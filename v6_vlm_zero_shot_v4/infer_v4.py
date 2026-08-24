#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(V3))

from clip_dense import clip_preprocess_tensor, load_clip
from common import list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed, topk_mean
from model_v4 import VLMAnomalyV4
from memory_v4 import build_memory, memory_maps
from object_prior import apply_prior
from view_consistency import apply_view_consistency


def sha256(path: Path | None):
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    model = VLMAnomalyV4(
        ckpt["dims"], clip_model, ckpt["decoder_width"], ckpt["fusion_width"], ckpt["prompt_ctx"]
    ).to(device)
    model.load_compact_state_dict(ckpt["model"])
    model.eval()
    dino = load_dinov2(ckpt["arch"], device, args.weight_dir).eval()
    return ckpt, dino, model


def calibrate(args, ckpt, dino, model, bank, feature_fn, train, categories, device):
    """Fit separate pixel and image calibration distributions.

    Pixel maps must be calibrated from normal *pixel values*.  The failed v3
    stream accidentally used image-level top-k quantiles as pixel bounds,
    turning much of the object into a low non-zero haze.  That improved
    P-AUROC but severely reduced P-AP/F1max.  Keep the two score domains
    explicitly separate here.
    """
    image_by_category: dict[str, list[float]] = {}
    image_global = []
    pixel_by_category: dict[str, list[np.ndarray]] = {}
    pixel_global: list[np.ndarray] = []
    limit = train[: args.calib_limit or len(train)]
    with torch.inference_mode():
        for index, sample in enumerate(limit):
            images = load_many(sample.paths)
            dino_maps, _ = per_layer_features(dino, images, ckpt["dino_size"], device, tuple(ckpt["layers"]))
            clip_x = clip_preprocess_tensor(images, ckpt["clip_size"], device)
            mem = torch.from_numpy(memory_maps(bank, dino_maps, [sample.category] * 5, device)).to(device)
            output = model(dino_maps, clip_x, seen=torch.ones(5, device=device), groups=1, memory_map=mem)
            maps = F.interpolate(output.final_map[:, None], (448, 448), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            prepared = [apply_prior(m, im, 0.03, 0.40) for m, im in zip(maps, images)]
            values = [topk_mean(m, 0.0015, 8) for m in prepared]
            score = 0.70 * max(values) + 0.30 * float(np.mean(sorted(values)[-2:]))
            image_by_category.setdefault(sample.category, []).append(score)
            image_global.append(score)
            # Deterministic sparse sampling keeps memory bounded while still
            # preserving the normal pixel distribution and its high tail.
            sampled = np.concatenate([m.ravel()[::37] for m in prepared]).astype(np.float32)
            pixel_by_category.setdefault(sample.category, []).append(sampled)
            pixel_global.append(sampled[::5])
            if index % 50 == 0:
                print(f"calibrate {index}/{len(limit)}", flush=True)
    image_fallback = (float(np.quantile(image_global, 0.50)), float(np.quantile(image_global, 0.995)))
    image_category = {
        key: (float(np.quantile(value, 0.50)), float(np.quantile(value, 0.995)))
        for key, value in image_by_category.items()
        if len(value) >= 3
    }
    pixel_pool = np.concatenate(pixel_global)
    pixel_fallback = (float(np.quantile(pixel_pool, args.pixel_lo_quantile)), float(np.quantile(pixel_pool, args.pixel_hi_quantile)))
    pixel_category = {}
    for key, arrays in pixel_by_category.items():
        pool = np.concatenate(arrays)
        pixel_category[key] = (
            float(np.quantile(pool, args.pixel_lo_quantile)),
            float(np.quantile(pool, args.pixel_hi_quantile)),
        )
    return pixel_category, pixel_fallback, image_category, image_fallback


def main(args):
    set_seed(args.seed)
    device = pick_device()
    ckpt, dino, model = load_model(args, device)
    train = list_samples(args.data_root / "Train")
    test = list_samples(args.data_root / args.split)
    seen_categories = {s.category for s in train}
    def feature_fn(images):
        return per_layer_features(dino, images, ckpt["dino_size"], device, tuple(ckpt["layers"]))
    bank = build_memory(feature_fn, train, args.memory_budget, args.global_memory_budget, args.seed)
    pixel_calibration, pixel_fallback, image_calibration, image_fallback = calibrate(
        args, ckpt, dino, model, bank, feature_fn, train, seen_categories, device
    )

    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / "predicted_masks").mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(test):
            images = load_many(sample.paths)
            dino_maps, _ = per_layer_features(dino, images, ckpt["dino_size"], device, tuple(ckpt["layers"]))
            clip_x = clip_preprocess_tensor(images, ckpt["clip_size"], device)
            seen_value = 1.0 if sample.category in seen_categories else 0.0
            mem = torch.from_numpy(memory_maps(bank, dino_maps, [sample.category] * 5, device)).to(device)
            output = model(
                dino_maps,
                clip_x,
                seen=torch.full((5,), seen_value, device=device),
                groups=1,
                memory_map=mem,
            )
            raw = F.interpolate(output.final_map[:, None], (448, 448), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            raw = [apply_prior(m, image, background_keep=0.025, boundary_keep=0.38) for m, image in zip(raw, images)]
            raw, _ = apply_view_consistency(raw)
            raw_view_scores = [topk_mean(m, 0.0015, 8) for m in raw]
            raw_dense_score = 0.70 * max(raw_view_scores) + 0.30 * float(np.mean(sorted(raw_view_scores)[-2:]))
            pixel_lo, pixel_hi = pixel_calibration.get(sample.category, pixel_fallback)
            maps = [np.clip((m - pixel_lo) / max(pixel_hi - pixel_lo, 1e-5), 0, 1) for m in raw]
            # Enforce an exact zero floor learned only from Train-normal pixel
            # statistics.  This prevents broad low-valued haze without doing
            # per-image percentile normalisation or using Test labels.
            maps = [np.where(m < args.zero_floor, 0.0, m).astype(np.float32) for m in maps]
            image_lo, image_hi = image_calibration.get(sample.category, image_fallback)
            dense_score = float(np.clip((raw_dense_score - image_lo) / max(image_hi - image_lo, 1e-5), 0, 1))
            mil_score = float(torch.sigmoid(output.sample_logits)[0].cpu())
            anomaly_score = float(np.clip(0.75 * dense_score + 0.25 * mil_score, 0, 1))
            rows.append((sample.key, anomaly_score))
            for view, score_map in enumerate(maps):
                path = args.output / "predicted_masks" / sample.category / sample.name / f"{view}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(np.round(score_map * 255)), "L").save(path, compress_level=1)
            if index % 20 == 0:
                print(f"test {index}/{len(test)}", flush=True)

    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_folder", "anomaly_score"])
        writer.writerows(rows)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(args.output / "submission.csv", "submission.csv")
        for path in sorted((args.output / "predicted_masks").rglob("*_mask.png")):
            archive.write(path, path.relative_to(args.output).as_posix())
    metadata = {
        "method": "DINOv2 + finetuned Prompt/CLIP adapter/local alignment/anomaly queries/multistate head/five-view MIL",
        "official_train_only": True,
        "external_anomaly_data": False,
        "online_api": False,
        "test_groups": len(test),
        "clip_weight_sha256": sha256(args.clip_weight),
        "checkpoint": str(args.checkpoint),
    }
    (args.output / "run_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--split", default="Test_A")
    parser.add_argument("--calib-limit", type=int, default=200)
    parser.add_argument("--memory-budget", type=int, default=1536)
    parser.add_argument("--global-memory-budget", type=int, default=8192)
    parser.add_argument("--pixel-lo-quantile", type=float, default=0.90)
    parser.add_argument("--pixel-hi-quantile", type=float, default=0.9995)
    parser.add_argument("--zero-floor", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260824)
    main(parser.parse_args())
