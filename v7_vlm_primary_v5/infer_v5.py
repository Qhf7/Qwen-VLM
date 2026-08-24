#!/usr/bin/env python3
"""Generate a VLM-primary v5 competition submission."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "v6_vlm_zero_shot_v4"
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(V4)); sys.path.insert(2, str(V3))

from calibration_v5 import assert_category_coverage, fit_calibration, stratified_samples
from clip_dense import clip_preprocess_tensor, load_clip
from common import list_samples, load_dinov2, load_many, pick_device, set_seed
from model_v5 import VLMPrimaryV5, load_trusted_checkpoint
from pipeline_v5 import calibrated_output, run_group
from progress_v5 import ProgressReporter
from reference_v5 import build_reference_bank


def load_models(args, device):
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    model = VLMPrimaryV5(
        checkpoint["dims"], clip_model, checkpoint["decoder_width"], checkpoint["vlm_width"],
        checkpoint["prompt_ctx"], checkpoint["max_dino_fraction"],
    ).to(device)
    model.load_compact_state_dict(checkpoint["model"])
    model.eval()
    dino = load_dinov2(checkpoint["arch"], device, args.weight_dir).eval()
    return checkpoint, dino, model


def build_calibration(args, checkpoint, dino, model, bank, train, device):
    pixel_values = defaultdict(list)
    image_values = defaultdict(list)
    selected = stratified_samples(train, args.calib_per_category, args.seed)
    expected = sorted({sample.category for sample in train})
    if {sample.category for sample in selected} != set(expected):
        raise RuntimeError("stratified calibration failed to cover every category")
    progress = ProgressReporter(
        "submission_calibration", len(selected), args.output.parent / "calibration_progress.json"
    )
    with torch.inference_mode():
        for index, sample in enumerate(selected):
            images = load_many(sample.paths)
            prediction = run_group(
                images, [sample.category] * 5, model, dino, bank,
                checkpoint["dino_size"], checkpoint["clip_size"], checkpoint["layers"], device,
                args.reference_scale,
            )
            sampled = np.concatenate([score.ravel()[::37] for score in prediction.raw_maps]).astype(np.float32)
            pixel_values[sample.category].append(sampled)
            image_values[sample.category].append(prediction.raw_image_score)
            if index % 10 == 0 or index + 1 == len(selected):
                progress.update(index + 1, {"category": sample.category})
    calibration = fit_calibration(pixel_values, image_values, args.pixel_lo_quantile, args.pixel_hi_quantile)
    assert_category_coverage(calibration, expected)
    print(f"v5 calibration categories={len(calibration.categories)}/{len(expected)}", flush=True)
    return calibration


def main(args):
    set_seed(args.seed)
    device = pick_device()
    checkpoint, dino, model = load_models(args, device)
    train = list_samples(args.data_root / "Train")
    test = list_samples(args.data_root / args.split)
    seen = {sample.category for sample in train}
    bank = build_reference_bank(
        model, clip_preprocess_tensor, train, checkpoint["clip_size"], device,
        args.reference_budget, args.global_reference_budget, args.seed,
        args.output.parent / "reference_progress.json", "submission_reference_bank",
    )
    calibration = build_calibration(args, checkpoint, dino, model, bank, train, device)
    if args.output.exists() or args.zip_path.exists():
        raise FileExistsError("refusing to overwrite an existing v5 output or ZIP")
    (args.output / "predicted_masks").mkdir(parents=True)
    rows = []
    progress = ProgressReporter("test_inference", len(test), args.output.parent / "infer_progress.json")
    with torch.inference_mode():
        for index, sample in enumerate(test):
            images = load_many(sample.paths)
            prediction = run_group(
                images, [sample.category] * 5, model, dino, bank,
                checkpoint["dino_size"], checkpoint["clip_size"], checkpoint["layers"], device,
                args.reference_scale,
            )
            maps, score = calibrated_output(
                prediction, sample.category, calibration, zero_floor=args.zero_floor
            )
            rows.append((sample.key, score))
            for view, score_map in enumerate(maps):
                path = args.output / "predicted_masks" / sample.category / sample.name / f"{view}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(np.round(np.clip(score_map, 0, 1) * 255)), "L").save(
                    path, compress_level=1
                )
            if index % 5 == 0 or index + 1 == len(test):
                progress.update(
                    index + 1,
                    {"category": sample.category, "sample": sample.name, "anomaly_score": float(score)},
                )
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_folder", "anomaly_score"])
        writer.writerows((key, f"{score:.8f}") for key, score in rows)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(args.output / "submission.csv", "submission.csv")
        for path in sorted((args.output / "predicted_masks").rglob("*_mask.png")):
            archive.write(path, path.relative_to(args.output).as_posix())
    metadata = {
        "method": "VLM-primary v5: WinCLIP local windows + AdaCLIP hybrid prompts + CLIP normal reference",
        "official_train_only": True,
        "external_anomaly_data": False,
        "online_api": False,
        "calibration_categories": len(calibration.categories),
        "seen_categories": len(seen),
        "test_groups": len(test),
        "checkpoint": str(args.checkpoint),
    }
    (args.output / "run_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"INFER_COMPLETE groups={len(rows)} zip={args.zip_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--split", default="Test_A")
    parser.add_argument("--calib-per-category", type=int, default=10)
    parser.add_argument("--reference-budget", type=int, default=1024)
    parser.add_argument("--global-reference-budget", type=int, default=8192)
    parser.add_argument("--reference-scale", type=float, default=0.18)
    parser.add_argument("--pixel-lo-quantile", type=float, default=0.90)
    parser.add_argument("--pixel-hi-quantile", type=float, default=0.9995)
    parser.add_argument("--zero-floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260825)
    main(parser.parse_args())
