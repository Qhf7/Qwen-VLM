#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from branches import MemoryBank
from common import LAYER_SETS, MASK_SIZE, Calibrator, greedy_coreset, list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed, topk_mean
from common import resize_map
from multiscale_decoder import LayerAwarePixelDecoder
from object_prior import apply_prior, foreground_grid
from view_consistency import apply_view_consistency


def _reservoir_add(store, key, values, capacity, rng):
    values = np.asarray(values, np.float16)
    current = store.get(key)
    if current is None:
        if len(values) > capacity:
            values = values[rng.choice(len(values), capacity, replace=False)]
        store[key] = values.copy()
        return
    merged = np.concatenate((current, values), axis=0)
    if len(merged) > capacity:
        merged = merged[rng.choice(len(merged), capacity, replace=False)]
    store[key] = merged


def build_bank(runner, samples, budget=2048, global_budget=8192):
    pools = {}
    global_pool = {}
    rng = np.random.default_rng(20260821)
    per_key_capacity = max(budget * 3, budget + 512)
    global_capacity = max(global_budget * 3, global_budget + 2048)
    for idx, sample in enumerate(samples):
        images = load_many(sample.paths)
        maps, _ = runner(images)
        cat = F.normalize(torch.cat([F.normalize(m, dim=-1) for m in maps], dim=-1), dim=-1).cpu().numpy()
        for view, arr in enumerate(cat):
            key = (sample.category, view)
            q = arr.reshape(-1, arr.shape[-1])[::8]
            _reservoir_add(pools, key, q, per_key_capacity, rng)
            _reservoir_add(global_pool, "global", q[::16], global_capacity, rng)
        if idx % 50 == 0:
            print(f"bank {idx}/{len(samples)}", flush=True)
    bank = MemoryBank(budget_per_key=budget, global_budget=global_budget, knn=3)
    for key, vectors in pools.items():
        x = vectors.astype(np.float32)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        chosen = greedy_coreset(x, min(budget, len(x)))
        bank.set(key, chosen)
    global_x = global_pool["global"].astype(np.float32)
    global_x = global_x / np.maximum(np.linalg.norm(global_x, axis=1, keepdims=True), 1e-6)
    bank.set_global(greedy_coreset(global_x, min(global_budget, len(global_x))))
    return bank


class Runner:
    def __init__(self, model, size, device, layers):
        self.model, self.size, self.device, self.layers = model, size, device, layers
    def __call__(self, images):
        return per_layer_features(self.model, images, self.size, self.device, self.layers)


def score_one(runner, decoder, bank, image, key, local=True):
    maps, side = runner([image])
    cat = F.normalize(torch.cat([F.normalize(m, dim=-1) for m in maps], dim=-1), dim=-1)[0].cpu().numpy()
    mem = bank.score(key, cat.reshape(-1, cat.shape[-1]), runner.device).reshape(side, side)
    with torch.inference_mode():
        logits, _ = decoder(maps)
        dense = torch.sigmoid(logits.float())[0].cpu().numpy()
    dense = resize_map(dense, MASK_SIZE)
    mem = resize_map(mem, MASK_SIZE)
    if local:
        # Two overlapping half-frame windows add local evidence at the same
        # generic DINO resolution. Scores are merged into the corresponding
        # canvas, so no external labels or prior predictions are used.
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        local_map = np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
        local_weight = np.zeros_like(local_map)
        for x0, y0 in ((0, 0), (w // 2, h // 2)):
            x1, y1 = min(w, x0 + w * 2 // 3), min(h, y0 + h * 2 // 3)
            crop = Image.fromarray(arr[y0:y1, x0:x1]).resize((max(280, w * 2 // 3), max(280, h * 2 // 3)), Image.Resampling.BICUBIC)
            cmaps, cside = runner([crop])
            with torch.inference_mode():
                clogits, _ = decoder(cmaps)
                cm = resize_map(torch.sigmoid(clogits.float())[0].cpu().numpy(), MASK_SIZE)
            xa, xb = int(x0 / w * MASK_SIZE), int(x1 / w * MASK_SIZE)
            ya, yb = int(y0 / h * MASK_SIZE), int(y1 / h * MASK_SIZE)
            if xb > xa and yb > ya:
                patch = np.asarray(Image.fromarray(np.uint8(cm * 255), "L").resize((xb-xa, yb-ya), Image.Resampling.BILINEAR), np.float32) / 255.0
                local_map[ya:yb, xa:xb] += patch
                local_weight[ya:yb, xa:xb] += 1
        dense = np.maximum(dense, local_map / np.maximum(local_weight, 1))
    raw = np.clip(0.68 * dense + 0.32 * mem, 0, 1)
    raw = apply_prior(raw, image, background_keep=0.04, boundary_keep=0.42)
    return raw


def main(args):
    set_seed(20260821)
    device = pick_device()
    layers = LAYER_SETS[args.arch]
    model = load_dinov2(args.arch, device, args.weight_dir)
    train = list_samples(args.data_root / "Train")
    test = list_samples(args.data_root / args.split)
    runner = Runner(model, args.size, device, layers)
    with torch.no_grad():
        probe = load_many(train[0].paths[:1])
        dims = [m.shape[-1] for m in runner(probe)[0]]
    decoder = LayerAwarePixelDecoder(dims, width=args.decoder_width).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(state["state"])
    decoder.eval()
    bank = build_bank(runner, train, args.budget, args.global_budget)
    # Train-only calibration; no test labels are read.
    normal_scores = {}
    for idx, sample in enumerate(train[:args.calib_limit or len(train)]):
        vals = []
        for view, image in enumerate(load_many(sample.paths)):
            vals.append(score_one(runner, decoder, bank, image, (sample.category, view), local=False))
        normal_scores.setdefault(sample.category, []).append(topk_mean(np.max(vals, axis=0)))
        if idx % 50 == 0:
            print(f"calibration {idx}/{len(train)}", flush=True)
    quant = {cat: (float(np.quantile(v, .50)), float(np.quantile(v, .995))) for cat, v in normal_scores.items()}
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    (out / "predicted_masks").mkdir(parents=True)
    rows = []
    for idx, sample in enumerate(test):
        maps = [score_one(runner, decoder, bank, image, (sample.category, view), local=not args.no_local) for view, image in enumerate(load_many(sample.paths))]
        maps, evidence = apply_view_consistency(maps)
        image_scores = []
        for view, (image, score_map) in enumerate(zip(load_many(sample.paths), maps)):
            lo, hi = quant.get(sample.category, (0.0, 1.0))
            calibrated = np.clip((score_map - lo) / max(hi - lo, 1e-5), 0, 1)
            image_scores.append(topk_mean(calibrated, .0015, 8))
            path = out / "predicted_masks" / sample.category / sample.name / f"{view}_mask.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.round(calibrated * 255)), "L").save(path, compress_level=1)
        rows.append((sample.key, float(np.clip(max(image_scores) * .70 + np.mean(sorted(image_scores)[-2:]) * .30, 0, 1))))
        if idx % 25 == 0:
            print(f"test {idx}/{len(test)}", flush=True)
    with (out / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["group_folder", "anomaly_score"]); writer.writerows(rows)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(out / "submission.csv", "submission.csv")
        for p in sorted((out / "predicted_masks").rglob("*_mask.png")):
            z.write(p, p.relative_to(out).as_posix())
    (out / "run_meta.json").write_text(json.dumps({"method":"v3 multi-layer DINOv2 + multi-scale decoder + local windows + hard-negative training + five-view consistency","external_data":False,"test_count":len(test)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--weight-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--split", default="Test_A"); ap.add_argument("--arch", default="dinov2_vitb14"); ap.add_argument("--size", type=int, default=1036)
    ap.add_argument("--decoder-width", type=int, default=192); ap.add_argument("--budget", type=int, default=2048); ap.add_argument("--global-budget", type=int, default=8192)
    ap.add_argument("--calib-limit", type=int, default=160); ap.add_argument("--no-local", action="store_true")
    main(ap.parse_args())
