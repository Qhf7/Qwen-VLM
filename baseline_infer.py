#!/usr/bin/env python3
"""Competition-compliant frozen ViT baseline.

Only the official Train normal images are used to build per-category/view
normal feature statistics. The torchvision ImageNet ViT-B/16 checkpoint is a
generic public backbone; no competition-specific checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16


def groups(root: Path):
    for category in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"):
        for sample in sorted(p for p in category.iterdir() if p.is_dir()):
            paths = [sample / f"{i}.png" for i in range(5)]
            if all(p.exists() for p in paths):
                yield category.name, sample.name, paths


def make_model(device: torch.device):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights.transforms()


@torch.inference_mode()
def features(model, batch: torch.Tensor):
    # torchvision ViT internals: keep patch tokens before the class token.
    x = model._process_input(batch)
    n = x.shape[0]
    cls = model.class_token.expand(n, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = model.encoder(x)
    cls_feature = torch.nn.functional.normalize(x[:, 0], dim=-1)
    patch = torch.nn.functional.normalize(x[:, 1:], dim=-1)
    return cls_feature.cpu(), patch.cpu()


def read_batch(paths, transform):
    def load_one(path):
        with Image.open(path) as im:
            return transform(im.convert("RGB"))
    workers = min(16, max(4, len(paths) // 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(pool.map(load_one, paths))
    return torch.stack(images)


def extract_split(root, model, transform, device, batch_size):
    items = list(groups(root))
    records = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        flat = [p for _, _, paths in chunk for p in paths]
        batch = read_batch(flat, transform).to(device)
        cls, patch = features(model, batch)
        cls = cls.view(len(chunk), 5, -1)
        patch = patch.view(len(chunk), 5, 196, -1)
        for j, (category, sample, _) in enumerate(chunk):
            records.append((category, sample, cls[j].numpy(), patch[j].numpy()))
        if (start // batch_size) % 10 == 0:
            print(f"extracted {min(start + batch_size, len(items))}/{len(items)} groups", flush=True)
    return records


def build_stats(train_records):
    # Per category/view/patch-position diagonal Gaussian statistics.
    by_key = {}
    global_by_view = {}
    for category, _sample, cls, patch in train_records:
        for view in range(5):
            key = (category, view)
            by_key.setdefault(key, []).append(patch[view])
            global_by_view.setdefault(view, []).append(patch[view])
    stats = {}
    for key, arrays in by_key.items():
        x = np.stack(arrays).astype(np.float32)  # [samples, 196, dim]
        stats[key] = (x.mean(0), x.std(0) + 0.05)
    global_stats = {}
    for view, arrays in global_by_view.items():
        x = np.concatenate(arrays, axis=0).astype(np.float32)
        global_stats[view] = (x.mean(0), x.std(0) + 0.05)
    return stats, global_stats


def score_one(patch, cls, mean, std):
    # Robust diagonal standardized distance, reduced over embedding channels.
    z = np.abs((patch - mean) / std)
    patch_score = np.percentile(z, 90, axis=1)
    # Ignore the most extreme single patch noise for group-level stability.
    image_score = float(np.percentile(patch_score, 99))
    return patch_score, image_score


def robust_scale(x, lo=1.0, hi=8.0):
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return x


def write_mask(path: Path, score):
    from PIL import ImageFilter
    # 14x14 patch grid -> 448x448; keep continuous grayscale values.
    grid = np.asarray(score, dtype=np.float32).reshape(14, 14)
    grid = robust_scale(grid, lo=1.0, hi=6.0)
    image = Image.fromarray(np.uint8(np.clip(grid * 255.0, 0, 255)), mode="L")
    image = image.resize((448, 448), Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--output", type=Path, default=Path("submission_vit_baseline"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)
    model, transform = make_model(device)
    t0 = time.time()
    train = extract_split(args.data_root / "Train", model, transform, device, args.batch_size)
    stats, global_stats = build_stats(train)
    test = extract_split(args.data_root / "Test_A", model, transform, device, args.batch_size)
    mask_root = args.output / "predicted_masks"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, sample, _cls, patch in test:
        view_scores = []
        for view in range(5):
            mean, std = stats.get((category, view), global_stats[view])
            patch_score, image_score = score_one(patch[view], _cls[view], mean, std)
            view_scores.append(image_score)
            write_mask(mask_root / category / sample / f"{view}_mask.png", patch_score)
        # Larger score means more anomalous; map to a stable [0, 1] score.
        normalized = [float(robust_scale(s, lo=1.0, hi=8.0)) for s in view_scores]
        group_score = 0.7 * max(normalized) + 0.3 * float(np.mean(normalized))
        rows.append({"group_folder": f"{category}/{sample}", "anomaly_score": f"{group_score:.8f}"})
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "method": "frozen torchvision ImageNet ViT-B/16 + official Train normal statistics",
        "train_groups": len(train),
        "test_groups": len(test),
        "views": 5,
        "mask_size": [448, 448],
        "device": str(device),
        "elapsed_seconds": round(time.time() - t0, 2),
        "external_specialized_weights": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
