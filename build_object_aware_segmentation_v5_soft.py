#!/usr/bin/env python3
"""Soft object-aware refinement for the strongest validated submission.

v4 looked visually cleaner but lost leaderboard score because it rebuilt masks
too aggressively and damaged pixel ranking/recall.  v5 is deliberately softer:
it keeps the validated mask ordering, applies a foreground prior, and only
attenuates broad whole-object/silhouette responses.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_object_aware_segmentation_v2 import foreground_mask, load_rgb, read_rows


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32) / 255.0


def save_mask(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(x, 0, 1) * 255)), "L").save(path, "PNG", compress_level=1)


def soft_foreground(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fg = foreground_mask(rgb).astype(np.float32)
    fg = cv2.GaussianBlur(fg, (0, 0), 2.0)
    fg = np.clip(fg, 0, 1)
    fg_u8 = (fg > 0.45).astype(np.uint8)
    inner = cv2.erode(fg_u8, np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)
    return fg, inner


def suppress_only_broad(mask: np.ndarray, fg: np.ndarray, inner: np.ndarray) -> np.ndarray:
    # Keep original ordering.  Only estimate the broad component and subtract a
    # small amount from it; this avoids the v4 failure of throwing away recall.
    x = np.clip(mask, 0, 1)
    gate = 0.18 + 0.82 * fg
    gated = x * gate

    broad = cv2.GaussianBlur(gated, (0, 0), 19)
    local = np.clip(gated - 0.35 * broad, 0, 1)

    hot = (gated > 0.52).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    area_total = gated.shape[0] * gated.shape[1]
    attenuation = np.ones_like(gated, np.float32)
    for comp in range(1, n):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area > 0.10 * area_total:
            attenuation[labels == comp] = 0.72
        elif area > 0.045 * area_total:
            attenuation[labels == comp] = 0.84

    boundary = np.clip(fg - inner, 0, 1)
    boundary_penalty = 1.0 - 0.22 * boundary
    out = (0.72 * gated + 0.28 * local) * attenuation * boundary_penalty
    return np.clip(out, 0, 1)


def process_one(task):
    data_root, base_root, out_masks, category, sample, view = task
    rgb = load_rgb(data_root / "Test_A" / category / sample / f"{view}.png")
    fg, inner = soft_foreground(rgb)
    base = load_mask(base_root / "predicted_masks" / category / sample / f"{view}_mask.png")
    refined = suppress_only_broad(base, fg, inner)
    save_mask(out_masks / category / sample / f"{view}_mask.png", refined)
    return {
        "fg_area": float(fg.mean()),
        "base_area": float((base > 0.5).mean()),
        "refined_area": float((refined > 0.5).mean()),
        "base_mean": float(base.mean()),
        "refined_mean": float(refined.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    rows = read_rows(args.base)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.base / "submission.csv", args.output / "submission.csv")
    out_masks = args.output / "predicted_masks"

    tasks = []
    for row in rows:
        category, sample = row["group_folder"].split("/", 1)
        for view in range(5):
            tasks.append((args.data_root, args.base, out_masks, category, sample, view))

    stats = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, stat in enumerate(ex.map(process_one, tasks), 1):
            stats.append(stat)
            if i % 500 == 0:
                print("processed masks", i, flush=True)

    meta = {
        "method": "soft object-aware foreground gate plus broad-response attenuation",
        "base": str(args.base),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "mean_fg_area": float(np.mean([x["fg_area"] for x in stats])),
        "mean_base_area_gt_0_5": float(np.mean([x["base_area"] for x in stats])),
        "mean_refined_area_gt_0_5": float(np.mean([x["refined_area"] for x in stats])),
        "mean_base_mask": float(np.mean([x["base_mean"] for x in stats])),
        "mean_refined_mask": float(np.mean([x["refined_mean"] for x in stats])),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
