#!/usr/bin/env python3
"""Balanced object-aware segmentation refinement.

This combines v1's foreground gating with v2's Train-normal residual evidence.
It avoids whole-object highlight while keeping enough recall for stains, dents,
and low-contrast defects inside the part.
"""
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

from build_object_aware_segmentation_v2 import (
    collect_calibration,
    foreground_mask,
    load_rgb,
    read_rows,
    residual_maps,
)


def load_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32) / 255.0


def save_mask(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(array, 0, 1) * 255)), "L").save(path, "PNG", compress_level=1)


def calibrate_residual_soft(residual, q):
    low = float(q[0])
    high = float(q[3])
    return np.clip((residual - low) / max(high - low, 1e-6), 0, 1)


def suppress_broad_response(base, fg):
    x = base * fg
    low = cv2.GaussianBlur(x, (0, 0), 17)
    local = np.clip(x - 0.58 * low, 0, 1)
    fg_bool = fg > 0.5
    if fg_bool.sum() == 0:
        return np.zeros_like(base)
    vals = local[fg_bool]
    floor = float(np.percentile(vals, 55))
    top = float(np.percentile(vals, 99.2))
    local = np.clip((local - floor) / max(top - floor, 1e-6), 0, 1)

    fg_u8 = fg_bool.astype(np.uint8)
    inner = cv2.erode(fg_u8, np.ones((9, 9), np.uint8), iterations=1).astype(np.float32)
    boundary = np.clip(fg - inner, 0, 1)
    local = local * (0.90 * inner + 0.35 * boundary)
    return local


def cap_large_components(x):
    hot = (x > 0.42).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    out = x.copy()
    area_total = x.shape[0] * x.shape[1]
    for comp in range(1, n):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area > 0.075 * area_total:
            out[labels == comp] *= 0.35
    return out


def process_one(task):
    data_root, base_root, out_masks, cal, category, sample, view = task
    rgb = load_rgb(data_root / "Test_A" / category / sample / f"{view}.png")
    fg = foreground_mask(rgb)
    residual, inner = residual_maps(rgb, fg)
    q = cal.get((category, view), cal[("__global__", -1)])
    residual_score = calibrate_residual_soft(residual, q)
    base = load_mask(base_root / "predicted_masks" / category / sample / f"{view}_mask.png")

    gated_base = np.clip(base * fg, 0, 1)
    local_base = suppress_broad_response(base, fg)
    supported = np.sqrt(np.clip(gated_base, 0, 1) * np.clip(residual_score, 0, 1))

    # Keep some base confidence for broad, low-contrast defects, but require
    # local support to become bright.  This is the balance missing in v1/v2.
    refined = 0.36 * gated_base + 0.42 * local_base + 0.34 * supported
    refined = refined * (inner + 0.18 * np.clip(fg - inner, 0, 1))
    refined = cap_large_components(np.clip(refined, 0, 1))
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
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
    cal = collect_calibration(args.data_root, args.workers)

    tasks = []
    for row in rows:
        category, sample = row["group_folder"].split("/", 1)
        for view in range(5):
            tasks.append((args.data_root, args.base, out_masks, cal, category, sample, view))

    stats = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, stat in enumerate(ex.map(process_one, tasks), 1):
            stats.append(stat)
            if i % 500 == 0:
                print("processed masks", i, flush=True)

    meta = {
        "method": "balanced foreground gating + local base residual + Train-normal image residual support",
        "base": str(args.base),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "mean_fg_area": float(np.mean([x["fg_area"] for x in stats])),
        "mean_base_area_gt_0_5": float(np.mean([x["base_area"] for x in stats])),
        "mean_refined_area_gt_0_5": float(np.mean([x["refined_area"] for x in stats])),
        "mean_base_mask": float(np.mean([x["base_mean"] for x in stats])),
        "mean_refined_mask": float(np.mean([x["refined_mean"] for x in stats])),
        "calibration_keys": len(cal),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
