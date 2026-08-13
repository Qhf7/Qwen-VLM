#!/usr/bin/env python3
"""Object-aware mask refinement for Real-IAD submissions.

The common failure mode of foundation-feature anomaly maps is highlighting the
whole object silhouette.  This builder uses the original test image only to
estimate a foreground object mask, then removes broad low-frequency responses
inside that object.  It does not use labels, external data, online models, or
test-time tuning feedback.
"""
import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image


def read_scores(root):
    with (root / "submission.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_path(data_root, category, sample, view):
    return data_root / "Test_A" / category / sample / f"{view}.png"


def load_gray(path, size=(448, 448)):
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.Resampling.BICUBIC)
        return np.asarray(im, np.float32) / 255.0


def load_rgb(path, size=(448, 448)):
    with Image.open(path) as im:
        im = im.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        return np.asarray(im, np.uint8)


def largest_components(binary, keep=3):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return binary.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = np.argsort(areas)[::-1][:keep] + 1
    out = np.isin(labels, ids)
    return out


def foreground_mask(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    border = np.concatenate(
        [
            lab[:12].reshape(-1, 3),
            lab[-12:].reshape(-1, 3),
            lab[:, :12].reshape(-1, 3),
            lab[:, -12:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)
    color_dist = np.linalg.norm(lab.astype(np.float32) - bg[None, None, :], axis=2)

    sat = hsv[:, :, 1].astype(np.float32)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    score = 0.70 * (color_dist / max(np.percentile(color_dist, 98), 1.0))
    score += 0.20 * (sat / max(np.percentile(sat, 98), 1.0))
    score += 0.10 * edges
    score = np.clip(score, 0, 1)

    thr = max(0.08, min(0.35, float(np.percentile(score, 72))))
    mask = score > thr
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = largest_components(mask, keep=4).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    # If the object is white on white, color distance can under-segment.  Fall
    # back to a broader central edge hull instead of letting the mask vanish.
    area = float(mask.mean())
    if area < 0.035:
        edge_mask = cv2.dilate((edges > 0).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=2)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
        mask = largest_components(edge_mask, keep=4).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    if mask.mean() > 0.80:
        # Over-broad foreground is dangerous for AP/F1.  Keep a conservative
        # object area from stronger color distance and edges.
        strict = ((color_dist > np.percentile(color_dist, 65)) | (edges > 0)).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        mask = largest_components(strict, keep=4).astype(np.uint8)

    return mask.astype(np.float32)


def local_sparse_anomaly(mask, fg):
    fg_bool = fg > 0.5
    if fg_bool.sum() < 64:
        return np.zeros_like(mask, dtype=np.float32)

    x = mask.astype(np.float32)
    x = x * fg
    low = cv2.GaussianBlur(x, (0, 0), 13)
    residual = np.clip(x - 0.72 * low, 0, 1)

    # Edge/silhouette responses are common false positives.  Erode the object
    # before measuring high-confidence anomalies, while retaining a faint band
    # for genuine edge defects.
    fg_u8 = fg_bool.astype(np.uint8)
    inner = cv2.erode(fg_u8, np.ones((9, 9), np.uint8), iterations=1).astype(np.float32)
    boundary = np.clip(fg - inner, 0, 1)
    residual = residual * (0.92 * inner + 0.25 * boundary)

    vals = residual[fg_bool]
    if vals.size == 0:
        return residual
    normal_floor = float(np.percentile(vals, 65))
    high = float(np.percentile(vals, 99.4))
    sparse = np.clip((residual - normal_floor) / max(high - normal_floor, 1e-6), 0, 1)

    # Tiny defects should remain visible, but broad object-wide activation
    # should not.  Suppress very large connected high-score regions.
    hot = (sparse > 0.45).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    if n > 1:
        cleaned = sparse.copy()
        image_area = sparse.shape[0] * sparse.shape[1]
        for comp in range(1, n):
            area = stats[comp, cv2.CC_STAT_AREA]
            if area > 0.10 * image_area:
                cleaned[labels == comp] *= 0.25
        sparse = cleaned
    sparse = cv2.GaussianBlur(sparse, (3, 3), 0)
    return np.clip(sparse * fg, 0, 1).astype(np.float32)


def save_mask(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(array, 0, 1) * 255)), "L").save(path, "PNG", compress_level=1)


def process_one(args_tuple):
    data_root, base, out_masks, category, sample, view, mask_weight, sparse_weight = args_tuple
    rgb = load_rgb(image_path(data_root, category, sample, view))
    fg = foreground_mask(rgb)
    base_mask = load_gray(base / "predicted_masks" / category / sample / f"{view}_mask.png")
    sparse = local_sparse_anomaly(base_mask, fg)
    refined = np.clip(mask_weight * base_mask * fg + sparse_weight * sparse, 0, 1)
    save_mask(out_masks / category / sample / f"{view}_mask.png", refined)
    return {
        "fg_area": float(fg.mean()),
        "base_area": float((base_mask > 0.5).mean()),
        "refined_area": float((refined > 0.5).mean()),
        "base_mean": float(base_mask.mean()),
        "refined_mean": float(refined.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--mask-weight", type=float, default=0.72)
    parser.add_argument("--sparse-weight", type=float, default=0.28)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = read_scores(args.base)
    args.output.mkdir(parents=True, exist_ok=True)
    out_masks = args.output / "predicted_masks"
    shutil.copyfile(args.base / "submission.csv", args.output / "submission.csv")

    tasks = []
    for row in rows:
        category, sample = row["group_folder"].split("/", 1)
        for view in range(5):
            tasks.append(
                (args.data_root, args.base, out_masks, category, sample, view, args.mask_weight, args.sparse_weight)
            )

    mask_stats = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, stat in enumerate(ex.map(process_one, tasks), 1):
            mask_stats.append(stat)
            if i % 500 == 0:
                print("processed masks", i, flush=True)

    summary = {
        "method": "object-aware foreground gating + low-frequency object-response removal",
        "base": str(args.base),
        "data_root": str(args.data_root),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "mean_fg_area": float(np.mean([x["fg_area"] for x in mask_stats])),
        "mean_base_area_gt_0_5": float(np.mean([x["base_area"] for x in mask_stats])),
        "mean_refined_area_gt_0_5": float(np.mean([x["refined_area"] for x in mask_stats])),
        "mean_base_mask": float(np.mean([x["base_mean"] for x in mask_stats])),
        "mean_refined_mask": float(np.mean([x["refined_mean"] for x in mask_stats])),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
