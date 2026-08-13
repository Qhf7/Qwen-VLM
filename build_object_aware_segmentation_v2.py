#!/usr/bin/env python3
"""Train-calibrated object-aware defect mask refinement.

v1 blackens background but can still leave a whole-object gray response.  v2
adds a Train-normal image residual model: normal object texture/edge residuals
define category/view thresholds, so test masks keep only local image evidence
that is unusual for that product view and supported by the base anomaly map.
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


def groups(root):
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            paths = [sample_dir / f"{i}.png" for i in range(5)]
            if all(p.exists() for p in paths):
                yield cat_dir.name, sample_dir.name, paths


def read_rows(root):
    with (root / "submission.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rgb(path, size=(448, 448)):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size, Image.Resampling.BICUBIC), np.uint8)


def load_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32) / 255.0


def largest_components(binary, keep=4):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return binary.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = np.argsort(areas)[::-1][:keep] + 1
    return np.isin(labels, ids)


def foreground_mask(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate(
        [lab[:12].reshape(-1, 3), lab[-12:].reshape(-1, 3), lab[:, :12].reshape(-1, 3), lab[:, -12:].reshape(-1, 3)],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)
    color_dist = np.linalg.norm(lab.astype(np.float32) - bg[None, None, :], axis=2)
    sat = hsv[:, :, 1].astype(np.float32)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    score = 0.72 * (color_dist / max(np.percentile(color_dist, 98), 1.0))
    score += 0.18 * (sat / max(np.percentile(sat, 98), 1.0))
    score += 0.10 * edges
    score = np.clip(score, 0, 1)
    mask = (score > max(0.08, min(0.35, float(np.percentile(score, 72))))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = largest_components(mask, keep=4).astype(np.uint8)
    if mask.mean() < 0.035:
        edge_mask = cv2.dilate((edges > 0).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=2)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
        mask = largest_components(edge_mask, keep=4).astype(np.uint8)
    if mask.mean() > 0.78:
        strict = ((color_dist > np.percentile(color_dist, 65)) | (edges > 0)).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        mask = largest_components(strict, keep=4).astype(np.uint8)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)


def residual_maps(rgb, fg):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    blur_gray = cv2.GaussianBlur(gray, (0, 0), 5)
    fine = np.abs(gray - blur_gray)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    blur_lab = cv2.GaussianBlur(lab, (0, 0), 7)
    color = np.linalg.norm(lab - blur_lab, axis=2) / 255.0
    residual = 0.50 * fine + 0.30 * lap + 0.20 * color
    residual = cv2.GaussianBlur(residual, (3, 3), 0)
    fg_u8 = (fg > 0.5).astype(np.uint8)
    inner = cv2.erode(fg_u8, np.ones((11, 11), np.uint8), iterations=1).astype(np.float32)
    boundary = np.clip(fg - inner, 0, 1)
    residual = residual * (inner + 0.18 * boundary)
    return np.clip(residual, 0, 1), inner


def collect_calibration(data_root, workers):
    train_items = list(groups(data_root / "Train"))

    def one(item):
        category, _, paths = item
        out = []
        for view, path in enumerate(paths):
            rgb = load_rgb(path)
            fg = foreground_mask(rgb)
            residual, inner = residual_maps(rgb, fg)
            valid = inner > 0.5
            vals = residual[valid]
            if vals.size:
                out.append(((category, view), np.percentile(vals, [90, 97, 99, 99.7]).astype(np.float32)))
                out.append((("__global__", -1), np.percentile(vals, [90, 97, 99, 99.7]).astype(np.float32)))
        return out

    buckets = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, result in enumerate(ex.map(one, train_items), 1):
            for key, q in result:
                buckets.setdefault(key, []).append(q)
            if i % 200 == 0:
                print("calibrated train groups", i, flush=True)
    return {key: np.median(np.stack(vals), axis=0) for key, vals in buckets.items()}


def calibrate_residual(residual, q):
    low = float(q[1])
    high = float(q[3])
    return np.clip((residual - low) / max(high - low, 1e-6), 0, 1)


def remove_large_components(x, max_area_ratio):
    hot = (x > 0.35).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    out = x.copy()
    image_area = x.shape[0] * x.shape[1]
    for comp in range(1, n):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area > max_area_ratio * image_area:
            out[labels == comp] *= 0.18
    return out


def refine_one(task):
    data_root, base, out_masks, cal, category, sample, view = task
    rgb = load_rgb(data_root / "Test_A" / category / sample / f"{view}.png")
    fg = foreground_mask(rgb)
    residual, inner = residual_maps(rgb, fg)
    q = cal.get((category, view), cal[("__global__", -1)])
    defect = calibrate_residual(residual, q)
    base_mask = load_mask(base / "predicted_masks" / category / sample / f"{view}_mask.png")

    supported = np.sqrt(np.clip(base_mask, 0, 1) * np.clip(defect, 0, 1))
    supported = supported * (inner + 0.10 * np.clip(fg - inner, 0, 1))
    supported = remove_large_components(supported, max_area_ratio=0.035)
    supported = cv2.GaussianBlur(supported, (3, 3), 0)
    refined = np.clip(supported ** 0.85, 0, 1)
    save_path = out_masks / category / sample / f"{view}_mask.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(refined * 255)), "L").save(save_path, "PNG", compress_level=1)
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
        for i, stat in enumerate(ex.map(refine_one, tasks), 1):
            stats.append(stat)
            if i % 500 == 0:
                print("processed masks", i, flush=True)

    meta = {
        "method": "Train-calibrated object-aware local residual mask refinement",
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
