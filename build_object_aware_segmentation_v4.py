#!/usr/bin/env python3
"""Template-calibrated object-aware mask refinement.

Goal:
  - background -> near-black
  - normal object silhouette/edges -> suppressed
  - only local deviations from the normal part template stay bright

This version uses official Train normal images to build a per-(category,view)
appearance template and residual quantiles.  Test masks are refined by the
intersection of:
  1. foreground object region,
  2. base anomaly evidence,
  3. deviation from the Train-normal template.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_object_aware_segmentation_v2 import foreground_mask, load_rgb, read_rows
from build_object_aware_segmentation_v3 import cap_large_components, suppress_broad_response


TEMPLATE_SIZE = 224


@dataclass
class TemplateStats:
    mean: np.ndarray
    std: np.ndarray
    fg_prior: np.ndarray
    quantiles: np.ndarray


def groups(root: Path):
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            paths = [sample_dir / f"{i}.png" for i in range(5)]
            if all(p.exists() for p in paths):
                yield cat_dir.name, sample_dir.name, paths


def load_gray(path: Path, size=(448, 448)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.Resampling.BICUBIC)
        return np.asarray(im, np.float32) / 255.0


def resize_float(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def build_templates(data_root: Path, workers: int) -> dict[tuple[str, int], TemplateStats]:
    train_items = list(groups(data_root / "Train"))
    by_key: dict[tuple[str, int], list[Path]] = {}
    for cat, _, paths in train_items:
        for view, path in enumerate(paths):
            by_key.setdefault((cat, view), []).append(path)

    def fit_key(item):
        key, paths = item
        grays = []
        fgs = []
        for path in paths:
            rgb = load_rgb(path)
            fg448 = foreground_mask(rgb)
            gray224 = resize_float(load_gray(path), TEMPLATE_SIZE)
            fg224 = resize_float(fg448, TEMPLATE_SIZE)
            grays.append(gray224)
            fgs.append(fg224)
        stack = np.stack(grays).astype(np.float32)
        fg_stack = np.stack(fgs).astype(np.float32)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        fg_prior = fg_stack.mean(axis=0)

        # Second pass: quantify what "normal deviation" looks like for this
        # (category, view), so test-time residuals can be measured relative to
        # train variation instead of per-image self-normalization.
        vals = []
        std_floor = np.maximum(std, 0.03)
        valid = fg_prior > 0.35
        for gray in stack:
            z = np.abs(gray - mean) / std_floor
            vals.append(z[valid][::2])
        q = np.percentile(np.concatenate(vals), [80, 92, 97.5, 99.5]).astype(np.float32)
        return key, TemplateStats(mean=mean, std=std.astype(np.float32), fg_prior=fg_prior.astype(np.float32), quantiles=q)

    out: dict[tuple[str, int], TemplateStats] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (key, stat) in enumerate(ex.map(fit_key, by_key.items()), 1):
            out[key] = stat
            if i % 40 == 0:
                print("built templates", i, flush=True)
    return out


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), np.float32) / 255.0


def upsample224(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), (448, 448), interpolation=cv2.INTER_LINEAR)


def template_deviation(gray224: np.ndarray, stats: TemplateStats) -> np.ndarray:
    z = np.abs(gray224 - stats.mean) / np.maximum(stats.std, 0.03)
    q = stats.quantiles
    return np.clip((z - float(q[1])) / max(float(q[3] - q[1]), 1e-6), 0, 1)


def final_foreground(rgb448: np.ndarray, stats: TemplateStats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fg448 = foreground_mask(rgb448)
    fg224 = resize_float(fg448, TEMPLATE_SIZE)
    prior = (stats.fg_prior > 0.30).astype(np.float32)
    prior = cv2.dilate(prior, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    fg224_final = np.clip(fg224 * (0.65 + 0.35 * prior), 0, 1)
    fg448_final = upsample224(fg224_final)
    fg448_final = np.clip(fg448_final * fg448, 0, 1)
    fg_u8 = (fg448_final > 0.5).astype(np.uint8)
    inner = cv2.erode(fg_u8, np.ones((9, 9), np.uint8), iterations=1).astype(np.float32)
    boundary = np.clip(fg448_final - inner, 0, 1)
    return fg448_final, inner, boundary


def refine_mask(base: np.ndarray, fg: np.ndarray, inner: np.ndarray, boundary: np.ndarray, template448: np.ndarray) -> np.ndarray:
    gated_base = np.clip(base * fg, 0, 1)
    local_base = suppress_broad_response(base, fg)
    template448 = np.clip(template448 * fg, 0, 1)
    support = np.sqrt(np.clip(gated_base, 0, 1) * np.clip(template448, 0, 1))

    # Base retains recall, template keeps only deviations from normal, and the
    # local branch suppresses broad silhouette responses.
    refined = 0.18 * gated_base + 0.32 * local_base + 0.50 * support + 0.20 * template448
    refined = refined * (0.94 * inner + 0.22 * boundary)
    refined = cap_large_components(np.clip(refined, 0, 1))
    refined = cv2.GaussianBlur(refined, (3, 3), 0)

    # Compress the middle so normal object regions do not remain gray.
    vals = refined[fg > 0.5]
    if vals.size:
        floor = float(np.percentile(vals, 70))
        top = float(np.percentile(vals, 99.4))
        refined = np.clip((refined - floor) / max(top - floor, 1e-6), 0, 1)
        refined = refined**1.35
    return np.clip(refined * fg, 0, 1)


def save_mask(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(array, 0, 1) * 255)), "L").save(path, "PNG", compress_level=1)


def process_one(task):
    data_root, base_root, out_masks, templates, category, sample, view = task
    stats = templates[(category, view)]
    img_path = data_root / "Test_A" / category / sample / f"{view}.png"
    rgb448 = load_rgb(img_path)
    gray224 = resize_float(load_gray(img_path), TEMPLATE_SIZE)
    fg448, inner, boundary = final_foreground(rgb448, stats)
    temp224 = template_deviation(gray224, stats)
    temp448 = upsample224(temp224)
    base = load_mask(base_root / "predicted_masks" / category / sample / f"{view}_mask.png")
    refined = refine_mask(base, fg448, inner, boundary, temp448)
    save_mask(out_masks / category / sample / f"{view}_mask.png", refined)
    return {
        "fg_area": float(fg448.mean()),
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

    templates = build_templates(args.data_root, args.workers)
    tasks = []
    for row in rows:
        category, sample = row["group_folder"].split("/", 1)
        for view in range(5):
            tasks.append((args.data_root, args.base, out_masks, templates, category, sample, view))

    stats = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, stat in enumerate(ex.map(process_one, tasks), 1):
            stats.append(stat)
            if i % 500 == 0:
                print("processed masks", i, flush=True)

    meta = {
        "method": "template-calibrated object-aware refinement",
        "base": str(args.base),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "template_size": TEMPLATE_SIZE,
        "mean_fg_area": float(np.mean([x["fg_area"] for x in stats])),
        "mean_base_area_gt_0_5": float(np.mean([x["base_area"] for x in stats])),
        "mean_refined_area_gt_0_5": float(np.mean([x["refined_area"] for x in stats])),
        "mean_base_mask": float(np.mean([x["base_mean"] for x in stats])),
        "mean_refined_mask": float(np.mean([x["refined_mean"] for x in stats])),
        "template_keys": len(templates),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
