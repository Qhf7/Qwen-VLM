#!/usr/bin/env python3
"""Held-category proxy evaluated on exact uint8 submission mask values."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "v7_vlm_primary_v5"), str(ROOT / "v6_vlm_zero_shot_v4"), str(ROOT / "v5_multiscale_v3")]

from calibration_v6 import quantize_mask
from common import list_samples, load_many, pick_device, set_seed
from metrics import PixelAccumulator, aggregate, category_scores, format_report
from pipeline_v6 import calibrated_output, run_group
from pseudo_v6 import HELDOUT_GENERATORS, make_targets
from runtime_v6 import fit_normal_calibration, load_models, load_or_build_banks
from progress_v6 import ProgressBar
from splits import split_categories


def main(args):
    set_seed(args.seed); device = pick_device()
    checkpoint, dino, model = load_models(args, device)
    all_samples = list_samples(args.data_root / "Train")
    fit, heldout = split_categories(all_samples, checkpoint["heldout_fold"], args.folds)
    selected = []
    for category in sorted({x.category for x in heldout}):
        selected.extend([x for x in heldout if x.category == category][:args.groups_per_category])
    heldout = selected
    banks = load_or_build_banks(args, checkpoint, model, dino, fit, device, args.bank_cache)
    calibration = fit_normal_calibration(args, checkpoint, dino, model, banks, fit, device)
    rng = np.random.default_rng(args.seed + 101)
    proxy_bar = ProgressBar("heldout_proxy", len(heldout), every=1)
    by_category = defaultdict(lambda: {"labels": [], "scores": [], "pixels": PixelAccumulator(args.neg_stride)})
    normal_means, normal_areas, normal_white, boundary_values, scores_all = [], [], [], [], []
    with torch.inference_mode():
        for index, sample in enumerate(heldout):
            normals = load_many(sample.paths)
            for anomalous in (False, True):
                targets = make_targets(normals, rng, force_normal=not anomalous, heldout=True)
                prediction = run_group(
                    [x.image for x in targets], [sample.category] * 5, model, dino, banks,
                    checkpoint["dino_size"], checkpoint["clip_size"], checkpoint["layers"], device,
                )
                # Held categories deliberately use global calibration, matching
                # the unseen-category path on Test B.
                maps = [quantize_mask(calibration.pixel_global.pixel(x)) for x in prediction.raw_maps]
                score = float(calibration.image_global.score(prediction.raw_image_score))
                slot = by_category[sample.category]
                slot["labels"].append(int(anomalous)); slot["scores"].append(score); scores_all.append(score)
                for view, target in enumerate(targets):
                    slot["pixels"].add(maps[view], target.binary_mask)
                    if not anomalous:
                        value = maps[view]
                        normal_means.append(float(value.mean()))
                        normal_areas.append(float((value > 0.5).mean()))
                        normal_white.append(float((value >= 1.0).mean()))
                        boundary = (target.state_mask == 2) | (target.state_mask == 3)
                        if boundary.any():
                            boundary_values.append(float(value[boundary].mean()))
            proxy_bar.update(index + 1, {"category": sample.category}, force=True)
    proxy_bar.close({"category": "complete"})
    per_category = {}
    for category, slot in by_category.items():
        labels, values = slot["pixels"].finish()
        per_category[category] = category_scores(
            np.asarray(slot["labels"]), np.asarray(slot["scores"]), labels, values
        )
    result = aggregate(per_category)
    rounded_scores = np.round(np.asarray(scores_all), 8)
    result.update({
        "heldout_fold": checkpoint["heldout_fold"], "heldout_categories": sorted(by_category),
        "heldout_generators": list(HELDOUT_GENERATORS), "proxy_uses_uint8_png_values": True,
        "normal_mask_mean": float(np.mean(normal_means)),
        "normal_area_gt_0_5": float(np.mean(normal_areas)),
        "normal_white_area": float(np.mean(normal_white)),
        "normal_boundary_mean": float(np.mean(boundary_values)),
        "score_saturation_ge_0_99": float(np.mean(rounded_scores >= 0.99)),
        "score_tie_fraction": 1.0 - len(np.unique(rounded_scores)) / max(len(rounded_scores), 1),
    })
    result["selection_penalty"] = (
        1.5 * max(0.0, result["normal_mask_mean"] - 0.02)
        + 2.0 * max(0.0, result["normal_area_gt_0_5"] - 0.01)
        + 1.0 * max(0.0, result["normal_boundary_mean"] - 0.06)
        + 0.5 * result["score_saturation_ge_0_99"]
        + 0.25 * result["score_tie_fraction"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_report(result, "v6 held-category/held-generator uint8 proxy"), flush=True)
    print(json.dumps({k: result[k] for k in result if k.startswith("normal_") or k.startswith("score_")}, indent=2), flush=True)
    print(f"PROXY_COMPLETE out={args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--weight-dir", type=Path, required=True)
    p.add_argument("--clip-weight", type=Path, required=True); p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank-cache", type=Path); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--groups-per-category", type=int, default=5); p.add_argument("--calib-per-category", type=int, default=8)
    p.add_argument("--reference-budget", type=int, default=128); p.add_argument("--global-reference-budget", type=int, default=2048)
    p.add_argument("--pixel-threshold-quantile", type=float, default=0.995)
    p.add_argument("--pixel-high-quantile", type=float, default=0.9998)
    p.add_argument("--neg-stride", type=int, default=17); p.add_argument("--seed", type=int, default=20260826)
    main(p.parse_args())
