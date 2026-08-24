#!/usr/bin/env python3
"""Held-category and held-generator proxy using the exact final inference path."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "v6_vlm_zero_shot_v4"
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(V4)); sys.path.insert(2, str(V3))

from calibration_v5 import assert_category_coverage, fit_calibration, stratified_samples
from clip_dense import clip_preprocess_tensor, load_clip
from common import list_samples, load_dinov2, load_many, pick_device, set_seed
from metrics import PixelAccumulator, aggregate, category_scores, format_report
from model_v5 import VLMPrimaryV5, load_trusted_checkpoint
from pipeline_v5 import calibrated_output, run_group
from pseudo_v5 import DEFAULT_HELDOUT_GENERATORS, make_group_targets_v5
from progress_v5 import ProgressReporter
from reference_v5 import build_reference_bank
from splits import split_categories


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


def fit_normal_calibration(args, checkpoint, dino, model, bank, fit, device):
    pixels, images = defaultdict(list), defaultdict(list)
    selected = stratified_samples(fit, args.calib_per_category, args.seed)
    expected = sorted({sample.category for sample in fit})
    progress = ProgressReporter(
        "proxy_calibration", len(selected), args.out.with_suffix(".calibration_progress.json")
    )
    with torch.inference_mode():
        for sample in selected:
            normal = load_many(sample.paths)
            pred = run_group(
                normal, [sample.category] * 5, model, dino, bank,
                checkpoint["dino_size"], checkpoint["clip_size"], checkpoint["layers"], device,
                args.reference_scale,
            )
            pixels[sample.category].append(
                np.concatenate([score.ravel()[::37] for score in pred.raw_maps]).astype(np.float32)
            )
            images[sample.category].append(pred.raw_image_score)
            if len(images[sample.category]) == 1 or sum(map(len, images.values())) % 10 == 0:
                progress.update(sum(map(len, images.values())), {"category": sample.category})
    calibration = fit_calibration(pixels, images, args.pixel_lo_quantile, args.pixel_hi_quantile)
    assert_category_coverage(calibration, expected)
    return calibration


def main(args):
    set_seed(args.seed)
    device = pick_device()
    checkpoint, dino, model = load_models(args, device)
    all_samples = list_samples(args.data_root / "Train")
    fit, heldout = split_categories(all_samples, checkpoint["heldout_fold"], args.folds)
    if args.groups_per_category:
        selected = []
        for category in sorted({sample.category for sample in heldout}):
            category_samples = [sample for sample in heldout if sample.category == category]
            selected.extend(category_samples[:args.groups_per_category])
        heldout = selected
    bank = build_reference_bank(
        model, clip_preprocess_tensor, fit, checkpoint["clip_size"], device,
        args.reference_budget, args.global_reference_budget, args.seed,
        args.out.with_suffix(".reference_progress.json"), "proxy_reference_bank",
    )
    calibration = fit_normal_calibration(args, checkpoint, dino, model, bank, fit, device)
    rng = np.random.default_rng(args.seed + 97)
    by_category = defaultdict(lambda: {"labels": [], "scores": [], "pixels": PixelAccumulator(args.neg_stride)})
    progress = ProgressReporter("heldout_proxy", len(heldout), args.out.with_suffix(".progress.json"))

    with torch.inference_mode():
        for index, sample in enumerate(heldout):
            normal = load_many(sample.paths)
            for anomalous in (False, True):
                targets = make_group_targets_v5(
                    normal,
                    rng,
                    force_normal=not anomalous,
                    generators=DEFAULT_HELDOUT_GENERATORS,
                )
                images = [target.image for target in targets]
                pred = run_group(
                    images, [sample.category] * 5, model, dino, bank,
                    checkpoint["dino_size"], checkpoint["clip_size"], checkpoint["layers"], device,
                    args.reference_scale,
                )
                # Held-out categories intentionally use the category-agnostic
                # normal calibration, exactly matching B-board unseen fallback.
                maps = [calibration.pixel_global.apply(score).astype(np.float32) for score in pred.raw_maps]
                maps = [np.where(score < args.zero_floor, 0.0, score) for score in maps]
                score = float(calibration.image_global.apply(pred.raw_image_score))
                slot = by_category[sample.category]
                slot["labels"].append(int(anomalous))
                slot["scores"].append(score)
                for view, target in enumerate(targets):
                    slot["pixels"].add(maps[view], target.binary_mask)
            if index % 5 == 0 or index + 1 == len(heldout):
                progress.update(index + 1, {"category": sample.category, "evaluations": 2 * (index + 1)})
    per_category = {}
    for category, slot in by_category.items():
        pixel_labels, pixel_scores = slot["pixels"].finish()
        per_category[category] = category_scores(
            np.asarray(slot["labels"]), np.asarray(slot["scores"]), pixel_labels, pixel_scores
        )
    result = aggregate(per_category)
    result.update({
        "heldout_fold": checkpoint["heldout_fold"],
        "heldout_categories": sorted(by_category),
        "heldout_generators": list(DEFAULT_HELDOUT_GENERATORS),
        "fit_calibration_categories": len(calibration.categories),
        "proxy_uses_final_pipeline": True,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_report(result, "VLM-primary v5 held-category + held-generator proxy"), flush=True)
    print(f"PROXY_COMPLETE out={args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--groups-per-category", type=int, default=5)
    parser.add_argument("--calib-per-category", type=int, default=10)
    parser.add_argument("--reference-budget", type=int, default=1024)
    parser.add_argument("--global-reference-budget", type=int, default=8192)
    parser.add_argument("--reference-scale", type=float, default=0.18)
    parser.add_argument("--pixel-lo-quantile", type=float, default=0.90)
    parser.add_argument("--pixel-hi-quantile", type=float, default=0.9995)
    parser.add_argument("--zero-floor", type=float, default=0.10)
    parser.add_argument("--neg-stride", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260825)
    main(parser.parse_args())
