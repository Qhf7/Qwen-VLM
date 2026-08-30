#!/usr/bin/env python3
"""Train the v12 candidate-region ranker from official Train images only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from branches import PseudoHead, ReconstructionBranch
from calibration_v10 import apply_pixel_calibration
from common import LAYER_SETS, MASK_SIZE, list_samples, load_dinov2, load_many, pick_device, set_seed
from fusion import fuse_branches, prepare_branch_map
from infer_stream_memory import (
    BRANCHES,
    ScaleRunner,
    branch_maps,
    build_memory,
    calibrate_branches,
    load_ckpt_head,
    load_ckpt_recon,
    stratified_samples,
)
from object_prior import apply_prior
from pseudo import synthesize
from region_ranker import (
    candidate_target, fit_ranker, generate_candidates, region_ranked_mask, save_ranker,
)
from metrics import PixelAccumulator, average_precision, auroc, f1_max


def candidate_metrics(model, metadata, features, targets):
    if not len(features):
        return {"ap": 0.0, "f1max": 0.0, "positive": 0, "negative": 0}
    from sklearn.metrics import average_precision_score
    mean = np.asarray(metadata["mean"], np.float32)
    std = np.asarray(metadata["std"], np.float32)
    with torch.inference_mode():
        probability = torch.sigmoid(
            model(torch.from_numpy((features - mean) / std).float())
        ).numpy()
    labels = targets >= 0.18
    ap = float(average_precision_score(labels.astype(np.uint8), probability)) if labels.any() else 0.0
    f1max = 0.0
    for threshold in np.unique(np.concatenate([probability, [metadata["reject_threshold"]]])):
        predicted = probability >= threshold
        tp = float(np.logical_and(predicted, labels).sum())
        fp = float(np.logical_and(predicted, ~labels).sum())
        fn = float(np.logical_and(~predicted, labels).sum())
        f1max = max(f1max, 2 * tp / max(2 * tp + fp + fn, 1.0))
    return {
        "ap": ap, "f1max": float(f1max), "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
        "accepted_at_reject_threshold": float((probability >= metadata["reject_threshold"]).mean()),
    }


def heldout_pixel_proxy(
    samples, runners, banks, recon, head, pix_cal, weights, scale_weights,
    ranker, metadata, batch_groups, background_keep, boundary_keep, seed=20260903,
):
    accumulator = PixelAccumulator(neg_stride=1, max_pos=2_000_000)
    rng = np.random.default_rng(seed)
    selected_areas, rejected = [], 0
    started = time.time()
    for start in range(0, len(samples), batch_groups):
        chunk = samples[start : start + batch_groups]
        images, keys, truths, _ = _pseudo_batch(chunk, rng)
        dense, evidence = _fused_batch(
            runners, images, keys, banks, recon, head, pix_cal, weights, scale_weights,
            background_keep, boundary_keep,
        )
        for image, score, branches, truth in zip(images, dense, evidence, truths):
            prediction, diagnostics = region_ranked_mask(score, branches, image, ranker, metadata)
            accumulator.add(prediction, truth > 0.20)
            selected_areas.append(float(diagnostics["selected_area"]))
            rejected += int(diagnostics["selected_count"] == 0)
        done = min(start + len(chunk), len(samples))
        elapsed = time.time() - started
        eta = elapsed / max(done, 1) * max(len(samples) - done, 0)
        print(
            f"V12_PROGRESS heldout_pixel_proxy {done}/{len(samples)} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True,
        )
    labels, scores = accumulator.finish()
    return {
        "p_auroc": auroc(labels, scores), "p_ap": average_precision(labels, scores),
        "p_f1max": f1_max(labels, scores),
        "mean_selected_area": float(np.mean(selected_areas)) if selected_areas else 0.0,
        "rejected_view_fraction": rejected / max(len(selected_areas), 1),
        "views": len(selected_areas), "negative_stride": 1,
    }


def category_split(samples, seed: int = 20260828):
    """Hold out 20% of product categories, then split seen normals 75/25.

    The held-out categories never enter ranker fitting, memory fitting or the
    rejection threshold.  Their pseudo anomalies form a zero-shot proxy for B.
    """
    by_category = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    rng = np.random.default_rng(seed)
    categories = sorted(by_category)
    heldout_count = max(1, round(len(categories) * 0.20))
    heldout_categories = set(rng.choice(categories, heldout_count, replace=False).tolist())
    fit, normal_calibration, proxy = [], [], []
    for category in sorted(by_category):
        values = sorted(by_category[category], key=lambda x: x.name)
        if category in heldout_categories:
            proxy.extend(values)
            continue
        order = rng.permutation(len(values))
        n_cal = max(1, round(len(values) * 0.25))
        normal_calibration.extend(values[int(i)] for i in order[:n_cal])
        fit.extend(values[int(i)] for i in order[n_cal:])
    return fit, normal_calibration, proxy


def _fused_batch(
    runners, images, keys, banks, recon, head, pix_cal, weights, scale_weights,
    background_keep: float, boundary_keep: float,
):
    per_scale = {
        runner.size: branch_maps(
            runner, images, keys, banks.get(runner.size), recon.get(runner.size), head.get(runner.size)
        )
        for runner in runners
    }
    dense_output, branch_output = [], []
    for index, key in enumerate(keys):
        fused_scales = []
        branch_scales = {name: [] for name in BRANCHES}
        for runner in runners:
            maps = per_scale[runner.size]
            calibrated = {}
            for name in BRANCHES:
                values = maps[name]
                if values is None:
                    calibrated[name] = None
                    continue
                prepared = prepare_branch_map(values[index], MASK_SIZE, highpass_frac=0.0)
                prepared = apply_prior(
                    prepared, images[index], background_keep=background_keep,
                    boundary_keep=boundary_keep,
                )
                lo, hi = pix_cal[(name, runner.size)]._bounds(key)
                calibrated[name] = apply_pixel_calibration(prepared, lo, hi)
                branch_scales[name].append((calibrated[name], scale_weights[runner.size]))
            fused_scales.append(
                (
                    fuse_branches(
                        calibrated["recon"], calibrated["memory"], calibrated["pseudo"],
                        weights=weights, mode="weighted",
                    ),
                    scale_weights[runner.size],
                )
            )
        total = sum(weight for _, weight in fused_scales) or 1.0
        dense = sum(values * (weight / total) for values, weight in fused_scales)
        dense = apply_prior(
            dense, images[index], background_keep=background_keep, boundary_keep=boundary_keep
        )
        evidence = []
        for name in BRANCHES:
            entries = branch_scales[name]
            if not entries:
                continue
            branch_total = sum(weight for _, weight in entries) or 1.0
            evidence.append(sum(values * (weight / branch_total) for values, weight in entries))
        dense_output.append(np.asarray(dense, np.float32))
        branch_output.append(evidence)
    return dense_output, branch_output


def _pseudo_batch(samples, rng):
    images, keys, truths, group_ids = [], [], [], []
    for group_id, sample in enumerate(samples):
        originals = load_many(sample.paths)
        for view, image in enumerate(originals):
            augmented, mask = synthesize(image, rng, n_defects=(1, 3))
            images.append(augmented)
            keys.append((sample.category, view))
            truth = np.asarray(
                mask.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR), np.float32
            ) / 255.0
            truths.append(truth)
            group_ids.append(group_id)
    return images, keys, truths, group_ids


def collect_pseudo_candidates(
    samples, runners, banks, recon, head, pix_cal, weights, scale_weights,
    batch_groups, seed, background_keep, boundary_keep, log_prefix,
):
    rng = np.random.default_rng(seed)
    feature_rows, targets, groups = [], [], []
    started = time.time()
    total_groups = len(samples)
    next_group_id = 0
    for start in range(0, total_groups, batch_groups):
        chunk = samples[start : start + batch_groups]
        images, keys, truths, local_groups = _pseudo_batch(chunk, rng)
        dense, evidence = _fused_batch(
            runners, images, keys, banks, recon, head, pix_cal, weights, scale_weights,
            background_keep, boundary_keep,
        )
        for image, score, branches, truth, group_id in zip(images, dense, evidence, truths, local_groups):
            for candidate in generate_candidates(score, branches, image):
                feature_rows.append(candidate.features)
                targets.append(candidate_target(candidate.mask, truth))
                groups.append(next_group_id + group_id)
        next_group_id += len(chunk)
        elapsed = time.time() - started
        done = min(start + len(chunk), total_groups)
        eta = elapsed / max(done, 1) * max(total_groups - done, 0)
        print(
            f"V12_PROGRESS {log_prefix} {done}/{total_groups} candidates={len(feature_rows)} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True,
        )
    if not feature_rows:
        return np.empty((0, 22), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    return np.stack(feature_rows), np.asarray(targets, np.float32), np.asarray(groups, np.int64)


def collect_normal_candidates(
    samples, runners, banks, recon, head, pix_cal, weights, scale_weights,
    batch_groups, background_keep, boundary_keep, return_view_ids=False,
):
    features, view_ids = [], []
    next_view_id = 0
    started = time.time()
    for start in range(0, len(samples), batch_groups):
        chunk = samples[start : start + batch_groups]
        images, keys = [], []
        for sample in chunk:
            for view, image in enumerate(load_many(sample.paths)):
                images.append(image)
                keys.append((sample.category, view))
        dense, evidence = _fused_batch(
            runners, images, keys, banks, recon, head, pix_cal, weights, scale_weights,
            background_keep, boundary_keep,
        )
        for image, score, branches in zip(images, dense, evidence):
            candidates = generate_candidates(score, branches, image)
            features.extend(candidate.features for candidate in candidates)
            view_ids.extend([next_view_id] * len(candidates))
            next_view_id += 1
        elapsed = time.time() - started
        done = min(start + len(chunk), len(samples))
        eta = elapsed / max(done, 1) * max(len(samples) - done, 0)
        print(
            f"V12_PROGRESS normal_rejection {done}/{len(samples)} candidates={len(features)} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True,
        )
    matrix = np.stack(features) if features else np.empty((0, 22), np.float32)
    if return_view_ids:
        return matrix, np.asarray(view_ids, np.int64)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--arch", default="dinov2_vitb14")
    parser.add_argument("--fine-size", type=int, default=1036)
    parser.add_argument("--coarse-size", type=int, default=518)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--batch-groups", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--global-budget", type=int, default=16384)
    parser.add_argument("--calib-limit", type=int, default=300)
    parser.add_argument("--rank-limit", type=int, default=300)
    parser.add_argument("--normal-limit", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--background-keep", type=float, default=0.08)
    parser.add_argument("--boundary-keep", type=float, default=0.55)
    args = parser.parse_args()
    if args.output.exists() or args.reference_cache.exists():
        raise FileExistsError(
            f"refusing to overwrite v12 ranker/cache: {args.output} / {args.reference_cache}"
        )

    set_seed()
    device = pick_device()
    layers = LAYER_SETS[args.arch]
    model = load_dinov2(args.arch, device, args.weight_dir)
    samples = list_samples(args.data_root / "Train")
    fit, normal_calibration, proxy = category_split(samples)
    memory_fit = list(fit)
    rank_fit = stratified_samples(fit, args.rank_limit, seed=20260828)
    normal_calibration = stratified_samples(normal_calibration, args.normal_limit, seed=20260829)
    proxy = stratified_samples(proxy, max(50, args.normal_limit // 2), seed=20260830)
    print(
        f"V12_SPLIT memory_fit={len(memory_fit)} rank_fit={len(rank_fit)} "
        f"normal_calibration={len(normal_calibration)} "
        f"proxy={len(proxy)} categories={len({x.category for x in samples})}", flush=True,
    )

    sizes = sorted({args.fine_size, args.coarse_size}, reverse=True)
    runners = [ScaleRunner(model, size, device, layers, agg_radius=1) for size in sizes]
    recon = {runner.size: None for runner in runners}
    head = {runner.size: None for runner in runners}
    trained_meta = json.loads((args.ckpt / "train_meta.json").read_text()) if (args.ckpt / "train_meta.json").exists() else {}
    trained_size = int(trained_meta.get("size", args.fine_size))
    recon_branch = load_ckpt_recon(args.ckpt / "recon.pt", device)
    head_branch = load_ckpt_head(args.ckpt / "head.pt", device)
    target_size = trained_size if trained_size in recon else sizes[0]
    recon[target_size] = recon_branch
    head[target_size] = head_branch

    # Memory and calibration never see Test A/B.  The proxy split remains
    # outside ranker optimisation and is used only for synthetic diagnostics.
    # Keep rejection-calibration normals out of both memory fitting and pixel
    # calibration.  Otherwise their own descriptors make their candidate
    # scores unrealistically easy and the rejection threshold is too low.
    banks = {}
    for runner in runners:
        print(f"V12_STAGE memory_{runner.size}", flush=True)
        banks[runner.size] = build_memory(
            runner, memory_fit, args.budget, args.global_budget, args.batch, knn=3
        )
    print("V12_STAGE train_normal_calibration", flush=True)
    pix_cal, _ = calibrate_branches(
        runners, memory_fit, banks, recon, head, args.batch, args.calib_limit
    )
    weights = (0.40, 0.35, 0.25)
    scale_weights = {sizes[0]: 0.65}
    for size in sizes[1:]:
        scale_weights[size] = 0.35 / max(1, len(sizes) - 1)

    print("V12_STAGE pseudo_candidate_collection", flush=True)
    features, targets, group_ids = collect_pseudo_candidates(
        rank_fit, runners, banks, recon, head, pix_cal, weights, scale_weights,
        args.batch_groups, 20260831, args.background_keep, args.boundary_keep, "pseudo_candidates",
    )
    print("V12_STAGE hard_normal_candidate_collection", flush=True)
    hard_normal_features = collect_normal_candidates(
        stratified_samples(rank_fit, min(len(rank_fit), 150), seed=20260902),
        runners, banks, recon, head, pix_cal, weights, scale_weights,
        args.batch_groups, args.background_keep, args.boundary_keep,
    )
    if len(hard_normal_features):
        first_group = int(group_ids.max()) + 1 if len(group_ids) else 0
        hard_group_ids = np.arange(first_group, first_group + len(hard_normal_features), dtype=np.int64)
        features = np.concatenate([features, hard_normal_features], axis=0)
        targets = np.concatenate([targets, np.zeros(len(hard_normal_features), np.float32)])
        group_ids = np.concatenate([group_ids, hard_group_ids])
    print("V12_STAGE heldout_normal_rejection", flush=True)
    normal_features, normal_view_ids = collect_normal_candidates(
        normal_calibration, runners, banks, recon, head, pix_cal, weights, scale_weights,
        args.batch_groups, args.background_keep, args.boundary_keep, return_view_ids=True,
    )
    print("V12_STAGE ranker_optimisation", flush=True)
    ranker, metadata = fit_ranker(
        features, targets, group_ids, normal_features, normal_view_ids=normal_view_ids,
        epochs=args.epochs
    )
    metadata.update(
        {
            "method": "v12 Train-only rejectable candidate-region ranking",
            "memory_fit_groups": len(memory_fit), "rank_fit_groups": len(rank_fit),
            "normal_calibration_groups": len(normal_calibration),
            "proxy_groups_reserved": len(proxy), "arch": args.arch, "sizes": sizes,
            "positive_candidates": int((targets >= 0.18).sum()),
            "negative_candidates": int((targets < 0.18).sum()),
            "external_data": False, "external_specialized_weights": False,
        }
    )
    print("V12_STAGE heldout_pseudo_proxy", flush=True)
    proxy_features, proxy_targets, _ = collect_pseudo_candidates(
        proxy, runners, banks, recon, head, pix_cal, weights, scale_weights,
        args.batch_groups, 20260901, args.background_keep, args.boundary_keep,
        "heldout_pseudo_proxy",
    )
    metadata["heldout_proxy_candidate_metrics"] = candidate_metrics(
        ranker, metadata, proxy_features, proxy_targets
    )
    print(
        "V12_PROXY " + json.dumps(metadata["heldout_proxy_candidate_metrics"], sort_keys=True),
        flush=True,
    )
    print("V12_STAGE heldout_category_pixel_proxy", flush=True)
    pixel_proxy_samples = stratified_samples(proxy, min(len(proxy), 50), seed=20260904)
    metadata["heldout_category_pixel_metrics"] = heldout_pixel_proxy(
        pixel_proxy_samples, runners, banks, recon, head, pix_cal, weights, scale_weights,
        ranker, metadata, args.batch_groups, args.background_keep, args.boundary_keep,
    )
    print(
        "V12_PIXEL_PROXY " + json.dumps(metadata["heldout_category_pixel_metrics"], sort_keys=True),
        flush=True,
    )
    save_ranker(args.output, ranker, metadata)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"V12_RANKER_COMPLETE output={args.output}", flush=True)

    # Build the production reference once with all 50 official Train classes.
    # Inference loads this cache instead of repeating an identical multi-hour
    # memory and calibration pass.
    print("V12_STAGE final_all_train_reference", flush=True)
    final_banks = {}
    for runner in runners:
        print(f"V12_STAGE final_memory_{runner.size}", flush=True)
        final_banks[runner.size] = build_memory(
            runner, samples, args.budget, args.global_budget, args.batch, knn=3
        )
    final_pix, final_img = calibrate_branches(
        runners, samples, final_banks, recon, head, args.batch, args.calib_limit
    )
    args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "banks": final_banks, "pix_cal": final_pix, "img_cal": final_img,
            "sizes": sizes, "arch": args.arch, "train_groups": len(samples),
            "official_train_only": True, "test_labels_used": False,
        },
        args.reference_cache,
    )
    print(f"V12_REFERENCE_COMPLETE output={args.reference_cache}", flush=True)


if __name__ == "__main__":
    main()
