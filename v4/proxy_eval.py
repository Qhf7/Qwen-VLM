#!/usr/bin/env python3
"""Local proxy evaluation: does resolution / sharpening actually help?

Runs the training-free coreset branch at several input resolutions on a
proxy set built from official Train normals plus synthetic defects, and
reports the official metric decomposition for each configuration.

This exists because the submission budget is 3/day.  Every hyper-parameter
that can be chosen offline should be chosen offline.

Caveat, stated up front: synthetic defects are easier than real ones, so
absolute proxy numbers run optimistic.  The harness is for ranking
configurations against each other, particularly on localisation sharpness,
which is where the real board is losing the most points.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from branches import MemoryBank
from common import (
    Calibrator,
    LAYER_SETS,
    MASK_SIZE,
    Sample,
    dino_features,
    greedy_coreset,
    list_samples,
    load_rgb,
    neighbourhood_aggregate,
    pick_device,
    set_seed,
)
from fusion import apply_image_gate, area_prior, prepare_branch_map, sharpen, view_score_from_map
from metrics import PixelAccumulator, aggregate, category_scores, format_report, split_train
from object_prior import apply_prior, descriptor_subset, foreground_grid
from pseudo import synthesize


def build_proxy(proxy_samples, seed: int, anomaly_frac: float = 0.5):
    """Assign each proxy sample a role and, if anomalous, which views carry
    defects.  Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    plan = []
    by_cat = {}
    for s in proxy_samples:
        by_cat.setdefault(s.category, []).append(s)
    for cat in sorted(by_cat):
        items = by_cat[cat]
        n_anom = int(round(len(items) * anomaly_frac))
        flags = np.zeros(len(items), bool)
        flags[:n_anom] = True
        flags = rng.permutation(flags)
        for s, is_anom in zip(items, flags):
            if is_anom:
                # a real defect is visible from a subset of views
                n_views = int(rng.integers(1, 4))
                views = sorted(rng.choice(5, size=n_views, replace=False).tolist())
            else:
                views = []
            plan.append((s, bool(is_anom), views, int(rng.integers(0, 2**31 - 1))))
    return plan


def render_proxy_view(sample: Sample, view: int, defect: bool, seed: int):
    """Return (image, gt_mask_448 or None)."""
    im = load_rgb(sample.paths[view])
    if not defect:
        return im, None
    rng = np.random.default_rng(seed + view * 7919)
    aug, mask = synthesize(im, rng)
    gt = np.asarray(
        mask.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR), np.float32
    ) > 96
    return aug, gt


def fit_memory(model, fit_samples, device, size, layers, budget, agg_radius, batch):
    """Per-(category,view) coreset from Train normals."""
    pools: dict[tuple, list] = {}
    for st in range(0, len(fit_samples), batch):
        chunk = fit_samples[st : st + batch]
        images, keys = [], []
        for s in chunk:
            for v in range(5):
                images.append(load_rgb(s.paths[v]))
                keys.append((s.category, v))
        _, patch, side = dino_features(model, images, size, device, layers)
        for i, key in enumerate(keys):
            fg = foreground_grid(images[i], side)
            p = descriptor_subset(
                neighbourhood_aggregate(patch[i], agg_radius),
                fg,
                fg_stride=2,
                bg_stride=28,
            )
            # deterministic stride subsample keeps memory bounded before coreset
            pools.setdefault(key, []).append(p[::3].astype(np.float16))
        if st % (batch * 10) == 0:
            print(f"  fit {min(st+batch, len(fit_samples))}/{len(fit_samples)}", flush=True)
    bank = MemoryBank(budget_per_key=budget, knn=3)
    all_small = []
    for key, arrs in pools.items():
        x = np.concatenate(arrs).astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        bank.set(key, greedy_coreset(x, budget))
        all_small.append(x[::17])
    g = np.concatenate(all_small)
    bank.set_global(greedy_coreset(g, min(16384, len(g))))
    return bank, side


def calibrate(model, fit_samples, device, size, layers, bank, agg_radius, batch, limit=200):
    """Normal-score quantiles per (category,view) and per category for images."""
    pix = Calibrator(lo_q=90.0, hi_q=99.9)
    img = Calibrator(lo_q=50.0, hi_q=99.0)
    pix_pool: dict[tuple, list] = {}
    img_pool: dict[str, list] = {}
    sub = fit_samples[:limit]
    for st in range(0, len(sub), batch):
        chunk = sub[st : st + batch]
        images, keys = [], []
        for s in chunk:
            for v in range(5):
                images.append(load_rgb(s.paths[v]))
                keys.append((s.category, v))
        _, patch, side = dino_features(model, images, size, device, layers)
        for i, key in enumerate(keys):
            p = neighbourhood_aggregate(patch[i], agg_radius).reshape(-1, patch.shape[-1])
            d = bank.score(key, p, device).reshape(side, side)
            d = apply_prior(d, images[i], background_keep=0.08, boundary_keep=0.55)
            pix_pool.setdefault(key, []).append(d.ravel()[::7])
            img_pool.setdefault(key[0], []).append(view_score_from_map(d))
    for k, v in pix_pool.items():
        pix.fit_key(k, np.concatenate(v))
    pix.fit_fallback(np.concatenate([np.concatenate(v) for v in pix_pool.values()])[::11])
    for k, v in img_pool.items():
        img.fit_key((k,), np.asarray(v))
    img.fit_fallback(np.concatenate([np.asarray(v) for v in img_pool.values()]))
    return pix, img


CONFIGS = [
    # (label, highpass_frac, gamma, gate_strength, area_hardness)
    ("raw (v3-like)", 0.0, 1.0, 0.0, 0.0),
    ("+highpass", 0.14, 1.0, 0.0, 0.0),
    ("+gamma2", 0.14, 2.0, 0.0, 0.0),
    ("+gate", 0.14, 2.0, 0.75, 0.0),
    ("+area", 0.14, 2.0, 0.75, 1.0),
    ("gamma3", 0.14, 3.0, 0.75, 1.0),
]


def evaluate(model, plan, device, size, layers, bank, pix_cal, img_cal, agg_radius, batch):
    """Score the proxy set once, then evaluate all post-processing configs."""
    # cache raw maps so all configs share one forward pass
    cache = []  # (category, is_anom, [ (raw_map, gt) x5 ])
    for st in range(0, len(plan), max(1, batch // 5)):
        chunk = plan[st : st + max(1, batch // 5)]
        images, metas = [], []
        for sample, is_anom, views, seed in chunk:
            for v in range(5):
                defect = is_anom and v in views
                im, gt = render_proxy_view(sample, v, defect, seed)
                images.append(im)
                metas.append((sample.category, v, gt))
        _, patch, side = dino_features(model, images, size, device, layers)
        idx = 0
        for sample, is_anom, views, seed in chunk:
            entry = []
            for v in range(5):
                cat, vv, gt = metas[idx]
                p = neighbourhood_aggregate(patch[idx], agg_radius).reshape(-1, patch.shape[-1])
                d = bank.score((cat, vv), p, device).reshape(side, side)
                d = apply_prior(d, images[idx], background_keep=0.08, boundary_keep=0.55)
                entry.append((d.astype(np.float32), gt))
                idx += 1
            cache.append((sample.category, is_anom, entry))
        if st % 20 == 0:
            print(f"  eval {min(st+len(chunk), len(plan))}/{len(plan)}", flush=True)

    results = {}
    for label, hp, gamma, gate_s, area_h in CONFIGS:
        by_cat: dict[str, dict] = {}
        for cat, is_anom, entry in cache:
            slot = by_cat.setdefault(
                cat, {"il": [], "isc": [], "acc": PixelAccumulator(neg_stride=23)}
            )
            raw_view_scores = [view_score_from_map(d) for d, _ in entry]
            from fusion import aggregate_views

            img_raw = aggregate_views(raw_view_scores)
            img_conf = float(img_cal.apply((cat,), img_raw))
            slot["il"].append(int(is_anom))
            slot["isc"].append(img_raw)
            for v, (d, gt) in enumerate(entry):
                m = prepare_branch_map(d, MASK_SIZE, highpass_frac=hp)
                m = pix_cal.apply((cat, v), m) if hp == 0 else np.clip(m / max(
                    float(pix_cal.apply_unclipped((cat, v), np.percentile(d, 99.9))), 1e-3
                ), 0, 1)
                m = sharpen(m, gamma=gamma, post_blur=0.6)
                if area_h > 0:
                    m = area_prior(m, 0.02, area_h)
                if gate_s > 0:
                    m = apply_image_gate(m, img_conf, gate_s)
                slot["acc"].add(m, gt if gt is not None else np.zeros((MASK_SIZE, MASK_SIZE), bool))
        per_cat = {}
        for cat, slot in by_cat.items():
            pl, ps = slot["acc"].finish()
            per_cat[cat] = category_scores(
                np.asarray(slot["il"]), np.asarray(slot["isc"]), pl, ps
            )
        results[label] = aggregate(per_cat)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--arch", default="dinov2_vits14")
    ap.add_argument("--weight-dir", type=Path, default=Path("weights"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[518, 1036])
    ap.add_argument("--fit-limit", type=int, default=0, help="0 = all")
    ap.add_argument("--proxy-limit", type=int, default=0)
    ap.add_argument("--categories", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--agg-radius", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("v4/proxy_report.json"))
    args = ap.parse_args()

    set_seed()
    device = pick_device()
    print("device", device, flush=True)
    layers = LAYER_SETS[args.arch]

    train = list_samples(args.data_root / "Train")
    if args.categories:
        cats = sorted({s.category for s in train})[: args.categories]
        train = [s for s in train if s.category in cats]
    fit, proxy = split_train(train, holdout_frac=0.25)
    if args.fit_limit:
        fit = fit[: args.fit_limit]
    if args.proxy_limit:
        proxy = proxy[: args.proxy_limit]
    plan = build_proxy(proxy, seed=777)
    print(f"fit={len(fit)} proxy={len(plan)} categories={len({s.category for s in train})}")

    from common import load_dinov2

    model = load_dinov2(args.arch, device, weight_dir=args.weight_dir)

    report = {}
    for size in args.sizes:
        side = size // 14
        print(f"\n=== size {size} -> {side}x{side} grid  ({MASK_SIZE/side:.1f} px/patch at 448)")
        t0 = time.time()
        bank, _ = fit_memory(
            model, fit, device, size, layers, args.budget, args.agg_radius, args.batch
        )
        pix_cal, img_cal = calibrate(
            model, fit, device, size, layers, bank, args.agg_radius, args.batch
        )
        res = evaluate(
            model, plan, device, size, layers, bank, pix_cal, img_cal, args.agg_radius, args.batch
        )
        for label, agg in res.items():
            print(format_report(agg, f"[{size}] {label}"))
        report[str(size)] = {k: v for k, v in res.items()}
        print(f"  elapsed {time.time()-t0:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
