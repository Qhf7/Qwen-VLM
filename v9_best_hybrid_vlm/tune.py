#!/usr/bin/env python3
"""Offline hyper-parameter search on the proxy validation set.

Submissions are limited to 3/day, so every post-processing parameter is
chosen here instead of on the board.  Raw branch score maps are computed
once and cached; the search then only re-runs cheap numpy post-processing,
which makes a few hundred configurations affordable.

Usage (local smoke):
    python v4/tune.py --categories 8 --fit-per-cat 12 --proxy-per-cat 6

Usage (cloud GPU, full):
    python v4/tune.py --arch dinov2_vitb14 --fine-size 1036
"""
from __future__ import annotations

import argparse
import itertools
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
    dino_features,
    greedy_coreset,
    list_samples,
    load_dinov2,
    load_rgb,
    neighbourhood_aggregate,
    pick_device,
    set_seed,
)
from fusion import (
    aggregate_views,
    apply_image_gate,
    area_prior,
    prepare_branch_map,
    sharpen,
    view_score_from_map,
)
from metrics import PixelAccumulator, aggregate, category_scores, format_report, split_train
from object_prior import apply_prior, descriptor_subset, foreground_grid
from pseudo import synthesize


def make_plan(proxy, seed=777, anomaly_frac=0.5):
    rng = np.random.default_rng(seed)
    by_cat = {}
    for s in proxy:
        by_cat.setdefault(s.category, []).append(s)
    plan = []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda s: s.name)
        n_anom = max(1, int(round(len(items) * anomaly_frac)))
        flags = np.zeros(len(items), bool)
        flags[:n_anom] = True
        flags = rng.permutation(flags)
        for s, is_anom in zip(items, flags):
            views = (
                sorted(rng.choice(5, size=int(rng.integers(1, 4)), replace=False).tolist())
                if is_anom
                else []
            )
            plan.append((s, bool(is_anom), views, int(rng.integers(0, 2**31 - 1))))
    return plan


def cache_scores(
    model, plan, fit, device, size, layers, budget, agg_radius, batch, knn=3,
    background_keep=0.08, boundary_keep=0.55,
):
    """Compute memory-branch raw maps for the whole proxy set once."""
    side = size // 14
    # ---- bank from fit normals
    pools = {}
    dim = None
    for st in range(0, len(fit), batch):
        chunk = fit[st : st + batch]
        images, keys = [], []
        for s in chunk:
            for v in range(5):
                images.append(load_rgb(s.paths[v]))
                keys.append((s.category, v))
        _, patch, _ = dino_features(model, images, size, device, layers)
        dim = patch.shape[-1]
        for i, key in enumerate(keys):
            fg = foreground_grid(images[i], side)
            qmap = neighbourhood_aggregate(patch[i], agg_radius)
            q = descriptor_subset(qmap, fg, fg_stride=2, bg_stride=28)
            pools.setdefault(key, []).append(q[::3].astype(np.float16))
        if st % (batch * 8) == 0:
            print(f"  bank {min(st+batch,len(fit))}/{len(fit)}", flush=True)
    bank = MemoryBank(budget_per_key=budget, knn=knn)
    small = []
    for key, arrs in pools.items():
        x = np.concatenate(arrs).astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        bank.set(key, greedy_coreset(x, budget))
        small.append(x[::17])
    g = np.concatenate(small)
    bank.set_global(greedy_coreset(g, min(8192, len(g))))

    # ---- calibration quantiles from fit normals
    pix_pool, img_pool = {}, {}
    for st in range(0, len(fit), batch):
        chunk = fit[st : st + batch]
        images, keys = [], []
        for s in chunk:
            for v in range(5):
                images.append(load_rgb(s.paths[v]))
                keys.append((s.category, v))
        _, patch, _ = dino_features(model, images, size, device, layers)
        for i, key in enumerate(keys):
            q = neighbourhood_aggregate(patch[i], agg_radius).reshape(-1, dim)
            d = bank.score(key, q, device).reshape(side, side)
            d = apply_prior(d, images[i], background_keep=background_keep, boundary_keep=boundary_keep)
            pix_pool.setdefault(key, []).append(d.ravel()[::5])
            img_pool.setdefault(key[0], []).append(view_score_from_map(d))
    pix = Calibrator(lo_q=90.0, hi_q=99.9)
    img = Calibrator(lo_q=50.0, hi_q=99.0)
    for k, v in pix_pool.items():
        pix.fit_key(k, np.concatenate(v))
    pix.fit_fallback(np.concatenate([np.concatenate(v) for v in pix_pool.values()])[::9])
    for k, v in img_pool.items():
        img.fit_key((k,), np.asarray(v))
    img.fit_fallback(np.concatenate([np.asarray(v) for v in img_pool.values()]))

    # ---- raw maps for the proxy set
    cache = []
    per_batch = max(1, batch // 5)
    for st in range(0, len(plan), per_batch):
        chunk = plan[st : st + per_batch]
        images, metas = [], []
        for sample, is_anom, views, seed in chunk:
            for v in range(5):
                im = load_rgb(sample.paths[v])
                if is_anom and v in views:
                    rng = np.random.default_rng(seed + v * 7919)
                    im, mask = synthesize(im, rng)
                    gt = (
                        np.asarray(
                            mask.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR),
                            np.float32,
                        )
                        > 96
                    )
                else:
                    gt = None
                images.append(im)
                metas.append((sample.category, v, gt))
        _, patch, _ = dino_features(model, images, size, device, layers)
        idx = 0
        for sample, is_anom, views, seed in chunk:
            entry = []
            for v in range(5):
                cat, vv, gt = metas[idx]
                q = neighbourhood_aggregate(patch[idx], agg_radius).reshape(-1, dim)
                d = bank.score((cat, vv), q, device).reshape(side, side)
                d = apply_prior(d, images[idx], background_keep=background_keep, boundary_keep=boundary_keep)
                entry.append((d.astype(np.float32), gt))
                idx += 1
            cache.append((sample.category, is_anom, entry))
        if st % (per_batch * 8) == 0:
            print(f"  proxy {min(st+len(chunk),len(plan))}/{len(plan)}", flush=True)
    return cache, pix, img, side


def evaluate_config(cache, pix, img, cfg):
    """Run post-processing for one configuration and return the aggregate."""
    hp = cfg["highpass"]
    by_cat = {}
    for cat, is_anom, entry in cache:
        slot = by_cat.setdefault(
            cat, {"il": [], "isc": [], "acc": PixelAccumulator(neg_stride=29)}
        )
        raw_views = [view_score_from_map(d, cfg["topk_frac"]) for d, _ in entry]
        img_raw = aggregate_views(raw_views, cfg["max_w"])
        conf = float(img.apply((cat,), img_raw))
        slot["il"].append(int(is_anom))
        slot["isc"].append(img_raw)
        for v, (d, gt) in enumerate(entry):
            prepared = prepare_branch_map(d, MASK_SIZE, highpass_frac=hp)
            lo, hi = pix._bounds((cat, v))
            m = np.clip(prepared / max(hi - lo, 1e-6), 0.0, 1.0)
            m = sharpen(m, gamma=cfg["gamma"], floor=cfg["floor"], post_blur=cfg["post_blur"])
            if cfg["area_hardness"] > 0:
                m = area_prior(m, cfg["area_target"], cfg["area_hardness"])
            if cfg["gate"] > 0:
                m = apply_image_gate(m, conf, cfg["gate"])
            slot["acc"].add(m, gt if gt is not None else np.zeros((MASK_SIZE, MASK_SIZE), bool))
    per_cat = {}
    for cat, slot in by_cat.items():
        pl, ps = slot["acc"].finish()
        per_cat[cat] = category_scores(np.asarray(slot["il"]), np.asarray(slot["isc"]), pl, ps)
    return aggregate(per_cat)


BASE = dict(
    highpass=0.0, gamma=1.0, floor=0.0, post_blur=0.6,
    area_target=0.02, area_hardness=0.5, gate=0.0,
    topk_frac=0.002, max_w=0.65,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--arch", default="dinov2_vits14")
    ap.add_argument("--weight-dir", type=Path, default=Path("weights"))
    ap.add_argument("--fine-size", type=int, default=1036)
    ap.add_argument("--categories", type=int, default=0)
    ap.add_argument("--fit-per-cat", type=int, default=0)
    ap.add_argument("--proxy-per-cat", type=int, default=0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--agg-radius", type=int, default=1)
    ap.add_argument("--background-keep", type=float, default=0.08)
    ap.add_argument("--boundary-keep", type=float, default=0.55)
    ap.add_argument("--out", type=Path, default=Path("v4/tune_report.json"))
    args = ap.parse_args()

    set_seed()
    device = pick_device()
    layers = LAYER_SETS[args.arch]
    print(f"device {device} arch {args.arch} size {args.fine_size}", flush=True)

    train = list_samples(args.data_root / "Train")
    if args.categories:
        cats = sorted({s.category for s in train})[: args.categories]
        train = [s for s in train if s.category in cats]
    fit, proxy = split_train(train, holdout_frac=0.3)
    if args.fit_per_cat:
        keep, seen = [], {}
        for s in fit:
            seen[s.category] = seen.get(s.category, 0) + 1
            if seen[s.category] <= args.fit_per_cat:
                keep.append(s)
        fit = keep
    if args.proxy_per_cat:
        keep, seen = [], {}
        for s in proxy:
            seen[s.category] = seen.get(s.category, 0) + 1
            if seen[s.category] <= args.proxy_per_cat:
                keep.append(s)
        proxy = keep
    plan = make_plan(proxy)
    print(f"fit={len(fit)} proxy={len(plan)} cats={len({s.category for s in train})}", flush=True)

    model = load_dinov2(args.arch, device, weight_dir=args.weight_dir)
    t0 = time.time()
    cache, pix, img, side = cache_scores(
        model, plan, fit, device, args.fine_size, layers,
        args.budget, args.agg_radius, args.batch,
        background_keep=args.background_keep,
        boundary_keep=args.boundary_keep,
    )
    print(f"cached in {time.time()-t0:.0f}s, grid {side}x{side}", flush=True)

    results = {}
    base_agg = evaluate_config(cache, pix, img, BASE)
    print(format_report(base_agg, "BASE"))
    results["BASE"] = base_agg

    # coordinate-wise sweeps: cheap and interpretable
    sweeps = {
        "highpass": [0.0, 0.03, 0.07, 0.14, 0.22],
        "gamma": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "floor": [0.0, 0.10, 0.20, 0.30],
        "post_blur": [0.0, 0.6, 1.2, 2.0],
        "gate": [0.0, 0.20, 0.35, 0.55, 0.70],
        "area_hardness": [0.0, 0.25, 0.5, 1.0],
        "area_target": [0.01, 0.02, 0.035, 0.05],
        "topk_frac": [0.0005, 0.001, 0.002, 0.005, 0.01],
        "max_w": [0.5, 0.65, 0.8],
    }
    best = dict(BASE)
    for key, values in sweeps.items():
        row = {}
        for val in values:
            cfg = dict(best)
            cfg[key] = val
            agg = evaluate_config(cache, pix, img, cfg)
            row[str(val)] = agg
            print(
                f"  {key}={val}: S_seg {agg['s_seg']:.4f} S_cls {agg['s_cls']:.4f} "
                f"SCORE_A {agg['score_a']:.3f}",
                flush=True,
            )
        # pick the value maximising the board score
        bestval = max(values, key=lambda v: row[str(v)]["score_a"])
        best[key] = bestval
        results[f"sweep_{key}"] = row
        print(f"  -> {key} := {bestval}", flush=True)

    final = evaluate_config(cache, pix, img, best)
    print(format_report(final, "BEST"))
    results["BEST"] = {"config": best, "agg": final}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
