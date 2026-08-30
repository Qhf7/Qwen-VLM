#!/usr/bin/env python3
"""v4 inference: three-branch fusion, multi-scale, submission builder.

Pipeline per view:
  1. DINOv2 multi-layer patch features at a fine scale (localisation) and a
     coarse scale (context).
  2. Branch scores: reconstruction distance, coreset k-NN distance,
     pseudo-head probability.
  3. Per-(category,view) calibration against Train-normal quantiles.
  4. High-pass -> fuse -> gamma sharpen -> area prior -> image-level gate.
  5. 448x448 uint8 mask.

Image score per group: per-branch top-k evidence, per-category calibrated,
combined with a CLIP semantic term when available, aggregated over views.

Compliance: official Train data for all banks/calibration/training; generic
public DINOv2 and CLIP weights; no test labels; the optional transductive
bank uses unlabelled test images only and is off by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from branches import MemoryBank, PseudoHead, ReconstructionBranch, reconstruction_score, robust_bank_from_samples
from common import (
    Calibrator,
    LAYER_SETS,
    MASK_SIZE,
    dino_features,
    greedy_coreset,
    list_samples,
    load_dinov2,
    load_many,
    load_rgb,
    neighbourhood_aggregate,
    per_layer_features,
    pick_device,
    set_seed,
)
from fusion import (
    aggregate_views,
    apply_image_gate,
    area_prior,
    combine_image_scores,
    fuse_branches,
    prepare_branch_map,
    sharpen,
    view_score_from_map,
)
from object_prior import apply_prior, descriptor_subset, foreground_grid
from calibration_v10 import apply_pixel_calibration
from region_ranker import load_ranker, region_ranked_mask


# --------------------------------------------------------------------------
# feature extraction for one batch of images at one scale
# --------------------------------------------------------------------------


class ScaleRunner:
    """Holds everything needed to score images at a single input scale."""

    def __init__(self, model, size, device, layers, agg_radius=1):
        self.model = model
        self.size = size
        self.device = device
        self.layers = layers
        self.agg_radius = agg_radius
        self.side = size // 14

    def features(self, images):
        maps, side = per_layer_features(self.model, images, self.size, self.device, self.layers)
        cat = torch.cat([F.normalize(m, dim=-1) for m in maps], dim=-1)
        cat = F.normalize(cat, dim=-1)
        return maps, cat, side

    def memory_query(self, cat_feat_np):
        return neighbourhood_aggregate(cat_feat_np, self.agg_radius)


def _append_reservoir(store: dict, key, arr: np.ndarray, cap: int, rng: np.random.Generator):
    """Keep a bounded, unbiased-ish descriptor reservoir per key.

    The original infer.py collected all Train descriptors first and only then
    ran the coreset selection.  At 1036px / 74x74 / multi-layer DINOv2 this can
    spike CPU memory near the end of the 1000-sample bank build.  This version
    preserves the same modelling idea and command-line hyper-parameters, but
    streams descriptors into a bounded reservoir before the final greedy coreset.
    """
    if arr.size == 0:
        return
    cur = store.get(key)
    if cur is None:
        store[key] = arr[-cap:].copy() if len(arr) > cap else arr.copy()
        return
    merged = np.concatenate([cur, arr], axis=0)
    if len(merged) <= cap:
        store[key] = merged
        return
    idx = rng.choice(len(merged), size=cap, replace=False)
    store[key] = merged[idx]


def build_memory(runner: ScaleRunner, samples, budget, global_budget, batch, knn=3, log=True):
    pools: dict[tuple, np.ndarray] = {}
    global_pool: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260819 + int(runner.size))
    per_key_cap = max(budget * 4, budget + 256)
    global_cap = max(global_budget * 4, global_budget + 1024)
    dim = None
    for st in range(0, len(samples), batch):
        chunk = samples[st : st + batch]
        images, keys = [], []
        for s in chunk:
            imgs = load_many(s.paths)
            for v, im in enumerate(imgs):
                images.append(im)
                keys.append((s.category, v))
        _, cat, side = runner.features(images)
        arr = cat.cpu().numpy()
        dim = arr.shape[-1]
        for i, key in enumerate(keys):
            fg = foreground_grid(images[i], side)
            qmap = runner.memory_query(arr[i])
            q = descriptor_subset(qmap, fg, fg_stride=2, bg_stride=28)
            _append_reservoir(
                pools, key, q[::3].astype(np.float16), per_key_cap, rng
            )
            _append_reservoir(
                global_pool, "global", q[::17].astype(np.float16), global_cap, rng
            )
        if log and st % (batch * 10) == 0:
            print(f"  memory {min(st+batch, len(samples))}/{len(samples)}", flush=True)
    bank = MemoryBank(budget_per_key=budget, global_budget=global_budget, knn=knn)
    for key, arrs in pools.items():
        x = arrs.astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        bank.set(key, greedy_coreset(x, budget))
    g = global_pool.get("global")
    if g is None:
        small = []
        for arrs in pools.values():
            x = arrs.astype(np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
            small.append(x[::17])
        g = np.concatenate(small)
    else:
        g = g.astype(np.float32)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    bank.set_global(greedy_coreset(g, min(global_budget, len(g))))
    return bank


def branch_maps(
    runner: ScaleRunner,
    images,
    keys,
    bank: MemoryBank | None,
    recon: ReconstructionBranch | None,
    head: PseudoHead | None,
):
    """Return dict of raw per-image score maps for each available branch."""
    maps, cat, side = runner.features(images)
    arr = cat.cpu().numpy()
    dim = arr.shape[-1]
    out = {"side": side, "recon": None, "memory": None, "pseudo": None}

    if recon is not None:
        feats = [m.reshape(m.shape[0], -1, m.shape[-1]) for m in maps]
        with torch.inference_mode():
            recons = recon(feats)
            s = reconstruction_score(feats, recons, side)
        out["recon"] = s.cpu().numpy()

    if bank is not None:
        mem = np.empty((len(images), side, side), np.float32)
        for i, key in enumerate(keys):
            q = runner.memory_query(arr[i]).reshape(-1, dim)
            mem[i] = bank.score(key, q, runner.device).reshape(side, side)
        out["memory"] = mem

    if head is not None:
        with torch.inference_mode():
            x = cat.permute(0, 3, 1, 2).contiguous()
            logits = head(x)
            out["pseudo"] = torch.sigmoid(logits.float()).cpu().numpy()

    return out


# --------------------------------------------------------------------------
# calibration on Train normals
# --------------------------------------------------------------------------

BRANCHES = ("recon", "memory", "pseudo")


def stratified_samples(samples, limit: int, seed: int = 20260827):
    """Select a deterministic, category-balanced normal calibration subset."""
    if not limit or limit >= len(samples):
        return list(samples)
    by_category = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    rng = np.random.default_rng(seed)
    categories = sorted(by_category)
    base, remainder = divmod(limit, len(categories))
    selected = []
    for index, category in enumerate(categories):
        values = sorted(by_category[category], key=lambda item: item.name)
        count = min(len(values), base + int(index < remainder))
        if count:
            indices = np.sort(rng.choice(len(values), count, replace=False))
            selected.extend(values[int(i)] for i in indices)
    return selected


def calibrate_branches(runners, samples, banks, recon, head, batch, limit, log=True):
    """Per-(category,view) pixel quantiles and per-category image quantiles."""
    pix = {(b, r.size): Calibrator(lo_q=90.0, hi_q=99.9) for b in BRANCHES for r in runners}
    img = {b: Calibrator(lo_q=50.0, hi_q=99.0) for b in BRANCHES}
    pix_pool: dict[tuple, list] = {}
    img_pool: dict[tuple, list] = {}
    sub = stratified_samples(samples, limit)
    for st in range(0, len(sub), batch):
        chunk = sub[st : st + batch]
        images, keys = [], []
        for s in chunk:
            for v, im in enumerate(load_many(s.paths)):
                images.append(im)
                keys.append((s.category, v))
        per_scale = {}
        for r in runners:
            per_scale[r.size] = branch_maps(r, images, keys, banks.get(r.size), recon.get(r.size), head.get(r.size))
        for b in BRANCHES:
            for r in runners:
                m = per_scale[r.size][b]
                if m is None:
                    continue
                for i, key in enumerate(keys):
                    fg = foreground_grid(images[i], m[i].shape[0])
                    mm = apply_prior(m[i], images[i], background_keep=0.10, boundary_keep=0.62)
                    pix_pool.setdefault((b, r.size, key), []).append(mm[fg > 0.10].ravel()[::7])
                    img_pool.setdefault((b, key[0]), []).append(view_score_from_map(mm))
        if log and st % (batch * 10) == 0:
            print(f"  calibrate {min(st+batch, len(sub))}/{len(sub)}", flush=True)
    for (b, size, key), vals in pix_pool.items():
        pix[(b, size)].fit_key(key, np.concatenate(vals))
    for b in BRANCHES:
        for r in runners:
            pool = [np.concatenate(v) for (bb, size, _), v in pix_pool.items() if bb == b and size == r.size]
            if pool:
                pix[(b, r.size)].fit_fallback(np.concatenate(pool)[::13])
    for (b, cat), vals in img_pool.items():
        img[b].fit_key((cat,), np.asarray(vals))
    for b in BRANCHES:
        pool = [np.asarray(v) for (bb, _), v in img_pool.items() if bb == b]
        if pool:
            img[b].fit_fallback(np.concatenate(pool))
    return pix, img


# --------------------------------------------------------------------------
# CLIP semantic term (optional)
# --------------------------------------------------------------------------


def build_clip(weight: Path | None, device):
    if weight is None or not Path(weight).exists():
        return None, None
    import clip

    model, preprocess = clip.load(str(weight), device=str(device), jit=False)
    model.eval()
    return model, preprocess


def clip_text_bank(model, categories, device):
    import clip

    bank = {}
    with torch.inference_mode():
        for cat in categories:
            name = cat.replace("_", " ")
            normal = [
                f"a photo of a normal {name} industrial part",
                f"a flawless {name} component",
                f"an intact clean {name}",
            ]
            abnormal = [
                f"a photo of a defective {name} industrial part",
                f"a {name} with a scratch",
                f"a {name} with a dent or crack",
                f"a {name} with a stain or contamination",
                f"a damaged broken {name}",
            ]
            z = model.encode_text(clip.tokenize(normal + abnormal).to(device))
            z = F.normalize(z.float(), dim=-1)
            bank[cat] = (z[: len(normal)].mean(0), z[len(normal) :].mean(0))
    return bank


@torch.inference_mode()
def clip_scores(model, preprocess, images, cats, text_bank, device):
    """Window-based CLIP anomaly evidence (WinCLIP-style local matching)."""
    tiles, owner = [], []
    for i, im in enumerate(images):
        base = im.resize((448, 448), Image.Resampling.BICUBIC)
        tiles.append(base)
        owner.append(i)
        for y in (0, 112, 224):
            for x in (0, 112, 224):
                tiles.append(base.crop((x, y, x + 224, y + 224)))
                owner.append(i)
    x = torch.stack([preprocess(t) for t in tiles]).to(device)
    z = F.normalize(model.encode_image(x).float(), dim=-1)
    out = np.zeros(len(images), np.float32)
    owner = np.asarray(owner)
    for i in range(len(images)):
        n, b = text_bank[cats[i]]
        sims = (z[owner == i] @ (b - n)).cpu().numpy()
        # global view plus the strongest local window
        out[i] = 0.4 * float(sims[0]) + 0.6 * float(np.sort(sims[1:])[-2:].mean())
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def load_ckpt_recon(path: Path, device):
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    branch = ReconstructionBranch(obj["dims"], depth=4, heads=8)
    branch.load_state_dict(obj["state"])
    return branch.to(device).eval()


def load_ckpt_head(path: Path, device):
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    head = PseudoHead(obj["dim"], hidden=256)
    head.load_state_dict(obj["state"])
    return head.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--test-split", default="Test_A")
    ap.add_argument("--arch", default="dinov2_vitb14")
    ap.add_argument("--weight-dir", type=Path, default=Path("weights"))
    ap.add_argument("--ckpt", type=Path, default=Path("v9_best_hybrid_vlm/ckpt"))
    ap.add_argument("--clip-weight", type=Path, default=None)
    ap.add_argument("--fine-size", type=int, default=1036)
    ap.add_argument("--coarse-size", type=int, default=518)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--global-budget", type=int, default=16384)
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--agg-radius", type=int, default=1)
    ap.add_argument("--calib-limit", type=int, default=300)
    # fusion / sharpening hyper-parameters (chosen on the proxy set)
    ap.add_argument("--w-recon", type=float, default=0.40)
    ap.add_argument("--w-memory", type=float, default=0.35)
    ap.add_argument("--w-pseudo", type=float, default=0.25)
    ap.add_argument("--fuse-mode", default="weighted", choices=["weighted", "gated", "max"])
    ap.add_argument("--highpass", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--post-blur", type=float, default=0.6)
    ap.add_argument("--gate-strength", type=float, default=0.0)
    ap.add_argument("--area-target", type=float, default=0.02)
    ap.add_argument("--area-hardness", type=float, default=0.5)
    ap.add_argument("--fine-weight", type=float, default=0.65)
    ap.add_argument("--background-keep", type=float, default=0.08,
                    help="soft foreground prior: residual score multiplier outside the part")
    ap.add_argument("--boundary-keep", type=float, default=0.55,
                    help="soft foreground prior: residual score multiplier near object boundary")
    ap.add_argument("--transductive", action="store_true",
                    help="build an extra coreset from unlabelled test images (off by default)")
    ap.add_argument("--region-ranker", type=Path, required=True,
                    help="Train-only v12 candidate-region ranker checkpoint")
    ap.add_argument("--region-reject-scale", type=float, default=1.0,
                    help="Train-only calibration multiplier for candidate rejection")
    ap.add_argument("--region-relative-margin", type=float, default=0.16)
    ap.add_argument("--region-max-area", type=float, default=0.018)
    ap.add_argument("--reference-cache", type=Path, default=None,
                    help="optional v12 all-Train memory/calibration cache")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", type=Path, default=None)
    args = ap.parse_args()

    zip_path = args.zip or args.output.with_suffix(".zip")
    if args.output.exists() or zip_path.exists():
        raise FileExistsError(
            f"refusing to overwrite v9 output: {args.output} / {zip_path}"
        )

    set_seed()
    device = pick_device()
    started = time.time()
    print(f"device {device} arch {args.arch}", flush=True)
    region_ranker, region_metadata = load_ranker(args.region_ranker)
    print(
        f"loaded v12 region ranker reject_threshold={region_metadata['reject_threshold']:.4f}",
        flush=True,
    )

    layers = LAYER_SETS[args.arch]
    model = load_dinov2(args.arch, device, weight_dir=args.weight_dir)

    sizes = [s for s in {args.fine_size, args.coarse_size} if s]
    sizes = sorted(sizes, reverse=True)
    runners = [ScaleRunner(model, s, device, layers, args.agg_radius) for s in sizes]

    train = list_samples(args.data_root / "Train")
    test = list_samples(args.data_root / args.test_split)
    categories = sorted({s.category for s in train + test})
    print(f"train {len(train)} test {len(test)} categories {len(categories)}", flush=True)

    recon = {r.size: None for r in runners}
    head = {r.size: None for r in runners}
    rc = load_ckpt_recon(args.ckpt / "recon.pt", device)
    hd = load_ckpt_head(args.ckpt / "head.pt", device)
    if rc is not None:
        meta = json.loads((args.ckpt / "train_meta.json").read_text()) if (args.ckpt / "train_meta.json").exists() else {}
        trained_size = meta.get("size", args.fine_size)
        recon[trained_size if trained_size in recon else sizes[0]] = rc
        print(f"loaded reconstruction branch (trained at {trained_size})", flush=True)
    if hd is not None:
        meta = json.loads((args.ckpt / "train_meta.json").read_text()) if (args.ckpt / "train_meta.json").exists() else {}
        trained_size = meta.get("size", args.fine_size)
        head[trained_size if trained_size in head else sizes[0]] = hd
        print(f"loaded pseudo head (trained at {trained_size})", flush=True)

    if args.reference_cache is not None:
        try:
            payload = torch.load(args.reference_cache, map_location="cpu")
        except Exception:
            payload = json.loads(args.reference_cache.read_text(encoding="utf-8"))
        if payload.get("manifest"):
            base = args.reference_cache.parent
            banks = {int(size): torch.load(Path(path), map_location="cpu") for size, path in payload["bank_paths"].items()}
            cal = {int(size): torch.load(Path(path), map_location="cpu") for size, path in payload["calibration_paths"].items()}
            pix_cal = {}
            img_cal = {}
            for size, item in cal.items():
                pix_cal.update({(branch, size): value for branch, value in item["pix_cal"].items()})
                img_cal.update(item["img_cal"])
        else:
            banks = payload["banks"]
            pix_cal = payload["pix_cal"]
            img_cal = payload["img_cal"]
        if set(banks) != set(sizes):
            raise RuntimeError(
                f"reference cache sizes {sorted(banks)} do not match inference sizes {sizes}"
            )
        print(f"loaded v12 all-Train reference cache {args.reference_cache}", flush=True)
    else:
        banks = {}
        for r in runners:
            print(f"building memory bank at {r.size} ({r.side}x{r.side})", flush=True)
            banks[r.size] = build_memory(r, train, args.budget, args.global_budget, args.batch, args.knn)

        print("calibrating on Train normals", flush=True)
        pix_cal, img_cal = calibrate_branches(
            runners, train, banks, recon, head, args.batch, args.calib_limit
        )

    clip_model, clip_pre = build_clip(args.clip_weight, device)
    text_bank = clip_text_bank(clip_model, categories, device) if clip_model is not None else None
    clip_cal = Calibrator(lo_q=50.0, hi_q=99.0)
    if clip_model is not None:
        print("calibrating CLIP on Train normals", flush=True)
        pool: dict[str, list] = {}
        sub = stratified_samples(train, args.calib_limit)
        for st in range(0, len(sub), args.batch):
            chunk = sub[st : st + args.batch]
            images, cats = [], []
            for s in chunk:
                for im in load_many(s.paths):
                    images.append(im)
                    cats.append(s.category)
            vals = clip_scores(clip_model, clip_pre, images, cats, text_bank, device)
            for c, v in zip(cats, vals):
                pool.setdefault(c, []).append(float(v))
        for c, v in pool.items():
            clip_cal.fit_key((c,), np.asarray(v))
        clip_cal.fit_fallback(np.concatenate([np.asarray(v) for v in pool.values()]))

    if args.transductive:
        print("building transductive banks from unlabelled test images", flush=True)
        for r in runners:
            pools: dict[tuple, list] = {}
            for st in range(0, len(test), args.batch):
                chunk = test[st : st + args.batch]
                images, keys = [], []
                for s in chunk:
                    for v, im in enumerate(load_many(s.paths)):
                        images.append(im)
                        keys.append((s.category, v))
                _, cat, side = r.features(images)
                arr = cat.cpu().numpy()
                for i, key in enumerate(keys):
                    q = r.memory_query(arr[i]).reshape(-1, arr.shape[-1])
                    pools.setdefault(key, []).append(q[::5].astype(np.float16))
            for key, arrs in pools.items():
                x = np.concatenate(arrs).astype(np.float32)
                x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
                extra = robust_bank_from_samples(x, args.budget, contamination_q=95.0)
                have = banks[r.size].banks.get(key)
                merged = extra if have is None else np.concatenate([have, extra])
                banks[r.size].set(key, merged)

    out_root = args.output / "predicted_masks"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    weights = (args.w_recon, args.w_memory, args.w_pseudo)
    scale_w = {sizes[0]: args.fine_weight}
    for s in sizes[1:]:
        scale_w[s] = (1.0 - args.fine_weight) / max(1, len(sizes) - 1)

    t_infer = time.time()
    region_totals = {"candidate_count": 0, "selected_count": 0, "selected_area": 0.0,
                     "rejected_views": 0}
    for st in range(0, len(test), args.batch):
        chunk = test[st : st + args.batch]
        images, keys, cats = [], [], []
        for s in chunk:
            for v, im in enumerate(load_many(s.paths)):
                images.append(im)
                keys.append((s.category, v))
                cats.append(s.category)
        per_scale = {
            r.size: branch_maps(r, images, keys, banks.get(r.size), recon.get(r.size), head.get(r.size))
            for r in runners
        }
        clip_vals = (
            clip_scores(clip_model, clip_pre, images, cats, text_bank, device)
            if clip_model is not None
            else np.zeros(len(images), np.float32)
        )

        for gi, s in enumerate(chunk):
            view_maps = []
            view_branch_maps = []
            view_parts = []
            for v in range(5):
                i = gi * 5 + v
                key = (s.category, v)
                fused_scales = []
                branch_scales = {name: [] for name in BRANCHES}
                raw_parts = {}
                for r in runners:
                    pm = per_scale[r.size]
                    calibrated = {}
                    for b in BRANCHES:
                        m = pm[b]
                        if m is None:
                            calibrated[b] = None
                            continue
                        raw = m[i]
                        raw_parts.setdefault(b, []).append(
                            (view_score_from_map(raw), r.size)
                        )
                        prepared = prepare_branch_map(
                            raw, MASK_SIZE, highpass_frac=args.highpass
                        )
                        prepared = apply_prior(
                            prepared, images[i],
                            background_keep=args.background_keep,
                            boundary_keep=args.boundary_keep,
                        )
                        # calibration bounds are defined on the raw scale; apply
                        # the same affine map to the high-passed version so the
                        # relative scale across categories stays meaningful
                        lo, hi = pix_cal[(b, r.size)]._bounds(key)
                        calibrated[b] = apply_pixel_calibration(prepared, lo, hi)
                        branch_scales[b].append((calibrated[b], scale_w[r.size]))
                    fused = fuse_branches(
                        calibrated["recon"], calibrated["memory"], calibrated["pseudo"],
                        weights=weights, mode=args.fuse_mode,
                    )
                    fused_scales.append((fused, scale_w[r.size]))
                total_w = sum(w for _, w in fused_scales) or 1.0
                mask = np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
                for m, w in fused_scales:
                    mask += m * (w / total_w)
                mask = apply_prior(
                    mask, images[i],
                    background_keep=args.background_keep,
                    boundary_keep=args.boundary_keep,
                )
                view_maps.append(mask)
                evidence_maps = []
                for b in BRANCHES:
                    entries = branch_scales[b]
                    if not entries:
                        continue
                    branch_total = sum(w for _, w in entries) or 1.0
                    evidence_maps.append(sum(m * (w / branch_total) for m, w in entries))
                view_branch_maps.append(evidence_maps)
                parts = {}
                for b, entries in raw_parts.items():
                    val = sum(v * scale_w[sz] for v, sz in entries) / total_w
                    parts[b] = float(img_cal[b].apply((s.category,), val))
                if clip_model is not None:
                    parts["clip"] = float(clip_cal.apply((s.category,), clip_vals[i]))
                view_parts.append(parts)

            view_scores = [
                combine_image_scores(
                    p,
                    {"recon": args.w_recon, "memory": args.w_memory,
                     "pseudo": args.w_pseudo, "clip": 0.20},
                )
                for p in view_parts
            ]
            group_score = aggregate_views(view_scores)
            for v in range(5):
                m, region_diag = region_ranked_mask(
                    view_maps[v], view_branch_maps[v], images[gi * 5 + v],
                    region_ranker, region_metadata,
                    reject_scale=args.region_reject_scale,
                    relative_margin=args.region_relative_margin,
                    max_area=args.region_max_area,
                )
                region_totals["candidate_count"] += int(region_diag["candidate_count"])
                region_totals["selected_count"] += int(region_diag["selected_count"])
                region_totals["selected_area"] += float(region_diag["selected_area"])
                region_totals["rejected_views"] += int(region_diag["selected_count"] == 0)
                m = sharpen(m, gamma=args.gamma, post_blur=args.post_blur)
                if args.gate_strength > 0:
                    m = apply_image_gate(m, group_score, args.gate_strength)
                path = out_root / s.category / s.name / f"{v}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(np.round(np.clip(m, 0, 1) * 255)), "L").save(
                    path, "PNG", compress_level=1
                )
            rows.append({"group_folder": s.key, "anomaly_score": f"{group_score:.8f}"})
        print(f"test {min(st+args.batch, len(test))}/{len(test)}", flush=True)

    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
        w.writeheader()
        w.writerows(rows)

    n_images = len(test) * 5
    meta = {
        "method": "v12 DINOv2 three-branch dense proposals plus Train-only rejectable region ranking",
        "calibration_fix": "subtract Train-normal lower quantile before range scaling",
        "arch": args.arch,
        "scales": sizes,
        "grids": [s // 14 for s in sizes],
        "branches_active": {
            "recon": any(v is not None for v in recon.values()),
            "memory": True,
            "pseudo": any(v is not None for v in head.values()),
            "clip": clip_model is not None,
        },
        "fusion": {
            "weights": list(weights), "mode": args.fuse_mode,
            "highpass": args.highpass, "gamma": args.gamma,
            "gate_strength": args.gate_strength,
            "area_target": args.area_target, "area_hardness": args.area_hardness,
            "fine_weight": args.fine_weight,
            "background_keep": args.background_keep, "boundary_keep": args.boundary_keep,
        },
        "transductive_bank": bool(args.transductive),
        "reference_cache": None if args.reference_cache is None else str(args.reference_cache),
        "region_ranking": {
            "checkpoint": str(args.region_ranker),
            "reject_threshold": float(region_metadata["reject_threshold"]),
            "candidate_count": int(region_totals["candidate_count"]),
            "selected_count": int(region_totals["selected_count"]),
            "rejected_views": int(region_totals["rejected_views"]),
            "mean_selected_area_per_view": float(region_totals["selected_area"] / max(n_images, 1)),
            "forced_top1": False,
            "max_regions_per_view": 3,
        },
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "test_labels_used": False,
        "train_groups": len(train),
        "test_groups": len(test),
        "calibration_groups": len(stratified_samples(train, args.calib_limit)),
        "calibration_categories": len({x.category for x in stratified_samples(train, args.calib_limit)}),
        "seconds_per_image": round((time.time() - t_infer) / max(n_images, 1), 3),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for p in sorted(out_root.glob("*/*/*_mask.png")):
            zf.write(p, p.relative_to(args.output).as_posix())
    print(json.dumps(meta, indent=2), flush=True)
    print(f"masks {len(list(out_root.glob('*/*/*_mask.png')))} zip {zip_path}", flush=True)


if __name__ == "__main__":
    main()
