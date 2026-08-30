#!/usr/bin/env python3
"""Resumable, low-peak-memory production reference builder for v12.

The original final reference pass retained a large descriptor reservoir for
every category/view key and then built both scales before calibration. This
builder uses bounded reservoirs, processes small input batches, persists each
scale independently, and writes the final cache atomically.
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch

from branches import MemoryBank
from common import Calibrator, LAYER_SETS, greedy_coreset, list_samples, load_dinov2, load_many, pick_device, set_seed
from fusion import view_score_from_map
from infer_stream_memory import BRANCHES, ScaleRunner, branch_maps, load_ckpt_head, load_ckpt_recon
from calibration_v10 import apply_pixel_calibration
from object_prior import apply_prior
from object_prior import descriptor_subset, foreground_grid


def _reservoir(store, key, values, cap, rng):
    if values.size == 0:
        return
    old = store.get(key)
    if old is None:
        store[key] = values[-cap:].copy() if len(values) > cap else values.copy()
        return
    merged = np.concatenate([old, values], axis=0)
    if len(merged) <= cap:
        store[key] = merged
    else:
        store[key] = merged[rng.choice(len(merged), cap, replace=False)]


def build_memory_lowmem(runner, samples, budget, global_budget, batch, log=True):
    pools = {}
    global_pool = {}
    rng = np.random.default_rng(20260829 + runner.size)
    # Keep the descriptor count bounded independently of the number of Train
    # images. This is the main protection against host-RAM OOM.
    per_key_cap = min(1024, max(384, budget))
    global_cap = min(4096, max(1024, global_budget))
    for start in range(0, len(samples), batch):
        chunk = samples[start : start + batch]
        images, keys = [], []
        for sample in chunk:
            for view, image in enumerate(load_many(sample.paths, workers=4)):
                images.append(image)
                keys.append((sample.category, view))
        _, cat, side = runner.features(images)
        arr = cat.detach().cpu().numpy()
        for index, key in enumerate(keys):
            qmap = runner.memory_query(arr[index])
            fg = foreground_grid(images[index], side)
            selected = descriptor_subset(qmap, fg, fg_stride=2, bg_stride=28)
            _reservoir(pools, key, selected[::3].astype(np.float16), per_key_cap, rng)
            _reservoir(global_pool, "global", selected[::17].astype(np.float16), global_cap, rng)
        del images, arr, cat, qmap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if log and (start == 0 or start % max(batch * 20, 1) == 0):
            print(
                f"V12_RECOVERY memory_{runner.size} "
                f"{min(start + batch, len(samples))}/{len(samples)}",
                flush=True,
            )

    bank = MemoryBank(budget_per_key=budget, global_budget=global_budget, knn=3)
    for key in sorted(pools, key=str):
        values = pools[key].astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
        bank.set(key, greedy_coreset(values, min(budget, len(values))))
    values = global_pool.get("global", np.empty((0, 1), np.float16)).astype(np.float32)
    if len(values):
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
    bank.set_global(greedy_coreset(values, min(global_budget, len(values))))
    return bank


def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_manifest(payload, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _bounded_append(store, key, values, cap=12000):
    values = np.asarray(values, np.float32).ravel()
    if not len(values):
        return
    old = store.get(key)
    if old is None:
        store[key] = values[-cap:].copy()
        return
    merged = np.concatenate([old, values])
    store[key] = merged[-cap:] if len(merged) > cap else merged


def calibrate_scale_lowmem(runner, samples, banks, recon, head, batch, limit, scale_index):
    """Calibrate one scale without retaining both scales' pixel pools."""
    pixel = {b: Calibrator(lo_q=90.0, hi_q=99.9) for b in BRANCHES}
    image = {b: Calibrator(lo_q=50.0, hi_q=99.0) for b in BRANCHES}
    pixel_pool, image_pool = {}, {}
    subset = samples if not limit or limit >= len(samples) else samples[:limit]
    for start in range(0, len(subset), batch):
        chunk = subset[start:start + batch]
        images, keys = [], []
        for sample in chunk:
            for view, im in enumerate(load_many(sample.paths, workers=4)):
                images.append(im)
                keys.append((sample.category, view))
        maps = branch_maps(runner, images, keys, banks, recon, head)
        for branch in BRANCHES:
            raw = maps[branch]
            if raw is None:
                continue
            for i, key in enumerate(keys):
                adjusted = apply_prior(raw[i], images[i], background_keep=0.10, boundary_keep=0.62)
                fg = np.ones_like(adjusted, bool)
                if adjusted.shape[0] == adjusted.shape[1]:
                    # The map is already on the runner grid; use a broad
                    # foreground gate only for collecting calibration pixels.
                    fg = adjusted >= 0.0
                _bounded_append(pixel_pool, (branch, key), adjusted[fg][::7])
                _bounded_append(image_pool, (branch, key[0]), [view_score_from_map(adjusted)])
        done = min(start + len(chunk), len(subset))
        if start == 0 or done % max(batch * 10, 1) == 0 or done == len(subset):
            print(
                f"V12_RECOVERY calibrate_{runner.size} {done}/{len(subset)}",
                flush=True,
            )
        del images, maps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for (branch, key), values in pixel_pool.items():
        pixel[branch].fit_key(key, values)
    for branch in BRANCHES:
        values = [v for (b, _), v in pixel_pool.items() if b == branch]
        if values:
            pixel[branch].fit_fallback(np.concatenate(values)[::5])
        values = [v for (b, _), v in image_pool.items() if b == branch]
        for (b, cat), vals in image_pool.items():
            if b == branch:
                image[branch].fit_key((cat,), vals)
        if values:
            image[branch].fit_fallback(np.concatenate(values))
    return pixel, image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--weight-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fine-size", type=int, default=1036)
    ap.add_argument("--coarse-size", type=int, default=518)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--global-budget", type=int, default=16384)
    ap.add_argument("--calib-limit", type=int, default=300)
    args = ap.parse_args()
    set_seed()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = load_dinov2("dinov2_vitb14", device, weight_dir=args.weight_dir)
    samples = list_samples(args.data_root / "Train")
    sizes = sorted({args.fine_size, args.coarse_size}, reverse=True)
    runners = [
        ScaleRunner(model, size, device, LAYER_SETS["dinov2_vitb14"], agg_radius=1)
        for size in sizes
    ]
    recon = {r.size: None for r in runners}
    head = {r.size: None for r in runners}
    rc = load_ckpt_recon(args.ckpt / "recon.pt", device)
    hd = load_ckpt_head(args.ckpt / "head.pt", device)
    meta_path = args.ckpt / "train_meta.json"
    trained_size = (
        int(json.loads(meta_path.read_text()).get("size", args.fine_size))
        if meta_path.exists() else args.fine_size
    )
    target_size = trained_size if trained_size in recon else sizes[0]
    recon[target_size] = rc
    head[target_size] = hd
    banks = {}
    for runner in runners:
        bank_path = args.output.with_name(f"reference_bank_{runner.size}.pt")
        if bank_path.exists() and bank_path.stat().st_size > 0:
            print(f"V12_RECOVERY reuse {bank_path}", flush=True)
            banks[runner.size] = torch.load(bank_path, map_location="cpu")
        else:
            print(f"V12_RECOVERY build memory_{runner.size}", flush=True)
            banks[runner.size] = build_memory_lowmem(
                runner, samples, args.budget, args.global_budget, args.batch
            )
            atomic_save(banks[runner.size], bank_path)
            print(f"V12_RECOVERY bank_complete {bank_path}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("V12_RECOVERY calibrate_start", flush=True)
    pix_cal, img_cal = {}, {}
    # Each scale is calibrated independently and immediately serialized. This
    # prevents the old two-scale pix_pool from growing until the host OOMs.
    for scale_index, runner in enumerate(runners):
        print(f"V12_RECOVERY calibrate_scale_{runner.size}", flush=True)
        pix_one, img_one = calibrate_scale_lowmem(
            runner, samples, banks[runner.size], recon[runner.size],
            head[runner.size], args.batch, args.calib_limit, scale_index
        )
        for branch in BRANCHES:
            pix_cal[(branch, runner.size)] = pix_one[branch]
            img_cal[branch] = img_one[branch]
        partial = args.output.with_name(f"calibration_scale_{runner.size}.pt")
        atomic_save({"pix_cal": pix_one, "img_cal": img_one, "size": runner.size}, partial)
        del pix_one, img_one
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Keep the large banks and calibration objects as separate files. A
    # single torch.save would require another full-size temporary copy and
    # overflow the competition machine's 50GB data volume.
    manifest = {
        "manifest": True, "bank_paths": {
            str(size): str(args.output.with_name(f"reference_bank_{size}.pt"))
            for size in sizes
        },
        "calibration_paths": {
            str(size): str(args.output.with_name(f"calibration_scale_{size}.pt"))
            for size in sizes
        },
        "sizes": sizes, "arch": "dinov2_vitb14", "train_groups": len(samples),
        "official_train_only": True, "test_labels_used": False,
        "lowmem_reference": True, "compact_reference": True,
        "per_key_reservoir_cap": 1024, "global_reservoir_cap": 4096,
    }
    atomic_manifest(manifest, args.output)
    print(f"V12_RECOVERY_REFERENCE_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
