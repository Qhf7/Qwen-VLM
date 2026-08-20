#!/usr/bin/env python3
"""Train the reconstruction decoder and the pseudo-anomaly head.

Both are trained on official Train normal images only.  The DINOv2 backbone
stays frozen throughout; only the small decoder / head weights are learned,
so this is normal-sample modelling plus rule-compliant synthetic defects,
not supervised anomaly training on external data.

Run on the cloud GPU:

    python v4/train_branches.py \
        --data-root Dataset --arch dinov2_vitb14 --size 1036 \
        --weight-dir weights --out v4/ckpt --epochs-recon 8 --epochs-head 6
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from branches import HeadLoss, PseudoHead, ReconstructionBranch, reconstruction_score
from common import (
    LAYER_SETS,
    list_samples,
    load_dinov2,
    load_many,
    load_rgb,
    per_layer_features,
    pick_device,
    set_seed,
)
from metrics import split_train
from object_prior import foreground_grid
from pseudo import synthesize


def layer_dims(model, size, device, layers):
    dummy = [Image.new("RGB", (size, size), (127, 127, 127))]
    maps, side = per_layer_features(model, dummy, size, device, layers)
    return [m.shape[-1] for m in maps], side


class NormalFeatureStream:
    """Yields frozen encoder features for normal Train images in batches."""

    def __init__(self, model, samples, size, device, layers, batch_images: int):
        self.model = model
        self.samples = samples
        self.size = size
        self.device = device
        self.layers = layers
        self.batch = batch_images

    def __iter__(self):
        flat = [(s, v) for s in self.samples for v in range(5)]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        order = rng.permutation(len(flat))
        for st in range(0, len(order), self.batch):
            idx = order[st : st + self.batch]
            # Parallel decode: each source PNG is 2000-3600 px, so serial
            # decoding leaves the GPU idle roughly 70% of the time.
            images = load_many([flat[i][0].paths[flat[i][1]] for i in idx], workers=self.batch)
            maps, side = per_layer_features(
                self.model, images, self.size, self.device, self.layers
            )
            yield [m.reshape(m.shape[0], -1, m.shape[-1]) for m in maps], side


def train_reconstruction(
    model, samples, device, size, layers, dims, epochs, batch_images, lr, out_dir, log_every=20
):
    branch = ReconstructionBranch(dims, depth=4, heads=8, noise=0.25).to(device)
    opt = torch.optim.AdamW(branch.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    stream = NormalFeatureStream(model, samples, size, device, layers, batch_images)
    steps_per_epoch = math.ceil(len(samples) * 5 / batch_images)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch), pct_start=0.25
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    branch.train()
    for ep in range(epochs):
        losses = []
        t0 = time.time()
        for step, (feats, side) in enumerate(stream):
            feats = [f.detach() for f in feats]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                recons = branch(feats)
                # cosine reconstruction loss with hard-patch mining: the
                # easiest 90% of patches are down-weighted so the decoder
                # spends capacity on genuinely varied normal structure
                # instead of flat background.
                loss = 0.0
                for f, r in zip(feats, recons):
                    d = 1.0 - F.cosine_similarity(
                        F.normalize(f, dim=-1), F.normalize(r, dim=-1), dim=-1
                    )
                    k = max(1, int(d.shape[1] * 0.10))
                    hard = torch.topk(d, k=k, dim=1).values
                    loss = loss + 0.5 * d.mean() + 0.5 * hard.mean()
                loss = loss / len(feats)
            if not torch.isfinite(loss):
                # Fail loudly: a silently-NaN run produces a checkpoint that
                # looks valid but encodes nothing.
                raise RuntimeError(
                    f"reconstruction loss became {float(loss)} at epoch {ep+1} step {step}; "
                    "aborting instead of saving a dead checkpoint"
                )
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
                opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            losses.append(float(loss.detach()))
            if step % log_every == 0:
                print(
                    f"  recon ep{ep+1} step{step}/{steps_per_epoch} loss {np.mean(losses[-log_every:]):.5f}",
                    flush=True,
                )
        print(f"recon epoch {ep+1}/{epochs} loss {np.mean(losses):.5f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(
            {"state": branch.state_dict(), "dims": dims, "size": size, "layers": list(layers), "epoch": ep + 1},
            out_dir / f"recon_epoch{ep+1}.pt",
        )
    torch.save(
        {"state": branch.state_dict(), "dims": dims, "size": size, "layers": list(layers)},
        out_dir / "recon.pt",
    )
    return branch


def train_pseudo_head(
    model, samples, device, size, layers, epochs, batch_images, lr, out_dir, log_every=20
):
    dims, side = layer_dims(model, size, device, layers)
    total_dim = sum(dims)
    head = PseudoHead(total_dim, hidden=256).to(device)
    loss_fn = HeadLoss(pos_weight=8.0, tversky_w=1.0, bce_w=0.5).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    flat = [(s, v) for s in samples for v in range(5)]
    steps_per_epoch = math.ceil(len(flat) / batch_images)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch), pct_start=0.3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    head.train()
    for ep in range(epochs):
        rng = np.random.default_rng(1000 + ep)
        order = rng.permutation(len(flat))
        losses = []
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = order[step * batch_images : (step + 1) * batch_images]
            if len(idx) == 0:
                continue
            # Parallel image loading: each PNG is 2000-3600 px, so serial
            # decoding leaves the GPU idle ~70% of the time.
            raw = load_many([flat[i][0].paths[flat[i][1]] for i in idx], workers=min(len(idx), 8))
            images, targets, fg_targets = [], [], []
            for im in raw:
                # half the batch stays normal so the head sees the true
                # normal distribution, not only defect-vs-defect contrast
                if rng.random() < 0.45:
                    images.append(im)
                    targets.append(np.zeros((side, side), np.float32))
                    fg_targets.append(foreground_grid(im, side))
                else:
                    aug, mask = synthesize(im, rng, n_defects=(1, 3))
                    images.append(aug)
                    m = np.asarray(
                        mask.resize((side, side), Image.Resampling.BILINEAR), np.float32
                    ) / 255.0
                    targets.append((m > 0.20).astype(np.float32))
                    fg_targets.append(foreground_grid(im, side))
            maps, _ = per_layer_features(model, images, size, device, layers)
            with torch.no_grad():
                x = torch.cat([F.normalize(m, dim=-1) for m in maps], dim=-1)
                x = x.permute(0, 3, 1, 2).contiguous()
            y = torch.from_numpy(np.stack(targets)).to(device)
            fg = torch.from_numpy(np.stack(fg_targets)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = head(x)
                loss = loss_fn(logits.float(), y)
                # Background should be strongly normal.  This is model-side
                # supervision from official Train images, not submission
                # post-processing; it attacks the "whole object/silhouette"
                # failure mode at the learned pixel head.
                bg = (fg < 0.20).float()
                if bg.sum() > 0:
                    loss = loss + 0.35 * F.binary_cross_entropy_with_logits(
                        logits.float(), torch.zeros_like(y), weight=bg, reduction="sum"
                    ) / bg.sum().clamp_min(1.0)
                # Silhouette and high-contrast part boundaries are frequent
                # false positives.  They are still normal pixels in Train, so
                # teach the head that foreground edges are not defects unless
                # a synthetic mask actually overlaps them.
                fg4 = fg[:, None]
                edge = F.max_pool2d(fg4, 3, stride=1, padding=1) - (
                    -F.max_pool2d(-fg4, 3, stride=1, padding=1)
                )
                edge = (edge[:, 0] > 0.12).float() * (y < 0.5).float()
                if edge.sum() > 0:
                    loss = loss + 0.20 * F.binary_cross_entropy_with_logits(
                        logits.float(), torch.zeros_like(y), weight=edge, reduction="sum"
                    ) / edge.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"pseudo-head loss became {float(loss)} at epoch {ep+1} step {step}; aborting"
                )
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            losses.append(float(loss.detach()))
            if step % log_every == 0:
                print(
                    f"  head ep{ep+1} step{step}/{steps_per_epoch} loss {np.mean(losses[-log_every:]):.5f}",
                    flush=True,
                )
        print(f"head epoch {ep+1}/{epochs} loss {np.mean(losses):.5f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(
            {"state": head.state_dict(), "dim": total_dim, "size": size, "layers": list(layers), "epoch": ep + 1},
            out_dir / f"head_epoch{ep+1}.pt",
        )
    torch.save(
        {"state": head.state_dict(), "dim": total_dim, "size": size, "layers": list(layers)},
        out_dir / "head.pt",
    )
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset"))
    ap.add_argument("--arch", default="dinov2_vitb14")
    ap.add_argument("--size", type=int, default=1036)
    ap.add_argument("--weight-dir", type=Path, default=Path("weights"))
    ap.add_argument("--out", type=Path, default=Path("v4/ckpt"))
    ap.add_argument("--epochs-recon", type=int, default=8)
    ap.add_argument("--epochs-head", type=int, default=6)
    ap.add_argument("--batch-images", type=int, default=8)
    ap.add_argument("--lr-recon", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=1.5e-3)
    ap.add_argument("--holdout", type=float, default=0.0, help=">0 keeps a proxy split unseen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-recon", action="store_true")
    ap.add_argument("--skip-head", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)
    layers = LAYER_SETS[args.arch]
    print(f"device {device} arch {args.arch} size {args.size} grid {args.size//14}", flush=True)

    model = load_dinov2(args.arch, device, weight_dir=args.weight_dir)
    samples = list_samples(args.data_root / "Train")
    if args.holdout > 0:
        samples, _ = split_train(samples, holdout_frac=args.holdout)
    if args.limit:
        samples = samples[: args.limit]
    print(f"training on {len(samples)} groups ({len(samples)*5} images)", flush=True)

    dims, side = layer_dims(model, args.size, device, layers)
    print("layer dims", dims, "grid side", side, flush=True)

    meta = {
        "arch": args.arch,
        "size": args.size,
        "layers": list(layers),
        "dims": dims,
        "side": side,
        "train_groups": len(samples),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
    }

    t0 = time.time()
    if not args.skip_recon:
        train_reconstruction(
            model, samples, device, args.size, layers, dims,
            args.epochs_recon, args.batch_images, args.lr_recon, args.out,
        )
    if not args.skip_head:
        train_pseudo_head(
            model, samples, device, args.size, layers,
            args.epochs_head, args.batch_images, args.lr_head, args.out,
        )
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
