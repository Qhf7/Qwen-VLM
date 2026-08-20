#!/usr/bin/env python3
"""Resume the Qwen-DINOv2 pseudo-anomaly head from a saved epoch.

The previous long-running job stopped during epoch 2.  This script resumes
from the preserved epoch-1 checkpoint and writes only new checkpoint names.
It does not touch the epoch-1 file or any previous submission.
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

from branches import HeadLoss, PseudoHead
from common import LAYER_SETS, list_samples, load_dinov2, load_rgb, per_layer_features, pick_device, set_seed
from object_prior import foreground_grid
from pseudo import synthesize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--arch", default="dinov2_vitb14")
    ap.add_argument("--size", type=int, default=1036)
    ap.add_argument("--weight-dir", type=Path, required=True)
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-images", type=int, default=8)
    ap.add_argument("--lr", type=float, default=7.5e-4)
    args = ap.parse_args()

    set_seed(20260814 + 100)
    device = pick_device()
    layers = LAYER_SETS[args.arch]
    samples = list_samples(args.data_root / "Train")
    model = load_dinov2(args.arch, device, weight_dir=args.weight_dir)

    checkpoint = torch.load(args.resume, map_location="cpu")
    head = PseudoHead(int(checkpoint["dim"]), hidden=256).to(device)
    head.load_state_dict(checkpoint["state"])
    head.train()
    side = args.size // 14
    flat = [(s, v) for s in samples for v in range(5)]
    steps = math.ceil(len(flat) / args.batch_images)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps), pct_start=0.25
    )
    loss_fn = HeadLoss(pos_weight=8.0, tversky_w=1.0, bce_w=0.5).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    start_epoch = int(checkpoint.get("epoch", 1))
    print(
        f"resume from epoch {start_epoch}; continue {args.epochs} epochs; "
        f"{len(samples)} groups/{len(flat)} views; grid {side}; device {device}",
        flush=True,
    )

    for local_ep in range(args.epochs):
        global_ep = start_epoch + local_ep + 1
        rng = np.random.default_rng(3000 + global_ep)
        order = rng.permutation(len(flat))
        losses = []
        t0 = time.time()
        for step in range(steps):
            idx = order[step * args.batch_images : (step + 1) * args.batch_images]
            if len(idx) == 0:
                continue
            images, targets, fg_targets = [], [], []
            for flat_idx in idx:
                sample, view = flat[int(flat_idx)]
                image = load_rgb(sample.paths[view])
                if rng.random() < 0.5:
                    images.append(image)
                    targets.append(np.zeros((side, side), np.float32))
                else:
                    augmented, mask = synthesize(image, rng)
                    images.append(augmented)
                    resized = np.asarray(
                        mask.resize((side, side), Image.Resampling.BILINEAR),
                        np.float32,
                    ) / 255.0
                    targets.append((resized > 0.20).astype(np.float32))
                fg_targets.append(foreground_grid(image, side))

            maps, _ = per_layer_features(model, images, args.size, device, layers)
            with torch.no_grad():
                x = torch.cat([F.normalize(m, dim=-1) for m in maps], dim=-1)
                x = x.permute(0, 3, 1, 2).contiguous()
            y = torch.from_numpy(np.stack(targets)).to(device)
            fg = torch.from_numpy(np.stack(fg_targets)).to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = head(x)
                loss = loss_fn(logits.float(), y)
                bg = (fg < 0.20).float()
                if bg.sum() > 0:
                    loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                        logits.float(), torch.zeros_like(y), weight=bg, reduction="sum"
                    ) / bg.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {global_ep}, step {step}")

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
            sched.step()
            losses.append(float(loss.detach()))
            if step % 20 == 0:
                print(
                    f"head ep{global_ep}/{start_epoch + args.epochs} "
                    f"step{step}/{steps} loss {np.mean(losses[-20:]):.5f}",
                    flush=True,
                )

        print(
            f"head epoch {global_ep} loss {np.mean(losses):.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        torch.save(
            {
                "state": head.state_dict(),
                "dim": int(checkpoint["dim"]),
                "size": args.size,
                "layers": list(layers),
                "epoch": global_ep,
                "resumed_from": str(args.resume),
            },
            args.out / f"head_epoch{global_ep}_resume.pt",
        )

    final = {
        "state": head.state_dict(),
        "dim": int(checkpoint["dim"]),
        "size": args.size,
        "layers": list(layers),
        "epoch": start_epoch + args.epochs,
        "resumed_from": str(args.resume),
    }
    torch.save(final, args.out / "head.pt")
    (args.out / "resume_meta.json").write_text(
        json.dumps(
            {
                "start_epoch": start_epoch,
                "end_epoch": start_epoch + args.epochs,
                "groups": len(samples),
                "views": len(flat),
                "size": args.size,
                "external_data": False,
                "test_labels_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out / 'head.pt'}", flush=True)


if __name__ == "__main__":
    main()
