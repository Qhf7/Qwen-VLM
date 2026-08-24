#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import LAYER_SETS, list_samples, load_dinov2, per_layer_features, pick_device, set_seed
from multiscale_decoder import LayerAwarePixelDecoder, MultiScalePixelLoss
from pseudo_v3 import make_training_targets


def train(args):
    set_seed(20260821)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)
    layers = LAYER_SETS[args.arch]
    model = load_dinov2(args.arch, device, args.weight_dir)
    samples = list_samples(args.data_root / "Train")
    if args.limit:
        samples = samples[:args.limit]
    with torch.no_grad():
        probe = Image.open(samples[0].paths[0]).convert("RGB")
        maps, side = per_layer_features(model, [probe], args.size, device, layers)
        dims = [int(x.shape[-1]) for x in maps]
    decoder = LayerAwarePixelDecoder(dims, width=args.decoder_width).to(device).train()
    loss_fn = MultiScalePixelLoss()
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(20260821)
    print(f"v3 device={device} groups={len(samples)} dims={dims} side={side}", flush=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        order = np.arange(len(samples))
        rng.shuffle(order)
        running = []
        for start in range(0, len(order), args.batch_images):
            chosen = [samples[int(i)] for i in order[start:start + args.batch_images]]
            images, targets, hard = [], [], []
            for sample in chosen:
                # All five views participate; view-level consistency is used
                # at inference because cameras are not pixel-registered.
                for path in sample.paths:
                    with Image.open(path) as im:
                        normal = im.convert("RGB")
                    defective, target, hard_map = make_training_targets(normal, rng)
                    images.append(defective)
                    targets.append(target)
                    hard.append(hard_map)
            with torch.no_grad():
                maps, _ = per_layer_features(model, images, args.size, device, layers)
            target_t = torch.from_numpy(np.stack(targets)).to(device)
            hard_t = torch.from_numpy(np.stack(hard)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, aux = decoder(maps)
                loss = loss_fn(logits.float(), aux.float(), target_t, hard_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite v3 loss: {loss}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running.append(float(loss.detach()))
            if len(running) % args.log_every == 0:
                print(f"v3 epoch {epoch+1}/{args.epochs} step {start}/{len(order)} loss {np.mean(running[-args.log_every:]):.5f}", flush=True)
        print(f"v3 epoch {epoch+1}/{args.epochs} mean_loss={np.mean(running):.5f} elapsed={time.time()-t0:.0f}s", flush=True)
        torch.save({"state": decoder.state_dict(), "dims": dims, "size": args.size, "layers": list(layers), "width": args.decoder_width, "epoch": epoch + 1}, args.out / f"decoder_epoch{epoch+1}.pt")
    torch.save({"state": decoder.state_dict(), "dims": dims, "size": args.size, "layers": list(layers), "width": args.decoder_width}, args.out / "decoder.pt")
    (args.out / "meta.json").write_text(json.dumps({"arch": args.arch, "size": args.size, "layers": list(layers), "dims": dims, "epochs": args.epochs, "train_groups": len(samples), "external_data": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--weight-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="dinov2_vitb14")
    ap.add_argument("--size", type=int, default=1036)
    ap.add_argument("--decoder-width", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-images", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    train(ap.parse_args())
