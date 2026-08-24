#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V3))

from clip_dense import clip_preprocess_tensor, load_clip
from common import LAYER_SETS, list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed
from losses_v4 import V4Loss
from model_v4 import VLMAnomalyV4
from pseudo_states import make_group_targets
from splits import split_categories


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}


def train(args):
    set_seed(args.seed)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=False)
    samples = list_samples(args.data_root / "Train")
    train_samples, heldout_samples = split_categories(samples, args.heldout_fold, args.folds)
    if args.limit:
        train_samples = train_samples[: args.limit]
    if not train_samples:
        raise RuntimeError("empty training split")

    dino = load_dinov2(args.arch, device, args.weight_dir)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    layers = LAYER_SETS[args.arch]
    probe = load_many(train_samples[0].paths[:1])
    with torch.no_grad():
        maps, _ = per_layer_features(dino, probe, args.dino_size, device, layers)
    dims = [int(x.shape[-1]) for x in maps]
    model = VLMAnomalyV4(dims, clip_model, args.decoder_width, args.fusion_width, args.prompt_ctx).to(device)
    if args.v3_checkpoint is not None:
        model.load_v3_decoder(torch.load(args.v3_checkpoint, map_location="cpu"))

    # CLIP/DINO backbones are referenced by the trainable wrappers but remain frozen.
    for p in clip_model.parameters():
        p.requires_grad_(False)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    loss_fn = V4Loss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    stats = trainable_summary(model)
    print(
        f"v4 device={device} train_groups={len(train_samples)} heldout_groups={len(heldout_samples)} "
        f"dims={dims} trainable={stats['trainable']:,}/{stats['total']:,}",
        flush=True,
    )

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_samples))
        epoch_parts: list[dict[str, float]] = []
        for step, start in enumerate(range(0, len(order), args.batch_groups)):
            selected = [train_samples[int(i)] for i in order[start : start + args.batch_groups]]
            images = []
            original_images = []
            binary = []
            states = []
            view_labels = []
            sample_labels = []
            for sample in selected:
                group_images = load_many(sample.paths)
                # Include fully-normal groups explicitly; otherwise every
                # training group would be positive and image-level AP would
                # have no useful negative supervision.
                force_normal = bool(rng.random() < args.normal_group_probability)
                targets = make_group_targets(group_images, rng, size=448, force_normal=force_normal)
                images.extend(t.image for t in targets)
                original_images.extend(t.original for t in targets)
                binary.extend(t.binary_mask for t in targets)
                states.extend(t.state_mask for t in targets)
                view_labels.extend(t.is_anomaly for t in targets)
                sample_labels.append(float(any(t.is_anomaly > 0.5 for t in targets)))

            with torch.no_grad():
                dino_maps, _ = per_layer_features(dino, images, args.dino_size, device, layers)
            clip_x = clip_preprocess_tensor(images, args.clip_size, device)
            binary_t = torch.from_numpy(np.stack(binary)).to(device)
            states_t = torch.from_numpy(np.stack(states)).to(device)
            view_t = torch.tensor(view_labels, device=device)
            sample_t = torch.tensor(sample_labels, device=device)
            # Memory-dropout simulates a B-board unseen category.  Although
            # all optimisation images come from the 40 fit categories, the
            # gate must also learn how to behave without a category bank.
            group_seen = [0.0 if rng.random() < args.unseen_sim_probability else 1.0 for _ in selected]
            seen = torch.tensor([value for value in group_seen for _ in range(5)], device=device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(dino_maps, clip_x, seen=seen, groups=len(selected))
                loss, parts = loss_fn(model, output, binary_t, states_t, view_t, sample_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} step={step}: {parts}")
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimiser)
            scaler.update()
            epoch_parts.append(parts)
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs} step {step}/{int(np.ceil(len(order)/args.batch_groups))} "
                    + " ".join(f"{key}={value:.4f}" for key, value in parts.items()),
                    flush=True,
                )
        scheduler.step()
        means = {key: float(np.mean([part[key] for part in epoch_parts])) for key in epoch_parts[0]}
        print(f"epoch {epoch+1} mean={means} elapsed={time.time()-started:.0f}s", flush=True)
        torch.save(
            {
                "model": model.compact_state_dict(),
                "dims": dims,
                "layers": list(layers),
                "arch": args.arch,
                "dino_size": args.dino_size,
                "clip_size": args.clip_size,
                "decoder_width": args.decoder_width,
                "fusion_width": args.fusion_width,
                "prompt_ctx": args.prompt_ctx,
                "heldout_fold": args.heldout_fold,
                "epoch": epoch + 1,
            },
            args.out / f"vlm_v4_epoch{epoch+1}.pt",
        )

    final = args.out / "vlm_v4.pt"
    torch.save(
        {
            "model": model.compact_state_dict(),
            "dims": dims,
            "layers": list(layers),
            "arch": args.arch,
            "dino_size": args.dino_size,
            "clip_size": args.clip_size,
            "decoder_width": args.decoder_width,
            "fusion_width": args.fusion_width,
            "prompt_ctx": args.prompt_ctx,
            "heldout_fold": args.heldout_fold,
        },
        final,
    )
    metadata = {
        "method": "frozen DINOv2 + frozen CLIP + prompt/adapter/query/multistate/gate/multiview training",
        "official_train_only": True,
        "external_anomaly_data": False,
        "heldout_fold": args.heldout_fold,
        "train_categories": sorted({s.category for s in train_samples}),
        "heldout_categories": sorted({s.category for s in heldout_samples}),
        "train_groups": len(train_samples),
        "trainable_parameters": stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.out / "train_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="dinov2_vitb14")
    parser.add_argument("--dino-size", type=int, default=1036)
    parser.add_argument("--clip-size", type=int, default=448)
    parser.add_argument("--decoder-width", type=int, default=192)
    parser.add_argument("--fusion-width", type=int, default=192)
    parser.add_argument("--prompt-ctx", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normal-group-probability", type=float, default=0.30)
    parser.add_argument("--unseen-sim-probability", type=float, default=0.25)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    train(parser.parse_args())
