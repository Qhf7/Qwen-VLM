#!/usr/bin/env python3
"""Train the lightweight VLM adaptation on official Train normals only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "v6_vlm_zero_shot_v4"
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(V4)); sys.path.insert(2, str(V3))

from clip_dense import clip_preprocess_tensor, load_clip
from common import LAYER_SETS, list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed
from losses_v5 import V5Loss
from model_v5 import VLMPrimaryV5, load_trusted_checkpoint
from pseudo_v5 import DEFAULT_TRAIN_GENERATORS, make_group_targets_v5
from reference_v5 import aligned_reference_map
from splits import split_categories


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}


def configure_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.clip_encoder.adapter,
        model.clip_encoder.patch_projection,
        model.prompt.dynamic,
        model.local_decoder,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for parameter in (
        model.prompt.context,
        model.prompt.state_residual,
        model.semantic_temperature,
        model.reference_weight_logit,
        model.dino_fraction_logit,
        model.clip_encoder.window_logits,
    ):
        parameter.requires_grad_(True)


def checkpoint(model, config, epoch, optimiser=None, scaler=None):
    payload = dict(config)
    payload.update({"model": model.compact_state_dict(), "epoch": epoch})
    if optimiser is not None:
        payload["optimiser"] = optimiser.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def main(args):
    set_seed(args.seed)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=False)
    samples = list_samples(args.data_root / "Train")
    train_samples, heldout = split_categories(samples, args.heldout_fold, args.folds)
    if args.heldout_fold < 0:
        train_samples, heldout = samples, []
    if args.limit:
        train_samples = train_samples[: args.limit]
    if not train_samples:
        raise RuntimeError("empty train split")

    dino = load_dinov2(args.arch, device, args.weight_dir).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    layers = LAYER_SETS[args.arch]
    with torch.inference_mode():
        probe, _ = per_layer_features(dino, load_many(train_samples[0].paths[:1]), args.dino_size, device, layers)
    dims = [int(feature.shape[-1]) for feature in probe]
    model = VLMPrimaryV5(
        dims, clip_model, args.decoder_width, args.vlm_width, args.prompt_ctx, args.max_dino_fraction
    ).to(device)
    provenance = {}
    if args.v4_checkpoint:
        provenance["v4_adaptation"] = model.load_v4_adaptation(
            load_trusted_checkpoint(args.v4_checkpoint), reuse_vlm=args.reuse_v4_vlm
        )
    if args.v3_checkpoint:
        model.load_v3_decoder(load_trusted_checkpoint(args.v3_checkpoint))
        provenance["v3_decoder"] = str(args.v3_checkpoint)
    configure_trainable(model)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = V5Loss().to(device)
    rng = np.random.default_rng(args.seed)
    stats = trainable_summary(model)
    config = {
        "dims": dims, "layers": list(layers), "arch": args.arch,
        "dino_size": args.dino_size, "clip_size": args.clip_size,
        "decoder_width": args.decoder_width, "vlm_width": args.vlm_width,
        "prompt_ctx": args.prompt_ctx, "max_dino_fraction": args.max_dino_fraction,
        "heldout_fold": args.heldout_fold,
        "train_generators": list(DEFAULT_TRAIN_GENERATORS),
    }
    print(
        f"v5 device={device} train_groups={len(train_samples)} heldout_groups={len(heldout)} "
        f"trainable={stats['trainable']:,}/{stats['total']:,} provenance={provenance}", flush=True
    )
    started = time.time()
    total_steps = int(np.ceil(len(train_samples) / args.batch_groups))

    for epoch in range(args.epochs):
        order = rng.permutation(len(train_samples))
        epoch_parts = []
        model.train()
        for step, start in enumerate(range(0, len(order), args.batch_groups)):
            selected = [train_samples[int(i)] for i in order[start:start + args.batch_groups]]
            images, originals, binary, states, view_labels, sample_labels = [], [], [], [], [], []
            for sample in selected:
                normals = load_many(sample.paths)
                targets = make_group_targets_v5(
                    normals,
                    rng,
                    size=448,
                    force_normal=bool(rng.random() < args.normal_group_probability),
                    generators=DEFAULT_TRAIN_GENERATORS,
                )
                images.extend(target.image for target in targets)
                originals.extend(target.original for target in targets)
                binary.extend(target.binary_mask for target in targets)
                states.extend(target.state_mask for target in targets)
                view_labels.extend(target.is_anomaly for target in targets)
                sample_labels.append(float(any(target.is_anomaly > 0.5 for target in targets)))

            with torch.no_grad():
                dino_maps, _ = per_layer_features(dino, images, args.dino_size, device, layers)
                normal_clip = clip_preprocess_tensor(originals, args.clip_size, device)
                _, normal_patch = model.encode_clip(normal_clip)
            clip_x = clip_preprocess_tensor(images, args.clip_size, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = model.encode_clip(clip_x)
                reference = aligned_reference_map(encoded[1], normal_patch)
                output = model(dino_maps, clip_x, reference_map=reference, groups=len(selected), encoded_clip=encoded)
                loss, parts = loss_fn(
                    model,
                    output,
                    torch.from_numpy(np.stack(binary)).to(device),
                    torch.from_numpy(np.stack(states)).to(device),
                    torch.tensor(view_labels, device=device),
                    torch.tensor(sample_labels, device=device),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch+1} step={step}: {parts}")
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimiser)
            scaler.update()
            epoch_parts.append(parts)
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs} step {step}/{total_steps} "
                    + " ".join(f"{key}={value:.4f}" for key, value in parts.items()), flush=True
                )
        scheduler.step()
        means = {key: float(np.mean([part[key] for part in epoch_parts])) for key in epoch_parts[0]}
        print(f"epoch {epoch+1} mean={means} elapsed={time.time()-started:.0f}s", flush=True)
        torch.save(checkpoint(model, config, epoch + 1, optimiser, scaler), args.out / f"vlm_primary_v5_epoch{epoch+1}.pt")

    final = args.out / "vlm_primary_v5.pt"
    torch.save(checkpoint(model, config, args.epochs), final)
    metadata = {
        "method": "VLM-primary WinCLIP+/AdaCLIP-inspired v5",
        "official_train_only": True,
        "external_anomaly_data": False,
        "train_groups": len(train_samples),
        "heldout_groups": len(heldout),
        "trainable_parameters": stats,
        "provenance": provenance,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.out / "train_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"TRAIN_COMPLETE final={final}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--v4-checkpoint", type=Path)
    parser.add_argument("--reuse-v4-vlm", action="store_true")
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="dinov2_vitb14")
    parser.add_argument("--dino-size", type=int, default=1036)
    parser.add_argument("--clip-size", type=int, default=448)
    parser.add_argument("--decoder-width", type=int, default=192)
    parser.add_argument("--vlm-width", type=int, default=192)
    parser.add_argument("--prompt-ctx", type=int, default=8)
    parser.add_argument("--max-dino-fraction", type=float, default=0.15)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-groups", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normal-group-probability", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    main(parser.parse_args())
