#!/usr/bin/env python3
"""Crash-resumable VLM-primary v5 training."""
from __future__ import annotations

import argparse
import json
import math
import random
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
from progress_v5 import ProgressReporter
from reference_v5 import aligned_reference_map
from splits import split_categories
from train_v5 import configure_trainable, trainable_summary


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture(model, optimiser, scaler, scheduler, config, epoch_index, next_step, order, parts, rng, path):
    atomic_save(
        {
            "format": "vlm_primary_v5_resume_v1",
            "model": model.compact_state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch_index": epoch_index,
            "next_step": next_step,
            "order": None if order is None else np.asarray(order, np.int64),
            "parts": parts,
            "numpy_rng": rng.bit_generator.state,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def restore_rng(payload, rng):
    rng.bit_generator.state = payload["numpy_rng"]
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def main(args):
    set_seed(args.seed)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)
    resume_path = args.out / "resume_latest.pt"
    if (args.out / "vlm_primary_v5.pt").exists():
        print(f"TRAIN_ALREADY_COMPLETE final={args.out / 'vlm_primary_v5.pt'}", flush=True)
        return
    existing = [path for path in args.out.iterdir() if path.name not in {"resume_latest.pt.tmp"}]
    if existing and not resume_path.exists():
        raise FileExistsError(f"refusing non-resume write into non-empty directory: {args.out}")

    samples = list_samples(args.data_root / "Train")
    train_samples, heldout = split_categories(samples, args.heldout_fold, args.folds)
    if args.heldout_fold < 0:
        train_samples, heldout = samples, []
    if args.limit:
        train_samples = train_samples[:args.limit]
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
    if not resume_path.exists():
        if args.v4_checkpoint:
            provenance["v4_adaptation"] = model.load_v4_adaptation(
                load_trusted_checkpoint(args.v4_checkpoint), reuse_vlm=args.reuse_v4_vlm
            )
        if args.v3_checkpoint:
            model.load_v3_decoder(load_trusted_checkpoint(args.v3_checkpoint))
            provenance["v3_decoder"] = str(args.v3_checkpoint)
    configure_trainable(model)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = V5Loss().to(device)
    rng = np.random.default_rng(args.seed)
    config = {
        "dims": dims, "layers": list(layers), "arch": args.arch,
        "dino_size": args.dino_size, "clip_size": args.clip_size,
        "decoder_width": args.decoder_width, "vlm_width": args.vlm_width,
        "prompt_ctx": args.prompt_ctx, "max_dino_fraction": args.max_dino_fraction,
        "heldout_fold": args.heldout_fold,
        "train_generators": list(DEFAULT_TRAIN_GENERATORS),
    }
    epoch_index, next_step, order, epoch_parts = 0, 0, None, []
    exact_resume = False
    if resume_path.exists():
        payload = load_trusted_checkpoint(resume_path)
        if payload.get("format") != "vlm_primary_v5_resume_v1":
            raise RuntimeError("unsupported v5 resume format")
        if payload["config"] != config:
            raise RuntimeError("resume configuration mismatch")
        model.load_compact_state_dict(payload["model"])
        optimiser.load_state_dict(payload["optimiser"])
        scaler.load_state_dict(payload["scaler"])
        scheduler.load_state_dict(payload["scheduler"])
        epoch_index = int(payload["epoch_index"])
        next_step = int(payload["next_step"])
        order = payload["order"]
        epoch_parts = list(payload["parts"])
        restore_rng(payload, rng)
        exact_resume = True
    else:
        initial = {**config, "model": model.compact_state_dict(), "epoch": 0}
        torch.save(initial, args.out / "vlm_primary_v5_initial.pt")
        capture(model, optimiser, scaler, scheduler, config, 0, 0, None, [], rng, resume_path)

    stats = trainable_summary(model)
    total_steps = int(math.ceil(len(train_samples) / args.batch_groups))
    started = time.time()
    print(
        f"v5-resume device={device} train_groups={len(train_samples)} heldout_groups={len(heldout)} "
        f"epoch={epoch_index+1}/{args.epochs} next_step={next_step} exact_resume={exact_resume} "
        f"trainable={stats['trainable']:,}/{stats['total']:,} provenance={provenance}", flush=True
    )

    while epoch_index < args.epochs:
        if order is None:
            order = rng.permutation(len(train_samples))
            next_step, epoch_parts = 0, []
        starts = list(range(0, len(order), args.batch_groups))
        model.train()
        print(f"epoch {epoch_index+1}/{args.epochs} start_step={next_step}/{total_steps}", flush=True)
        progress = ProgressReporter(
            stage=f"train_epoch_{epoch_index+1}_of_{args.epochs}",
            total=total_steps,
            initial=next_step,
            progress_path=args.out.parent / "train_progress.json",
        )
        if next_step:
            progress.update(next_step, {"resume": True}, force=True)
        for step in range(next_step, len(starts)):
            start = starts[step]
            selected = [train_samples[int(i)] for i in order[start:start + args.batch_groups]]
            images, originals, binary, states, view_labels, sample_labels = [], [], [], [], [], []
            for sample in selected:
                normals = load_many(sample.paths)
                targets = make_group_targets_v5(
                    normals, rng, size=448,
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
                _, normal_patch = model.encode_clip(clip_preprocess_tensor(originals, args.clip_size, device))
            clip_x = clip_preprocess_tensor(images, args.clip_size, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = model.encode_clip(clip_x)
                reference = aligned_reference_map(encoded[1], normal_patch)
                output = model(dino_maps, clip_x, reference_map=reference, groups=len(selected), encoded_clip=encoded)
                loss, parts = loss_fn(
                    model, output,
                    torch.from_numpy(np.stack(binary)).to(device),
                    torch.from_numpy(np.stack(states)).to(device),
                    torch.tensor(view_labels, device=device),
                    torch.tensor(sample_labels, device=device),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch_index+1} step={step}: {parts}")
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimiser); scaler.update()
            epoch_parts.append(parts); next_step = step + 1
            if step % args.log_every == 0:
                recent = epoch_parts[-min(len(epoch_parts), args.loss_window):]
                progress.update(
                    next_step,
                    {
                        "loss": float(parts["total"]),
                        "loss_avg": float(np.mean([item["total"] for item in recent])),
                        "vlm": float(parts["vlm"]),
                        "semantic": float(parts["semantic"]),
                        "state": float(parts["state"]),
                        "region": float(parts["region"]),
                        "reference": float(parts["reference"]),
                        "lr": float(optimiser.param_groups[0]["lr"]),
                        "dino_fraction": float(parts["dino_fraction"]),
                    },
                )
            if next_step % args.checkpoint_every == 0 or next_step == total_steps:
                capture(
                    model, optimiser, scaler, scheduler, config, epoch_index, next_step,
                    order, epoch_parts, rng, resume_path,
                )
                print(f"resume_saved epoch={epoch_index+1} next_step={next_step}/{total_steps}", flush=True)
        progress.update(
            total_steps,
            {"loss_avg": float(np.mean([item["total"] for item in epoch_parts]))},
            force=True,
        )
        means = {key: float(np.mean([part[key] for part in epoch_parts])) for key in epoch_parts[0]}
        print(f"epoch {epoch_index+1} mean={means} elapsed={time.time()-started:.0f}s", flush=True)
        torch.save({**config, "model": model.compact_state_dict(), "epoch": epoch_index + 1},
                   args.out / f"vlm_primary_v5_epoch{epoch_index+1}.pt")
        scheduler.step(); epoch_index += 1; next_step = 0; order = None; epoch_parts = []
        capture(model, optimiser, scaler, scheduler, config, epoch_index, 0, None, [], rng, resume_path)

    final = args.out / "vlm_primary_v5.pt"
    torch.save({**config, "model": model.compact_state_dict(), "epoch": args.epochs}, final)
    metadata = {
        "method": "VLM-primary v5 crash-resumable training",
        "official_train_only": True,
        "external_anomaly_data": False,
        "train_groups": len(train_samples), "heldout_groups": len(heldout),
        "trainable_parameters": stats, "provenance": provenance,
        "exact_resume_used": exact_resume, "elapsed_seconds_this_invocation": round(time.time()-started, 2),
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
    parser.add_argument("--loss-window", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    main(parser.parse_args())
