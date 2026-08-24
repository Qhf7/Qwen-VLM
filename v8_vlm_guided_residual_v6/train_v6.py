#!/usr/bin/env python3
"""Crash-resumable training for DINO-primary/VLM-guided v6."""
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
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "v7_vlm_primary_v5"), str(ROOT / "v6_vlm_zero_shot_v4"), str(ROOT / "v5_multiscale_v3")]

from clip_dense import clip_preprocess_tensor, load_clip
from common import LAYER_SETS, list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed
from hard_negatives import hard_negative_map
from losses_v6 import V6Loss
from model_v6 import VLMGuidedResidualV6, load_trusted_checkpoint
from pseudo_v6 import TRAIN_GENERATORS, make_targets
from progress_v6 import ProgressBar
from reference_v6 import aligned_clip_reference, aligned_dino_reference
from splits import split_categories


def configure_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.dino_decoder,
        model.clip_encoder.adapter,
        model.clip_encoder.patch_projection,
        model.prompt.dynamic,
        model.local_semantic,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for parameter in (
        model.prompt.context, model.prompt.state_residual, model.semantic_temperature,
        model.reference_weight_logit, model.candidate_threshold,
        model.candidate_temperature_log, model.negative_residual_fraction_logit,
        model.clip_encoder.window_logits,
    ):
        parameter.requires_grad_(True)


def trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_progress(path, stage, current, total, started, values):
    elapsed = time.time() - started
    rate = current / elapsed if elapsed > 0 else 0.0
    # Keep progress metadata separate from the loss dictionary. ``parts``
    # also contains a metric named ``total``; merging it after the metadata
    # silently changed the displayed total step count in earlier runs.
    payload = {
        **values,
        "loss": float(values["total"]) if "total" in values else None,
        "stage": stage,
        "current": current,
        "total_steps": total,
        "percent": 100.0 * current / max(total, 1),
        "elapsed_seconds": elapsed,
        "eta_seconds": (total - current) / rate if rate > 0 else None,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


def capture(model, optimiser, scaler, scheduler, config, epoch, next_step, order, rng, path):
    atomic_save({
        "format": "vlm_guided_residual_v6_resume_v1", "model": model.compact_state_dict(),
        "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(), "config": config, "epoch": epoch,
        "next_step": next_step, "order": order, "numpy_rng": rng.bit_generator.state,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def main(args):
    set_seed(args.seed)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)
    final_path = args.out / "vlm_guided_residual_v6.pt"
    resume_path = args.out / "resume_latest.pt"
    if final_path.exists():
        print(f"TRAIN_ALREADY_COMPLETE final={final_path}", flush=True)
        return

    samples = list_samples(args.data_root / "Train")
    train_samples, heldout = split_categories(samples, args.heldout_fold, args.folds)
    if args.heldout_fold < 0:
        train_samples, heldout = samples, []
    if args.limit:
        train_samples = train_samples[:args.limit]
    dino = load_dinov2(args.arch, device, args.weight_dir).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    layers = LAYER_SETS[args.arch]
    with torch.inference_mode():
        probe, _ = per_layer_features(dino, load_many(train_samples[0].paths[:1]), args.dino_size, device, layers)
    dims = [int(x.shape[-1]) for x in probe]
    model = VLMGuidedResidualV6(
        dims, clip_model, args.decoder_width, args.vlm_width, args.prompt_ctx, args.max_vlm_residual
    ).to(device)
    provenance = {}
    if args.initial_checkpoint and not resume_path.exists():
        provenance["initial_checkpoint"] = model.load_v4_adaptation(
            load_trusted_checkpoint(args.initial_checkpoint), reuse_vlm=args.reuse_vlm
        )
    configure_trainable(model)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = V6Loss().to(device)
    rng = np.random.default_rng(args.seed)
    config = {
        "dims": dims, "layers": list(layers), "arch": args.arch, "dino_size": args.dino_size,
        "clip_size": args.clip_size, "decoder_width": args.decoder_width, "vlm_width": args.vlm_width,
        "prompt_ctx": args.prompt_ctx, "max_vlm_residual": args.max_vlm_residual,
        "heldout_fold": args.heldout_fold, "train_generators": list(TRAIN_GENERATORS),
    }
    epoch, next_step, order = 0, 0, None
    if resume_path.exists():
        payload = load_trusted_checkpoint(resume_path)
        if payload.get("format") != "vlm_guided_residual_v6_resume_v1" or payload["config"] != config:
            raise RuntimeError("incompatible v6 resume checkpoint")
        model.load_compact_state_dict(payload["model"])
        optimiser.load_state_dict(payload["optimiser"]); scaler.load_state_dict(payload["scaler"])
        scheduler.load_state_dict(payload["scheduler"])
        epoch, next_step, order = int(payload["epoch"]), int(payload["next_step"]), payload["order"]
        rng.bit_generator.state = payload["numpy_rng"]; random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    else:
        torch.save({**config, "model": model.compact_state_dict(), "epoch": 0}, args.out / "epoch0.pt")

    stats = trainable_summary(model)
    steps_per_epoch = math.ceil(len(train_samples) / args.batch_groups)
    total_steps = steps_per_epoch * args.epochs
    started = time.time()
    print(
        f"v6 device={device} train_groups={len(train_samples)} heldout={len(heldout)} "
        f"epoch={epoch+1}/{args.epochs} step={next_step}/{steps_per_epoch} "
        f"trainable={stats['trainable']:,}/{stats['total']:,} provenance={provenance}", flush=True
    )
    while epoch < args.epochs:
        if order is None:
            order = rng.permutation(len(train_samples))
            next_step = 0
        starts = list(range(0, len(order), args.batch_groups))
        model.train()
        epoch_parts = []
        bar = ProgressBar(
            f"train_epoch_{epoch+1}_of_{args.epochs}", len(starts), next_step,
            every=max(1, args.log_every),
        )
        for step in range(next_step, len(starts)):
            selected = [train_samples[int(i)] for i in order[starts[step]:starts[step] + args.batch_groups]]
            images, originals, binary, states, hard, view_labels, sample_labels = [], [], [], [], [], [], []
            for sample in selected:
                normals = load_many(sample.paths)
                targets = make_targets(
                    normals, rng, force_normal=bool(rng.random() < args.normal_group_probability)
                )
                images.extend(t.image for t in targets); originals.extend(t.original for t in targets)
                binary.extend(t.binary_mask for t in targets); states.extend(t.state_mask for t in targets)
                hard.extend(hard_negative_map(t.original, 448) for t in targets)
                view_labels.extend(t.is_anomaly for t in targets)
                sample_labels.append(float(any(t.is_anomaly > 0.5 for t in targets)))
            with torch.no_grad():
                current_dino, _ = per_layer_features(dino, images, args.dino_size, device, layers)
                normal_dino, _ = per_layer_features(dino, originals, args.dino_size, device, layers)
                normal_clip = clip_preprocess_tensor(originals, args.clip_size, device)
                _, normal_patch = model.encode_clip(normal_clip)
                dino_reference = aligned_dino_reference(current_dino, normal_dino)
            clip_x = clip_preprocess_tensor(images, args.clip_size, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = model.encode_clip(clip_x)
                clip_reference = aligned_clip_reference(encoded[1], normal_patch)
                output = model(
                    current_dino, clip_x, dino_reference=dino_reference,
                    clip_reference=clip_reference, groups=len(selected), encoded_clip=encoded,
                )
                loss, parts = loss_fn(
                    model, output, torch.from_numpy(np.stack(binary)).to(device),
                    torch.from_numpy(np.stack(states)).to(device),
                    torch.from_numpy(np.stack(hard)).to(device),
                    torch.tensor(view_labels, device=device), torch.tensor(sample_labels, device=device),
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch+1} step={step}: {parts}")
            optimiser.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimiser); scaler.update()
            next_step = step + 1; epoch_parts.append(parts)
            absolute = epoch * steps_per_epoch + next_step
            if step % args.log_every == 0 or next_step == steps_per_epoch:
                recent = epoch_parts[-min(20, len(epoch_parts)):]
                values = {key: float(np.mean([x[key] for x in recent])) for key in parts}
                report = save_progress(
                    args.out / "train_progress.json", f"epoch_{epoch+1}_of_{args.epochs}",
                    absolute, total_steps, started, values,
                )
                print(
                    f"PROGRESS {report['percent']:.2f}% epoch={epoch+1}/{args.epochs} "
                    f"step={next_step}/{steps_per_epoch} loss={parts['total']:.5f} "
                    f"p_normal={parts['normal']:.4f} p_boundary={parts['boundary']:.4f} "
                    f"p_area={parts['area']:.4f} eta={report['eta_seconds']:.0f}s", flush=True
                )
                bar.update(
                    next_step,
                    {
                        "loss": float(parts["total"]),
                        "normal": float(parts["normal"]),
                        "boundary": float(parts["boundary"]),
                        "area": float(parts["area"]),
                    },
                    force=True,
                )
            if next_step % args.checkpoint_every == 0 or next_step == steps_per_epoch:
                capture(model, optimiser, scaler, scheduler, config, epoch, next_step, order, rng, resume_path)
        bar.close({"loss": float(np.mean([x["total"] for x in epoch_parts]))})
        torch.save({**config, "model": model.compact_state_dict(), "epoch": epoch + 1}, args.out / f"epoch{epoch+1}.pt")
        scheduler.step(); epoch += 1; next_step = 0; order = None
        capture(model, optimiser, scaler, scheduler, config, epoch, 0, None, rng, resume_path)
    torch.save({**config, "model": model.compact_state_dict(), "epoch": args.epochs}, final_path)
    (args.out / "train_meta.json").write_text(json.dumps({
        "method": "DINO-primary plus bounded VLM residual v6", "official_train_only": True,
        "external_anomaly_data": False, "online_api": False, "train_groups": len(train_samples),
        "heldout_groups": len(heldout), "trainable_parameters": stats, "provenance": provenance,
        "elapsed_seconds": time.time() - started,
    }, indent=2), encoding="utf-8")
    print(f"TRAIN_COMPLETE final={final_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--weight-dir", type=Path, required=True)
    p.add_argument("--clip-weight", type=Path, required=True); p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--initial-checkpoint", type=Path); p.add_argument("--reuse-vlm", action="store_true")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--arch", default="dinov2_vitb14")
    p.add_argument("--dino-size", type=int, default=1036); p.add_argument("--clip-size", type=int, default=448)
    p.add_argument("--decoder-width", type=int, default=192); p.add_argument("--vlm-width", type=int, default=192)
    p.add_argument("--prompt-ctx", type=int, default=8); p.add_argument("--max-vlm-residual", type=float, default=1.25)
    p.add_argument("--heldout-fold", type=int, default=0); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch-groups", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--normal-group-probability", type=float, default=0.50); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5); p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260826)
    main(p.parse_args())
