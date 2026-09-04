"""Mixed-precision fold1 training for frozen-Qwen E1."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from e1.balanced_sampler import BalancedBatchSampler
from e1.data import E1TrainingDataset, encode_examples, list_collate
from e1.evaluate_fold1 import evaluate_model, sha256_file
from e1.evaluate_fold1 import build_model


class FixedBatchSampler:
    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_factor(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    best_macro_aupr: float,
    best_step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "best_macro_aupr": best_macro_aupr,
        "best_step": best_step,
        "decoder": model.decoder.state_dict(),
        "prompt_learner": model.prompt_learner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--config", type=Path, default=Path("e1/config.json"))
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--pool", type=Path, default=Path("artifacts/e1/sampler_pool.parquet"))
    parser.add_argument("--cache-root", type=Path, default=Path("Train448"))
    parser.add_argument("--dev-cache-root", type=Path, default=Path("Dataset448"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--cache-manifest", type=Path, default=Path("artifacts/e1/train448_cache_manifest.csv"))
    parser.add_argument("--schedule", type=Path, default=Path("artifacts/e1/fold1_schedule.parquet"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_fold1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--eval-steps", type=str)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--empty-mask-dice-only-nonempty",
        action="store_true",
        help="E1 ablation A: exclude empty GT masks from the Dice term",
    )
    parser.add_argument(
        "--lambda-seg",
        type=float,
        help="E1 ablation B: override only the segmentation-loss weight",
    )
    parser.add_argument(
        "--feature-layers",
        type=str,
        default="7,15,23,31",
        help="comma-separated visual block indices for decoder features",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    feature_layers = tuple(int(value.strip()) for value in args.feature_layers.split(",") if value.strip())
    if not feature_layers:
        raise ValueError("feature-layers must contain at least one layer")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    lambda_seg = (
        float(args.lambda_seg)
        if args.lambda_seg is not None
        else float(config["loss"]["lambda_seg"])
    )
    if lambda_seg <= 0:
        raise ValueError("lambda-seg must be positive")
    max_steps = args.max_steps or int(training["max_steps"])
    run_until_step = args.stop_after_step or max_steps
    if run_until_step > max_steps:
        raise ValueError("stop-after-step cannot exceed the frozen max_steps")
    if args.skip_eval:
        eval_steps = set()
    else:
        eval_steps = (
            {int(value) for value in args.eval_steps.split(",") if value}
            if args.eval_steps else set(int(value) for value in training["evaluation_steps"])
        )
        eval_steps = {value for value in eval_steps if value <= max_steps}
        if max_steps not in eval_steps:
            eval_steps.add(max_steps)
    logical_batch_size = int(training["logical_batch_size"])
    micro_batch_size = int(training["micro_batch_size"])
    if micro_batch_size != 1 or logical_batch_size % micro_batch_size:
        raise ValueError("the frozen fold1 protocol requires micro_batch_size=1")
    seed = int(training["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train = pd.read_parquet(args.train)
    dev = pd.read_parquet(args.dev)
    pool = pd.read_parquet(args.pool)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    overlap = sorted((set(train.class_name) | set(dev.class_name)) & holdout)
    if overlap:
        raise RuntimeError(f"fold1 train/dev contains holdout classes: {overlap}")
    cache_manifest = pd.read_csv(args.cache_manifest)
    if len(cache_manifest) != len(train) or set(cache_manifest.image_path) != set(train.image_path):
        raise RuntimeError("train448 cache manifest does not match fold1/train")
    schedule = pd.read_parquet(args.schedule) if args.schedule.is_file() else None
    if schedule is not None:
        schedule = schedule[schedule.step.le(max_steps)].copy()
        if len(schedule) != max_steps * logical_batch_size:
            raise RuntimeError("fixed schedule does not cover every requested step")
        expected = pool.iloc[schedule.dataset_index.to_numpy()].reset_index(drop=True)
        if not (
            expected.image_path.astype(str).eq(schedule.image_path.astype(str)).all()
            and expected.source_type.astype(str).eq(schedule.source_type.astype(str)).all()
        ):
            raise RuntimeError("fixed schedule no longer matches sampler_pool")
        active_schedule = schedule[schedule.step.le(run_until_step)]
        required_images = set(active_schedule.image_path) | (set(active_schedule.donor_path) - {""})
        required_masks = set(active_schedule.mask_path) - {""}
    else:
        required_images = set(train.image_path)
        required_masks = set(train.loc[train.view_label.eq(1), "mask_path"])
    missing_images = [path for path in required_images if not (args.cache_root / path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"missing {len(missing_images)} scheduled training images")
    missing_masks = [path for path in required_masks if not (args.cache_root / path).is_file()]
    if missing_masks:
        raise FileNotFoundError(f"missing {len(missing_masks)} scheduled training masks")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": config["protocol"],
        "config": config,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "pool_sha256": sha256_file(args.pool),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "holdout_intersection": overlap,
        "max_steps": max_steps,
        "run_until_step": run_until_step,
        "evaluation_steps": sorted(eval_steps),
        "schedule_sha256": sha256_file(args.schedule) if schedule is not None else None,
        "scheduled_unique_images": len(required_images),
        "scheduled_unique_masks": len(required_masks),
        "empty_mask_dice_only_nonempty": bool(args.empty_mask_dice_only_nonempty),
        "lambda_seg": lambda_seg,
        "feature_layers": list(feature_layers),
        "experiment_variant": (
            "E1-C-feature-layers"
            if tuple(feature_layers) != (7, 15, 23, 31)
            else (
                "E1-B-lambda-seg"
                if args.lambda_seg is not None
                else (
                    "E1-A-empty-mask-dice-nonempty"
                    if args.empty_mask_dice_only_nonempty
                    else "E1-B0-original"
                )
            )
        ),
    }
    (args.out_dir / "frozen_training_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    precision_name = str(training["autocast_dtype"])
    precision_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if precision_name not in precision_dtypes:
        raise ValueError(f"unsupported autocast dtype: {precision_name}")
    amp_dtype = precision_dtypes[precision_name]
    if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frozen E1 protocol requires CUDA BF16 support")

    device = torch.device("cuda:0")
    model, processor = build_model(
        args.model_id,
        None,
        device,
        dtype=amp_dtype,
        dice_nonempty_only=args.empty_mask_dice_only_nonempty,
        lambda_seg=lambda_seg,
        feature_layers=feature_layers,
    )
    summary = model.trainable_parameter_summary()
    if summary["base_model"] != 0:
        raise RuntimeError("base model contains trainable parameters")
    parameter_groups = [
        {
            "params": list(model.decoder.parameters()),
            "lr": float(config["optimizer"]["decoder_lr"]),
        },
        {
            "params": list(model.prompt_learner.parameters()),
            "lr": float(config["optimizer"]["prompt_lr"]),
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda zero_based_step: learning_rate_factor(
            zero_based_step, int(training["warmup_steps"]), max_steps
        ),
    )
    # FP16 scaling overflowed deterministic outlier batches even after moving
    # reductions to FP32.  The audited 4090D path therefore uses native BF16
    # autocast (same mixed-precision memory benefit, FP32-like exponent range)
    # without loss scaling.  FP16 remains supported for explicit experiments.
    scaler_enabled = bool(training["grad_scaler_enabled"])
    if scaler_enabled != (amp_dtype is torch.float16):
        raise ValueError("GradScaler must be enabled exactly for the FP16 protocol")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=float(training["grad_scaler_init_scale"]),
        growth_interval=2000,
    )
    start_step = 0
    best_macro_aupr = -float("inf")
    best_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior_config = state.get("config", {})
        prior_ablation = bool(prior_config.get("empty_mask_dice_only_nonempty", False))
        if prior_ablation != bool(args.empty_mask_dice_only_nonempty):
            raise RuntimeError("resume checkpoint ablation mode does not match this run")
        prior_lambda_seg = float(prior_config.get("lambda_seg", 0.2))
        if not math.isclose(prior_lambda_seg, lambda_seg, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("resume checkpoint lambda_seg does not match this run")
        model.decoder.load_state_dict(state["decoder"])
        model.prompt_learner.load_state_dict(state["prompt_learner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        best_macro_aupr = float(state["best_macro_aupr"])
        best_step = int(state["best_step"])
        print(f"[resume] step={start_step} best={best_macro_aupr:.6f}@{best_step}", flush=True)

    dataset = E1TrainingDataset(pool, args.cache_root, global_seed=seed)
    if schedule is not None:
        batches = [
            part.sort_values("batch_position").dataset_index.astype(int).tolist()
            for _, part in schedule.groupby("step", sort=True)
        ]
        sampler = FixedBatchSampler(batches)
    else:
        sampler = BalancedBatchSampler(
            pool,
            batch_size=logical_batch_size,
            steps_per_epoch=max_steps,
            seed=seed,
        )
    dataset.set_epoch(0)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=list_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    history_path = args.out_dir / "training_history.csv"
    if history_path.is_file():
        prior_history = pd.read_csv(history_path)
        history = prior_history[prior_history.step.le(start_step)].to_dict("records")
    else:
        history = []
    run_started = time.perf_counter()
    first_gradient_gate_done = start_step > 0

    for step, examples in enumerate(loader, 1):
        if step <= start_step:
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sums = {
            "loss": 0.0,
            "classification": 0.0,
            "segmentation": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "region": 0.0,
        }
        source_stats = {
            source: {
                "loss": [],
                "classification": [],
                "segmentation": [],
                "focal": [],
                "dice": [],
                "region": [],
                "mask_probability_mean": [],
                "predicted_area": [],
                "dice_nonempty": [],
                "dice_empty": [],
                "dice_nonempty_count": 0,
                "dice_empty_count": 0,
            }
            for source in ("real_anomaly", "pseudo_anomaly", "normal", "hard_normal")
        }
        all_dice_nonempty = []
        all_dice_empty = []
        step_started = time.perf_counter()
        for example in examples:
            inputs, targets = encode_examples(processor, [example], device)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(inputs, **targets)
                scaled_loss = output.loss / len(examples)
            scaler.scale(scaled_loss).backward()
            loss_sums["loss"] += float(output.loss.detach())
            loss_sums["classification"] += float(output.classification_loss.detach())
            loss_sums["segmentation"] += float(output.segmentation_loss.detach())
            loss_sums["focal"] += float(output.focal_loss.detach())
            loss_sums["dice"] += float(output.dice_loss.detach())
            loss_sums["region"] += float(output.region_loss.detach())
            source = str(example["source_type"])
            stats = source_stats[source]
            stats["loss"].append(float(output.loss.detach()))
            stats["classification"].append(float(output.classification_loss.detach()))
            stats["segmentation"].append(float(output.segmentation_loss.detach()))
            stats["focal"].append(float(output.focal_loss.detach()))
            stats["dice"].append(float(output.dice_loss.detach()))
            stats["region"].append(float(output.region_loss.detach()))
            stats["mask_probability_mean"].append(float(output.mask_probability_mean))
            stats["predicted_area"].append(float(output.predicted_area))
            if output.dice_nonempty_count:
                value = float(output.dice_loss_nonempty.detach())
                stats["dice_nonempty"].append(value)
                all_dice_nonempty.append(value)
            if output.dice_empty_count:
                value = float(output.dice_loss_empty.detach())
                stats["dice_empty"].append(value)
                all_dice_empty.append(value)
            stats["dice_nonempty_count"] += int(output.dice_nonempty_count)
            stats["dice_empty_count"] += int(output.dice_empty_count)
            del output, inputs, targets, scaled_loss

        if not first_gradient_gate_done:
            if any(parameter.grad is not None for parameter in model.base_model.parameters()):
                raise RuntimeError("a frozen base parameter received a gradient")
            first_gradient_gate_done = True
        scaler.unscale_(optimizer)
        trainable = list(model.decoder.parameters()) + list(model.prompt_learner.parameters())
        grad_norm = float(clip_grad_norm_(trainable, float(training["gradient_clip_norm"])))
        if not math.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite unscaled gradient at step {step}; update was not accepted"
            )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if scale_after < scale_before:
            raise RuntimeError(f"GradScaler skipped a finite-gate update at step {step}")
        scheduler.step()
        mean_losses = {key: value / len(examples) for key, value in loss_sums.items()}
        record = {
            "step": step,
            **mean_losses,
            "grad_norm": grad_norm,
            "grad_scale": scale_after,
            "lr": optimizer.param_groups[0]["lr"],
            "step_s": time.perf_counter() - step_started,
            "elapsed_s": time.perf_counter() - run_started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        for source, stats in source_stats.items():
            prefix = f"source_{source}"
            record[f"{prefix}_n"] = len(stats["loss"])
            for key in ("loss", "classification", "segmentation", "focal", "dice", "region"):
                record[f"{prefix}_{key}_mean"] = (
                    float(np.mean(stats[key])) if stats[key] else float("nan")
                )
            for key in ("mask_probability_mean", "predicted_area"):
                values = stats[key]
                record[f"{prefix}_{key}_mean"] = float(np.mean(values)) if values else float("nan")
                record[f"{prefix}_{key}_p50"] = float(np.percentile(values, 50)) if values else float("nan")
                record[f"{prefix}_{key}_p95"] = float(np.percentile(values, 95)) if values else float("nan")
            record[f"{prefix}_dice_nonempty_mean"] = (
                float(np.mean(stats["dice_nonempty"])) if stats["dice_nonempty"] else float("nan")
            )
            record[f"{prefix}_dice_empty_mean"] = (
                float(np.mean(stats["dice_empty"])) if stats["dice_empty"] else float("nan")
            )
            record[f"{prefix}_dice_score_nonempty_mean"] = (
                1.0 - record[f"{prefix}_dice_nonempty_mean"]
                if stats["dice_nonempty"] else float("nan")
            )
            record[f"{prefix}_dice_score_empty_mean"] = (
                1.0 - record[f"{prefix}_dice_empty_mean"]
                if stats["dice_empty"] else float("nan")
            )
            record[f"{prefix}_dice_nonempty_count"] = stats["dice_nonempty_count"]
            record[f"{prefix}_dice_empty_count"] = stats["dice_empty_count"]
        record["dice_nonempty_mean"] = (
            float(np.mean(all_dice_nonempty)) if all_dice_nonempty else float("nan")
        )
        record["dice_empty_mean"] = (
            float(np.mean(all_dice_empty)) if all_dice_empty else float("nan")
        )
        record["dice_score_nonempty_mean"] = (
            1.0 - record["dice_nonempty_mean"] if all_dice_nonempty else float("nan")
        )
        record["dice_score_empty_mean"] = (
            1.0 - record["dice_empty_mean"] if all_dice_empty else float("nan")
        )
        record["dice_nonempty_count"] = len(all_dice_nonempty)
        record["dice_empty_count"] = len(all_dice_empty)
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if step % 10 == 0 or step == 1:
            print(
                f"[train] {step}/{max_steps} loss={mean_losses['loss']:.4f} "
                f"grad={grad_norm:.3f} step={record['step_s']:.2f}s",
                flush=True,
            )
        if step % 50 == 0:
            save_checkpoint(
                args.out_dir / "latest.pt", model, optimizer, scheduler, scaler,
                step, best_macro_aupr, best_step, protocol,
            )

        if step in eval_steps:
            tag = f"step{step:04d}"
            metrics = evaluate_model(
                model,
                processor,
                dev,
                args.dev_cache_root,
                device,
                args.out_dir,
                tag,
                gt_root=args.gt_root,
            )
            macro_aupr = float(metrics["macro_aupr"])
            print(
                f"[dev] step={step} macro_aupr={macro_aupr:.4f} "
                f"macro_f1={metrics['macro_f1max']:.4f} "
                f"small_recall={metrics['small_defect_recall']:.4f} "
                f"area_ok={metrics['predicted_area']['ok_view_mean']:.4f}",
                flush=True,
            )
            if macro_aupr > best_macro_aupr:
                best_macro_aupr = macro_aupr
                best_step = step
                save_checkpoint(
                    args.out_dir / "best.pt", model, optimizer, scheduler, scaler,
                    step, best_macro_aupr, best_step, protocol,
                )
            save_checkpoint(
                args.out_dir / "latest.pt", model, optimizer, scheduler, scaler,
                step, best_macro_aupr, best_step, protocol,
            )
        if step >= run_until_step:
            break

    final = {
        "completed_steps": run_until_step,
        "best_step": best_step,
        "best_macro_aupr": best_macro_aupr,
        "trainable_parameters": summary,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_s": time.perf_counter() - run_started,
        "holdout_intersection": overlap,
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
