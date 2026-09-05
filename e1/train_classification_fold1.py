"""Stage-1 fold1 training with image-level classification loss only."""

from __future__ import annotations

import argparse
import hashlib
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

from b0.b0plus_protocol import P4_CLASSIFICATION_ONLY
from e1.data import E1ClassificationDataset, RealClassificationDataset, encode_classification_examples, list_collate
from e1.evaluate_classification_fold1 import (
    build_classification_model,
    evaluate_classification,
    sha256_file,
)


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
    protocol: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "classification_only": True,
        "step": step,
        "best_macro_aupr": best_macro_aupr,
        "best_step": best_step,
        "decoder": model.decoder.state_dict(),
        "prompt_learner": model.prompt_learner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "protocol": protocol,
    }, path)


def audit_split(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    train_classes = set(train.class_name)
    dev_classes = set(dev.class_name)
    if train_classes != dev_classes or len(train_classes) != 18:
        raise RuntimeError("train/dev must contain the same 18 component classes")
    group_overlap = set(train.group_path) & set(dev.group_path)
    visual_overlap = set(train.visual_group_id) & set(dev.visual_group_id)
    if group_overlap or visual_overlap:
        raise RuntimeError("train/dev group or visual-group leakage detected")
    dev_counts = dev.groupby(["class_name", "sample_label"]).group_path.nunique()
    if not dev_counts.eq(20).all():
        raise RuntimeError("Dev must contain 20 OK and 20 NG groups per class")
    return {
        "classes": sorted(train_classes),
        "train_views": len(train),
        "train_groups": int(train.group_path.nunique()),
        "train_ok_groups": int(train.loc[train.sample_label.eq(0), "group_path"].nunique()),
        "train_ng_groups": int(train.loc[train.sample_label.eq(1), "group_path"].nunique()),
        "dev_views": len(dev),
        "dev_groups": int(dev.group_path.nunique()),
        "dev_groups_per_class_and_label": 20,
        "group_overlap": 0,
        "visual_group_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--config", type=Path, default=Path("e1/config.json"))
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--pool", type=Path, default=Path("artifacts/e1/sampler_pool.parquet"))
    parser.add_argument("--schedule", type=Path, default=Path("artifacts/e1/fold1_schedule.parquet"))
    parser.add_argument("--cache-root", type=Path, default=Path("Train448"))
    parser.add_argument("--dev-cache-root", type=Path, default=Path("Dataset448"))
    parser.add_argument("--cache-manifest", type=Path, default=Path("artifacts/e1/train448_cache_manifest.csv"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_classification_fold1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--eval-steps", type=str)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--feature-layers", default="7,15,23,31")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    max_steps = args.max_steps or int(training["max_steps"])
    run_until_step = args.stop_after_step or max_steps
    if run_until_step > max_steps:
        raise ValueError("stop-after-step cannot exceed max-steps")
    eval_steps = set() if args.skip_eval else (
        {int(value) for value in args.eval_steps.split(",") if value}
        if args.eval_steps
        else set(int(value) for value in training["evaluation_steps"])
    )
    eval_steps = {value for value in eval_steps if value <= max_steps}
    if not args.skip_eval:
        eval_steps.add(max_steps)
    logical_batch_size = int(training["logical_batch_size"])
    if int(training["micro_batch_size"]) != 1:
        raise ValueError("classification stage currently requires micro_batch_size=1")
    seed = int(training["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train = pd.read_parquet(args.train)
    dev = pd.read_parquet(args.dev)
    pool = pd.read_parquet(args.pool)
    schedule = pd.read_parquet(args.schedule)
    split_audit = audit_split(train, dev)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    overlap = sorted((set(train.class_name) | set(dev.class_name)) & holdout)
    if overlap:
        raise RuntimeError(f"fold1 train/dev contains holdout classes: {overlap}")

    cache_manifest = pd.read_csv(args.cache_manifest)
    if not args.real_only and (len(cache_manifest) != len(train) or set(cache_manifest.image_path) != set(train.image_path)):
        raise RuntimeError("train448 cache manifest does not match fold1/train")
    schedule = schedule[schedule.step.le(max_steps)].copy()
    if len(schedule) != max_steps * logical_batch_size:
        raise RuntimeError("fixed schedule does not cover every requested step")
    expected = pool.iloc[schedule.dataset_index.to_numpy()].reset_index(drop=True)
    if not (
        expected.image_path.astype(str).eq(schedule.image_path.astype(str)).all()
        and expected.source_type.astype(str).eq(schedule.source_type.astype(str)).all()
    ):
        raise RuntimeError("fixed schedule no longer matches sampler_pool")
    per_step_source_counts = schedule.groupby(["step", "source_type"]).size().unstack(fill_value=0)
    expected_sources = {"real_anomaly", "normal"} if args.real_only else {"real_anomaly", "pseudo_anomaly", "normal", "hard_normal"}
    expected_quota = logical_batch_size // len(expected_sources)
    if set(per_step_source_counts.columns) != expected_sources or not per_step_source_counts.eq(expected_quota).all().all():
        raise RuntimeError("classification schedule has incorrect source quotas")

    active_schedule = schedule[schedule.step.le(run_until_step)]
    required_images = set(active_schedule.image_path) | (set(active_schedule.donor_path) - {""})
    missing_images = [path for path in required_images if not (args.cache_root / path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"missing {len(missing_images)} scheduled images; first={missing_images[0]}")

    feature_layers = tuple(int(value) for value in args.feature_layers.split(",") if value.strip())
    prompt_sha = hashlib.sha256(P4_CLASSIFICATION_ONLY.encode()).hexdigest()
    protocol = {
        "protocol": "E1-stage1-classification-only-real-v1" if args.real_only else "E1-stage1-classification-only-v1",
        "causal_question": "Does removing mask and region objectives prevent classification collapse?",
        "single_variable_relative_to_original_e1": "classification objective only; no mask/region loss or mask-conditioned pooling",
        "prompt_id": "P4_CLASSIFICATION_ONLY",
        "prompt_sha256": prompt_sha,
        "loss": "binary Yes/No next-token BCE only",
        "segmentation_loss": None,
        "region_loss": None,
        "mask_targets_loaded": False,
        "source_schedule": "balanced real_anomaly/normal views only" if args.real_only else "unchanged 2/2/2/2 real/pseudo/normal/hard-normal",
        "feature_layers": list(feature_layers),
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "pool_sha256": sha256_file(args.pool),
        "schedule_sha256": sha256_file(args.schedule),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "split_audit": split_audit,
        "holdout_intersection": overlap,
        "max_steps": max_steps,
        "run_until_step": run_until_step,
        "evaluation_steps": sorted(eval_steps),
        "optimizer": config["optimizer"],
        "training": training,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "frozen_training_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    precision_name = str(training["autocast_dtype"])
    precision_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    amp_dtype = precision_dtypes[precision_name]
    if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("classification stage requires CUDA BF16 support")
    device = torch.device("cuda:0")
    model, processor = build_classification_model(
        args.model_id, None, device, dtype=amp_dtype, feature_layers=feature_layers
    )
    summary = model.trainable_parameter_summary()
    if summary["base_model"] != 0:
        raise RuntimeError("base model contains trainable parameters")
    decoder_parameters = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    prompt_parameters = [parameter for parameter in model.prompt_learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_parameters, "lr": float(config["optimizer"]["decoder_lr"])},
            {"params": prompt_parameters, "lr": float(config["optimizer"]["prompt_lr"])},
        ],
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_factor(
            step, int(training["warmup_steps"]), max_steps
        ),
    )
    scaler_enabled = amp_dtype is torch.float16
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
        if not state.get("classification_only", False):
            raise RuntimeError("resume checkpoint is not classification-only")
        model.decoder.load_state_dict(state["decoder"])
        model.prompt_learner.load_state_dict(state["prompt_learner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        best_macro_aupr = float(state["best_macro_aupr"])
        best_step = int(state["best_step"])

    dataset_cls = RealClassificationDataset if args.real_only else E1ClassificationDataset
    dataset = dataset_cls(pool, args.cache_root, global_seed=seed)
    batches = [
        part.sort_values("batch_position").dataset_index.astype(int).tolist()
        for _, part in schedule.groupby("step", sort=True)
    ]
    loader = DataLoader(
        dataset,
        batch_sampler=FixedBatchSampler(batches),
        collate_fn=list_collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dataset.set_epoch(0)
    history_path = args.out_dir / "training_history.csv"
    history = []
    if history_path.is_file() and start_step:
        history = pd.read_csv(history_path).query("step <= @start_step").to_dict("records")
    run_started = time.perf_counter()
    first_gradient_gate_done = start_step > 0

    for step, examples in enumerate(loader, 1):
        if step <= start_step:
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        probabilities = []
        source_losses = {source: [] for source in expected_sources}
        source_probabilities = {source: [] for source in expected_sources}
        step_started = time.perf_counter()
        for example in examples:
            inputs, labels = encode_classification_examples(processor, [example], device)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(inputs, sample_labels=labels)
                scaled_loss = output.loss / len(examples)
            scaler.scale(scaled_loss).backward()
            loss_value = float(output.loss.detach())
            probability = float(torch.sigmoid(output.anomaly_logits[0]).detach())
            source = str(example["source_type"])
            losses.append(loss_value)
            probabilities.append(probability)
            source_losses[source].append(loss_value)
            source_probabilities[source].append(probability)
            del output, inputs, labels, scaled_loss

        if not first_gradient_gate_done:
            if any(parameter.grad is not None for parameter in model.base_model.parameters()):
                raise RuntimeError("a frozen base parameter received a gradient")
            if any(parameter.grad is not None for parameter in model.decoder.mask_head.parameters()):
                raise RuntimeError("classification loss reached the frozen mask head")
            if any(parameter.grad is not None for parameter in model.decoder.region_head.parameters()):
                raise RuntimeError("classification loss reached the frozen region head")
            first_gradient_gate_done = True
        scaler.unscale_(optimizer)
        trainable = decoder_parameters + prompt_parameters
        grad_norm = float(clip_grad_norm_(trainable, float(training["gradient_clip_norm"])))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < scale_before:
            raise RuntimeError(f"GradScaler skipped update at step {step}")
        scheduler.step()

        record = {
            "step": step,
            "loss": float(np.mean(losses)),
            "classification_loss": float(np.mean(losses)),
            "p_yes_mean": float(np.mean(probabilities)),
            "p_yes_std": float(np.std(probabilities)),
            "grad_norm": grad_norm,
            "grad_scale": float(scaler.get_scale()),
            "lr": optimizer.param_groups[0]["lr"],
            "step_s": time.perf_counter() - step_started,
            "elapsed_s": time.perf_counter() - run_started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        for source in sorted(expected_sources):
            record[f"source_{source}_loss_mean"] = float(np.mean(source_losses[source]))
            record[f"source_{source}_p_yes_mean"] = float(np.mean(source_probabilities[source]))
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if step % 10 == 0 or step == 1:
            print(
                f"[classification train] {step}/{max_steps} loss={record['loss']:.4f} "
                f"p_yes={record['p_yes_mean']:.4f} grad={grad_norm:.3f} "
                f"step={record['step_s']:.2f}s",
                flush=True,
            )
        if step % 50 == 0:
            save_checkpoint(
                args.out_dir / "latest.pt", model, optimizer, scheduler, scaler,
                step, best_macro_aupr, best_step, protocol,
            )

        if step in eval_steps:
            tag = f"step{step:04d}"
            metrics = evaluate_classification(
                model, processor, dev, args.dev_cache_root, device, args.out_dir, tag
            )
            macro_aupr = float(metrics["macro_aupr"])
            print(
                f"[classification dev] step={step} macro_aupr={macro_aupr:.4f} "
                f"macro_f1={metrics['macro_f1max']:.4f} "
                f"small_recall={metrics['small_defect_recall']:.4f} "
                f"score_std={metrics['score_diagnostics']['sample_std']:.6f}",
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
                args.out_dir / f"{tag}.pt", model, optimizer, scheduler, scaler,
                step, best_macro_aupr, best_step, protocol,
            )
            save_checkpoint(
                args.out_dir / "latest.pt", model, optimizer, scheduler, scaler,
                step, best_macro_aupr, best_step, protocol,
            )
        if step >= run_until_step:
            break

    final = {
        "classification_only": True,
        "completed_steps": run_until_step,
        "best_step": best_step,
        "best_macro_aupr": best_macro_aupr,
        "trainable_parameters": summary,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_s": time.perf_counter() - run_started,
        "holdout_intersection": [],
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
