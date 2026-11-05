"""Auditable microbatch-one Qwen3-VL QLoRA SFT, without Trainer loss shortcuts.

Run: python -m industrial_vlm.train --model /path/to/model --train-file ...
--image-root ... --output ... --max-steps 32. This starts a new adapter run;
loading --adapter initializes from adapter weights, not optimizer-state resume.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from .modeling import (
    audit_trainable_parameters, load_model, make_inputs, move_inputs, set_training_mode,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _emit(path: Path, value: dict[str, Any]) -> None:
    value = {"time_unix": time.time(), **value}
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", type=Path)
    config_arg, _ = early.parse_known_args()
    defaults = json.loads(config_arg.config.read_text()) if config_arg.config else {}
    if not isinstance(defaults, dict):
        raise ValueError("Config must be a JSON object with argparse destination names")
    parser = argparse.ArgumentParser(description=__doc__, parents=[early])
    parser.add_argument("--model")
    parser.add_argument("--adapter", help="Optional adapter initialization; not optimizer-state resume")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=32, help="Optimizer updates, each containing accumulation microsteps")
    parser.add_argument("--smoke", action="store_true", help="Exactly one optimizer update, then save and reload-check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-order", choices=["shuffled", "sequential"], default="shuffled",
                        help="Sequential consumes a preregistered exposure schedule without reshuffling")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, choices=[1], default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--val-max-samples", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--skip-reload-check", action="store_true")
    allowed = {action.dest for action in parser._actions}
    unknown = set(defaults) - allowed
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for name in ("model", "train_file", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required (CLI or config)")
    for name in ("train_file", "val_file", "image_root", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    for name in ("max_steps", "gradient_accumulation_steps", "max_pixels", "max_length", "lora_r"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        parser.error("learning_rate/max_grad_norm must be positive and weight_decay nonnegative")
    if args.micro_batch_size != 1:
        parser.error("Only micro_batch_size=1 is implemented")
    if args.smoke:
        args.max_steps = 1
    return args


def _read_rows(path: Path, image_root: Path) -> list[dict[str, Any]]:
    rows, ids = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row["is_anomalous"], bool):
                    raise ValueError("is_anomalous must be boolean")
                if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                    raise ValueError("prompt must be a nonempty string")
                if not isinstance(row["answer"], str):
                    raise ValueError("answer must be a JSON string")
                target = json.loads(row["answer"])
                if not isinstance(target, dict) or not isinstance(target.get("is_anomalous"), bool):
                    raise ValueError("answer must contain boolean is_anomalous")
                if target["is_anomalous"] != row["is_anomalous"]:
                    raise ValueError("answer and metadata disagree")
                boxes = row["boxes"]
                if not isinstance(boxes, list):
                    raise ValueError("boxes must be a list")
                for box in boxes:
                    if (not isinstance(box, list) or len(box) != 4
                            or any(type(coordinate) is not int or not 0 <= coordinate <= 1000 for coordinate in box)
                            or box[0] >= box[2] or box[1] >= box[3]):
                        raise ValueError(f"Invalid normalized0-1000 integer box: {box!r}")
                if not row["is_anomalous"] and boxes:
                    raise ValueError("Normal sample cannot have anomaly boxes")
                if "boxes" in target and target["boxes"] != boxes:
                    raise ValueError("answer boxes and metadata disagree")
                if set(target) - {"is_anomalous", "boxes"}:
                    raise ValueError("First-version target permits only is_anomalous and boxes")
                image = Path(row["image"]).expanduser()
                image = (image_root / image if not image.is_absolute() else image).resolve()
                if not image.is_file():
                    raise ValueError(f"Image not found: {image}")
                if str(row["id"]) in ids:
                    raise ValueError(f"Duplicate sample id: {row['id']}")
                ids.add(str(row["id"]))
                rows.append({**row, "_image_path": str(image)})
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No samples in {path}")
    return rows


def _encode(processor: Any, row: dict[str, Any], args: argparse.Namespace, *, target: bool = True) -> dict:
    try:
        return make_inputs(processor, row["_image_path"], row["prompt"],
                           row["answer"] if target else None, max_length=args.max_length)
    except Exception as exc:
        raise RuntimeError(f"Encoding failed for sample {row['id']}: {exc}") from exc


def _save(model: Any, processor: Any, directory: Path, step: int, args: argparse.Namespace) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    processor.save_pretrained(directory)
    _write_json(directory / "training_state.json", {
        "optimizer_step": step,
        "base_model": args.model,
        "seed": args.seed,
        "note": "Adapter-only checkpoint. Optimizer/scheduler state is not saved; not an exact resume checkpoint.",
        "coordinate_format": "xyxy integer normalized0-1000",
    })


def _validate(model: Any, processor: Any, rows: list[dict], args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    loss_sum, tokens = 0.0, 0
    selected = rows[:args.val_max_samples] if args.val_max_samples > 0 else rows
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in selected:
            batch = move_inputs(_encode(processor, row, args), model)
            n_tokens = int((batch["labels"][:, 1:] != -100).sum())
            output = model(**batch, use_cache=False)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Nonfinite validation loss at sample {row['id']}")
            loss_sum += float(output.loss) * n_tokens
            tokens += n_tokens
    set_training_mode(model)
    return {"validation_loss": loss_sum / tokens, "validation_target_tokens": tokens,
            "validation_samples": len(selected), "metric_type": "teacher_forced_assistant_CE"}


def _last_logits(model: Any, inputs: dict) -> torch.Tensor:
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Native Qwen3-VL supports logits_to_keep; avoid allocating all prompt logits.
        output = model(**move_inputs(inputs, model), use_cache=False, logits_to_keep=1)
        return output.logits[0, -1].float().cpu()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output must be empty to avoid mixing runs: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "metrics.jsonl"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_rows = _read_rows(args.train_file, args.image_root)
    val_rows = _read_rows(args.val_file, args.image_root) if args.val_file else []
    if val_rows:
        train_images = {row["_image_path"] for row in train_rows}
        overlap = train_images & {row["_image_path"] for row in val_rows}
        if overlap:
            raise ValueError(f"Training/validation share {len(overlap)} resolved image paths")
    serial_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    _write_json(args.output / "run_config.json", {
        **serial_args, "versions": versions, "train_sha256": _sha256(args.train_file),
        "val_sha256": _sha256(args.val_file) if args.val_file else None,
        "coordinate_format": "xyxy integer normalized0-1000",
        "loss": "token-weighted assistant-only causal cross entropy including assistant end marker",
        "world_size": 1, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one process on one GPU; distributed training is not implemented")
    start_load = time.perf_counter()
    model, processor = load_model(
        args.model, adapter=args.adapter, training=True, max_pixels=args.max_pixels,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    audit = {**audit_trainable_parameters(model), **model._industrial_vlm_audit}
    initial_digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            initial_digest.update(name.encode())
            initial_digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    audit["initial_trainable_sha256"] = initial_digest.hexdigest()
    _write_json(args.output / "parameter_audit.json", audit)
    _emit(log_path, {"event": "loaded", "load_seconds": time.perf_counter() - start_load,
                     "train_samples": len(train_rows), "val_samples": len(val_rows),
                     "trainable_parameters": audit["trainable_parameters"],
                     "gpu": torch.cuda.get_device_name(),
                     "allocated_gib": torch.cuda.memory_allocated() / 2**30})
    first = _encode(processor, train_rows[0], args)
    labels = first["labels"]
    _write_json(args.output / "first_sample_audit.json", {
        "id": train_rows[0]["id"], "input_tokens": int(first["input_ids"].shape[-1]),
        "target_tokens": int((labels[:, 1:] != -100).sum()),
        "masked_prefix_tokens": int((labels[0] == -100).sum()),
        "native_fields": {key: list(value.shape) for key, value in first.items() if isinstance(value, torch.Tensor)},
        "supervised_text": processor.tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=False),
    })
    del first, labels
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False)

    def multiplier(index: int) -> float:
        if args.warmup_steps and index < args.warmup_steps:
            return (index + 1) / args.warmup_steps
        return max(0.0, (args.max_steps - index) / max(1, args.max_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    rng = random.Random(args.seed)
    order: list[int] = []
    cursor, epoch = 0, -1
    total_targets, total_inputs, total_images = 0, 0, 0
    last_loss = None
    exposure_path = args.output / "exposures.jsonl"
    source_digest = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, target_tokens, input_tokens = 0.0, 0, 0
        step_exposures = []
        for microstep in range(args.gradient_accumulation_steps):
            if cursor >= len(order):
                order = list(range(len(train_rows)))
                if args.sample_order == "shuffled":
                    rng.shuffle(order)
                cursor, epoch = 0, epoch + 1
            row = train_rows[order[cursor]]
            cursor += 1
            batch = move_inputs(_encode(processor, row, args), model)
            n_targets = int((batch["labels"][:, 1:] != -100).sum())
            n_inputs = int(batch["attention_mask"].sum())
            source_id = str(row.get("source_id", row["image"]))
            step_exposures.append({"optimizer_step": step, "microstep": microstep,
                                   "id": row["id"], "source_id": source_id,
                                   "task": "region" if "boxes" in json.loads(row["answer"]) else "decision",
                                   "target_tokens": n_targets, "input_tokens": n_inputs})
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at optimizer step {step}, sample {row['id']}")
            # Accumulate SUM token losses, then normalize gradients once by
            # the actual supervised-token count. Unequal answer lengths do
            # not silently receive equal per-example weight.
            (loss * n_targets).backward()
            loss_sum += float(loss.detach()) * n_targets
            target_tokens += n_targets
            input_tokens += n_inputs
            del output, loss, batch
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(target_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        if step == 1:
            nonzero = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is not None and bool(parameter.grad.abs().max() > 0)]
            if not nonzero:
                raise RuntimeError("No nonzero LoRA gradients on the first optimizer step")
            _emit(log_path, {"event": "gradient_audit", "nonzero_gradient_tensors": nonzero})
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        with exposure_path.open("a", encoding="utf-8") as exposure_log:
            for exposure in step_exposures:
                exposure_log.write(json.dumps(exposure, ensure_ascii=False) + "\n")
                source_digest.update((exposure["source_id"] + "\n").encode())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        total_targets += target_tokens
        total_inputs += input_tokens
        total_images += args.gradient_accumulation_steps
        last_loss = loss_sum / target_tokens
        _emit(log_path, {
            "event": "train", "optimizer_step": step, "epoch": epoch, "loss": last_loss,
            "learning_rate": learning_rate, "grad_norm_before_clip": float(grad_norm),
            "target_tokens": target_tokens, "input_tokens": input_tokens,
            "microsteps": args.gradient_accumulation_steps, "seconds": elapsed,
            "input_tokens_per_second": input_tokens / elapsed,
            "target_tokens_per_second": target_tokens / elapsed,
            "images_per_second": args.gradient_accumulation_steps / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        if val_rows and args.eval_steps > 0 and (step % args.eval_steps == 0 or step == args.max_steps):
            _emit(log_path, {"event": "validation", "optimizer_step": step,
                             **_validate(model, processor, val_rows, args)})
        if args.save_steps > 0 and step % args.save_steps == 0 and step != args.max_steps:
            _save(model, processor, args.output / f"checkpoint-{step}", step, args)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    final_dir = args.output / "adapter"
    _save(model, processor, final_dir, args.max_steps, args)
    summary = {
        "status": "training_complete", "optimizer_steps": args.max_steps,
        "microsteps": total_images, "last_train_loss": last_loss,
        "total_target_tokens": total_targets, "total_input_tokens": total_inputs,
        "source_exposure_sha256": source_digest.hexdigest(),
        "exposure_trace_sha256": _sha256(exposure_path),
        "initial_trainable_sha256": audit["initial_trainable_sha256"],
        "train_wall_seconds_including_validation": train_seconds,
        "trainable_parameters": audit["trainable_parameters"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "adapter": str(final_dir), "reload_check": "skipped" if args.skip_reload_check else "pending",
    }
    _write_json(args.output / "summary.json", summary)
    if not args.skip_reload_check:
        check_inputs = _encode(processor, train_rows[0], args, target=False)
        before = _last_logits(model, check_inputs)
        # Release optimizer and base before loading another base on a 24GB GPU.
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, parameters, model, parameter
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, _ = load_model(args.model, adapter=final_dir, training=False, max_pixels=args.max_pixels)
        after = _last_logits(reloaded, check_inputs)
        difference = float((before - after).abs().max())
        passed = bool(torch.allclose(before, after, rtol=1e-3, atol=1e-3))
        summary["reload_check"] = {"passed": passed, "max_absolute_logit_difference": difference,
                                    "same_top1": int(before.argmax()) == int(after.argmax()),
                                    "rtol": 1e-3, "atol": 1e-3,
                                    "scope": "first-sample prompt final-token logits; fresh base + saved adapter"}
        summary["status"] = "complete" if passed else "reload_check_failed"
        _write_json(args.output / "summary.json", summary)
        _emit(log_path, {"event": "reload_check", **summary["reload_check"]})
        if not passed:
            raise RuntimeError("Saved adapter reload did not reproduce prompt logits within tolerance")
    _emit(log_path, {"event": "complete", **summary})


if __name__ == "__main__":
    main()
