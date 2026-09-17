"""Train the frozen-Qwen segmentation decoder with full images and positive 3x3 patches.

This is an isolated experiment entry point.  It deliberately reuses QwenE1's
existing segmentation_only method, so the classifier, frozen vision backbone,
decoder and Focal + Dice loss stay unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from b0.single_view import candidate_ids
from e1.model import QwenE1
from e1.patch9 import crop_resize, crop_resize_mask, grid_boxes, positive_boxes, read_mask, read_rgb

SIDE = 448
EVAL_STEPS = (250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def safe_dataset_path(root: Path, value: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"dataset path must be relative: {value}")
    return root / relative


def load_original(row, root: Path, gt_root: Path) -> tuple[np.ndarray, np.ndarray]:
    image = read_rgb(safe_dataset_path(root, row.image_path))
    if int(getattr(row, "view_label", 1)) == 0 or pd.isna(row.mask_path):
        mask = np.zeros(image.shape[:2], dtype=bool)
    else:
        mask = read_mask(safe_dataset_path(gt_root, row.mask_path))
    if image.shape[:2] != mask.shape:
        raise ValueError(f"invalid image/mask pair: {row.image_path}")
    return image, mask


def encode(processor, image: np.ndarray) -> dict[str, torch.Tensor]:
    batch = processor.image_processor(images=Image.fromarray(image), return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in batch.items()}
    if inputs["image_grid_thw"].tolist() != [[1, 28, 28]]:
        raise ValueError(f"unexpected Qwen3 image grid: {inputs['image_grid_thw']}")
    return inputs


def target_from(mask: np.ndarray) -> torch.Tensor:
    value = np.asarray(mask, dtype=np.float32)
    if value.shape != (SIDE, SIDE):
        raise ValueError(f"expected 448x448 target, got {value.shape}")
    return torch.from_numpy(value[None, None]).cuda()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def check_gradients(model):
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not name.startswith("decoder.") or not parameter.requires_grad:
                raise RuntimeError(f"unexpected gradient: {name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"nonfinite gradient: {name}")
            names.append(name)
        elif parameter.requires_grad:
            raise RuntimeError(f"disconnected trainable parameter: {name}")
    if not names:
        raise RuntimeError("no decoder gradients")
    return names


def build_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    layers = tuple(base.config.vision_config.deepstack_visual_indexes)
    if layers != (5, 11, 17):
        raise ValueError(f"unexpected Qwen3 feature layers: {layers}")
    model = QwenE1(
        base, candidate_ids(processor.tokenizer, "Yes"), candidate_ids(processor.tokenizer, "No"),
        feature_layers=layers, decoder_channels=288,
    ).cuda()
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.decoder.region_head.requires_grad_(False)
    return model, processor, layers


def audit(train: pd.DataFrame, dev: pd.DataFrame, root: Path, gt_root: Path) -> dict:
    report = {}
    for name, frame in (("train", train), ("dev", dev)):
        if "view_label" not in frame or frame.image_path.duplicated().any():
            raise ValueError(f"{name} has invalid image rows")
        positives = []
        for row in frame.itertuples(index=False):
            image, mask = load_original(row, root, gt_root)
            boxes = grid_boxes(image.shape[1], image.shape[0])
            indices = positive_boxes(mask, boxes)
            positives.append(len(indices))
        report[name] = {
            "rows": len(frame),
            "classes": int(frame.class_name.nunique()),
            "positive_patch_min": min(positives),
            "positive_patch_max": max(positives),
            "positive_patch_mean": float(np.mean(positives)), "empty_mask_rows": int((frame.view_label == 0).sum()),
        }
    if set(train.class_name) != set(dev.class_name):
        raise ValueError("train/dev class sets differ")
    return report


def prepare_examples(row, root: Path, gt_root: Path, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, str]]:
    image, mask = load_original(row, root, gt_root)
    full_image = np.asarray(Image.fromarray(image).resize((SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.uint8)
    full_mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIDE, SIDE), Image.Resampling.NEAREST), dtype=bool)
    if not mask.any():
        return [(full_image, full_mask, "full_empty")]
    boxes = grid_boxes(image.shape[1], image.shape[0])
    positive = positive_boxes(mask, boxes)
    box_index = int(rng.choice(positive))
    box = boxes[box_index]
    # Two entries per row keep the logical batch half full-image and half patch.
    patch_image = crop_resize(image, box, SIDE)
    patch_mask = crop_resize_mask(mask, box, SIDE)
    # A very tiny defect can legitimately disappear when the *full* original
    # image is reduced to 448px.  Do not invent a positive pixel: keep the
    # high-resolution positive patch and omit only this invalid full-image
    # example.  The existing segmentation_only loss intentionally accepts
    # non-empty masks only.
    examples = [(patch_image, patch_mask, f"patch{box_index}")]
    if full_mask.any():
        examples.insert(0, (full_image, full_mask, "full"))
    return examples


def validate(model, processor, row, root: Path, gt_root: Path, out: Path) -> dict:
    image, mask = load_original(row, root, gt_root)
    boxes = grid_boxes(image.shape[1], image.shape[0])
    patch_index = positive_boxes(mask, boxes)[0]
    examples = prepare_examples(row, root, gt_root, np.random.default_rng(0))
    model.train(); model.zero_grad(set_to_none=True)
    losses = []
    for image_448, mask_448, _ in examples:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.segmentation_only(encode(processor, image_448), target_from(mask_448), allow_empty=True)
        if not torch.isfinite(result.loss) or tuple(result.mask_logits.shape) != (1, 1, SIDE, SIDE):
            raise RuntimeError("invalid patch forward output")
        (result.loss / len(examples)).backward()
        losses.append(float(result.loss.detach()))
    names = check_gradients(model)
    norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("zero or nonfinite decoder gradient")
    report = {
        "passed": True, "examples": [tag for _, _, tag in examples], "first_patch_index": patch_index,
        "losses": losses, "gradient_norm": norm, "gradient_parameters": names,
        "trainable_parameters": sum(p.numel() for p in trainable(model)),
        "base_trainable": any(p.requires_grad for p in model.base_model.parameters()),
        "loss_protocol": "existing Focal(gamma=2, positive_alpha=0.25) + Soft Dice(nonempty)",
    }
    model.zero_grad(set_to_none=True)
    write_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/segmentation_qwen3_patch9_12000"))
    parser.add_argument("--steps", type=int, default=12000, choices=[1000, 3000, 12000])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    train = pd.read_parquet(args.train).reset_index(drop=True)
    dev = pd.read_parquet(args.dev).reset_index(drop=True)
    report = audit(train, dev, args.root, args.gt_root)
    write_json(args.out / "data_validation.json", report)
    model, processor, layers = build_model(args.model_id)
    validation_row = train[train.view_label.eq(1)].iloc[0]
    forward = validate(model, processor, validation_row, args.root, args.gt_root, args.out / "forward_validation.json")
    print(json.dumps({"data": report, "forward": forward}, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return
    if (args.out / "training_history.csv").exists() and not args.resume:
        raise FileExistsError("an existing patch9 experiment must not be overwritten")
    seed = 20260917
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    protocol = {
        "experiment": "segmentation_qwen3_patch9_12000", "model": args.model_id,
        "feature_layers": list(layers), "decoder_channels": 288, "seed": seed,
        "steps": args.steps, "evaluation_steps": list(EVAL_STEPS),
        "input": "anomaly: full image + positive 3x3 patch; normal: full image + empty mask; balanced 4/4 examples",
        "patch_window_fraction": 0.4, "patch_stride_fraction": 0.3,
        "loss": "existing focal(gamma=2,positive_alpha=0.25)+soft_Dice; empty masks allowed",
        "classification": "unchanged; classifier checkpoint is not touched",
        "trainable": "decoder only; Qwen vision/language base frozen",
        "train_sha256": digest(args.train), "dev_sha256": digest(args.dev),
    }
    write_json(args.out / "frozen_training_protocol.json", protocol)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/50 if s < 50 else .5*(1+math.cos(math.pi*(s-50)/max(1,args.steps-50))))
    fields = ["step", "loss", "focal_loss", "dice_loss", "grad_norm", "lr", "step_s", "elapsed_s", "input_mix"]
    started = time.perf_counter(); accumulation = 8
    best = -1.0
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        print(f"[patch9 resume] restored step {start_step}", flush=True)
    history_path = args.out / "training_history.csv"
    history_mode = "a" if args.resume and history_path.exists() else "x"
    with history_path.open(history_mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if history_mode == "x":
            writer.writeheader()
        for step in range(start_step + 1, args.steps + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); begin = time.perf_counter(); values = np.zeros(3); mix = []
            anomaly_rows = train[train.view_label.eq(1)]
            normal_rows = train[train.view_label.eq(0)]
            # Two anomaly rows yield four examples (full + positive patch);
            # four normal rows yield four full-image empty-mask examples.
            for _ in range(accumulation // 4):
                row = anomaly_rows.iloc[int(rng.integers(len(anomaly_rows)))]
                for image_448, mask_448, tag in prepare_examples(row, args.root, args.gt_root, rng):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model.segmentation_only(encode(processor, image_448), target_from(mask_448), allow_empty=True)
                    (result.loss / accumulation).backward(); values += [float(result.loss.detach()), float(result.focal_loss.detach()), float(result.dice_loss.detach())]; mix.append(tag)
            for _ in range(accumulation // 2):
                row = normal_rows.iloc[int(rng.integers(len(normal_rows)))]
                for image_448, mask_448, tag in prepare_examples(row, args.root, args.gt_root, rng):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model.segmentation_only(encode(processor, image_448), target_from(mask_448), allow_empty=True)
                    (result.loss / accumulation).backward(); values += [float(result.loss.detach()), float(result.focal_loss.detach()), float(result.dice_loss.detach())]; mix.append(tag)
            check_gradients(model); norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)); optimizer.step(); scheduler.step(); values /= accumulation
            writer.writerow(dict(zip(fields, [step, *values, norm, optimizer.param_groups[0]["lr"], time.perf_counter()-begin, time.perf_counter()-started, "full=" + str(sum(x == "full" for x in mix)) + ",patch=" + str(sum(x.startswith("patch") for x in mix))]))); stream.flush()
            if step % 50 == 0:
                state = {"step": step, "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "protocol": protocol}
                torch.save(state, args.out / "latest.pt")
            if step in EVAL_STEPS:
                print(f"[patch9 train] reached evaluation node {step}/{args.steps}; checkpoint saved", flush=True)
                if step == args.steps:
                    torch.save(state, args.out / f"step{step:05d}.pt")
        write_json(args.out / "training_summary.json", {"completed_steps": args.steps, "best_macro_dice": best, "elapsed_s": time.perf_counter()-started})


if __name__ == "__main__":
    main()
