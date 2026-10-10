"""Patch9 v5: v4 recipe + small-defect-oriented loss.

What changes vs v4 (everything else is byte-identical)
------------------------------------------------------
v4  = 24000 steps, 6 anomaly rows + 2 normal rows + 2 defect-free patches
      (accumulation 16), Focal(gamma=2, pos_alpha=0.25) + Soft Dice.
v5  = the same data pipeline and schedule, but the loss is re-weighted so a
      *small-area defect* is no longer a cheap example to get wrong:

  1. **Sample-level area weight** (the core change).
     With plain Focal+Dice a tiny defect contributes ~1e-4 of the pixels, so
     its gradient is two orders of magnitude smaller than a large defect's.
     v5 multiplies the whole example loss by a bounded bucket weight derived
     from the *row-level* (original-image) defect area ratio:

         ratio < 0.005        -> 2.0      (tiny)
         0.005 <= ratio < 0.02 -> 1.5     (small)
         ratio >= 0.02        -> 1.0      (large)
         empty crop           -> 1.0      (normals and defect-free patches
                                           are NEVER up-weighted)

     Two deliberate details:
       * the weight comes from the row (full-image) ratio, not from the
         448x448 crop, so a magnified patch keeps the *same* weight as its
         full image instead of being punished for the zoom;
       * any crop with an empty GT mask is pinned to 1.0, otherwise the
         "stay silent" signal would be amplified together with the defect
         signal and the false-positive suppression learned since v2 would be
         undone.

  2. **Focal Tversky instead of plain Soft Dice** (mild FN emphasis).
     Tversky = TP / (TP + alpha*FP + beta*FN); alpha=beta=0.5 is Dice.
     v5 uses alpha=0.4 / beta=0.6, i.e. a *slight* bias towards penalising
     missed defect pixels, which is exactly the failure mode of tiny
     defects. It stays close to Dice on purpose: the leaderboard's pixel AP
     is precision-sensitive, so an aggressive beta=0.7 would trade away the
     false-positive suppression that produced v2 -> v4's gains.

     loss = w_area * (Focal + Tversky)

Both knobs are switchable (``--no-area-weight``, ``--tversky-alpha 0.5
--tversky-beta 0.5`` restores plain Dice) so the two effects can be
attributed later without editing code.
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
from e1.dataset_config import BASE_DEV, BASE_TRAIN, assert_original_paths, assert_original_split

SIDE = 448
EVAL_STEPS = (
    250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000,
    18000, 20000, 22000, 24000,
)
# Nodes at which a standalone step<NNNNN>.pt checkpoint is written.
CHECKPOINT_NODES = (8000, 12000, 16000, 20000, 24000)

# Small-defect area-weight buckets, applied to the row-level defect ratio.
TINY_RATIO = 0.005
SMALL_RATIO = 0.02
W_TINY, W_SMALL, W_LARGE = 2.0, 1.5, 1.0


def area_weight(ratio: float) -> float:
    """Bounded sample weight; empty targets are always 1.0."""
    if ratio <= 0:
        return W_LARGE
    if ratio < TINY_RATIO:
        return W_TINY
    if ratio < SMALL_RATIO:
        return W_SMALL
    return W_LARGE


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """1 - Tversky, averaged over the batch (alpha=beta=0.5 reproduces Dice)."""
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    flat_p = probabilities.flatten(1)
    flat_t = targets.flatten(1)
    tp = (flat_p * flat_t).sum(1)
    fp = (flat_p * (1 - flat_t)).sum(1)
    fn = ((1 - flat_p) * flat_t).sum(1)
    per_example = 1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)
    return per_example.mean()


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
        ratios = []
        all_positive_rows = 0
        for row in frame.itertuples(index=False):
            image, mask = load_original(row, root, gt_root)
            ratio = float(mask.mean())
            if int(row.view_label) == 1:
                ratios.append(ratio)
            boxes = grid_boxes(image.shape[1], image.shape[0])
            indices = positive_boxes(mask, boxes)
            positives.append(len(indices))
            if len(indices) == len(boxes) and int(row.view_label) == 1:
                all_positive_rows += 1
        array = np.asarray(ratios, dtype=float)
        report[name] = {
            "rows": len(frame),
            "classes": int(frame.class_name.nunique()),
            "positive_patch_min": min(positives),
            "positive_patch_max": max(positives),
            "positive_patch_mean": float(np.mean(positives)),
            "empty_mask_rows": int((frame.view_label == 0).sum()),
            "all_positive_patch_anomaly_rows": all_positive_rows,
            "anomaly_area_ratio_mean": float(array.mean()) if array.size else 0.0,
            "anomaly_area_ratio_median": float(np.median(array)) if array.size else 0.0,
            "anomaly_area_ratio_min": float(array.min()) if array.size else 0.0,
            "weight_bucket_tiny_frac": float((array < TINY_RATIO).mean()) if array.size else 0.0,
            "weight_bucket_small_frac": float(((array >= TINY_RATIO) & (array < SMALL_RATIO)).mean()) if array.size else 0.0,
            "weight_bucket_large_frac": float((array >= SMALL_RATIO).mean()) if array.size else 0.0,
        }
    if set(train.class_name) != set(dev.class_name):
        raise ValueError("train/dev class sets differ")
    return report


def prepare_examples(row, root: Path, gt_root: Path, rng: np.random.Generator) -> tuple[list[tuple[np.ndarray, np.ndarray, str]], float]:
    """Return the crops for one row plus the row-level (full-image) defect ratio."""
    image, mask = load_original(row, root, gt_root)
    ratio = float(mask.mean())
    full_image = np.asarray(Image.fromarray(image).resize((SIDE, SIDE), Image.Resampling.BILINEAR), dtype=np.uint8)
    full_mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((SIDE, SIDE), Image.Resampling.NEAREST), dtype=bool)
    if not mask.any():
        return [(full_image, full_mask, "full_empty")], ratio
    boxes = grid_boxes(image.shape[1], image.shape[0])
    positive = positive_boxes(mask, boxes)
    box_index = int(rng.choice(positive))
    box = boxes[box_index]
    patch_image = crop_resize(image, box, SIDE)
    patch_mask = crop_resize_mask(mask, box, SIDE)
    examples = [(patch_image, patch_mask, f"patch{box_index}")]
    if full_mask.any():
        examples.insert(0, (full_image, full_mask, "full"))
    return examples, ratio


def prepare_negative_patch(row, root: Path, gt_root: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    """One defect-free patch with an empty mask (inference-distribution alignment)."""
    image, mask = load_original(row, root, gt_root)
    boxes = grid_boxes(image.shape[1], image.shape[0])
    positive = set(positive_boxes(mask, boxes))
    negative = [index for index in range(len(boxes)) if index not in positive]
    if not negative:
        raise LookupError(f"row has no defect-free patch: {row.image_path}")
    index = int(rng.choice(negative))
    box = boxes[index]
    patch_image = crop_resize(image, box, SIDE)
    patch_mask = np.zeros((SIDE, SIDE), dtype=bool)
    return patch_image, patch_mask, f"patch_empty{index}"


def validate(model, processor, train: pd.DataFrame, root: Path, gt_root: Path, out: Path,
             use_area_weight: bool, tversky_alpha: float, tversky_beta: float) -> dict:
    anomaly_row = train[train.view_label.eq(1)].iloc[0]
    rng = np.random.default_rng(0)
    examples, ratio = prepare_examples(anomaly_row, root, gt_root, rng)
    negative_ok = True
    try:
        neg_image, neg_mask, neg_tag = prepare_negative_patch(anomaly_row, root, gt_root, rng)
        examples.append((neg_image, neg_mask, neg_tag))
    except LookupError:
        negative_ok = False
        for _, row in train[train.view_label.eq(1)].iloc[1:200].iterrows():
            try:
                neg_image, neg_mask, neg_tag = prepare_negative_patch(row, root, gt_root, rng)
                examples.append((neg_image, neg_mask, neg_tag))
                negative_ok = True
                break
            except LookupError:
                continue
    if not negative_ok:
        raise RuntimeError("could not sample any negative patch from the first 200 anomaly rows")
    model.train(); model.zero_grad(set_to_none=True)
    losses = []
    weights = []
    for image_448, mask_448, _ in examples:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.segmentation_only(encode(processor, image_448), target_from(mask_448), allow_empty=True)
        if not torch.isfinite(result.loss) or tuple(result.mask_logits.shape) != (1, 1, SIDE, SIDE):
            raise RuntimeError("invalid patch forward output")
        empty = not bool(np.asarray(mask_448).any())
        weight = 1.0 if (not use_area_weight or empty) else area_weight(ratio)
        tv = tversky_loss(result.mask_logits, target_from(mask_448), tversky_alpha, tversky_beta)
        total = weight * (result.focal_loss + tv)
        if not torch.isfinite(total):
            raise RuntimeError("nonfinite v5 loss")
        (total / len(examples)).backward()
        losses.append(float(total.detach()))
        weights.append(weight)
    names = check_gradients(model)
    norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("zero or nonfinite decoder gradient")
    report = {
        "passed": True, "examples": [tag for _, _, tag in examples],
        "losses": losses, "sample_weights": weights, "row_area_ratio": ratio,
        "gradient_norm": norm, "gradient_parameters": names,
        "trainable_parameters": sum(p.numel() for p in trainable(model)),
        "base_trainable": any(p.requires_grad for p in model.base_model.parameters()),
        "loss_protocol": (
            f"w_area x (Focal(gamma=2, positive_alpha=0.25) + Tversky(alpha={tversky_alpha}, beta={tversky_beta}))"
            if use_area_weight else
            f"Focal(gamma=2, positive_alpha=0.25) + Tversky(alpha={tversky_alpha}, beta={tversky_beta})"
        ),
    }
    model.zero_grad(set_to_none=True)
    write_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train", type=Path, default=BASE_TRAIN)
    parser.add_argument("--dev", type=Path, default=BASE_DEV)
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/segmentation_qwen3_patch9_mixed_v5_24000"))
    parser.add_argument("--steps", type=int, default=24000, choices=[1000, 3000, 12000, 24000])
    parser.add_argument("--anomaly-rows", type=int, default=6)
    parser.add_argument("--normal-rows", type=int, default=2)
    parser.add_argument("--empty-patch-rows", type=int, default=2,
                        help="defect-free patch examples per step (RC-1); 2 matches the v4 run that scored best")
    parser.add_argument("--area-weight", dest="area_weight", action="store_true", default=True)
    parser.add_argument("--no-area-weight", dest="area_weight", action="store_false")
    parser.add_argument("--tversky-alpha", type=float, default=0.4,
                        help="FP coefficient; 0.5 together with beta=0.5 restores plain Soft Dice")
    parser.add_argument("--tversky-beta", type=float, default=0.6,
                        help="FN coefficient; larger = stronger penalty for missed defect pixels")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hard-normal-pool", type=Path,
                        help="optional CSV of verified normal training views; 25%% of normal-row draws come from this pool")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if abs(args.tversky_alpha + args.tversky_beta - 1.0) > 1e-6:
        raise SystemExit("tversky alpha + beta must equal 1.0")
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    train = pd.read_parquet(args.train).reset_index(drop=True)
    dev = pd.read_parquet(args.dev).reset_index(drop=True)
    assert_original_split(train, "train")
    assert_original_split(dev, "dev")
    assert_original_paths(train, "train")
    assert_original_paths(dev, "dev")
    report = audit(train, dev, args.root, args.gt_root)
    hard_paths = set()
    if args.hard_normal_pool:
        hard = pd.read_csv(args.hard_normal_pool)
        hard_paths = set(hard.image_path.astype(str))
        train_normals = set(train.loc[train.view_label.eq(0), "image_path"].astype(str))
        if not hard_paths or not hard_paths.issubset(train_normals):
            raise ValueError("hard-normal pool must be a non-empty subset of training normal views")
    write_json(args.out / "data_validation.json", report)
    model, processor, layers = build_model(args.model_id)
    forward = validate(model, processor, train, args.root, args.gt_root, args.out / "forward_validation.json",
                       args.area_weight, args.tversky_alpha, args.tversky_beta)
    print(json.dumps({"data": report, "forward": forward}, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return
    if (args.out / "training_history.csv").exists() and not args.resume:
        raise FileExistsError("an existing patch9 v5 experiment must not be overwritten")
    seed = 20260921
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    accumulation = args.anomaly_rows * 2 + args.normal_rows + args.empty_patch_rows
    protocol = {
        "experiment": "segmentation_qwen3_patch9_mixed_v5_24000", "model": args.model_id,
        "feature_layers": list(layers), "decoder_channels": 288, "seed": seed,
        "steps": args.steps, "evaluation_steps": list(EVAL_STEPS),
        "accumulation": accumulation,
        "input": (
            f"per step: {args.anomaly_rows} anomaly rows (full + positive patch each "
            f"= {args.anomaly_rows * 2} non-empty) + {args.normal_rows} normal rows (full empty mask)"
            + (f" + {args.empty_patch_rows} defect-free patch (empty mask)" if args.empty_patch_rows else "")
            + f"; accumulation={accumulation}"
        ),
        "change_vs_v4": (
            "identical data pipeline, schedule (24000 steps, lr 1e-4 cosine) and batch mix; "
            "only the loss changed. "
            + (f"sample-level area weight {W_TINY}/{W_SMALL}/{W_LARGE} for row defect ratio "
               f"<{TINY_RATIO}/<{SMALL_RATIO}/>={SMALL_RATIO} (empty crops pinned to 1.0) " if args.area_weight
               else "no area weighting ")
            + f"+ Focal Tversky(alpha={args.tversky_alpha}, beta={args.tversky_beta}) replacing Soft Dice. "
            "Goal: raise tiny/small-defect recall without giving up the false-positive "
            "suppression that pixel AP depends on."
        ),
        "patch_window_fraction": 0.4, "patch_stride_fraction": 0.3,
        "loss": (
            f"w_area x (focal(gamma=2,positive_alpha=0.25) + tversky(alpha={args.tversky_alpha},"
            f"beta={args.tversky_beta}))" if args.area_weight else
            f"focal(gamma=2,positive_alpha=0.25) + tversky(alpha={args.tversky_alpha},beta={args.tversky_beta})"
        ),
        "area_weight_buckets": {
            "tiny_ratio_lt": TINY_RATIO, "small_ratio_lt": SMALL_RATIO,
            "weights": {"tiny": W_TINY, "small": W_SMALL, "large": W_LARGE, "empty_crop": 1.0},
            "ratio_source": "row-level full-image defect ratio (patches inherit their row's weight)",
        },
        "classification": "unchanged; classifier checkpoint is not touched",
        "hard_normal_pool": str(args.hard_normal_pool) if args.hard_normal_pool else None,
        "hard_normal_fraction_of_normal_draws": 0.25 if args.hard_normal_pool else 0.0,
        "trainable": "decoder only; Qwen vision/language base frozen",
        "train_sha256": digest(args.train), "dev_sha256": digest(args.dev),
    }
    write_json(args.out / "frozen_training_protocol.json", protocol)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(trainable(model), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/50 if s < 50 else .5*(1+math.cos(math.pi*(s-50)/max(1,args.steps-50))))
    fields = ["step", "loss", "focal_loss", "tversky_loss", "area_w_mean", "small_frac",
              "grad_norm", "lr", "step_s", "elapsed_s", "input_mix"]
    started = time.perf_counter()
    best = -1.0
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        print(f"[patch9v5 resume] restored step {start_step}", flush=True)
    history_path = args.out / "training_history.csv"
    history_mode = "a" if args.resume and history_path.exists() else "x"
    anomaly_rows = train[train.view_label.eq(1)]
    normal_all = train[train.view_label.eq(0)]
    if hard_paths:
        normal_rows = normal_all[~normal_all.image_path.astype(str).isin(hard_paths)].reset_index(drop=True)
        hard_normal_rows = normal_all[normal_all.image_path.astype(str).isin(hard_paths)].reset_index(drop=True)
    else:
        normal_rows = normal_all.reset_index(drop=True)
        hard_normal_rows = normal_all.iloc[0:0].copy()

    def forward_example(image_448, mask_448, ratio: float):
        target = target_from(mask_448)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.segmentation_only(encode(processor, image_448), target, allow_empty=True)
        empty = not bool(np.asarray(mask_448).any())
        weight = 1.0 if (not args.area_weight or empty) else area_weight(ratio)
        tv = tversky_loss(result.mask_logits, target, args.tversky_alpha, args.tversky_beta)
        total = weight * (result.focal_loss + tv)
        (total / accumulation).backward()
        return [float(total.detach()), float(result.focal_loss.detach()),
                float(tv.detach()), weight]

    with history_path.open(history_mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if history_mode == "x":
            writer.writeheader()
        for step in range(start_step + 1, args.steps + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); begin = time.perf_counter(); values = np.zeros(4); mix = []
            ratios = []
            # Anomaly (foreground) supervision: each row contributes its full image
            # plus one randomly chosen defect-containing patch.
            for _ in range(args.anomaly_rows):
                row = anomaly_rows.iloc[int(rng.integers(len(anomaly_rows)))]
                examples, ratio = prepare_examples(row, args.root, args.gt_root, rng)
                for image_448, mask_448, tag in examples:
                    values[:4] += forward_example(image_448, mask_448, ratio); mix.append(tag); ratios.append(ratio)
            # Normal rows: full image with an empty mask (silence-on-normal signal).
            for _ in range(args.normal_rows):
                source = hard_normal_rows if hard_paths and rng.random() < 0.25 else normal_rows
                row = source.iloc[int(rng.integers(len(source)))]
                examples, ratio = prepare_examples(row, args.root, args.gt_root, rng)
                for image_448, mask_448, tag in examples:
                    values[:4] += forward_example(image_448, mask_448, ratio); mix.append(tag); ratios.append(ratio)
            # Defect-free patch supervision (RC-1); kept ON, matching the best v4 run.
            made = 0; attempts = 0
            while made < args.empty_patch_rows and attempts < 40:
                attempts += 1
                source = anomaly_rows if rng.random() < 0.5 else normal_rows
                row = source.iloc[int(rng.integers(len(source)))]
                try:
                    image_448, mask_448, tag = prepare_negative_patch(row, args.root, args.gt_root, rng)
                except LookupError:
                    continue
                values[:4] += forward_example(image_448, mask_448, 0.0); mix.append(tag); ratios.append(0.0); made += 1
            if made < args.empty_patch_rows:
                raise RuntimeError("failed to sample defect-free patches")
            check_gradients(model); norm = float(torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)); optimizer.step(); scheduler.step(); values /= accumulation
            full_n = sum(x == "full" for x in mix)
            patch_n = sum(x.startswith("patch") and not x.startswith("patch_empty") for x in mix)
            patch_empty_n = sum(x.startswith("patch_empty") for x in mix)
            full_empty_n = sum(x == "full_empty" for x in mix)
            input_mix = f"full={full_n},patch={patch_n},patch_empty={patch_empty_n},full_empty={full_empty_n}"
            array = np.asarray(ratios, dtype=float)
            small_frac = float(((array > 0) & (array < SMALL_RATIO)).mean()) if array.size else 0.0
            writer.writerow(dict(zip(fields, [step, *values[:3], values[3], small_frac,
                                             norm, optimizer.param_groups[0]["lr"],
                                             time.perf_counter()-begin, time.perf_counter()-started, input_mix])))
            stream.flush()
            state = {"step": step, "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "protocol": protocol}
            if step % 50 == 0:
                torch.save(state, args.out / "latest.pt")
            if step in EVAL_STEPS:
                print(f"[patch9v5 train] reached evaluation node {step}/{args.steps}", flush=True)
                if step in CHECKPOINT_NODES or step == args.steps:
                    torch.save(state, args.out / f"step{step:05d}.pt")
                    print(f"[patch9v5 train] checkpoint written: step{step:05d}.pt", flush=True)
        write_json(args.out / "training_summary.json", {"completed_steps": args.steps, "best_macro_dice": best, "elapsed_s": time.perf_counter()-started})


if __name__ == "__main__":
    main()
