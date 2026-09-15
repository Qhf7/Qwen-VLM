"""Evaluate the trained classifier -> segmenter cascade on a labeled image split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from b0.single_view import candidate_ids
from e1.evaluate_classification_fold1 import (
    build_classification_model,
    evaluate_classification,
)
from e1.model import QwenE1


SIDE = 448


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def dataset_path(root: Path, value: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a dataset-relative path: {value}")
    return root / relative


def load_gt_mask(row, gt_root: Path) -> np.ndarray:
    if int(row.view_label) == 0:
        return np.zeros((SIDE, SIDE), dtype=bool)
    if pd.isna(row.mask_path):
        raise ValueError(f"An anomalous image has no mask path: {row.image_path}")
    with Image.open(dataset_path(gt_root, row.mask_path)) as image:
        mask = np.asarray(
            image.convert("L").resize((SIDE, SIDE), Image.Resampling.NEAREST)
        ) > 0
    if not mask.any():
        raise ValueError(f"An anomalous image has an empty mask: {row.image_path}")
    return mask


def encode_segmentation(processor, image_path: Path) -> dict[str, torch.Tensor]:
    with Image.open(image_path) as image:
        batch = processor.image_processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in batch.items()}
    if inputs["image_grid_thw"].tolist() != [[1, 28, 28]]:
        raise ValueError(f"Unexpected Qwen3 image grid for {image_path}")
    return inputs


def build_segmentation_model(
    model_id: str, checkpoint: Path
) -> tuple[QwenE1, AutoProcessor, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = state.get("protocol", {})
    layers = tuple(int(value) for value in protocol.get("feature_layers", (5, 11, 17)))
    channels = int(protocol.get("decoder_channels", 288))
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=SIDE**2, max_pixels=SIDE**2
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to("cuda")
    model = QwenE1(
        base,
        candidate_ids(processor.tokenizer, "Yes"),
        candidate_ids(processor.tokenizer, "No"),
        feature_layers=layers,
        decoder_channels=channels,
    ).cuda()
    model.requires_grad_(False)
    model.decoder.load_state_dict(state["decoder"])
    model.eval()
    return model, processor, state


@torch.inference_mode()
def predict_segmentation(
    model: QwenE1,
    processor,
    image_path: Path,
) -> np.ndarray:
    inputs = encode_segmentation(processor, image_path)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features, _, _ = model._vision_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        low_resolution, _, _ = model.decoder(features)
        logits = F.interpolate(
            low_resolution, size=(SIDE, SIDE), mode="bilinear", align_corners=False
        )
    probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError(f"Non-finite segmentation output: {image_path}")
    return probability


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_policy(
    frame: pd.DataFrame, threshold: float
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gated = frame.p_yes.ge(threshold)
    anomaly = frame.view_label.eq(1)
    normal = ~anomaly
    predicted_pixels = np.where(gated, frame.raw_pred_area, 0).astype(np.int64)
    tp = np.where(gated, frame.raw_tp, 0).astype(np.int64)
    fp = np.where(gated, frame.raw_fp, 0).astype(np.int64)
    fn = frame.gt_area.to_numpy(np.int64) - tp
    denominator_dice = 2 * tp + fp + fn
    denominator_iou = tp + fp + fn
    dice = np.divide(
        2 * tp,
        denominator_dice,
        out=np.full(len(frame), np.nan),
        where=denominator_dice > 0,
    )
    iou = np.divide(
        tp,
        denominator_iou,
        out=np.full(len(frame), np.nan),
        where=denominator_iou > 0,
    )
    evaluated = frame.copy()
    evaluated["gate_threshold"] = threshold
    evaluated["gate_positive"] = gated
    evaluated["pred_area"] = predicted_pixels
    evaluated["tp"] = tp
    evaluated["fp"] = fp
    evaluated["fn"] = fn
    evaluated["dice"] = dice
    evaluated["iou"] = iou

    tn_i, fp_i, fn_i, tp_i = confusion_matrix(
        frame.view_label.astype(int), gated.astype(int), labels=[0, 1]
    ).ravel()
    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    selected_anomaly = anomaly & gated
    false_positive_images = normal & gated
    anomaly_frame = evaluated[anomaly]
    selected_anomaly_frame = evaluated[selected_anomaly]
    normal_pixels = int(normal.sum()) * SIDE * SIDE
    summary = {
        "classification_threshold": float(threshold),
        "images": len(frame),
        "segmented_images": int(gated.sum()),
        "segmented_fraction": float(gated.mean()),
        "classification": {
            "tn": int(tn_i),
            "fp": int(fp_i),
            "fn": int(fn_i),
            "tp": int(tp_i),
            "accuracy": safe_ratio(tp_i + tn_i, len(frame)),
            "precision": safe_ratio(tp_i, tp_i + fp_i),
            "recall": safe_ratio(tp_i, tp_i + fn_i),
            "f1": safe_ratio(2 * tp_i, 2 * tp_i + fp_i + fn_i),
        },
        "end_to_end_pixels": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": safe_ratio(total_tp, total_tp + total_fp),
            "recall": safe_ratio(total_tp, total_tp + total_fn),
            "dice": safe_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
            "iou": safe_ratio(total_tp, total_tp + total_fp + total_fn),
        },
        "anomaly_image_macro": {
            "n": int(anomaly.sum()),
            "dice": float(anomaly_frame.dice.mean()),
            "iou": float(anomaly_frame.iou.mean()),
        },
        "conditional_segmentation_on_classifier_tp": {
            "n": int(selected_anomaly.sum()),
            "dice": float(selected_anomaly_frame.dice.mean()),
            "iou": float(selected_anomaly_frame.iou.mean()),
        },
        "normal_false_positive_masks": {
            "classification_fp_images": int(false_positive_images.sum()),
            "nonempty_mask_images": int((false_positive_images & frame.raw_pred_area.gt(0)).sum()),
            "predicted_pixels": int(np.where(false_positive_images, frame.raw_pred_area, 0).sum()),
            "pixel_rate_over_all_normal_images": safe_ratio(
                int(np.where(false_positive_images, frame.raw_pred_area, 0).sum()), normal_pixels
            ),
        },
    }

    metrics = ["dice", "iou"]
    per_class = anomaly_frame.groupby("class_name")[metrics].mean()
    per_class["anomaly_n"] = anomaly_frame.groupby("class_name").size()
    per_class["classification_recall"] = anomaly_frame.groupby("class_name")[
        "gate_positive"
    ].mean()
    buckets = pd.cut(
        anomaly_frame.defect_ratio,
        [0, 0.001, 0.01, 1],
        labels=["tiny_0_0.1pct", "small_0.1_1pct", "large_1_100pct"],
    )
    per_area = anomaly_frame.groupby(buckets, observed=True)[metrics].mean()
    per_area["anomaly_n"] = anomaly_frame.groupby(buckets, observed=True).size()
    per_area["classification_recall"] = anomaly_frame.groupby(buckets, observed=True)[
        "gate_positive"
    ].mean()
    return summary, evaluated, per_class.reset_index(), per_area.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--classification-checkpoint",
        type=Path,
        default=Path("artifacts/e1_real_classification_qwen3_full_12000/best.pt"),
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation_qwen3_decoder_only_12000/best.pt"),
    )
    parser.add_argument("--test", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--image-root", type=Path, default=Path("Dataset448"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/two_stage_12000_test"))
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(args.test).reset_index(drop=True)
    if test.image_path.duplicated().any() or not test.view_label.isin([0, 1]).all():
        raise ValueError("Test rows must be unique images with binary view labels")
    if len(test) != 3600 or test.class_name.nunique() != 18:
        raise ValueError("Expected the frozen 3600-image, 18-class evaluation split")

    classification_state = torch.load(
        args.classification_checkpoint, map_location="cpu", weights_only=False
    )
    classification_checkpoint_step = int(classification_state["step"])
    feature_layers = tuple(classification_state["protocol"]["feature_layers"])
    classifier, classification_processor = build_classification_model(
        args.model_id,
        args.classification_checkpoint,
        torch.device("cuda:0"),
        dtype=torch.bfloat16,
        feature_layers=feature_layers,
    )
    classification_result = evaluate_classification(
        classifier,
        classification_processor,
        test,
        args.image_root,
        torch.device("cuda:0"),
        args.out,
        "classification",
    )
    classification_views = pd.read_csv(args.out / "classification_views.csv")
    # The shared classification evaluator intentionally writes only its
    # classification fields; restore segmentation ground-truth metadata from
    # the frozen test manifest by the unique image path.
    metadata = test[["image_path", "mask_path", "defect_ratio", "view_label"]].copy()
    classification_views = classification_views.drop(
        columns=["mask_path", "defect_ratio", "view_label"], errors="ignore"
    ).merge(metadata, on="image_path", how="left", validate="one_to_one")
    if classification_views.mask_path.isna().any():
        raise RuntimeError("Failed to join test mask metadata to classification outputs")
    del classifier, classification_processor, classification_state
    gc.collect()
    torch.cuda.empty_cache()

    thresholds = sorted(
        {
            0.5,
            float(classification_result["overall"]["f1max_threshold"]),
        }
    )
    inference_threshold = min(thresholds)
    segmenter, segmentation_processor, segmentation_state = build_segmentation_model(
        args.model_id, args.segmentation_checkpoint
    )
    mask_dir = args.out / "segmentation_masks"
    records = []
    selected = classification_views.p_yes.ge(inference_threshold)
    started = time.perf_counter()
    for position, row in enumerate(classification_views.itertuples(index=False), 1):
        gt = load_gt_mask(row, args.gt_root)
        probability = None
        prediction = np.zeros((SIDE, SIDE), dtype=bool)
        if bool(selected.iloc[position - 1]):
            probability = predict_segmentation(
                segmenter,
                segmentation_processor,
                dataset_path(args.image_root, row.image_path),
            )
            prediction = probability >= args.segmentation_threshold
            destination = mask_dir / row.image_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction.astype(np.uint8) * 255).save(destination)
        tp = int((prediction & gt).sum())
        fp = int((prediction & ~gt).sum())
        fn = int((~prediction & gt).sum())
        records.append(
            {
                **row._asdict(),
                "segmentation_was_run": bool(selected.iloc[position - 1]),
                "raw_pred_area": int(prediction.sum()),
                "raw_tp": tp,
                "raw_fp": fp,
                "raw_fn": fn,
                "gt_area": int(gt.sum()),
                "mask_probability_mean": float(probability.mean()) if probability is not None else 0.0,
            }
        )
        if position % 100 == 0:
            print(
                f"[two-stage segmentation] {position}/{len(classification_views)} "
                f"run={int(selected.iloc[:position].sum())}/{int(selected.sum())}",
                flush=True,
            )
    raw = pd.DataFrame(records)
    raw.to_csv(args.out / "raw_two_stage_views.csv", index=False)

    policies = {}
    for threshold in thresholds:
        name = "fixed_0_5" if threshold == 0.5 else "classification_f1max"
        summary, views, per_class, per_area = summarize_policy(raw, threshold)
        policies[name] = summary
        views.to_csv(args.out / f"{name}_views.csv", index=False)
        per_class.to_csv(args.out / f"{name}_per_class.csv", index=False)
        per_area.to_csv(args.out / f"{name}_per_area.csv", index=False)

    result = {
        "task": "two-stage image classification then pixel segmentation",
        "test_split": str(args.test),
        "test_sha256": sha256_file(args.test),
        "classification_checkpoint": str(args.classification_checkpoint),
        "classification_checkpoint_step": classification_checkpoint_step,
        "segmentation_checkpoint": str(args.segmentation_checkpoint),
        "segmentation_checkpoint_step": int(segmentation_state["step"]),
        "segmentation_threshold": args.segmentation_threshold,
        "classification_evaluation": classification_result,
        "segmentation_runtime_s": time.perf_counter() - started,
        "policies": policies,
    }
    write_json(args.out / "two_stage_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
