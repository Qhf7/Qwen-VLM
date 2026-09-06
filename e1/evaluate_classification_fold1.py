"""Evaluate classification-only E1 checkpoints on frozen fold1 Dev720."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                            accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from transformers import AutoModelForImageTextToText, AutoProcessor

from b0.b0plus_protocol import P4_CLASSIFICATION_ONLY
from b0.single_view import candidate_ids
from e1.classification_model import QwenE1Classification


SIDE = 448
EXPECTED_CLASSES = 18
EXPECTED_GROUPS = 720
EXPECTED_VIEWS = 3600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1max_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.argmax(values))
    return float(values[index]), float(thresholds[index])


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    f1, threshold = f1max_threshold(labels, scores)
    predicted = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "threshold": 0.5,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "f1max": f1,
        "f1max_threshold": threshold,
    }


def summarize(views: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if views.image_path.duplicated().any() or not views.view_label.isin([0, 1]).all():
        raise ValueError("Every evaluation row must be a unique image with a binary view label")
    overall = binary_metrics(views.view_label.to_numpy(), views.p_yes.to_numpy())
    class_rows = []
    for class_name, part in views.groupby("class_name", sort=True):
        class_rows.append({
            "class_name": class_name,
            **binary_metrics(part.view_label.to_numpy(), part.p_yes.to_numpy()),
            "n_views": len(part),
            "n_ok": int(part.view_label.eq(0).sum()),
            "n_ng": int(part.view_label.eq(1).sum()),
        })
    per_class = pd.DataFrame(class_rows)
    threshold = overall["f1max_threshold"]
    small = views[views.view_label.eq(1) & views.defect_ratio.gt(0) & views.defect_ratio.le(0.01)]
    ok = views[views.view_label.eq(0)].p_yes
    ng = views[views.view_label.eq(1)].p_yes
    result = {
        "task": "image-level classification only",
        "evaluation_unit": "single image",
        "target": "view_label",
        "overall": overall,
        "macro_auroc": float(per_class.auroc.mean()),
        "macro_aupr": float(per_class.aupr.mean()),
        "macro_f1max": float(per_class.f1max.mean()),
        "small_defect_n": len(small),
        "small_defect_recall": float((small.p_yes >= 0.5).mean()) if len(small) else None,
        "small_defect_recall_at_f1max": float((small.p_yes >= threshold).mean()) if len(small) else None,
        "score_diagnostics": {
            "view_std": float(views.p_yes.std(ddof=0)),
            "ok_mean": float(ok.mean()),
            "ok_median": float(ok.median()),
            "ng_mean": float(ng.mean()),
            "ng_median": float(ng.median()),
            "median_separation_ng_minus_ok": float(ng.median() - ok.median()),
            "positive_rate_at_0_5": float((views.p_yes >= 0.5).mean()),
            "positive_rate_at_f1max": float((views.p_yes >= threshold).mean()),
        },
    }
    return result, views, per_class


def encode(processor, image: Image.Image, device: torch.device) -> dict[str, torch.Tensor]:
    conversation = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": P4_CLASSIFICATION_ONLY},
        ],
    }]]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    return {key: value.to(device) for key, value in inputs.items()}


def evaluate_classification(
    model: QwenE1Classification,
    processor,
    dev: pd.DataFrame,
    cache_root: Path,
    device: torch.device,
    out_dir: Path,
    tag: str,
    max_views: int | None = None,
) -> dict:
    rows = dev.sort_values(
        ["class_name", "condition", "group_path", "view_idx"], kind="mergesort"
    )
    if max_views is not None:
        rows = rows.head(max_views)
    records = []
    model.eval()
    started = time.perf_counter()
    for index, row in enumerate(rows.itertuples(index=False), 1):
        with Image.open(cache_root / row.image_path) as source:
            image = source.convert("RGB")
            if image.size != (SIDE, SIDE):
                raise RuntimeError(f"Dev cache is not direct448: {row.image_path}")
            inputs = encode(processor, image.copy(), device)
        torch.cuda.synchronize()
        one_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=getattr(model, "autocast_dtype", torch.bfloat16)
        ):
            output = model(inputs)
        torch.cuda.synchronize()
        records.append({
            "class_name": row.class_name,
            "group_path": row.group_path,
            "image_path": row.image_path,
            "view_idx": row.view_idx,
            "condition": row.condition,
            "sample_label": row.sample_label,
            "view_label": row.view_label,
            "defect_ratio": row.defect_ratio,
            "p_yes": float(torch.sigmoid(output.anomaly_logits[0]).cpu()),
            "forward_s": time.perf_counter() - one_started,
        })
        if index % 100 == 0:
            print(f"[classification eval {tag}] {index}/{len(rows)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    views = pd.DataFrame(records)
    views.to_csv(out_dir / f"{tag}_views.csv", index=False)
    result = {
        "tag": tag,
        "n_views": len(views),
        "elapsed_s": time.perf_counter() - started,
        "forward_p50_s": float(np.percentile(views.forward_s, 50)),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if max_views is None:
        if (
            len(views) != EXPECTED_VIEWS
            or views.group_path.nunique() != EXPECTED_GROUPS
            or views.class_name.nunique() != EXPECTED_CLASSES
        ):
            raise RuntimeError("classification evaluation is not complete Dev720")
        metrics, _, per_class = summarize(views)
        result.update(metrics)
        per_class.to_csv(out_dir / f"{tag}_per_class.csv", index=False)
    (out_dir / f"{tag}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_classification_model(
    model_id: str,
    checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    feature_layers: tuple[int, ...] = (7, 15, 23, 31),
):
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE
    )
    processor.tokenizer.padding_side = "left"
    base = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    model = QwenE1Classification(
        base,
        yes_token_ids=candidate_ids(processor.tokenizer, "Yes"),
        no_token_ids=candidate_ids(processor.tokenizer, "No"),
        feature_layers=feature_layers,
    ).to(device)
    model.autocast_dtype = dtype
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not state.get("classification_only", False):
            raise RuntimeError("checkpoint is not a classification-only stage-1 checkpoint")
        model.decoder.load_state_dict(state["decoder"])
        model.prompt_learner.load_state_dict(state["prompt_learner"])
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--cache-root", type=Path, default=Path("Dataset448"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_classification_fold1"))
    parser.add_argument("--tag", default="best")
    parser.add_argument("--max-views", type=int)
    parser.add_argument("--feature-layers", default="7,15,23,31")
    args = parser.parse_args()

    dev = pd.read_parquet(args.dev)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    if set(dev.class_name) & holdout:
        raise RuntimeError("fold1 Dev contains a holdout class")
    if args.max_views is None and (
        len(dev) != EXPECTED_VIEWS
        or dev.group_path.nunique() != EXPECTED_GROUPS
        or dev.class_name.nunique() != EXPECTED_CLASSES
    ):
        raise RuntimeError("fold1 Dev is not the frozen 18-class Dev720")
    feature_layers = tuple(int(value) for value in args.feature_layers.split(",") if value.strip())
    device = torch.device("cuda:0")
    model, processor = build_classification_model(
        args.model_id, args.checkpoint, device, feature_layers=feature_layers
    )
    result = evaluate_classification(
        model, processor, dev, args.cache_root, device, args.out_dir, args.tag, args.max_views
    )
    result["dev_sha256"] = sha256_file(args.dev)
    result["holdout_intersection"] = []
    (args.out_dir / f"{args.tag}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
