"""Evaluate E1 on frozen fold1 Dev-mini without touching holdout images."""

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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor

from b0.b0plus_protocol import P0
from b0.run_dev720 import candidate_ids
from e1.model import QwenE1


SIDE = 448
EXPECTED_VIEWS = 3600
EXPECTED_GROUPS = 720
MACRO_AUPR_GATE = 0.7795
MACRO_F1_GATE = 0.7380
SMALL_DEFECT_RECALL_GATE = 0.9880952380952381
OK_MEAN_AREA_GATE = 0.10
OVERALL_AREA_P95_GATE = 0.25


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
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "f1max": f1,
        "f1max_threshold": threshold,
    }


def build_sample_scores(views: pd.DataFrame) -> pd.DataFrame:
    return (
        views.groupby(["class_name", "group_path"], as_index=False)
        .agg(
            sample_label=("sample_label", "first"),
            score=("p_yes", "max"),
            max_defect_ratio=("defect_ratio", "max"),
            predicted_area=("predicted_area", "max"),
        )
        .sort_values(["class_name", "group_path"], kind="mergesort")
    )


def summarize(views: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    samples = build_sample_scores(views)
    overall = binary_metrics(samples.sample_label.to_numpy(), samples.score.to_numpy())
    class_rows = []
    for class_name, part in samples.groupby("class_name", sort=True):
        class_rows.append({
            "class_name": class_name,
            **binary_metrics(part.sample_label.to_numpy(), part.score.to_numpy()),
            "n_samples": len(part),
        })
    per_class = pd.DataFrame(class_rows)
    threshold = overall["f1max_threshold"]
    small = samples[samples.sample_label.eq(1) & samples.max_defect_ratio.le(0.01)]
    ok_views = views[views.sample_label.eq(0)]
    summary = {
        "overall": overall,
        "macro_auroc": float(per_class.auroc.mean()),
        "macro_aupr": float(per_class.aupr.mean()),
        "macro_f1max": float(per_class.f1max.mean()),
        "small_defect_n": len(small),
        "small_defect_recall": float((small.score >= threshold).mean()),
        "predicted_area": {
            "view_mean": float(views.predicted_area.mean()),
            "view_p95": float(np.percentile(views.predicted_area, 95)),
            "ok_view_mean": float(ok_views.predicted_area.mean()),
            "ng_view_mean": float(views.loc[views.sample_label.eq(1), "predicted_area"].mean()),
            "sample_max_mean": float(samples.predicted_area.mean()),
        },
    }
    summary["stage_gate"] = {
        "macro_aupr": summary["macro_aupr"] >= MACRO_AUPR_GATE,
        "macro_f1max": summary["macro_f1max"] >= MACRO_F1_GATE,
        "small_defect_recall": summary["small_defect_recall"] >= SMALL_DEFECT_RECALL_GATE,
        "ok_predicted_area": summary["predicted_area"]["ok_view_mean"] <= OK_MEAN_AREA_GATE,
        "area_p95": summary["predicted_area"]["view_p95"] <= OVERALL_AREA_P95_GATE,
    }
    summary["stage_gate"]["fold1_pass"] = all(summary["stage_gate"].values())
    return summary, samples, per_class


def evaluate_model(
    model: QwenE1,
    processor,
    dev: pd.DataFrame,
    cache_root: Path,
    device: torch.device,
    out_dir: Path,
    tag: str,
    max_views: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
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
            conversation = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": image.copy()},
                    {"type": "text", "text": P0},
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
        inputs = {key: value.to(device) for key, value in inputs.items()}
        torch.cuda.synchronize()
        one_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=getattr(model, "autocast_dtype", torch.bfloat16)
        ):
            output = model(inputs)
        torch.cuda.synchronize()
        probability = float(torch.sigmoid(output.anomaly_logits[0]).cpu())
        predicted_area = float((torch.sigmoid(output.mask_logits[0, 0]) >= 0.5).float().mean().cpu())
        records.append({
            "class_name": row.class_name,
            "group_path": row.group_path,
            "image_path": row.image_path,
            "view_idx": row.view_idx,
            "sample_label": row.sample_label,
            "view_label": row.view_label,
            "defect_ratio": row.defect_ratio,
            "p_yes": probability,
            "predicted_area": predicted_area,
            "forward_s": time.perf_counter() - one_started,
        })
        if index % 100 == 0:
            print(f"[eval {tag}] {index}/{len(rows)}", flush=True)

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
        if len(views) != EXPECTED_VIEWS or views.group_path.nunique() != EXPECTED_GROUPS:
            raise RuntimeError("full fold1 Dev evaluation is incomplete")
        metrics, samples, per_class = summarize(views)
        result.update(metrics)
        samples.to_csv(out_dir / f"{tag}_samples.csv", index=False)
        per_class.to_csv(out_dir / f"{tag}_per_class.csv", index=False)
    (out_dir / f"{tag}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_model(
    model_id: str,
    checkpoint: Path | None,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
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
    model = QwenE1(
        base,
        yes_token_ids=candidate_ids(processor.tokenizer, "Yes"),
        no_token_ids=candidate_ids(processor.tokenizer, "No"),
    ).to(device)
    model.autocast_dtype = dtype
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_fold1"))
    parser.add_argument("--tag", default="best")
    parser.add_argument("--max-views", type=int)
    args = parser.parse_args()

    dev = pd.read_parquet(args.dev)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    if set(dev.class_name) & holdout:
        raise RuntimeError("fold1 Dev contains a holdout class")
    if args.max_views is None and (
        len(dev) != EXPECTED_VIEWS or dev.group_path.nunique() != EXPECTED_GROUPS
    ):
        raise RuntimeError("fold1 Dev is not the frozen 3600-view split")
    device = torch.device("cuda:0")
    model, processor = build_model(args.model_id, args.checkpoint, device)
    result = evaluate_model(
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
