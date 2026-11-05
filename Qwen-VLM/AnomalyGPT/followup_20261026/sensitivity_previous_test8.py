"""CPU-only sensitivity analysis excluding eight previously evaluated normals.

Main 2,005-source results remain unchanged. This script reads complete prediction
JSONL files and writes a separate sensitivity report; pixel NPZs are not read or
rescored. The exclusion input may be the original test8 cache manifest (normal
and synthetic variants repeat each source) or an explicit source JSON list.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from summarize_followup import classification, ranking


SEEDS = (42, 43, 44)
METRICS = ("generation_macro_f1", "generation_anomaly_recall", "generation_normal_fpr",
           "generation_invalid_rate", "map_image_auroc", "map_image_ap",
           "answer_image_auroc", "answer_image_ap")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def normalize_source(value, category=None):
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if category and category in parts:
        return str(PurePosixPath(*parts[parts.index(category):]))
    if "visa_data" in parts:
        return str(PurePosixPath(*parts[parts.index("visa_data") + 1:]))
    return str(PurePosixPath(str(value).replace("\\", "/")))


def exclusions(path):
    data = read_json(path)
    records = data.get("records", data.get("sources", [])) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Exclusion JSON must contain a records/sources list")
    unique = {}
    for item in records:
        if isinstance(item, str):
            image = normalize_source(item)
            category = image.split("/", 1)[0]
        elif isinstance(item, dict):
            category = item.get("class_name", item.get("object"))
            value = item.get("source_path", item.get("image", item.get("image_relative")))
            if not value:
                raise ValueError("Exclusion record lacks a source path")
            image = normalize_source(value, category)
            category = category or image.split("/", 1)[0]
        else:
            raise ValueError("Invalid exclusion record")
        if "/Data/Images/Normal/" not in image:
            raise ValueError("This analysis expects the eight previous normal sources: " + image)
        unique[image] = {"source_path": image, "class_name": category,
                         "source_id": hashlib.sha256(image.encode()).hexdigest()[:24], "expected_label": 0}
    if len(unique) != 8:
        raise ValueError(f"Exactly eight unique previous test sources are required, got {len(unique)}")
    return {"input": str(Path(path).resolve()), "input_sha256": digest(path),
            "unique_source_count": len(unique), "input_record_count": len(records),
            "records": [unique[key] for key in sorted(unique)]}


def scores(rows):
    generated = classification(rows)
    map_rank = ranking([row["label"] for row in rows], [row["map_score"] for row in rows])
    answer_rank = ranking([row["label"] for row in rows], [row["answer_log_odds"] for row in rows])
    return {"generation_macro_f1": generated["macro_f1"],
            "generation_anomaly_recall": generated["anomaly_recall"],
            "generation_normal_fpr": generated["normal_fpr"],
            "generation_invalid_rate": generated["invalid_rate"],
            "map_image_auroc": map_rank["auroc"], "map_image_ap": map_rank["ap"],
            "answer_image_auroc": answer_rank["auroc"], "answer_image_ap": answer_rank["ap"]}


def audit_and_analyze(directory, excluded):
    prediction_path = directory / "predictions.jsonl"
    run_path, metrics_path = directory / "run.json", directory / "metrics.json"
    base = {"name": directory.name, "predictions": str(prediction_path.resolve()), "status": "partial_or_unverified"}
    if not run_path.is_file() or not metrics_path.is_file():
        return {**base, "reason": "missing run.json or metrics.json"}
    run, metrics = read_json(run_path), read_json(metrics_path)
    if run.get("completed") is not True or run.get("completed_num_sources") != 2005 or metrics.get("num_sources") != 2005:
        return {**base, "reason": "completion metadata does not verify 2,005 sources"}
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    prediction_sha = digest(prediction_path)
    if prediction_sha != run.get("prediction_sha256"):
        raise ValueError("Prediction hash mismatch: " + directory.name)
    paths = [row["source_path"] for row in rows]
    if len(rows) != 2005 or len(set(paths)) != 2005:
        raise ValueError("Complete evaluation source count or uniqueness mismatch: " + directory.name)
    for row in rows:
        expected_id = hashlib.sha256(row["source_path"].encode()).hexdigest()[:24]
        if row.get("source_id") != expected_id or row.get("label") not in (0, 1) or row.get("prediction") not in (-1, 0, 1):
            raise ValueError("Invalid source ID, label or prediction: " + directory.name)
        if not all(np.isfinite(row[key]) for key in ("map_score", "answer_log_odds")):
            raise ValueError("Nonfinite ranking scores: " + directory.name)
    excluded_paths = {row["source_path"] for row in excluded["records"]}
    removed = [row for row in rows if row["source_path"] in excluded_paths]
    if len(removed) != 8 or any(row["label"] != 0 for row in removed):
        raise ValueError("Eight excluded sources must all be present with normal GT: " + directory.name)
    kept = [row for row in rows if row["source_path"] not in excluded_paths]
    main, sensitivity = scores(rows), scores(kept)
    return {**base, "status": "complete", "prediction_sha256": prediction_sha,
            "main_source_count": len(rows), "sensitivity_source_count": len(kept),
            "main_class_counts": dict(Counter(int(row["label"]) for row in rows)),
            "sensitivity_class_counts": dict(Counter(int(row["label"]) for row in kept)),
            "removed_sources": [{key: row[key] for key in ("source_path", "source_id", "class_name", "label", "prediction")} for row in removed],
            "main_metrics_recomputed": main, "sensitivity_metrics": sensitivity,
            "difference_sensitivity_minus_main": {key: sensitivity[key] - main[key]
                                                  if sensitivity[key] is not None and main[key] is not None else None for key in METRICS}}


def groups(runs):
    definitions = {"historical_frozen": "historical_frozen", "historical_lora": "historical_lora",
                   "matched_qwen_frozen": "matched_qwen_frozen", "matched_qwen_lora": "matched_qwen_lora",
                   "matched_vicuna": "matched_vicuna"}
    by_name = {row["name"]: row for row in runs if row["status"] == "complete"}
    result = {}
    for name, prefix in definitions.items():
        available = [(seed, by_name[prefix + str(seed)]) for seed in SEEDS if prefix + str(seed) in by_name]
        entry = {"status": "complete" if len(available) == 3 else "pending",
                 "completed_seeds": [seed for seed, _ in available], "expected_seeds": list(SEEDS), "metrics": {}}
        for metric in METRICS:
            entry["metrics"][metric] = {}
            for field in ("main_metrics_recomputed", "sensitivity_metrics", "difference_sensitivity_minus_main"):
                values = [row[field][metric] for _, row in available if row[field][metric] is not None]
                entry["metrics"][metric][field] = {"n": len(values), "values": values,
                    "mean": float(np.mean(values)) if values else None,
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        result[name] = entry
    return result


def markdown(report):
    lines = ["# 旧 test8 正常来源排除敏感性分析", "",
             "主分析的2,005张来源及其原有结果保持不变。本分析仅另行剔除8张曾参与早期 synthetic 评测的正常源图，剩余1,997张（917正常、1,080异常）。所有值由现有逐图预测重算，未重跑GPU推理，也未重算像素指标。", "",
             "| 模型 | 状态 | 剔除后生成F1 | 异常召回 | 正常FPR | 无效回答 | 热图图像AUROC | 回答AUROC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    selected = ("generation_macro_f1", "generation_anomaly_recall", "generation_normal_fpr", "generation_invalid_rate", "map_image_auroc", "answer_image_auroc")
    for row in report["runs"]:
        if row["status"] == "complete":
            values = [f"{row['sensitivity_metrics'][key] * 100:.4f}" if row["sensitivity_metrics"][key] is not None else "—" for key in selected]
        else:
            values = ["待完成"] * len(selected)
        lines.append("| " + " | ".join([row["name"], row["status"], *values]) + " |")
    lines += ["", "所有表格数值为百分比；差值、AP和三seed均值／样本标准差见JSON。异常召回的分母与异常来源不变；正常FPR和图像排序分母改变。该敏感性分析不将已查看的主集合重新定义为独立确认集。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Final/progress results directory containing real_evaluations/")
    parser.add_argument("--excluded-sources", type=Path, default=Path(__file__).with_name("previous_test8_excluded_sources.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = exclusions(args.excluded_sources)
    runs = []
    for path in sorted((root / "real_evaluations").glob("*/predictions.jsonl")):
        try:
            runs.append(audit_and_analyze(path.parent, excluded))
        except (ValueError, OSError, KeyError, TypeError) as error:
            runs.append({"name": path.parent.name, "status": "audit_failed", "error": str(error)})
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "exclusion": excluded, "runs": runs, "three_seed_groups": groups(runs),
              "source_sha256": digest(__file__),
              "scope": "Separate CPU sensitivity diagnostic only; main 2,005-source protocol and files unchanged; pixel maps not reread or rescored",
              "direction": "difference_sensitivity_minus_main; rate units are fractions; sample SD uses ddof=1"}
    output = args.out or root / "final_report/sensitivity_previous_test8.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"out": str(output.resolve()), "complete": sum(row["status"] == "complete" for row in runs),
                      "runs": len(runs), "excluded_sources": excluded["unique_source_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
