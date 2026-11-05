"""CPU diagnostics for real-evaluation prediction JSONL; no model inference.

Usage: python diagnose_generation.py --root /path/to/followup_20261026 --out diagnostics.json
The JSON and a short Chinese Markdown companion distinguish full 2,005-source
prediction sets from partial runs. Statistics describe outputs, not whether an
explanation is faithful to the model's internal reasoning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np


PREDICTION_NAMES = {0: "no", 1: "yes", -1: "invalid"}


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def gt_label(row):
    value = row.get("label")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        return {"0": 0, "1": 1, "normal": 0, "anomaly": 1, "false": 0, "true": 1}.get(value.lower().strip())
    return None


def parsed_prediction(row):
    text = str(row.get("answer", ""))
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.I)
    return -1 if match is None else int(match.group(1).lower() == "yes")


def normalized_answer(answer):
    return " ".join(unicodedata.normalize("NFKC", str(answer)).casefold().split())


def source_key(row):
    if row.get("source_path"):
        return "path:" + str(row["source_path"])
    if row.get("source_id"):
        return "id:" + str(row["source_id"])
    return None


def quantiles(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "mean": float(array.mean()),
            **{f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95)},
            "max": float(array.max())}


def average_ranks(values):
    """NumPy fallback matching scipy rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
    return ranks


def correlation(rows):
    pairs = [(finite_number(row.get("map_score")), finite_number(row.get("answer_log_odds"))) for row in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 2:
        return {"count": len(pairs), "spearman": None, "reason": "fewer_than_two_finite_pairs"}
    values = np.asarray(pairs, dtype=np.float64)
    if np.ptp(values[:, 0]) == 0 or np.ptp(values[:, 1]) == 0:
        return {"count": len(pairs), "spearman": None, "reason": "constant_input"}
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(values[:, 0], values[:, 1]).statistic)
        backend = "scipy.stats.spearmanr"
    except ImportError:
        rho = float(np.corrcoef(average_ranks(values[:, 0]), average_ranks(values[:, 1]))[0, 1])
        backend = "numpy_average_tie_ranks"
    return {"count": len(pairs), "spearman": rho, "backend": backend}


def confusion(rows):
    counts = {"gt_normal": {"no": 0, "yes": 0, "invalid": 0},
              "gt_anomaly": {"no": 0, "yes": 0, "invalid": 0}}
    unknown = 0
    for row in rows:
        gt = gt_label(row)
        if gt is None:
            unknown += 1
        else:
            counts["gt_anomaly" if gt else "gt_normal"][PREDICTION_NAMES[parsed_prediction(row)]] += 1
    return {"num_sources": len(rows), "counts": counts, "unknown_gt": unknown}


def load_predictions(path):
    rows, errors = [], []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("prediction must be an object")
                rows.append(row)
            except (json.JSONDecodeError, ValueError) as error:
                errors.append({"line": line_number, "error": type(error).__name__})
    return rows, errors


def diagnose_run(directory, expected):
    path = directory / "predictions.jsonl"
    rows, parse_errors = load_predictions(path)
    keys = [source_key(row) for row in rows]
    frequencies = Counter(key for key in keys if key is not None)
    duplicate_keys = {key: count for key, count in frequencies.items() if count > 1}
    complete = len(rows) == expected and len(frequencies) == expected and not parse_errors
    metadata = {}
    for name in ("run.json", "metrics.json"):
        file = directory / name
        if file.is_file():
            try:
                data = json.loads(file.read_text())
                metadata[name] = {key: data[key] for key in ("completed", "completed_num_sources", "num_sources", "backend", "seed") if key in data}
            except (json.JSONDecodeError, OSError):
                metadata[name] = {"read_error": True}
    counts = Counter(PREDICTION_NAMES[parsed_prediction(row)] for row in rows)
    answer_counts = Counter(normalized_answer(row.get("answer", "")) for row in rows)
    class_groups = defaultdict(list)
    for row in rows:
        class_groups[str(row.get("class_name", "unknown"))].append(row)
    termination = Counter(str(row.get("termination", "missing")) for row in rows)
    raw_lengths = [len(row["generated_token_ids"]) if isinstance(row.get("generated_token_ids"), list) else None for row in rows]
    finite_map_rows = [(row, finite_number(row.get("map_score"))) for row in rows]
    finite_map_rows = [(row, score) for row, score in finite_map_rows if score is not None]
    high_threshold = float(np.quantile([score for _, score in finite_map_rows], 0.75)) if finite_map_rows else None
    high = [row for row, score in finite_map_rows if score >= high_threshold] if high_threshold is not None else []
    high_no = [row for row in high if parsed_prediction(row) == 0]
    zero_foreground_anomalies = [row for row in rows if gt_label(row) == 1 and finite_number(row.get("foreground_pixels_224")) == 0]
    stored_prediction_disagreements = sum(row.get("prediction") in (-1, 0, 1) and row.get("prediction") != parsed_prediction(row) for row in rows)
    out = {
        "name": directory.name, "predictions": str(path.resolve()), "status": f"complete_{expected}" if complete else "partial_or_invalid",
        "expected_sources": expected, "num_records": len(rows), "unique_source_keys": len(frequencies),
        "records_without_source_key": sum(key is None for key in keys), "duplicate_source_keys": duplicate_keys,
        "jsonl_parse_errors": parse_errors, "completion_metadata": metadata,
        "answer_counts": {key: counts.get(key, 0) for key in ("yes", "no", "invalid")},
        "answer_fractions": {key: counts.get(key, 0) / len(rows) if rows else None for key in ("yes", "no", "invalid")},
        "top_normalized_answers": [{"answer": answer, "count": count, "fraction": count / len(rows)} for answer, count in answer_counts.most_common(10)],
        "stored_vs_reparsed_prediction_disagreements": int(stored_prediction_disagreements),
        "confusion": confusion(rows), "per_class_confusion": {key: confusion(group) for key, group in sorted(class_groups.items())},
        "raw_generated_token_length": quantiles(raw_lengths), "missing_token_id_records": sum(length is None for length in raw_lengths),
        "termination_counts": dict(sorted(termination.items())),
        "raw_token_lengths_by_termination": {reason: quantiles([length for row, length in zip(rows, raw_lengths) if str(row.get("termination", "missing")) == reason]) for reason in sorted(termination)},
        "map_score_quantiles": quantiles([score for _, score in finite_map_rows]),
        "map_answer_log_odds_correlation": correlation(rows),
        "per_class_map_answer_spearman": {key: correlation(group) for key, group in sorted(class_groups.items())},
        "high_map_quartile_no_answers": {
            "threshold_q75": high_threshold, "selection": "map_score >= q75; boundary ties retained",
            "selected_sources": len(high), "no_answer_sources": len(high_no),
            "no_answer_gt_normal": sum(gt_label(row) == 0 for row in high_no),
            "no_answer_gt_anomaly": sum(gt_label(row) == 1 for row in high_no),
            "no_answer_gt_unknown": sum(gt_label(row) is None for row in high_no),
            "per_class": dict(sorted(Counter(str(row.get("class_name", "unknown")) for row in high_no).items())),
        },
        "official_anomaly_zero_foreground_224": {"count": len(zero_foreground_anomalies),
            "per_class": dict(sorted(Counter(str(row.get("class_name", "unknown")) for row in zero_foreground_anomalies).items())),
            "source_paths": [row.get("source_path", row.get("source_id")) for row in zero_foreground_anomalies]},
    }
    return out, rows


def paired_flips(name, left_name, right_name, all_rows, reports):
    out = {"name": name, "left": left_name, "right": right_name}
    if left_name not in all_rows or right_name not in all_rows:
        return {**out, "status": "missing_run", "missing": [run for run in (left_name, right_name) if run not in all_rows]}
    if reports[left_name]["duplicate_source_keys"] or reports[right_name]["duplicate_source_keys"]:
        return {**out, "status": "duplicate_source_keys_no_pairing"}
    left = {source_key(row): row for row in all_rows[left_name] if source_key(row) is not None}
    right = {source_key(row): row for row in all_rows[right_name] if source_key(row) is not None}
    common = sorted(left.keys() & right.keys())
    transitions = Counter()
    by_gt = defaultdict(Counter)
    label_disagreement = []
    corrected, regressed, changed = 0, 0, 0
    for key in common:
        a, b = left[key], right[key]
        if gt_label(a) != gt_label(b):
            label_disagreement.append(key)
            continue
        old, new, gt = parsed_prediction(a), parsed_prediction(b), gt_label(a)
        transition = PREDICTION_NAMES[old] + "->" + PREDICTION_NAMES[new]
        transitions[transition] += 1
        by_gt["anomaly" if gt == 1 else "normal" if gt == 0 else "unknown"][transition] += 1
        changed += old != new
        if gt is not None:
            corrected += old != gt and new == gt
            regressed += old == gt and new != gt
    return {**out, "status": "paired_full_sets" if reports[left_name]["status"].startswith("complete_") and reports[right_name]["status"].startswith("complete_") and len(common) == reports[left_name]["expected_sources"] else "paired_intersection_partial",
            "left_records": len(left), "right_records": len(right), "common_sources": len(common),
            "left_only": len(left.keys() - right.keys()), "right_only": len(right.keys() - left.keys()),
            "gt_disagreement_keys": label_disagreement, "prediction_changed_sources": int(changed),
            "incorrect_to_correct": int(corrected), "correct_to_incorrect": int(regressed),
            "transitions": dict(sorted(transitions.items())),
            "transitions_by_gt": {key: dict(sorted(values.items())) for key, values in sorted(by_gt.items())}}


def markdown(report):
    lines = ["# 真实异常生成输出诊断", "", f"完整集合以 {report['expected_sources']} 个不重复来源为准；其他输出标为 partial。下表只描述输出统计。", "",
             "| 运行 | 状态 | 记录数 | Yes / No / 无法解析 | 最常见答案占比 | map–答案分数 Spearman | 高 map 四分位中 No（异常 GT） | 异常 GT 零前景 |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for name, run in report["runs"].items():
        counts = run["answer_counts"]
        common = run["top_normalized_answers"][0]["fraction"] if run["top_normalized_answers"] else None
        rho = run["map_answer_log_odds_correlation"]["spearman"]
        high = run["high_map_quartile_no_answers"]
        lines.append(f"| {name} | {run['status']} | {run['num_records']} | {counts['yes']} / {counts['no']} / {counts['invalid']} | {common:.2%}" +
                     (f" | {rho:.4f}" if rho is not None else " | —") + f" | {high['no_answer_sources']}（{high['no_answer_gt_anomaly']}） | {run['official_anomaly_zero_foreground_224']['count']} |") if common is not None else lines.append(f"| {name} | {run['status']} | 0 | 0 / 0 / 0 | — | — | — | — |")
    lines += ["", "| 配对比较 | 状态 | 共有来源 | 预测改变 | 错→对 | 对→错 |", "|---|---|---:|---:|---:|---:|"]
    for pair in report["paired_flips"]:
        lines.append(f"| {pair['name']} | {pair['status']} | {pair.get('common_sources', '—')} | {pair.get('prediction_changed_sources', '—')} | {pair.get('incorrect_to_correct', '—')} | {pair.get('correct_to_incorrect', '—')} |")
    lines += ["", "Yes/No 按答案开头重新解析；分位数边界保留并列值。像素前景为零不改变官方图像标签。完整答案、按类混淆、终止方式、token 长度和配对翻转明细见同名 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-sources", type=int, default=2005)
    args = parser.parse_args()
    if args.expected_sources < 1:
        parser.error("expected-sources must be positive")
    runs, all_rows = {}, {}
    for predictions in sorted((args.root / "real_evaluations").glob("*/predictions.jsonl")):
        directory = predictions.parent
        runs[directory.name], all_rows[directory.name] = diagnose_run(directory, args.expected_sources)
    pairs = [paired_flips("historical_frozen42_to_lora42", "historical_frozen42", "historical_lora42", all_rows, runs),
             paired_flips("matched_frozen42_to_lora42", "matched_qwen_frozen42", "matched_qwen_lora42", all_rows, runs)]
    report = {"root": str(args.root.resolve()), "expected_sources": args.expected_sources,
              "num_runs": len(runs), "complete_runs": [name for name, run in runs.items() if run["status"].startswith("complete_")],
              "partial_runs": [name for name, run in runs.items() if not run["status"].startswith("complete_")],
              "runs": runs, "paired_flips": pairs,
              "definitions": {"answer_normalization": "Unicode NFKC, casefold, collapse whitespace; no content substitution",
                              "complete": "Expected record count and unique source keys, with no malformed JSONL lines; finalization metadata reported separately",
                              "correlation": "Spearman correlation of finite map maxima and Yes-No continuation log-odds, average ranks for ties",
                              "faithfulness": "No conclusion about explanation faithfulness is inferred from these output statistics."}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    args.out.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"out": str(args.out), "runs": len(runs), "complete": report["complete_runs"], "partial": report["partial_runs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
