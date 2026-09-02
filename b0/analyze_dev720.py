"""冻结 Dev-mini 720 样本的聚合、宏平均、bootstrap 与分层分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


AGGREGATIONS = ("max", "top2_mean", "mean")
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260901


def aggregate(values: pd.Series, method: str) -> float:
    scores = np.sort(values.to_numpy(dtype=float))
    if method == "max":
        return float(scores[-1])
    if method == "top2_mean":
        return float(scores[-2:].mean())
    if method == "mean":
        return float(scores.mean())
    raise ValueError(method)


def f1max(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    values = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(values))


def f1max_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.argmax(values))
    return float(values[index]), float(thresholds[index])


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "f1max": f1max(labels, scores),
    }


def macro_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    rows = []
    for _, part in frame.groupby("class_name", sort=True):
        rows.append(metrics(part.sample_label.to_numpy(), part[score_column].to_numpy()))
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "macro_aupr": float(np.mean([row["aupr"] for row in rows])),
        "macro_f1max": float(np.mean([row["f1max"] for row in rows])),
    }


def summarize(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    labels = frame.sample_label.to_numpy()
    scores = frame[score_column].to_numpy()
    result = {f"overall_{key}": value for key, value in metrics(labels, scores).items()}
    result.update(macro_metrics(frame, score_column))
    _, threshold = f1max_threshold(labels, scores)
    result.update({
        "f1max_threshold": threshold,
        "accuracy_at_0.5": float(((scores >= 0.5).astype(int) == labels).mean()),
        "hard_yes_rate_at_0.5": float((scores >= 0.5).mean()),
        "ok_median": float(np.median(scores[labels == 0])),
        "ng_median": float(np.median(scores[labels == 1])),
        "n_samples": len(frame),
    })
    return result


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_samples(views: pd.DataFrame) -> pd.DataFrame:
    keys = ["class_name", "group_path"]
    metadata = (
        views.assign(long_side=views[["width", "height"]].max(axis=1))
        .groupby(keys, as_index=False)
        .agg(
            sample_label=("sample_label", "first"),
            resolution=("long_side", "median"),
            max_defect_ratio=("defect_ratio", "max"),
        )
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    for method in AGGREGATIONS:
        scores = (
            views.groupby(keys, as_index=False)
            .agg(score=("p_yes", lambda values, method=method: aggregate(values, method)))
            .rename(columns={"score": method})
        )
        metadata = metadata.merge(scores, on=keys, validate="one_to_one")
    return metadata


def bootstrap(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [
        part.index.to_numpy()
        for _, part in samples.groupby(["class_name", "sample_label"], sort=True)
    ]
    draws = [
        np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        for _ in range(BOOTSTRAP_REPEATS)
    ]
    records = []
    for repeat, indices in enumerate(draws):
        sampled = samples.loc[indices]
        for method in AGGREGATIONS:
            row = {"repeat": repeat, "aggregation": method}
            row.update(summarize(sampled, method))
            records.append(row)
    raw = pd.DataFrame(records)
    metric_columns = [
        "overall_auroc", "overall_aupr", "overall_f1max",
        "macro_auroc", "macro_aupr", "macro_f1max",
    ]
    summaries = []
    for method, part in raw.groupby("aggregation", sort=False):
        for metric in metric_columns:
            summaries.append({
                "aggregation": method,
                "metric": metric,
                "ci_low": float(np.percentile(part[metric], 2.5)),
                "ci_median": float(np.percentile(part[metric], 50)),
                "ci_high": float(np.percentile(part[metric], 97.5)),
            })
    return raw, pd.DataFrame(summaries)


def paired_deltas(raw: pd.DataFrame) -> pd.DataFrame:
    metrics_to_compare = ["macro_auroc", "macro_aupr", "macro_f1max"]
    rows = []
    base = raw[raw.aggregation.eq("max")].set_index("repeat")
    for method in ("top2_mean", "mean"):
        candidate = raw[raw.aggregation.eq(method)].set_index("repeat")
        for metric in metrics_to_compare:
            delta = candidate[metric] - base[metric]
            rows.append({
                "candidate": method,
                "relative_to": "max",
                "metric": metric,
                "point_delta": np.nan,
                "ci_low": float(np.percentile(delta, 2.5)),
                "ci_median": float(np.percentile(delta, 50)),
                "ci_high": float(np.percentile(delta, 97.5)),
                "probability_positive": float((delta > 0).mean()),
            })
    return pd.DataFrame(rows)


def stratified_tables(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = samples.copy()
    samples["resolution_stratum"] = pd.cut(
        samples.resolution,
        [-np.inf, 640, 1280, np.inf],
        labels=["small<=640", "medium641-1280", "large>1280"],
    )
    resolution_rows = []
    for method in AGGREGATIONS:
        for name, part in samples.groupby("resolution_stratum", observed=True, sort=True):
            row = {"aggregation": method, "resolution_stratum": str(name)}
            row.update(metrics(part.sample_label.to_numpy(), part[method].to_numpy()))
            row.update({
                "n_samples": len(part),
                "n_ok": int((part.sample_label == 0).sum()),
                "n_ng": int((part.sample_label == 1).sum()),
            })
            resolution_rows.append(row)

    ng = samples[samples.sample_label.eq(1)].copy()
    ng["area_stratum"] = pd.cut(
        ng.max_defect_ratio,
        [-np.inf, 0.01, 0.03, np.inf],
        labels=["small<=1%", "medium1-3%", "large>3%"],
    )
    ok = samples[samples.sample_label.eq(0)]
    area_rows = []
    for method in AGGREGATIONS:
        _, threshold = f1max_threshold(samples.sample_label.to_numpy(), samples[method].to_numpy())
        for name, positives in ng.groupby("area_stratum", observed=True, sort=True):
            comparison = pd.concat([ok, positives], ignore_index=True)
            row = {"aggregation": method, "area_stratum": str(name)}
            row.update(metrics(comparison.sample_label.to_numpy(), comparison[method].to_numpy()))
            row.update({
                "n_ng": len(positives),
                "n_ok_reference": len(ok),
                "recall_at_full_f1_threshold": float((positives[method] >= threshold).mean()),
                "full_f1_threshold": threshold,
            })
            area_rows.append(row)
    return pd.DataFrame(resolution_rows), pd.DataFrame(area_rows)


def compare_smoke(samples: pd.DataFrame, views: pd.DataFrame, smoke_path: Path) -> dict | None:
    if not smoke_path.is_file():
        return None
    smoke = pd.read_csv(smoke_path)[["image_path", "p_yes"]]
    overlap = views[["image_path", "p_yes"]].merge(
        smoke, on="image_path", suffixes=("_cuda", "_mps"), validate="one_to_one"
    )
    if overlap.empty:
        return None
    return {
        "n_overlap_views": len(overlap),
        "max_abs_probability_delta": float(
            np.max(np.abs(overlap.p_yes_cuda - overlap.p_yes_mps))
        ),
        "median_abs_probability_delta": float(
            np.median(np.abs(overlap.p_yes_cuda - overlap.p_yes_mps))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/b0plus_dev720"))
    parser.add_argument("--smoke-views", type=Path, default=Path("artifacts/b0plus/P0_views.csv"))
    args = parser.parse_args()
    views = pd.read_csv(args.out_dir / "P0_views.csv")
    if len(views) != 3600 or views.image_path.nunique() != 3600:
        raise RuntimeError("P0_views.csv 不完整")
    samples = build_samples(views)
    if len(samples) != 720 or samples.class_name.nunique() != 18:
        raise RuntimeError("样本聚合结果不完整")
    samples.to_csv(args.out_dir / "sample_scores.csv", index=False)

    overall_rows = []
    class_rows = []
    for method in AGGREGATIONS:
        overall_rows.append({"aggregation": method, **summarize(samples, method)})
        for class_name, part in samples.groupby("class_name", sort=True):
            class_rows.append({
                "aggregation": method,
                "class_name": class_name,
                **metrics(part.sample_label.to_numpy(), part[method].to_numpy()),
                "n_samples": len(part),
            })
    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(class_rows)
    overall.to_csv(args.out_dir / "aggregation_metrics.csv", index=False)
    per_class.to_csv(args.out_dir / "per_class_metrics.csv", index=False)

    raw_bootstrap, ci = bootstrap(samples)
    raw_bootstrap.to_csv(args.out_dir / "bootstrap_raw.csv", index=False)
    ci.to_csv(args.out_dir / "bootstrap_ci.csv", index=False)
    deltas = paired_deltas(raw_bootstrap)
    for index, row in deltas.iterrows():
        candidate = row.candidate
        metric = row.metric
        point = (
            overall.loc[overall.aggregation.eq(candidate), metric].iloc[0]
            - overall.loc[overall.aggregation.eq("max"), metric].iloc[0]
        )
        deltas.loc[index, "point_delta"] = point
    deltas.to_csv(args.out_dir / "aggregation_delta_ci.csv", index=False)

    resolution, area = stratified_tables(samples)
    resolution.to_csv(args.out_dir / "resolution_strata.csv", index=False)
    area.to_csv(args.out_dir / "defect_area_strata.csv", index=False)
    timings = pd.read_csv(args.out_dir / "group_timings.csv")
    decision_path = args.out_dir / "execution_decision.json"
    execution_decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file() else {"execution_mode": "unknown"}
    )
    validation = pd.read_csv(args.out_dir / "cuda_validation.csv")
    latency = {
        "n_groups": len(timings),
        "execution_mode": execution_decision["execution_mode"],
        "total_p50_s": float(np.percentile(timings.total_s, 50)),
        "total_p95_s": float(np.percentile(timings.total_s, 95)),
        "forward_p50_s": float(np.percentile(timings.forward_s, 50)),
        "forward_p95_s": float(np.percentile(timings.forward_s, 95)),
        "formal_single_view_forward_estimate_p50_s": float(
            np.percentile(timings.forward_s / 5, 50)
        ),
        "validation_batch5_forward_s": float(validation.batch5_forward_s.iloc[0]),
    }
    smoke_comparison = compare_smoke(samples, views, args.smoke_views)
    summary = {
        "primary_configuration": "P0 + direct448 + B0f + max",
        "primary_aggregation_remains_frozen": True,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "latency": latency,
        "cuda_equivalence_gate": execution_decision,
        "cross_hardware_smoke_comparison": smoke_comparison,
    }
    (args.out_dir / "analysis_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_display = overall[[
        "aggregation", "overall_auroc", "overall_aupr", "overall_f1max",
        "macro_auroc", "macro_aupr", "macro_f1max", "hard_yes_rate_at_0.5",
    ]]
    ci_display = ci[ci.metric.isin(["macro_auroc", "macro_aupr", "macro_f1max"])]
    delta_display = deltas[deltas.metric.isin(["macro_auroc", "macro_aupr", "macro_f1max"])]
    report = [
        "# B0+ Dev-mini 720 最终统计基线",
        "",
        "冻结配置：Qwen2.5-VL-3B-Instruct + P0 + direct 448 + B0f；主聚合仍为 max。",
        "本次只读取 fold1/dev.parquet，未读取或推理 holdout 样本。",
        "",
        "## 总体与 18 类宏平均",
        "",
        markdown_table(metric_display),
        "",
        "## 2000 次分层 bootstrap 95% 区间",
        "",
        markdown_table(ci_display),
        "",
        "## 相对 max 的配对 bootstrap 差值",
        "",
        markdown_table(delta_display),
        "",
        "## 分辨率分层",
        "",
        markdown_table(resolution),
        "",
        "## NG 最大缺陷面积分层",
        "",
        "每个面积档的正样本与全部 360 个 OK 样本比较；recall 使用全 Dev 的 F1max 阈值。",
        "",
        markdown_table(area),
        "",
        "## 4090D 延迟",
        "",
        f"正式执行模式为 {latency['execution_mode']}。每组 5 视角总延迟 p50/p95 = "
        f"{latency['total_p50_s']:.4f}/{latency['total_p95_s']:.4f}s；5 次逐图 forward "
        f"之和 p50/p95 = {latency['forward_p50_s']:.4f}/"
        f"{latency['forward_p95_s']:.4f}s，折算单视角 forward p50 = "
        f"{latency['formal_single_view_forward_estimate_p50_s']:.4f}s。",
        f"独立的 5 视角 batch prefill forward 测量为 "
        f"{latency['validation_batch5_forward_s']:.4f}s；由于 batch/single 最大分数差 "
        f"{execution_decision['batch_single_max_abs']:.6g} 超过冻结容差 "
        f"{execution_decision['batch_single_atol']:.6g}，它不参与正式打分。",
        "",
    ]
    if smoke_comparison:
        report.extend([
            "## 跨硬件复核",
            "",
            f"与 MPS smoke20 重叠 {smoke_comparison['n_overlap_views']} 个视角；"
            f"P(Yes) 最大/中位绝对差为 "
            f"{smoke_comparison['max_abs_probability_delta']:.6g}/"
            f"{smoke_comparison['median_abs_probability_delta']:.6g}。",
            "",
        ])
    report.extend([
        "## 决策纪律",
        "",
        "max 是预先冻结的主结果；top2_mean 和 mean 是同一批视角分数的离线诊断，"
        "完整保留但不因单次 Dev 结果事后替换主口径。holdout-zs 继续封存。",
        "",
    ])
    (args.out_dir / "DEV720_BASELINE.md").write_text("\n".join(report), encoding="utf-8")
    print(args.out_dir / "DEV720_BASELINE.md")


if __name__ == "__main__":
    main()
