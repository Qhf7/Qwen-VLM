#!/usr/bin/env python3
"""Plot only a complete, frozen 360-prediction Jev development baseline.

This script never calls a model API and does not modify the runner, frozen plan,
contract, predictions, or metrics. It refuses incomplete or mismatched results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent
SHOTS = (0, 1, 4)
MODEL = "jev-1.14.0"
COLORS = {0: "#2469A0", 1: "#D88027", 4: "#19856A"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("All plotting inputs and outputs must stay within the experiment directory")
    return path


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities >= 0.5
    bins = np.minimum((probabilities * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(float(np.mean(labels[bins == b]))
              - float(np.mean(probabilities[bins == b]))) for b in range(10) if np.any(bins == b))
    return {"n": len(labels), "normal": int(np.sum(labels == 0)),
            "anomalous": int(np.sum(labels == 1)),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "ap": float(average_precision_score(labels, probabilities)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", labels=[0, 1], zero_division=0)),
            "recall": float(np.mean(predicted[labels == 1])),
            "fpr": float(np.mean(predicted[labels == 0])),
            "brier": float(brier_score_loss(labels, probabilities)), "ece_10_equal_width": float(ece)}


def compare_metrics(expected: dict, observed: dict, context: str) -> None:
    for key, value in expected.items():
        other = observed.get(key)
        if type(other) not in (float, int) or not math.isclose(float(value), float(other), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Metric mismatch: {context}/{key}")


def load_complete(run_dir: Path, plan_dir: Path) -> dict:
    # Read the completion gate before loading any metrics or prediction values.
    status = read_json(run_dir / "status.json")
    if (status.get("status") != "complete" or status.get("completed") != 360
            or status.get("expected") != 360 or status.get("mode") != "formal_development_baseline"):
        raise ValueError("Refusing to plot: formal baseline status is not complete with 360 predictions")
    summary = read_json(run_dir / "metrics.json")
    if summary.get("status") != "complete" or summary.get("model") != MODEL:
        raise ValueError("Refusing to plot incomplete or mismatched metrics")
    config = read_json(plan_dir / "config.json")
    plan = read_json(plan_dir / "plan.json")
    labels = read_json(plan_dir / "labels.json")
    contract = read_json(run_dir / "contract.json")
    if config.get("size") != 448 or config.get("shots") != list(SHOTS) or config.get("model") != MODEL:
        raise ValueError("Expected frozen 448-pixel Jev 0/1/4-shot protocol")
    if config.get("plan_sha256") != canonical_sha(plan) or config.get("labels_sha256") != canonical_sha(labels):
        raise ValueError("Frozen plan or label hash mismatch")
    if config.get("runner_sha256") != file_sha(ROOT / "runner.py"):
        raise ValueError("Runner differs from frozen version")
    if (contract.get("mode") != "formal_development_baseline" or contract.get("limit_queries") is not None
            or contract.get("config_sha256") != canonical_sha(config)
            or contract.get("runner_sha256") != config["runner_sha256"]):
        raise ValueError("Formal run contract mismatch")
    materialized = read_json(plan_dir / "materialized.json")
    if (materialized.get("status") != "complete" or materialized.get("failures")
            or materialized.get("config_sha256") != canonical_sha(config)
            or contract.get("materialized_records_sha256") != canonical_sha(materialized["records"])):
        raise ValueError("Materialized image provenance mismatch")
    queries = plan["queries"]
    query_ids = [row["source_id"] for row in queries]
    if len(queries) != 120 or len(set(query_ids)) != 120 or set(labels) != set(query_ids):
        raise ValueError("Expected 120 unique frozen queries and their exact label map")
    if summary.get("data_role") != plan.get("data_role") or summary.get("threshold") != 0.5:
        raise ValueError("Metric data role or threshold mismatch")
    y = np.asarray([labels[key] for key in query_ids], dtype=int)
    if not np.isin(y, [0, 1]).all() or int(np.sum(y)) != 60:
        raise ValueError("Expected 60 normal and 60 anomalous development queries")
    categories = [row["category"] for row in queries]
    if len(set(categories)) != 12:
        raise ValueError("Expected 12 frozen categories")
    schedule = [{"source_id": sid, "shots": shots} for sid in query_ids for shots in SHOTS]
    if contract.get("schedule") != schedule:
        raise ValueError("Contract does not contain the complete fixed 360-query schedule")
    predictions_path = run_dir / "predictions.jsonl"
    if summary.get("prediction_file_sha256") != file_sha(predictions_path):
        raise ValueError("Prediction hash differs from the completed metric artifact")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 360:
        raise ValueError("Exactly 360 prediction rows are required")
    expected_pairs = {(sid, s) for sid in query_ids for s in SHOTS}
    by_id = {}
    cat_by_id = dict(zip(query_ids, categories))
    contract_sha = canonical_sha(contract)
    for row in rows:
        pair = row.get("source_id"), row.get("shots")
        if pair not in expected_pairs or pair in by_id:
            raise ValueError("Unexpected or duplicate prediction identity")
        if row.get("contract_sha256") != contract_sha or row.get("response_model") != MODEL:
            raise ValueError("Prediction contract or model mismatch")
        if row.get("category") != cat_by_id[pair[0]]:
            raise ValueError("Prediction category mismatch")
        probability = row.get("probability")
        if type(probability) not in (float, int) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Prediction probability is invalid")
        by_id[pair] = row
    if set(by_id) != expected_pairs:
        raise ValueError("Formal prediction coverage is incomplete")
    probabilities = {s: np.asarray([by_id[sid, s]["probability"] for sid in query_ids]) for s in SHOTS}
    if set(summary.get("shots", {})) != {str(s) for s in SHOTS}:
        raise ValueError("Metric shot coverage mismatch")
    for shots in SHOTS:
        compare_metrics(compute_metrics(y, probabilities[shots]), summary["shots"][str(shots)], f"shots={shots}")
    for category in sorted(set(categories)):
        indices = np.asarray([i for i, name in enumerate(categories) if name == category])
        if len(indices) != 10 or np.sum(y[indices]) != 5:
            raise ValueError("Each category must contain 5 normal and 5 anomalous queries")
        for shots in SHOTS:
            compare_metrics(compute_metrics(y[indices], probabilities[shots][indices]),
                            summary["by_category"][category][str(shots)], f"category={category},shots={shots}")
    return {"summary": summary, "labels": y, "probabilities": probabilities,
            "categories": categories, "config": config, "contract": contract,
            "prediction_sha256": file_sha(predictions_path)}


def render(data: dict, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 8.5, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    summary, y, probabilities = data["summary"], data["labels"], data["probabilities"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
    fig.suptitle("Jev 1.14.0 | 448 x 448 | 0 / 1 / 4 normal reference images", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.947, "Same 120 VisA development queries: 60 normal + 60 anomalous; 360 completed predictions", ha="center", fontsize=11, color="#444444")
    ax = axes[0, 0]
    x = np.arange(3)
    for offset, key, title, alpha in ((-0.18, "auroc", "AUROC", 1.0), (0.18, "ap", "AP", 0.58)):
        values = [summary["shots"][str(s)][key] for s in SHOTS]
        bars = ax.bar(x + offset, values, width=0.34, color=[COLORS[s] for s in SHOTS], alpha=alpha, label=title)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=[f"{s}-shot" for s in SHOTS], ylim=(0, 1.12), ylabel="Ranking metric", title="A  |  AUROC and average precision")
    ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc="lower right")
    ax = axes[0, 1]
    for s in SHOTS:
        fpr, tpr, _ = roc_curve(y, probabilities[s], drop_intermediate=False)
        ax.plot(fpr, tpr, color=COLORS[s], lw=2, label=f"{s}-shot (AUC {summary['shots'][str(s)]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#909090", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate", title="B  |  ROC curves"); ax.legend(loc="lower right")
    ax = axes[0, 2]
    for s in SHOTS:
        precision, recall, _ = precision_recall_curve(y, probabilities[s])
        ax.step(recall, precision, where="post", color=COLORS[s], lw=2, label=f"{s}-shot (AP {summary['shots'][str(s)]['ap']:.3f})")
    ax.axhline(float(np.mean(y)), ls="--", color="#909090", lw=1, label="Anomaly prevalence")
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel="Recall", ylabel="Precision", title="C  |  Precision-recall curves"); ax.legend(loc="lower left")
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], ls="--", color="#909090", lw=1, label="Perfect calibration")
    for s in SHOTS:
        p = probabilities[s]; bins = np.minimum((p * 10).astype(int), 9)
        occupied = [b for b in range(10) if np.any(bins == b)]
        confidence = [float(np.mean(p[bins == b])) for b in occupied]
        observed = [float(np.mean(y[bins == b])) for b in occupied]
        counts = [int(np.sum(bins == b)) for b in occupied]
        ece = summary["shots"][str(s)]["ece_10_equal_width"]; brier = summary["shots"][str(s)]["brier"]
        ax.plot(confidence, observed, color=COLORS[s], lw=1.6, alpha=0.9, label=f"{s}-shot | ECE {ece:.3f}, Brier {brier:.3f}")
        ax.scatter(confidence, observed, s=[22 + 2.2 * n for n in counts], color=COLORS[s], edgecolors="white", linewidths=0.8, zorder=3, alpha=0.85)
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel="Mean predicted anomaly probability", ylabel="Observed anomaly fraction", title="D  |  Reliability (10 equal-width bins)"); ax.legend(loc="upper left", fontsize=7.5)
    ax.text(0.98, 0.03, "Marker area increases with bin count", transform=ax.transAxes, ha="right", fontsize=7.5, color="#555555")
    categories = sorted(set(data["categories"]))
    x = np.arange(len(categories))
    for ax, key, title in ((axes[1, 1], "recall", "E  |  Recall by object category"),
                           (axes[1, 2], "fpr", "F  |  False-positive rate by category")):
        for offset, s in zip((-0.25, 0, 0.25), SHOTS):
            values = [summary["by_category"][category][str(s)][key] for category in categories]
            ax.bar(x + offset, values, width=0.23, color=COLORS[s], label=f"{s}-shot")
        ax.set(xticks=x, xticklabels=categories, ylim=(0, 1.04), ylabel="Rate at probability >= 0.5", title=title)
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        for tick in ax.get_xticklabels(): tick.set_ha("right")
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(loc="upper right", ncol=3, fontsize=7.5)
    for ax in axes.flat:
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.18, color="#7A8694", linewidth=0.6)
    fig.text(0.5, 0.018, "Development validation, not a new blind test. Fixed threshold: 0.5. Per-category denominators: 5 anomalies / 5 normals.", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), h_pad=2.7, w_pad=2.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        destination = output_dir / ("jev_448_0_1_4_comparison." + suffix)
        fd, temporary = tempfile.mkstemp(prefix=".jev_plot_", suffix="." + suffix, dir=output_dir)
        os.close(fd)
        try:
            fig.savefig(temporary, format=suffix, dpi=220, facecolor="white", metadata={"Creator": "Qwen-VLM plot_results.py"})
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        outputs.append(destination)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "modelgo_448")
    parser.add_argument("--plan-dir", type=Path, default=ROOT / "frozen")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    run_dir, plan_dir, output_dir = (contained(path, ROOT) for path in (args.run_dir, args.plan_dir, args.output_dir))
    data = load_complete(run_dir, plan_dir)
    outputs = render(data, output_dir)
    print(json.dumps({"status": "complete", "unique_predictions": 360,
                      "prediction_sha256": data["prediction_sha256"],
                      "figures": [str(path) for path in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
