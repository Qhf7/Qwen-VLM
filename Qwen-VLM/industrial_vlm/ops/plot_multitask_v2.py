"""Render completed multitask-v2 validation and training evidence as static figures.

Example: python ops/plot_multitask_v2.py --root /root/autodl-tmp/industrial_vlm
All five validation evaluations and all three 192-update runs must be complete.
The script reads JSON/JSONL only; missing observations are never zero-filled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = {
    "region_region": {"label": "RR: region + region", "short": "RR", "color": "#0072B2"},
    "decision_decision": {"label": "DD: decision + decision", "short": "DD", "color": "#009E73"},
    "region_decision": {"label": "RD: region + decision", "short": "RD", "color": "#D55E00"},
}
EVALUATIONS = {
    "region_region_region": ("region_region", "region"),
    "region_region_decision": ("region_region", "decision"),
    "decision_decision_decision": ("decision_decision", "decision"),
    "region_decision_region": ("region_decision", "region"),
    "region_decision_decision": ("region_decision", "decision"),
}
N_VALIDATION = 120
FINAL_UPDATE = 192


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, context: str, *, minimum: float = 0.0) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value >= minimum, f"Invalid observed number at {context}: {value!r}")
    return float(value)


def _rate(value: Any, context: str) -> float:
    result = _number(value, context)
    _require(result <= 1.0, f"Rate outside [0,1] at {context}: {result}")
    return result


def _jsonl(path: Path) -> list[dict]:
    _require(path.is_file(), f"Missing observed input: {path}; rerun after the pipeline completes")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), f"Expected JSON objects in {path}")
    return rows


def _valid_decision(row: dict) -> bool | None:
    parsed = row.get("parsed")
    value = parsed.get("is_anomalous") if isinstance(parsed, dict) else None
    return value if row.get("parse_error") is None and isinstance(value, bool) else None


def load_observed(root: Path) -> tuple[dict, list[dict]]:
    """Load complete observations and verify paired-source validation conditions."""
    inputs: set[Path] = {Path(__file__).resolve()}
    evaluation_data, prediction_maps = {}, {}
    source_set, ground_truth = None, None
    for name, (arm, task) in EVALUATIONS.items():
        folder = root / "evaluations/multitask_v2" / name
        metrics_path, predictions_path = folder / "metrics.json", folder / "predictions.jsonl"
        _require(metrics_path.is_file(), f"Missing observed input: {metrics_path}; rerun after the pipeline completes")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        overall, metadata = payload["overall"], payload["metadata"]
        _require(metadata.get("complete") is True, f"Incomplete evaluation: {name}")
        _require(overall["n"] == metadata["n_expected"] == N_VALIDATION, f"Expected complete val120: {name}")
        _require(overall["n_sources"] == N_VALIDATION and overall["n_normal"] == overall["n_anomalous"] == 60,
                 f"Unexpected validation population: {name}")
        _require(Path(metadata["data"]).name == f"val_{task}_120.jsonl", f"Not the declared validation prompt data: {name}")
        _require(metadata.get("max_pixels") == 524288 and metadata.get("max_length") == 2048,
                 f"Unexpected input budget: {name}")
        _require(metadata.get("generation", {}).get("do_sample") is False, f"Expected greedy evaluation: {name}")
        rows = _jsonl(predictions_path)
        by_source = {row["source_id"]: row for row in rows}
        _require(len(rows) == len(by_source) == N_VALIDATION, f"Duplicate/missing validation source: {name}")
        _require(all(isinstance(row.get("label"), bool) for row in rows), f"Invalid ground-truth labels: {name}")
        labels = {sid: row["label"] for sid, row in by_source.items()}
        if source_set is None:
            source_set, ground_truth = set(by_source), labels
        _require(set(by_source) == source_set and labels == ground_truth, f"Validation sources/labels differ: {name}")
        _require(sum(_valid_decision(row) is not None for row in rows) == overall["n_valid"],
                 f"Prediction validity disagrees with reported metrics: {name}")
        keys = ["macro_f1", "anomaly_recall", "fpr", "format_valid_rate"]
        if task == "region":
            keys.append("union_box_recall_iou_0.5")
        values = {key: _rate(overall[key], f"{name}:{key}") for key in keys}
        intervals = {}
        for key in keys:
            interval = payload.get("bootstrap", {}).get("intervals", {}).get(key)
            if interval is not None:
                low, high = _rate(interval["low"], f"{name}:{key}:low"), _rate(interval["high"], f"{name}:{key}:high")
                _require(low <= high, f"Reversed bootstrap interval: {name}:{key}")
                intervals[key] = {"low": low, "high": high}
        evaluation_data[name] = {
            "arm": arm, "task": task, "n": overall["n"], "n_normal": overall["n_normal"],
            "n_anomalous": overall["n_anomalous"], "values": values, "intervals": intervals,
            "bootstrap": {key: payload.get("bootstrap", {}).get(key) for key in ("unit", "method", "seed", "n_resamples")},
            "metadata": metadata,
        }
        prediction_maps[name] = by_source
        inputs.update((metrics_path, predictions_path))
    for key in ("model", "max_pixels", "max_length", "max_new_tokens", "generation", "quantization"):
        conditions = {json.dumps(item["metadata"].get(key), sort_keys=True) for item in evaluation_data.values()}
        _require(len(conditions) == 1, f"Unmatched evaluation condition: {key}")
    for task in ("region", "decision"):
        hashes = {item["metadata"]["data_sha256"] for item in evaluation_data.values() if item["task"] == task}
        _require(len(hashes) == 1, f"Different validation data hashes under {task} prompt")

    consistency = {}
    for arm in ("region_region", "region_decision"):
        region, decision = prediction_maps[arm + "_region"], prediction_maps[arm + "_decision"]
        counts = Counter({"valid_agreement": 0, "valid_disagreement": 0, "any_invalid": 0})
        invalid_region = invalid_decision = invalid_both = 0
        paired = []
        for source_id in sorted(region):
            first, second = _valid_decision(region[source_id]), _valid_decision(decision[source_id])
            invalid_region += first is None
            invalid_decision += second is None
            invalid_both += first is None and second is None
            status = "any_invalid" if first is None or second is None else "valid_agreement" if first == second else "valid_disagreement"
            counts[status] += 1
            paired.append({"source_id": source_id, "label": region[source_id]["label"],
                           "region_decision": first, "decision_decision": second, "status": status})
        consistency[arm] = {"n": N_VALIDATION, "counts": dict(counts),
                            "fractions_all_sources": {key: count / N_VALIDATION for key, count in counts.items()},
                            "invalid_region": invalid_region, "invalid_decision": invalid_decision,
                            "invalid_both": invalid_both, "source_pairs": paired}

    training = {}
    for arm in ARMS:
        path = root / "runs/multitask_v2" / arm / "metrics.jsonl"
        events = _jsonl(path)
        steps = [event for event in events if event.get("event") == "train"]
        completed = [event for event in events if event.get("event") == "complete"]
        _require(len(completed) == 1 and completed[0].get("status") == "complete", f"Training not complete: {arm}")
        final = completed[0]
        _require(final["optimizer_steps"] == FINAL_UPDATE, f"Unexpected completed update count: {arm}")
        _require([event["optimizer_step"] for event in steps] == list(range(1, FINAL_UPDATE + 1)),
                 f"Training steps missing/duplicated/out of order: {arm}")
        train_series, validation_series = [], []
        for event in steps:
            train_series.append({"step": event["optimizer_step"], "loss": _number(event["loss"], arm + ":train_loss"),
                                 "target_tokens": int(_number(event["target_tokens"], arm + ":target_tokens")),
                                 "input_tokens": int(_number(event["input_tokens"], arm + ":input_tokens")),
                                 "peak_allocated_gib": _number(event["peak_allocated_gib"], arm + ":peak_allocated_gib")})
        for event in events:
            if event.get("event") == "validation":
                validation_series.append({"step": event["optimizer_step"],
                                          "loss": _number(event["validation_loss"], arm + ":validation_loss"),
                                          "samples": event["validation_samples"],
                                          "target_tokens": event["validation_target_tokens"]})
        _require([event["step"] for event in validation_series] == [64, 128, 192]
                 and all(event["samples"] == 24 for event in validation_series), f"Unexpected validation-loss audit: {arm}")
        total_target = sum(event["target_tokens"] for event in train_series)
        total_input = sum(event["input_tokens"] for event in train_series)
        _require(total_target == final["total_target_tokens"] and total_input == final["total_input_tokens"],
                 f"Logged per-update tokens disagree with completion event: {arm}")
        peak = _number(final["peak_allocated_gib"], arm + ":complete_peak_allocated_gib")
        _require(peak + 1e-6 >= max(event["peak_allocated_gib"] for event in train_series), f"Completion peak memory lower than logged update peak: {arm}")
        training[arm] = {"optimizer_steps": FINAL_UPDATE, "train": train_series, "validation": validation_series,
                         "total_target_tokens": total_target, "total_input_tokens": total_input,
                         "peak_allocated_gib": peak,
                         "train_wall_seconds_including_validation": _number(final["train_wall_seconds_including_validation"], arm + ":wall_time")}
        inputs.add(path)
    hashes = []
    for path in sorted(inputs):
        content = path.read_bytes()
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        hashes.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    return {
        "schema_version": "industrial-vlm-multitask-v2-figures-v1",
        "population": "Fixed VisA validation120: 12 categories x2 labels x5 images; one training seed; final update192",
        "definitions": {
            "macro_f1": "Mean of normal-class and anomaly-class F1, not a mean across categories",
            "box_recall": "Union-rectangle recall at IoU>=0.5 over all60 anomalous sources, including detection/format failures",
            "prompt_consistency": "Both prompts valid and predict the same boolean / all120 paired sources; any invalid output is a separate retained status",
            "error_bars": "Observed source-bootstrap metric intervals from evaluation metrics, when present; pointwise intervals are not paired-difference evidence",
            "ce": "Raw logged assistant-only cross-entropy. Target tasks/token budgets differ; CE is not directly comparable across tasks",
            "memory": "Observed maximum allocated CUDA memory in training completion event, in GiB; not reserved VRAM or device-wide usage",
        },
        "evaluations": evaluation_data, "prompt_consistency": consistency, "training": training,
    }, hashes


def _style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "grid.linewidth": 0.5, "grid.alpha": 0.25,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "industrial-vlm-multitask-v2", "savefig.facecolor": "white",
    })


def _save(fig: Any, output: Path, name: str) -> list[str]:
    import matplotlib.pyplot as plt
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = output / (name + "." + suffix)
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {}
        # Preserve exactly 12x7 inches; layout reserves space for all captions.
        fig.savefig(path, dpi=300, metadata=metadata)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _metric_bar(ax: Any, x: float, run: dict, metric: str, color: str, width: float) -> None:
    value = 100 * run["values"][metric]
    ax.bar(x, value, width=width, color=color, edgecolor="white", linewidth=0.5, zorder=2)
    interval = run["intervals"].get(metric)
    top = value
    if interval is not None:
        low, high = 100 * interval["low"], 100 * interval["high"]
        ax.vlines(x, low, high, color="#333333", linewidth=0.8, zorder=3)
        ax.hlines([low, high], x - width * 0.12, x + width * 0.12, color="#333333", linewidth=0.8, zorder=3)
        top = max(top, high)
    ax.text(x, top + 1.3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)


def plot_validation(data: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1), left=0.075, right=0.975,
                            bottom=0.20, top=0.83, hspace=0.53, wspace=0.25)
    common, region, consistency = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    runs = data["evaluations"]
    metrics = (("macro_f1", "Macro F1"), ("anomaly_recall", "Anomaly recall"), ("fpr", "Normal FPR (lower is better)"))
    width = 0.20
    for arm_index, (arm, config) in enumerate(ARMS.items()):
        for metric_index, (metric, _) in enumerate(metrics):
            _metric_bar(common, metric_index + (arm_index - 1) * width, runs[arm + "_decision"], metric, config["color"], width)
    common.set_xticks(range(3), [label for _, label in metrics])
    common.set_title("(a) Common decision prompt | validation120", loc="left", pad=9)
    for arm_index, arm in enumerate(("region_region", "region_decision")):
        config = ARMS[arm]
        for index, metric in enumerate(("macro_f1", "union_box_recall_iou_0.5")):
            _metric_bar(region, index + (arm_index - 0.5) * 0.26, runs[arm + "_region"], metric, config["color"], 0.26)
    region.set_xticks((0, 1), ("Macro F1", "Box recall @ IoU 0.5"))
    region.set_title("(b) Region prompt | validation120", loc="left", pad=9)
    for ax in (common, region):
        ax.set_ylim(0, 113)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Percent (%)")
        ax.grid(axis="y", zorder=0)
    statuses = (("valid_agreement", "Valid agreement", "#4C78A8"),
                ("valid_disagreement", "Valid disagreement", "#E69F00"),
                ("any_invalid", "Any invalid output", "#8C8C8C"))
    for index, arm in enumerate(("region_region", "region_decision")):
        bottom = 0.0
        item = data["prompt_consistency"][arm]
        for key, _, color in statuses:
            height = 100 * item["fractions_all_sources"][key]
            consistency.barh(index, height, left=bottom, height=0.48, color=color, edgecolor="white", linewidth=0.5)
            if height >= 9:
                consistency.text(bottom + height / 2, index, f"{item['counts'][key]}/120", ha="center", va="center", fontsize=8,
                                 color="white" if key == "valid_agreement" else "#222222")
            bottom += height
    consistency.set_title("(c) Agreement between prompts | validation120", loc="left", pad=9)
    consistency.set_yticks((0, 1), ("RR", "RD"))
    consistency.invert_yaxis()
    consistency.set_xlim(0, 100)
    consistency.set_xlabel("All paired sources (%)")
    consistency.grid(axis="x", zorder=0)
    consistency.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in statuses],
                       loc="upper center", bbox_to_anchor=(0.48, -0.27), ncol=3, frameon=False, fontsize=7.5,
                       handlelength=1.0, columnspacing=0.8)
    fig.suptitle("Multitask SFT: decision, localization, and prompt consistency", y=0.97, fontsize=14)
    fig.text(0.5, 0.925, "Fixed validation sources | final update 192 | 1,536 matched source exposures per arm | one seed", ha="center", fontsize=10, color="#444444")
    fig.legend(handles=[Patch(facecolor=config["color"], label=config["label"]) for config in ARMS.values()],
               loc="upper center", bbox_to_anchor=(0.5, 0.907), ncol=3, frameon=False)
    fig.text(0.075, 0.042, "Pointwise source-bootstrap intervals where available; localization denominator: all 60 anomalies.\n"
             "Prompt agreement retains all 120 sources: any invalid prediction is separate. Agreement does not imply correctness.",
             ha="left", va="bottom", fontsize=8, color="#444444", linespacing=1.5)
    return _save(fig, output, "validation_comparison")


def plot_training(data: dict, output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.13, top=0.80, hspace=0.43, wspace=0.25)
    train_ax, val_ax, token_ax, memory_ax = axes.flat
    for index, (arm, config) in enumerate(ARMS.items()):
        item = data["training"][arm]
        for ax, key in ((train_ax, "train"), (val_ax, "validation")):
            series = item[key]
            ax.plot([row["step"] for row in series], [row["loss"] for row in series],
                    color=config["color"], linewidth=1.0 if key == "train" else 1.5,
                    marker="o" if key == "validation" else None, markersize=4,
                    alpha=0.8 if key == "train" else 1.0, label=config["label"])
        for ax, value, label in (
            (token_ax, item["total_target_tokens"] / 1000, f"{item['total_target_tokens']:,}"),
            (memory_ax, item["peak_allocated_gib"], f"{item['peak_allocated_gib']:.3f}"),
        ):
            ax.bar(index, value, width=0.56, color=config["color"], edgecolor="white", linewidth=0.5, zorder=2)
            ax.annotate(label, (index, value), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for ax, title in ((train_ax, "(a) Raw training assistant CE"), (val_ax, "(b) Validation assistant CE | fixed first 24")):
        ax.set_title(title, loc="left", pad=9)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Cross-entropy")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
    train_ax.set_xlim(1, FINAL_UPDATE)
    val_ax.set_xticks((64, 128, 192))
    token_ax.set_title("(c) Actual supervised target tokens", loc="left", pad=9)
    token_ax.set_ylabel("Thousand tokens")
    memory_ax.set_title("(d) Peak allocated training GPU memory", loc="left", pad=9)
    memory_ax.set_ylabel("GiB")
    for ax in (token_ax, memory_ax):
        ax.set_xticks(range(3), [config["short"] for config in ARMS.values()])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.grid(axis="y", zorder=0)
    fig.suptitle("Training dynamics and observed resource use", y=0.97, fontsize=14)
    fig.text(0.5, 0.925, "Task-specific CE: values across different target tasks are not directly comparable", ha="center", fontsize=10, color="#7A3E00")
    handles, labels = train_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=3, frameon=False)
    fig.text(0.075, 0.04, "All arms: fresh base + LoRA, 192 updates, accumulation 8, same 768 sources exposed twice, one seed.\n"
             "Validation CE uses region targets for RR/RD and decision targets for DD. Token counts are measured; budgets differ.",
             ha="left", va="bottom", fontsize=8, color="#444444", linespacing=1.5)
    return _save(fig, output, "training_dynamics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: ROOT/reports/multitask_v2/figures")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "reports/multitask_v2/figures"
    data, hashes = load_observed(root)
    _style()
    output.mkdir(parents=True, exist_ok=True)
    paths = plot_validation(data, output) + plot_training(data, output)
    (output / "figure_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "input_hashes.json").write_text(json.dumps({"schema_version": "industrial-vlm-figure-inputs-v1", "inputs": hashes,
                                                         "figures": paths, "figure_size_inches": [12, 7], "png_dpi": 300,
                                                         "confirmation_data_read": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "figures": paths, "observed_inputs": len(hashes),
                      "figure_data": str(output / "figure_data.json"), "input_hashes": str(output / "input_hashes.json")}, indent=2))


if __name__ == "__main__":
    main()
