"""Paired source-image bootstrap comparisons for the preregistered pilot.

Run after evaluation: python -m industrial_vlm.compare --evaluations evaluations
--output reports. No model, CUDA, or third-party numerical library is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import SCHEMA_VERSION, _summary, parse_prediction, validate_boxes

PAIRS = (
    ("quality_region", "base_region", "base_vs_sft"),
    ("quality_region", "random_region", "dataquality_sampling_ablation"),
    ("quality_region_decision_prompt", "quality_decision", "region_supervision_ablation"),
    ("external_quality_region", "external_base_region", "external_pilot"),
)
METRICS = ("macro_f1", "balanced_accuracy", "fpr", "fnr", "union_box_recall_iou_0.5")
REQUIRED_CONDITIONS = ("model", "max_pixels", "max_length", "max_new_tokens", "generation", "quantization")
OPTIONAL_CONDITIONS = ("base_config_sha256", "limit")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_run(root: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    directory = root / name
    paths = {key: directory / filename for key, filename in (
        ("meta", "meta.json"), ("metrics", "metrics.json"), ("predictions", "predictions.jsonl"))}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return None, "missing files: " + ", ".join(missing)
    meta, metrics = _read_json(paths["meta"]), _read_json(paths["metrics"])
    if meta.get("status") != "complete" or metrics.get("metadata", {}).get("complete") is not True:
        return None, "evaluation is not complete in both meta.json and metrics.json"
    if meta.get("schema_version") != SCHEMA_VERSION or metrics.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name}: unsupported or inconsistent evaluation schema")
    config = meta.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{name}: missing evaluation config")
    metrics_config = {k: v for k, v in metrics["metadata"].items() if k != "complete"}
    if metrics_config != config:
        raise ValueError(f"{name}: metadata and metrics configs differ; stale metrics cannot be paired")
    for key in REQUIRED_CONDITIONS:
        if key not in config or config[key] is None:
            raise ValueError(f"{name}: missing comparison condition {key}")
    rows, ids = [], set()
    with paths["predictions"].open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{name}:{line_number}: unsupported prediction schema")
            if not isinstance(row.get("id"), str) or row["id"] in ids:
                raise ValueError(f"{name}:{line_number}: missing or duplicate id")
            ids.add(row["id"])
            if not isinstance(row.get("source_id"), str) or not row["source_id"]:
                raise ValueError(f"{name}:{line_number}: source_id is required")
            if type(row.get("label")) is not bool or type(row.get("localization_eligible")) is not bool:
                raise ValueError(f"{name}:{line_number}: label and localization_eligible must be booleans")
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"{name}:{line_number}: prompt is required")
            validate_boxes(row.get("gt_boxes"))
            if not row["label"] and row["gt_boxes"]:
                raise ValueError(f"{name}:{line_number}: normal GT cannot have anomaly boxes")
            if row["label"] and row["localization_eligible"] and not row["gt_boxes"]:
                raise ValueError(f"{name}:{line_number}: anomalous region GT requires boxes")
            if "parsed" not in row:
                raise ValueError(f"{name}:{line_number}: parsed field is required, including null for failures")
            if row["parsed"] is not None:
                _, error = parse_prediction(json.dumps(row["parsed"]), row["localization_eligible"])
                if error:
                    raise ValueError(f"{name}:{line_number}: invalid parsed prediction: {error}")
            rows.append(row)
    expected = config.get("n_expected")
    if not rows or any(value != len(rows) for value in (
        expected, meta.get("n_completed"), metrics.get("overall", {}).get("n"))):
        raise ValueError(f"{name}: prediction count disagrees with completion metadata")
    return {"name": name, "directory": str(directory.resolve()), "config": config, "rows": rows}, None


def _align(left: dict[str, Any], right: dict[str, Any]) -> tuple[list[str], dict[str, list[tuple[dict, dict]]]]:
    for key in REQUIRED_CONDITIONS + OPTIONAL_CONDITIONS:
        if left["config"].get(key) != right["config"].get(key):
            raise ValueError(f"{left['name']} vs {right['name']}: mismatched {key}")
    left_rows = {row["id"]: row for row in left["rows"]}
    right_rows = {row["id"]: row for row in right["rows"]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"{left['name']} vs {right['name']}: task ID sets differ")
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    fields = ("source_id", "label", "gt_boxes", "category", "prompt", "localization_eligible", "anomaly_area_fraction")
    for identifier in sorted(left_rows):
        a, b = left_rows[identifier], right_rows[identifier]
        for key in fields:
            if a.get(key) != b.get(key):
                raise ValueError(f"{left['name']} vs {right['name']}: {identifier} mismatched {key}")
        groups[a["source_id"]].append((a, b))
    return sorted(groups), dict(groups)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_comparison(left: dict[str, Any], right: dict[str, Any], *, samples: int = 2000, seed: int = 17) -> dict[str, Any]:
    """Estimate left-minus-right differences, resampling identical source clusters."""
    if samples < 20:
        raise ValueError("At least 20 bootstrap samples are required")
    sources, grouped = _align(left, right)
    left_point, right_point = _summary(left["rows"]), _summary(right["rows"])
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {key: [] for key in METRICS}
    for _ in range(samples):
        pairs = [pair for _ in sources for pair in grouped[rng.choice(sources)]]
        a, b = _summary([pair[0] for pair in pairs]), _summary([pair[1] for pair in pairs])
        for key in METRICS:
            if a[key] is not None and b[key] is not None:
                distributions[key].append(a[key] - b[key])
    estimates = {}
    for key in METRICS:
        values = sorted(distributions[key])
        defined = left_point[key] is not None and right_point[key] is not None
        estimates[key] = {
            "left": left_point[key], "right": right_point[key],
            "difference": left_point[key] - right_point[key] if defined else None,
            "ci95_low": _quantile(values, 0.025), "ci95_high": _quantile(values, 0.975),
            "n_defined_resamples": len(values),
            "preferred_direction": "negative" if key in {"fpr", "fnr"} else "positive",
        }
    return {
        "left": left["name"], "right": right["name"],
        "left_directory": left["directory"], "right_directory": right["directory"],
        "n_rows": len(left["rows"]), "n_sources": len(sources), "n_resamples": samples, "seed": seed,
        "comparison_conditions": {key: left["config"].get(key) for key in REQUIRED_CONDITIONS + OPTIONAL_CONDITIONS},
        "left_adapter": left["config"].get("adapter"), "right_adapter": right["config"].get("adapter"),
        "left_adapter_files_sha256": left["config"].get("adapter_files_sha256"),
        "right_adapter_files_sha256": right["config"].get("adapter_files_sha256"),
        "metrics": estimates,
    }


def build_comparisons(evaluations: Path, *, samples: int = 2000, seed: int = 17) -> dict[str, Any]:
    loaded: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    comparisons, missing = [], []
    for left_name, right_name, branch in PAIRS:
        for name in (left_name, right_name):
            if name not in loaded:
                loaded[name] = _load_run(evaluations, name)
        absent = {name: loaded[name][1] for name in (left_name, right_name) if loaded[name][0] is None}
        if absent:
            missing.append({"left": left_name, "right": right_name, "branch": branch, "reason": absent})
            continue
        comparison = paired_comparison(loaded[left_name][0], loaded[right_name][0], samples=samples, seed=seed)
        comparisons.append({"branch": branch, **comparison})
    return {
        "schema_version": "industrial-vlm-paired-difference-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "partial",
        "method": "paired source_id cluster bootstrap; identical sampled sources and task multiplicities in both runs; percentile 95% intervals with linear interpolation",
        "difference_direction": "left minus right; values stored as fractions, not percentage points",
        "n_resamples": samples, "seed": seed, "comparisons": comparisons, "missing": missing,
        "limitations": [
            "One training seed and fixed small pilot test sets; intervals quantify source-sampling uncertainty conditional on these trained models.",
            "Intervals do not include training-seed variation, dataset shift, or benchmark pretraining contamination.",
            "Four planned comparisons and multiple metrics are descriptive; no multiplicity correction or SOTA claim is made.",
            "macro_f1 averages the normal/anomalous label F1 values, not product-category F1 values.",
            "Format failures remain in metric denominators under the shared evaluation policy. Decision-only localization is undefined, not zero.",
            "Bootstrap draws lacking a required label yield undefined rates; those draws are omitted for that metric and their defined counts are reported.",
            "Model path and base config hash are checked; this is not a hash audit of all base weight shards.",
        ],
    }


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}" if signed else f"{value * 100:.2f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Paired source-image bootstrap differences", "",
             "All differences are left minus right in percentage points (pp). Lower FPR/FNR is better; higher F1, balanced accuracy, and union-box recall is better.", "",
             f"Bootstrap: {result['n_resamples']} paired source-cluster draws, seed {result['seed']}; percentile 95% intervals.", "",
             "| Left − right | Metric | Left % | Right % | Difference pp | 95% CI pp | Defined draws |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for comparison in result["comparisons"]:
        pair = f"{comparison['left']} − {comparison['right']}"
        for metric, estimate in comparison["metrics"].items():
            interval = f"[{_percent(estimate['ci95_low'], signed=True)}, {_percent(estimate['ci95_high'], signed=True)}]"
            if estimate["ci95_low"] is None:
                interval = "—"
            lines.append(f"| {pair} | {metric} | {_percent(estimate['left'])} | {_percent(estimate['right'])} | "
                         f"{_percent(estimate['difference'], signed=True)} | {interval} | {estimate['n_defined_resamples']} |")
    if not result["comparisons"]:
        lines.extend(["", "No complete matched pairs are available."])
    if result["missing"]:
        lines.extend(["", "## Missing pairs", ""])
        for item in result["missing"]:
            lines.append(f"- {item['left']} vs {item['right']}: " + "; ".join(f"{name}: {reason}" for name, reason in item["reason"].items()))
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {text}" for text in result["limitations"])
    lines.extend(["", "A CI excluding zero describes this pilot contrast under the stated resampling model. It does not establish broad superiority or state-of-the-art performance.", ""])
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be at least 20")
    if not args.evaluations.is_dir():
        raise FileNotFoundError(args.evaluations)
    # Validate every available pair before replacing any prior report.
    result = build_comparisons(args.evaluations, samples=args.bootstrap_samples, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.output / "paired_difference.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _atomic_write(args.output / "paired_difference.md", render_markdown(result))
    print(json.dumps({"comparisons": len(result["comparisons"]), "missing": len(result["missing"]),
                      "json": str(args.output / "paired_difference.json"), "markdown": str(args.output / "paired_difference.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
