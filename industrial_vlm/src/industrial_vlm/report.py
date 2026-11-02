"""Summarize completed evaluation runs without inventing missing metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_BRANCHES = ("dataquality_sampling_ablation", "region_supervision_ablation")


def _discover(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            found.update(p.resolve() for p in path.rglob("metrics.json"))
            found.update(p.resolve() for p in path.rglob("*.metrics.json"))
        else:
            raise FileNotFoundError(path)
    return sorted(found)


def _meta_path(path: Path) -> Path:
    if path.name == "metrics.json":
        return path.with_name("meta.json")
    return path.with_name(path.name.removesuffix(".metrics.json") + ".meta.json")


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, percent: bool = True) -> str:
    if value is None:
        return "—"
    if not isinstance(value, (float, int)):
        return "—"
    return f"{100 * value:.2f}" if percent else f"{value:.3f}"


def build_report(paths: list[Path]) -> tuple[dict[str, Any], str]:
    runs, warnings = [], []
    for path in _discover(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("overall"), dict):
            warnings.append(f"Skipped {path}: missing overall metrics")
            continue
        meta_file = _meta_path(path)
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        config = {**meta.get("config", {}), **payload.get("metadata", {})}
        complete = config.get("complete", meta.get("status") == "complete")
        if not complete:
            warnings.append(f"Excluded incomplete evaluation: {path}")
            continue
        branch = config.get("branch", config.get("ablation_type", "unspecified"))
        name = config.get("run_name") or path.parent.name
        # Dataset SHA prevents falsely calling different prompts/subsets matched.
        comparison = {key: config.get(key) for key in (
            "model", "data_sha256", "subset_sha256", "max_pixels", "max_length",
            "max_new_tokens", "generation", "quantization", "limit")}
        if any(comparison[key] is None for key in ("data_sha256", "subset_sha256", "max_pixels")):
            warnings.append(f"{name}: insufficient metadata to establish matched inputs")
        runs.append({"name": name, "branch": branch, "metrics_file": str(path),
                     "adapter": config.get("adapter"), "comparison_condition": comparison,
                     "metadata": config, "overall": payload["overall"],
                     "by_category": payload.get("by_category", {}),
                     "by_anomaly_area": payload.get("by_anomaly_area", {}),
                     "bootstrap": payload.get("bootstrap")})
    groups: dict[str, list[str]] = {}
    for run in runs:
        signature = json.dumps(run["comparison_condition"], sort_keys=True, ensure_ascii=False)
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(run["name"])
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "runs": runs, "matched_condition_groups": list(groups.values()), "warnings": warnings,
              "interpretation": {
                  "dataquality_sampling_ablation": "Matched category×label quotas; quality covers defect-area strata, random samples within the same quotas. This evaluates area-coverage sampling, not general label cleansing.",
                  "region_supervision_ablation": "Decision-only has no localization result. Compare classification with explicit task/prompt metadata; do not treat absent localization as zero.",
                  "comparison": "Only groups with identical model/input-budget/data/prompt metadata have matched conditions. Different task prompts require a separate protocol explanation.",
                  "uncertainty": "Confidence intervals exist only when present in source metrics; no significance claim is inferred.",
              }}
    lines = ["# Industrial VLM experiment matrix", "",
             "Values below are observed evaluation outputs. Rates are percentages; — means undefined or unavailable.", "",
             "| Run | Branch | N / sources | Valid | Anomaly F1 | Macro F1 | BAcc | FPR | FNR | Normal failure | Empty boxes on normal | Union-box R@.3 | Union-box R@.5 | Latency s | Peak GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        m = run["overall"]
        memory = m.get("max_peak_memory_allocated_bytes")
        cells = [_escape(run["name"]), _escape(run["branch"]), f"{m.get('n', '—')} / {m.get('n_sources', '—')}",
                 *[_number(m.get(key)) for key in ("format_valid_rate", "anomaly_f1", "macro_f1", "balanced_accuracy",
                                                  "fpr", "fnr", "normal_failure_rate", "normal_empty_box_rate",
                                                  "union_box_recall_iou_0.3", "union_box_recall_iou_0.5")],
                 _number(m.get("mean_latency_seconds"), False),
                 _number(memory / 1024 ** 3 if memory is not None else None, False)]
        lines.append("| " + " | ".join(cells) + " |")
    if not runs:
        lines.extend(["", "No completed runs with metrics were found."])
    lines.extend(["", "## Protocol and interpretation", "",
                  "Format failures remain in denominators. FPR counts explicit anomalous predictions on normal images; normal failure additionally counts invalid normal outputs. FNR includes invalid anomalous outputs.", "",
                  "Union-box recall uses the enclosing rectangle of all boxes and all anomalous region-task rows. It is neither instance-matched detection recall nor pixel/mask segmentation. Decision-only runs have no localization score.", "",
                  "The data-quality sampling branch compares category×label-matched random selection with defect-area coverage. It does not establish general label-noise removal.", "",
                  "## Matched evaluation conditions", ""])
    for index, group in enumerate(result["matched_condition_groups"], 1):
        lines.append(f"- Condition {index}: " + ", ".join(_escape(name) for name in group))
    for run in runs:
        if run.get("bootstrap"):
            lines.extend(["", f"## {_escape(run['name'])}: source-cluster bootstrap intervals", "",
                          "| Metric | 95% interval |", "|---|---:|"])
            for key, interval in run["bootstrap"].get("intervals", {}).items():
                lines.append(f"| {_escape(key)} | {_number(interval.get('low'))}–{_number(interval.get('high'))} |")
    if warnings:
        lines.extend(["", "## Missing evidence", ""])
        lines.extend(f"- {_escape(item)}" for item in warnings)
    lines.extend(["", "Per-category and anomaly-area breakdowns, source paths, and evaluation metadata are retained in matrix.json.", ""])
    return result, "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="report directory")
    args = parser.parse_args(argv)
    result, markdown = build_report(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "matrix.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"runs": len(result["runs"]), "markdown": str(args.output / "matrix.md"),
                      "json": str(args.output / "matrix.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
