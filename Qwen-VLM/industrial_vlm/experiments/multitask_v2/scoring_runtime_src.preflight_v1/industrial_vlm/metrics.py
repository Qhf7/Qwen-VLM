"""Dependency-free metrics for structured industrial anomaly predictions.

No AUROC is computed: generated labels are discrete, not calibrated scores.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

SCHEMA_VERSION = "industrial-vlm-eval-v1"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_boxes(value: Any) -> list[list[float]]:
    """Validate normalized [x1,y1,x2,y2] boxes, without clipping or rescaling."""
    if not isinstance(value, list):
        raise ValueError("boxes must be a list")
    result = []
    for box in value:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("each box must contain four coordinates")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError("coordinates must be finite numbers")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("boxes must have positive area within [0,1000]")
        result.append([float(x) for x in box])
    return result


def parse_prediction(raw_text: str, require_boxes: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    """Accept a single JSON object, optionally enclosed in one JSON code fence.

    No substring extraction, label coercion, coordinate repair, or box clipping is
    performed. Nonempty boxes on normal predictions remain valid structurally;
    their semantic inconsistency is measured separately.
    """
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))
        if not isinstance(value, dict):
            raise ValueError("prediction must be a JSON object")
        if type(value.get("is_anomalous")) is not bool:
            raise ValueError("is_anomalous must be a JSON boolean")
        if set(value) - {"is_anomalous", "boxes"}:
            raise ValueError("unexpected fields; allowed fields are is_anomalous and boxes")
        if require_boxes and "boxes" not in value:
            raise ValueError("boxes is required for the region task")
        parsed = {"is_anomalous": value["is_anomalous"]}
        if "boxes" in value:
            parsed["boxes"] = validate_boxes(value["boxes"])
        return parsed, None
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def enclosing_box(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def union_box_iou(predicted: list[list[float]], target: list[list[float]]) -> float:
    """IoU of the two enclosing rectangles, NOT a polygon/mask union IoU."""
    a, b = enclosing_box(predicted), enclosing_box(target)
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def union_area_fraction(boxes: list[list[float]]) -> float:
    """Exact union area of rectangles / image area; used only for GT strata."""
    if not boxes:
        return 0.0
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        length, end = 0.0, -1.0
        for low, high in intervals:
            length += max(0.0, high - max(low, end))
            end = max(end, high)
        area += (right - left) * length
    return area / 1_000_000.0


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int, support: int) -> float | None:
    if not support:
        return None
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid_normal = invalid_anomalous = valid = 0
    normals = sum(not r["label"] for r in rows)
    anomalies = len(rows) - normals
    region_rows = [r for r in rows if r.get("localization_eligible", False)]
    region_anomalies = [r for r in region_rows if r["label"]]
    normal_empty = inconsistent = 0
    for row in rows:
        pred = row.get("parsed")
        if pred is None:
            if row["label"]:
                invalid_anomalous += 1
            else:
                invalid_normal += 1
            continue
        valid += 1
        if row["label"]:
            if pred["is_anomalous"]:
                tp += 1
            else:
                fn += 1
        elif pred["is_anomalous"]:
            fp += 1
        else:
            tn += 1
        if row.get("localization_eligible"):
            if not row["label"] and pred.get("boxes") == []:
                normal_empty += 1
            if not pred["is_anomalous"] and pred.get("boxes"):
                inconsistent += 1
    anomaly_f1 = _f1(tp, fp, fn + invalid_anomalous, anomalies)
    normal_f1 = _f1(tn, fn, fp + invalid_normal, normals)
    f1s = [x for x in (anomaly_f1, normal_f1) if x is not None]
    recalls = [x for x in (_ratio(tp, anomalies), _ratio(tn, normals)) if x is not None]
    result: dict[str, Any] = {
        "n": len(rows), "n_sources": len({r["source_id"] for r in rows}),
        "n_normal": normals, "n_anomalous": anomalies,
        "n_valid": valid, "n_invalid_normal": invalid_normal,
        "n_invalid_anomalous": invalid_anomalous,
        "tp": tp, "tn": tn, "fp": fp, "fn_valid": fn,
        "accuracy": _ratio(tp + tn, len(rows)),
        "anomaly_precision": _ratio(tp, tp + fp),
        "anomaly_recall": _ratio(tp, anomalies),
        "anomaly_f1": anomaly_f1, "normal_f1": normal_f1,
        "macro_f1": mean(f1s) if f1s else None,
        "balanced_accuracy": mean(recalls) if recalls else None,
        "fpr": _ratio(fp, normals),
        "fnr": _ratio(fn + invalid_anomalous, anomalies),
        "normal_failure_rate": _ratio(fp + invalid_normal, normals),
        "format_valid_rate": _ratio(valid, len(rows)),
        "n_localization": len(region_rows), "n_localization_anomalous": len(region_anomalies),
        "normal_empty_box_rate": _ratio(normal_empty, sum(not r["label"] for r in region_rows)),
        "n_normal_prediction_with_nonempty_boxes": inconsistent,
    }
    for threshold in (0.3, 0.5):
        successes = sum(
            r.get("parsed") is not None and r["parsed"]["is_anomalous"]
            and union_box_iou(r["parsed"].get("boxes", []), r["gt_boxes"]) >= threshold
            for r in region_anomalies
        )
        result[f"union_box_recall_iou_{threshold:.1f}"] = _ratio(successes, len(region_anomalies))
    latencies = [r["latency_seconds"] for r in rows if isinstance(r.get("latency_seconds"), (int, float))]
    result["mean_latency_seconds"] = mean(latencies) if latencies else None
    memories = [r["peak_memory_allocated_bytes"] for r in rows if isinstance(r.get("peak_memory_allocated_bytes"), (int, float))]
    result["max_peak_memory_allocated_bytes"] = max(memories) if memories else None
    return result


def _area_group(row: dict[str, Any]) -> str:
    if not row["label"]:
        return "normal"
    # mask fraction is preferred; otherwise this is explicitly a BOX-area proxy.
    fraction = row.get("anomaly_area_fraction")
    if fraction is None:
        fraction = union_area_fraction(row["gt_boxes"])
    if fraction <= 0:
        return "anomalous_area_unknown"
    if fraction < 0.01:
        return "anomalous_lt_1pct"
    if fraction < 0.05:
        return "anomalous_1_to_5pct"
    return "anomalous_ge_5pct"


def compute_metrics(rows: Iterable[dict[str, Any]], bootstrap_samples: int = 0, seed: int = 17) -> dict[str, Any]:
    records = list(rows)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    areas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if type(row.get("label")) is not bool:
            raise ValueError("label must be a boolean")
        if not row.get("source_id"):
            raise ValueError("source_id is required")
        validate_boxes(row["gt_boxes"])
        if row["label"] and row.get("localization_eligible") and not row["gt_boxes"]:
            raise ValueError("region-task anomalous ground truth requires boxes")
        if row.get("parsed") is not None:
            validated, error = parse_prediction(json.dumps(row["parsed"]), bool(row.get("localization_eligible")))
            if error:
                raise ValueError(f"parsed field violates prediction schema: {error}")
        categories[str(row.get("category", "unknown"))].append(row)
        areas[_area_group(row)].append(row)
    result = {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "classification_unit": "one prediction row; repeated source images remain separate tasks",
            "invalid_outputs": "retained; invalid anomaly counts as anomaly FN, invalid normal counts as normal-class FN",
            "fpr": "valid anomalous predictions / all normal rows; format failures separately included in normal_failure_rate",
            "fnr": "(valid normal predictions + invalid predictions) / all anomalous rows",
            "macro_f1": "mean normal/anomalous F1 over classes with ground-truth support",
            "balanced_accuracy": "mean normal/anomalous recall over classes with ground-truth support",
            "normal_empty_box_rate": "valid empty-box predictions / all normal region-task rows, including format failures",
            "localization": "end-to-end recall over ALL anomalous region-task rows, including false negatives and format failures",
            "union_box": "IoU of minimal enclosing rectangle of all predicted boxes vs minimal enclosing rectangle of all GT boxes; not mask union IoU",
            "area_strata": "mask area fraction when provided, otherwise rectangle-union area fraction proxy; <1%, [1%,5%), >=5%",
            "latency": "warm model input transfer + generation + decoding; excludes preprocessing and initial warm-up",
            "auroc": "not computed; generated boolean predictions are not continuous scores",
        },
        "overall": _summary(records),
        "by_category": {k: _summary(v) for k, v in sorted(categories.items())},
        "by_anomaly_area": {k: _summary(v) for k, v in sorted(areas.items())},
    }
    area_sources = {"mask" if r.get("anomaly_area_fraction") is not None else "box_proxy" for r in records if r["label"]}
    result["area_stratification_sources"] = sorted(area_sources)
    if bootstrap_samples:
        if bootstrap_samples < 20:
            raise ValueError("bootstrap_samples must be 0 or at least 20")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[row["source_id"]].append(row)
        source_ids = list(grouped)
        rng = random.Random(seed)
        keys = ("anomaly_f1", "macro_f1", "balanced_accuracy", "fpr", "fnr", "format_valid_rate",
                "union_box_recall_iou_0.3", "union_box_recall_iou_0.5")
        samples: dict[str, list[float]] = {key: [] for key in keys}
        for _ in range(bootstrap_samples):
            selected = [row for _ in source_ids for row in grouped[rng.choice(source_ids)]]
            summary = _summary(selected)
            for key in keys:
                if summary[key] is not None:
                    samples[key].append(summary[key])
        result["bootstrap"] = {"unit": "source_id cluster", "method": "percentile 95% CI", "seed": seed,
                               "n_resamples": bootstrap_samples, "n_sources": len(source_ids), "intervals": {}}
        for key, values in samples.items():
            values.sort()
            result["bootstrap"]["intervals"][key] = {
                "low": values[int(0.025 * (len(values) - 1))] if values else None,
                "high": values[int(0.975 * (len(values) - 1))] if values else None,
                "n_defined_resamples": len(values),
            }
    return result
