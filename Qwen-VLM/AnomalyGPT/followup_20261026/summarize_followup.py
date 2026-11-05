"""Summarize complete or partial follow-up outputs; never fill missing scores.

Uses only the standard library and NumPy. It verifies real-evaluation source
identity and recomputes generation/image-ranking metrics from prediction rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)
RELEASED_INIT_SHA256 = "a7e858586d9118c3322ff8ca3b60a8c7c521859ec8c1825e733df6666acab7d6"
SYNTHETIC_VERSION = "four_layer_average_image_label_v2"
METRIC_PATHS = {
    "generation_macro_f1": ("generation", "macro_f1"),
    "generation_anomaly_recall": ("generation", "anomaly_recall"),
    "generation_normal_fpr": ("generation", "normal_fpr"),
    "generation_invalid_rate": ("generation", "invalid_rate"),
    "pooled_map_image_auroc": ("map_image", "auroc"),
    "pooled_answer_image_auroc": ("answer_image", "auroc"),
    "category_macro_map_image_auroc": ("category_macro", "map_image", "auroc"),
    "category_macro_pixel_auroc": ("category_macro", "pixel", "auroc"),
    "category_macro_pixel_ap": ("category_macro", "pixel", "ap"),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path, errors=None):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(), parse_constant=reject_json_constant)
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value
    except (ValueError, OSError) as error:
        if errors is not None:
            errors.append({"path": str(path), "error": str(error)})
        return None


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def flat_metrics(metrics):
    return {name: get(metrics, *path) for name, path in METRIC_PATHS.items()}


def stats(values, seeds=None):
    pairs = [(seed, value) for seed, value in zip(seeds if seeds is not None else range(len(values)), values)
             if finite_number(value)]
    vals = [float(value) for _, value in pairs]
    return {"n": len(vals), "seeds": [seed for seed, _ in pairs], "values": vals,
            "mean": float(np.mean(vals)) if vals else None,
            "sample_sd": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def classification(rows):
    if not rows:
        return None
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    result = {}
    f1s = []
    for value in (0, 1):
        tp = int(((labels == value) & (predictions == value)).sum())
        fp = int(((labels != value) & (predictions == value)).sum())
        fn = int(((labels == value) & (predictions != value)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    normal, anomalous = labels == 0, labels == 1
    result.update(macro_f1=float(np.mean(f1s)), accuracy=float((labels == predictions).mean()),
                  anomaly_recall=float((predictions[anomalous] == 1).mean()) if anomalous.any() else None,
                  normal_recall=float((predictions[normal] == 0).mean()) if normal.any() else None,
                  normal_fpr=float((predictions[normal] == 1).mean()) if normal.any() else None,
                  invalid_rate=float((predictions == -1).mean()),
                  n_normal=int(normal.sum()), n_anomaly=int(anomalous.sum()))
    return result


def ranking(labels, scores):
    """Tie-aware exact AUROC and non-interpolated AP, without sklearn."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or not len(labels) or labels.shape != scores.shape or not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid ranking arrays")
    labels = labels.astype(np.uint8)
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    true_positive = np.cumsum(y, dtype=np.int64)[ends]
    false_positive = 1 + ends - true_positive
    positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
    auroc = None
    ap = None
    if positive and negative:
        tpr = np.r_[0.0, true_positive / positive]
        fpr = np.r_[0.0, false_positive / negative]
        auroc = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2))
    if positive:
        recall = np.r_[0.0, true_positive / positive]
        precision = true_positive / (true_positive + false_positive)
        ap = float(np.sum(np.diff(recall) * precision))
    return {"auroc": auroc, "ap": ap}


def read_predictions(path):
    if not path.is_file():
        return None, []
    rows, errors = [], []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=reject_json_constant)
            if not isinstance(row, dict):
                raise ValueError("prediction must be an object")
            for field in ("source_id", "source_path", "class_name", "label", "prediction"):
                if field not in row:
                    raise ValueError(f"missing {field}")
            if row["prediction"] not in (-1, 0, 1):
                raise ValueError("invalid prediction label")
            if row["label"] not in (0, 1):
                raise ValueError("invalid ground-truth label")
            if not all(isinstance(row[key], str) and row[key] for key in ("source_id", "source_path", "class_name")):
                raise ValueError("source identity fields must be nonempty strings")
            rows.append(row)
        except (ValueError, TypeError) as error:
            errors.append({"line": index, "error": str(error)})
    return rows, errors


def protocol(root):
    choices = [root / "real_cache/sources.json", root / "real_sources.json"]
    path = next((path for path in choices if path.is_file()), None)
    read_errors = []
    data = read_json(path, read_errors) if path else None
    if data is None:
        return {"status": "invalid" if read_errors else "missing", "expected_source_count": 2005,
                "expected_normal_count": 925, "expected_anomaly_count": 1080, "problems": read_errors}, {}
    records = data.get("records", [])
    problems = []
    expected = {}
    if not isinstance(records, list):
        records = []
        problems.append("records must be an array")
    for index, row in enumerate(records):
        try:
            if row["label_int"] not in (0, 1) or not all(isinstance(row[key], str) and row[key] for key in ("image", "object")):
                raise ValueError("invalid image path, class or binary label")
            expected[row["image"]] = {"label": int(row["label_int"]), "class_name": row["object"],
                                       "source_id": hashlib.sha256(row["image"].encode()).hexdigest()[:24]}
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"record {index}: {error}")
    counts = Counter(row["label"] for row in expected.values())
    if len(expected) != len(records):
        problems.append("duplicate source paths in frozen protocol")
    for actual, target, description in ((len(expected), 2005, "sources"), (counts[0], 925, "normals"), (counts[1], 1080, "anomalies")):
        if actual != target:
            problems.append(f"{description}: actual {actual}, planned {target}")
    if data.get("source_count") != len(expected):
        problems.append("source_count does not match unique records")
    if len({row["class_name"] for row in expected.values()}) != 12:
        problems.append("frozen protocol does not contain exactly 12 classes")
    if data.get("geometry") != "square224_RGB_bicubic_mask_index_positive_nearest" or data.get("reference_mode") != "none":
        problems.append("geometry or reference mode differs from the frozen protocol")
    return {"status": "verified" if not problems else "protocol_mismatch", "path": str(path),
            "sha256": digest(path), "num_sources": len(expected), "normal_count": counts[0],
            "anomaly_count": counts[1], "num_classes": len({r['class_name'] for r in expected.values()}),
            "geometry": data.get("geometry"), "reference_mode": data.get("reference_mode"),
            "exclusions": data.get("exclusions"), "per_class_label": data.get("per_class_label"),
            "evaluation_role": "development_diagnostic_after_this_run", "problems": problems}, expected


def evaluation_names():
    names = [(f"historical_{arm}{seed}", "historical", arm, seed)
             for seed in SEEDS for arm in ("frozen", "lora")]
    names.append(("shared_initial_qwen42", "initial", "qwen", 42))
    names.extend((f"matched_{arm}{seed}", "matched", arm, seed)
                 for seed in SEEDS for arm in ("qwen_frozen", "qwen_lora", "vicuna"))
    return names


def compare_metrics(recomputed, reported, errors, differences, prefix=""):
    """Verify reported metrics including missing and undefined values."""
    for key, value in recomputed.items():
        field = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            compare_metrics(value, get(reported, key), errors, differences, field)
            continue
        if not isinstance(reported, dict) or key not in reported:
            errors.append({"missing_reported_metric": field})
            continue
        reference = reported[key]
        if value is None:
            if reference is not None:
                errors.append({"undefined_metric_mismatch": field})
        elif not finite_number(reference):
            errors.append({"invalid_reported_metric": field})
        else:
            delta = abs(float(reference) - float(value))
            differences[field] = delta
            if delta > 1e-9:
                errors.append({"recomputed_metric_mismatch": field, "absolute_difference": delta})


def audit_evaluation(root, name, stage, arm, seed, expected, protocol_verified=False):
    directory = root / "real_evaluations" / name
    read_errors = []
    metrics = read_json(directory / "metrics.json", read_errors)
    run = read_json(directory / "run.json", read_errors)
    rows, line_errors = read_predictions(directory / "predictions.jsonl")
    result = {"name": name, "stage": stage, "arm": arm, "seed": seed,
              "directory": str(directory), "status": "missing", "metrics": flat_metrics(metrics),
              "reported_num_sources": get(metrics, "num_sources"), "audit": {},
              "checkpoint_sha256": get(run, "bridge", "sha256"), "initial_sha256": get(run, "init_sha256"),
              "backend": get(metrics, "backend"), "source_sha256": get(run, "source_sha256"),
              "cache_manifest_sha256": get(run, "cache_manifest_sha256"),
              "cache_hashes_sha256": hashlib.sha256(json.dumps(run["cache_hashes"], sort_keys=True).encode()).hexdigest()
              if isinstance(get(run, "cache_hashes"), dict) and run["cache_hashes"] else None}
    if rows is None and metrics is None and not read_errors:
        return result, None
    if rows is None:
        result.update(status="audit_failed" if read_errors else "unverified_missing_predictions", audit={"errors": read_errors})
        return result, None
    errors = read_errors + list(line_errors)
    source_ids = [row["source_id"] for row in rows]
    paths = [row["source_path"] for row in rows]
    duplicate_ids = [value for value, count in Counter(source_ids).items() if count > 1]
    duplicate_paths = [value for value, count in Counter(paths).items() if count > 1]
    if duplicate_ids or duplicate_paths:
        errors.append({"duplicate_source_ids": duplicate_ids, "duplicate_source_paths": duplicate_paths})
    mismatch = []
    for row in rows:
        reference = expected.get(row["source_path"])
        if reference and any(row[key] != reference[key] for key in ("source_id", "class_name", "label")):
            mismatch.append(row["source_path"])
    if mismatch:
        errors.append({"source_metadata_mismatch": mismatch})
    actual_set, expected_set = set(paths), set(expected)
    count_matches = len(rows) == len(expected) if expected else None
    full_set = actual_set == expected_set if expected else None
    if metrics and get(metrics, "num_sources") != len(rows):
        errors.append({"metric_count_mismatch": [get(metrics, "num_sources"), len(rows)]})
    recomputed = {"generation": classification(rows)} if rows else {}
    for family, score in (("map_image", "map_score"), ("answer_image", "answer_log_odds")):
        if rows and all(score in row for row in rows):
            recomputed[family] = ranking([row["label"] for row in rows], [row[score] for row in rows])
        elif metrics:
            errors.append({"missing_prediction_score": score})
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_name"], []).append(row)
    class_metrics = {}
    for category, group in by_class.items():
        entry = {"num_sources": len(group), "generation": classification(group)}
        for family, score in (("map_image", "map_score"), ("answer_image", "answer_log_odds")):
            if all(score in row for row in group):
                entry[family] = ranking([row["label"] for row in group], [row[score] for row in group])
        class_metrics[category] = entry
    if rows:
        recomputed["per_class"] = class_metrics
        recomputed["category_macro"] = {}
        for family, keys in (("map_image", ("auroc", "ap")), ("answer_image", ("auroc", "ap")),
                             ("generation", ("macro_f1", "anomaly_recall", "normal_fpr"))):
            recomputed["category_macro"][family] = {}
            for key in keys:
                values = [get(entry, family, key) for entry in class_metrics.values()]
                values = [value for value in values if value is not None]
                recomputed["category_macro"][family][key] = float(np.mean(values)) if values else None
    metric_differences = {}
    if metrics:
        compare_metrics(recomputed, metrics, errors, metric_differences)
        reported_classes = get(metrics, "per_class")
        if not isinstance(reported_classes, dict) or set(reported_classes) != set(by_class):
            errors.append({"reported_per_class_set_mismatch": True})
        else:
            pixel_macro = {}
            for key in ("auroc", "ap"):
                values = [get(entry, "pixel", key) for entry in reported_classes.values()]
                if any(not finite_number(value) or not 0 <= value <= 1 for value in values):
                    errors.append({"invalid_per_class_pixel_metric": key})
                else:
                    pixel_macro[key] = float(np.mean(values))
            compare_metrics(pixel_macro, get(metrics, "category_macro", "pixel"), errors,
                            metric_differences, "category_macro.pixel")
        for key, value in flat_metrics(metrics).items():
            if not finite_number(value) or not 0 <= value <= 1:
                errors.append({"invalid_main_metric": key})
        if get(metrics, "seed") != seed:
            errors.append({"seed_mismatch": get(metrics, "seed")})
        if get(metrics, "backend") != ("vicuna" if arm == "vicuna" else "qwen"):
            errors.append({"backend_mismatch": get(metrics, "backend")})
        wants_lora = arm in ("lora", "qwen_lora")
        if bool(get(metrics, "lora_adapter")) != wants_lora:
            errors.append({"lora_adapter_intent_mismatch": True})
    prediction_hash = digest(directory / "predictions.jsonl")
    if get(run, "completed") is True and get(run, "prediction_sha256") != prediction_hash:
        errors.append({"prediction_sha256_mismatch": True})
    erased = [row["source_id"] for row in rows if bool(row["label"]) and row.get("foreground_pixels_224") == 0]
    unknown_area = sum(row.get("foreground_pixels_224") is None for row in rows)
    result["audit"] = {
        "prediction_rows": len(rows), "unique_source_ids": len(set(source_ids)), "unique_paths": len(actual_set),
        "normal_count": sum(not row["label"] for row in rows), "anomaly_count": sum(bool(row["label"]) for row in rows),
        "source_set_matches_protocol": full_set, "count_matches_protocol": count_matches,
        "source_set_sha256": hashlib.sha256("\n".join(sorted(source_ids)).encode()).hexdigest(),
        "missing_source_count": len(expected_set - actual_set) if expected else None,
        "extra_source_count": len(actual_set - expected_set) if expected else None,
        "missing_source_examples": sorted(expected_set - actual_set)[:10] if expected else None,
        "extra_source_examples": sorted(actual_set - expected_set)[:10] if expected else None,
        "official_anomaly_with_empty_224_mask_count": len(erased) if not unknown_area else None,
        "official_anomaly_with_empty_224_mask_ids": erased, "unknown_foreground_area_rows": unknown_area,
        "recomputed_metrics": recomputed, "max_metric_absolute_difference": max(metric_differences.values(), default=None),
        "pixel_audit_scope": "Per-class metric presence/range and macro consistency; pixel NPZ arrays are not reread by this summarizer.",
        "errors": errors, "run_completed_marker": get(run, "completed"),
        "prediction_sha256": prediction_hash}
    if errors:
        result["status"] = "audit_failed"
    elif metrics is None:
        result["status"] = "partial_predictions"
    elif not expected:
        result["status"] = "unverified_missing_protocol"
    elif not protocol_verified:
        result["status"] = "unverified_protocol_mismatch"
    elif not full_set or not count_matches:
        result["status"] = "partial_or_source_mismatch"
    elif get(run, "completed") is not True:
        result["status"] = "metrics_present_run_not_complete"
    else:
        result["status"] = "complete"
    return result, {row["source_id"]: row for row in rows}


def grouped_real(evaluations):
    result = {}
    for stage, arm in (("historical", "frozen"), ("historical", "lora"),
                       ("matched", "qwen_frozen"), ("matched", "qwen_lora"), ("matched", "vicuna")):
        rows = [row for row in evaluations if row["stage"] == stage and row["arm"] == arm]
        complete = [row for row in rows if row["status"] == "complete"]
        seeds = [row["seed"] for row in complete]
        result[f"{stage}_{arm}"] = {"status": "complete" if set(seeds) == set(SEEDS) else "pending",
                                     "completed_seeds": seeds, "expected_seeds": list(SEEDS),
                                     "metrics": {key: stats([row["metrics"][key] for row in complete], seeds)
                                                 for key in METRIC_PATHS}}
    return result


def pair_results(evaluations, predictions, matching):
    by_name = {row["name"]: row for row in evaluations}
    result = {}
    comparisons = (("historical_lora_minus_frozen", "historical_lora", "historical_frozen"),
                   ("matched_qwen_lora_minus_frozen", "matched_qwen_lora", "matched_qwen_frozen"),
                   ("matched_qwen_frozen_minus_vicuna", "matched_qwen_frozen", "matched_vicuna"),
                   ("matched_qwen_lora_minus_vicuna", "matched_qwen_lora", "matched_vicuna"))
    for key, first, second in comparisons:
        pairs = []
        for seed in SEEDS:
            a, b = by_name[first + str(seed)], by_name[second + str(seed)]
            if a["status"] != "complete" or b["status"] != "complete":
                continue
            ra, rb = predictions[a["name"]], predictions[b["name"]]
            same = set(ra) == set(rb) and all(ra[s]["label"] == rb[s]["label"] for s in ra)
            pair = {"seed": seed, "source_ids_and_labels_match": same, "num_sources": len(ra),
                    "metric_difference": {metric: a["metrics"][metric] - b["metrics"][metric]
                                           if a["metrics"][metric] is not None and b["metrics"][metric] is not None else None
                                           for metric in METRIC_PATHS}}
            if same:
                correctness = Counter((ra[s]["prediction"] == int(ra[s]["label"]),
                                       rb[s]["prediction"] == int(rb[s]["label"])) for s in ra)
                pair["generation_correctness"] = {"both_correct": correctness[(True, True)],
                                                  "first_only_correct": correctness[(True, False)],
                                                  "second_only_correct": correctness[(False, True)],
                                                  "both_incorrect": correctness[(False, False)]}
            pairs.append(pair)
        valid = [pair for pair in pairs if pair["source_ids_and_labels_match"]]
        controlled = all(row["controlled_comparison_verified"] for row in matching) if key.startswith("matched_") else None
        result[key] = {"status": "complete" if len(valid) == 3 else "pending", "pairs": pairs,
                       "controlled_comparison_verified": controlled,
                       "difference_statistics": {metric: stats([pair["metric_difference"][metric] for pair in valid],
                                                                [pair["seed"] for pair in valid]) for metric in METRIC_PATHS},
                       "interpretation": "First minus second. Paired by seed and identical source IDs; descriptive mean/sample SD, no significance test."}
    return result


def training_runs(root, evaluations):
    result = []
    for seed in SEEDS:
        for arm in ("qwen_frozen", "qwen_lora", "vicuna"):
            directory = root / "matched_runs" / f"{arm}_{seed}"
            errors = []
            run = read_json(directory / "run.json", errors)
            budget = read_json(directory / "token_budget.json", errors)
            schedule = read_json(directory / "sample_order.json", errors)
            rows = get(budget, "tokens_and_sources_by_file")
            truncated = get(budget, "truncated_assistant_records")
            if truncated is not None and not isinstance(truncated, list):
                errors.append({"invalid_truncation_records": True})
            zero = None
            if isinstance(rows, dict):
                zero = [name for name, item in rows.items() if isinstance(item, dict) and item.get("retained_supervised_tokens") == 0]
                if not all(isinstance(item, dict) for item in rows.values()):
                    errors.append({"invalid_token_budget_record": True})
            if rows is not None and not isinstance(rows, dict):
                errors.append({"invalid_token_budget_rows": True})
            observed_schedule = digest(directory / "sample_order.json") if schedule else None
            if observed_schedule and get(run, "sample_order_sha256") and observed_schedule != run["sample_order_sha256"]:
                errors.append({"sample_order_hash_mismatch": True})
            observed_budget = digest(directory / "token_budget.json") if budget else None
            if observed_budget and get(run, "token_budget_sha256") and observed_budget != run["token_budget_sha256"]:
                errors.append({"token_budget_hash_mismatch": True})
            batches = get(schedule, "batches")
            schedule_exposures = None
            if isinstance(batches, list) and all(isinstance(batch, list) for batch in batches):
                schedule_exposures = sum(map(len, batches))
                if get(budget, "actual_variant_exposures") != schedule_exposures:
                    errors.append({"schedule_exposure_mismatch": True})
            elif schedule is not None:
                errors.append({"invalid_schedule_batches": True})
            complete = run and get(run, "steps") == 16 and get(run, "checkpoint_sha256") and budget and schedule
            result.append({"arm": arm, "seed": seed, "status": "audit_failed" if errors else ("complete" if complete else "pending"),
                           "directory": str(directory), "steps": get(run, "steps"),
                           "available_cache_variants": get(run, "num_files"),
                           "actual_unique_sources": get(budget, "actual_unique_source_paths"),
                           "actual_unique_variants": get(budget, "unique_variants"),
                           "actual_variant_exposures": get(budget, "actual_variant_exposures"),
                           "max_tgt_len": get(budget, "max_tgt_len"),
                           "truncated_assistant_records_count": len(truncated) if isinstance(truncated, list) else None,
                           "zero_supervision_files": zero,
                           "total_supervised_tokens": get(run, "total_supervised_tokens"),
                           "class_label_counts": get(budget, "class_label_counts"),
                           "class_variant_counts": get(budget, "class_variant_counts"),
                           "normal_vs_positive_mask_count": get(budget, "normal_vs_positive_mask_count"),
                           "sample_order_sha256": get(run, "sample_order_sha256"),
                           "initial_bridge_sha256": get(run, "initial_bridge_sha256") or get(run, "initial_checkpoint_sha256"),
                           "actual_initial_bridge_fingerprint": get(run, "actual_initial_bridge_fingerprint"),
                           "checkpoint_sha256": get(run, "checkpoint_sha256"),
                           "seconds": get(run, "seconds"), "peak_allocated_bytes": get(run, "peak_allocated_bytes"),
                           "trainable_parameters": get(run, "trainable_parameters"),
                           "matching": get(run, "matching"),
                           "budget_sha256": observed_budget, "observed_schedule_sha256": observed_schedule,
                           "schedule_exposures_from_file": schedule_exposures, "errors": errors})
    matching = []
    for seed in SEEDS:
        rows = [row for row in result if row["seed"] == seed]
        schedules = [row["sample_order_sha256"] for row in rows]
        bridges = [row["initial_bridge_sha256"] for row in rows]
        fingerprints = [row["actual_initial_bridge_fingerprint"] for row in rows]
        evals = [next(item for item in evaluations if item["name"] == f"matched_{row['arm']}{seed}") for row in rows]
        checkpoint_matches = [row["checkpoint_sha256"] == evaluation.get("checkpoint_sha256")
                              if row["checkpoint_sha256"] and evaluation.get("checkpoint_sha256") else None
                              for row, evaluation in zip(rows, evals)]
        cache_hashes = [row.get("cache_hashes_sha256") for row in evals]
        all_complete = all(row["status"] == "complete" for row in rows)
        same_schedule = len(set(schedules)) == 1 if all(schedules) else None
        same_bridge = len(set(bridges)) == 1 if all(bridges) else None
        same_fingerprint = len(set(fingerprints)) == 1 if all(fingerprints) else None
        same_cache = len(set(cache_hashes)) == 1 if all(cache_hashes) else None
        budgets = all(row["max_tgt_len"] == 128 and row["actual_variant_exposures"] == 32
                      and row["schedule_exposures_from_file"] == 32 and row["zero_supervision_files"] == []
                      and row["truncated_assistant_records_count"] == 0 for row in rows)
        controlled = all_complete and all(value is True for value in
                     (same_schedule, same_bridge, same_fingerprint, same_cache, budgets, *checkpoint_matches))
        matching.append({"seed": seed,
                         "same_schedule_hash": same_schedule, "same_initial_bridge_hash": same_bridge,
                         "same_actual_initial_bridge_fingerprint": same_fingerprint,
                         "evaluation_training_checkpoint_matches": dict(zip((row["arm"] for row in rows), checkpoint_matches)),
                         "same_evaluation_cache_hashes": same_cache,
                         "budget_checks_passed": budgets,
                         "all_complete": all_complete, "controlled_comparison_verified": controlled})
    return result, matching


def synthetic(root):
    result = []
    names = [("initial", "initial", None)] + [(f"{arm}{seed}", arm, seed) for seed in SEEDS for arm in ("frozen", "lora")]
    for name, arm, seed in names:
        path = root / "synthetic_corrected" / (name + ".json")
        errors = []
        data = read_json(path, errors)
        metrics = {f"{scope}_{metric}": get(data, scope, metric)
                   for scope in ("pooled", "macro") for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")}
        if data:
            if data.get("evaluation_version") != SYNTHETIC_VERSION:
                errors.append({"evaluation_version_mismatch": data.get("evaluation_version")})
            if get(data, "pooled", "num_variants") != 48:
                errors.append({"variant_count_mismatch": get(data, "pooled", "num_variants")})
            if get(data, "macro", "num_classes") != 12:
                errors.append({"class_count_mismatch": get(data, "macro", "num_classes")})
            for key, value in metrics.items():
                if not finite_number(value) or not 0 <= value <= 1:
                    errors.append({"invalid_main_metric": key})
            if not get(data, "checkpoint_sha256") or not isinstance(data.get("label_audit"), dict):
                errors.append({"missing_checkpoint_or_label_audit": True})
        result.append({"name": name, "arm": arm, "seed": seed,
                       "status": "audit_failed" if errors else ("complete" if data else "missing"),
                       "path": str(path), "metrics": metrics, "checkpoint_sha256": get(data, "checkpoint_sha256"),
                       "label_audit": get(data, "label_audit"), "num_variants": get(data, "pooled", "num_variants"),
                       "evaluation_version": get(data, "evaluation_version"), "errors": errors})
    groups = {}
    for arm in ("frozen", "lora"):
        rows = [row for row in result if row["arm"] == arm and row["status"] == "complete"]
        keys = result[0]["metrics"]
        groups[arm] = {"status": "complete" if len(rows) == 3 else "pending",
                       "metrics": {key: stats([row["metrics"][key] for row in rows], [row["seed"] for row in rows]) for key in keys}}
    return {"runs": result, "groups": groups, "required_evaluation_version": SYNTHETIC_VERSION,
            "revision": "Historical pixel AUROC/AP pooled layer maps as extra samples. Corrected metrics score the final four-layer mean map once per variant; not a retraining improvement.",
            "image_label_revision": "v2 uses explicit image labels, otherwise conversation Yes/No, and only then threshold-mask fallback. Pixel masks remain >0.3; a semantically positive image can have no pixel above that threshold. v1 image metrics are superseded.",
            "legacy_teacher_forcing": "Historical mean_batch_token_acc is a teacher-forcing token diagnostic, not free-generation answer accuracy."}


def historical_source_reuse(root, expected):
    """Verify earlier synthetic-test normal sources against the frozen real set.

    The explicitly saved sensitivity list is evidence, not an instruction to
    change the 2,005-source evaluation protocol.
    """
    result = {
        "status": "unverified_missing_evidence", "evidence_path": None,
        "prior_unique_source_count": None, "reused_source_count": None,
        "reused_normal_source_count": None, "reused_source_paths": None,
        "source_manifest_verified": None, "problems": [],
        "main_evaluation_unchanged": True,
        "role": "Historical source reuse audit; the existing real evaluation remains a development diagnostic.",
    }
    choices = [root / "previous_test8_excluded_sources.json",
               root.parent / "previous_test8_excluded_sources.json"]
    path = next((candidate for candidate in choices if candidate.is_file()), None)
    if path is None:
        result["problems"].append("Earlier test8 source evidence file is unavailable; overlap is not assumed zero.")
        return result
    read_errors = []
    data = read_json(path, read_errors)
    result.update(evidence_path=str(path), evidence_sha256=digest(path))
    if data is None:
        result.update(status="unverified_invalid_evidence", problems=read_errors)
        return result
    records = data.get("records")
    if not isinstance(records, list) or not records:
        result.update(status="unverified_invalid_evidence", problems=["Evidence records must be a nonempty array."])
        return result
    prior = {}
    for index, record in enumerate(records):
        try:
            source = record["source_path"]
            label = record["expected_label"]
            category = record["class_name"]
            if not isinstance(source, str) or not source or label not in (0, 1):
                raise ValueError("invalid path or label")
            source_id = hashlib.sha256(source.encode()).hexdigest()[:24]
            if record.get("source_id") != source_id:
                raise ValueError("source_id does not match source_path")
            if source in prior:
                raise ValueError("duplicate source_path in evidence")
            prior[source] = {"label": int(label), "class_name": category, "source_id": source_id}
        except (KeyError, TypeError, ValueError) as error:
            result["problems"].append(f"record {index}: {error}")
    if result["problems"]:
        result["status"] = "unverified_invalid_evidence"
        return result
    result["prior_unique_source_count"] = len(prior)
    if not expected:
        result.update(status="unverified_missing_frozen_protocol")
        result["problems"].append("Frozen real source protocol is unavailable.")
        return result
    overlap = sorted(set(prior) & set(expected))
    metadata_mismatch = [source for source in overlap if prior[source] != expected[source]]
    if metadata_mismatch:
        result["problems"].append({"frozen_source_metadata_mismatch": metadata_mismatch})
    result.update(
        reused_source_count=len(overlap),
        reused_normal_source_count=sum(expected[source]["label"] == 0 for source in overlap),
        reused_source_paths=overlap,
        prior_source_paths_absent_from_real_set=sorted(set(prior) - set(expected)),
        reused_class_counts=dict(Counter(expected[source]["class_name"] for source in overlap)),
    )
    manifest_name = data.get("source_manifest")
    manifest_choices = ([path.parent / manifest_name, root / manifest_name, root.parent / manifest_name]
                        if isinstance(manifest_name, str) else [])
    manifest_choices.extend([root.parent / "cache_visa_test8/manifest.json",
                             root / "server_snapshot/cache_visa_test8/manifest.json",
                             root.parent / "server_snapshot/cache_visa_test8/manifest.json"])
    manifest_path = next((candidate for candidate in manifest_choices if candidate.is_file()), None)
    if manifest_path is None:
        result["status"] = "frozen_overlap_verified_prior_manifest_unavailable" if not result["problems"] else "audit_failed"
        result["problems"].append("Original cache_visa_test8 manifest is unavailable; evidence origin remains unverified.")
        return result
    manifest_errors = []
    manifest = read_json(manifest_path, manifest_errors)
    actual_hash = digest(manifest_path)
    result.update(source_manifest_path=str(manifest_path), source_manifest_sha256=actual_hash,
                  expected_source_manifest_sha256=data.get("source_manifest_sha256"))
    if actual_hash != data.get("source_manifest_sha256"):
        result["problems"].append("Original test8 manifest SHA256 differs from the saved evidence.")
    raw_records = manifest.get("records") if manifest else None
    if not isinstance(raw_records, list):
        result["problems"].extend(manifest_errors or ["Original test8 manifest records are unavailable."])
    else:
        original_sources = set()
        for record in raw_records:
            source = record.get("source_path", "") if isinstance(record, dict) else ""
            # The original manifest prepends ../visa_data/ to official paths.
            matching = [candidate for candidate in prior
                        if source == candidate or source.endswith("/" + candidate)] if isinstance(source, str) else []
            if len(matching) != 1:
                result["problems"].append({"unmatched_original_source": source})
            else:
                original_sources.add(matching[0])
        if original_sources != set(prior):
            result["problems"].append("Evidence source set differs from the original test8 manifest.")
        result["original_manifest_variant_records"] = len(raw_records)
        result["original_manifest_unique_sources"] = len(original_sources)
    result["source_manifest_verified"] = not result["problems"]
    result["status"] = "verified" if not result["problems"] else "audit_failed"
    return result


def provenance(root, evaluations, expected=None):
    protocol_data = read_json(root / "matched_inputs/protocol.json")
    observed = {value for row in evaluations if (value := row.get("initial_sha256"))}
    if get(protocol_data, "init_sha256"):
        observed.add(protocol_data["init_sha256"])
    files = []
    for path in (root.parent / "ckpt/train_visa/pytorch_model.pt", root.parent / "code/ckpt/train_visa/pytorch_model.pt"):
        if path.is_file():
            value = digest(path)
            files.append({"path": str(path), "sha256": value})
            observed.add(value)
    return {"released_train_visa_sha256": RELEASED_INIT_SHA256, "observed_initialization_hashes": sorted(observed),
            "observed_hashes_match_released": all(value == RELEASED_INIT_SHA256 for value in observed) if observed else None,
            "locally_hashed_files": files,
            "source": "FantasticGNU/AnomalyGPT released train_visa checkpoint; current task independently compared local downloaded bytes and server bytes.",
            "initialization": "Image decoder and PromptLearner are warm-started from released VisA-trained modules, not random heads. Projection is separately seeded and shared only within each matched seed.",
            "language_interface_boundary": "Released PromptLearner was paired with Vicuna. Matching its 4096-dimensional output and sharing bridge weights does not ensure equally adapted prompt semantics for Qwen. This is a simplified no-reference migration pilot; normal-reference inference and PandaGPT mixed training from the original setup are omitted.",
            "remaining_boundary": "Complete released-checkpoint pretraining source provenance is unavailable. New-stage source isolation does not by itself prove the full chain was unseen.",
            "initial_comparison": "shared_initial_qwen42 uses the saved projection for seed42. Historical projection initialization is not guaranteed to be reconstructible; its pixel decoder baseline is the same released decoder.",
            "historical_source_reuse": historical_source_reuse(root, expected or {})}


def execution_states(root):
    errors = []
    legacy_path, safe_path = root / "state.json", root / "safe_state.json"
    legacy = read_json(legacy_path, errors)
    safe = read_json(safe_path, errors)
    # A present but malformed safe state must not silently fall back to the
    # superseded legacy runner state.
    use_safe = safe_path.is_file()
    return {
        "state": legacy, "legacy_state": legacy, "safe_state": safe,
        "effective_state": safe if use_safe else legacy,
        "state_selection": {
            "effective_source": str(safe_path if use_safe else legacy_path),
            "effective_kind": "safe" if use_safe else "legacy",
            "legacy_source": str(legacy_path),
            "legacy_field": "state", "legacy_status": "superseded" if use_safe else "fallback",
            "safe_state_present": use_safe,
            "effective_state_valid": isinstance(safe if use_safe else legacy, dict),
            "read_errors": errors,
        },
    }


def conclusions(summary):
    text = []
    completed = sum(row["status"] == "complete" for row in summary["real_evaluations"])
    text.append("真实 VisA 的 16/16 项评测均完整通过统一来源与指标核查。" if completed == 16 else
                f"真实 VisA 统一来源核查通过 {completed}/16 项评测；未完成项继续标记待完成。")
    groups = summary["real_groups"]
    frozen, lora, vicuna = (groups[f"matched_{name}"] for name in ("qwen_frozen", "qwen_lora", "vicuna"))
    verified_matching = all(row["controlled_comparison_verified"] for row in summary["training_match_audit"])
    if all(group["status"] == "complete" for group in (frozen, lora, vicuna)) and verified_matching:
        ff, lf, vf = (group["metrics"]["generation_macro_f1"]["mean"] for group in (frozen, lora, vicuna))
        if all(value is not None for value in (ff, lf, vf)):
            text.append(f"匹配三 seed 的生成 Macro-F1 均值：冻结 Qwen {ff:.4f}、Qwen LoRA {lf:.4f}、Vicuna {vf:.4f}。")
            text.append("LoRA 的自由回答均值高于冻结 Qwen；可继续检查增益是否伴随异常召回提高和正常误报可接受。" if lf > ff else
                        "LoRA 的自由回答均值未高于冻结 Qwen；当前优先排查回答与软提示利用，扩训不应只依据旧 token accuracy。")
            text.append("冻结 Qwen 的自由回答均值高于 Vicuna；结论限于当前 warm-start、同预算 pilot。" if ff > vf else
                        "冻结 Qwen 的自由回答均值未高于 Vicuna；当前实验尚不支持仅更换语言骨干即可改善回答。")
        map_values = [(label, group["metrics"]["category_macro_pixel_auroc"]["mean"]) for label, group in
                      (("冻结 Qwen", frozen), ("Qwen LoRA", lora), ("Vicuna", vicuna))]
        text.append("真实定位的类别宏 Pixel AUROC 均值：" + "、".join(f"{name} {value:.4f}" if value is not None else f"{name} 待完成" for name, value in map_values) + "。语言与像素指标分别解读。")
    else:
        text.append("三臂三 seed 的训练、真实评测与初始化／顺序／缓存／checkpoint关联核查仍待全部完成，暂不据部分结果决定语言骨干优劣或扩大训练。")
    text.append("本轮是 16-step、小规模新增训练和已有 VisA decoder/PromptLearner 的迁移诊断；不支持从零训练、未见类别零样本或统计显著优越性结论。")
    text.append("发布的 PromptLearner 原本与 Vicuna 配套；共享权重仅控制初值，不能保证 Qwen 已获得同等语言接口适配。Qwen 较差时优先检查 bridge alignment 与软提示利用，再判断扩训；该 pilot 不代表骨干总体能力排名。")
    text.append("这 2,005 张官方测试来源经本轮查看后属于开发诊断集。后续配方选择不能再把它称为未见的最终确认。")
    return text


def number(value, scale=100):
    return "待完成" if value is None else f"{value * scale:.2f}"


def mean_sd(value, require_three=True):
    if value is None or value.get("mean") is None:
        return "待完成"
    if require_three and value.get("n") != 3:
        return f"待完成（已有{value.get('n', 0)}/3）"
    mean, sd = value["mean"], value.get("sample_sd")
    return f"{mean * 100:.2f} ± {sd * 100:.2f}" if sd is not None else f"{mean * 100:.2f}（单次）"


def markdown(summary):
    protocol_data = summary["protocol"]
    exclusions = protocol_data.get("exclusions") or {}
    reuse = get(summary, "provenance", "historical_source_reuse") or {}
    if reuse.get("status") == "verified":
        reuse_text = (
            f"早期 `cache_visa_test8` 的来源复用核查：原 manifest 有 {reuse.get('original_manifest_unique_sources')} 个独立来源、"
            f"{reuse.get('original_manifest_variant_records')} 个 variant 记录；其中 {reuse.get('reused_source_count')} 张仍在冻结的真实评测集合中，"
            f"均核对标签后正常图计 {reuse.get('reused_normal_source_count')} 张。"
            f"证据清单 `{reuse.get('evidence_path')}` 与原 manifest 的 SHA256、source_id、类别及标签均已交叉核对。"
            "这部分属于既有正常来源复用，不能称为全部来源此前均未评测；本次主 2,005 图集合及已有指标保持原样，另行做排除这些来源的敏感性分析。"
        )
    else:
        count = reuse.get("reused_source_count")
        count_text = f"当前可核对的重用来源数为 {count}；" if count is not None else "重用来源数未核验，不能按 0 处理；"
        reuse_text = f"早期 `cache_visa_test8` 来源复用审计状态：{reuse.get('status', 'unverified')}。{count_text}证据路径 `{reuse.get('evidence_path') or '未找到'}`。主评测集合与指标不据此改写，详见 summary.json 的 provenance.historical_source_reuse。"
    state_selection = summary.get("state_selection") or {}
    state_text = (
        f"执行状态采用 `{state_selection.get('effective_source', '待核验')}`（{state_selection.get('effective_kind', '待核验')}）；"
        "summary.json 的 `state` 字段保留旧 runner 的 legacy 状态以兼容既有读取，当前进度应读 `effective_state`。"
    )
    lines = ["# 真实 VisA 与匹配语言骨干对照", "", *summary["conclusions"], "", "## 数据与初始化", "",
             f"冻结协议状态：{protocol_data['status']}。真实评测 {protocol_data.get('num_sources', '待核验')} 张，正常 {protocol_data.get('normal_count', '待核验')}、真实异常 {protocol_data.get('anomaly_count', '待核验')}，类别数 {protocol_data.get('num_classes', '待核验')}。协议记录排除原生封存确认来源 {exclusions.get('native_confirmation_path', '待核验')} 张及既有 synthetic test 来源 {exclusions.get('previous_synthetic_test_source', '待核验')} 张。",
             reuse_text, "", state_text, "",
             "图像全图 bicubic 缩放到 224²；真实类别索引掩码先 >0 二值化，再 nearest 缩放。无正常参考图。此评测与旧 synthetic cache 的软 mask >0.3 分开报告。", "",
             f"初始化使用发布的 train_visa decoder 与 PromptLearner，SHA256 `{RELEASED_INIT_SHA256}`；本地下载与服务器字节已核对一致。它们不是随机头。语言投影在匹配实验中按 seed 单独初始化，同 seed 三臂共享。完整预训练来源清单尚未核验。",
             "shared_initial_qwen42 使用保存的 seed42 投影；历史语言投影不保证能完整重建，因此该初始点对历史回答性能的解释有限；像素基线对应同一发布 decoder。", "", "## 逐模型真实结果", "",
             "指标单位为百分比。生成 Macro-F1 是正常／异常二标签宏平均；类别宏指标对 12 种产品等权。未完成或来源核查失败的行不填分数。", "",
             "| 模型 | 状态/进度 | 生成 F1 | 异常召回 | 正常 FPR | 无效回答 | 热图 I-AUROC | 回答 I-AUROC | 类宏热图 I-AUROC | 类宏 P-AUROC | 类宏 P-AP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["real_evaluations"]:
        progress = get(row, "audit", "prediction_rows")
        status = "完成" if row["status"] == "complete" else f"{row['status']}（{progress if progress is not None else '—'}/2005）"
        values = [number(row["metrics"][key]) if row["status"] == "complete" else "待完成" for key in METRIC_PATHS]
        lines.append("| " + " | ".join([row["name"], status, *values]) + " |")
    lines += ["", "## 三 seed 汇总", "", "均值 ± 样本标准差（ddof=1）；三 seed 均完整通过来源核查后才展示正式均值。初始模型为单次结果，见上表。", "",
              "| 组别 | 生成 F1 | 异常召回 | 正常 FPR | 无效回答 | 热图 I-AUROC | 回答 I-AUROC | 类宏热图 I-AUROC | 类宏 P-AUROC | 类宏 P-AP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in summary["real_groups"].items():
        lines.append("| " + " | ".join([name, *[mean_sd(group["metrics"][key]) for key in METRIC_PATHS]]) + " |")
    lines += ["", "历史组使用旧 max_tgt_len=64；新匹配组为128。两组不作为同一控制实验求训练提升。旧 teacher-forcing token accuracy 不等于自由回答准确率。", "", "## 匹配预算与实际覆盖", "",
              "训练缓存有48个source、96个variant，但16 steps×batch2只曝光32个variant。表中报告实际被用到的来源；不同 seed 的抽样覆盖可不同。相同 seed 的三臂共享显式顺序与完整 bridge，tokenizer及实际监督 token 数不同。", "",
              "| 训练臂/seed | 状态 | steps | 实际source | 独立variant | 曝光 | max_tgt_len | 截断回答数 | 监督token总数 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["training_runs"]:
        keys = ("steps", "actual_unique_sources", "actual_unique_variants", "actual_variant_exposures", "max_tgt_len", "truncated_assistant_records_count", "total_supervised_tokens")
        values = [str(row[key]) if row[key] is not None else "待完成" for key in keys]
        lines.append("| " + " | ".join([f"{row['arm']}_{row['seed']}", row["status"], *values]) + " |")
    lines += ["", "| seed | 三臂训练完成 | 顺序hash相同 | 初始文件hash相同 | 实际tensor指纹相同 | 缓存相同 | 控制核查通过 |", "|---|---|---|---|---|---|---|"]
    for row in summary["training_match_audit"]:
        lines.append(f"| {row['seed']} | {row['all_complete']} | {row['same_schedule_hash']} | {row['same_initial_bridge_hash']} | {row['same_actual_initial_bridge_fingerprint']} | {row['same_evaluation_cache_hashes']} | {row['controlled_comparison_verified']} |")
    lines += ["", "Vicuna 使用可用的 Vicuna-7B-v1.1，非原论文 v0。NF4精度、参考禁用、损失及图片曝光预算统一；相同图片／步数不代表相同 token 数或计算量。发布 PromptLearner 原本与 Vicuna 配套，共享4096维 bridge 初值不保证各语言模型得到相同程度的语义适配。当前省略正常参考图及 PandaGPT 混训，是简化迁移对照。", "", "## 配对差异", "",
              "下表是前者减后者的百分点差，按同 seed 及相同 source_id 配对；仅为描述统计，没有显著性检验。正常FPR和无效回答差值为负表示下降。匹配因果解释还需上表控制核查全部通过；历史三seed不保证同投影初值与顺序。", "",
              "| 比较 | 生成F1差 | 异常召回差 | 正常FPR差 | 类宏P-AUROC差 | 类宏P-AP差 |", "|---|---:|---:|---:|---:|---:|"]
    selected = ("generation_macro_f1", "generation_anomaly_recall", "generation_normal_fpr", "category_macro_pixel_auroc", "category_macro_pixel_ap")
    for name, comparison in summary["paired_differences"].items():
        lines.append("| " + " | ".join([name, *[mean_sd(comparison["difference_statistics"][key]) for key in selected]]) + " |")
    lines += ["", "## Synthetic 指标修订", "", summary["synthetic_corrected"]["revision"],
              summary["synthetic_corrected"]["image_label_revision"],
              "旧 Pixel AUROC/AP 不能继续标作最终热图成绩；GT随每层同步追加，问题是层间混合排序。四层平均后的修订值独立列出，不将口径改变写成训练收益。", "",
              "| 模型 | 状态 | pooled P-AUROC | pooled P-AP | 类宏 P-AUROC | 类宏 P-AP |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["synthetic_corrected"]["runs"]:
        keys = ("pooled_pixel_auroc", "pooled_pixel_ap", "macro_pixel_auroc", "macro_pixel_ap")
        lines.append("| " + " | ".join([row["name"], row["status"], *[number(row["metrics"][key]) if row["status"] == "complete" else "待完成" for key in keys]]) + " |")
    for arm, group in summary["synthetic_corrected"]["groups"].items():
        keys = ("pooled_pixel_auroc", "pooled_pixel_ap", "macro_pixel_auroc", "macro_pixel_ap")
        lines.append("| " + " | ".join([arm + " 三seed", group["status"], *[mean_sd(group["metrics"][key]) for key in keys]]) + " |")
    lines += ["", "## 来源与缩放审计", "", "每个完整模型均核对2,005个逐图 source_id、标签、类别与冻结清单一致，并重算生成F1、图像AUROC/AP及其类别宏平均。像素指标核对逐类值与宏平均的一致性，本汇总器未重新读取 NPZ 计算像素排名。掩码缩放后前景为空的真实异常仍保留官方异常标签。", "",
              "| 模型 | 逐图数 | 来源集合一致 | 异常空224mask | 指标重算最大误差 |", "|---|---:|---|---:|---:|"]
    for row in summary["real_evaluations"]:
        audit = row["audit"]
        error = audit.get("max_metric_absolute_difference")
        values = [row["name"], str(audit.get("prediction_rows", "待完成")), str(audit.get("source_set_matches_protocol", "待核验")),
                  str(audit.get("official_anomaly_with_empty_224_mask_count", "待核验")), f"{error:.3g}" if error is not None else "待核验"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## 后续选择", "", *["- " + value for value in summary["conclusions"]], "",
              "若匹配结果支持保留某个候选，下一轮先预先固定训练／验证协议，再扩大正常来源与更新预算；保留初始化基线、真实像素指标和自由回答指标。只有文本拟合变好时，应先检查定位软提示被利用的证据。未完成任务在 summary.json 中保留状态与缺失字段。", ""]
    return "\n".join(lines)


def summarize(root):
    protocol_data, expected = protocol(root)
    evaluations, predictions = [], {}
    for name, stage, arm, seed in evaluation_names():
        try:
            result, rows = audit_evaluation(root, name, stage, arm, seed, expected, protocol_data["status"] == "verified")
        except (ValueError, OSError, TypeError, KeyError) as error:
            result, rows = {"name": name, "stage": stage, "arm": arm, "seed": seed,
                            "status": "audit_failed", "metrics": flat_metrics(None), "audit": {"errors": [str(error)]}}, None
        evaluations.append(result)
        predictions[name] = rows
    training, matching = training_runs(root, evaluations)
    summary = {"schema_version": "1.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "root": str(root), "source_sha256": digest(__file__), "protocol": protocol_data,
               "provenance": provenance(root, evaluations, expected), "real_evaluations": evaluations,
               "real_groups": grouped_real(evaluations), "paired_differences": pair_results(evaluations, predictions, matching),
               "synthetic_corrected": synthetic(root), "training_runs": training, "training_match_audit": matching,
               **execution_states(root),
               "definitions": {"missing": "null; never imputed as zero", "sample_sd": "ddof=1; undefined for n<2",
                               "rates": "JSON values are fractions in [0,1]; Markdown shows percentage units",
                               "paired_difference": "first minus second; Markdown percentage points",
                               "partial_group": "available values may be present in JSON with n; Markdown formal mean requires n=3",
                               "state": "state remains the legacy runner value for compatibility; safe_state takes precedence in effective_state when its file exists"}}
    summary["conclusions"] = conclusions(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="followup_20261026 directory, or its experiment parent")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for summary.json and RESULTS_REAL_AND_MATCHED.md")
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "followup_20261026").is_dir() and not (root / "real_evaluations").exists():
        root = root / "followup_20261026"
    summary = summarize(root)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, text in (("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("RESULTS_REAL_AND_MATCHED.md", markdown(summary))):
        path = args.out / filename
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(json.dumps({"out": str(args.out.resolve()),
                      "complete_real_evaluations": sum(row["status"] == "complete" for row in summary["real_evaluations"]),
                      "expected_real_evaluations": 16,
                      "complete_training_runs": sum(row["status"] == "complete" for row in summary["training_runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
