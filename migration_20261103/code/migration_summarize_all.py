"""JSON-only progress/contract summary for the five VisA migration routes.

Standard library only. Never loads checkpoints, arrays, images, model code,
or a GPU. Missing/partially written outputs remain ongoing. No best-run
selection, significance claim, test result, or cross-route ranking is made.

  python migration_summarize_all.py --root runs/migration_20261103
  python migration_summarize_all.py --root runs/migration_20261103 --out runs/migration_20261103/all_summary.json
  python migration_summarize_all.py --cpu-test-only

The optional output is a derived JSON report. Inputs are read-only. Dense
completion requires its recorded best-validation reload and completion JSON;
GT/KD deltas additionally require an identical matched_run_key and plan.
ROI pilot and full validation stay separate even when both are available.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile


DEFAULT_ROOT = "/root/autodl-tmp/AnomalyGPT/runs/migration_20261103"
SCHEMA_VERSION = "visa-five-route-json-progress-v1"
PIXEL_KEYS = ("pixel_ap_macro", "pixel_auroc_macro", "pixel_auroc_weighted", "image_auroc_from_map99")
IMAGE_KEYS = ("image_auroc", "image_ap", "accuracy", "macro_f1", "recall", "fpr", "invalid_rate")
ROI_CONDITIONS = ("original_only", "original_center_crop", "original_random_crop",
                  "original_teacher_roi", "original_teacher_roi_reference")
DENSE_GT = (("imagebind", 224), ("pre", 224), ("post", 224), ("pre", 448), ("post", 448))
DENSE_KD = (("pre", 448), ("post", 448))
JSON_BASENAMES = {"run.json", "plan.json", "val.json", "report.json", "train_complete.json",
                  "complete.json", "best_validation_reloaded.json", "metrics.json",
                  "val_metrics.json", "failure.json", "teacher_val.json",
                  "parity_gate.json", "auxiliary_cache_check.json", "selected_auxiliary_cache_check.json",
                  "native_parity.json", "teacher_cache.json"}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nested(value, path, default=None):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    if not path.is_file():
        return None, {"path": str(path), "state": "missing"}
    try:
        raw = path.read_bytes()
        def reject_constant(value):
            raise ValueError("Nonfinite JSON constant: " + value)
        value = json.loads(raw, parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as error:
        # Atomic writers normally avoid this; a live copy can still be partial.
        return None, {"path": str(path), "state": "ongoing_unreadable", "error": str(error)}
    if not isinstance(value, dict):
        return None, {"path": str(path), "state": "invalid", "error": "Expected JSON object"}
    return value, {"path": str(path), "state": "read", "sha256": hashlib.sha256(raw).hexdigest()}


def load_documents(root, output=None):
    documents, inventory = {}, []
    if not root.is_dir():
        return documents, [{"path": str(root), "state": "missing_directory"}]
    for path in sorted(root.rglob("*.json")):
        if output is not None and path.resolve() == output.resolve():
            continue
        name = path.name
        if (name not in JSON_BASENAMES and not name.endswith("_status.json")
                and not (name.startswith("report") and name.endswith(".json"))
                and not name.endswith("_metrics.json")):
            continue
        # Never inspect prediction arrays or historical test metric files.
        if "prediction" in name or "test" in name.lower():
            continue
        value, info = read_json(path)
        inventory.append(info)
        if value is not None:
            documents[path] = value
    return documents, inventory


def metric_subset(value, keys=PIXEL_KEYS):
    return {key: value.get(key) if isinstance(value, dict) else None for key in keys}


def finite_metrics(value, keys):
    return isinstance(value, dict) and all(numeric(value.get(key)) for key in keys)


def image_channels(value):
    overall = value.get("overall", {}) if isinstance(value, dict) else {}
    return {"scores": metric_subset(overall.get("scores"), IMAGE_KEYS),
            "generated": metric_subset(overall.get("generated"), IMAGE_KEYS),
            "score_n": nested(overall, "scores.n"),
            "generated_n": nested(overall, "generated.n")}


def status(errors, pending):
    return "invalid" if errors else "ongoing" if pending else "complete"


def scope(n, full_n, split):
    if split != "val":
        return "unknown_or_nonval"
    if isinstance(n, int) and isinstance(full_n, int) and 0 < n < full_n:
        return "pilot"
    if isinstance(n, int) and n == full_n and n > 0:
        return "full_validation"
    return "unknown"


def unique_result(candidates):
    if not candidates:
        return {"status": "ongoing", "pending": ["No matching result JSON"], "candidates": []}
    complete = [value for value in candidates if value["status"] == "complete"]
    if len(complete) > 1:
        return {"status": "ambiguous", "pending": ["Multiple complete runs; choose explicitly instead of best-run selection"],
                "candidates": candidates}
    if len(candidates) == 1:
        return candidates[0]
    return {"status": complete[0]["status"] if complete else "invalid" if any(x["status"] == "invalid" for x in candidates) else "ongoing",
            "selected_complete": complete[0] if complete else None, "candidates": candidates,
            "selection_rule": "sole completed contract only; all other candidates retained"}


def selected_result(value):
    return value.get("selected_complete", value) if value.get("status") == "complete" else None


def summarize_gates(documents):
    gates = []
    seen = {}
    for path, report in documents.items():
        version = str(report.get("pipeline_version", ""))
        if "parity" not in version or "cases" not in report:
            continue
        fingerprint = json_sha(report)
        if fingerprint in seen:
            seen[fingerprint]["mirrored_paths"].append(str(path))
            continue
        ran = report.get("mode") == "run"
        cases = report.get("cases", [])
        passed = ran and report.get("passed") is True and bool(cases) and all(case.get("passed") is True for case in cases)
        if report.get("selected_cache_indices") is not None:
            passed = passed and len(cases) == len(report["selected_cache_indices"])
        case_summaries = []
        for case in report.get("cases", []):
            checks = case.get("checks", {})
            case_summaries.append({"cache_index": case.get("cache_index"), "item": case.get("item"),
                                   "passed": case.get("passed"), "auxiliary_tokens": case.get("auxiliary_tokens"),
                                   "image_grid_thw": case.get("image_grid_thw"),
                                   "check_pass_flags": {name: check.get("passed") for name, check in checks.items() if isinstance(check, dict)},
                                   "same_shape_logits": {name: checks.get(name) for name in ("prefill_logits", "cached_step_logits", "full_replay_logits")},
                                   "official_cached_vs_official_full_replay": case.get("official_cached_vs_official_full_replay"),
                                   "manual_cached_vs_manual_full_replay": case.get("manual_cached_vs_manual_full_replay")})
        diagnostics = [c["official_cached_vs_official_full_replay"] for c in case_summaries
                       if isinstance(c["official_cached_vs_official_full_replay"], dict)]
        differences = [d["max_abs_diff"] for d in diagnostics if numeric(d.get("max_abs_diff"))]
        gates.append({"path": str(path), "pipeline_version": version,
                      "status": "complete" if passed else "failed" if ran and (report.get("error") or report.get("status") == "failed") else "ongoing",
                      "passed_recorded": report.get("passed"), "mode": report.get("mode"),
                      "cached_decoding_exercised": report.get("all_cases_exercised_cached_decoding"),
                      "n_cases": len(report.get("cases", [])),
                      "bridge_script_sha256": report.get("bridge_script_sha256"),
                      "script_sha256": report.get("script_sha256"),
                      "source_files": report.get("source_files"),
                      "preceding_parity_report": report.get("preceding_parity_report"),
                      "gate_definition": report.get("gate_definition"),
                      "diagnostic_definition": report.get("diagnostic_definition"),
                      "case_summaries": case_summaries,
                      "official_cached_full_max_abs_range": [min(differences), max(differences)] if differences else None,
                      "runtime": report.get("runtime"), "error": report.get("error"),
                      "interpretation": "Recorded gate outcome only; distinct gate versions/cases are not interchangeable"})
        gates[-1].update(source_report_sha256=fingerprint, mirrored_paths=[])
        seen[fingerprint] = gates[-1]
    return gates


def summarize_memory(documents, expected_val_n):
    records = []
    for path, report in documents.items():
        if report.get("pipeline_version") != "qwen3vl-bf16-normal-memory-square-v1":
            continue
        config = report.get("config", {})
        for resolution in report.get("resolutions", []):
            if not isinstance(resolution, dict):
                continue
            for key, metrics in resolution.get("metrics", {}).items():
                errors, pending = [], []
                if not isinstance(metrics, dict):
                    continue
                n = metrics.get("n_images")
                if metrics.get("split") != "val" or report.get("test_set_used") is not False:
                    errors.append("Validation-only/test-unused contract missing or false")
                if n != expected_val_n:
                    errors.append("Expected full validation n=%d, got %r" % (expected_val_n, n))
                if not finite_metrics(metrics, PIXEL_KEYS):
                    pending.append("Required pixel/image-map metrics not yet finite")
                if nested(report, "source_audit.passed") is not True:
                    errors.append("Source audit did not pass")
                if report.get("complete") is not True:
                    pending.append("Full visual report is not complete")
                if report.get("error"):
                    errors.append("Visual report has a recorded error")
                feature = metrics.get("feature", key.split(":")[0])
                shot = metrics.get("shots")
                if shot is None:
                    try:
                        shot = int(key.split(":")[1])
                    except (ValueError, IndexError):
                        errors.append("Unrecognized feature:shot key")
                records.append({"status": status(errors, pending), "errors": errors, "pending": pending,
                                "path": str(path), "feature": feature, "shots": shot, "seed": config.get("seed"),
                                "requested_size": resolution.get("requested_square_resolution"),
                                "actual_geometries": resolution.get("actual_geometries"),
                                "scope": "full_validation", "split": "val", "n": n,
                                "metrics": metric_subset(metrics), "head_parameters": 0,
                                "precision": {"encoder": "BF16", "bank_and_cosine": "FP32"},
                                "teacher_used": False, "support_definition": "same-category training normals only; nested16/64",
                                "manifest_sha256": nested(report, "data.manifest_sha256"),
                                "source_sha256": nested(report, "source_audit.source_sha256"),
                                "report_complete": report.get("complete"), "report_error": report.get("error"),
                                "peak_gpu_allocated_bytes_report": report.get("peak_gpu_allocated_bytes"),
                                "memory": {category: values.get(key) for category, values in resolution.get("memory", {}).items()}})
    matrix = []
    for size in (224, 448, 672):
        for feature in ("pre", "post"):
            for shots in (16, 64):
                matching = [r for r in records if (r["requested_size"], r["feature"], r["shots"], r["seed"]) == (size, feature, shots, 42)]
                matrix.append({"requested_size": size, "feature": feature, "shots": shots, "seed": 42,
                               "result": unique_result(matching)})
    return {"status": "complete" if all(x["result"]["status"] == "complete" for x in matrix) else "ongoing",
            "expected_cells": 12, "matrix": matrix, "observed_records": records}


def dense_parameter_count(config):
    channels, width = config.get("in_channels"), config.get("width")
    if not isinstance(channels, int) or not isinstance(width, int):
        return None
    return channels * width + 1153 * width + 74497


def inspect_dense(path, run, documents, expected_val_n):
    directory = path.parent
    config = run.get("config", {})
    compatibility = run.get("compatibility", {})
    complete = documents.get(directory / "train_complete.json")
    reload = documents.get(directory / "best_validation_reloaded.json")
    errors, pending = [], []
    if not complete:
        pending.append("train_complete.json missing")
    if not reload:
        pending.append("best_validation_reloaded.json missing")
    if directory / "failure.json" in documents:
        errors.append("failure.json present")
    if run.get("test_set_used") is not False or run.get("validation_images") != expected_val_n:
        errors.append("Expected full validation with unused test set")
    if nested(run, "source_audit.passed") is not True:
        errors.append("Source audit did not pass")
    if not run.get("matched_run_key") or not run.get("train_plan_sha256"):
        pending.append("Matched run key/plan SHA missing")
    metrics = {} if reload is None else reload.get("validation_reloaded", {})
    if reload:
        if reload.get("passed") is not True or reload.get("split") != "val":
            errors.append("Best checkpoint reload did not pass on val")
        if metrics.get("split") != "val" or metrics.get("teacher_used") is not False or metrics.get("n_images") != expected_val_n:
            errors.append("Reloaded metrics must cover full val without teacher")
        if not finite_metrics(metrics, PIXEL_KEYS):
            pending.append("Reloaded required metrics are missing/nonfinite")
    if complete:
        for key, expected in (("steps", config.get("steps")), ("matched_run_key", run.get("matched_run_key")), ("test_set_used", False)):
            if complete.get(key) != expected:
                errors.append("Completion/run mismatch: " + key)
        if reload and complete.get("selected_best_step") != reload.get("selected_best_step"):
            errors.append("Selected best step mismatch")
    teacher = run.get("distillation", {})
    weight = teacher.get("weight", config.get("distill_weight"))
    if not numeric(weight) or weight < 0:
        errors.append("Invalid distillation weight")
    if numeric(weight) and weight > 0 and (not teacher.get("manifest_sha256") or not teacher.get("teacher_checkpoint")):
        pending.append("KD teacher manifest/checkpoint provenance missing")
    recorded_params = run.get("head_parameters")
    derived_params = dense_parameter_count(compatibility.get("head_config", {}))
    if recorded_params is not None and derived_params is not None and recorded_params != derived_params:
        errors.append("Recorded head parameter count differs from configured SpatialHead")
    return {"status": status(errors, pending), "path": str(path), "directory": str(directory),
            "errors": errors, "pending": pending, "feature": compatibility.get("features"),
            "requested_size": compatibility.get("size"), "seed": config.get("seed"),
            "distill_weight": weight, "arm": "GT" if weight == 0 else "GT_KD",
            "split": "val", "scope": "full_validation", "n": metrics.get("n_images"),
            "metrics": metric_subset(metrics), "best_step": None if reload is None else reload.get("selected_best_step"),
            "head_parameters_recorded": recorded_params, "head_parameters_derived": derived_params,
            "precision_recorded": run.get("numerical_protocol"),
            "teacher": teacher, "feature_source": run.get("feature_source"),
            "matched_run_key": run.get("matched_run_key"), "plan_sha256": run.get("train_plan_sha256"),
            "compatibility": compatibility, "config": config,
            "elapsed_seconds": None if complete is None else complete.get("elapsed_seconds"),
            "peak_gpu_allocated_bytes": None if complete is None else complete.get("peak_gpu_allocated_bytes")}


def summarize_dense(documents, expected_val_n, kd_weight):
    records = [inspect_dense(path, run, documents, expected_val_n) for path, run in documents.items()
               if path.name == "run.json" and run.get("config", {}).get("run") is True and run.get("pipeline_version") in {
                   "qwen3vl-frozen-singlelayer-spatialhead-v1", "imagebind-fp16-lastlayer-spatialhead-v1"}]
    matrix, pairs = [], []
    for feature, size in DENSE_GT:
        arms = {}
        expected_arms = (("GT", 0), ("GT_KD", kd_weight)) if (feature, size) in DENSE_KD else (("GT", 0),)
        for arm, weight in expected_arms:
            matching = [r for r in records if (r["feature"], r["requested_size"], r["seed"], r["distill_weight"]) == (feature, size, 42, weight)]
            arms[arm] = unique_result(matching)
            matrix.append({"feature": feature, "requested_size": size, "seed": 42,
                           "arm": arm, "distill_weight": weight, "result": arms[arm]})
        if (feature, size) not in DENSE_KD:
            continue
        gt, kd = selected_result(arms["GT"]), selected_result(arms["GT_KD"])
        checks = {}
        if gt and kd:
            for key in ("matched_run_key", "plan_sha256", "compatibility", "head_parameters_derived"):
                checks[key] = gt[key] is not None and gt[key] == kd[key]
            for key in ("seed", "steps", "bs", "lr", "val_every"):
                checks["config." + key] = gt["config"].get(key) is not None and gt["config"].get(key) == kd["config"].get(key)
        pair_status = "complete" if gt and kd and all(checks.values()) else "invalid" if gt and kd else "ongoing"
        pairs.append({"feature": feature, "requested_size": size, "seed": 42, "status": pair_status,
                      "contract_checks": checks, "delta_KD_minus_GT": {key: kd["metrics"][key] - gt["metrics"][key] for key in PIXEL_KEYS}
                      if pair_status == "complete" else None,
                      "interpretation": "Single planned seed, fixed KD weight; descriptive matched pair only"})
    return {"status": "complete" if all(p["status"] == "complete" for p in pairs) and all(x["result"]["status"] == "complete" for x in matrix) else "ongoing",
            "expected_cells": 7, "GT_cells": 5, "KD_cells": 2, "kd_weight": kd_weight, "matrix": matrix, "pairs": pairs,
            "observed_records": records,
            "comparison_boundary": "Identical head structure; input widths/parameter counts and FP16/BF16 encoder precision differ across feature sources. No capacity-matched or pure-encoder causal claim."}


def summarize_native(documents, expected_val_n):
    records = []
    for path, run in documents.items():
        if path.name != "run.json" or run.get("pipeline_version") != "qwen3vl-native-map-increment-v1":
            continue
        directory, errors, pending = path.parent, [], []
        metrics = documents.get(directory / "val_metrics.json", {})
        complete = documents.get(directory / "complete.json")
        config = run.get("config", {})
        baseline = config.get("baseline") is True
        if not complete or complete.get("passed") is not True:
            pending.append("Final complete.json not present/passed")
        n = metrics.get("val_n", run.get("val_n"))
        full_n = metrics.get("full_validation_n", nested(run, "counts.val"))
        evidence_scope = scope(n, full_n, metrics.get("split"))
        if evidence_scope != "full_validation" or n != expected_val_n or run.get("smoke") is True:
            pending.append("Full planned validation run not complete (pilot/smoke is separate)")
        conditions = ("real_map",) if baseline else ("real_map", "zero_map", "donor_opposite_label_map", "spatial_shuffle_map")
        channels = {name: image_channels(value) for name, value in metrics.get("conditions", {}).items()}
        if any(name not in channels for name in conditions):
            pending.append("Native control conditions incomplete")
        if metrics.get("generated_n") != n or metrics.get("generated_is_subset") is not False:
            pending.append("Full selected-run generation incomplete")
        if run.get("selection_split") != "val":
            errors.append("Native selection must be val")
        for gate_name in ("auxiliary_cache_check",):
            if nested(run, gate_name + ".passed") is not True:
                pending.append("Recorded native auxiliary/cache check not passed: " + gate_name)
        records.append({"status": status(errors, pending), "errors": errors, "pending": pending,
                        "path": str(path), "baseline": baseline, "seed": config.get("seed"),
                        "scope": evidence_scope, "n": n, "full_validation_n": full_n,
                        "precision": run.get("precision"), "trainable_parameters": run.get("trainable_parameters"),
                        "teacher_used": not baseline, "teacher_decoder_sha256": run.get("decoder_sha256"),
                        "cache_sha256": run.get("cache_sha256"), "conditions": channels,
                        "generated_real_map": image_channels(metrics.get("generated_real_map")),
                        "selected_step": metrics.get("selected_step"),
                        "selected_auxiliary_cache_check": metrics.get("selected_auxiliary_cache_check"),
                        "parity_report_sha256": run.get("parity_report_sha256"),
                        "model_manifest_sha256": run.get("model_manifest_sha256"),
                        "source_sha256": run.get("source_sha256"),
                        "estimand": run.get("estimand"), "completion": complete})
    matrix = [{"baseline": baseline, "seed": 42, "result": unique_result([r for r in records if r["baseline"] == baseline and r["seed"] == 42 and r["scope"] == "full_validation"])}
              for baseline in (True, False)]
    return {"status": "complete" if all(x["result"]["status"] == "complete" for x in matrix) else "ongoing",
            "matrix": matrix, "observed_records": records,
            "comparison_boundary": "Native baseline versus native plus learned map tokens; no direct native-vision effect can be isolated from phase1."}


def summarize_roi(documents, expected_val_n):
    records = []
    for path, run in documents.items():
        if path.name not in {"run.json", "plan.json"} or run.get("pipeline_version") != "qwen3vl-native-roi-val-v1":
            continue
        if path.name == "plan.json" and path.with_name("run.json") in documents:
            continue
        n, full_n = run.get("n"), run.get("full_validation_n")
        evidence_scope = scope(n, full_n, run.get("split"))
        for condition in run.get("conditions", []):
            metrics = run.get("results", {}).get(condition) or documents.get(path.parent / (condition + "_metrics.json"))
            errors, pending = [], []
            if run.get("split") != "val" or full_n != expected_val_n:
                errors.append("Expected val-only source with full_validation_n=%d" % expected_val_n)
            if not metrics:
                pending.append("Condition metrics not written")
            elif metrics.get("n") != n:
                errors.append("Condition count differs from run selection")
            if evidence_scope not in {"pilot", "full_validation"}:
                pending.append("Cannot classify pilot versus full validation")
            if run.get("status") == "failed":
                errors.append("ROI run status failed")
            elif run.get("status") != "complete":
                pending.append("Full ROI run is not complete")
            record = {"status": status(errors, pending), "errors": errors, "pending": pending,
                      "path": str(path), "scope": evidence_scope, "split": "val", "n": n,
                      "condition": condition, "run_status": run.get("status"),
                      "metrics": image_channels({} if not metrics else metrics.get("metrics", {})),
                      "precision": nested(run, "runtime.precision"), "trainable_parameters": 0,
                      "teacher_used_for_roi": "teacher" in condition,
                      "teacher_cache_audit": run.get("cache_audit"),
                      "reference_selection": run.get("reference_definition") if condition.endswith("reference") else None,
                      "selected_manifest_sha256": run.get("selected_manifest_sha256"),
                      "score_definition": run.get("score_definition"),
                      "actual_image_grids": None if not metrics else metrics.get("actual_image_grids"),
                      "visual_tokens_total_min_max": None if not metrics else metrics.get("visual_tokens_total_min_max"),
                      "elapsed_seconds": None if not metrics else metrics.get("elapsed_seconds"),
                      "peak_gpu_allocated_bytes": None if not metrics else metrics.get("peak_gpu_allocated_bytes")}
            if metrics and (record["metrics"]["score_n"] != n or record["metrics"]["generated_n"] != n):
                record["errors"].append("Score/generation channel count differs from selected n")
                record["status"] = "invalid"
            if metrics and (not finite_metrics(record["metrics"]["scores"], IMAGE_KEYS)
                            or not finite_metrics(record["metrics"]["generated"], ("accuracy", "macro_f1", "recall", "fpr", "invalid_rate"))):
                record["pending"].append("Required score/generation metrics not finite")
                record["status"] = status(record["errors"], record["pending"])
            records.append(record)
    matrices = {}
    for evidence_scope in ("pilot", "full_validation"):
        matrix = [{"condition": condition, "result": unique_result([r for r in records if r["scope"] == evidence_scope and r["condition"] == condition])}
                  for condition in ROI_CONDITIONS]
        matrices[evidence_scope] = {"status": "complete" if all(x["result"]["status"] == "complete" for x in matrix) else "ongoing", "matrix": matrix}
    return {**matrices, "observed_records": records,
            "comparison_boundary": "Pilot and full val are different evidence surfaces. Teacher ROI is external localization. Reference adds a third image/token budget; factual explanation quality is not measured."}


def performance_model():
    """Analytical array sizes and exposure bounds; deliberately no runtime estimate."""
    capacity = 1024 ** 3
    dense = []
    for feature, size, h, w, c, params, dtype in (
        ("imagebind", 224, 16, 16, 1280, 697345, "FP16"),
        ("pre", 224, 14, 14, 1152, 664577, "BF16"),
        ("post", 224, 7, 7, 4096, 1418241, "BF16"),
        ("pre", 448, 28, 28, 1152, 664577, "BF16"),
        ("post", 448, 14, 14, 4096, 1418241, "BF16")):
        nbytes = h * w * c * 2
        dense.append({"feature": feature, "requested_size": size, "feature_hw_channels": [h, w, c],
                      "feature_bytes_per_image": nbytes, "one_GiB_cache_image_capacity_floor": capacity // nbytes,
                      "eight_GiB_runner_cache_image_capacity_floor": 8 * capacity // nbytes,
                      "head_parameters": params, "encoder_and_cache_dtype": dtype, "head_dtype": "FP32",
                      "optimizer_state_parameter_storage_bytes_approx": params * 16})
    pixels = 380 * 224 * 224
    memory = []
    for requested, pre_side, post_side in ((224, 14, 7), (448, 28, 14), (672, 42, 21)):
        pre_patches, post_patches = min(pre_side, 14) ** 2, min(post_side, 14) ** 2
        bank_bytes = sum(shot * (pre_patches * 1152 + post_patches * 4096) * 4 for shot in (16, 64))
        memory.append({"requested_size": requested, "actual_hw_explicit_no_resize": [pre_side * 16] * 2,
                       "pre_hw": [pre_side] * 2, "post_hw": [post_side] * 2,
                       "one_category_four_banks_FP32_bytes": bank_bytes,
                       "encoder_forwards_upper_bound_12classes64support380val": 12 * 64 + 380})
    return {"evidence": "Code-derived bounds/array payloads, not measured timing or total RSS/VRAM",
            "dense_default": {"steps": 1200, "batch_size": 4, "training_exposures": 4800,
                              "runner_cache_GiB": 8, "trainer_default_cache_GiB": 1,
                              "scheduled_validation_passes": 6, "selected_reload_validation_passes": 1,
                              "full_validation_exposures": 7 * 380,
                              "encoder_forward_upper_bound_if_all_cache_misses": 4800 + 7 * 380,
                              "risk": "Per-image tower extraction on cache misses dominates potential latency; each validation inserts380 images and evicts training entries. A1GiB cache bounds payload, not hit rate.",
                              "features": dense, "validation_score_and_GT_float32_payload_bytes": 2 * pixels * 4},
            "load_and_activation_notes": ["Qwen loader instantiates only indexed model.visual weights, not the8B language model. Frozen extraction is inference_mode; head alone retains backward activations.",
                                           "ImageBind constructor creates the full multi-modality model and strict-loads a4,803,584,173-byte checkpoint on CPU before pruning unused modalities and moving vision FP16 to GPU. Simultaneous constructor/state tensors can require roughly twice checkpoint tensor payload plus overhead in host RAM; this is not measured peakRSS.",
                                           "SpatialHead optimizer/parameter storage is small relative to the visual tower, but FP32 spatial activations and convolution workspace add GPU memory beyond the parameter estimate."],
            "normal_memory": {"settings": memory, "all_three_resolution_encoder_forward_upper_bound": 3 * (12 * 64 + 380),
                              "similarity_matrix_chunk_FP32_bytes": 256 * 2048 * 4,
                              "four_scoremaps_plus_shared_GT_float32_payload_bytes": 5 * pixels * 4,
                              "risk": "One category's banks reside on GPU at a time; sklearn ranking/sorting and intermediate concatenations add CPU memory beyond array payload. No fixed completion time follows from source inspection."},
            "execution_recommendation": ["Run the12 memory cells in one invocation to share pre/post features and nested16/64 supports.",
                                         "Run dense arms sequentially with the identical seed42 plan,1200steps,B4,LR1e-3,val-every200. Keep KD weight fixed before results.",
                                         "Use logged elapsed time and recorded peak allocation from actual runs; do not infer measured runtime or OOM safety from these payload sizes."]}


def minimum_matrix(kd_weight):
    return {"seed_policy": "Seed42 is the minimum exploratory matrix; paired43/44 replication is follow-up, not silently substituted or best-seed selected.",
            "normal_memory": {"cells": 12, "requested_resolutions": [224, 448, 672], "features": ["pre", "post"],
                              "training_normal_supports": [16, 64], "seed": 42,
                              "single_invocation_args": ["--resolutions", "224", "448", "672", "--features", "pre", "post", "--shots", "16", "64", "--seed", "42"],
                              "outcome_surface": "full val380; pixel macro AP/AUROC and map-p99 image AUROC"},
            "dense": {"cells": 7, "GT_feature_resolution": [[feature, size] for feature, size in DENSE_GT],
                      "KD_feature_resolution": [[feature, size] for feature, size in DENSE_KD],
                      "arms": [{"name": "GT", "distill_weight": 0}, {"name": "GT_KD", "distill_weight": kd_weight}],
                      "seed": 42, "steps": 1200, "bs": 4, "lr": 1e-3, "val_every": 200,
                      "selection": "val macro pixel AP; strict best reload",
                      "pairing": "same feature/size/head/seed/plan/optimizer/schedule/budget; KD teacher is train-only"},
            "native": {"cells": 2, "arms": ["native baseline", "native plus18 map tokens"],
                       "required_gate": "passed native prefill/cache and same-shape auxiliary gate matching current script/source"},
            "roi": {"pilot_cells": 5, "pilot_n": 48, "full_val_followup_cells": 5,
                    "full_val_n": 380, "conditions": list(ROI_CONDITIONS),
                    "reporting": "pilot and full validation are separate; equal-view crop controls and third-view reference confound retained"},
            "phase1": {"cells": 6, "backends": ["vicuna", "qwen"], "seeds": [42, 43, 44]}}


def summarize(root, expected_val_n=380, kd_weight=.5, output=None):
    documents, inventory = load_documents(root, output)
    try:
        # This known standard-library-only helper reuses the stricter six-run audit.
        from migration_summarize_phase1 import summarize as summarize_phase1
        phase1 = summarize_phase1(root)
    except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
        phase1 = {"status": "ongoing", "pending": ["Phase1 helper could not summarize existing JSON: " + str(error)]}
    memory = summarize_memory(documents, expected_val_n)
    dense = summarize_dense(documents, expected_val_n, kd_weight)
    native = summarize_native(documents, expected_val_n)
    roi = summarize_roi(documents, expected_val_n)
    teacher = [{"path": str(path), "result": value} for path, value in documents.items()
               if path.name == "teacher_val.json" and value.get("split") == "val"]
    all_complete = all(x["status"] == "complete" for x in (phase1, memory, dense, native, roi["full_validation"]))
    gates = summarize_gates(documents)
    auxiliary = [gate for gate in gates if gate["pipeline_version"] == "qwen3vl-aux-official-same-shape-parity-v1"]
    stage2 = {"auxiliary_official_gate": unique_result(auxiliary),
              "native_training_completion": "complete" if native["status"] == "complete" else "ongoing",
              "native_run_json_count": len(native["observed_records"]),
              "interpretation": ["The same-shaped official/manual gate checks prefill, cached step, full replay, LM inputs, MRoPE, masks, and cache lengths. A passed gate is interface evidence, not training or detection performance.",
                                   "Official cached versus official full replay changes query length and BF16 arithmetic. Its difference is a diagnostic, not an official/manual parity threshold. The prior cross-shape rejection does not identify an error in the manual bridge.",
                                   "Four selected validation examples cover only recorded cases; they do not establish all-prompt correctness or trained map-branch quality."]}
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "status": "complete" if all_complete else "ongoing",
            "input_scope": "Existing JSON result/config/completion/gate files only; no checkpoints, model inference, raw predictions, test metrics, or training execution.",
            "expected_val_n": expected_val_n, "minimum_matrix": minimum_matrix(kd_weight),
            "routes": {"phase1_language_backbone": phase1, "native_map_increment": native,
                       "frozen_normal_memory": memory, "supervised_dense_GT_and_KD": dense,
                       "native_ROI": roi},
            "parity_gates": gates, "stage2_gate_findings": stage2, "teacher_val_references": teacher,
            "teacher_cache_provenance": [{"path": str(path), "result": value} for path, value in documents.items() if path.name == "teacher_cache.json"],
            "performance_model": performance_model(), "input_files": inventory,
            "stage_status_files": [{"path": str(path), "state": value.get("state"), "stage": value.get("stage"),
                                    "current_job": value.get("current_job"), "jobs": value.get("jobs")}
                                   for path, value in documents.items() if path.name.endswith("_status.json")],
            "interpretation_boundaries": ["Supervised custom60/20/20 uses real anomaly training images; this is not the official normal-only benchmark protocol.",
                                          "All score tables are validation development evidence. Pilot scores are not full-validation scores; historical test456 has been used in prior development.",
                                          "Image Yes-No candidate scores, free generation, pixel maps, and memory distances are separate outcome surfaces.",
                                          "No missing result becomes zero; no pilot, failed gate, unmatched pair, or incomplete seed aggregate becomes a completed result.",
                                          "Frozen Qwen failures/successes from one route do not establish presence or absence of anomaly information in its features."]}


def fmt(value):
    return "—" if not numeric(value) else "%.6f" % value


def moment(value):
    if not isinstance(value, dict):
        return "—"
    return fmt(value.get("mean")) + " ± " + fmt(value.get("sample_sd", value.get("sd")))


def markdown(report):
    routes = report["routes"]
    phase1 = routes["phase1_language_backbone"]
    native, memory, dense, roi = (routes[name] for name in ("native_map_increment", "frozen_normal_memory", "supervised_dense_GT_and_KD", "native_ROI"))
    lines = ["# 五路线迁移实验统一进度", "",
             "本地快照生成时间（UTC）：`%s`。状态：**%s**。仅读现有 JSON；缺失结果为 ongoing，不计为0。" % (report["created_at_utc"], report["status"]), "",
             "| 路线 | 状态 | 已有证据 / 最小完整矩阵 |",
             "|---|---|---|",
             "| 1. 语言骨干替换 | %s | Vicuna/Qwen × seeds42/43/44，完整val380 |" % phase1.get("status"),
             "| 2. 原生视觉＋18 map token | %s | native baseline＋learned map；gate与训练结果分别报告 |" % native["status"],
             "| 3. Qwen视觉特征及独立定位头 | memory %s；GT %s | memory12格；GT5格（IB224/pre224/post224/pre448/post448） |" % (memory["status"], "complete" if all(x["result"]["status"] == "complete" for x in dense["matrix"] if x["arm"] == "GT") else "ongoing"),
             "| 4. ROI / 正常参考 | pilot %s；full val %s | 原图、中心、随机、teacher ROI、teacher ROI＋参考，共5条件 |" % (roi["pilot"]["status"], roi["full_validation"]["status"]),
             "| 5. 同feature教师蒸馏 | %s | 与路线3匹配的KD2格（pre448/post448），KD权重%s |" % (dense["status"], fmt(dense["kd_weight"])), "",
             "## 第一阶段三seed", "",
             "配对完成：%s / 3；按预定seed42/43/44汇总，未挑选最好seed。数值均为验证集0–1，mean ± sample SD，n=3。" % len(phase1.get("completed_paired_seeds", [])), ""]
    aggregate = phase1.get("aggregate")
    if isinstance(aggregate, dict):
        lines.extend(["| 指标 | Vicuna | Qwen | 配对Δ（Qwen−Vicuna） |", "|---|---:|---:|---:|"])
        for key in ("image_auroc", "image_ap", "generated_macro_f1", "generated_recall", "generated_fpr", "generated_invalid_rate"):
            lines.append("| %s | %s | %s | %s |" % (key, moment(nested(aggregate, "by_backend.vicuna." + key)),
                         moment(nested(aggregate, "by_backend.qwen." + key)), moment(nested(aggregate, "paired_delta_qwen_minus_vicuna." + key))))
    else:
        lines.append("三seed完整且配置匹配之前不计算最终均值。")
    lines.extend(["", "两臂冻结语言骨干、仅训练重初始化PromptLearner与CLS投影；不使用Qwen原生视觉。详细协议和各seed控制条件保留于 `PHASE1_RESULTS.md`。", "",
                  "## 第二阶段接口门禁", "",
                  "| Gate | 状态 | 样本数 | 模式 |", "|---|---|---:|---|"])
    for gate in report["parity_gates"]:
        lines.append("| %s | %s | %d | %s |" % (gate["pipeline_version"], gate["status"], gate["n_cases"], gate.get("mode")))
    for gate in report["parity_gates"]:
        if gate.get("mirrored_paths"):
            lines.extend(["", "同一接口检查报告存在%d份内容相同的镜像，仅记一次检查；所有路径保留在JSON的mirrored_paths，不构成独立重复实验。" % len(gate["mirrored_paths"])])
    aux = selected_result(report["stage2_gate_findings"]["auxiliary_official_gate"])
    if aux is not None:
        lines.extend(["", "官方与manual的相同形状对照已通过：prefill、单步cached、完整replay的logits及LM输入逐位一致；MRoPE、mask、DeepStack输入与cache长度检查通过。", "",
                      "| Cache index | 产品 / 标签 | Prefill最大差 | Cached最大差 | Full replay最大差 | 官方cached/full最大差（诊断） |",
                      "|---:|---|---:|---:|---:|---:|"])
        for case in aux["case_summaries"]:
            lines.append("| %s | %s / %s | %s | %s | %s | %s |" % (case.get("cache_index"), nested(case, "item.category"), nested(case, "item.label"),
                         fmt(nested(case, "same_shape_logits.prefill_logits.max_abs_diff")),
                         fmt(nested(case, "same_shape_logits.cached_step_logits.max_abs_diff")),
                         fmt(nested(case, "same_shape_logits.full_replay_logits.max_abs_diff")),
                         fmt(nested(case, "official_cached_vs_official_full_replay.max_abs_diff"))))
        limits = aux.get("official_cached_full_max_abs_range")
        if limits:
            lines.extend(["", "官方自身BF16 cached/full replay的最大logit差为%s–%s。该比较改变attention query长度，记录为数值诊断；原先用此跨形状差异作桥接正确性gate导致误拒绝。相同形状官方/manual结果才直接检验桥接实现。" % (fmt(limits[0]), fmt(limits[1]))])
        lines.extend(["", "4个样本属于接口检查，不能替代训练收敛、完整验证或检测效果结论。下一token Yes/No诊断也不是图像首答案检测分数。"])
    smoke_records = [r for r in native["observed_records"] if r.get("scope") == "pilot"]
    full_records = [r for r in native["observed_records"] if r.get("scope") == "full_validation"]
    if smoke_records:
        for record in smoke_records:
            smoke_complete = isinstance(record.get("completion"), dict) and record["completion"].get("passed") is True
            lines.extend(["", "Native smoke局部检查：val%s，生成%s；complete.json=%s。该smoke属于pilot，不进入fullval380主结果。" % (record.get("n"), nested(record, "generated_real_map.generated_n"), smoke_complete)])
    if not full_records:
        lines.extend(["", "当前本地快照尚无native main完整val380结果；正在运行的远程job以stage status为准。normal memory、dense、ROI与KD无完整结果时均保留ongoing。"])
    else:
        lines.extend(["", "## 原生视觉完整验证", "",
                      "| 方案 | 状态 | Selected step | AUROC | AP | 生成macro F1 | 生成invalid |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for cell in native["matrix"]:
            result = cell["result"]
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                "native baseline" if cell["baseline"] else "native + learned map", result["status"],
                result.get("selected_step", "—"), fmt(nested(result, "conditions.real_map.scores.image_auroc")),
                fmt(nested(result, "conditions.real_map.scores.image_ap")),
                fmt(nested(result, "generated_real_map.generated.macro_f1")),
                fmt(nested(result, "generated_real_map.generated.invalid_rate"))))
        lines.extend(["", "Baseline未经过canonical回答监督，prompt未强制Yes/No首词；生成invalid与格式适配收益需结合NATIVE_PROTOCOL_AUDIT.md解读。两臂都是val开发结果。"])
    memory_records = [x for x in memory["observed_records"] if x.get("metrics")]
    if memory_records:
        lines.extend(["", "## 正常样本记忆库", "",
                      "| 分辨率 | 特征 | Shots | 状态 | Pixel AUROC | Pixel AP | Map image AUROC |",
                      "|---:|---|---:|---|---:|---:|---:|"])
        for r in memory_records:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["requested_size"], r["feature"], r["shots"], r["status"],
                fmt(r["metrics"].get("pixel_auroc_macro")), fmt(r["metrics"].get("pixel_ap_macro")),
                fmt(r["metrics"].get("image_auroc_from_map99"))))
    dense_records = [r for r in dense["observed_records"] if r.get("metrics", {}).get("pixel_ap_macro") is not None]
    if dense_records:
        lines.extend(["", "## 独立定位头", "",
                      "| 特征 | 分辨率 | 训练 | 状态 | Best step | Pixel AUROC | Pixel AP |",
                      "|---|---:|---|---|---:|---:|---:|"])
        for r in dense_records:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["feature"], r["requested_size"], r["arm"], r["status"], r.get("best_step"),
                fmt(r["metrics"].get("pixel_auroc_macro")), fmt(r["metrics"].get("pixel_ap_macro"))))
    completed_pairs = [pair for pair in dense["pairs"] if pair["status"] == "complete"]
    if completed_pairs:
        lines.extend(["", "同特征GT/KD按相同head、初始化seed、训练计划、优化器和预算匹配。差值为KD−GT，单位0–1；单seed描述性结果。", "",
                      "| 特征/分辨率 | 合同 | Δ Pixel AP | Δ Pixel AUROC | Δ Map image AUROC |",
                      "|---|---|---:|---:|---:|"])
        for pair in completed_pairs:
            delta = pair["delta_KD_minus_GT"]
            lines.append("| %s/%s | %s | %s | %s | %s |" % (pair["feature"], pair["requested_size"], pair["status"], fmt(delta.get("pixel_ap_macro")), fmt(delta.get("pixel_auroc_macro")), fmt(delta.get("image_auroc_from_map99"))))
        lines.extend(["", "KD teacher仅在训练使用；当前无teacher验证/推理结论仅覆盖独立定位头，蒸馏map尚未接入语言模型或ROI系统。"])
    roi_records = [r for r in roi["observed_records"] if r.get("metrics", {}).get("score_n") is not None]
    if roi_records:
        lines.extend(["", "## ROI及正常参考", "",
                      "| 范围 | N | 条件 | 状态 | AUROC | 生成macro F1 | Invalid |",
                      "|---|---:|---|---|---:|---:|---:|"])
        for r in roi_records:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                r["scope"], r["n"], r["condition"], r["status"], fmt(nested(r, "metrics.scores.image_auroc")),
                fmt(nested(r, "metrics.generated.macro_f1")), fmt(nested(r, "metrics.generated.invalid_rate"))))
    lines.extend(["", "## 共享教师参考", ""])
    for reference in report["teacher_val_references"]:
        value = reference["result"]
        lines.append("冻结教师单独在val%d上的pixel AUROC=%s、pixel AP=%s、map-p99 image AUROC=%s。该项是缓存参考，不是LM分数或新dense结果。" % (value.get("n", 0),
                     fmt(nested(value, "pixel.pixel_auroc_macro")), fmt(nested(value, "pixel.pixel_ap_macro")), fmt(nested(value, "pixel.image_auroc_from_map99"))))
    for cache in report["teacher_cache_provenance"]:
        value = cache["result"]
        lines.append("共享缓存：teacher seed=%s，%s/%s，batch=%s；decoder SHA `%s`。完整cache/manifest SHA保留于JSON，训练与蒸馏需核对同一教师来源。" % (
            value.get("teacher_seed"), value.get("dtype"), value.get("geometry"), value.get("cache_batch_size"), value.get("decoder_sha256")))
    lines.extend(["", "## 本轮矩阵与复现协议", "",
                  "- Normal memory：seed42，requested224/448/672 ×pre/post ×16/64；一次调用共享visual前向，16是64支持集的前缀；仅训练正常样本建库。当前extractor显式do_resize=False，实际输入等于请求分辨率；224对应pre14×14/post7×7。native LM路径仍使用官方smart resize，224请求可到256，二者分别记录。",
                  "- Dense GT：ImageBind224、pre224、post224、pre448、post448共5格；KD仅pre448与post448共2格。1200updates、B4、lr1e-3、cache8GiB、每200步val380、按macro pixel AP选best并重载。两对GT/KD须同head、seed、plan、优化器、调度和预算。",
                  "- Head参数：ImageBind697,345（FP16 tower/cache），pre664,577（BF16），post1,418,241（BF16）；head均FP32。结构相同不代表参数量或精度完全匹配。",
                  "- ROI pilot48与fullval380分别报告。teacher ROI依赖外部定位教师，参考图仅从同类别训练正常图选择；第三张参考图同时增加视觉token预算。",
                  "- 完整matrix的seed42属于探索性对照；后续43/44应做匹配复现，不能从多个seed/权重中挑最好结果。", "",
                  "## 代码推导的资源风险", "",
                  "以下是张量payload/曝光上界，不是实测耗时、总RSS或OOM保证。", "",
                  "| Dense特征 | 单图缓存bytes | 1GiB图数 | 当前runner8GiB图数 | Head参数 |",
                  "|---|---:|---:|---:|---:|"])
    for row in report["performance_model"]["dense_default"]["features"]:
        lines.append("| %s%s | %s | %s | %s | %s |" % (row["feature"], row["requested_size"], row["feature_bytes_per_image"],
                     row["one_GiB_cache_image_capacity_floor"], row["eight_GiB_runner_cache_image_capacity_floor"], row["head_parameters"]))
    lines.extend(["", "Dense1200×B4有4800次训练曝光；6轮val380和1轮best重载val380共2660次，若全cache miss最多7460次逐图visual前向。Trainer默认1GiB可能因val插入驱逐训练项；当前runner显式8GiB，是否容纳全部实际唯一训练路径＋val由training_plan决定。速度须读实际elapsed/peak指标，不能用曝光数冒充测量。",
                  "", "ImageBind在CPU构造完整多模态模型并strict load约4.80GB权重后才裁掉非视觉模块，加载瞬时CPU tensor payload可接近两份权重并另有开销；Qwen只加载visual。Head的FP32激活/卷积workspace不包含在参数量估计内。",
                  "", "Normal memory全矩阵共享提取时最多3444次visual前向；每次仅一类别四个FP32 bank驻GPU，448/672约314MiB，similarity块2MiB。CPU四组scoremap与共享GT的float32 payload约364MiB，指标排序/拼接另需内存，不能视为总RAM。",
                  "", "## 证据边界", ""])
    lines.extend("- " + text for text in report["interpretation_boundaries"])
    lines.extend(["", "源JSON路径与SHA256保存在 `all_summary.json` 的input_files；仅读结果、配置、完成状态和gate，不读取checkpoint、预测数组或test指标。", ""])
    return "\n".join(lines)


def cpu_self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        empty = summarize(root)
        assert empty["status"] == "ongoing" and len(empty["routes"]["frozen_normal_memory"]["matrix"]) == 12
        assert len(empty["routes"]["supervised_dense_GT_and_KD"]["matrix"]) == 7
        def put(relative, value):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def dense_fixture(name, weight, key="matched"):
            compatibility = {"features": "pre", "size": 448, "head_config": {"in_channels": 1152, "width": 256}}
            metrics = {**{k: .5 for k in PIXEL_KEYS}, "split": "val", "teacher_used": False, "n_images": 380}
            run = {"pipeline_version": "qwen3vl-frozen-singlelayer-spatialhead-v1", "compatibility": compatibility,
                   "source_audit": {"passed": True}, "test_set_used": False, "validation_images": 380,
                   "head_parameters": 664577, "matched_run_key": key, "train_plan_sha256": "plan",
                   "config": {"run": True, "seed": 42, "steps": 1200, "bs": 4, "lr": .001, "val_every": 200, "distill_weight": weight},
                   "distillation": {"weight": weight, "manifest_sha256": "teacher", "teacher_checkpoint": "teacher"}}
            put(name + "/run.json", run)
            put(name + "/train_complete.json", {"steps": 1200, "matched_run_key": key, "test_set_used": False, "selected_best_step": 400})
            put(name + "/best_validation_reloaded.json", {"passed": True, "split": "val", "selected_best_step": 400, "validation_reloaded": metrics})
        dense_fixture("gt", 0)
        dense_fixture("kd", .5)
        paired = summarize(root)["routes"]["supervised_dense_GT_and_KD"]["pairs"][0]
        assert paired["status"] == "complete" and all(v == 0 for v in paired["delta_KD_minus_GT"].values())
        dense_fixture("kd", .5, "wrong")
        assert summarize(root)["routes"]["supervised_dense_GT_and_KD"]["pairs"][0]["status"] == "invalid"
        channels = {"overall": {"scores": {"n": 48, **{k: .5 for k in IMAGE_KEYS}},
                                "generated": {"n": 48, **{k: .5 for k in IMAGE_KEYS}}}}
        put("roi/run.json", {"pipeline_version": "qwen3vl-native-roi-val-v1", "status": "complete", "split": "val",
                            "n": 48, "full_validation_n": 380, "conditions": ["original_only"],
                            "results": {"original_only": {"n": 48, "metrics": channels}}})
        roi = summarize(root)["routes"]["native_ROI"]
        assert roi["pilot"]["matrix"][0]["result"]["status"] == "complete"
        assert roi["full_validation"]["matrix"][0]["result"]["status"] == "ongoing"
        (root / "partial").mkdir()
        (root / "partial/run.json").write_text("{")
        report = summarize(root)
        assert any(info["state"] == "ongoing_unreadable" for info in report["input_files"])
        assert report["status"] == "ongoing"
        assert performance_model()["dense_default"]["encoder_forward_upper_bound_if_all_cache_misses"] == 7460
        json.dumps(report, allow_nan=False)
        assert "pilot" in markdown(report) and "ongoing" in markdown(report)
        assert "torch" not in __import__("sys").modules
    return {"passed": True, "missing_ongoing": True, "matched_pair_required": True,
            "pilot_fullval_separate": True, "partial_json_tolerated": True, "no_torch_or_model_import": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", help="Optional derived summary JSON; default prints JSON to stdout")
    parser.add_argument("--markdown", help="Optional derived Markdown report from the same JSON snapshot")
    parser.add_argument("--expected-val-n", type=int, default=380)
    parser.add_argument("--kd-weight", type=float, default=.5, help="Fixed exploratory matrix weight; no weight selection")
    parser.add_argument("--cpu-test-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_test_only:
        print(json.dumps(cpu_self_test(), indent=2))
        return
    if args.expected_val_n < 1 or not math.isfinite(args.kd_weight) or args.kd_weight <= 0:
        parser.error("Positive expected validation count and finite KD weight required")
    root = Path(args.root).resolve()
    out = None if args.out is None else Path(args.out).resolve()
    if out is not None and (out.name in JSON_BASENAMES or out.name.endswith("_metrics.json") or out.name.endswith("_status.json")):
        parser.error("--out must be a new summary filename, not a source result/config filename")
    report = summarize(root, args.expected_val_n, args.kd_weight, out)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if out is None:
        print(rendered)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(out)
        print(json.dumps({"status": report["status"], "summary": str(out),
                          "input_files": len(report["input_files"]), "gpu_used": False}))
    if args.markdown:
        path = Path(args.markdown).resolve()
        if path.suffix.lower() not in {".md", ".markdown"}:
            parser.error("--markdown requires a Markdown output filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(markdown(report), encoding="utf-8")
        temporary.replace(path)


if __name__ == "__main__":
    main()
