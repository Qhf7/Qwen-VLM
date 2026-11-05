"""Audit and analyze the completed train-normal curation / resolution study.

Ordinary Python only. No torch, model, GPU, or dataset image is opened. Fails
closed before replacing reports when any required run is absent/inconsistent.
The final confirmation set is used only as a source-ID/hash exclusion list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from industrial_vlm.compare import _load_run, _quantile  # noqa: E402
from industrial_vlm.evaluate import _atomic_json, _sha256  # noqa: E402
from industrial_vlm.metrics import _summary, compute_metrics, parse_prediction  # noqa: E402
from industrial_vlm.score import (  # noqa: E402
    SCHEMA_VERSION as SCORE_SCHEMA, _canonical_hash, _read_scores,
    normalized_score, ranking_metrics, threshold_metrics,
)

ARMS = ("normal_random", "normal_hard", "region_highres")
LOW, HIGH = 524288, 1048576
SPECS = {
    "normal_random_decision": ("normal_random", "decision", LOW),
    "normal_hard_decision": ("normal_hard", "decision", LOW),
    "region_low_low_region": ("region_lowres", "region", LOW),
    "region_low_high_region": ("region_lowres", "region", HIGH),
    "region_high_low_region": ("region_highres", "region", LOW),
    "region_high_high_region": ("region_highres", "region", HIGH),
}
GEN_METRICS = ("macro_f1", "fpr", "anomaly_recall", "union_box_recall_iou_0.5")
MATCHED_TRAIN = (
    "model", "adapter", "max_steps", "seed", "sample_order", "max_length",
    "micro_batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "max_grad_norm", "warmup_steps", "lora_r",
    "lora_alpha", "lora_dropout", "loss", "coordinate_format", "versions",
)
BOUNDARY_AUDIT = "native full token IDs == prompt IDs + offset-verified continuation IDs"


def obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same(a: Any, b: Any) -> bool:
    if isinstance(b, float):
        return type(a) in (int, float) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(b, dict):
        return isinstance(a, dict) and set(a) == set(b) and all(same(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def source_hash(source_id: str) -> str:
    match = re.search(r"_([a-f0-9]{64})$", source_id)
    if not match:
        raise ValueError(f"No source SHA256 suffix: {source_id}")
    return match.group(1)


def sequence_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256("".join(r["source_id"] + "\n" for r in records).encode()).hexdigest()


def local_path(root: Path, value: str) -> Path:
    """Translate the recorded remote project prefix, never arbitrary paths."""
    path = Path(value)
    if not path.is_absolute():
        candidate = root / path
    elif path.is_relative_to(root):
        candidate = path
    elif "industrial_vlm" in path.parts:
        index = max(i for i, part in enumerate(path.parts) if part == "industrial_vlm")
        candidate = root.joinpath(*path.parts[index + 1:])
    else:
        raise ValueError(f"Recorded artifact is outside the project: {value}")
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Artifact escapes project root: {value}")
    return candidate


class Audit:
    def __init__(self, root: Path):
        self.root, self.checks, self.inputs = root, {}, set()

    def check(self, name: str, condition: Any) -> None:
        self.checks[name] = bool(condition)
        if not condition:
            raise ValueError("v3 analysis audit failed: " + name)

    def read(self, path: Path) -> dict[str, Any]:
        self.inputs.add(path)
        return obj(path)

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        self.inputs.add(path)
        return rows(path)

    def adapter(self, label: str, directory: Path, files: Any) -> None:
        self.check(label + ":adapter_inventory", isinstance(files, dict) and
                   bool(files) and "adapter_model.safetensors" in files)
        for filename, digest in files.items():
            self.check(label + ":adapter_safe_name:" + filename, Path(filename).name == filename)
            path = directory / filename
            self.inputs.add(path)
            self.check(label + ":adapter_bytes:" + filename, _sha256(path) == digest)

    def artifacts(self) -> dict[str, str]:
        return {str(p.relative_to(self.root)) if p.is_relative_to(self.root) else str(p): _sha256(p)
                for p in sorted(self.inputs)}


def unique_sources(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        source = record["source_id"]
        if source in output:
            for field in ("image", "category", "is_anomalous", "anomaly_area_fraction", "boxes"):
                if output[source].get(field) != record.get(field):
                    raise ValueError(f"Repeated source ground truth differs: {source}: {field}")
        else:
            output[source] = record
    return output


def train_audit(audit: Audit) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root, summaries, prepared, configs = audit.root, {}, {}, {}
    for arm in ARMS + ("region_lowres",):
        directory = root / ("runs/multitask_v2/region_region" if arm == "region_lowres" else f"runs/curation_v3/{arm}")
        summary = audit.read(directory / "summary.json")
        config = audit.read(directory / "run_config.json")
        parameter = audit.read(directory / "parameter_audit.json")
        trace_path = directory / "exposures.jsonl"
        trace = audit.read_rows(trace_path)
        data_path = local_path(root, config["train_file"])
        data = audit.read_rows(data_path)
        expected_pixels = HIGH if arm == "region_highres" else LOW
        audit.check(arm + ":complete_reload", summary.get("status") == "complete" and
                    summary.get("reload_check", {}).get("passed") is True)
        audit.check(arm + ":budget", (summary.get("optimizer_steps"), summary.get("microsteps"), len(trace), len(data))
                    == (192, 1536, 1536, 1536))
        audit.check(arm + ":fresh_sequential", config.get("adapter") is None and
                    config.get("sample_order") == "sequential" and config.get("seed") == 42 and
                    config.get("max_steps") == 192 and config.get("gradient_accumulation_steps") == 8 and
                    config.get("max_pixels") == expected_pixels)
        audit.check(arm + ":training_data_hash", config.get("train_sha256") == _sha256(data_path))
        counts = Counter(r["source_id"] for r in trace)
        audit.check(arm + ":two_exposures_per_source", len(counts) == 768 and set(counts.values()) == {2})
        expected_task = "region" if arm.startswith("region") else "decision"
        totals = Counter()
        for index, (actual, planned) in enumerate(zip(trace, data)):
            task = "region" if "boxes" in json.loads(planned["answer"]) else "decision"
            expected = (index // 8 + 1, index % 8, planned["id"], planned["source_id"], task)
            values = tuple(actual.get(k) for k in ("optimizer_step", "microstep", "id", "source_id", "task"))
            if values != expected or task != expected_task:
                raise ValueError(f"{arm}: exposure {index} does not match its frozen prepared row/task")
            for key in ("target_tokens", "input_tokens"):
                if type(actual.get(key)) is not int or actual[key] <= 0:
                    raise ValueError(f"{arm}: invalid {key} at exposure {index}")
                totals[key] += actual[key]
        audit.check(arm + ":actual_exposure_sequence", True)
        audit.check(arm + ":trace_hashes", summary.get("source_exposure_sha256") == sequence_hash(trace) and
                    summary.get("exposure_trace_sha256") == _sha256(trace_path))
        audit.check(arm + ":token_totals", totals["target_tokens"] == summary.get("total_target_tokens") and
                    totals["input_tokens"] == summary.get("total_input_tokens"))
        initial = summary.get("initial_trainable_sha256")
        audit.check(arm + ":initial_weight_audit", isinstance(initial, str) and len(initial) == 64 and
                    initial == parameter.get("initial_trainable_sha256"))
        sources = unique_sources(data)
        audit.check(arm + ":balanced_training", Counter(r["is_anomalous"] for r in sources.values()) ==
                    Counter({False: 384, True: 384}) and
                    len({r["category"] for r in sources.values()}) == 12 and
                    set(Counter((r["category"], r["is_anomalous"]) for r in sources.values()).values()) == {32})
        summaries[arm] = {**summary, "unique_sources": len(sources), "run_config": config,
                          "directory": str(directory.relative_to(root)),
                          "train_manifest": str(data_path.relative_to(root))}
        configs[arm], prepared[arm] = config, data
    reference = configs["normal_random"]
    for arm, config in configs.items():
        audit.check(arm + ":matched_optimizer_and_runtime", all(config.get(k) == reference.get(k) for k in MATCHED_TRAIN))
    audit.check("all_training_same_initial_weights", len({s["initial_trainable_sha256"] for s in summaries.values()}) == 1)
    a, b = prepared["normal_random"], prepared["normal_hard"]
    audit.check("normal_pair_category_label_schedule",
                [(r["category"], r["is_anomalous"]) for r in a] ==
                [(r["category"], r["is_anomalous"]) for r in b])
    audit.check("normal_pair_identical_anomaly_sequence", [r["source_id"] for r in a if r["is_anomalous"]] ==
                [r["source_id"] for r in b if r["is_anomalous"]])
    high, low = prepared["region_highres"], prepared["region_lowres"]
    audit.check("resolution_same_source_sequence", [r["source_id"] for r in high] == [r["source_id"] for r in low])
    audit.check("resolution_same_targets_prompts", all(all(a.get(k) == b.get(k) for k in
                ("answer", "prompt", "boxes", "category", "is_anomalous", "anomaly_area_fraction")) for a, b in zip(high, low)))
    fixed_anomalies = {s for s, r in unique_sources(low).items() if r["is_anomalous"]}
    for arm in ("normal_random", "normal_hard"):
        audit.check(arm + ":fixed_v2_anomalies", {s for s, r in unique_sources(prepared[arm]).items() if r["is_anomalous"]} == fixed_anomalies)
    return summaries, prepared


def small_definition(audit: Audit, prepared: dict[str, list[dict[str, Any]]], plan: dict[str, Any]) -> dict[str, Any]:
    values = [r["anomaly_area_fraction"] for r in unique_sources(prepared["region_lowres"]).values() if r["is_anomalous"]]
    audit.check("small_threshold_384_training_mask_fractions", len(values) == 384 and
                all(type(x) in (int, float) and math.isfinite(x) and 0 < x <= 1 for x in values))
    threshold = _quantile(sorted(values), 0.25)
    recorded = plan.get("small_area_threshold_train_anomaly_q1")
    if recorded is not None:
        if isinstance(recorded, dict):
            recorded = recorded.get("threshold", recorded.get("value"))
        audit.check("small_threshold_matches_frozen_plan", same(recorded, threshold))
    return {"threshold": threshold, "comparison": "0 < anomaly_area_fraction <= threshold",
            "quantile": 0.25, "method": "linear interpolation at (n-1)*q",
            "n_unique_training_anomalies": len(values),
            "origin": "fixed 384 v2 region-training anomaly sources; GT mask-area fraction",
            "uses_validation_for_threshold": False}


def generation_audit(audit: Audit, training: dict[str, Any], small: dict[str, Any],
                     manifest: dict[str, list[dict[str, Any]]], confirmation: set[str]) -> dict[str, Any]:
    root = audit.root
    references = audit.read(root / "experiments/curation_v3/evaluation_refs.json")
    # Both a flat mapping and an explicit named mapping are supported; values
    # themselves must resolve to the exact preregistered directories below.
    refs = references.get("evaluations", references.get("refs", references))
    output, shared_policy = {}, None
    for name, (arm, task, pixels) in SPECS.items():
        expected_relative = ("evaluations/multitask_v2/region_region_region" if name == "region_low_low_region"
                             else f"evaluations/curation_v3/{name}")
        value = refs.get(name)
        if isinstance(value, dict):
            value = value.get("directory", value.get("path", value.get("reference")))
        audit.check(name + ":reference_manifest", isinstance(value, str) and
                    local_path(root, value) == root / expected_relative)
        directory = root / expected_relative
        run, error = _load_run(directory.parent, directory.name)
        if run is None:
            raise ValueError(name + ": " + str(error))
        run["name"] = name
        for filename in ("meta.json", "metrics.json", "predictions.jsonl"):
            audit.inputs.add(directory / filename)
        meta, stored, config = obj(directory / "meta.json"), obj(directory / "metrics.json"), run["config"]
        expected = {r["id"]: r for r in manifest[task]}
        audit.check(name + ":balanced_validation_120", len(run["rows"]) == len(expected) == 120 and
                    len({r["source_id"] for r in run["rows"]}) == 120 and
                    sum(r["label"] for r in run["rows"]) == 60 and
                    set(expected) == {r["id"] for r in run["rows"]})
        manifest_path = root / f"prepared/curation_v3/val_{task}_120.jsonl"
        audit.check(name + ":frozen_data_hash", config.get("data_sha256") == _sha256(manifest_path))
        audit.check(name + ":input_budget", config.get("max_pixels") == pixels and
                    config.get("max_length") == 2048 and config.get("max_new_tokens") == 128 and config.get("limit") is None)
        policy = {k: config.get(k) for k in ("model", "max_length", "max_new_tokens", "generation", "quantization", "base_config_sha256")}
        shared_policy = policy if shared_policy is None else shared_policy
        audit.check(name + ":shared_generation_policy", policy == shared_policy)
        expected_adapter = root / training[arm]["directory"] / "adapter"
        audit.check(name + ":adapter_path", local_path(root, config["adapter"]) == expected_adapter)
        audit.adapter(name, expected_adapter, config.get("adapter_files_sha256"))
        for prediction in run["rows"]:
            expected_row = expected[prediction["id"]]
            for pk, dk in (("source_id", "source_id"), ("image", "image"), ("label", "is_anomalous"),
                           ("gt_boxes", "boxes"), ("category", "category"), ("prompt", "prompt"),
                           ("anomaly_area_fraction", "anomaly_area_fraction")):
                if not same(prediction.get(pk), expected_row.get(dk)):
                    raise ValueError(f"{name}: {prediction['id']} differs in {pk}")
            parsed, error = parse_prediction(prediction["raw_text"], require_boxes=task == "region")
            if parsed != prediction.get("parsed") or error != prediction.get("parse_error"):
                raise ValueError(f"{name}: stored parser result differs from raw generation")
            if prediction["localization_eligible"] != (task == "region"):
                raise ValueError(f"{name}: wrong task eligibility")
        audit.check(name + ":source_gt_and_raw_parser", True)
        audit.check(name + ":no_confirmation_sources", not confirmation.intersection(r["source_id"] for r in run["rows"]))
        recomputed = compute_metrics(run["rows"])
        for key in ("overall", "by_category", "by_anomaly_area", "definitions", "area_stratification_sources"):
            audit.check(name + ":metrics_recomputed:" + key, same(stored.get(key), recomputed.get(key)))
        anomalies = [r for r in run["rows"] if r["label"]]
        small_rows = [r for r in anomalies if 0 < r["anomaly_area_fraction"] <= small["threshold"]]
        other_rows = [r for r in anomalies if r["anomaly_area_fraction"] > small["threshold"]]
        output[name] = {**run, "train_arm": arm, "task": task, "inference_max_pixels": pixels,
                        "by_category": recomputed["by_category"],
                        "overall": recomputed["overall"],
                        "small_anomalies": _summary(small_rows),
                        "non_small_anomalies": _summary(other_rows),
                        "source_artifact": expected_relative, "reused_v2": name == "region_low_low_region"}
    return output


def score_audit(audit: Audit, name: str, expected: list[dict[str, Any]], adapter: Path | None,
                role: str, confirmation: set[str]) -> dict[str, Any]:
    directory = audit.root / "evaluations/curation_v3" / name
    for filename in ("scores.jsonl", "meta.json", "metrics.json"):
        audit.inputs.add(directory / filename)
    records, meta = _read_scores(directory)  # Verifies completed-file hash and raw likelihood-derived fields.
    config, stored = meta["config"], obj(directory / "metrics.json")
    contract = config["score_model_contract"]
    expected_by_id = {r["id"]: r for r in expected}
    audit.check(name + ":complete_schema", meta.get("status") == "complete" and
                meta.get("n_completed") == config.get("n_expected") == len(records) == len(expected) and
                meta.get("schema_version") == stored.get("schema_version") == SCORE_SCHEMA)
    audit.check(name + ":unique_expected_ids", len(expected_by_id) == len(expected) ==
                len({r["source_id"] for r in records}) and set(expected_by_id) == {r["id"] for r in records})
    data_path = local_path(audit.root, config["data"])
    audit.inputs.add(data_path)
    audit.check(name + ":data_hash", config["data_sha256"] == _sha256(data_path) and rows(data_path) == expected)
    audit.check(name + ":subset_hash", config.get("subset_sha256") ==
                _canonical_hash([(r["id"], r["source_id"], r["prompt"]) for r in expected]))
    audit.check(name + ":role_and_budget", config.get("data_role") == role and config.get("limit") is None and
                config.get("calibration_sha256") is None and contract.get("max_pixels") == LOW and
                contract.get("max_length") == 2048)
    audit.check(name + ":contract_hash", contract.get("contract_sha256") ==
                _canonical_hash({k: v for k, v in contract.items() if k != "contract_sha256"}))
    audit.check(name + ":adapter_policy", (adapter is None) == (config.get("adapter") is None) and
                (adapter is None) == (contract.get("adapter_files_sha256") is None))
    if adapter is not None:
        audit.check(name + ":adapter_path", local_path(audit.root, config["adapter"]) == adapter)
        audit.adapter(name, adapter, contract["adapter_files_sha256"])
    audit.check(name + ":image_hash_inventory", set(config["image_sha256_by_id"]) == set(expected_by_id))
    for record in records:
        source = expected_by_id[record["id"]]
        for key in ("source_id", "image", "category", "prompt"):
            if record.get(key) != source.get(key):
                raise ValueError(f"{name}: score provenance mismatch: {record['id']} {key}")
        if record["label"] != source["is_anomalous"] or record["prompt"] != contract["prompt"]:
            raise ValueError(f"{name}: score label/prompt mismatch")
        if record["image_sha256"] != source_hash(record["source_id"]) or config["image_sha256_by_id"][record["id"]] != record["image_sha256"]:
            raise ValueError(f"{name}: score image hash mismatch")
        derived = normalized_score(record["log_likelihood_true"], record["log_likelihood_false"])
        if any(not same(record.get(k), v) for k, v in derived.items()):
            raise ValueError(f"{name}: raw likelihood and derived score mismatch")
        if record.get("token_boundary_audit") != BOUNDARY_AUDIT:
            raise ValueError(f"{name}: missing native token-boundary evidence")
        for value in ("true", "false"):
            n = record["n_value_tokens"][value]
            total = record["prompt_tokens"] + len(record["common_json_prefix_token_ids"]) + n + len(record["common_suffix_token_ids"])
            if not (n == len(record["value_token_ids"][value]) > 0 and
                    record["candidate_input_tokens"][value] == total <= contract["max_length"]):
                raise ValueError(f"{name}: candidate token budget mismatch")
    audit.check(name + ":raw_likelihood_provenance_and_native_tokens", True)
    audit.check(name + ":no_confirmation_sources", not confirmation.intersection(r["source_id"] for r in records))
    ranking, equal = ranking_metrics(records), threshold_metrics(records, 0.0)
    audit.check(name + ":ranking_recomputed", same(stored["overall"], ranking))
    audit.check(name + ":equal_threshold_recomputed", same(stored["default_equal_candidate_threshold"], equal))
    audit.check(name + ":metrics_metadata", same(stored["metadata"], {**config, "complete": True}))
    category_records = defaultdict(list)
    for record in records:
        category_records[record["category"]].append(record)
    by_category = {key: ranking_metrics(value) for key, value in sorted(category_records.items())}
    audit.check(name + ":category_ranking_recomputed", same(stored.get("by_category"), by_category))
    timestamps = {key: meta.get(key) for key in ("created_at", "finished_at")}
    elapsed = None
    if all(timestamps.values()):
        elapsed = (datetime.fromisoformat(timestamps["finished_at"]) - datetime.fromisoformat(timestamps["created_at"])).total_seconds()
        audit.check(name + ":elapsed_nonnegative", elapsed >= 0)
    return {"name": name, "records": records, "config": config, "ranking": ranking,
            "equal_candidate_threshold": equal, "by_category": by_category,
            "cost": {"meta_elapsed_seconds": elapsed,
                     "meta_elapsed_scope": "after config/image/base hashing; includes model loading, warmup and scoring, not pre-meta hashing",
                     "sum_record_latency_seconds": sum(r["latency_seconds"] for r in records),
                     "mean_record_latency_seconds": sum(r["latency_seconds"] for r in records) / len(records),
                     "record_latency_scope": stored.get("latency_definition"),
                     "max_peak_memory_allocated_bytes": max(r["peak_memory_allocated_bytes"] for r in records)},
            "timestamps": timestamps}


def align(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    a, b = {r["source_id"]: r for r in left}, {r["source_id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Paired analysis needs one row per common source")
    for source in a:
        for key in ("id", "label", "category", "prompt", "gt_boxes", "anomaly_area_fraction", "localization_eligible"):
            if a[source].get(key) != b[source].get(key):
                raise ValueError(f"Paired source differs in {key}: {source}")
    return sorted(a), a, b


def metric_values(records: list[dict[str, Any]], *, scoring: bool) -> dict[str, Any]:
    if scoring:
        # Bootstrap repeats a source; ranking_metrics intentionally forbids that
        # in saved predictions, so each resampled occurrence gets a draw ID.
        records = [{**r, "source_id": f"bootstrap_{i}"} for i, r in enumerate(records)]
        ranking, threshold = ranking_metrics(records), threshold_metrics(records, 0.0)
        return {"auroc": ranking["auroc"], "average_precision": ranking["average_precision"],
                "equal_threshold_fpr": threshold["fpr"], "equal_threshold_recall": threshold["recall"],
                "equal_threshold_macro_f1": threshold["macro_f1"]}
    summary = _summary(records)
    return {key: summary[key] for key in GEN_METRICS}


def paired(left: dict[str, Any], right: dict[str, Any], *, samples: int, seed: int, scoring: bool = False) -> dict[str, Any]:
    field = "records" if scoring else "rows"
    sources, a, b = align(left[field], right[field])
    if not sources:
        raise ValueError("Empty paired comparison")
    point_a, point_b = metric_values(left[field], scoring=scoring), metric_values(right[field], scoring=scoring)
    values, rng = defaultdict(list), random.Random(seed)
    for _ in range(samples):
        draw = [rng.choice(sources) for _ in sources]
        ma = metric_values([a[s] for s in draw], scoring=scoring)
        mb = metric_values([b[s] for s in draw], scoring=scoring)
        for key in point_a:
            if ma[key] is not None and mb[key] is not None:
                values[key].append(ma[key] - mb[key])
    estimates = {}
    for key in point_a:
        distribution = sorted(values[key])
        estimates[key] = {"left": point_a[key], "right": point_b[key],
                          "difference": point_a[key] - point_b[key] if point_a[key] is not None and point_b[key] is not None else None,
                          "ci95_low": _quantile(distribution, 0.025), "ci95_high": _quantile(distribution, 0.975),
                          "n_defined_resamples": len(distribution)}
    return {"left": left["name"], "right": right["name"], "n_sources": len(sources),
            "n_resamples": samples, "seed": seed, "metrics": estimates}


def resolution_interaction(runs: dict[str, Any], *, samples: int, seed: int) -> dict[str, Any]:
    names = ("region_low_low_region", "region_low_high_region", "region_high_low_region", "region_high_high_region")
    reference = runs[names[0]]["rows"]
    maps = {}
    for name in names:
        sources, _, maps[name] = align(reference, runs[name]["rows"])
    def calculate(draw):
        metrics = {name: metric_values([maps[name][s] for s in draw], scoring=False) for name in names}
        return {k: metrics[names[3]][k] - metrics[names[2]][k] - metrics[names[1]][k] + metrics[names[0]][k]
                if all(metrics[n][k] is not None for n in names) else None for k in GEN_METRICS}
    point, values, rng = calculate(sources), defaultdict(list), random.Random(seed)
    for _ in range(samples):
        value = calculate([rng.choice(sources) for _ in sources])
        for key, result in value.items():
            if result is not None:
                values[key].append(result)
    return {"definition": "(high-train/high-infer - high-train/low-infer) - (low-train/high-infer - low-train/low-infer)",
            "n_sources": len(sources), "n_resamples": samples, "seed": seed,
            "metrics": {key: {"difference": value, "ci95_low": _quantile(sorted(values[key]), .025),
                                "ci95_high": _quantile(sorted(values[key]), .975),
                                "n_defined_resamples": len(values[key])} for key, value in point.items()}}


def mining_audit(audit: Audit, mining: dict[str, Any], prepared: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    pool = mining["records"]
    audit.check("mining_1152_train_normals", len(pool) == 1152 and not any(r["label"] for r in pool) and
                len({r["category"] for r in pool}) == 12 and
                set(Counter(r["category"] for r in pool).values()) == {96})
    pool_sources = {r["source_id"] for r in pool}
    selections = {}
    grouped = defaultdict(list)
    for record in pool:
        grouped[record["category"]].append(record)
    for arm in ("normal_random", "normal_hard"):
        selected = [r for r in unique_sources(prepared[arm]).values() if not r["is_anomalous"]]
        selected_ids = {r["source_id"] for r in selected}
        audit.check(arm + ":selected_from_pool", len(selected_ids) == 384 and selected_ids <= pool_sources)
        selections[arm] = selected_ids
    hard_details = {}
    for category, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda r: (-r["anomaly_log_odds"], r["source_id"]))
        top = {r["source_id"] for r in ordered[:16]}
        chosen = selections["normal_hard"].intersection(r["source_id"] for r in records)
        audit.check(category + ":hard_includes_top16", top <= chosen and len(chosen) == 32)
        hard_details[category] = {"n_pool": len(records), "n_selected": len(chosen),
                                  "n_top16_included": len(top & chosen),
                                  "n_other_selected": len(chosen - top),
                                  "top16_tiebreak": "descending anomaly_log_odds, ascending source_id"}
    return {"pool_sources": 1152, "normal_sources_per_arm": 384,
            "selected_normal_overlap": len(selections["normal_random"] & selections["normal_hard"]),
            "hard_selection_by_category": hard_details,
            "cost": mining["cost"], "selection_is_annotation_quality_audit": False,
            "interpretation": "Hardness is frozen base-model false-anomaly tendency on existing train-normal labels; not expert relabeling or a verified quality label."}


def build(root: Path, *, samples: int = 2000, seed: int = 17) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = Audit(root)
    audit.check("no_torch_import", "torch" not in sys.modules)
    audit.inputs.add(Path(__file__).resolve())
    for module in ("industrial_vlm.metrics", "industrial_vlm.compare", "industrial_vlm.score", "industrial_vlm.evaluate"):
        audit.inputs.add(Path(sys.modules[module].__file__).resolve())
    plan = audit.read(root / "prepared/curation_v3/plan.json")
    state = audit.read(root / "experiments/curation_v3/state.json")
    audit.check("runner_complete", state.get("status") == "complete" and state.get("confirmation_evaluated") is False)
    manifest = {task: audit.read_rows(root / f"prepared/curation_v3/val_{task}_120.jsonl") for task in ("decision", "region")}
    for task, records in manifest.items():
        old = root / f"prepared/multitask_v2/val_{task}_120.jsonl"
        audit.inputs.add(old)
        audit.check(task + ":unchanged_v2_validation", _sha256(old) == _sha256(root / f"prepared/curation_v3/val_{task}_120.jsonl"))
        audit.check(task + ":unique_balanced_120", len(records) == len({r["source_id"] for r in records}) == 120 and
                    Counter(r["is_anomalous"] for r in records) == Counter({True: 60, False: 60}))
    audit.check("same_validation_sources_across_tasks", {r["source_id"] for r in manifest["decision"]} ==
                {r["source_id"] for r in manifest["region"]})
    confirmation_path = root / "prepared/multitask_v2/confirmation_sources.jsonl"
    confirmation_rows = audit.read_rows(confirmation_path)
    confirmation = {r["source_id"] for r in confirmation_rows}
    audit.check("confirmation_metadata_240", len(confirmation_rows) == len(confirmation) == 240)
    for path in (root / "evaluations").rglob("meta.json"):
        config = audit.read(path).get("config", {})
        audit.check("no_confirmation_invocation:" + str(path.relative_to(root)), config.get("data_role") != "confirmation" and
                    "confirmation" not in str(config.get("data", "")).lower())
    training, prepared = train_audit(audit)
    small = small_definition(audit, prepared, plan)
    for arm, data in prepared.items():
        ids = set(unique_sources(data))
        audit.check(arm + ":train_validation_confirmation_disjoint", not ids.intersection(confirmation) and
                    not ids.intersection(r["source_id"] for r in manifest["decision"]))
    generated = generation_audit(audit, training, small, manifest, confirmation)
    scores = {}
    for arm in ("normal_random", "normal_hard"):
        scores[arm] = score_audit(audit, arm + "_validation", manifest["decision"],
                                  root / training[arm]["directory"] / "adapter", "validation", confirmation)
    policies = [{k: v for k, v in score["config"]["score_model_contract"].items()
                 if k not in {"adapter_files_sha256", "contract_sha256"}} for score in scores.values()]
    audit.check("validation_scores_shared_base_and_policy", policies[0] == policies[1])
    mining_meta = audit.read(root / "evaluations/curation_v3/mining_base_train_normal/meta.json")
    mining_path = local_path(root, mining_meta["config"]["data"])
    mining_manifest = audit.read_rows(mining_path)
    mining_run = score_audit(audit, "mining_base_train_normal", mining_manifest, None, "train_mining", confirmation)
    mining = mining_audit(audit, mining_run, prepared)
    mining_ids = {r["source_id"] for r in mining_manifest}
    audit.check("mining_validation_disjoint", not mining_ids.intersection(r["source_id"] for r in manifest["decision"]))
    # The full canonical train source inventory, if present, verifies partition
    # membership without reading raw images; scorer already records byte hashes.
    train_inventory = root / "prepared/visa/train.jsonl"
    if train_inventory.is_file():
        train_ids = {r["source_id"] for r in audit.read_rows(train_inventory)}
        audit.check("mining_pool_in_train_partition", mining_ids <= train_ids)
    else:
        # Prepared pool rows must retain the split provenance if canonical
        # inventory follows a different name. No missing audit is called passed.
        candidates = sorted((root / "prepared/visa").glob("*train*.jsonl"))
        train_ids = set()
        for path in candidates:
            train_ids.update(r["source_id"] for r in audit.read_rows(path))
        audit.check("mining_pool_in_train_partition", bool(train_ids) and mining_ids <= train_ids)
    generation_pairs = [
        ("normal_hard_decision", "normal_random_decision", "hard_minus_random"),
        ("region_low_high_region", "region_low_low_region", "inference_resolution_given_low_training"),
        ("region_high_high_region", "region_high_low_region", "inference_resolution_given_high_training"),
        ("region_high_low_region", "region_low_low_region", "training_resolution_given_low_inference"),
        ("region_high_high_region", "region_low_high_region", "training_resolution_given_high_inference"),
        ("region_high_high_region", "region_low_low_region", "both_resolution_changes"),
    ]
    comparisons = [{"contrast": label, **paired(generated[left], generated[right], samples=samples, seed=seed)}
                   for left, right, label in generation_pairs]
    score_comparison = paired(scores["normal_hard"], scores["normal_random"], samples=samples, seed=seed, scoring=True)
    interaction = resolution_interaction(generated, samples=samples, seed=seed)
    # Historical v2 DD remains explicitly descriptive, never part of the new
    # hard-vs-random matched-selection contrast or an independent test claim.
    history_path = root / "reports/multitask_v2/scoring_matrix.json"
    history = None
    if history_path.is_file():
        old = audit.read(history_path)
        history = {"scope": "historical validation development reference only; different normal-source selection",
                   "decision_decision_validation": old["models"]["decision_decision"]["validation"]}
    audit.check("no_torch_import_after_analysis", "torch" not in sys.modules)
    limits = [
        "单个训练seed=42；配对bootstrap只刻画当前模型的验证来源采样不确定性，不包含训练seed波动。",
        "相同120张开发验证来源被多轮使用，结果属于开发比较，不是独立测试；本轮没有外域评估。",
        "三组新训练固定最终192更新；曝光匹配不等于token、算力或总耗时匹配，高分辨率及困难挖掘增加成本。",
        "困难正常样本来自训练分区已有标签；base分数只代表模型难度，不能作为专家复核或标签质量改善的证据。",
        "区域低训练/低推理使用冻结v2 RR历史产物；其余格子新运行，硬件时序与缓存影响延迟因果解释。",
        "小缺陷阈值仅用固定384训练异常的mask面积q1确定；验证小子集样本少，必须同时展示分母。",
        "区域指标为包围矩形IoU的端到端召回，不是像素/实例分割；漏检与格式错误保留在分母。",
        "Macro F1为正常/异常两个标签的平均；均衡验证集的AP/F1不代表真实生产异常比例。",
        "布尔候选归一分数不是校准置信度；本轮只报告排序及预先固定log-odds=0，不在同一验证集调阈值后宣称低误报。",
        "source SHA256隔离不保证物理工件身份或近重复独立；公共基准的预训练污染未知。",
        "多组差异和多个指标均为描述性比较，未作多重比较校正；未达到跨seed或SOTA证据标准。",
        "分析器只核验已记录的图像/基础权重hash一致性，不重新打开数据图像或重新读取基础权重。",
    ]
    created = datetime.now(timezone.utc).isoformat()
    report = {"schema_version": "industrial-vlm-curation-v3-analysis-v1", "status": "complete",
              "audit_status": "passed", "created_at": created, "scope": "validation_development_only",
              "checks": audit.checks, "training": training, "small_anomaly_definition": small,
              "mining": mining, "generation": {name: {k: v for k, v in run.items() if k != "rows"}
                                                     for name, run in generated.items()},
              "scores": {name: {k: v for k, v in run.items() if k != "records"} for name, run in scores.items()},
              "historical_reference": history,
              "confirmation": {"metadata_sources_checked": 240, "images_opened": False,
                               "predictions_generated_or_read": False, "evaluation_metadata_invocations_found": 0},
              "limits": limits, "input_sha256": audit.artifacts(), "figures": []}
    differences = {"schema_version": "industrial-vlm-curation-v3-paired-v1", "status": "complete",
                   "created_at": created, "scope": "validation_development_only",
                   "method": "paired source bootstrap with replacement; percentile95% interval with linear interpolation",
                   "difference_direction": "left minus right; JSON stores fractions, Markdown percentage points",
                   "n_resamples": samples, "seed": seed, "generation": comparisons,
                   "scoring": score_comparison, "resolution_interaction": interaction,
                   "input_sha256": report["input_sha256"], "limits": limits}
    return report, differences


def pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci_text(metric: dict[str, Any]) -> str:
    return "—" if metric["ci95_low"] is None else f"[{metric['ci95_low'] * 100:+.2f}, {metric['ci95_high'] * 100:+.2f}]"


def render(report: dict[str, Any], differences: dict[str, Any]) -> str:
    hard = differences["generation"][0]["metrics"]
    auc = differences["scoring"]["metrics"]["auroc"]
    lines = ["# 训练正常样本挖掘与分辨率消融 v3", "",
             "本轮只在既有120张开发验证图上比较，最终240张确认集保持封存。结论取决于实际差异、区间和成本，不预设困难挖掘或高分辨率有效。", "",
             f"困难正常采样相对随机采样：生成FPR差{hard['fpr']['difference'] * 100:+.2f}pp，95%区间{ci_text(hard['fpr'])}；"
             f"Macro F1差{hard['macro_f1']['difference'] * 100:+.2f}pp，区间{ci_text(hard['macro_f1'])}；"
             f"连续评分AUROC差{auc['difference'] * 100:+.2f}pp，区间{ci_text(auc)}。", "",
             "## 生成结果", "",
             "| 实验 | Macro F1 % | 异常召回 % | 正常FPR % | 框R@IoU.5 % | 有效格式 % | 秒/图 | 峰值GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in report["generation"].items():
        m = run["overall"]
        memory = m["max_peak_memory_allocated_bytes"]
        lines.append(f"| {name}{'（复用v2）' if run['reused_v2'] else ''} | {pct(m['macro_f1'])} | {pct(m['anomaly_recall'])} | "
                     f"{pct(m['fpr'])} | {pct(m['union_box_recall_iou_0.5'])} | {pct(m['format_valid_rate'])} | "
                     f"{number(m['mean_latency_seconds'])} | {number(memory / 2**30 if memory is not None else None)} |")
    lines += ["", "Macro F1平均正常/异常两个标签，非12类别宏平均；框指标分母包括分类漏检和格式错误。", "",
              "## 连续评分", "",
              "| 正常样本方案 | AUROC % | AP % | 固定margin=0 Macro F1 % | 固定margin=0 FPR % | 固定margin=0召回 % |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, score in report["scores"].items():
        rank, equal = score["ranking"], score["equal_candidate_threshold"]
        lines.append(f"| {name} | {pct(rank['auroc'])} | {pct(rank['average_precision'])} | {pct(equal['macro_f1'])} | {pct(equal['fpr'])} | {pct(equal['recall'])} |")
    lines += ["", "只使用true/false候选值token的原始条件似然；排序处理相同分数，AP为非插值average precision。"
              "固定margin=0意味着两候选等似然，未用本轮验证标签优化阈值；分数不等于校准置信度。", "",
              "## 分辨率2×2", "",
              "low=524288 max_pixels，high=1048576 max_pixels，均沿用原生Qwen动态分辨率处理；上限翻倍不保证每张图token恰好翻倍。"
              "low/high的第一个维度是训练上限，第二个是推理上限，四格分开回答训练输入与推理输入的作用。", "",
              "| 训练 / 推理 | Macro F1 % | 框R@IoU.5 % | 小缺陷数 | 小缺陷框R % | 非小缺陷数 | 非小缺陷框R % |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in SPECS:
        if not name.startswith("region_"):
            continue
        run = report["generation"][name]
        small, other = run["small_anomalies"], run["non_small_anomalies"]
        lines.append(f"| {name} | {pct(run['overall']['macro_f1'])} | {pct(run['overall']['union_box_recall_iou_0.5'])} | "
                     f"{small['n_anomalous']} | {pct(small['union_box_recall_iou_0.5'])} | {other['n_anomalous']} | {pct(other['union_box_recall_iou_0.5'])} |")
    definition = report["small_anomaly_definition"]
    lines += ["", f"小缺陷定义为mask面积比例≤{definition['threshold']:.9g}，阈值来自固定384张训练异常的线性插值q1；未按验证分布选阈值。", "",
              "## 配对差异", "",
              f"同来源配对bootstrap {differences['n_resamples']}次，seed={differences['seed']}；下列差值单位pp，FPR越低越好。", "",
              "| 对比（左−右） | 指标 | 差值pp | 95%区间pp | 有定义抽样数 |", "|---|---|---:|---:|---:|"]
    for comparison in differences["generation"] + [differences["scoring"]]:
        for key, metric in comparison["metrics"].items():
            difference = "—" if metric["difference"] is None else f"{100 * metric['difference']:+.2f}"
            lines.append(f"| {comparison['left']} − {comparison['right']} | {key} | {difference} | {ci_text(metric)} | {metric['n_defined_resamples']} |")
    lines += ["", "训练/推理分辨率交互项=(高训练高推理−高训练低推理)−(低训练高推理−低训练低推理)。", "",
              "| 交互指标 | 差值pp | 95%区间pp |", "|---|---:|---:|"]
    for key, metric in differences["resolution_interaction"]["metrics"].items():
        lines.append(f"| {key} | {100 * metric['difference']:+.2f} | {ci_text(metric)} |")
    lines += ["", "## 成本与预算", "",
              "| 训练臂 | 更新/曝光 | 输入token | 监督token | 训练秒* | 峰值allocated GiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, summary in report["training"].items():
        lines.append(f"| {arm} | {summary['optimizer_steps']}/{summary['microsteps']} | {summary['total_input_tokens']} | "
                     f"{summary['total_target_tokens']} | {number(summary.get('train_wall_seconds_including_validation'))} | {number(summary.get('peak_allocated_gib'))} |")
    cost = report["mining"]["cost"]
    lines += ["", "*训练秒含训练循环内验证CE，不含初始加载和最终重载。相同曝光不等于等计算预算。", "",
              f"正常训练候选池1152张，挖掘评分记录延迟合计{number(cost['sum_record_latency_seconds'])}秒；"
              f"meta时间跨度{number(cost['meta_elapsed_seconds'])}秒（写meta之后，含模型加载/暖机，不含此前hash）。"
              "这笔成本必须纳入困难采样方案，不能只比较SFT循环。", "",
              "## 分类别结果", "",
              "| 实验 | 类别 | N | Macro F1 % | FPR % | 异常召回 % | 框R@IoU.5 % |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, run in report["generation"].items():
        for category, metric in run["by_category"].items():
            lines.append(f"| {name} | {category} | {metric['n']} | {pct(metric['macro_f1'])} | {pct(metric['fpr'])} | "
                         f"{pct(metric['anomaly_recall'])} | {pct(metric['union_box_recall_iou_0.5'])} |")
    lines += ["", "## 可信度与解释边界", ""]
    lines.extend("- " + limit for limit in report["limits"])
    lines += ["", f"审计通过{len(report['checks'])}项；逐条真实曝光、初始化、训练目标/来源、adapter字节、预测原文解析、"
              "连续原始似然与派生值、文件hash和确认来源排除均可见[机读报告](analysis.json)。", "",
              "复核命令：`python ops/analyze_curation_v3.py --plots`。脚本只读产物元数据，不加载模型，不打开数据图像。", ""]
    if report["figures"]:
        lines += ["## 图表", ""]
        lines.extend(f"- [{Path(p).name}](figures/{Path(p).name})" for p in report["figures"])
        lines.append("")
    return "\n".join(lines)


def plots(report: dict[str, Any], differences: dict[str, Any], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    artifacts = []
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
    names = ("normal_random_decision", "normal_hard_decision")
    for axis, key, title in zip(axes, ("macro_f1", "fpr", "anomaly_recall"),
                                ("Decision Macro F1", "Normal false-positive rate", "Anomaly recall")):
        values = [100 * report["generation"][n]["overall"][key] for n in names]
        bars = axis.bar(["Random", "Hard + random"], values, color=["#0072b2", "#d55e00"], width=.58)
        axis.set(title=title, ylabel="Percent", ylim=(0, 108))
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.grid(axis="y", alpha=.18)
    fig.suptitle("Training-normal curation: development validation (120 images; seed 42)", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        p = output / f"normal_curation_comparison.{ext}"
        fig.savefig(p, dpi=300)
        artifacts.append(str(p))
    plt.close(fig)
    names = ("region_low_low_region", "region_low_high_region", "region_high_low_region", "region_high_high_region")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
    for axis, key, title in zip(axes, ("macro_f1", "union_box_recall_iou_0.5", "small"),
                                ("Macro F1", "Box recall @ IoU 0.5", "Small-defect box recall")):
        points = []
        for name in names:
            run = report["generation"][name]
            points.append(100 * (run["small_anomalies"]["union_box_recall_iou_0.5"] if key == "small" else run["overall"][key]))
        for index, label, color in ((0, "Low-res training", "#0072b2"), (2, "High-res training", "#d55e00")):
            axis.plot([0, 1], points[index:index + 2], marker="o", color=color, label=label)
            for x, value in enumerate(points[index:index + 2]):
                axis.annotate(f"{value:.2f}", (x, value), xytext=(0, 7 if index == 0 else -14),
                              textcoords="offset points", ha="center", fontsize=8)
        axis.set(xticks=[0, 1], xticklabels=["Low", "High"], xlabel="Inference resolution limit",
                 ylabel="Percent", ylim=(0, 108), title=title)
        axis.grid(alpha=.18)
        axis.legend(loc="lower right", fontsize=8)
    small_n = report["generation"][names[0]]["small_anomalies"]["n_anomalous"]
    fig.suptitle(f"Resolution 2 x 2: validation development set (small-defect n={small_n}; train-only q1 threshold)", fontsize=11)
    for ext in ("png", "pdf", "svg"):
        p = output / f"resolution_factorial.{ext}"
        fig.savefig(p, dpi=300)
        artifacts.append(str(p))
    plt.close(fig)
    figure_data = {"generation": report["generation"], "scores": report["scores"],
                   "small_anomaly_definition": report["small_anomaly_definition"],
                   "differences": differences, "input_sha256": report["input_sha256"]}
    _atomic_json(output / "figure_data.json", figure_data)
    return artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be at least 20")
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "reports/curation_v3"
    if output == root / "reports" or output.is_relative_to(root / "reports/multitask_v2"):
        parser.error("v3 output must not replace v1/v2 report directories")
    report, differences = build(root, samples=args.bootstrap_samples, seed=args.seed)
    output.mkdir(parents=True, exist_ok=True)
    if args.plots:
        report["figures"] = plots(report, differences, output / "figures")
    document = render(report, differences)
    _atomic_json(output / "analysis.json", report)
    _atomic_json(output / "paired_differences.json", differences)
    temporary = output / "实验报告.md.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "实验报告.md")
    print(json.dumps({"status": "complete", "audit_status": "passed", "n_checks": len(report["checks"]),
                      "n_generation_runs": len(report["generation"]), "n_validation_score_runs": len(report["scores"]),
                      "confirmation_evaluated": False, "report": str(output / "实验报告.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
