"""Auditable boolean continuation scores; these are not calibrated confidence.

Score both fixed candidate continuations under one decision prompt. Only the
tokens spelling the JSON boolean contribute to the conditional log likelihood;
the common JSON prefix, closing brace and chat end marker do not. Native Qwen
image processing, grids and positions come exclusively from ``make_inputs``.

``score`` needs the existing CUDA NF4 runtime. ``calibrate`` and metric helpers
use only Python's standard library and never load a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate import _atomic_json, _load_data, _package_versions, _read_predictions, _sha256

SCHEMA_VERSION = "industrial-vlm-boolean-score-v1"
JSON_PREFIX = '{"is_anomalous":'
VALUES = ("true", "false")


def normalized_score(log_true: float, log_false: float) -> dict[str, float]:
    """Two-option log-softmax, without treating likelihood as confidence."""
    if not all(math.isfinite(v) for v in (log_true, log_false)):
        raise ValueError("candidate log likelihoods must be finite")
    margin = log_true - log_false
    if not math.isfinite(margin):
        raise ValueError("candidate log likelihood difference must be finite")
    correction = math.log1p(math.exp(-abs(margin)))
    log_normalized_true = min(margin, 0.0) - correction
    log_normalized_false = min(-margin, 0.0) - correction
    return {"anomaly_score": math.exp(log_normalized_true),
            "log_normalized_true": log_normalized_true,
            "log_normalized_false": log_normalized_false,
            "anomaly_log_odds": margin}


def boolean_token_span(ids: list[int], offsets: list[tuple[int, int]], text: str,
                       start: int, end: int) -> tuple[int, int]:
    """Find a complete, non-straddling token interval for the boolean value.

    Fast-tokenizer offsets must cover every character of the value exactly.
    Fail closed on a merged JSON/value token or an unreliable offset mapping.
    The returned end is exclusive, in token coordinates.
    """
    if len(ids) != len(offsets) or not (0 <= start < end <= len(text)):
        raise ValueError("invalid tokenizer offsets or boolean character span")
    selected: list[int] = []
    for i, (left, right) in enumerate(offsets):
        if not (0 <= left <= right <= len(text)):
            raise ValueError("tokenizer returned invalid character offsets")
        if left < start < right or left < end < right:
            raise ValueError("boolean boundary is inside a tokenizer token")
        if left < end and right > start:
            selected.append(i)
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("boolean tokens are absent or noncontiguous")
    position = start
    for i in selected:
        left, right = offsets[i]
        if left != position or right <= left:
            raise ValueError("boolean offsets contain a gap, overlap or empty token")
        position = right
    if position != end:
        raise ValueError("boolean offsets do not cover the exact value")
    return selected[0], selected[-1] + 1


def candidate_plan(tokenizer: Any, continuation: str, value: str) -> dict[str, Any]:
    if value not in VALUES or not continuation.startswith(JSON_PREFIX + value + "}"):
        raise ValueError("unexpected fixed candidate continuation")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with verified offsets is required")
    encoded = tokenizer(continuation, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    begin, end = boolean_token_span(ids, offsets, continuation,
                                    len(JSON_PREFIX), len(JSON_PREFIX) + len(value))
    decoded = tokenizer.decode(ids[begin:end], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded != value:
        raise ValueError(f"boolean token decoding differs from {value!r}: {decoded!r}")
    return {"continuation_ids": ids, "value_begin": begin, "value_end": end,
            "value_token_ids": ids[begin:end], "prefix_ids": ids[:begin],
            "suffix_ids": ids[end:]}


def ranking_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """AUROC and non-interpolated AP, grouping all tied scores together.

    Ranking uses log odds to avoid sigmoid rounding collapsing extreme scores.
    AP is the sum of precision times recall increments at distinct thresholds,
    rather than a trapezoidal PR area. One-class samples return null AUROC.
    """
    if not records:
        raise ValueError("cannot compute metrics for empty scores")
    grouped: dict[float, list[bool]] = defaultdict(list)
    sources = set()
    for row in records:
        if type(row.get("label")) is not bool:
            raise ValueError("score label must be a boolean")
        value = row.get("anomaly_log_odds")
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("finite anomaly_log_odds required")
        grouped[float(value)].append(row["label"])
        source_id = row.get("source_id")
        if not source_id or source_id in sources:
            raise ValueError("one unique source_id is required per scored image")
        sources.add(source_id)
    positive = sum(row["label"] for row in records)
    negative = len(records) - positive
    tp = fp = 0
    auc_numerator = ap = 0.0
    for score in sorted(grouped, reverse=True):
        labels = grouped[score]
        add_tp = sum(labels)
        add_fp = len(labels) - add_tp
        # Descending ROC trapezoids in count coordinates, tie-safe.
        auc_numerator += (tp + add_tp / 2) * add_fp
        tp += add_tp
        fp += add_fp
        if positive:
            ap += (add_tp / positive) * (tp / (tp + fp))
    return {"n": len(records), "n_anomalous": positive, "n_normal": negative,
            "auroc": auc_numerator / (positive * negative) if positive and negative else None,
            "average_precision": ap if positive else None,
            "auprc_definition": "non-interpolated average precision; tied values grouped",
            "ranking_field": "anomaly_log_odds",
            "confidence_calibrated": False}


def threshold_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    tp = fp = tn = fn = 0
    for row in records:
        prediction = row["anomaly_log_odds"] > threshold
        if row["label"]:
            tp += prediction
            fn += not prediction
        else:
            fp += prediction
            tn += not prediction
    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None
    f1_positive = ratio(2 * tp, 2 * tp + fp + fn)
    f1_negative = ratio(2 * tn, 2 * tn + fp + fn)
    return {"threshold_log_odds": threshold, "comparison": "anomaly_log_odds > threshold",
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn),
            "precision": ratio(tp, tp + fp), "anomaly_f1": f1_positive,
            "macro_f1": (f1_positive + f1_negative) / 2
            if f1_positive is not None and f1_negative is not None else None}


def calibrate_normal_threshold(records: list[dict[str, Any]], target_fpr: float) -> dict[str, Any]:
    """Select a normal-only order statistic with a strict greater-than rule.

    This controls empirical calibration-set FPR only. It gives no finite-sample
    or distribution-shift guarantee about future FPR, including the 5% setting.
    """
    ranking_metrics(records)  # Validate labels, source uniqueness and values.
    if not 0 <= target_fpr < 1:
        raise ValueError("target FPR must be in [0,1)")
    normals = sorted((r["anomaly_log_odds"] for r in records if not r["label"]), reverse=True)
    if not normals:
        raise ValueError("threshold calibration needs validation normal images")
    allowed = math.floor(target_fpr * len(normals) + 1e-12)
    threshold = normals[allowed]
    false_positives = sum(value > threshold for value in normals)
    return {"threshold_log_odds": threshold, "comparison": "anomaly_log_odds > threshold",
            "target_empirical_fpr": target_fpr, "n_calibration_normals": len(normals),
            "allowed_calibration_false_positives": allowed,
            "actual_calibration_false_positives": false_positives,
            "empirical_calibration_fpr": false_positives / len(normals),
            "selection": "normal-only descending order statistic; strict > excludes boundary ties",
            "limitation": "Empirical validation FPR only; no guarantee of future or cross-domain FPR.",
            "confidence_calibrated": False}


def _paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix == ".jsonl":
        return output, output.with_suffix(".metrics.json"), output.with_suffix(".meta.json")
    return output / "scores.jsonl", output / "metrics.json", output / "meta.json"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _model_files(path: Path) -> dict[str, str]:
    """Hash local weights and tokenizer/config files, not just their filenames."""
    files = [p for p in sorted(path.iterdir()) if p.is_file()
             and p.suffix in {".safetensors", ".bin", ".json", ".txt", ".model", ".tiktoken"}]
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"no local model/adapter weights found in {path}")
    return {p.name: _sha256(p) for p in files}


def _input_images(data: list[dict[str, Any]], root: Path) -> dict[str, str]:
    hashes = {}
    for row in data:
        path = Path(row["image"])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        match = re.search(r"_([a-f0-9]{64})$", row["source_id"])
        expected = row.get("image_sha256") or (match.group(1) if match else None)
        if expected and expected != actual:
            raise ValueError(f"source image SHA256 mismatch: {row['source_id']}")
        hashes[row["id"]] = actual
    return hashes


def _score_image(model: Any, processor: Any, image: Path, prompt: str,
                 max_length: int) -> dict[str, Any]:
    """No label/answer argument: both label-independent hypotheses are scored."""
    import torch
    from .modeling import make_inputs, move_inputs

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    prefix_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_inputs = make_inputs(processor, image, prompt, max_length=max_length)
    prompt_ids = prompt_inputs["input_ids"][0].tolist()
    if "labels" in prompt_inputs:
        raise ValueError("label-free prompt unexpectedly contains labels")
    candidates = {}
    plans = {}
    for value in VALUES:
        answer = JSON_PREFIX + value + "}"
        full_text = processor.apply_chat_template(
            messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if not full_text.startswith(prefix_text):
            raise ValueError("native chat-template text prefix is not stable")
        continuation = full_text[len(prefix_text):]
        plan = candidate_plan(processor.tokenizer, continuation, value)
        inputs = make_inputs(processor, image, prompt, answer=answer, max_length=max_length)
        inputs.pop("labels", None)  # Candidate labels are never passed to model.forward.
        full_ids = inputs["input_ids"][0].tolist()
        if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[len(prompt_ids):] != plan["continuation_ids"]:
            raise ValueError("isolated continuation tokenization differs from native full inputs")
        if not torch.equal(inputs["image_grid_thw"], prompt_inputs["image_grid_thw"]):
            raise ValueError("native image grid changed across boolean candidates")
        candidates[value] = inputs
        plans[value] = plan
    if (plans["true"]["prefix_ids"] != plans["false"]["prefix_ids"]
            or plans["true"]["suffix_ids"] != plans["false"]["suffix_ids"]):
        raise ValueError("candidates do not have the exact same token prefix and suffix")
    lengths, raw = {}, {}
    for value in VALUES:
        plan = plans[value]
        inputs = move_inputs(candidates.pop(value), model)
        positions = torch.arange(len(prompt_ids) + plan["value_begin"],
                                 len(prompt_ids) + plan["value_end"], device=inputs["input_ids"].device)
        if int(positions.min()) <= 0:
            raise ValueError("boolean has no causal predecessor token")
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
            value_logits = output.logits[0, positions - 1].float()
            targets = inputs["input_ids"][0, positions]
            log_probabilities = value_logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            raw[value] = float(log_probabilities.sum().item())
        lengths[value] = int(inputs["input_ids"].shape[-1])
        del output, value_logits, targets, log_probabilities, inputs
    return {"log_likelihood_true": raw["true"], "log_likelihood_false": raw["false"],
            **normalized_score(raw["true"], raw["false"]),
            "value_token_ids": {v: plans[v]["value_token_ids"] for v in VALUES},
            "n_value_tokens": {v: len(plans[v]["value_token_ids"]) for v in VALUES},
            "prompt_tokens": len(prompt_ids), "candidate_input_tokens": lengths,
            "common_json_prefix_token_ids": plans["true"]["prefix_ids"],
            "common_suffix_token_ids": plans["true"]["suffix_ids"],
            "token_boundary_audit": "native full token IDs == prompt IDs + offset-verified continuation IDs"}


def _read_scores(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores_path, _, meta_path = _paths(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("status") != "complete":
        raise ValueError("offline metrics/calibration require complete scoring metadata")
    if meta.get("scores_sha256") != _sha256(scores_path):
        raise ValueError("completed score file SHA256 differs from scoring metadata")
    records = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != meta["config"]["n_expected"]:
        raise ValueError("score row count differs from completed metadata")
    ranking_metrics(records)
    for record in records:
        expected = normalized_score(record["log_likelihood_true"], record["log_likelihood_false"])
        if any(not math.isclose(record[k], v, rel_tol=1e-10, abs_tol=1e-12) for k, v in expected.items()):
            raise ValueError("stored score differs from raw candidate likelihoods")
    return records, meta


def _calibration(args: argparse.Namespace) -> None:
    records, meta = _read_scores(args.scores)
    if meta["config"]["data_role"] != "validation":
        raise ValueError("calibration is restricted to scores explicitly marked as validation")
    if args.output.exists():
        raise FileExistsError(args.output)
    value = {"schema_version": SCHEMA_VERSION, "kind": "validation_normal_threshold",
             "created_at": datetime.now(timezone.utc).isoformat(),
             "source_scores": str(args.scores.resolve()),
             "source_scores_sha256": _sha256(_paths(args.scores)[0]),
             "source_data_sha256": meta["config"]["data_sha256"],
             "source_subset_sha256": meta["config"]["subset_sha256"],
             "calibration_source_ids": sorted(r["source_id"] for r in records),
             "score_model_contract": meta["config"]["score_model_contract"],
             **calibrate_normal_threshold(records, args.target_fpr)}
    value["calibration_set_metrics_not_independent_test"] = threshold_metrics(records, value["threshold_log_odds"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def _run(args: argparse.Namespace) -> None:
    from .data import DECISION_PROMPT
    data = _load_data(args.data, args.limit)
    if any(r["prompt"] != DECISION_PROMPT or r["localization_eligible"] for r in data):
        raise ValueError("use a decision-only manifest with the project's exact fixed DECISION_PROMPT")
    if len({r["source_id"] for r in data}) != len(data):
        raise ValueError("scoring requires one row per unique source image")
    scores_path, metrics_path, meta_path = _paths(args.output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {"schema_version": SCHEMA_VERSION, "prompt": DECISION_PROMPT,
                "json_prefix": JSON_PREFIX, "candidates": list(VALUES),
                "likelihood": "sum log p(boolean token | native image, prompt, common JSON prefix, preceding boolean tokens)",
                "normalization": "two-option log-softmax; no length normalization; not calibrated confidence",
                "base_files_sha256": _model_files(Path(args.model)),
                "adapter_files_sha256": _model_files(Path(args.adapter)) if args.adapter else None,
                "max_pixels": args.max_pixels, "max_length": args.max_length,
                "quantization": "shared modeling.load_model language-only NF4 / visual BF16 / SDPA",
                "code_sha256": {name: _sha256(Path(__file__).with_name(name))
                                for name in ("modeling.py", "score.py", "data.py")},
                "versions": _package_versions()}
    contract["contract_sha256"] = _canonical_hash(contract)
    image_hashes = _input_images(data, args.image_root)
    config = {"model": str(Path(args.model).resolve()),
              "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
              "data": str(args.data.resolve()), "data_sha256": _sha256(args.data),
              "image_root": str(args.image_root.resolve()), "image_sha256_by_id": image_hashes,
              "data_role": args.data_role, "limit": args.limit, "run_name": args.run_name,
              "n_expected": len(data), "score_model_contract": contract,
              "subset_sha256": _canonical_hash([(r["id"], r["source_id"], r["prompt"]) for r in data]),
              "calibration_sha256": _sha256(args.calibration) if args.calibration else None}
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        if calibration.get("kind") != "validation_normal_threshold" or calibration.get("score_model_contract") != contract:
            raise ValueError("calibration model/prompt/input budget differs from this scorer")
        overlap = {r["source_id"] for r in data}.intersection(calibration["calibration_source_ids"])
        if overlap:
            raise ValueError("applying a calibration artifact requires disjoint evaluation source images")
    if args.resume:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("config") != config:
            raise ValueError("resume data/model/adapter/images/config differ; use a new output")
        if old_meta.get("status") == "complete":
            _read_scores(args.output)  # Preserve the completed file hash before any resume repair or rewrite.
        records = _read_predictions(scores_path, {r["id"] for r in data})
        expected_by_id = {r["id"]: r for r in data}
        for record in records:
            source = expected_by_id[record["id"]]
            if any(record.get(k) != v for k, v in {
                "source_id": source["source_id"], "image": source["image"],
                "label": source["is_anomalous"], "image_sha256": image_hashes[source["id"]],
                "schema_version": SCHEMA_VERSION}.items()):
                raise ValueError("resumed score provenance differs from manifest")
            recomputed = normalized_score(record["log_likelihood_true"], record["log_likelihood_false"])
            if any(record.get(k) != v for k, v in recomputed.items()):
                raise ValueError("resumed score disagrees with raw log likelihoods")
        if records:
            ranking_metrics(records)
        meta = old_meta
    else:
        if any(p.exists() for p in (scores_path, metrics_path, meta_path)):
            raise FileExistsError("output exists; use --resume or a new directory")
        records = []
        meta = {"schema_version": SCHEMA_VERSION, "config": config, "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(meta_path, meta)
    completed = {r["id"] for r in records}
    pending = [r for r in data if r["id"] not in completed]
    try:
        if pending:
            import torch
            from .modeling import input_device, load_model
            model, processor = load_model(args.model, adapter=args.adapter, training=False,
                                          max_pixels=args.max_pixels)
            device = input_device(model)
            if device.type != "cuda":
                raise RuntimeError("shared NF4 CUDA scorer required")
            meta.update(hardware={"gpu": torch.cuda.get_device_name(device), "device": str(device)},
                        model_loading_audit=getattr(model, "_industrial_vlm_audit", {}), status="running")
            _atomic_json(meta_path, meta)
            # First complete image pass is outside reported latency/memory.
            first = Path(pending[0]["image"])
            _score_image(model, processor, first if first.is_absolute() else args.image_root / first,
                         DECISION_PROMPT, args.max_length)
            torch.cuda.synchronize(device)
            with scores_path.open("a", encoding="utf-8") as output:
                for row in pending:
                    path = Path(row["image"])
                    path = path if path.is_absolute() else args.image_root / path
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    value = _score_image(model, processor, path, DECISION_PROMPT, args.max_length)
                    torch.cuda.synchronize(device)
                    record = {"schema_version": SCHEMA_VERSION, "id": row["id"],
                              "source_id": row["source_id"], "image": row["image"],
                              "image_sha256": image_hashes[row["id"]], "category": row["category"],
                              "label": row["is_anomalous"], "prompt": DECISION_PROMPT, **value,
                              "latency_seconds": time.perf_counter() - start,
                              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    records.append(record)
                    if len(records) % 10 == 0:
                        print(f"Scored {len(records)}/{len(data)}", flush=True)
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            categories[record["category"]].append(record)
        metrics = {"schema_version": SCHEMA_VERSION, "overall": ranking_metrics(records),
                   "by_category": {k: ranking_metrics(v) for k, v in sorted(categories.items())},
                   "default_equal_candidate_threshold": threshold_metrics(records, 0.0),
                   "latency_definition": "warm model; includes preprocessing and two full candidate forwards; excludes loading, hashes and warmup",
                   "mean_latency_seconds": sum(r["latency_seconds"] for r in records) / len(records),
                   "metadata": {**config, "complete": len(records) == len(data)}}
        if calibration:
            metrics["validation_calibrated_threshold"] = threshold_metrics(records, calibration["threshold_log_odds"])
            metrics["validation_calibration_warning"] = calibration["limitation"]
        _atomic_json(metrics_path, metrics)
        meta.update(status="complete", n_completed=len(records), scores_sha256=_sha256(scores_path),
                    finished_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(meta_path, meta)
        print(json.dumps({"scores": str(scores_path), "metrics": str(metrics_path),
                          "overall": metrics["overall"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        meta.update(status="failed", n_completed=len(records), error=f"{type(exc).__name__}: {exc}")
        _atomic_json(meta_path, meta)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    score = modes.add_parser("score", help="CUDA native-input two-candidate scores")
    score.add_argument("--model", required=True)
    score.add_argument("--adapter")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--image-root", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--data-role", required=True,
                       choices=("validation", "regression", "confirmation", "external_pilot"))
    score.add_argument("--max-pixels", type=int, default=524288)
    score.add_argument("--max-length", type=int, default=2048)
    score.add_argument("--limit", type=int)
    score.add_argument("--run-name")
    score.add_argument("--resume", action="store_true")
    score.add_argument("--calibration", type=Path, help="optional frozen validation-normal threshold JSON")
    calibration = modes.add_parser("calibrate", help="offline normal-only validation threshold")
    calibration.add_argument("--scores", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.mode == "score":
        if min(args.max_pixels, args.max_length) <= 0 or (args.limit is not None and args.limit <= 0):
            parser.error("pixel/token limits and optional --limit must be positive")
        _run(args)
    else:
        if not 0 <= args.target_fpr < 1:
            parser.error("target FPR must be in [0,1)")
        _calibration(args)


if __name__ == "__main__":
    main()
