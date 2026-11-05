"""Diagnose language use of anomaly-map prompts on a fixed 48-image subset.

For matched seed-42 Qwen, Qwen-LoRA and Vicuna, retain each image's global
feature and swap only the map entering PromptLearner with an opposite-label
image from the same category. This subset is a diagnostic, not an independent
test. Run after the main follow-up pipeline has released its GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import inspect
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time
import traceback

import numpy as np
import torch
from torch import nn

import evaluate_models as evaluation


ARMS = ("qwen_frozen", "qwen_lora", "vicuna")
SOURCE_COUNT = 2005
GEOMETRY = "square224_RGB_bicubic_mask_index_positive_nearest"


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    data = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def binary_label(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Image label must be a scalar")
        value = value.item()
    if value not in (0, 1):
        raise ValueError(f"Image label must be 0 or 1: {value!r}")
    return int(value)


def verify_hash(path, expected):
    actual = evaluation.sha256(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def validate_cache(root):
    """Verify all source identities and cache bytes against completed main runs."""
    cache = root / "real_cache"
    sources_path, manifest_path = cache / "sources.json", cache / "manifest.json"
    sources, manifest = read_json(sources_path), read_json(manifest_path)
    if sources.get("geometry") != GEOMETRY or sources.get("reference_mode") != "none":
        raise ValueError("Real source geometry/reference mode differs from the frozen protocol")
    if manifest.get("completed") is not True or manifest.get("geometry") != GEOMETRY or manifest.get("reference_mode") != "none":
        raise ValueError("Real cache manifest is incomplete or uses a different geometry/reference mode")
    expected = {}
    for row in sources.get("records", []):
        source = row["image"]
        if not isinstance(source, str) or not source or source in expected:
            raise ValueError("Invalid or duplicate path in frozen sources")
        expected[source] = {"class_name": row["object"], "label": binary_label(row["label_int"]),
                            "image_sha256": row["image_sha256"]}
    counts = Counter(row["label"] for row in expected.values())
    if (len(expected) != SOURCE_COUNT or sources.get("source_count") != SOURCE_COUNT
            or counts != {0: 925, 1: 1080} or len({r["class_name"] for r in expected.values()}) != 12):
        raise ValueError("Frozen sources must contain exactly 2005 images, 925 normal/1080 anomaly, 12 classes")
    records = manifest.get("records", [])
    if len(records) != SOURCE_COUNT:
        raise ValueError("Real cache manifest does not contain 2005 records")
    by_file, by_source = {}, {}
    for row in records:
        filename, source = row["file"], row["source_path"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".pt") or filename in by_file or source in by_source:
            raise ValueError("Invalid or duplicate cache filename/source in manifest")
        if source not in expected or any(row.get(key) != value for key, value in expected[source].items()):
            raise ValueError(f"Cache manifest source metadata differs from frozen protocol: {source}")
        by_file[filename], by_source[source] = row, row
    if set(by_source) != set(expected):
        raise ValueError("Manifest source set differs from frozen sources")
    if {path.name for path in cache.glob("*.pt")} != set(by_file):
        raise ValueError("Cache files differ from the complete manifest")
    manifest_sha256 = evaluation.sha256(manifest_path)
    frozen_training = read_json(root / "matched_inputs/protocol.json")
    if (frozen_training.get("steps") != 16 or frozen_training.get("batch_size") != 2
            or frozen_training.get("max_tgt_len") != 128 or frozen_training.get("cache_manifest_sha256")
            != evaluation.sha256(root.parent / "cache_balanced_train48/manifest.json")):
        raise ValueError("Matched training protocol is incomplete or changed")
    expected_runs = {row["seed"]: row for row in frozen_training.get("runs", [])}
    if 42 not in expected_runs:
        raise ValueError("Frozen seed-42 initialization/order hashes are absent")
    frozen_seed = expected_runs[42]
    anchors, training_anchors, cache_hashes = {}, {}, None
    for arm in ARMS:
        directory = root / "real_evaluations" / f"matched_{arm}42"
        run_path = directory / "run.json"
        run = read_json(run_path)
        if run.get("completed") is not True or run.get("completed_num_sources") != SOURCE_COUNT or run.get("num_sources") != SOURCE_COUNT:
            raise ValueError(f"Main matched evaluation is not complete: {arm}")
        if run.get("cache_manifest_sha256") != manifest_sha256:
            raise ValueError(f"Main matched evaluation used a different manifest: {arm}")
        hashes = run.get("cache_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(by_file):
            raise ValueError(f"Main matched evaluation cache hashes are incomplete: {arm}")
        if cache_hashes is not None and hashes != cache_hashes:
            raise ValueError("Main matched evaluations used different cache bytes")
        cache_hashes = hashes
        training_dir = root / "matched_runs" / f"{arm}_42"
        training_run_path = training_dir / "run.json"
        training = read_json(training_run_path)
        if training.get("steps") != 16 or not training.get("checkpoint_sha256"):
            raise ValueError(f"Matched training is incomplete: {arm}")
        if training.get("checkpoint_sha256") != run.get("bridge", {}).get("sha256"):
            raise ValueError(f"Main evaluation does not use the completed training checkpoint: {arm}")
        schedule_path = training_dir / "sample_order.json"
        schedule_hash = evaluation.sha256(schedule_path)
        if schedule_hash != training.get("sample_order_sha256") or schedule_hash != frozen_seed.get("sample_order_sha256"):
            raise ValueError(f"Matched training sample order differs from the frozen seed-42 schedule: {arm}")
        if (training.get("initial_bridge_sha256", training.get("initial_checkpoint_sha256"))
                != frozen_seed.get("bridge_init_sha256")):
            raise ValueError(f"Matched training bridge initialization file differs from protocol: {arm}")
        if (training_dir / "pytorch_model.pt").is_file():
            verify_hash(training_dir / "pytorch_model.pt", training["checkpoint_sha256"])
        if arm == "qwen_lora":
            train_adapter = training_dir / "lora_adapter"
            eval_adapter_hashes = run.get("adapter_hashes", {})
            if not eval_adapter_hashes:
                raise ValueError("Completed Qwen LoRA evaluation has no adapter hashes")
            for filename, expected_hash in eval_adapter_hashes.items():
                if Path(filename).name != filename:
                    raise ValueError("Adapter filename is not a simple filename")
                verify_hash(train_adapter / filename, expected_hash)
        elif run.get("adapter_hashes"):
            raise ValueError(f"Unexpected language adapter in {arm} evaluation")
        training_anchors[arm] = {"run_sha256": evaluation.sha256(training_run_path),
                                 "initial_bridge_fingerprint": training.get("actual_initial_bridge_fingerprint"),
                                 "sample_order_sha256": training.get("sample_order_sha256"),
                                 "checkpoint_sha256": training.get("checkpoint_sha256")}
        prediction_path = directory / "predictions.jsonl"
        verify_hash(prediction_path, run.get("prediction_sha256"))
        prediction_rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        prediction_sources = set()
        for row in prediction_rows:
            source = row["source_path"]
            reference = expected.get(source)
            if (reference is None or source in prediction_sources
                    or row.get("cache_file") not in by_file or by_file[row["cache_file"]]["source_path"] != source
                    or row.get("label") != reference["label"] or row.get("class_name") != reference["class_name"]
                    or row.get("source_id") != hashlib.sha256(source.encode()).hexdigest()[:24]):
                raise ValueError(f"Main evaluation source identity differs from frozen protocol: {arm}, {source}")
            prediction_sources.add(source)
        if len(prediction_rows) != SOURCE_COUNT or prediction_sources != set(expected):
            raise ValueError(f"Main evaluation predictions are not the full source set: {arm}")
        anchors[arm] = {"run_sha256": evaluation.sha256(run_path), "prediction_sha256": run["prediction_sha256"],
                        "bridge_sha256": run.get("bridge", {}).get("sha256"),
                        "adapter_hashes": run.get("adapter_hashes", {})}
    if (len({row["initial_bridge_fingerprint"] for row in training_anchors.values()}) != 1
            or None in {row["initial_bridge_fingerprint"] for row in training_anchors.values()}):
        raise ValueError("Matched seed-42 arms do not share the same actual initial bridge tensors")
    if (len({row["sample_order_sha256"] for row in training_anchors.values()}) != 1
            or None in {row["sample_order_sha256"] for row in training_anchors.values()}):
        raise ValueError("Matched seed-42 arms do not share one frozen sample order")
    for index, (filename, expected_hash) in enumerate(sorted(cache_hashes.items()), 1):
        verify_hash(cache / filename, expected_hash)
        if index % 250 == 0 or index == SOURCE_COUNT:
            print(json.dumps({"phase": "cache_integrity", "verified": index, "total": SOURCE_COUNT}), flush=True)
    source_copy = root / "real_sources.json"
    if source_copy.is_file() and evaluation.sha256(source_copy) != evaluation.sha256(sources_path):
        raise ValueError("Frozen source copies differ")
    return records, {"completed": True, "num_sources": SOURCE_COUNT, "normal_count": 925, "anomaly_count": 1080,
                     "num_classes": 12, "geometry": GEOMETRY, "reference_mode": "none",
                     "sources_sha256": evaluation.sha256(sources_path), "cache_manifest_sha256": manifest_sha256,
                     "cache_hashes": cache_hashes, "main_evaluation_anchors": anchors,
                     "training_anchors": training_anchors,
                     "cache_hashes_sha256": hashlib.sha256(json.dumps(cache_hashes, sort_keys=True).encode()).hexdigest()}


class SwapPromptLearner(nn.Module):
    """Change only the batch correspondence of maps entering the same module."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, anomaly_maps):
        if anomaly_maps.shape[0] != 4:
            raise ValueError("The diagnostic requires a category batch of four")
        return self.original(torch.roll(anomaly_maps, shifts=2, dims=0))


def select_sources(records, cache_hashes):
    """Use labels only to freeze two normal and two anomalous sources per class."""
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["class_name"]][binary_label(record["label"])].append(record)
    if len(grouped) != 12:
        raise ValueError(f"Expected 12 VisA categories, found {len(grouped)}")
    selected = []
    for class_name in sorted(grouped):
        batch = []
        for label in (0, 1):
            candidates = sorted(grouped[class_name][label], key=lambda item: item["source_path"])
            if len(candidates) < 2:
                raise ValueError(f"Insufficient class/label sources: {class_name}, {label}")
            for record in candidates[:2]:
                batch.append({"class_name": class_name, "label": label,
                              "source_path": record["source_path"], "cache_file": record["file"],
                              "image_sha256": record["image_sha256"],
                              "cache_sha256": cache_hashes[record["file"]]})
        for index, item in enumerate(batch):
            item["swap_source_path"] = batch[(index - 2) % 4]["source_path"]
            item["swap_cache_file"] = batch[(index - 2) % 4]["cache_file"]
        selected.append(batch)
    return selected


def load_selected(cache, item):
    path = Path(cache) / item["cache_file"]
    verify_hash(path, item["cache_sha256"])
    record = evaluation.load_torch(path)
    if (record.get("source_path") != item["source_path"] or record.get("class_name") != item["class_name"]
            or binary_label(record.get("label")) != item["label"] or record.get("image_sha256") != item["image_sha256"]):
        raise ValueError(f"Selected cache metadata changed: {path}")
    if tuple(record["mask"].shape) != (1, 224, 224):
        raise ValueError(f"Selected real mask has the wrong shape: {path}")
    return record


def inference_records(cache, batch, with_masks=False):
    """Drop evaluation labels/masks before passing records into model functions."""
    keep = ("global_feature", "patch_features", "text_features", "class_name",
            "source_path", "question")
    records, masks = [], []
    for item in batch:
        record = load_selected(cache, item)
        records.append({key: record[key] for key in keep})
        if with_masks:
            masks.append(record["mask"][0].cpu().numpy() > 0)
    if len({record["question"] for record in records}) != 1:
        raise ValueError("Selected category batch has inconsistent questions")
    return (records, masks) if with_masks else records


def map_statistics(array):
    return {"min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std()),
            "median": float(np.median(array)), "q95": float(np.quantile(array, 0.95))}


def difference_statistics(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Difference inputs have mismatched shapes or non-finite values")
    delta = (second - first).reshape(-1)
    absolute = np.abs(delta)
    return {"num_values": int(delta.size), "exact_equal": bool(np.array_equal(first, second)),
            "l2": float(np.linalg.norm(delta)), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "mean_absolute": float(absolute.mean()), "max_absolute": float(absolute.max()),
            "signed_mean": float(delta.mean())}


def candidate_predictions(model, backend, prefix, attention, audit):
    """Repeat the same continuation scoring path, without another generation."""
    first_logits, past = evaluation.lm_forward(model, backend, prefix, attention)
    scores = {name: evaluation.continuation_logp(model, backend, prefix, attention,
                                              audit[name]["ids"], first_logits)
              for name in ("yes", "no")}
    values = []
    for yes, no in zip(scores["yes"].tolist(), scores["no"].tolist()):
        values.append({"yes_logp": float(yes), "no_logp": float(no),
                       "answer_log_odds": float(yes - no), "candidate_prediction": int(yes > no)})
    del past, first_logits, scores
    return values


def flush_rows(path, rows_by_source, selected):
    """Atomically save every completed condition; a partial row stays explicit."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for item in (item for batch in selected for item in batch):
            row = rows_by_source.get(item["source_path"])
            if row is not None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def error_details(error):
    return {"type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}


def evidence_summary(rows, key):
    entries = [row[key] for row in rows]
    return {"num_sources": len(entries), "exact_equal_count": sum(item["exact_equal"] for item in entries),
            **{f"{field}_{aggregation}": float(function([item[field] for item in entries]))
               for field in ("l2", "rms", "mean_absolute", "max_absolute")
               for aggregation, function in (("mean", np.mean), ("max", np.max))}}


def summarize_pairs(rows):
    metrics = {"num_sources": len(rows), "conditions": {}}
    for condition in ("actual", "swapped"):
        values = [{"label": row["label"], **row[condition]} for row in rows]
        metrics["conditions"][condition] = {
            "generation": evaluation.classification(values),
            "candidate_classification": evaluation.classification(values, "candidate_prediction"),
            "answer_image": evaluation.ranking(
                [row["label"] for row in values], [row["answer_log_odds"] for row in values]),
        }
    answer_flips = [row["actual"]["prediction"] != row["swapped"]["prediction"] for row in rows]
    valid_pairs = [row for row in rows if row["actual"]["prediction"] != -1
                   and row["swapped"]["prediction"] != -1]
    valid_flips = [row["actual"]["prediction"] != row["swapped"]["prediction"]
                   for row in valid_pairs]
    deltas = np.array([row["swapped"]["answer_log_odds"] - row["actual"]["answer_log_odds"]
                       for row in rows], dtype=np.float64)
    metrics["sensitivity"] = {
        "answer_label_flip_count": int(sum(answer_flips)),
        "answer_label_flip_rate": float(np.mean(answer_flips)),
        "both_valid_answer_pairs": len(valid_pairs),
        "both_valid_answer_flip_rate": float(np.mean(valid_flips)) if valid_flips else None,
        "candidate_flip_count": sum(row["actual"]["candidate_prediction"] != row["swapped"]["candidate_prediction"]
                                    for row in rows),
        "candidate_flip_rate": float(np.mean([
            row["actual"]["candidate_prediction"] != row["swapped"]["candidate_prediction"] for row in rows])),
        "answer_text_change_rate": float(np.mean([
            row["actual"]["answer"] != row["swapped"]["answer"] for row in rows])),
        "absolute_log_odds_delta_mean": float(np.abs(deltas).mean()),
        "absolute_log_odds_delta_median": float(np.median(np.abs(deltas))),
        "absolute_log_odds_delta_max": float(np.abs(deltas).max()),
        "signed_log_odds_delta_mean": float(deltas.mean()),
    }
    metrics["sensitivity"]["signed_log_odds_delta_by_actual_label"] = {
        str(label): float(np.mean([delta for delta, row in zip(deltas, rows) if row["label"] == label]))
        for label in (0, 1)
    }
    repeated_rows = [row for row in rows if "actual_repeat_candidate" in row]
    repeats = np.asarray([row["actual_repeat_candidate"]["answer_log_odds"] - row["actual"]["answer_log_odds"]
                          for row in repeated_rows], dtype=np.float64)
    noise_max = float(np.abs(repeats).max()) if len(repeats) else None
    metrics["same_actual_repeat"] = {
        "num_sources": len(repeated_rows), "generation_repeated": False,
        "absolute_log_odds_delta_mean": float(np.abs(repeats).mean()) if len(repeats) else None,
        "absolute_log_odds_delta_median": float(np.median(np.abs(repeats))) if len(repeats) else None,
        "absolute_log_odds_delta_max": noise_max,
        "candidate_flip_count": sum(row["actual_repeat_candidate"]["candidate_prediction"] != row["actual"]["candidate_prediction"]
                                    for row in repeated_rows),
        "swap_delta_above_max_observed_repeat_delta_count": int((np.abs(deltas) > noise_max).sum()) if noise_max is not None else None,
        "interpretation": "Same actual prefix is rescored with identical candidate-scoring/cache settings and no generation. This is an observed numerical reference, not a noise bound or a significance test.",
    }
    metrics["intervention_evidence"] = {
        **{key: evidence_summary(rows, key) for key in
           ("donor_map_difference", "soft_prompt_difference", "adaptive_prompt_difference", "base_prompt_difference",
            "soft_prompt_donor_alignment")},
        "same_actual_repeat_log_odds": ({"num_sources": len(repeated_rows),
                                          "absolute_log_odds_delta_mean": float(np.abs(repeats).mean()),
                                          "absolute_log_odds_delta_median": float(np.median(np.abs(repeats))),
                                          "absolute_log_odds_delta_max": noise_max}
                                         if len(repeated_rows) else {"num_sources": 0}),
    }
    metrics["interpretation"] = (
        "Small changes under this map swap do not establish that the language model ignores prompts. "
        "Consider donor-map separation, adaptive PromptLearner output separation, and the same-actual numerical reference. "
        "The intervention is a selected diagnostic and does not establish localization correctness or anomaly-detection gains.")
    return metrics


def map_metrics(rows, destination):
    """Original image/map quality is separate from swapped language conditions."""
    from exact_bf16_ranking import ranking as pixel_ranking
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
    per_class = {}
    for class_name, group in sorted(by_class.items()):
        masks, scores = [], []
        for row in group:
            with np.load(destination / row["map_file"]) as values:
                masks.append(values["mask"].reshape(-1))
                scores.append(values["anomaly_map"].reshape(-1))
        per_class[class_name] = {
            "pixel": pixel_ranking(np.concatenate(masks), np.concatenate(scores), unsupported="raise"),
            "image": evaluation.ranking([row["label"] for row in group],
                                         [row["original_map_statistics"]["max"] for row in group]),
        }
    return {
        "image_pooled": evaluation.ranking([row["label"] for row in rows],
                                            [row["original_map_statistics"]["max"] for row in rows]),
        "per_class": per_class,
        "category_macro": {
            family: {key: float(np.mean([value[family][key] for value in per_class.values()
                                        if value[family][key] is not None]))
                     for key in ("auroc", "ap")}
            for family in ("pixel", "image")
        },
        "definition": "Only original per-image averaged maps are scored against their own GT; swapping changes language conditioning, not these maps.",
    }


def model_arguments(experiment, root, arm, device, vicuna_ckpt):
    checkpoint = root / "matched_runs" / f"{arm}_42" / "pytorch_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    backend = "vicuna" if arm == "vicuna" else "qwen"
    return SimpleNamespace(
        backend=backend, code_dir=str(experiment / "code"),
        model_factory="train_qwen_matched:factory" if backend == "qwen" else "",
        qwen_ckpt="/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct",
        vicuna_ckpt=str(vicuna_ckpt), init_ckpt=str(experiment / "ckpt/train_visa/pytorch_model.pt"),
        checkpoint=str(checkpoint),
        lora_adapter=str(checkpoint.parent / "lora_adapter") if arm == "qwen_lora" else "",
        imagebind_ckpt="", vicuna_full_precision=False, device=device, seed=42,
        max_tgt_len=128, export_initial_bridge="", export_only=False,
    )


def run_arm(experiment, root, destination, selected, arm, device, max_new_tokens, vicuna_ckpt, integrity):
    destination.mkdir()
    (destination / "maps").mkdir()
    prediction_path = destination / "predictions.jsonl"
    rows_by_source, maps_by_class, soft_by_class = {}, {}, {}
    metadata = {"arm": arm, "seed": 42, "max_new_tokens": max_new_tokens,
                "source_sha256": evaluation.sha256(__file__),
                "evaluation_source_sha256": evaluation.sha256(evaluation.__file__),
                "selection_sha256": evaluation.sha256(destination.parent / "selection.json"),
                "conditions": ["actual", "actual_repeat_candidate", "swapped"],
                "candidate_audits": {}, "completed": False, "status": "initializing", "phase": "build_model"}
    evaluation.write_json(destination / "run.json", metadata)
    flush_rows(prediction_path, rows_by_source, selected)
    model = original_prompt = hook = None
    prefix = attention = maps = records = predictions = repeated = None
    captured = []
    started = time.time()

    def persist(phase):
        metadata.update(phase=phase, prediction_rows=len(rows_by_source),
                        condition_counts={condition: sum(condition in row for row in rows_by_source.values())
                                          for condition in ("actual", "actual_repeat_candidate", "swapped")})
        flush_rows(prediction_path, rows_by_source, selected)
        metadata["prediction_sha256"] = evaluation.sha256(prediction_path)
        evaluation.write_json(destination / "run.json", metadata)

    def capture_prompt(module, inputs, output):
        captured.append(output.detach().float().cpu().numpy().copy())

    def take_prompt():
        if len(captured) != 1:
            raise ValueError("Expected exactly one PromptLearner output per prefix")
        value = captured.pop()
        if value.shape[0] != 4 or value.shape[1] != 18 or not np.isfinite(value).all():
            raise ValueError("PromptLearner output must contain 9 base + 9 adaptive tokens")
        return value

    try:
        args = model_arguments(experiment, root, arm, device, vicuna_ckpt)
        args.max_new_tokens = max_new_tokens
        metadata["args"] = vars(args)
        train_run_path = Path(args.checkpoint).parent / "run.json"
        anchor = integrity["main_evaluation_anchors"][arm]
        verify_hash(train_run_path, integrity["training_anchors"][arm]["run_sha256"])
        verify_hash(args.checkpoint, anchor["bridge_sha256"])
        if args.lora_adapter:
            metadata["adapter_hashes"] = {path.name: evaluation.sha256(path)
                                          for path in sorted(Path(args.lora_adapter).glob("*")) if path.is_file()}
            if metadata["adapter_hashes"] != anchor["adapter_hashes"]:
                raise ValueError("Diagnostic adapter files differ from the completed main evaluation")
        metadata["train_run_sha256"] = evaluation.sha256(train_run_path)
        metadata["init_sha256"] = evaluation.sha256(args.init_ckpt)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        model, bridge_audit = evaluation.build_model(args)
        if bridge_audit["sha256"] != anchor["bridge_sha256"]:
            raise ValueError("Loaded diagnostic bridge differs from the main evaluation")
        metadata["bridge"] = bridge_audit
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_source = inspect.getsourcefile(type(model))
        metadata["model_source_sha256"] = evaluation.sha256(model_source) if model_source and Path(model_source).is_file() else None
        original_prompt = model.prompt_learner
        hook = original_prompt.register_forward_hook(capture_prompt)
        metadata["status"] = "running"
        cache = root / "real_cache"
        with torch.inference_mode():
            for batch in selected:
                class_name = batch[0]["class_name"]
                metadata.update(phase="actual", current_class=class_name)
                evaluation.write_json(destination / "run.json", metadata)
                records, masks = inference_records(cache, batch, with_masks=True)
                maps, prefix, attention, question = evaluation.map_and_prefix(model, records)
                soft = take_prompt()
                arrays = maps.float().cpu().numpy()
                maps_by_class[class_name], soft_by_class[class_name] = arrays.copy(), soft.copy()
                audit = evaluation.candidate_audit(model.llama_tokenizer, question)
                metadata["candidate_audits"][class_name] = audit
                predictions = evaluation.language_predictions(model, args.backend, prefix, attention, audit, max_new_tokens)
                if len(predictions) != len(batch):
                    raise ValueError("Actual prediction count differs from category batch")
                for index, (item, array, mask, prediction) in enumerate(zip(batch, arrays, masks, predictions)):
                    map_file = f"maps/{class_name}_{index}.npz"
                    np.savez_compressed(destination / map_file, anomaly_map=array, mask=mask.astype(np.uint8))
                    rows_by_source[item["source_path"]] = {**item, "map_file": map_file,
                        "original_map_statistics": map_statistics(array), "actual": prediction}
                persist("actual")
                # Same batch/prefix/cache settings as the original candidate scores;
                # no second autoregressive generation is performed.
                metadata["phase"] = "actual_repeat_candidate"
                evaluation.write_json(destination / "run.json", metadata)
                repeated = candidate_predictions(model, args.backend, prefix, attention, audit)
                if len(repeated) != len(batch):
                    raise ValueError("Repeat candidate count differs from category batch")
                for item, prediction in zip(batch, repeated):
                    rows_by_source[item["source_path"]]["actual_repeat_candidate"] = prediction
                persist("actual_repeat_candidate")
                print(json.dumps({"arm": arm, "condition": "actual_and_candidate_repeat", "class": class_name}), flush=True)
            model.prompt_learner = SwapPromptLearner(original_prompt).eval()
            for batch in selected:
                class_name = batch[0]["class_name"]
                metadata.update(phase="swapped", current_class=class_name)
                evaluation.write_json(destination / "run.json", metadata)
                records = inference_records(cache, batch)
                maps, prefix, attention, question = evaluation.map_and_prefix(model, records)
                swapped_soft = take_prompt()
                arrays = maps.float().cpu().numpy()
                original_maps, original_soft = maps_by_class[class_name], soft_by_class[class_name]
                if not np.array_equal(arrays, original_maps):
                    raise ValueError("Original anomaly maps changed between conditions")
                predictions = evaluation.language_predictions(model, args.backend, prefix, attention,
                                                               metadata["candidate_audits"][class_name], max_new_tokens)
                if len(predictions) != len(batch):
                    raise ValueError("Swapped prediction count differs from category batch")
                for index, (item, prediction) in enumerate(zip(batch, predictions)):
                    donor = (index - 2) % 4
                    if batch[donor]["label"] == item["label"]:
                        raise ValueError("Donor does not have the opposite official image label")
                    row = rows_by_source[item["source_path"]]
                    row.update(swapped=prediction, original_map_unchanged=True,
                        donor_map_difference=difference_statistics(original_maps[index], original_maps[donor]),
                        soft_prompt_difference=difference_statistics(original_soft[index], swapped_soft[index]),
                        adaptive_prompt_difference=difference_statistics(original_soft[index, 9:], swapped_soft[index, 9:]),
                        base_prompt_difference=difference_statistics(original_soft[index, :9], swapped_soft[index, :9]),
                        soft_prompt_donor_alignment=difference_statistics(original_soft[donor], swapped_soft[index]),
                        absolute_log_odds_delta=abs(prediction["answer_log_odds"] - row["actual"]["answer_log_odds"]),
                        answer_label_flipped=prediction["prediction"] != row["actual"]["prediction"])
                    if not row["base_prompt_difference"]["exact_equal"]:
                        raise ValueError("Base prompt tokens changed under a map-only intervention")
                persist("swapped")
                print(json.dumps({"arm": arm, "condition": "swapped", "class": class_name}), flush=True)
        rows = [rows_by_source[item["source_path"]] for batch in selected for item in batch]
        if len(rows) != 48 or any(any(condition not in row for condition in ("actual", "actual_repeat_candidate", "swapped")) for row in rows):
            raise ValueError("Diagnostic does not contain 48 fully paired sources")
        metrics = summarize_pairs(rows)
        metrics["per_class"] = {batch[0]["class_name"]: summarize_pairs(
            [rows_by_source[item["source_path"]] for item in batch]) for batch in selected}
        metrics["original_map_quality"] = map_metrics(rows, destination)
        metrics.update(seconds=time.time() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        evaluation.write_json(destination / "metrics.json", metrics)
        metadata.update(completed=True, status="complete", num_sources=len(rows), phase="complete", current_class=None)
        persist("complete")
        return metrics
    except BaseException as error:
        metadata.update(completed=False, status="failed", failed=error_details(error), seconds=time.time() - started)
        persist(metadata.get("phase", "initializing"))
        raise
    finally:
        if hook is not None:
            hook.remove()
        if model is not None and original_prompt is not None:
            model.prompt_learner = original_prompt
        # Drop cached GPU prefixes before releasing this arm.
        prefix = attention = maps = records = predictions = repeated = None
        model = original_prompt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Follow-up directory; default EXPERIMENT/followup_20261026")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    experiment = args.experiment.resolve()
    root = args.root.resolve() if args.root else experiment / "followup_20261026"
    if args.out.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {args.out}")
    args.out.mkdir(parents=True)
    state = {"running": "preflight", "completed": [], "finished": False, "failed": None,
             "max_new_tokens": args.max_new_tokens,
             "arms": {arm: {"status": "pending", "completed": False} for arm in ARMS}}
    evaluation.write_json(args.out / "state.json", state)
    try:
        for state_path in (root / "state.json", root / "safe_state.json"):
            if state_path.is_file() and read_json(state_path).get("running"):
                raise RuntimeError("Main follow-up pipeline still has a running job; wait for GPU release")
        resource_path = root / "vicuna_resource.json"
        resource = read_json(resource_path)
        if resource.get("all_tensors_equal") is not True:
            raise ValueError("Vicuna conversion lacks tensor-equivalence audit")
        vicuna_ckpt = Path(resource["safetensors_path"])
        if not vicuna_ckpt.is_dir():
            raise FileNotFoundError(vicuna_ckpt)
        records, integrity = validate_cache(root)
        evaluation.write_json(args.out / "cache_integrity.json", integrity)
        selected = select_sources(records, integrity["cache_hashes"])
        if sum(map(len, selected)) != 48:
            raise ValueError("The diagnostic requires exactly 48 selected sources")
        selection = {
            "num_sources": 48, "per_category": "2 normal + 2 anomaly", "max_new_tokens": args.max_new_tokens,
            "selection": "Each class/label sorted by source path; take the first two; no model scores used",
            "batch_order": "normal, normal, anomaly, anomaly; torch.roll(+2) swaps opposite labels within class",
            "label_usage": "Labels select the diagnostic subset and are used for posthoc metrics; model inference receives neither label nor mask",
            "evaluation_scope": "Diagnostic subset of the already used real evaluation cache; not an independent test",
            "intervention": "Only predicted anomaly maps entering PromptLearner are swapped; own global ImageBind feature and question remain fixed",
            "same_actual_repeat": "All 48 actual prefixes receive candidate-only repeat scoring; no second generation",
            "batches": selected, "cache_manifest_sha256": integrity["cache_manifest_sha256"],
            "sources_sha256": integrity["sources_sha256"], "cache_integrity_sha256": evaluation.sha256(args.out / "cache_integrity.json"),
            "vicuna_resource_sha256": evaluation.sha256(resource_path),
        }
        evaluation.write_json(args.out / "selection.json", selection)
        if not torch.cuda.is_available():
            raise RuntimeError("This diagnostic uses the same NF4 CUDA models as the matched experiment")
        torch.cuda.set_device(torch.device(args.device))
        results = {}
        for arm in ARMS:
            state["running"] = arm
            state["arms"][arm] = {"status": "running", "completed": False}
            evaluation.write_json(args.out / "state.json", state)
            results[arm] = run_arm(experiment, root, args.out / arm, selected, arm,
                                   args.device, args.max_new_tokens, vicuna_ckpt, integrity)
            state["arms"][arm] = {"status": "complete", "completed": True}
            state["completed"].append(arm)
            evaluation.write_json(args.out / "state.json", state)
        evaluation.write_json(args.out / "summary.json", {
            "scope": selection["evaluation_scope"], "num_sources": 48, "max_new_tokens": args.max_new_tokens,
            "selection_sha256": evaluation.sha256(args.out / "selection.json"), "models": results,
            "interpretation": "Map-swap sensitivity is diagnostic. Small changes do not prove prompts are ignored; inspect donor-map/adaptive-prompt differences and candidate-only repeat variation. This does not establish localization correctness or detection gains.",
        })
        state.update(running=None, finished=True)
        evaluation.write_json(args.out / "state.json", state)
        print("PROMPT_MAP_ABLATION_COMPLETE", flush=True)
    except BaseException as error:
        running = state.get("running")
        details = error_details(error)
        if running in state["arms"]:
            state["arms"][running] = {"status": "failed", "completed": False, "failed": details}
        for arm in ARMS:
            if state["arms"][arm]["status"] == "pending":
                state["arms"][arm] = {"status": "not_run", "completed": False,
                                       "reason": "Stopped after preflight or another arm failed"}
        state.update(running=None, finished=False, failed={"phase_or_arm": running, **details})
        evaluation.write_json(args.out / "state.json", state)
        raise


if __name__ == "__main__":
    main()
