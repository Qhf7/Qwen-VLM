"""One fixed validation query for native 448px three-image ROI execution.

Default: CPU plan only. --run explicitly loads the official BF16 CUDA model
and reuses migration_roi_eval scoring and greedy cached generation. This is
an execution/geometry/memory check; no detection-performance metric is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import migration_roi_eval as roi


PIPELINE_VERSION = "qwen3vl-native-roi-preflight-v1"
IMAGE_SIZE = 448
EXPECTED_GRID = [[1, 28, 28]] * 3
EXPECTED_VIEW_ROLES = ["target_original", "target_roi", "normal_training_reference"]
REQUIRED_CHECKS = (
    "source_and_parity", "model_bf16", "validation_query", "train_normal_reference",
    "three_view_roles", "actual_448_grids", "visual_tokens_588", "input_pixels_bf16",
    "finite_forward_logits", "finite_yes_no_score", "generation_output_contract",
    "official_score_and_generation_prefill", "cached_generation_observed", "memory_recorded",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=roi.DEFAULT_MODEL)
    parser.add_argument("--cache", default="runs/migration_20261103/teacher_cache.pt")
    parser.add_argument("--parity-report", default="runs/migration_20261103/native_parity/report.json")
    parser.add_argument("--out", default="runs/migration_20261103/roi_preflight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-area", type=float, default=1.0 / 9.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    require(args.max_new_tokens >= 2, "Need at least two generation tokens to exercise cached decoding")
    roi.shifted_square_bbox([0.5, 0.5], args.roi_area)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "run.json").exists(), "Existing preflight result is preserved; choose a fresh output")
    cache, cache_audit = roi.load_shared_cache(args.cache)
    items = cache["items"]
    index = roi.select_validation_indices(items, limit=1, seed=args.seed)[0]
    query = items[index]
    bank = roi.build_reference_bank(items, cache["cls"])
    reference = roi.select_reference(index, items, cache["cls"], bank)
    reference = {**reference, "label": int(items[reference["cache_index"]]["label"])}
    parity = json.loads(Path(args.parity_report).read_text())
    require(parity.get("passed") is True and parity.get("all_cases_exercised_cached_decoding") is True
            and parity.get("source_audit", {}).get("passed") is True, "Native cached parity report must pass")
    parity_source = parity["source_audit"].get("source_sha256")
    require(isinstance(parity_source, str) and len(parity_source) == 64, "Missing native parity source identity")
    views, metadata = roi.build_views(query, cache["maps"][index], roi.REFERENCE_CONDITION,
                                     args.seed, args.roi_area, IMAGE_SIZE, reference)
    parity_audit = {"passed": True, "path": args.parity_report, "sha256": roi.sha256_file(args.parity_report),
                    "source_sha256": parity_source, "all_cases_exercised_cached_decoding": True}
    plan = {"pipeline_version": PIPELINE_VERSION, "roi_pipeline_version": roi.PIPELINE_VERSION,
            "status": "planned", "passed": False, "split": "val", "n": 1,
            "condition": roi.REFERENCE_CONDITION, "image_size": IMAGE_SIZE,
            "config": vars(args), "cache_index": index, "query": query, "reference": reference,
            "subset_selection": "first fixed category/label-round-robin validation query, same ROI selector and seed",
            "expected_image_grids": EXPECTED_GRID, "expected_image_tokens_per_view": [196] * 3,
            "expected_image_tokens_total": 588, "view_metadata": metadata, "question": roi.QUESTION,
            "cache_audit": cache_audit, "parity_audit": parity_audit, "gpu_model_loaded": False,
            "required_checks": list(REQUIRED_CHECKS), "performance_metrics_computed": False,
            "roi_script_sha256": roi.sha256_file(Path(roi.__file__)),
            "script_sha256": roi.sha256_file(Path(__file__))}
    plan["plan_sha256"] = canonical_sha(plan)
    roi.atomic_json(plan, out / "plan.json")
    if not args.run:
        print(json.dumps({"status": "planned", "gpu_model_loaded": False, "n": 1,
                          "condition": roi.REFERENCE_CONDITION, "plan": str(out / "plan.json")}))
        return

    run = {**plan, "status": "running", "checks": {name: False for name in REQUIRED_CHECKS}}
    roi.atomic_json(run, out / "run.json")
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
        from migration_native_parity import audit_dtypes, source_audit

        require(args.device.startswith("cuda") and torch.cuda.is_available(), "--run requires CUDA")
        require(torch.cuda.is_bf16_supported(), "Native ROI preflight requires BF16 support")
        audit = source_audit(args.model)
        run["source_audit"] = audit
        require(audit.get("passed") is True and audit["source_sha256"] == parity_source,
                "Current official source and native parity must match")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": args.device},
            attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True).eval()
        model.requires_grad_(False)
        run["gpu_model_loaded"] = True
        run["dtype_audit"] = audit_dtypes(model)
        require(run["dtype_audit"]["passed"] is True, "All native model parameters must be BF16")
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        yes_id, no_id = (roi.single_token_id(processor.tokenizer, value) for value in ("Yes", "No"))
        cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
                               bos_token_id=model.generation_config.bos_token_id,
                               eos_token_id=model.generation_config.eos_token_id,
                               pad_token_id=model.generation_config.pad_token_id)
        run["runtime"] = {"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
                          "device": args.device, "attention_backend": "sdpa", "encoder_dtype": "torch.bfloat16",
                          "yes_id": yes_id, "no_id": no_id, "inference_mode": True,
                          "generation_use_cache": True, "max_new_tokens": args.max_new_tokens}
        events = []

        def capture_forward(module, positional, kwargs, output):
            logits = output.logits
            pixel = kwargs.get("pixel_values")
            position = kwargs.get("cache_position")
            events.append({"use_cache": kwargs.get("use_cache") is True,
                           "has_pixel_values": pixel is not None,
                           "pixel_values_dtype": None if pixel is None else str(pixel.dtype),
                           "cache_position_first": None if position is None else int(position.reshape(-1)[0].item()),
                           "input_tokens": int(kwargs["input_ids"].shape[1]),
                           "logits_dtype": str(logits.dtype), "logits_shape": list(logits.shape),
                           "logits_finite": bool(torch.isfinite(logits).all().item()),
                           "inference_mode": torch.is_inference_mode_enabled()})

        with torch.inference_mode():
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            baseline = int(torch.cuda.memory_allocated(args.device))
            free, total = torch.cuda.mem_get_info(args.device)
            batch, tokens = roi.prepare_processor_inputs(processor, model, views, args.device, IMAGE_SIZE)
            tensor_audit = {key: {"dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device)}
                            for key, value in batch.items() if isinstance(value, torch.Tensor)}
            handle = model.register_forward_hook(capture_forward, with_kwargs=True)
            try:
                prediction = roi.score_and_generate(model, processor, batch, yes_id, no_id, cfg, args.device)
            finally:
                handle.remove()
            torch.cuda.synchronize(args.device)
            memory = {"baseline_gpu_allocated_bytes": baseline,
                      "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                      "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
                      "free_gpu_bytes_before_query": int(free), "total_gpu_bytes": int(total)}
            del batch, views
        record = {**query, "cache_index": index, "condition": roi.REFERENCE_CONDITION,
                  **metadata, **tokens, **prediction}
        run.update(record=record, tensor_audit=tensor_audit, forward_trace=events, memory=memory)
        cached = [event for event in events if event["use_cache"] and not event["has_pixel_values"]
                  and event["cache_position_first"] is not None and event["cache_position_first"] > 0]
        run["generation_cached_steps"] = len(cached)
        generated_ids = prediction["generated_token_ids"]
        checks = {
            "source_and_parity": audit["passed"] is True and audit["source_sha256"] == parity_source,
            "model_bf16": run["dtype_audit"]["passed"] is True,
            "validation_query": query["split"] == "val",
            "train_normal_reference": reference["split"] == "train" and reference["label"] == 0
                and reference["category"] == query["category"] and reference["path"] != query["path"],
            "three_view_roles": tokens["n_images"] == 3 and metadata["view_roles"] == EXPECTED_VIEW_ROLES,
            "actual_448_grids": tokens["image_grid_thw"] == EXPECTED_GRID,
            "visual_tokens_588": tokens["image_tokens_per_view"] == [196] * 3 and tokens["image_tokens_total"] == 588,
            "input_pixels_bf16": tensor_audit["pixel_values"]["dtype"] == "torch.bfloat16",
            "finite_forward_logits": bool(events) and all(event["logits_finite"] and event["inference_mode"] for event in events),
            "finite_yes_no_score": all(math.isfinite(prediction[name]) for name in ("yes_logit", "no_logit", "score_yes_minus_no")),
            "generation_output_contract": isinstance(prediction["generated_text"], str)
                and 1 <= len(generated_ids) <= args.max_new_tokens
                and all(type(token) is int and token >= 0 for token in generated_ids),
            "official_score_and_generation_prefill": len(events) >= 2
                and events[0]["has_pixel_values"] and not events[0]["use_cache"]
                and events[1]["has_pixel_values"] and events[1]["use_cache"]
                and events[1]["cache_position_first"] == 0,
            "cached_generation_observed": bool(cached),
            "memory_recorded": memory["peak_gpu_reserved_bytes"] >= memory["peak_gpu_allocated_bytes"] >= baseline >= 0
                and memory["total_gpu_bytes"] > 0,
        }
        run["checks"] = checks
        run["failed_checks"] = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
        require(not run["failed_checks"], "Preflight execution checks failed: " + ", ".join(run["failed_checks"]))
        run.update(status="complete", passed=True)
    except Exception as error:
        run.update(status="failed", passed=False, error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        roi.atomic_json(run, out / "run.json")
    print(json.dumps({"status": run["status"], "passed": run["passed"], "n": 1,
                      "visual_tokens": 588, "memory": run["memory"], "run": str(out / "run.json")}))


if __name__ == "__main__":
    main()
