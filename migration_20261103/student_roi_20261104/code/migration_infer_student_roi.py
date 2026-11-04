"""Teacher-free single RGB -> Qwen pre448 map -> native two-image answer.

Default writes a CPU-audited plan. --run explicitly enables serial CUDA
inference: the visual-only map stage is released before the complete native
Qwen is loaded. No label/category/mask/cache/teacher/ImageBind input is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import migration_dense_infer as dense


PIPELINE_VERSION = "qwen3vl-single-image-student-native-roi-v1"
IMAGE_SIZE = 448
ROI_AREA = 1.0 / 9.0


def atomic_bytes(data, destination, exclusive=True):
    """Publish from a unique sibling temp; never remove another run's temp."""
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(value, destination, exclusive=True):
    atomic_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode(), destination, exclusive)


def audit_inputs(model_path, head_path, image_path):
    """Audit on CPU; return a live CPU head plus JSON-only provenance."""
    image = Path(image_path).resolve(strict=True)
    head = Path(head_path).resolve(strict=True)
    dense.require(image.is_file() and head.is_file(), "Image and head must be files")
    heads, source, size = dense.audit_heads(model_path, {"student": str(head)})
    selected = heads["student"]
    dense.require(selected["features"] == "pre" and size == IMAGE_SIZE, "This inference protocol requires a Qwen pre448 head")
    identity = dense.audit_weight_files(model_path, source)
    metadata = {key: value for key, value in selected.items() if key != "module"}
    return heads, {"image": str(image), "image_sha256": dense.sha256_file(image),
                   "head": metadata, "source_audit": source, "visual_weight_identity": identity,
                   "source_scripts_sha256": {name: dense.sha256_file(Path(__file__).with_name(name))
                       for name in ("migration_dense_infer.py", "migration_visual_probe.py",
                                    "migration_student_roi.py", "migration_roi_eval.py")}}


def infer_student_map(model_path, image_path, image_sha256, heads, device):
    """Only a CPU probability array and JSON metadata escape this scope."""
    import torch
    from transformers import AutoImageProcessor
    from migration_visual_probe import VisualExtractor, load_visual_tower

    visual = extractor = processor = None
    try:
        torch.cuda.synchronize(device)
        load_started = time.perf_counter()
        visual, visual_metadata = load_visual_tower(model_path, device)
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
        extractor = VisualExtractor(visual, processor, device)
        heads["student"]["module"].to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started
        dense.require(dense.sha256_file(image_path) == image_sha256, "Image changed before dense inference")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        maps, geometry = dense.predict_image(image_path, extractor, heads)
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - started
        dense.require(dense.sha256_file(image_path) == image_sha256, "Image changed during dense inference")
        probability = maps["student"]
        return probability, {"geometry": geometry, "visual_model": visual_metadata,
                             "visual_load_seconds": load_seconds, "inference_seconds": inference_seconds}
    finally:
        try:
            if extractor is not None:
                extractor.close()
        finally:
            extractor = visual = processor = None
            heads.clear()
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def infer_native_answer(model_path, image_path, image_sha256, probability, device, max_new_tokens, expected_source):
    """Reuse the fixed QUESTION, ROI rule and official native inference path."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from migration_metrics import parse_generated
    from migration_native_parity import audit_dtypes, source_audit
    from migration_student_roi import map_roi, views_for_roi
    from migration_roi_eval import QUESTION, prepare_processor_inputs, score_and_generate, single_token_id

    dense.require(probability.dtype == np.float32 and probability.shape == (224, 224)
                  and np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1,
                  "Student map must be finite float32 [224,224] probabilities")
    current = source_audit(model_path)
    dense.require(current.get("passed") is True and all(current[key] == expected_source[key]
                  for key in ("source_sha256", "model_config", "processor_config", "api")),
                  "Native model source/config/processor/API changed after dense audit")
    dense.require(dense.sha256_file(image_path) == image_sha256, "Image changed before native inference")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True).eval()
    model.requires_grad_(False)
    dtype_audit = audit_dtypes(model)
    dense.require(dtype_audit.get("passed") is True, "The complete native model must be BF16")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    yes_id, no_id = (single_token_id(processor.tokenizer, text) for text in ("Yes", "No"))
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        bos_token_id=model.generation_config.bos_token_id, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id)
    with torch.inference_mode():
        started = time.perf_counter()
        spec = map_roi(probability, ROI_AREA)
        views, metadata = views_for_roi(image_path, spec, IMAGE_SIZE)
        batch, tokens = prepare_processor_inputs(processor, model, views, device, IMAGE_SIZE)
        dense.require(tokens["image_grid_thw"] == [[1, 28, 28]] * 2
                      and tokens["image_tokens_per_view"] == [196, 196]
                      and tokens["image_tokens_total"] == 392 and tokens["prompt_tokens_total"] == 470
                      and tokens["n_images"] == 2, "Native two-image geometry/token protocol changed")
        tensor_audit = {key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                        for key, value in batch.items() if isinstance(value, torch.Tensor)}
        dense.require(tensor_audit["pixel_values"]["dtype"] == "torch.bfloat16", "Native pixels must be BF16")
        prepared = time.perf_counter()
        prediction = score_and_generate(model, processor, batch, yes_id, no_id, cfg, device)
        dense.require(all(math.isfinite(prediction[key]) for key in ("yes_logit", "no_logit", "score_yes_minus_no")), "Native logits must be finite")
        token_ids = prediction["generated_token_ids"]
        dense.require(isinstance(prediction["generated_text"], str) and 1 <= len(token_ids) <= max_new_tokens
                      and all(type(token) is int and token >= 0 for token in token_ids), "Invalid generation output contract")
        del batch, views
    dense.require(dense.sha256_file(image_path) == image_sha256, "Image changed during native inference")
    label = parse_generated(prediction["generated_text"])
    return {**prediction, **metadata, **tokens, "question": QUESTION, "dtype_audit": dtype_audit,
            "native_source_audit": current, "tensor_audit": tensor_audit,
            "score_decision": "Yes" if prediction["score_yes_minus_no"] > 0 else "No",
            "generated_decision": None if label is None else ("Yes" if label else "No"),
            "generated_invalid": label is None, "native_load_seconds": load_seconds,
            "prepare_seconds": prepared - started, "yes_id": yes_id, "no_id": no_id}


def save_probability(probability, map_path, heatmap_path=None):
    """Exclusive atomic probability export; fixed [0,1] heatmap if requested."""
    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    np.save(buffer, probability, allow_pickle=False)
    atomic_bytes(buffer.getvalue(), map_path)
    result = {"probability_npy": str(map_path), "probability_npy_sha256": dense.sha256_file(map_path),
              "array_sha256": hashlib.sha256(probability.tobytes(order="C")).hexdigest(),
              "shape": [224, 224], "dtype": "float32", "min": float(probability.min()),
              "max": float(probability.max()), "probability_scale": [0.0, 1.0]}
    if heatmap_path is not None:
        rgb = np.stack([probability, 1 - np.abs(2 * probability - 1), 1 - probability], axis=-1)
        pixels = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        atomic_bytes(buffer.getvalue(), heatmap_path)
        result.update(heatmap_png=str(heatmap_path), heatmap_png_sha256=dense.sha256_file(heatmap_path),
                      heatmap_scale="fixed probability0 blue to1 red; no per-image normalization")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=dense.DEFAULT_MODEL)
    parser.add_argument("--head", required=True, help="Selected Qwen pre448 GT or KD head checkpoint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--outjson", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--run", action="store_true", help="Enable CUDA; default writes only an audited CPU plan")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    dense.require(1 <= args.max_new_tokens <= 32, "Generation budget must be between 1 and 32")
    out = Path(args.outjson).resolve()
    dense.require(out.suffix.lower() == ".json", "--outjson must have .json suffix")
    map_path, plan_path, status_path = out.with_suffix(".map.npy"), out.with_suffix(".plan.json"), out.with_suffix(".status.json")
    heatmap_path = out.with_suffix(".map.heatmap.png") if args.heatmap else None
    destinations = [out, map_path, plan_path, status_path] + ([heatmap_path] if heatmap_path else [])
    dense.require(len(set(destinations)) == len(destinations), "Output paths conflict")
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Existing inference artifact is preserved: " + str(destination))
    out.parent.mkdir(parents=True, exist_ok=True)
    status = {"pipeline_version": PIPELINE_VERSION, "status": "auditing", "gpu_model_loaded": False,
              "config": vars(args), "teacher_model_used": False, "teacher_cache_read": False,
              "imagebind_used": False, "gt_masks_read": False, "train_fit": False,
              "script_sha256": dense.sha256_file(Path(__file__))}
    atomic_json(status, status_path)
    started = time.perf_counter()
    heads = None
    try:
        heads, provenance = audit_inputs(args.model, args.head, args.image)
        status.update(provenance=provenance, image_size=IMAGE_SIZE, roi_area=ROI_AREA,
                      visual_tokens=392, prompt_tokens=470, status="planned")
        plan = dict(status)
        plan["plan_sha256"] = dense.canonical_sha(plan)
        atomic_json(plan, plan_path)
        if not args.run:
            print(__import__("json").dumps({"status": "planned", "gpu_model_loaded": False, "plan": str(plan_path)}))
            return plan
        import torch

        dense.require(args.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "--run requires BF16-capable CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        dense.verify_weight_files(provenance["visual_weight_identity"])
        dense.require(dense.sha256_file(args.head) == provenance["head"]["sha256"], "Head changed after CPU audit")
        cudnn_keys = ("allow_tf32", "benchmark", "deterministic")
        native_cudnn = {key: getattr(torch.backends.cudnn, key) for key in cudnn_keys}
        dense_cudnn = dict(allow_tf32=False, benchmark=False, deterministic=True)
        status.update(status="running", numerical_protocol={"seed_per_stage": 42, "matmul_allow_tf32": False,
                      "dense_cudnn": dense_cudnn, "native_cudnn_restored": native_cudnn})
        atomic_json(status, status_path, exclusive=False)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        try:
            for key, value in dense_cudnn.items():
                setattr(torch.backends.cudnn, key, value)
            probability, map_audit = infer_student_map(args.model, provenance["image"], provenance["image_sha256"], heads, args.device)
        finally:
            for key, value in native_cudnn.items():
                setattr(torch.backends.cudnn, key, value)
        heads = None
        gc.collect()
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
        status["gpu_model_loaded"] = True
        map_audit["gpu_allocated_bytes_after_dense_release"] = int(torch.cuda.memory_allocated(args.device))
        dense.verify_weight_files(provenance["visual_weight_identity"])
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        native = infer_native_answer(args.model, provenance["image"], provenance["image_sha256"], probability,
                                     args.device, args.max_new_tokens, provenance["source_audit"])
        dense.verify_weight_files(provenance["visual_weight_identity"])
        torch.cuda.synchronize(args.device)
        serialization_started = time.perf_counter()
        artifacts = save_probability(probability, map_path, heatmap_path)
        result = {**status, **native, "status": "complete", "map_audit": map_audit, "artifacts": artifacts,
                  "plan_sha256": plan["plan_sha256"], "generation": {"method": "greedy", "use_cache": True,
                   "max_new_tokens": args.max_new_tokens},
                  "serialization_seconds": time.perf_counter() - serialization_started,
                  "elapsed_seconds": time.perf_counter() - started,
                  "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                  "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
                  "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
                   "precision": "dense visual BF16/head FP32; full native BF16", "attention_backend": "sdpa"},
                  "timing_scope": "elapsed includes CPU audit/weight hashing and sequential dense/native loads; stage timings are separate"}
        result["payload_sha256"] = dense.canonical_sha(result)
        atomic_json(result, out)
        status.update(status="complete", outjson=str(out), outjson_sha256=dense.sha256_file(out),
                      elapsed_seconds=result["elapsed_seconds"])
        print(__import__("json").dumps({"status": "complete", "outjson": str(out),
              "score": native["score_yes_minus_no"], "generated_text": native["generated_text"]}, ensure_ascii=False))
        return result
    except Exception as error:
        status.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      partial_artifacts=[str(path) for path in destinations if path != status_path and path.exists()],
                      elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        if heads is not None:
            heads.clear()
        atomic_json(status, status_path, exclusive=False)


if __name__ == "__main__":
    main()
