"""Validation-only native Qwen3-VL ROI controls using the shared teacher cache.

This module never loads a model on import. The CLI defaults to a CPU plan;
``--run`` explicitly enables BF16 GPU evaluation. Every scoring/generation
call uses the official processor and top-level Qwen3VLForConditionalGeneration
with its complete native MRoPE/DeepStack path.

Pilot: python migration_roi_eval.py --limit 48 --out runs/roi_pilot --run
Full validation: use --limit 0. No test split or model fitting is supported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from migration_metrics import evaluate_predictions, validate_cache
from visa_pipeline import PIPELINE_VERSION as TEACHER_PIPELINE_VERSION


PIPELINE_VERSION = "qwen3vl-native-roi-val-v1"
DEFAULT_MODEL = "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"
CONDITIONS = ("original_only", "original_center_crop", "original_random_crop",
              "original_teacher_roi")
REFERENCE_CONDITION = "original_teacher_roi_reference"
QUESTION = (
    "The first image is the target product. If a second image is supplied, "
    "it is a close-up crop of the same target. If a third image is supplied, "
    "it is a normal reference product of the same category. "
    "Is there any anomaly in the target product? "
    "Answer with Yes or No first, followed by one brief sentence."
)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def stable_seed(seed, path):
    text = str(seed) + "\0" + str(path)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def select_validation_indices(items, limit=48, seed=42):
    """Fixed pilot strata; labels affect subset balance, never ROI inference."""
    if limit < 0:
        raise ValueError("Validation limit must be nonnegative")
    val = [i for i, item in enumerate(items) if item["split"] == "val"]
    if not val:
        raise ValueError("The teacher cache has no validation items")
    if limit == 0 or limit >= len(val):
        return val
    pools = {}
    for index in val:
        item = items[index]
        pools.setdefault((str(item["category"]), int(item["label"])), []).append(index)
    for pool in pools.values():
        pool.sort(key=lambda i: (stable_seed(seed, items[i]["path"]), i))
    selected, offset = [], 0
    while len(selected) < limit:
        for key in sorted(pools):
            if offset < len(pools[key]):
                selected.append(pools[key][offset])
                if len(selected) == limit:
                    return selected
        offset += 1
    return selected


def shifted_square_bbox(center_xy, area_fraction=1.0 / 9.0):
    """A square in normalized raw-image coordinates, translated at borders.

    Width and height are each sqrt(area_fraction) of the original image.
    Thus a non-square source gives a rectangular pixel crop of the same
    relative area. Floating bounds avoid condition-dependent pixel rounding.
    """
    if not math.isfinite(area_fraction) or not 0 < area_fraction <= 1:
        raise ValueError("ROI area fraction must be in (0,1]")
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all() or ((center < 0) | (center > 1)).any():
        raise ValueError("ROI center must be finite normalized x,y coordinates")
    side = math.sqrt(area_fraction)
    left = min(max(float(center[0]) - side / 2, 0.0), 1.0 - side)
    top = min(max(float(center[1]) - side / 2, 0.0), 1.0 - side)
    return [left, top, left + side, top + side]


def roi_spec(kind, heatmap, path, seed=42, area_fraction=1.0 / 9.0):
    if kind == "center":
        center = [0.5, 0.5]
        detail = {}
    elif kind == "random":
        side = math.sqrt(area_fraction)
        if not math.isfinite(side) or not 0 < side <= 1:
            raise ValueError("ROI area fraction must be in (0,1]")
        rng = np.random.default_rng(stable_seed(seed, path))
        left, top = rng.uniform(0.0, 1.0 - side, size=2)
        center = [float(left + side / 2), float(top + side / 2)]
        detail = {"random_seed": stable_seed(seed, path)}
    elif kind == "teacher":
        values = as_numpy(heatmap)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("Teacher ROI requires a finite 2D probability map")
        row, col = np.unravel_index(int(values.argmax()), values.shape)
        center = [(int(col) + 0.5) / values.shape[1], (int(row) + 0.5) / values.shape[0]]
        detail = {"teacher_peak_row_col": [int(row), int(col)],
                  "teacher_peak_value": float(values[row, col]),
                  "teacher_map_hw": list(values.shape), "argmax_tie_rule": "first in row-major order"}
    else:
        raise ValueError("Unknown ROI kind: " + str(kind))
    bbox = shifted_square_bbox(center, area_fraction)
    return {"kind": kind, "requested_center_xy_relative": center,
            "bbox_xyxy_relative": bbox, "area_fraction": float(area_fraction),
            "coordinate_source": "original RGB; teacher square224 coordinates map linearly to raw x/y",
            **detail}


def crop_raw_rgb(image, bbox_xyxy_relative, size=448):
    """Sample fractional raw-RGB bounds directly, then output a fixed square."""
    left, top, right, bottom = bbox_xyxy_relative
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Crop bounds must be ordered and contained in the original RGB")
    width, height = image.size
    pixel_bbox = [left * width, top * height, right * width, bottom * height]
    crop = image.transform((size, size), Image.Transform.EXTENT, pixel_bbox,
                           resample=Image.Resampling.BICUBIC)
    return crop, pixel_bbox


def build_reference_bank(items, cls):
    """Only known-normal training images are eligible reference sources."""
    values = as_numpy(cls).astype(np.float32, copy=False)
    result = {}
    for category in sorted({str(item["category"]) for item in items}):
        indices = [i for i, item in enumerate(items)
                   if item["split"] == "train" and item["label"] == 0
                   and str(item["category"]) == category]
        if not indices:
            continue
        features = values[indices]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Normal reference CLS vectors must have finite nonzero norms")
        result[category] = (indices, features / norms)
    return result


def select_reference(query_index, items, cls, bank):
    query = items[query_index]
    if query["split"] != "val":
        raise ValueError("ROI evaluation queries must come from validation")
    category = str(query["category"])
    if category not in bank:
        raise ValueError("No normal training reference for category " + category)
    vector = as_numpy(cls[query_index]).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Query CLS must have a finite nonzero norm")
    indices, features = bank[category]
    cosine = features @ (vector / norm)
    position = int(cosine.argmax())
    index = indices[position]
    source = items[index]
    if not (source["split"] == "train" and source["label"] == 0
            and source["category"] == query["category"] and source["path"] != query["path"]):
        raise AssertionError("Reference eligibility failed")
    return {"cache_index": int(index), "path": source["path"], "category": source["category"],
            "split": "train", "source": "normal training RGB",
            "selection": "same-category nearest cached ImageBind CLS cosine; first cache index on ties",
            "cosine_similarity": float(cosine[position])}


def load_shared_cache(path):
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    audit = validate_cache(cache, expected_pipeline_version=TEACHER_PIPELINE_VERSION)
    if cache.get("geometry") != "square224":
        raise ValueError("ROI coordinates require the shared direct square224 teacher geometry")
    sidecar = json.loads(Path(path).with_suffix(".json").read_text())
    digest = sha256_file(path)
    if digest != sidecar.get("cache_sha256"):
        raise ValueError("Shared cache file SHA256 does not match its sidecar")
    manifest_sha = hashlib.sha256(json.dumps(cache["items"], sort_keys=True).encode()).hexdigest()
    for source in (cache, sidecar):
        if source.get("manifest_sha256") != manifest_sha:
            raise ValueError("Shared cache item manifest SHA256 mismatch")
        for key in ("pipeline_version", "decoder_sha256"):
            if source.get(key) != cache[key]:
                raise ValueError("Shared cache sidecar mismatch for " + key)
    return cache, {**audit, "cache_sha256": digest, "manifest_sha256": manifest_sha,
                   "geometry": cache["geometry"], "cache_batch_size": cache.get("cache_batch_size"),
                   "singleton_checks": cache.get("singleton_checks")}


def build_views(item, heatmap, condition, seed, area_fraction, size, reference=None):
    with Image.open(item["path"]) as source:
        raw = source.convert("RGB")
    views = [raw.resize((size, size), Image.Resampling.BICUBIC)]
    metadata = {"target_path": item["path"], "source_image_wh": list(raw.size),
                "view_sources": [item["path"]], "view_roles": ["target_original"],
                "processed_image_wh": [size, size], "roi": None, "reference": None}
    if condition != "original_only":
        kind = {"original_center_crop": "center", "original_random_crop": "random",
                "original_teacher_roi": "teacher", REFERENCE_CONDITION: "teacher"}[condition]
        spec = roi_spec(kind, heatmap, item["path"], seed, area_fraction)
        crop, pixel_bbox = crop_raw_rgb(raw, spec["bbox_xyxy_relative"], size)
        spec["bbox_xyxy_raw_pixels_float"] = pixel_bbox
        metadata["roi"] = spec
        views.append(crop)
        metadata["view_sources"].append(item["path"])
        metadata["view_roles"].append("target_roi")
    if condition == REFERENCE_CONDITION:
        if reference is None:
            raise ValueError("Reference condition requires a normal-training reference")
        with Image.open(reference["path"]) as source:
            normal = source.convert("RGB")
        views.append(normal.resize((size, size), Image.Resampling.BICUBIC))
        metadata["reference"] = {**reference, "source_image_wh": list(normal.size)}
        metadata["view_sources"].append(reference["path"])
        metadata["view_roles"].append("normal_training_reference")
    if not 1 <= len(views) <= 3 or any(view.size != (size, size) for view in views):
        raise AssertionError("Native ROI conditions require one to three equal-resolution images")
    return views, metadata


def prepare_processor_inputs(processor, model, views, device, size):
    import torch

    content = [{"type": "image"} for _ in views] + [{"type": "text", "text": QUESTION}]
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    # The already-square images are divisible by patch_size * merge_size.
    # Use the public processor option so no dynamic resizing changes a control.
    encoded = processor(text=[text], images=views, return_tensors="pt",
                        images_kwargs={"do_resize": False})
    grids = encoded["image_grid_thw"].tolist()
    image_processor = processor.image_processor
    patch = int(image_processor.patch_size)
    merge = int(image_processor.merge_size)
    if len(grids) != len(views):
        raise ValueError("Processor image count changed")
    per_image_tokens = []
    for temporal, height, width in grids:
        if temporal != 1 or height * patch != size or width * patch != size:
            raise ValueError("Processor changed the requested fixed square image resolution: " + str(grids))
        if temporal * height * width % (merge * merge):
            raise ValueError("Image grid is not divisible by the spatial merge size")
        per_image_tokens.append(temporal * height * width // (merge * merge))
    image_tokens = int((encoded["input_ids"] == model.config.image_token_id).sum())
    if image_tokens != sum(per_image_tokens) or len(set(per_image_tokens)) != 1:
        raise ValueError("Image-placeholder count or per-image visual-token equality failed")
    if not bool((encoded["attention_mask"] == 1).all()):
        raise ValueError("ROI evaluation intentionally uses one unpadded query at a time")
    info = {"image_grid_thw": grids, "image_tokens_per_view": per_image_tokens,
            "image_tokens_total": image_tokens, "prompt_tokens_total": int(encoded["input_ids"].shape[1]),
            "patch_size": patch, "spatial_merge_size": merge, "n_images": len(views)}
    batch = {key: value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
             if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
    return batch, info


def single_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("First-answer-token scoring requires a single tokenizer token for " + text)
    return int(ids[0])


def score_and_generate(model, processor, batch, yes_id, no_id, generation_config, device):
    import torch

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.model.rope_deltas = None
    output = model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
    logits = output.logits[0, -1].float()
    yes, no = float(logits[yes_id]), float(logits[no_id])
    if not math.isfinite(yes) or not math.isfinite(no):
        raise ValueError("Nonfinite native Yes/No logits")
    del output, logits
    torch.cuda.synchronize(device)
    scored = time.perf_counter()
    model.model.rope_deltas = None
    generated = model.generate(**batch, generation_config=generation_config, use_model_defaults=False)
    suffix = generated[0, batch["input_ids"].shape[1]:].detach().cpu().tolist()
    text = processor.tokenizer.decode(suffix, skip_special_tokens=True)
    del generated
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    return {"score_yes_minus_no": yes - no, "yes_logit": yes, "no_logit": no,
            "generated_text": text, "generated_token_ids": suffix,
            "score_seconds": scored - start, "generate_seconds": finished - scored}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", default="runs/migration_20261103/teacher_cache.pt")
    parser.add_argument("--out", default="runs/migration_20261103/roi_eval")
    parser.add_argument("--run", action="store_true", help="Explicitly enable GPU evaluation; default is a CPU plan")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=48, help="48-image pilot by default; 0 evaluates all validation images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-area", type=float, default=1.0 / 9.0)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--condition", action="append", choices=CONDITIONS, help="Repeat to restrict the four default controls")
    parser.add_argument("--with-reference", action="store_true", help="Also evaluate target + teacher ROI + normal training reference")
    parser.add_argument("--parity-report", help="Optional preceding native parity report; when supplied it must have passed")
    return parser


def main():
    args = make_parser().parse_args()
    if args.limit < 0 or args.image_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Limit must be nonnegative; image size and generation budget must be positive")
    shifted_square_bbox([0.5, 0.5], args.roi_area)
    conditions = list(dict.fromkeys(args.condition or CONDITIONS))
    if args.with_reference:
        conditions.append(REFERENCE_CONDITION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() or any(out.glob("*_predictions.jsonl")):
        raise FileExistsError("Existing ROI evaluation is preserved; choose a fresh output directory")
    cache, cache_audit = load_shared_cache(args.cache)
    items = cache["items"]
    indices = select_validation_indices(items, args.limit, args.seed)
    selected = [items[index] for index in indices]
    bank = build_reference_bank(items, cache["cls"]) if args.with_reference else {}
    references = {i: select_reference(i, items, cache["cls"], bank) for i in indices} if args.with_reference else {}
    parity = None
    if args.parity_report:
        parity = json.loads(Path(args.parity_report).read_text())
        if parity.get("passed") is not True or parity.get("all_cases_exercised_cached_decoding") is not True:
            raise ValueError("The supplied native parity report did not pass actual cached decoding")
    plan = {"pipeline_version": PIPELINE_VERSION, "split": "val", "config": vars(args),
            "conditions": conditions, "n": len(indices), "full_validation_n": sum(x["split"] == "val" for x in items),
            "subset_selection": "fixed category/label round robin; labels used only for pilot balance and metrics",
            "selected_cache_indices": indices, "selected_paths": [x["path"] for x in selected],
            "selected_manifest_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
            "cache_audit": cache_audit, "question": QUESTION,
            "score_definition": "unprocessed first-answer-token logit(Yes) - logit(No); threshold 0",
            "roi_definition": "one normalized square, fixed area, translated at raw-image borders; no GT masks",
            "reference_definition": "same-category nearest cached CLS among normal training images only",
            "token_budget_comparison": "crop controls have equal image count/ROI area/per-view resolution; original_only has fewer visual tokens; reference adds a third image",
            "parity_report": None if parity is None else {"path": args.parity_report, "sha256": sha256_file(args.parity_report)},
            "gpu_model_loaded": False, "status": "planned"}
    atomic_json(plan, out / "plan.json")
    if not args.run:
        print(json.dumps({"status": "planned", "gpu_model_loaded": False, "n": len(indices),
                          "conditions": conditions, "plan": str(out / "plan.json")}))
        return

    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration
    from migration_native_parity import source_audit, audit_dtypes

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--run requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native ROI evaluation requires BF16 support")
    audit = source_audit(args.model)
    if not audit["passed"]:
        raise RuntimeError("Installed native Qwen3-VL source failed the audited SHA/version gate")
    if parity is not None and parity["source_audit"]["source_sha256"] != audit["source_sha256"]:
        raise RuntimeError("Native parity report and current modeling source SHA differ")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True).eval()
    model.requires_grad_(False)
    dtype_audit = audit_dtypes(model)
    if not dtype_audit["passed"]:
        raise RuntimeError("Full native model, including vision/mergers/DeepStack, must be BF16")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    factor = int(processor.image_processor.patch_size) * int(processor.image_processor.merge_size)
    if args.image_size % factor:
        raise ValueError("Image size must be divisible by patch_size * merge_size")
    yes_id, no_id = (single_token_id(processor.tokenizer, value) for value in ("Yes", "No"))
    cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
                           bos_token_id=model.generation_config.bos_token_id,
                           eos_token_id=model.generation_config.eos_token_id,
                           pad_token_id=model.generation_config.pad_token_id)
    run = {**plan, "status": "running", "gpu_model_loaded": True, "source_audit": audit,
           "dtype_audit": dtype_audit, "runtime": {"torch_version": torch.__version__,
           "gpu": torch.cuda.get_device_name(args.device), "precision": "full model BF16; no quantization",
           "attention_backend": "sdpa", "yes_id": yes_id, "no_id": no_id}, "results": {}}
    atomic_json(run, out / "run.json")
    try:
        with torch.inference_mode():
            for condition in conditions:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                baseline = torch.cuda.memory_allocated(args.device)
                started = time.perf_counter()
                records, scores, texts = [], [], []
                with open(out / (condition + "_predictions.jsonl"), "x") as handle:
                    for position, index in enumerate(indices):
                        began = time.perf_counter()
                        views, metadata = build_views(items[index], cache["maps"][index], condition,
                                                      args.seed, args.roi_area, args.image_size,
                                                      references.get(index))
                        batch, tokens = prepare_processor_inputs(processor, model, views, args.device, args.image_size)
                        prepared = time.perf_counter()
                        prediction = score_and_generate(model, processor, batch, yes_id, no_id, cfg, args.device)
                        record = {**items[index], "cache_index": index, "condition": condition,
                                  **metadata, **tokens, **prediction,
                                  "prepare_seconds": prepared - began, "elapsed_seconds": time.perf_counter() - began}
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                        records.append(record)
                        scores.append(prediction["score_yes_minus_no"])
                        texts.append(prediction["generated_text"])
                        del batch, views
                        if (position + 1) % 12 == 0 or position + 1 == len(indices):
                            print(json.dumps({"condition": condition, "done": position + 1,
                                              "n": len(indices), "elapsed_seconds": time.perf_counter() - started}), flush=True)
                torch.cuda.synchronize(args.device)
                summary = {"metrics": evaluate_predictions(selected, scores=scores, generated_texts=texts),
                           "n": len(indices), "elapsed_seconds": time.perf_counter() - started,
                           "score_seconds_total": sum(x["score_seconds"] for x in records),
                           "generate_seconds_total": sum(x["generate_seconds"] for x in records),
                           "baseline_gpu_allocated_bytes": baseline,
                           "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                           "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                           "actual_image_grids": sorted({tuple(grid) for x in records for grid in x["image_grid_thw"]}),
                           "visual_tokens_total_min_max": [min(x["image_tokens_total"] for x in records), max(x["image_tokens_total"] for x in records)],
                           "prompt_tokens_total_min_max": [min(x["prompt_tokens_total"] for x in records), max(x["prompt_tokens_total"] for x in records)]}
                atomic_json(summary, out / (condition + "_metrics.json"))
                run["results"][condition] = summary
                atomic_json(run, out / "run.json")
        run["status"] = "complete"
    except Exception as error:
        run["status"] = "failed"
        run["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_json(run, out / "run.json")
    print(json.dumps({"status": "complete", "split": "val", "n": len(indices), "run": str(out / "run.json")}))


if __name__ == "__main__":
    main()
