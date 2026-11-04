"""Export Qwen pre/post SpatialHead probabilities; default is a CPU plan.

Batch inputs are {"items":[{"path":absolute,"category":str,"split":"val",...}]}.
Labels are retained for downstream metrics and never passed into predict_image.
Repeat --head NAME=CKPT for heads sharing one image resolution and visual tower.
No teacher, ImageBind, GT masks, fitting, language model, or test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time


SCHEMA_VERSION = "qwen-dense-map-export-v1"
PIPELINE_VERSION = "qwen3vl-dense-infer-v1"
DEFAULT_MODEL = "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_sha256(state):
    """Canonical tensor identity, independent of checkpoint serialization."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_bytes(data, destination, exclusive=True):
    """Publish in the destination directory, without replacing prior exports."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value, destination, exclusive=True):
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()
    atomic_bytes(data, destination, exclusive)


def load_items(manifest=None, image=None, category=None):
    """Validate val-only batches or a label-free standalone RGB query."""
    if manifest is not None:
        require(image is None and category is None, "Use --manifest or --image/--category, not both")
        path = Path(manifest).resolve(strict=True)
        before = sha256_file(path)
        value = json.loads(path.read_text())
        require(before == sha256_file(path), "Input manifest changed while being read")
        items = value.get("items") if isinstance(value, dict) else value
        source = {"path": str(path), "sha256": before}
        if isinstance(value, dict) and "selected_manifest_sha256" in value:
            source["selected_manifest_sha256"] = value["selected_manifest_sha256"]
    else:
        require(image is not None and isinstance(category, str) and category, "Provide --manifest or --image and --category")
        items = [{"path": str(Path(image).resolve(strict=True)), "category": category, "split": "inference"}]
        source = None
    require(isinstance(items, list) and bool(items), "Manifest items must be a nonempty list")
    seen, resolved = set(), set()
    for item in items:
        require(isinstance(item, dict), "Each item must be an object")
        path, category_name = item.get("path"), item.get("category")
        require(isinstance(path, str) and Path(path).is_absolute(), "Image paths must be absolute")
        require(isinstance(category_name, str) and bool(category_name), "Every item requires a category")
        require(item.get("split") == ("val" if manifest is not None else "inference"), "Batch manifests accept val only; test/train rejected")
        if "label" in item:
            require(type(item["label"]) is int and item["label"] in (0, 1), "Labels, when present, must be integer 0/1")
        actual = Path(path).resolve(strict=True)
        require(actual.is_file(), "Image is not a file: " + path)
        require(path not in seen and str(actual) not in resolved, "Image paths and resolved paths must be unique")
        seen.add(path)
        resolved.add(str(actual))
    return items, source


def parse_head_specs(specifications):
    result = {}
    for specification in specifications:
        name, separator, filename = specification.partition("=")
        require(separator and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name), "Use --head SAFE_NAME=CHECKPOINT")
        require(name not in result and bool(filename), "Head names must be unique and checkpoints nonempty")
        path = Path(filename).resolve(strict=True)
        require(path.is_file(), "Head checkpoint is not a file")
        result[name] = str(path)
    require(bool(result), "At least one --head NAME=CKPT is required")
    return result


def audit_heads(model_path, specifications):
    """CPU-only source audit and strict head restoration, before CUDA access."""
    import torch
    from migration_dense_train import SpatialHead, feature_configuration, load_head_checkpoint
    from migration_visual_probe import visual_source_audit

    source = visual_source_audit(model_path)
    require(source.get("passed") is True, "Official Qwen/processor source audit failed")
    vision = source["model_config"]["vision_config"]
    factor = int(vision["patch_size"]) * int(vision["spatial_merge_size"])
    heads, size = {}, None
    for name, path in specifications.items():
        digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        compatibility = checkpoint.get("compatibility", {})
        feature, resolution = compatibility.get("features"), compatibility.get("size")
        require(feature in ("pre", "post"), "Only Qwen pre/post checkpoints are supported")
        require(type(resolution) is int and resolution > 0 and resolution % factor == 0, "Invalid explicit checkpoint resolution")
        require(size is None or size == resolution, "All heads must share the same image resolution")
        size = resolution
        config, expected = feature_configuration(feature, resolution, source)
        head = SpatialHead(config["in_channels"], width=config["width"]).float()
        loaded = load_head_checkpoint(head, path, expected)
        require(sha256_file(path) == digest, "Head file changed while being loaded")
        require(all(parameter.dtype == torch.float32 and bool(torch.isfinite(parameter).all())
                    for parameter in head.parameters()), "Head parameters must be finite FP32")
        step = loaded.get("step")
        require(type(step) is int and step > 0, "Head checkpoint requires a positive selected step")
        weight = loaded.get("config", {}).get("distill_weight")
        require(isinstance(weight, (int, float)) and not isinstance(weight, bool)
                and math.isfinite(weight) and weight >= 0, "Invalid checkpoint distillation identity")
        head.eval().requires_grad_(False)
        heads[name] = {"module": head, "checkpoint": path, "sha256": digest, "features": feature,
                       "size": resolution, "compatibility": expected, "step": step, "distill_weight": weight,
                       "matched_run_key": loaded.get("matched_run_key"),
                       "state_sha256": state_sha256(head.state_dict())}
    return heads, source, size


def audit_weight_files(model_path, source):
    """Record the actual shards used by the visual-only loader, on CPU."""
    root = Path(model_path).resolve(strict=True)
    index = root / "model.safetensors.index.json"
    require(sha256_file(index) == source["visual_checkpoint"]["index_sha256"], "Qwen index changed after source audit")
    shards = []
    for filename in source["visual_checkpoint"]["shards"]:
        path = (root / filename).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), "Visual shard must lie in the model directory")
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Visual shard changed during hashing")
        shards.append({"path": str(path), "sha256": digest, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
    require(bool(shards), "No visual checkpoint shards")
    return {"model": str(root), "index_sha256": sha256_file(index), "shards": shards,
            "scope": "current visual loader shard files; historical head compatibility does not bind full shard hashes"}


def verify_weight_files(identity):
    require(sha256_file(Path(identity["model"]) / "model.safetensors.index.json") == identity["index_sha256"], "Model index changed")
    for record in identity["shards"]:
        current = Path(record["path"]).stat()
        require(current.st_size == record["bytes"] and current.st_mtime_ns == record["mtime_ns"], "Model shard changed after CPU audit")


def predict_image(image_path, extractor, heads):
    """Return {name: FP32 map224}, geometry; no label/category/GT inputs."""
    import numpy as np
    import torch

    sizes = {head["size"] for head in heads.values()}
    require(len(sizes) == 1, "Shared extraction requires one common resolution")
    size = next(iter(sizes))
    with torch.inference_mode():
        features, geometry = extractor(str(image_path), size)
        require(geometry.get("requested_image_hw") == [size, size]
                and geometry.get("actual_image_hw") == [size, size]
                and geometry.get("processor_do_resize") is False
                and all(geometry.get("geometry_policy") == head["compatibility"]["geometry_policy"]
                        for head in heads.values()), "Extractor explicit geometry failed")
        maps = {}
        for name, record in heads.items():
            feature = features[record["features"]]
            require(feature.ndim == 3 and feature.dtype == torch.bfloat16
                    and bool(torch.isfinite(feature).all()), "Expected finite BF16 Qwen spatial features")
            logits = record["module"](feature.unsqueeze(0))
            require(tuple(logits.shape) == (1, 1, 224, 224) and logits.dtype == torch.float32
                    and bool(torch.isfinite(logits).all()), "Head output must be finite FP32 [1,1,224,224]")
            probability = np.array(logits.sigmoid()[0, 0].cpu().numpy(), dtype=np.float32, copy=True, order="C")
            require(np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1, "Invalid probability map")
            maps[name] = probability
        return maps, geometry


def export_maps(items, heads, extractor, output, input_hashes, source_sha256, device=None):
    """Reuse one visual extraction per path; serialize label-free predictions."""
    import numpy as np
    import torch

    records = {name: {} for name in heads}
    for index, item in enumerate(items, 1):
        path = item["path"]
        require(sha256_file(path) == input_hashes[path], "Input image changed before prediction")
        if device is not None:
            torch.cuda.synchronize(device)
        prediction_started = time.perf_counter()
        maps, geometry = predict_image(path, extractor, heads)
        if device is not None:
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - prediction_started
        require(sha256_file(path) == input_hashes[path], "Input image changed during prediction")
        require(set(maps) == set(heads), "Prediction names do not cover every head")
        for name, probability in maps.items():
            serialization_started = time.perf_counter()
            relative = "maps/" + name + "/" + hashlib.sha256(path.encode()).hexdigest() + ".npy"
            buffer = io.BytesIO()
            np.save(buffer, probability, allow_pickle=False)
            destination = output / relative
            atomic_bytes(buffer.getvalue(), destination)
            records[name][path] = {"path": relative, "sha256": sha256_file(destination),
                "array_sha256": hashlib.sha256(probability.tobytes(order="C")).hexdigest(),
                "bytes": destination.stat().st_size, "shape": [224, 224], "dtype": "float32",
                "min": float(probability.min()), "max": float(probability.max()),
                "input_sha256": input_hashes[path], "head_sha256": heads[name]["sha256"],
                "source_sha256": source_sha256, "geometry": geometry,
                "inference_seconds": inference_seconds,
                "serialization_seconds": time.perf_counter() - serialization_started,
                "inference_timing_scope": "shared RGB preparation, one visual extraction, all heads, CPU map transfer; repeated per head, count once per image"}
        if index == 1 or index % 12 == 0 or index == len(items):
            print(json.dumps({"images_done": index, "n_images": len(items), "heads": list(heads)}), flush=True)
    return records


def verify_exports(output, manifest):
    """Read back the complete export and reject misalignment or tampering."""
    import numpy as np

    root = Path(output).resolve()
    paths = {item["path"] for item in manifest["items"]}
    for name, head in manifest["heads"].items():
        require(set(head["maps"]) == paths, "Head map path coverage differs from items: " + name)
        for path, record in head["maps"].items():
            relative = Path(record["path"])
            destination = (root / relative).resolve(strict=True)
            require(not relative.is_absolute() and destination.is_relative_to(root), "Map must be relative and stay inside output")
            require(destination.stat().st_size == record["bytes"] and sha256_file(destination) == record["sha256"], "Map file identity mismatch")
            value = np.load(destination, allow_pickle=False)
            require(value.dtype == np.float32 and value.shape == (224, 224) and value.flags.c_contiguous
                    and np.isfinite(value).all() and 0 <= value.min() <= value.max() <= 1, "Map array contract mismatch")
            require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == record["array_sha256"]
                    and float(value.min()) == record["min"] and float(value.max()) == record["max"]
                    and record["head_sha256"] == head["sha256"], "Map array/head identity mismatch")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--image")
    parser.add_argument("--category")
    parser.add_argument("--head", action="append", required=True, help="Repeat NAME=CHECKPOINT; feature and size come from checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
           "status": "auditing", "config": vars(args), "gpu_model_loaded": False,
           "no_teacher": True, "no_imagebind": True, "no_gt_masks": True,
           "test_set_used": False, "train_fit": False, "language_model_instantiated": False,
           "script_sha256": sha256_file(Path(__file__))}
    atomic_json(run, output / "run.json")
    extractor = None
    try:
        items, source_manifest = load_items(args.manifest, args.image, args.category)
        specifications = parse_head_specs(args.head)
        heads, audit, size = audit_heads(args.model, specifications)
        identity = audit_weight_files(args.model, audit)
        input_hashes = {item["path"]: sha256_file(item["path"]) for item in items}
        geometry_policy = next(iter(heads.values()))["compatibility"]["geometry_policy"]
        metadata_heads = {name: {key: value for key, value in record.items() if key != "module"}
                          for name, record in heads.items()}
        run.update(status="planned", items=items, n_images=len(items), items_sha256=canonical_sha(items),
                   source_manifest=source_manifest, source_audit=audit, visual_weight_identity=identity,
                   input_sha256=input_hashes, heads=metadata_heads,
                   geometry={"coordinate_system": "direct-square-224", "map_hw": [224, 224],
                             "requested_image_hw": [size, size], "processor_do_resize": False,
                             "geometry_policy": geometry_policy})
        plan = dict(run)
        plan["plan_sha256"] = canonical_sha(plan)
        run["plan_sha256"] = plan["plan_sha256"]
        atomic_json(plan, output / "plan.json")
        atomic_json(run, output / "run.json", exclusive=False)
        if not args.run:
            print(json.dumps({"status": "planned", "gpu_model_loaded": False, "n_images": len(items),
                              "heads": list(heads), "plan": str(output / "plan.json")}))
            return run
        import torch
        from transformers import AutoImageProcessor
        from migration_visual_probe import VisualExtractor, load_visual_tower

        require(args.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "--run requires BF16-capable CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        run["numerical_protocol"] = {"seed": 42, "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                     "cudnn_benchmark": False, "cudnn_deterministic": True}
        verify_weight_files(identity)
        for record in heads.values():
            require(sha256_file(record["checkpoint"]) == record["sha256"], "Head changed after CPU audit")
        run["status"] = "running"
        atomic_json(run, output / "run.json", exclusive=False)
        torch.cuda.synchronize(args.device)
        visual_load_started = time.perf_counter()
        visual, visual_meta = load_visual_tower(args.model, args.device)
        torch.cuda.synchronize(args.device)
        run["visual_load_seconds"] = time.perf_counter() - visual_load_started
        run["gpu_model_loaded"] = True
        processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True, use_fast=True)
        extractor = VisualExtractor(visual, processor, args.device)
        for record in heads.values():
            record["module"].to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        export_started = time.perf_counter()
        records = export_maps(items, heads, extractor, output, input_hashes, audit["source_sha256"], args.device)
        torch.cuda.synchronize(args.device)
        run["export_map_seconds"] = time.perf_counter() - export_started
        first_records = records[next(iter(heads))]
        run["inference_seconds_total"] = sum(record["inference_seconds"] for record in first_records.values())
        run["serialization_seconds_total"] = sum(record["serialization_seconds"]
                                                   for head_records in records.values() for record in head_records.values())
        run["timing_scope"] = {"elapsed_seconds": "entire job including CPU audit, hashing, loading and export",
                                "visual_load_seconds": "visual-only weight construction/load and CUDA transfer",
                                "export_map_seconds": "all input checks, synchronized prediction and serialization",
                                "inference_seconds_total": "count shared prediction once per image; includes RGB processor and CPU map transfer"}
        verify_weight_files(identity)
        if source_manifest is not None:
            require(sha256_file(source_manifest["path"]) == source_manifest["sha256"], "Source manifest changed")
        manifest = {**run, "status": "complete", "visual_model": visual_meta,
                    "heads": {name: {**metadata_heads[name], "maps": records[name]} for name in heads},
                    "feature_extractions": len(items), "shared_visual_features_across_heads": True}
        verify_exports(output, manifest)
        torch.cuda.synchronize(args.device)
        manifest["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(args.device))
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["payload_sha256"] = canonical_sha(manifest)
        atomic_json(manifest, output / "manifest.json")
        run.update(status="complete", manifest="manifest.json", manifest_sha256=sha256_file(output / "manifest.json"),
                   payload_sha256=manifest["payload_sha256"], feature_extractions=len(items),
                   elapsed_seconds=manifest["elapsed_seconds"], peak_gpu_allocated_bytes=manifest["peak_gpu_allocated_bytes"])
        print(json.dumps({"status": "complete", "n_images": len(items), "heads": list(heads),
                          "manifest": str(output / "manifest.json")}), flush=True)
        return manifest
    except Exception as error:
        run.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                   elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        try:
            if extractor is not None:
                extractor.close()
        finally:
            atomic_json(run, output / "run.json", exclusive=False)


if __name__ == "__main__":
    main()
