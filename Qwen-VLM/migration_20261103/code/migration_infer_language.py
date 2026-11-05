"""Single-image inference for a selected stage-one frozen-language bridge.

No label, mask, split file or teacher cache is needed. Recompute ImageBind
CLS and the repaired BigDec probability map for this image, release both
teacher modules, then load the frozen BF16 language backend and the trained
PromptLearner/CLS projection from the selected stage-one checkpoint.

  python migration_infer_language.py --backend qwen \
      --ckpt runs/migration_20261103/lm_qwen_s42/best.pt \
      --image /path/to/image.JPG --category macaroni1 \
      --outjson runs/demo/qwen.json --heatmap

For Vicuna, --model or VICUNA_BF16_MODEL may select the converted checkpoint;
otherwise the existing /dev/shm/migration_vicuna_bf16 is preferred. This CLI
requires the converted safetensors files, config and tokenizer.model.

The singleton FP16 teacher uses the same square224 pipeline as training.
Different singleton/batched kernels may cause small differences from the
cached validation scores. The map is a raw float32 probability .npy array;
an optional heatmap uses a fixed [0,1] scale without per-image normalization.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from migration_language import DESCRIPTIONS, FrozenLanguageBridge, QWEN_PATH, VICUNA_PATH
from migration_metrics import parse_generated
from visa_pipeline import PIPELINE_VERSION, load_image_tensor


REPO = Path(os.environ.get("ANOMALYGPT_REPO", "/root/autodl-tmp/AnomalyGPT"))
DEMO_VERSION = "imagebind-bigdec-frozen-language-single-image-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def validate_bridge_checkpoint(checkpoint, backend):
    if checkpoint.get("backend") != backend:
        raise ValueError("The bridge checkpoint backend does not match --backend")
    if checkpoint.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("The bridge checkpoint uses a different image/map pipeline")
    if checkpoint.get("training_protocol") != "supervised_custom_60_20_20":
        raise ValueError("Expected a stage-one supervised_custom_60_20_20 checkpoint")
    if not isinstance(checkpoint.get("decoder_sha256"), str) or len(checkpoint["decoder_sha256"]) != 64:
        raise ValueError("The bridge checkpoint must identify its repaired BigDec teacher by SHA256")
    for name in ("prompt_learner", "cls_proj"):
        state = checkpoint.get(name)
        if not isinstance(state, dict) or not state:
            raise ValueError("Missing trained bridge state: " + name)
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError("Bridge state must contain finite tensors: %s.%s" % (name, key))
    if checkpoint["prompt_learner"].get("base_prompts", torch.empty(0)).shape != (9, 4096):
        raise ValueError("Expected original PromptLearner base_prompts [9,4096]")
    if (checkpoint["cls_proj"].get("weight", torch.empty(0)).shape != (4096, 1024)
            or checkpoint["cls_proj"].get("bias", torch.empty(0)).shape != (4096,)):
        raise ValueError("Expected the trained Linear(1024,4096) CLS projection")
    return checkpoint


def resolve_decoder(explicit, checkpoint):
    if explicit:
        return Path(explicit)
    cache_path = checkpoint.get("config", {}).get("cache")
    if cache_path:
        sidecar_path = Path(cache_path).with_suffix(".json")
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text())
            if (sidecar.get("decoder_sha256") == checkpoint["decoder_sha256"]
                    and sidecar.get("cache_sha256") == checkpoint.get("cache_sha256")
                    and sidecar.get("decoder")):
                return Path(sidecar["decoder"])
    return REPO / "runs/dec_big_fixed_s42/best.pt"


def _check_model_directory(path, backend):
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError("Language model config is missing: " + str(path))
    tokenizer = "tokenizer.model" if backend == "vicuna" else "tokenizer.json"
    if not (path / tokenizer).is_file():
        raise FileNotFoundError("Language tokenizer is missing: " + str(path / tokenizer))
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = [str(path / shard) for shard in sorted(shards) if not (path / shard).is_file()]
        if not shards or missing:
            raise FileNotFoundError("Language safetensors shards are missing: " + str(missing))
    elif not (path / "model.safetensors").is_file():
        raise FileNotFoundError("Use the complete safetensors language checkpoint: " + str(path))
    return path


def resolve_language_model(backend, explicit=None, checkpoint=None):
    if explicit:
        return _check_model_directory(explicit, backend)
    variable = "VICUNA_BF16_MODEL" if backend == "vicuna" else "QWEN3VL_MODEL"
    if os.environ.get(variable):
        return _check_model_directory(os.environ[variable], backend)
    candidates = []
    if backend == "vicuna":
        candidates.append(Path("/dev/shm/migration_vicuna_bf16"))
    saved_path = (checkpoint or {}).get("initialization_audit", {}).get("model_path")
    if saved_path:
        candidates.append(Path(saved_path))
    candidates.append(VICUNA_PATH if backend == "vicuna" else QWEN_PATH)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir():
            return _check_model_directory(candidate, backend)
    raise FileNotFoundError("No local language checkpoint found; pass --model or set " + variable)


def validate_decoder(path, expected_sha256):
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError("Actual BigDec SHA256 differs from the selected bridge's teacher")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("pipeline_version") != PIPELINE_VERSION or not isinstance(checkpoint.get("dec"), dict):
        raise ValueError("Expected a repaired BigDec checkpoint from the same pipeline")
    return checkpoint, digest


def teacher_features(image_path, imagebind_path, decoder_checkpoint, device):
    """Only CPU FP32 CLS/map escape this function; teacher GPU references do not."""
    model_directory = str(REPO / "code/code/model")
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)
    from ImageBind import imagebind_model
    from ImageBind.models.imagebind_model import ModalityType
    from dec_big import BigDec, patch_features

    started = time.perf_counter()
    encoder = decoder = pixels = encoded = patches = None
    try:
        encoder, _ = imagebind_model.imagebind_huge({"layers": [7, 15, 23, 31]})
        weights = torch.load(imagebind_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(weights, strict=True)
        del weights
        encoder.requires_grad_(False).half().to(device).eval()
        decoder = BigDec().float()
        decoder.load_state_dict(decoder_checkpoint["dec"], strict=True)
        decoder.requires_grad_(False).to(device).eval()
        with torch.inference_mode():
            pixels = load_image_tensor(image_path, device=device, size=224).half()
            encoded = encoder({ModalityType.VISION: pixels})["vision"]
            patches = patch_features(encoded[1], 1)
            cls = encoded[0].float().cpu()
            amap = decoder(patches).sigmoid()[0, 0].float().cpu()
        # Clone outside inference_mode so consumers also receive ordinary tensors.
        cls, amap = cls.clone(), amap.clone()
        if cls.shape != (1, 1024) or amap.shape != (224, 224):
            raise ValueError("Teacher must produce CLS[1,1024] and map[224,224]")
        if (not bool(torch.isfinite(cls).all()) or not bool(torch.isfinite(amap).all())
                or bool(((amap < 0) | (amap > 1)).any())):
            raise ValueError("Teacher CLS/map must be finite and map must be a probability")
        metadata = {
            "seconds": time.perf_counter() - started,
            "imagebind_dtype": "torch.float16", "decoder_dtype": "torch.float32",
            "cls_dtype": str(cls.dtype), "map_dtype": str(amap.dtype),
            "cls_shape": list(cls.shape), "map_shape": list(amap.shape),
            "cls_norm": float(cls.norm()), "map_min": float(amap.min()),
            "map_max": float(amap.max()), "map_mean": float(amap.mean()),
            "map_p99": float(torch.quantile(amap.flatten(), .99)),
            "imagebind_layers": [7, 15, 23, 31],
            "feature_layout": "four [257,1,1280] -> transpose/remove CLS -> [1,256,1280]",
            "source_sha256": {
                "imagebind_model.py": sha256_file(imagebind_model.__file__),
                "dec_big.py": sha256_file(inspect.getsourcefile(BigDec)),
            },
        }
        return cls, amap, metadata
    finally:
        encoder = decoder = pixels = encoded = patches = None
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def load_trained_bridge(backend, checkpoint, model_path, device):
    seed = int(checkpoint.get("config", {}).get("seed", checkpoint.get("seed", 42)))
    model = FrozenLanguageBridge(backend, device=device, init_seed=seed, model_path=str(model_path))
    # Initialization audit belongs to construction, not to the loaded inference
    # weights. Record only the checkpoint states verified after strict loading.
    model.prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
    model.cls_proj.load_state_dict(checkpoint["cls_proj"], strict=True)
    model.eval().requires_grad_(False)
    hashes = {}
    for name in ("prompt_learner", "cls_proj"):
        actual = getattr(model, name).state_dict()
        expected_hash = state_hash(checkpoint[name])
        hashes[name] = state_hash(actual)
        if hashes[name] != expected_hash:
            raise AssertionError("Loaded trained module differs from checkpoint: " + name)
    original = checkpoint.get("initialization_audit", {})
    current = model.initialization_audit
    for key in ("yes_id", "no_id", "training_eos_id", "generation_stop_ids", "pad_id", "bos_id"):
        if key in original and original[key] != current[key]:
            raise ValueError("Language tokenizer differs from training: " + key)
    with torch.no_grad():
        loaded_norm = float(model.prompt_learner.base_prompts.float().norm(dim=-1).mean())
    return model, {"loaded_state_sha256": hashes, "loaded_base_prompt_norm_mean": loaded_norm,
                   "weights_source": "selected checkpoint strict state load",
                   "yes_id": model.yes_id, "no_id": model.no_id,
                   "training_eos_id": model.eos_id, "generation_stop_ids": sorted(model.stop_ids),
                   "frozen_language_dtype": "torch.bfloat16", "bridge_dtype": "torch.float32"}


def save_map(amap, path, heatmap=False):
    path = Path(path)
    values = amap.detach().cpu().numpy().astype(np.float32, copy=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)
    result = {"probability_npy": str(path.resolve()), "probability_npy_sha256": sha256_file(path),
              "shape": list(values.shape), "dtype": "float32", "probability_scale": [0.0, 1.0]}
    if heatmap:
        # Blue -> cyan/green -> red, using probability itself as the fixed scale.
        rgb = np.stack([values, 1 - np.abs(2 * values - 1), 1 - values], axis=-1)
        pixels = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        destination = path.with_suffix(".heatmap.png")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
        temporary.replace(destination)
        result.update(heatmap_png=str(destination.resolve()), heatmap_scale="fixed probability0 blue to1 red")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen", "vicuna"], required=True)
    parser.add_argument("--ckpt", required=True, help="Selected stage-one lm_*/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--category", choices=sorted(DESCRIPTIONS), required=True)
    parser.add_argument("--outjson", required=True)
    parser.add_argument("--decoder", help="Actual BigDec file; SHA256 must match the bridge checkpoint")
    parser.add_argument("--imagebind", default=str(REPO / "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"))
    parser.add_argument("--model", help="Local backend model; Vicuna requires complete converted safetensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda") or args.max_new_tokens < 1:
        parser.error("A CUDA device and positive generation token limit are required")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    out = Path(args.outjson)
    map_path = out.with_suffix(".map.npy")
    destinations = [out, map_path] + ([map_path.with_suffix(".heatmap.png")] if args.heatmap else [])
    if any(path.exists() for path in destinations):
        raise FileExistsError("Inference output already exists; choose a fresh --outjson")
    for path in (args.ckpt, args.image, args.imagebind):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    checkpoint = validate_bridge_checkpoint(torch.load(args.ckpt, map_location="cpu", weights_only=True), args.backend)
    decoder_path = resolve_decoder(args.decoder, checkpoint)
    decoder_checkpoint, decoder_sha256 = validate_decoder(decoder_path, checkpoint["decoder_sha256"])
    model_path = resolve_language_model(args.backend, args.model, checkpoint)
    # Every provenance gate above runs before allocating a GPU model.
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required for this inference demo")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(args.device)
    provenance = {
        "bridge_checkpoint": str(Path(args.ckpt).resolve()), "bridge_checkpoint_sha256": sha256_file(args.ckpt),
        "selected_step": checkpoint.get("step"), "selected_val_auroc": checkpoint.get("val_auroc"),
        "decoder": str(decoder_path.resolve()), "decoder_sha256": decoder_sha256,
        "imagebind": str(Path(args.imagebind).resolve()), "imagebind_sha256": sha256_file(args.imagebind),
        "model": str(model_path.resolve()),
        "model_configuration_sha256": {name: sha256_file(model_path / name)
                                       for name in ("config.json", "tokenizer_config.json", "conversion.json")
                                       if (model_path / name).is_file()},
        "training_cache_sha256": checkpoint.get("cache_sha256"),
        "training_cache_manifest_sha256": checkpoint.get("cache_manifest_sha256"),
        "current_source_sha256": {
            Path(__file__).name: sha256_file(__file__),
            "migration_language.py": sha256_file(inspect.getsourcefile(FrozenLanguageBridge)),
            "visa_pipeline.py": sha256_file(inspect.getsourcefile(load_image_tensor)),
        },
    }
    cls, amap, teacher = teacher_features(args.image, args.imagebind, decoder_checkpoint, args.device)
    del decoder_checkpoint
    gc.collect()
    teacher["gpu_allocated_bytes_before_language_load"] = torch.cuda.memory_allocated(args.device)
    language_started = time.perf_counter()
    model, loaded = load_trained_bridge(args.backend, checkpoint, model_path, args.device)
    del checkpoint
    gc.collect()
    with torch.inference_mode():
        packed = model.pack(cls, amap, args.category)
        score = float(model.score(packed).reshape(-1)[0])
        text = model.generate(packed, max_new_tokens=args.max_new_tokens)[0]
    torch.cuda.synchronize(args.device)
    if not np.isfinite(score):
        raise ValueError("Nonfinite Yes-minus-No score")
    generated_label = parse_generated(text)
    score_label = int(score > 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = save_map(amap, map_path, args.heatmap)
    result = {
        "demo_version": DEMO_VERSION, "pipeline_version": PIPELINE_VERSION,
        "backend": args.backend, "image": str(Path(args.image).resolve()), "image_sha256": sha256_file(args.image),
        "category": args.category, "input_geometry": "RGB direct bicubic square224, no crop",
        "score": score, "score_definition": "first-answer logit(Yes)-logit(No); score>0 isYes, elseNo",
        "score_decision": "Yes" if score_label else "No", "generated_answer": text,
        "generated_decision": None if generated_label is None else "Yes" if generated_label else "No",
        "generated_invalid": generated_label is None,
        "score_generation_agree": generated_label == score_label if generated_label is not None else None,
        "generation": {"method": "greedy", "max_new_tokens": args.max_new_tokens, "EOS_omitted": True},
        "teacher": teacher, "loaded_bridge": loaded, "provenance": provenance, "artifacts": artifacts,
        "language_load_and_inference_seconds": time.perf_counter() - language_started,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
    }
    atomic_json(result, out)
    print(json.dumps({"outjson": str(out.resolve()), "score": score, "score_decision": result["score_decision"],
                      "generated_answer": text, "probability_npy": artifacts["probability_npy"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
