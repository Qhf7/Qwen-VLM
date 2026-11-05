"""Qwen3-VL native-path parity gate and a minimal, explicit bridge API.

Audited against the installed Transformers 4.57.1 Qwen3-VL source. Importing
this module does not load a checkpoint. By default the CLI only audits source
and configuration; --run explicitly enables a GPU parity check. No ImageBind
dependency is used.

Example (run only after the preceding experiment has finished):
  python migration_native_parity.py --run --out runs/native_parity/report.json

Reusable bridge draft: prepare_native_inputs -> manual_prefill ->
manual_decode_step. Optional continuous auxiliary/map tokens go AFTER the
last complete vision_start ... vision_end block. Shadow ordinary text IDs
let the official get_rope_index recompute all MRoPE positions and deltas;
auxiliary tokens are never image placeholders and never receive DeepStack.
This API intentionally supports one unpadded, still-image prompt per call;
it fails on videos/batches rather than inventing their positional semantics.
"""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


AUDITED_TRANSFORMERS_VERSION = "4.57.1"
AUDITED_MODELING_SHA256 = "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da"
DEFAULT_MODEL = "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"
PIPELINE_VERSION = "qwen3vl-native-mrope-deepstack-parity-v1"


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def source_audit(model_path):
    """Read installed source/config only; do not instantiate a neural network."""
    import transformers
    from transformers.models.qwen3_vl import modeling_qwen3_vl as modeling

    source_path = Path(inspect.getsourcefile(modeling))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    processor = json.loads((model_path / "preprocessor_config.json").read_text())
    methods = [
        (modeling.Qwen3VLModel, "get_image_features"),
        (modeling.Qwen3VLModel, "get_placeholder_mask"),
        (modeling.Qwen3VLModel, "get_rope_index"),
        (modeling.Qwen3VLModel, "forward"),
        (modeling.Qwen3VLTextModel, "forward"),
        (modeling.Qwen3VLForConditionalGeneration, "forward"),
        (modeling.Qwen3VLForConditionalGeneration, "prepare_inputs_for_generation"),
    ]
    api = {}
    for cls, name in methods:
        method = getattr(cls, name)
        api[cls.__name__ + "." + name] = {
            "signature": str(inspect.signature(method)),
            "source_sha256": hashlib.sha256(inspect.getsource(method).encode()).hexdigest(),
        }
    passed = (transformers.__version__ == AUDITED_TRANSFORMERS_VERSION
              and digest == AUDITED_MODELING_SHA256 and config["model_type"] == "qwen3_vl")
    return {
        "passed": passed, "transformers_version": transformers.__version__,
        "source_file": str(source_path), "source_sha256": digest,
        "expected_source_sha256": AUDITED_MODELING_SHA256, "api": api,
        "model_config": config, "processor_config": processor,
    }


@dataclass
class NativeInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    visual_pos_masks: torch.Tensor
    deepstack_visual_embeds: list
    cache_position: torch.Tensor
    shadow_input_ids: torch.Tensor
    image_grid_thw: torch.Tensor
    auxiliary_mask: torch.Tensor

    def lm_kwargs(self, use_cache=True):
        return dict(
            input_ids=None, inputs_embeds=self.inputs_embeds,
            attention_mask=self.attention_mask, position_ids=self.position_ids,
            cache_position=self.cache_position, use_cache=use_cache,
            visual_pos_masks=self.visual_pos_masks,
            deepstack_visual_embeds=self.deepstack_visual_embeds,
        )


@dataclass
class NativeDecodeState:
    past_key_values: object
    attention_mask: torch.Tensor
    rope_deltas: torch.Tensor


def _validate_prompt(model, batch):
    ids, mask = batch["input_ids"], batch["attention_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not bool((mask == 1).all()):
        raise ValueError("Native bridge draft requires one unpadded prompt")
    if batch.get("pixel_values_videos") is not None or batch.get("video_grid_thw") is not None:
        raise ValueError("This native bridge draft handles still images only")
    if "pixel_values" not in batch or "image_grid_thw" not in batch:
        raise ValueError("Expected the official processor image inputs")
    starts = (ids[0] == model.config.vision_start_token_id).nonzero().flatten()
    ends = (ids[0] == model.config.vision_end_token_id).nonzero().flatten()
    if len(starts) == 0 or len(starts) != len(ends):
        raise ValueError("Expected complete vision_start ... vision_end blocks")
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        if end <= start + 1 or (index and start <= ends[index - 1]):
            raise ValueError("Malformed or overlapping image blocks")
        if not bool((ids[0, start + 1:end] == model.config.image_token_id).all()):
            raise ValueError("Only ordinary still-image blocks are supported")


def prepare_native_inputs(model, batch, auxiliary_tokens=None,
                          auxiliary_placeholder_id=None, freeze_vision=True):
    """Mirror Qwen3VLModel.forward's image scatter, DeepStack and MRoPE.

    Auxiliary tokens are optional [1, N, text_hidden_size] continuous values.
    For training, leave this function outside inference_mode; freeze_vision
    only suppresses gradients through the base embeddings and visual tower.
    The auxiliary cat/cast keeps its gradient path. A normal text token ID
    supplied by the caller acts only as a positional shadow, never as data
    sent through the language embedding layer for these auxiliary tokens.
    """
    _validate_prompt(model, batch)
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    context = torch.no_grad() if freeze_vision else nullcontext()
    with context:
        embeddings = model.get_input_embeddings()(ids)
        images, deepstack = model.model.get_image_features(
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"]
        )
        image_embeddings = torch.cat(images, dim=0).to(embeddings.device, embeddings.dtype)
        image_mask, _ = model.model.get_placeholder_mask(
            ids, inputs_embeds=embeddings, image_features=image_embeddings
        )
        embeddings = embeddings.masked_scatter(image_mask, image_embeddings)
    visual_mask = image_mask[..., 0]
    auxiliary_mask = torch.zeros_like(visual_mask)

    if auxiliary_tokens is not None:
        if (auxiliary_tokens.ndim != 3 or auxiliary_tokens.shape[0] != 1
                or auxiliary_tokens.shape[2] != embeddings.shape[2]):
            raise ValueError("Auxiliary tokens must be [1,N,text_hidden_size]")
        n = auxiliary_tokens.shape[1]
        special_ids = {model.config.image_token_id, model.config.video_token_id,
                       model.config.vision_start_token_id, model.config.vision_end_token_id}
        if auxiliary_placeholder_id is None or auxiliary_placeholder_id in special_ids:
            raise ValueError("Supply an ordinary text-token ID as auxiliary_placeholder_id")
        if not 0 <= auxiliary_placeholder_id < model.config.text_config.vocab_size:
            raise ValueError("Auxiliary shadow token ID is outside the text vocabulary")
        # Full vision block remains intact. In particular insertion is NOT before vision_end.
        insert = int((ids[0] == model.config.vision_end_token_id).nonzero()[-1]) + 1
        auxiliary = auxiliary_tokens.to(device=embeddings.device, dtype=embeddings.dtype)
        embeddings = torch.cat((embeddings[:, :insert], auxiliary, embeddings[:, insert:]), dim=1)
        shadow = ids.new_full((1, n), int(auxiliary_placeholder_id))
        ids = torch.cat((ids[:, :insert], shadow, ids[:, insert:]), dim=1)
        attention = torch.cat((attention[:, :insert], attention.new_ones((1, n)), attention[:, insert:]), dim=1)
        visual_mask = torch.cat((visual_mask[:, :insert], visual_mask.new_zeros((1, n)), visual_mask[:, insert:]), dim=1)
        auxiliary_mask = torch.cat((auxiliary_mask[:, :insert], auxiliary_mask.new_ones((1, n)), auxiliary_mask[:, insert:]), dim=1)
    positions, deltas = model.model.get_rope_index(
        input_ids=ids, image_grid_thw=batch["image_grid_thw"], attention_mask=attention
    )
    if bool((auxiliary_mask & visual_mask).any()):
        raise AssertionError("Auxiliary tokens must not receive visual DeepStack features")
    return NativeInputs(
        embeddings, attention, positions, deltas, visual_mask, deepstack,
        torch.arange(ids.shape[1], device=ids.device), ids, batch["image_grid_thw"], auxiliary_mask,
    )


def manual_prefill(model, prepared, logits_to_keep=0):
    """Direct language_model + lm_head prefill, with explicit 3-axis positions."""
    output = model.model.language_model(**prepared.lm_kwargs(use_cache=True))
    hidden = output.last_hidden_state[:, -logits_to_keep:] if logits_to_keep else output.last_hidden_state
    logits = model.lm_head(hidden)
    state = NativeDecodeState(output.past_key_values, prepared.attention_mask, prepared.rope_deltas)
    return SimpleNamespace(logits=logits, state=state)


def manual_decode_step(model, token_ids, state):
    """One cached text step; cache slots include all prompt/auxiliary tokens."""
    if token_ids.shape != (1, 1):
        raise ValueError("Cached bridge draft takes one text token at a time")
    seen = state.past_key_values.get_seq_length()
    if seen != state.attention_mask.shape[1]:
        raise ValueError("Cache length and complete attention mask disagree")
    cache_position = torch.tensor([seen], device=token_ids.device, dtype=torch.long)
    position_ids = cache_position.view(1, 1, 1).expand(3, 1, 1) + state.rope_deltas.view(1, 1, 1)
    attention = torch.cat((state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1)
    output = model.model.language_model(
        input_ids=None, inputs_embeds=model.get_input_embeddings()(token_ids),
        attention_mask=attention, position_ids=position_ids,
        past_key_values=state.past_key_values, cache_position=cache_position,
        use_cache=True, visual_pos_masks=None, deepstack_visual_embeds=None,
    )
    state = NativeDecodeState(output.past_key_values, attention, state.rope_deltas)
    return SimpleNamespace(logits=model.lm_head(output.last_hidden_state), state=state)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    return value


@contextmanager
def capture_lm_inputs(model):
    traces = []
    fields = ("inputs_embeds", "position_ids", "attention_mask", "cache_position",
              "visual_pos_masks", "deepstack_visual_embeds")

    def capture(module, args, kwargs):
        traces.append({field: _snapshot(kwargs.get(field)) for field in fields})

    handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        yield traces
    finally:
        handle.remove()


def compare_tensor(reference, candidate, atol=0.0, rtol=0.0):
    if reference is None or candidate is None:
        return {"passed": reference is None and candidate is None,
                "reference_none": reference is None, "candidate_none": candidate is None}
    if reference.shape != candidate.shape:
        return {"passed": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    difference = (reference.float() - candidate.float()).abs()
    finite_difference = difference[torch.isfinite(difference)]
    return {"passed": finite and bool(torch.allclose(reference.float(), candidate.float(), atol=atol, rtol=rtol)),
            "shape": list(reference.shape), "finite": finite,
            "reference_nonfinite": int((~torch.isfinite(reference)).sum()),
            "candidate_nonfinite": int((~torch.isfinite(candidate)).sum()),
            "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype),
            "max_abs_diff": float(finite_difference.max()) if finite_difference.numel() else None,
            "mean_abs_diff": float(finite_difference.mean()) if finite_difference.numel() else None}


def compare_traces(reference, candidate):
    result = {"same_call_count": len(reference) == len(candidate), "steps": []}
    for a, b in zip(reference, candidate):
        comparisons = {}
        for field in ("inputs_embeds", "position_ids", "attention_mask", "cache_position", "visual_pos_masks"):
            comparisons[field] = compare_tensor(a[field], b[field])
        a_deep, b_deep = a["deepstack_visual_embeds"], b["deepstack_visual_embeds"]
        if a_deep is None or b_deep is None:
            comparisons["deepstack"] = {"passed": a_deep is None and b_deep is None}
        else:
            layers = [compare_tensor(x, y) for x, y in zip(a_deep, b_deep)]
            comparisons["deepstack"] = {"passed": len(a_deep) == len(b_deep) and all(x["passed"] for x in layers), "layers": layers}
        result["steps"].append(comparisons)
    result["passed"] = result["same_call_count"] and all(
        comparison["passed"] for step in result["steps"] for comparison in step.values()
    )
    return result


def audit_dtypes(model):
    """Require BF16 vision blocks, main merger and every DeepStack merger."""
    counts = {}
    violations = []
    for name, parameter in model.named_parameters():
        group = "vision" if name.startswith("model.visual.") else "language_and_head"
        key = group + ":" + str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
        if parameter.dtype != torch.bfloat16 or parameter.__class__.__name__ == "Params4bit":
            violations.append({"name": name, "dtype": str(parameter.dtype), "class": parameter.__class__.__name__})
    expected_modules = ["model.visual", "model.visual.merger", "model.visual.deepstack_merger_list"]
    modules = dict(model.named_modules())
    present = {name: name in modules for name in expected_modules}
    vision_buffers = {name: str(buffer.dtype) for name, buffer in model.named_buffers() if name.startswith("model.visual.")}
    return {"passed": not violations and all(present.values()), "parameter_counts": counts,
            "required_modules": present, "violations": violations, "vision_buffers": vision_buffers,
            "quantization": "none; all model parameters BF16"}


def structured_image(width, height):
    yy, xx = np.indices((height, width))
    return Image.fromarray(np.stack(((xx * 255 // max(width - 1, 1)),
                                     (yy * 255 // max(height - 1, 1)),
                                     (((xx // 17 + yy // 23) % 2) * 200 + 25)), axis=-1).astype(np.uint8))


def prepare_processor_batch(processor, model, image, device):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe the visible shapes briefly."}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors="pt")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}


def run_case(model, processor, batch, steps, atol, rtol):
    from transformers import GenerationConfig

    prepared = prepare_native_inputs(model, batch)
    length = batch["input_ids"].shape[1]
    model.model.rope_deltas = None
    with capture_lm_inputs(model) as official_trace:
        official = model(**batch, cache_position=prepared.cache_position, use_cache=True, logits_to_keep=0)
    expected_deltas = official.rope_deltas.detach().clone()
    official_logits = official.logits.detach().cpu()
    del official  # Release the official KV cache before creating the manual one.
    with capture_lm_inputs(model) as manual_trace:
        manual = manual_prefill(model, prepared)
    prefill = compare_tensor(official_logits, manual.logits, atol, rtol)
    last_argmax_equal = bool(torch.equal(official_logits[:, -1].argmax(-1), manual.logits[:, -1].argmax(-1).cpu()))
    prefill_trace = compare_traces(official_trace, manual_trace)
    delta_check = compare_tensor(expected_deltas, prepared.rope_deltas)
    del official_logits, manual

    # A fresh clean config avoids model-shipped sampling/logit processors.
    cfg = GenerationConfig(
        max_new_tokens=steps, do_sample=False, num_beams=1, use_cache=True,
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=model.generation_config.pad_token_id,
        repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0,
        return_dict_in_generate=True, output_scores=True,
    )
    model.model.rope_deltas = None
    with capture_lm_inputs(model) as generation_trace:
        generated = model.generate(**batch, generation_config=cfg, use_model_defaults=False)
    expected_tokens = generated.sequences[:, length:].detach().cpu()
    expected_scores = [score.detach().cpu() for score in generated.scores]
    del generated
    actual_tokens, score_checks = [], []
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    with capture_lm_inputs(model) as manual_generation_trace:
        # HF generate requests one prefill logit. Match the lm_head GEMM shape:
        # projecting all positions can differ by one BF16 ULP on some logits.
        output = manual_prefill(model, prepared, logits_to_keep=1)
        for index in range(len(expected_scores)):
            next_logits = output.logits[:, -1]
            score_checks.append(compare_tensor(expected_scores[index], next_logits, atol, rtol))
            token = next_logits.argmax(-1, keepdim=True)
            actual_tokens.append(int(token.item()))
            if int(token.item()) in eos or index + 1 == len(expected_scores):
                break
            output = manual_decode_step(model, token, output.state)
    generation_inputs = compare_traces(generation_trace, manual_generation_trace)
    token_equal = actual_tokens == expected_tokens[0].tolist()
    numeric = all(check["passed"] for check in score_checks)
    # Record complete small positional arrays so order/axis differences remain reviewable.
    trace_positions = [{"position_ids": trace["position_ids"].tolist(),
                        "cache_position": trace["cache_position"].tolist()}
                       for trace in generation_trace]
    return {
        "passed": prefill["passed"] and last_argmax_equal and prefill_trace["passed"]
                  and delta_check["passed"] and token_equal and numeric and generation_inputs["passed"],
        "processor_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "image_grid_thw": batch["image_grid_thw"].tolist(),
        "merged_image_tokens": int(prepared.visual_pos_masks.sum()),
        "deepstack_shapes": [list(value.shape) for value in prepared.deepstack_visual_embeds],
        "visual_positions": prepared.visual_pos_masks[0].nonzero().flatten().tolist(),
        "vision_start_positions": (batch["input_ids"][0] == model.config.vision_start_token_id).nonzero().flatten().tolist(),
        "vision_end_positions": (batch["input_ids"][0] == model.config.vision_end_token_id).nonzero().flatten().tolist(),
        "mrope_position_ids": prepared.position_ids.tolist(), "rope_deltas": prepared.rope_deltas.tolist(),
        "prefill_logits": prefill, "prefill_last_argmax_equal": last_argmax_equal,
        "prefill_lm_inputs": prefill_trace, "rope_delta_check": delta_check,
        "greedy": {"token_ids_equal": token_equal, "official_tokens": expected_tokens[0].tolist(),
                   "manual_tokens": actual_tokens, "text": processor.tokenizer.decode(actual_tokens),
                   "step_logits": score_checks, "lm_inputs": generation_inputs,
                   "official_position_and_cache_steps": trace_positions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run", action="store_true", help="Explicitly enable GPU model loading and parity execution")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", help="Optional real source image; resized per aspect-ratio case")
    parser.add_argument("--case", action="append", help="WIDTHxHEIGHT; repeatable (default224x224,448x224,224x448)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.0001)
    parser.add_argument("--out", default="runs/native_parity/report.json")
    args = parser.parse_args()
    if args.steps < 2 or args.atol < 0 or args.rtol < 0:
        parser.error("Need at least two generated tokens to test cached decoding, and nonnegative tolerances")
    cases = []
    for text in args.case or ("224x224", "448x224", "224x448"):
        width, height = map(int, text.lower().split("x"))
        if min(width, height) <= 0:
            parser.error("Case dimensions must be positive")
        cases.append((width, height))
    report = {"pipeline_version": PIPELINE_VERSION, "mode": "run" if args.run else "source_only",
              "bridge_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_audit": source_audit(args.model), "cases": [],
              "auxiliary_interface_status": "API draft; excluded from the no-extra-token parity gate"}
    atomic_json(report, args.out)
    if not report["source_audit"]["passed"]:
        raise RuntimeError("Installed Qwen3-VL source differs from the audited source; re-audit before GPU execution")
    if not args.run:
        print(json.dumps({"source_audit_passed": True, "gpu_model_loaded": False, "report": args.out}))
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--run requires CUDA; source-only audit does not")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This parity gate requires BF16 support")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report["dtype_audit"] = audit_dtypes(model)
    report["runtime"] = {"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
                         "device": args.device, "attention_backend": "sdpa", "atol": args.atol, "rtol": args.rtol}
    if not report["dtype_audit"]["passed"]:
        atomic_json(report, args.out)
        raise RuntimeError("BF16 dtype audit failed, including visual/merger/DeepStack modules")
    source_image = Image.open(args.image).convert("RGB") if args.image else None
    try:
        with torch.inference_mode():
            for width, height in cases:
                image = (source_image.resize((width, height), Image.Resampling.BICUBIC)
                         if source_image is not None else structured_image(width, height))
                batch = prepare_processor_batch(processor, model, image, args.device)
                result = run_case(model, processor, batch, args.steps, args.atol, args.rtol)
                result["requested_image_wh"] = [width, height]
                result["source"] = args.image or "synthetic structured RGB"
                report["cases"].append(result)
                atomic_json(report, args.out)
                print(json.dumps({"case": [width, height], "grid": result["image_grid_thw"],
                                  "passed": result["passed"]}), flush=True)
                del batch
        # At least one actual cached step is required, even if a prompt immediately emits EOS.
        has_cached_step = all(len(case["greedy"]["step_logits"]) >= 2 for case in report["cases"])
        report["passed"] = all(case["passed"] for case in report["cases"]) and has_cached_step
        report["all_cases_exercised_cached_decoding"] = has_cached_step
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    except Exception as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if source_image is not None:
            source_image.close()
        atomic_json(report, args.out)
    if not report["passed"]:
        raise RuntimeError("Native parity gate failed; inspect per-case report before migration")
    print(json.dumps({"passed": True, "report": args.out, "peak_gpu_allocated_bytes": report["peak_gpu_allocated_bytes"]}))


if __name__ == "__main__":
    main()
