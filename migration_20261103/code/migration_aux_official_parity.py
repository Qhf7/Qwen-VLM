"""Exact auxiliary-token parity against the complete official Qwen3-VL path.

No model loads on import. The default CLI validates source/cache and writes a
CPU plan; --run explicitly loads NativeMapBridge for inference only. An
embedding forward hook replaces only the known auxiliary placeholder slots
in official prefill. Official Qwen handles image scatter, DeepStack and MRoPE.
NativeMapBridge baseline uses the unchanged original processor batch with no
embedding hook and zero auxiliary tokens.

Gate: exact, finite same-shaped official/manual prefill and cached-step
logits, language inputs, positions, masks and DeepStack streams. Official
cached-vs-full replay is a separate diagnostic, never a tolerance gate.

Example (GPU execution is an explicit caller decision):
  python migration_aux_official_parity.py --run --limit 2 \
      --out runs/native_aux_official/report.json
"""
import argparse
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import time

import torch

from migration_native_parity import (
    DEFAULT_MODEL, capture_lm_inputs, compare_tensor, compare_traces,
    manual_prefill, manual_decode_step, source_audit, atomic_json,
)
PIPELINE_VERSION = "qwen3vl-aux-official-same-shape-parity-v1"
EXPECTED_AUXILIARY_TOKENS = 18


def finite_json(value):
    """Keep failure diagnostics writable even if a model emits NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def save_report(report, path):
    atomic_json(finite_json(report), path)


def exact_tensor(reference, candidate):
    result = compare_tensor(reference, candidate, atol=0.0, rtol=0.0)
    if reference is not None and candidate is not None:
        result["dtype_equal"] = reference.dtype == candidate.dtype
        result["bitwise_equal"] = bool(torch.equal(reference.detach().cpu(), candidate.detach().cpu()))
        result["passed"] = result["passed"] and result["dtype_equal"] and result["bitwise_equal"]
    return result


def exact_traces(reference, candidate):
    result = compare_traces(reference, candidate)
    dtype_errors = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        for field in ("inputs_embeds", "position_ids", "attention_mask", "cache_position", "visual_pos_masks"):
            if left[field] is not None and right[field] is not None and left[field].dtype != right[field].dtype:
                dtype_errors.append({"step": index, "field": field})
        if left["deepstack_visual_embeds"] is not None and right["deepstack_visual_embeds"] is not None:
            for layer, (a, b) in enumerate(zip(left["deepstack_visual_embeds"], right["deepstack_visual_embeds"])):
                if a.dtype != b.dtype:
                    dtype_errors.append({"step": index, "field": "deepstack", "layer": layer})
    result["dtype_mismatches"] = dtype_errors
    result["passed"] = result["passed"] and not dtype_errors
    return result


@contextmanager
def auxiliary_embedding_hook(qwen, shadow_ids, auxiliary_mask, auxiliary_tokens, placeholder_id):
    """Replace exact slots once; ordinary occurrences of the ID stay untouched."""
    if shadow_ids.ndim != 2 or shadow_ids.shape[0] != 1 or auxiliary_mask.shape != shadow_ids.shape:
        raise ValueError("Embedding replacement requires one exact prompt and aligned auxiliary mask")
    if auxiliary_mask.dtype != torch.bool or int(auxiliary_mask.sum()) != auxiliary_tokens.shape[1]:
        raise ValueError("Auxiliary embedding replacement count/mask mismatch")
    if auxiliary_tokens.ndim != 3 or auxiliary_tokens.shape[0] != 1:
        raise ValueError("Auxiliary embeddings must be [1,N,H]")
    if not bool((shadow_ids[auxiliary_mask] == placeholder_id).all()):
        raise ValueError("Every replaced slot must contain the selected ordinary placeholder ID")
    visual_ids = {qwen.config.image_token_id, qwen.config.video_token_id,
                  qwen.config.vision_start_token_id, qwen.config.vision_end_token_id}
    if placeholder_id in visual_ids:
        raise ValueError("Auxiliary placeholder must be an ordinary text ID")
    audit = {"enabled": True, "calls": 0, "replaced_slots": int(auxiliary_mask.sum()),
             "positions": auxiliary_mask[0].nonzero().flatten().detach().cpu().tolist(),
             "nonauxiliary_unchanged": False, "replacement_exact": False}

    def replace(module, args, output):
        audit["calls"] += 1
        if audit["calls"] != 1 or not args or not torch.equal(args[0], shadow_ids):
            raise AssertionError("Auxiliary hook may only run once on the exact official prefill IDs")
        if output.shape[:2] != shadow_ids.shape or output.shape[-1] != auxiliary_tokens.shape[-1]:
            raise AssertionError("Official embedding shape changed")
        replacement = auxiliary_tokens.to(device=output.device, dtype=output.dtype).reshape(-1, output.shape[-1])
        modified = output.clone()
        modified[auxiliary_mask] = replacement
        audit["nonauxiliary_unchanged"] = bool(torch.equal(modified[~auxiliary_mask], output[~auxiliary_mask]))
        audit["replacement_exact"] = bool(torch.equal(modified[auxiliary_mask], replacement))
        if not audit["nonauxiliary_unchanged"] or not audit["replacement_exact"]:
            raise AssertionError("Auxiliary hook changed an unintended embedding slot")
        return modified

    handle = qwen.get_input_embeddings().register_forward_hook(replace)
    try:
        yield audit
    finally:
        handle.remove()
    if audit["calls"] != 1:
        raise AssertionError("Official prefill did not call the embedding replacement hook exactly once")


def logits_diagnostics(left, right, yes_id, no_id):
    """No pass threshold: compare two differently shaped attention computations."""
    result = compare_tensor(left, right, atol=0.0, rtol=0.0)
    result.pop("passed", None)
    result["diagnostic_only"] = True
    result["argmax_equal"] = bool(torch.equal(left.argmax(-1), right.argmax(-1)))
    result["left_argmax_token_id"] = int(left.argmax(-1).item())
    result["right_argmax_token_id"] = int(right.argmax(-1).item())
    a = float(left[0, -1, yes_id].float() - left[0, -1, no_id].float())
    b = float(right[0, -1, yes_id].float() - right[0, -1, no_id].float())
    result["yes_no_scores"] = {"cached": a, "full_replay": b}
    result["yes_no_score_scope"] = "next-token candidate scores after the recorded forced/argmax replay token; not the image first-answer score"
    result["yes_no_score_abs_error"] = abs(a - b)
    result["yes_no_decision_equal"] = (a > 0) == (b > 0)
    return result


def trace_summary(traces):
    return [{
        "input_embedding_shape": None if t["inputs_embeds"] is None else list(t["inputs_embeds"].shape),
        "position_ids": None if t["position_ids"] is None else t["position_ids"].tolist(),
        "cache_position": None if t["cache_position"] is None else t["cache_position"].tolist(),
        "attention_mask": None if t["attention_mask"] is None else t["attention_mask"].tolist(),
        "visual_positions": None if t["visual_pos_masks"] is None else t["visual_pos_masks"][0].nonzero().flatten().tolist(),
        "deepstack_shapes": None if t["deepstack_visual_embeds"] is None else [list(x.shape) for x in t["deepstack_visual_embeds"]],
    } for t in traces]


def official_prefill(qwen, batch, shadow_ids, mask, auxiliary, placeholder_id, use_cache=True):
    qwen.model.rope_deltas = None
    if auxiliary is None:
        if bool(mask.any()) or not torch.equal(shadow_ids, batch["input_ids"]):
            raise ValueError("Baseline official prefill requires unchanged original IDs and no auxiliary slots")
        if batch["attention_mask"].shape != shadow_ids.shape:
            raise ValueError("Baseline official attention must match the original processor IDs")
        # Direct complete official call: install no embedding hook. The batch
        # is the original processor batch (or that batch plus one replay token).
        output = qwen(**batch, cache_position=torch.arange(shadow_ids.shape[1], device=shadow_ids.device),
                      use_cache=use_cache, logits_to_keep=1, return_dict=True)
        return output, {"enabled": False, "calls": 0, "replaced_slots": 0, "positions": [],
                        "nonauxiliary_unchanged": True, "replacement_exact": None}
    with auxiliary_embedding_hook(qwen, shadow_ids, mask, auxiliary, placeholder_id) as hook:
        output = qwen(
            input_ids=shadow_ids, attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
            cache_position=torch.arange(shadow_ids.shape[1], device=shadow_ids.device),
            use_cache=use_cache, logits_to_keep=1, return_dict=True,
        )
    return output, hook


@torch.no_grad()
def run_case(bridge, item, amap, decode_choice="yes"):
    """Same interface for a 0-token native baseline or 18-token map bridge.

    This callable deliberately does not import the trainer. The trainer may
    import it lazily inside cache_self_check without a circular dependency.
    """
    if decode_choice not in {"yes", "no", "argmax"}:
        raise ValueError("decode_choice must be yes, no or argmax")
    bridge.eval()
    qwen = bridge.qwen
    batch, _ = bridge.batch(item, answer=False)
    auxiliary = bridge.auxiliary(amap)
    baseline = bool(getattr(bridge, "baseline", auxiliary is None))
    prepared = bridge.prepare(batch, auxiliary)
    length = prepared.shadow_input_ids.shape[1]
    auxiliary_n = int(prepared.auxiliary_mask.sum())
    expected_n = 0 if baseline else EXPECTED_AUXILIARY_TOKENS
    if auxiliary_n != expected_n or ((auxiliary is None) != baseline):
        raise AssertionError("Baseline requires auxiliary=None/0 slots; map bridge requires 18 actual map tokens")
    if baseline:
        if not torch.equal(prepared.shadow_input_ids, batch["input_ids"]):
            raise AssertionError("Baseline preparation must preserve the original processor IDs")
        if not torch.equal(prepared.attention_mask, batch["attention_mask"]):
            raise AssertionError("Baseline preparation must preserve the original processor attention")
        official_batch = dict(batch)
    else:
        official_batch = dict(batch, attention_mask=prepared.attention_mask)
    placeholder_id = None if baseline else bridge.placeholder_id

    # Complete one official sequence before creating a manual cache. No hook
    # survives prefill; the cached call supplies no image/grid/visual inputs.
    with capture_lm_inputs(qwen) as official_trace:
        official, hook = official_prefill(qwen, official_batch, prepared.shadow_input_ids,
                                          prepared.auxiliary_mask, auxiliary, placeholder_id)
        official_logits = official.logits.detach().cpu().clone()
        official_delta = official.rope_deltas.detach().cpu().clone()
        if official.past_key_values.get_seq_length() != length:
            raise AssertionError("Official prefill cache omits an auxiliary/prompt slot")
        if decode_choice == "argmax":
            token_id = int(official_logits[0, -1].argmax())
        else:
            token_id = bridge.yes_id if decode_choice == "yes" else bridge.no_id
        if token_id in {qwen.config.image_token_id, qwen.config.video_token_id,
                         qwen.config.vision_start_token_id, qwen.config.vision_end_token_id}:
            raise ValueError("Cached comparison token must be an ordinary nonvisual text token")
        token = prepared.shadow_input_ids.new_tensor([[token_id]])
        attention = torch.cat([prepared.attention_mask, torch.ones_like(token)], dim=1)
        cache = official.past_key_values
        del official
        official_step = qwen(input_ids=token, attention_mask=attention, past_key_values=cache,
                             cache_position=token.new_tensor([length]), use_cache=True,
                             logits_to_keep=1, return_dict=True)
        official_step_logits = official_step.logits.detach().cpu().clone()
        official_step_delta = official_step.rope_deltas.detach().cpu().clone()
        official_cache_length = int(official_step.past_key_values.get_seq_length())
        del official_step, cache

    with capture_lm_inputs(qwen) as manual_trace:
        manual = manual_prefill(qwen, prepared, logits_to_keep=1)
        manual_logits = manual.logits.detach().cpu().clone()
        if manual.state.past_key_values.get_seq_length() != length:
            raise AssertionError("Manual prefill cache omits an auxiliary/prompt slot")
        manual_step = manual_decode_step(qwen, token, manual.state)
        manual_step_logits = manual_step.logits.detach().cpu().clone()
        manual_cache_length = int(manual_step.state.past_key_values.get_seq_length())
        del manual_step, manual

    # Full replay is official, independent of the manual position API. Repeat
    # the same RGB and continuous auxiliary tensor, append one known token.
    replay_shadow = torch.cat([prepared.shadow_input_ids, token], dim=1)
    replay_mask = torch.cat([prepared.auxiliary_mask, torch.zeros_like(token, dtype=torch.bool)], dim=1)
    replay_batch = dict(batch, input_ids=replay_shadow, attention_mask=attention)
    with capture_lm_inputs(qwen) as official_full_trace:
        official_full, replay_hook = official_prefill(qwen, replay_batch, replay_shadow, replay_mask,
                                                     auxiliary, placeholder_id)
        official_full_logits = official_full.logits.detach().cpu().clone()
        official_full_delta = official_full.rope_deltas.detach().cpu().clone()
        official_full_cache_length = int(official_full.past_key_values.get_seq_length())
        del official_full
    manual_replay_batch = dict(batch)
    manual_replay_batch["input_ids"] = torch.cat([batch["input_ids"], token], dim=1)
    manual_replay_batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(token)], dim=1)
    prepared_full = bridge.prepare(manual_replay_batch, auxiliary)
    with capture_lm_inputs(qwen) as manual_full_trace:
        manual_full = manual_prefill(qwen, prepared_full, logits_to_keep=1)
        manual_full_logits = manual_full.logits.detach().cpu().clone()
        del manual_full

    if len(official_trace) != 2 or len(manual_trace) != 2 or len(official_full_trace) != 1 or len(manual_full_trace) != 1:
        raise AssertionError("Expected prefill+one cached step and one separate full replay per path")
    no_cached_visual = all(t["visual_pos_masks"] is None and t["deepstack_visual_embeds"] is None
                          for t in (official_trace[1], manual_trace[1]))
    full_position = official_full_trace[0]["position_ids"][:, :, -1:]
    cached_position = official_trace[1]["position_ids"]
    checks = {
        "prefill_logits": exact_tensor(official_logits, manual_logits),
        "cached_step_logits": exact_tensor(official_step_logits, manual_step_logits),
        "prefill_and_cached_language_inputs": exact_traces(official_trace, manual_trace),
        "prefill_rope_delta": exact_tensor(official_delta, prepared.rope_deltas.detach().cpu()),
        "cached_rope_delta": exact_tensor(official_step_delta, prepared.rope_deltas.detach().cpu()),
        "full_replay_logits": exact_tensor(official_full_logits, manual_full_logits),
        "full_replay_language_inputs": exact_traces(official_full_trace, manual_full_trace),
        "full_replay_shadow_ids": exact_tensor(replay_shadow, prepared_full.shadow_input_ids),
        "full_replay_rope_delta": exact_tensor(official_full_delta, prepared_full.rope_deltas.detach().cpu()),
        "official_full_vs_cached_last_position": exact_tensor(full_position, cached_position),
        "cache_lengths": {"passed": official_cache_length == manual_cache_length == official_full_cache_length == length + 1,
                            "official_cached": official_cache_length, "manual_cached": manual_cache_length,
                            "official_full_replay": official_full_cache_length},
        "cached_step_has_no_visual_inputs": {"passed": no_cached_visual},
    }
    return {"passed": all(check["passed"] for check in checks.values()),
            "item": {key: item[key] for key in ("path", "category", "split", "label")},
            "baseline": baseline,
            "prefix_length_with_auxiliary": length, "auxiliary_tokens": auxiliary_n,
            "auxiliary_dtype": None if auxiliary is None else str(auxiliary.dtype),
            "auxiliary_placeholder_id": placeholder_id,
            "image_grid_thw": batch["image_grid_thw"].detach().cpu().tolist(),
            "prefill_embedding_hook": hook, "full_replay_embedding_hook": replay_hook,
            "cached_input_token_id": token_id, "cached_token_choice": decode_choice,
            "cached_step_scope": "one next-token forward after the recorded prefill continuation token",
            "checks": checks,
            "official_cached_vs_official_full_replay": logits_diagnostics(official_step_logits, official_full_logits, bridge.yes_id, bridge.no_id),
            "manual_cached_vs_manual_full_replay": logits_diagnostics(manual_step_logits, manual_full_logits, bridge.yes_id, bridge.no_id),
            "official_prefill_cached_trace": trace_summary(official_trace),
            "manual_prefill_cached_trace": trace_summary(manual_trace),
            "official_full_replay_trace": trace_summary(official_full_trace),
            "manual_full_replay_trace": trace_summary(manual_full_trace)}


def load_prompt_checkpoint(path, bridge, cache, current):
    from migration_native_train import sha256_file, state_hash

    if bool(getattr(bridge, "baseline", False)):
        raise ValueError("A baseline has no PromptLearner checkpoint to load")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    run = checkpoint.get("run", checkpoint)
    expected = {"cache_sha256": cache["cache_sha256"], "decoder_sha256": cache["decoder_sha256"],
                "cache_manifest_sha256": cache["manifest_sha256"], "source_sha256": current["source_sha256"],
                "auxiliary_tokens": EXPECTED_AUXILIARY_TOKENS, "cls_tokens": 0}
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError("Prompt checkpoint provenance differs for " + key)
    weights = checkpoint.get("prompt_learner")
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint does not contain learned PromptLearner state")
    bridge.prompt_learner.load_state_dict(weights, strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in bridge.prompt_learner.parameters()):
        raise ValueError("Prompt checkpoint contains nonfinite parameters")
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
            "step": checkpoint.get("step"), "prompt_state_sha256": state_hash(bridge.prompt_learner)}


def main():
    # CLI-only dependency: run_case remains importable by the trainer without
    # importing migration_native_train back at module initialization time.
    from migration_native_train import (
        NativeMapBridge, DEFAULT_CACHE, check_parity_report, load_cache,
        fixed_subset, sha256_file, state_hash,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--parity-report", help="Optional preceding no-extra-token GPU parity report; validated if supplied")
    parser.add_argument("--ckpt", help="Optional native-map PromptLearner checkpoint; otherwise same-seed fresh initialization")
    parser.add_argument("--baseline", action="store_true", help="Check native Qwen with 0 auxiliary tokens and no embedding hook")
    parser.add_argument("--out", default="runs/migration_20261103/native_aux_official/report.json")
    parser.add_argument("--run", action="store_true", help="Explicitly enable BF16 CUDA inference; default is a CPU plan")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=2, help="Fixed validation subset size; 0 selects all val")
    parser.add_argument("--cache-index", action="append", type=int, help="Repeat explicit validation cache indices instead of --limit")
    parser.add_argument("--decode-token", choices=("yes", "no", "argmax"), default="yes")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("Limit must be nonnegative")
    if args.baseline and args.ckpt:
        parser.error("--baseline has no PromptLearner; do not supply --ckpt")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Existing parity report is preserved: " + str(out))
    report = {"pipeline_version": PIPELINE_VERSION, "mode": "run" if args.run else "plan",
              "status": "starting", "gpu_model_loaded": False, "passed": False,
              "config": vars(args), "cases": [],
              "gate_definition": "finite, dtype-equal, exact same-shaped official/manual prefill+cached+full-replay logits and LM inputs; exact masks/positions/cache lengths",
              "diagnostic_definition": "official cached vs official full replay changes attention query length; report errors without tolerance-based pass/fail"}
    save_report(report, out)
    try:
        current = source_audit(args.model)
        report["source_audit"] = current
        if current.get("passed") is not True:
            raise RuntimeError("Current official modeling source/version/config failed the audited SHA gate")
        if args.parity_report:
            previous = json.loads(Path(args.parity_report).read_text())
            check_parity_report(previous, current)
            report["preceding_parity_report"] = {"path": str(Path(args.parity_report).resolve()),
                                                    "sha256": sha256_file(args.parity_report)}
        cache, indices = load_cache(args.cache)
        selected = list(dict.fromkeys(args.cache_index)) if args.cache_index else fixed_subset(indices["val"], cache["items"], args.limit)
        if not selected or any(i not in set(indices["val"]) for i in selected):
            raise ValueError("All requested cache indices must be validation items")
        report["selected_cache_indices"] = selected
        report["cache_provenance"] = {key: cache.get(key) for key in ("pipeline_version", "decoder_sha256", "cache_sha256", "manifest_sha256", "geometry")}
        paths = [Path(__file__).resolve(), Path(inspect.getsourcefile(NativeMapBridge)),
                 Path(inspect.getsourcefile(manual_prefill))]
        report["source_files"] = {str(p): sha256_file(p) for p in paths}
        report["status"] = "planned"
        save_report(report, out)
        if not args.run:
            print(json.dumps({"status": "planned", "gpu_model_loaded": False, "n": len(selected), "report": str(out)}))
            return
        if not args.device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("--run requires a BF16-capable CUDA device")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        bridge = NativeMapBridge(args.model, args.device, args.seed, baseline=args.baseline).eval()
        report["gpu_model_loaded"] = True
        report["dtype_audit"] = bridge.dtype_audit
        report["initialization_audit"] = bridge.initialization_audit
        if args.ckpt:
            report["checkpoint"] = load_prompt_checkpoint(args.ckpt, bridge, cache, current)
        report["prompt_state_sha256"] = None if args.baseline else state_hash(bridge.prompt_learner)
        if bridge.prompt_class_path is not None:
            report["source_files"][str(bridge.prompt_class_path)] = sha256_file(bridge.prompt_class_path)
        report["runtime"] = {"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
                               "device": args.device, "attention_backend": "sdpa",
                               "precision": "frozen full Qwen BF16; unchanged FP32 PromptLearner and teacher map"}
        report["status"] = "running"
        save_report(report, out)
        for index in selected:
            torch.cuda.synchronize(args.device)
            began = time.perf_counter()
            try:
                result = run_case(bridge, cache["items"][index],
                                  None if args.baseline else cache["maps"][index], args.decode_token)
            except Exception as error:
                report["cases"].append({"passed": False, "cache_index": index,
                                         "item": cache["items"][index],
                                         "error": {"type": type(error).__name__, "message": str(error)}})
                save_report(report, out)
                raise
            torch.cuda.synchronize(args.device)
            result.update(cache_index=index, elapsed_seconds=time.perf_counter() - began)
            report["cases"].append(result)
            save_report(report, out)
            print(json.dumps({"cache_index": index, "passed": result["passed"],
                              "official_cached_full_max_abs": result["official_cached_vs_official_full_replay"]["max_abs_diff"]}), flush=True)
        report["passed"] = len(report["cases"]) == len(selected) and all(case["passed"] for case in report["cases"])
        report["status"] = "complete" if report["passed"] else "failed"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        if not report["passed"]:
            raise RuntimeError("Auxiliary official/manual exact same-shape gate failed; inspect per-case checks")
    except Exception as error:
        report["passed"] = False
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save_report(report, out)
    print(json.dumps({"passed": True, "report": str(out), "n": len(report["cases"])}))


if __name__ == "__main__":
    main()
