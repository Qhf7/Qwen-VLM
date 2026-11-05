"""Evaluate retained AnomalyGPT bridges on a frozen, real-image feature cache.

All labels and masks stay outside the language-model inference functions. The
same averaged four-level map conditions generation and supplies map metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


BRIDGE_PREFIXES = ("image_decoder.", "prompt_learner.", "llama_proj.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_torch(path):
    # These are the user's own experiment checkpoints and cache records.
    return torch.load(path, map_location="cpu", weights_only=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def bridge_state(model):
    return {key: value.detach().cpu() for name in ("image_decoder", "prompt_learner", "llama_proj")
            for key, value in ((name + "." + key, value)
                               for key, value in getattr(model, name).state_dict().items())}


def load_bridge(model, path):
    state = load_torch(path)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    expected = bridge_state(model)
    actual = {key: value for key, value in state.items() if key.startswith(BRIDGE_PREFIXES)}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Incomplete bridge checkpoint: missing={missing}, unexpected={unexpected}")
    for key in expected:
        if tuple(expected[key].shape) != tuple(actual[key].shape):
            raise ValueError(f"Bridge shape mismatch: {key}: {actual[key].shape} != {expected[key].shape}")
    for name in ("image_decoder", "prompt_learner", "llama_proj"):
        prefix = name + "."
        getattr(model, name).load_state_dict(
            {key[len(prefix):]: value for key, value in actual.items() if key.startswith(prefix)}, strict=True)
    return {"keys": len(actual), "sha256": sha256(path), "path": str(Path(path).resolve())}


def build_model(args):
    if args.code_dir:
        sys.path.insert(0, str(Path(args.code_dir).resolve()))
    if args.model_factory:
        module, name = args.model_factory.rsplit(":", 1)
        model = getattr(importlib.import_module(module), name)(args)
    elif args.backend == "vicuna":
        from train_vicuna_matched import VicunaCachedModel
        model = VicunaCachedModel(
            vicuna_ckpt=args.vicuna_ckpt, init_ckpt=args.init_ckpt or None,
            code_dir=args.code_dir, max_tgt_len=128, device=args.device,
            load_in_4bit=not args.vicuna_full_precision)
    else:
        if not args.qwen_ckpt or not args.imagebind_ckpt:
            raise ValueError("Qwen requires --qwen-ckpt and --imagebind-ckpt")
        from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel
        model = OpenLLAMAPEFTQwenStrictModel(
            imagebind_ckpt_path=args.imagebind_ckpt,
            vicuna_ckpt_path=args.qwen_ckpt,
            init_ckpt_path=args.init_ckpt or None,
            max_tgt_len=128, stage=1)
    # Cached inference only needs the language model and the three bridge modules.
    if hasattr(model, "visual_encoder"):
        model.visual_encoder = None
    if args.backend == "qwen" and hasattr(model, "llama_model"):
        outer = getattr(model.llama_model, "model", None)
        if outer is not None and hasattr(outer, "visual"):
            outer.visual = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.export_initial_bridge:
        initial_path = Path(args.export_initial_bridge)
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        if initial_path.exists():
            raise FileExistsError(f"Initial bridge already exists: {initial_path}")
        torch.save(bridge_state(model), initial_path)
        print(json.dumps({"export_initial_bridge": str(initial_path), "sha256": sha256(initial_path)}), flush=True)
    bridge_audit = load_bridge(model, args.checkpoint) if args.checkpoint else None
    if args.lora_adapter:
        if args.backend != "qwen":
            raise ValueError("The recorded adapters are Qwen language-only adapters")
        from peft import PeftModel
        if not hasattr(model.lm, "prepare_inputs_for_generation"):
            model.lm.prepare_inputs_for_generation = lambda input_ids=None, past_key_values=None, **kwargs: kwargs
        model.lm = PeftModel.from_pretrained(model.lm, args.lora_adapter, is_trainable=False)
    model.eval()
    model.lm.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, bridge_audit


def tokenize(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not isinstance(ids, list) or (ids and isinstance(ids[0], list)):
        raise TypeError("Expected one token-ID sequence")
    return ids


def embedder(model):
    if hasattr(model.lm, "get_input_embeddings"):
        layer = model.lm.get_input_embeddings()
        if layer is not None:
            return layer
    return model.lm.embed_tokens


def candidate_audit(tokenizer, question):
    prefix_text = question + "\n### Assistant:"
    prefix_ids = tokenize(tokenizer, prefix_text)
    result = {}
    for candidate in ("Yes", "No"):
        ids = tokenize(tokenizer, candidate)
        full_ids = tokenize(tokenizer, prefix_text + candidate)
        boundary_stable = full_ids[:len(prefix_ids)] == prefix_ids
        # Training tokenizes user and assistant turns separately. Record both
        # representations; score exactly the appended training representation.
        result[candidate.lower()] = {
            "text": candidate, "ids": ids,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            "full_text_prefix_stable": boundary_stable,
            "full_text_suffix_ids": full_ids[len(prefix_ids):] if boundary_stable else None,
            "matches_training_segmented_tokens": full_ids == prefix_ids + ids,
        }
        if not ids:
            raise ValueError(f"Empty candidate tokens: {candidate}")
        if result[candidate.lower()]["decoded"].strip() != candidate:
            raise ValueError(f"Candidate token round-trip failed: {result[candidate.lower()]}")
    if result["yes"]["ids"] == result["no"]["ids"]:
        raise ValueError("Yes and No have identical tokenizer IDs")
    return result


def decode_prediction(text):
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.I)
    return -1 if not match else int(match.group(1).lower() == "yes")


def model_device_dtype(model):
    parameter = next(model.llama_proj.parameters())
    return parameter.device, parameter.dtype


def map_and_prefix(model, records):
    device, dtype = model_device_dtype(model)
    patches = [torch.cat([row["patch_features"][level] for row in records], dim=1).to(device, dtype=dtype)
               for level in range(4)]
    for patch in patches:
        if patch.ndim != 3 or patch.shape[1] != len(records) or patch.shape[-1] != 1280:
            raise ValueError(f"Invalid cached patch shape: {patch.shape}")
    patch_tokens = model.image_decoder(patches)
    text_features = torch.stack([row["text_features"] for row in records]).to(device, dtype=dtype)
    if text_features.shape != (len(records), 2, 1024):
        raise ValueError(f"Invalid text-feature shape: {text_features.shape}")
    maps = []
    for tokens in patch_tokens:
        tokens = F.normalize(tokens, dim=-1)
        logits = 100.0 * (tokens @ text_features.transpose(-2, -1))
        batch, length, _ = logits.shape
        side = math.isqrt(length)
        if side * side != length:
            raise ValueError("Patch token count is not a square")
        logits = F.interpolate(logits.permute(0, 2, 1).reshape(batch, 2, side, side),
                               size=(224, 224), mode="bilinear", align_corners=True)
        maps.append(logits.softmax(dim=1)[:, 1])
    anomaly_map = torch.stack(maps).mean(dim=0).unsqueeze(1)
    if not torch.isfinite(anomaly_map).all():
        raise ValueError("Non-finite anomaly map")
    globals_ = torch.stack([row["global_feature"] for row in records]).to(device, dtype=dtype)
    if globals_.shape != (len(records), 1024):
        raise ValueError(f"Invalid global-feature shape: {globals_.shape}")
    image_embeds = model.llama_proj(globals_).unsqueeze(1)
    soft = model.prompt_learner(anomaly_map)
    questions = {row["question"] for row in records}
    if len(questions) != 1:
        raise ValueError("Batch must have an identical question and token length")
    question = next(iter(questions))
    ids = tokenize(model.llama_tokenizer, question + "\n### Assistant:")
    input_ids = torch.tensor([ids] * len(records), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    targets = torch.full_like(input_ids, -100)
    embeddings, _, attention_mask = model.prompt_wrap(image_embeds, input_ids, targets, attention_mask, soft)
    if embeddings.shape[:2] != attention_mask.shape:
        raise ValueError("Prefix embedding/attention shape mismatch")
    return anomaly_map[:, 0], embeddings, attention_mask, question


def lm_forward(model, backend, embeddings, attention_mask, start=0, past=None, use_cache=True):
    batch, length = embeddings.shape[:2]
    positions = torch.arange(start, start + length, device=embeddings.device)
    if backend == "qwen":
        position_ids = positions.view(1, 1, -1).expand(3, batch, -1)
    else:
        position_ids = positions.view(1, -1).expand(batch, -1)
    kwargs = dict(inputs_embeds=embeddings, attention_mask=attention_mask,
                  position_ids=position_ids, past_key_values=past,
                  use_cache=use_cache, return_dict=True)
    if backend == "qwen":
        kwargs["cache_position"] = positions
    output = model.lm(**kwargs)
    return model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1].float(), output.past_key_values


def continuation_logp(model, backend, prefix, attention_mask, candidate_ids, first_logits):
    ids = torch.tensor(candidate_ids, device=prefix.device, dtype=torch.long)
    if len(candidate_ids) == 1:
        return first_logits.log_softmax(dim=-1)[:, int(ids[0])]
    batch, prefix_length = prefix.shape[:2]
    previous = embedder(model)(ids[:-1].unsqueeze(0).expand(batch, -1))
    embeddings = torch.cat([prefix, previous], dim=1)
    attn = torch.cat([attention_mask, torch.ones((batch, len(ids) - 1), device=prefix.device,
                                                dtype=attention_mask.dtype)], dim=1)
    positions = torch.arange(embeddings.shape[1], device=prefix.device)
    position_ids = (positions.view(1, 1, -1).expand(3, batch, -1) if backend == "qwen"
                    else positions.view(1, -1).expand(batch, -1))
    kwargs = dict(inputs_embeds=embeddings, attention_mask=attn, position_ids=position_ids,
                  use_cache=False, return_dict=True)
    if backend == "qwen":
        kwargs["cache_position"] = positions
    output = model.lm(**kwargs)
    hidden = output.last_hidden_state[:, prefix_length - 1:prefix_length + len(ids) - 1]
    log_probs = model.lm_head(hidden).float().log_softmax(dim=-1)
    targets = ids.view(1, -1, 1).expand(batch, -1, -1)
    return log_probs.gather(-1, targets).squeeze(-1).sum(-1)


def eos_ids(model):
    ids = set()
    tokenizer = model.llama_tokenizer
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    full = getattr(model, "llama_model", None)
    config = getattr(full, "generation_config", None)
    value = getattr(config, "eos_token_id", None)
    if isinstance(value, int):
        ids.add(value)
    elif value:
        ids.update(int(item) for item in value)
    return ids


def language_predictions(model, backend, prefix, attention_mask, audit, max_new_tokens):
    first_logits, past = lm_forward(model, backend, prefix, attention_mask)
    scores = {name: continuation_logp(model, backend, prefix, attention_mask, audit[name]["ids"], first_logits)
              for name in ("yes", "no")}
    next_ids = first_logits.argmax(dim=-1)
    batch = prefix.shape[0]
    generated = [[] for _ in range(batch)]
    done = [False] * batch
    reasons = ["max_new_tokens"] * batch
    stop_ids = eos_ids(model)
    tokenizer = model.llama_tokenizer
    stop_text_ids = tokenize(tokenizer, "\n###")
    for step in range(max_new_tokens):
        for index, token in enumerate(next_ids.tolist()):
            if done[index]:
                continue
            generated[index].append(token)
            if token in stop_ids:
                done[index] = True
                reasons[index] = "eos"
            elif stop_text_ids and generated[index][-len(stop_text_ids):] == stop_text_ids:
                done[index] = True
                reasons[index] = "training_turn_delimiter"
            elif "\n###" in tokenizer.decode(generated[index], skip_special_tokens=True):
                # BPE can merge the newline with the preceding punctuation.
                # Match the decoded new-token stream as well as standalone IDs.
                done[index] = True
                reasons[index] = "training_turn_delimiter_text"
        if all(done) or step + 1 == max_new_tokens:
            break
        attention_mask = torch.cat([attention_mask, torch.ones((batch, 1), device=prefix.device,
                                                              dtype=attention_mask.dtype)], dim=1)
        embeddings = embedder(model)(next_ids[:, None])
        logits, past = lm_forward(model, backend, embeddings, attention_mask,
                                  start=prefix.shape[1] + step, past=past)
        next_ids = logits.argmax(dim=-1)
    result = []
    for index, ids in enumerate(generated):
        text_ids = list(ids)
        while text_ids and text_ids[-1] in stop_ids:
            text_ids.pop()
        if stop_text_ids and text_ids[-len(stop_text_ids):] == stop_text_ids:
            text_ids = text_ids[:-len(stop_text_ids)]
        text = tokenizer.decode(text_ids, skip_special_tokens=True).split("\n###", 1)[0]
        yes = float(scores["yes"][index].item())
        no = float(scores["no"][index].item())
        result.append(dict(answer=text, generated_token_ids=ids, prediction=decode_prediction(text),
                           termination=reasons[index], yes_logp=yes, no_logp=no,
                           answer_log_odds=yes - no,
                           candidate_prediction=int(yes > no)))
    return result


def ranking(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores contain non-finite values")
    return dict(auroc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
                ap=float(average_precision_score(labels, scores)) if labels.sum() > 0 else None)


def classification(rows, key="prediction"):
    labels = np.asarray([int(row["label"]) for row in rows])
    pred = np.asarray([row[key] for row in rows])
    f1 = []
    for label in (0, 1):
        tp = int(((labels == label) & (pred == label)).sum())
        fp = int(((labels != label) & (pred == label)).sum())
        fn = int(((labels == label) & (pred != label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    n_normal = int((labels == 0).sum())
    n_anomaly = int((labels == 1).sum())
    return dict(macro_f1=float(np.mean(f1)), accuracy=float((labels == pred).mean()),
                anomaly_recall=float(((labels == 1) & (pred == 1)).sum() / n_anomaly) if n_anomaly else None,
                normal_recall=float(((labels == 0) & (pred == 0)).sum() / n_normal) if n_normal else None,
                normal_fpr=float(((labels == 0) & (pred == 1)).sum() / n_normal) if n_normal else None,
                invalid_rate=float((pred == -1).mean()), n_normal=n_normal, n_anomaly=n_anomaly)


def aggregate(rows, out):
    groups = defaultdict(list)
    for row in rows:
        groups[row["class_name"]].append(row)
    metrics = {"num_sources": len(rows), "generation": classification(rows),
               "candidate_classification": classification(rows, "candidate_prediction"),
               "map_image": ranking([row["label"] for row in rows], [row["map_score"] for row in rows]),
               "answer_image": ranking([row["label"] for row in rows], [row["answer_log_odds"] for row in rows]),
               "per_class": {}}
    for name, group in sorted(groups.items()):
        labels, scores = [], []
        for row in group:
            with np.load(out / row["map_file"]) as value:
                labels.append(value["mask"].reshape(-1))
                scores.append(value["anomaly_map"].reshape(-1))
        pixels = ranking(np.concatenate(labels), np.concatenate(scores))
        metrics["per_class"][name] = {
            "num_sources": len(group), "generation": classification(group), "pixel": pixels,
            "map_image": ranking([row["label"] for row in group], [row["map_score"] for row in group]),
            "answer_image": ranking([row["label"] for row in group], [row["answer_log_odds"] for row in group])}
    metrics["category_macro"] = {}
    for family, keys in (("pixel", ("auroc", "ap")), ("map_image", ("auroc", "ap")),
                         ("answer_image", ("auroc", "ap")), ("generation", ("macro_f1", "anomaly_recall", "normal_fpr"))):
        metrics["category_macro"][family] = {}
        for key in keys:
            values = [value[family][key] for value in metrics["per_class"].values() if value[family][key] is not None]
            metrics["category_macro"][family][key] = float(np.mean(values)) if values else None
    metrics["definitions"] = {
        "pixel": "Four-level maps are averaged before scoring; per-class pixel metrics pool normal and anomaly images; category_macro weights products equally.",
        "generation_macro_f1": "Normal/anomalous label macro F1 over all sources; invalid answers remain false negatives for their GT class.",
        "candidate": "Sum continuation conditional log probabilities using the same segmented tokenization as training; log_odds=Yes-No.",
        "map_image_score": "Maximum of the averaged 224x224 map, without per-image rescaling.",
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "vicuna"), default="qwen")
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--lora-adapter", "--lora_adapter", default="")
    parser.add_argument("--init-ckpt", "--init_ckpt", default="")
    parser.add_argument("--qwen-ckpt", "--qwen_ckpt", default="")
    parser.add_argument("--vicuna-ckpt", "--vicuna_ckpt", default="")
    parser.add_argument("--imagebind-ckpt", "--imagebind_ckpt", default="")
    parser.add_argument("--code-dir", default="")
    parser.add_argument("--model-factory", default="", help="Optional module:function(args) returning the common model interface")
    parser.add_argument("--vicuna-full-precision", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--export-initial-bridge", default="")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        raise ValueError("Invalid batch size, generation length or limit")
    if not args.export_only and not args.cache:
        raise ValueError("Evaluation requires --cache")
    if args.export_only and not args.export_initial_bridge:
        raise ValueError("--export-only requires --export-initial-bridge")
    for name in ("checkpoint", "init_ckpt"):
        if getattr(args, name) and not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model, bridge_audit = build_model(args)
    metadata = {"args": vars(args), "source_sha256": sha256(__file__), "bridge": bridge_audit,
                "torch": torch.__version__, "python": sys.version, "candidate_audits": {},
                "init_sha256": sha256(args.init_ckpt) if args.init_ckpt else None}
    try:
        source = inspect.getsourcefile(type(model))
        metadata["model_source"] = {"path": source, "sha256": sha256(source)} if source else None
    except (OSError, TypeError):
        metadata["model_source"] = None
    if args.lora_adapter:
        metadata["adapter_hashes"] = {str(path.relative_to(args.lora_adapter)): sha256(path)
                                      for path in sorted(Path(args.lora_adapter).glob("*")) if path.is_file()}
    write_json(out / "run.json", metadata)
    if args.export_only:
        return
    cache = Path(args.cache)
    files = sorted(cache.glob("*.pt"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No cache records found")
    groups = defaultdict(list)
    source_paths = set()
    cache_hashes = {}
    for file in files:
        row = load_torch(file)
        for key in ("global_feature", "patch_features", "text_features", "mask", "class_name", "source_path", "label", "question"):
            if key not in row:
                raise KeyError(f"{file}: missing {key}")
        if len(row["patch_features"]) != 4 or tuple(row["mask"].shape) != (1, 224, 224):
            raise ValueError(f"{file}: incorrect cached shape")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"{file}: invalid binary label")
        if row["source_path"] in source_paths:
            raise ValueError(f"Duplicate source: {row['source_path']}")
        source_paths.add(row["source_path"])
        groups[(row["class_name"], row["question"])].append(file)
        cache_hashes[file.name] = sha256(file)
    metadata["cache_hashes"] = cache_hashes
    manifest_path = cache / "manifest.json"
    metadata["cache_manifest_sha256"] = sha256(manifest_path) if manifest_path.exists() else None
    metadata["num_sources"] = len(files)
    (out / "maps").mkdir()
    rows = []
    started = time.time()
    with (out / "predictions.jsonl").open("w") as stream, torch.inference_mode():
        for (class_name, question), group in sorted(groups.items()):
            audit = candidate_audit(model.llama_tokenizer, question)
            metadata["candidate_audits"][class_name] = audit
            write_json(out / "run.json", metadata)
            for offset in range(0, len(group), args.batch_size):
                batch_files = group[offset:offset + args.batch_size]
                batch = [load_torch(path) for path in batch_files]
                t0 = time.time()
                anomaly_maps, prefix, attention_mask, _ = map_and_prefix(model, batch)
                predictions = language_predictions(model, args.backend, prefix, attention_mask, audit, args.max_new_tokens)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                for file, record, anomaly, prediction in zip(batch_files, batch, anomaly_maps, predictions):
                    source_id = hashlib.sha256(str(record["source_path"]).encode()).hexdigest()[:24]
                    map_file = "maps/" + source_id + ".npz"
                    array = anomaly.float().cpu().numpy()
                    mask = (record["mask"][0].numpy() > 0).astype(np.uint8)
                    np.savez_compressed(out / map_file, anomaly_map=array, mask=mask)
                    result = dict(source_id=source_id, source_path=record["source_path"],
                                  class_name=class_name, label=bool(record["label"]), cache_file=file.name,
                                  map_file=map_file, map_score=float(array.max()),
                                  foreground_pixels_224=int(mask.sum()), batch_seconds=elapsed,
                                  batch_size=len(batch), **prediction)
                    rows.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                print(json.dumps({"progress": len(rows), "total": len(files), "class": class_name,
                                  "batch_seconds": round(elapsed, 3)}, ensure_ascii=False), flush=True)
    metrics = aggregate(rows, out)
    metrics["seconds"] = time.time() - started
    metrics["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    metrics["checkpoint"] = args.checkpoint or "constructor_initialization"
    metrics["lora_adapter"] = args.lora_adapter or None
    metrics["backend"] = args.backend
    metrics["seed"] = args.seed
    write_json(out / "metrics.json", metrics)
    metadata["completed"] = True
    metadata["completed_num_sources"] = len(rows)
    metadata["prediction_sha256"] = sha256(out / "predictions.jsonl")
    write_json(out / "run.json", metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
