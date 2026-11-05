"""Cached ImageBind -> Vicuna baseline with an explicit shared initialization/order.

This uses the available Vicuna-7B-v1.1 checkpoint, not the original paper's
Vicuna-v0 checkpoint.  The visual cache, bridge initialization and sample order
can be shared with the Qwen arms; no ImageBind model is instantiated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BRIDGE_MODULES = ("image_decoder", "prompt_learner", "llama_proj")
PROMPT_START = "### Human: <Img>"


def resolve_code_dir(code_dir: str | Path | None = None) -> Path:
    candidates = ([Path(code_dir)] if code_dir is not None else []) + [
        Path(__file__).resolve().parent.parent / "code",
        Path(__file__).resolve().parent / "server_snapshot" / "code",
        Path.cwd(),
    ]
    for path in candidates:
        if (path / "model" / "AnomalyGPT_models.py").is_file():
            return path.resolve()
    raise FileNotFoundError("Pass --code-dir containing model/AnomalyGPT_models.py")


def import_file(name: str, path: Path):
    # Import by filename to avoid model/__init__.py constructing CUDA text inputs.
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tensors(path: str | Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bridge_fingerprint(model):
    """Hash actual bridge tensors, one at a time, after device/dtype conversion."""
    digest = hashlib.sha256()
    entries = sorted(
        (f"{module_name}.{key}", value)
        for module_name in BRIDGE_MODULES
        for key, value in getattr(model, module_name).state_dict().items()
    )
    for key, tensor in entries:
        value = tensor.detach().contiguous().cpu()
        header = json.dumps(
            {"key": key, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        raw_bytes = value.view(torch.uint8).numpy().tobytes()
        # Length prefixes prevent ambiguous key/shape/raw-byte concatenations.
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
        del value, raw_bytes
    return digest.hexdigest()


def build_one_instance(tokenizer, conversation):
    """Retain the strict Qwen arm's manual conversation formatting verbatim."""
    input_ids, target_ids = [], []
    for i, turn in enumerate(conversation):
        role = turn["from"]
        if i == 0:
            if role != "human":
                raise ValueError("Conversation must begin with a human question")
            text = turn["value"] + "\n### Assistant:"
            supervised = False
        elif role == "human":
            text = "Human: " + turn["value"] + "\n### Assistant:"
            supervised = False
        elif role == "gpt":
            text = turn["value"] + "\n###"
            supervised = True
        else:
            raise ValueError(f"Unknown role: {role}")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids.extend(ids)
        target_ids.extend(ids if supervised else [-100] * len(ids))
    return torch.tensor(input_ids), torch.tensor(target_ids)


def process_batch_instance(tokenizer, conversations, max_tgt_len):
    pairs = [build_one_instance(tokenizer, item) for item in conversations]
    ids = pad_sequence([p[0] for p in pairs], batch_first=True,
                       padding_value=tokenizer.pad_token_id)[:, :max_tgt_len]
    targets = pad_sequence([p[1] for p in pairs], batch_first=True,
                           padding_value=-100)[:, :max_tgt_len]
    # Match the existing strict arm's padding rule. Assistant labels remain -100
    # on padding. Neither tokenizer's actual EOS is inserted in the target text.
    attention = ids.ne(tokenizer.pad_token_id).long()
    return ids.long(), targets.long(), attention


def token_budget_audit(tokenizer, cache, schedule, max_tgt_len):
    """Audit actual scheduled exposure, source coverage and assistant-token caps.

    Nominal synthetic variants and masks with positive pixels are counted
    separately because synthesis can return an empty mask.
    """
    cache = Path(cache)
    exposure_counts = Counter(name for batch in schedule for name in batch)
    unique_names = sorted(exposure_counts)
    truncated, zero_supervision = [], []
    per_file, source_paths = {}, set()
    class_label_counts, exposure_class_label_counts = Counter(), Counter()
    class_variant_counts = Counter()
    mask_counts, exposure_mask_counts = Counter(), Counter()
    for name in unique_names:
        record = load_tensors(cache / name)
        full_ids, full_labels = build_one_instance(tokenizer, record["text"])
        labels = full_labels[:max_tgt_len]
        retained = int(labels.ne(-100).sum())
        full_count = int(full_labels.ne(-100).sum())
        source_path = str(record.get("source_path", ""))
        if source_path:
            source_paths.add(source_path)
        class_name = str(record.get("class_name", "unknown"))
        positive_mask = bool(torch.as_tensor(record["mask"]).gt(0.3).any())
        mask_label = "positive" if positive_mask else "normal"
        variant = str(record.get("variant", "unknown"))
        exposure = exposure_counts[name]
        class_label_counts[(class_name, mask_label)] += 1
        exposure_class_label_counts[(class_name, mask_label)] += exposure
        class_variant_counts[(class_name, variant)] += 1
        mask_counts[mask_label] += 1
        exposure_mask_counts[mask_label] += exposure
        per_file[name] = {
            "retained_supervised_tokens": retained, "full_supervised_tokens": full_count,
            "full_sequence_tokens": int(full_ids.numel()),
            "retained_sequence_tokens": min(int(full_ids.numel()), max_tgt_len),
            "source_path": source_path, "class_name": class_name,
            "variant": record.get("variant"), "positive_mask": positive_mask,
            "exposures": exposure,
        }
        if retained < full_count:
            truncated.append({"file": name, "retained": retained, "full": full_count})
        if retained == 0:
            zero_supervision.append(name)
    if zero_supervision:
        raise ValueError(f"No assistant tokens after truncation: {zero_supervision}")

    def nested_counts(counts):
        result = {}
        for (category, label), count in sorted(counts.items()):
            result.setdefault(category, {})[label] = count
        return result

    return {
        "scheduled_steps": len(schedule),
        "actual_variant_exposures": sum(exposure_counts.values()),
        "unique_variants": len(unique_names),
        "actual_unique_source_paths": len(source_paths),
        "actual_source_paths": sorted(source_paths),
        "class_label_counts": nested_counts(class_label_counts),
        "exposure_class_label_counts": nested_counts(exposure_class_label_counts),
        "class_variant_counts": nested_counts(class_variant_counts),
        "normal_vs_positive_mask_count": dict(mask_counts),
        "exposure_normal_vs_positive_mask_count": dict(exposure_mask_counts),
        "label_definition": "mask > 0.3 contains at least one positive pixel; independent of nominal variant",
        "max_tgt_len": max_tgt_len,
        "truncated_assistant_records": truncated,
        "supervised_tokens_by_file": {name: value["retained_supervised_tokens"]
                                      for name, value in per_file.items()},
        "tokens_and_sources_by_file": per_file,
    }


class VicunaCachedModel(nn.Module):
    """Expose the same bridge/LM handles used by the cache-only Qwen evaluator."""

    def __init__(self, vicuna_ckpt, init_ckpt, code_dir=None, max_tgt_len=128,
                 device="cuda:0", projection_init_checkpoint=None,
                 load_in_4bit=True):
        super().__init__()
        self.device = torch.device(device)
        self.code_dir = resolve_code_dir(code_dir)
        architecture = import_file("matched_anomaly_architecture",
                                   self.code_dir / "model" / "AnomalyGPT_models.py")
        losses = import_file("matched_anomaly_losses", self.code_dir / "utils" / "loss.py")
        self.image_decoder = architecture.LinearLayer(1280, 1024, 4)
        self.prompt_learner = architecture.PromptLearner(1, 4096)
        self.loss_focal = losses.FocalLoss()
        self.loss_dice = losses.BinaryDiceLoss()
        kwargs = {"device_map": {"": str(self.device)}, "torch_dtype": torch.bfloat16,
                  "low_cpu_mem_usage": True}
        if load_in_4bit:
            if self.device.type != "cuda":
                raise ValueError("The NF4 baseline requires a CUDA device")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.llama_model = AutoModelForCausalLM.from_pretrained(vicuna_ckpt, **kwargs)
        self.llama_model.eval()
        self.llama_model.config.use_cache = False
        for parameter in self.llama_model.parameters():
            parameter.requires_grad = False
        self.lm = self.llama_model.model
        self.lm_head = self.llama_model.lm_head
        if self.lm.config.hidden_size != 4096:
            raise ValueError("The retained PromptLearner requires hidden_size=4096")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(vicuna_ckpt, use_fast=False)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.llama_tokenizer.bos_token_id is None:
            raise ValueError("Vicuna tokenizer must define a BOS token")
        self.llama_tokenizer.padding_side = "right"
        self.llama_proj = nn.Linear(1024, 4096)
        self.max_tgt_len = max_tgt_len
        self.pixel_weight = 1.0
        self.load_bridge(init_ckpt, modules=("image_decoder", "prompt_learner"))
        if projection_init_checkpoint:
            self.load_bridge(projection_init_checkpoint, modules=("llama_proj",))
        for name in BRIDGE_MODULES:
            getattr(self, name).to(self.device, dtype=torch.bfloat16)

    def load_bridge(self, path, modules=BRIDGE_MODULES):
        state = load_tensors(path)
        if "state_dict" in state:
            state = state["state_dict"]
        for name in modules:
            prefix = name + "."
            subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not subset:
                raise ValueError(f"{path}: no {name} parameters")
            getattr(self, name).load_state_dict(subset, strict=True)
        del state

    def bridge_state(self):
        return {f"{name}.{key}": value.detach().cpu()
                for name in BRIDGE_MODULES
                for key, value in getattr(self, name).state_dict().items()}

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask,
                    anomaly_embedding=None):
        input_ids, target_ids, attention_mask = [
            item.to(self.device) for item in (input_ids, target_ids, attention_mask)
        ]
        batch_size = img_embeds.shape[0]

        def text_embedding(text):
            ids = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            return self.lm.embed_tokens(ids.to(self.device)).expand(batch_size, -1, -1)

        bos = torch.full((batch_size, 1), self.llama_tokenizer.bos_token_id,
                         device=self.device, dtype=torch.long)
        prefix = [self.lm.embed_tokens(bos), text_embedding(PROMPT_START),
                  img_embeds, text_embedding("</Img> ")]
        if anomaly_embedding is not None:
            prefix.append(anomaly_embedding)
        prefix = torch.cat(prefix, dim=1)
        embeddings = torch.cat([prefix, self.lm.embed_tokens(input_ids)], dim=1)
        empty_targets = torch.full(prefix.shape[:2], -100, device=self.device, dtype=torch.long)
        targets = torch.cat([empty_targets, target_ids], dim=1)
        attention = torch.cat([torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
                               attention_mask], dim=1)
        if embeddings.shape[:2] != targets.shape or attention.shape != targets.shape:
            raise ValueError("Embedding/label/attention shape mismatch")
        return embeddings, targets, attention

    def _forward_lm(self, inputs_embeds, attention_mask, targets):
        sequence_length = inputs_embeds.shape[1]
        positions = torch.arange(sequence_length, device=self.device).unsqueeze(0)
        positions = positions.expand(inputs_embeds.shape[0], -1)
        output = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         position_ids=positions, use_cache=False, return_dict=True)
        logits = self.lm_head(output.last_hidden_state)
        shifted_labels = targets[:, 1:]
        supervised = shifted_labels.ne(-100).sum()
        if supervised.item() == 0:
            raise ValueError("No assistant tokens survive max_tgt_len")
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               shifted_labels.reshape(-1), ignore_index=-100,
                               reduction="sum") / supervised
        return SimpleNamespace(loss=loss, logits=logits)

    def forward_cached(self, inputs):
        dtype = torch.bfloat16
        image_embeds = self.llama_proj(inputs["global_features"].to(self.device, dtype)).unsqueeze(1)
        patches = [item.to(self.device, dtype) for item in inputs["patch_features"]]
        tokens = self.image_decoder(patches)
        text = inputs["text_features"].to(self.device, dtype)
        gt = inputs["masks"].to(self.device).float().gt(0.3).float()
        maps, pixel_loss = [], 0.0
        for token in tokens:
            token = token / token.norm(dim=-1, keepdim=True)
            similarity = 100.0 * (token @ text.transpose(-2, -1))
            batch_size, length, _ = similarity.shape
            side = math.isqrt(length)
            if side * side != length:
                raise ValueError("Patch lattice must be square")
            logits = similarity.permute(0, 2, 1).reshape(batch_size, 2, side, side)
            probabilities = F.interpolate(logits, size=224, mode="bilinear",
                                            align_corners=True).softmax(dim=1)
            pixel_loss = pixel_loss + self.loss_focal(probabilities, gt)
            pixel_loss = pixel_loss + self.loss_dice(probabilities[:, 1], gt)
            maps.append(probabilities[:, 1])
        anomaly = torch.stack(maps).mean(0).unsqueeze(1)
        soft = self.prompt_learner(anomaly)
        ids, targets, attention = process_batch_instance(
            self.llama_tokenizer, inputs["texts"], self.max_tgt_len)
        embeddings, targets, attention = self.prompt_wrap(image_embeds, ids, targets, attention, soft)
        output = self._forward_lm(embeddings, attention, targets)
        labels = targets[:, 1:]
        valid = labels.ne(-100)
        predictions = output.logits[:, :-1].argmax(-1)
        correct = ((predictions == labels) & valid).sum().item()
        supervised = valid.sum().item()
        return output.loss + self.pixel_weight * pixel_loss, {
            "assistant_ce": float(output.loss.detach()),
            "pixel_loss": float(pixel_loss.detach()),
            "token_correct": correct, "supervised_tokens": supervised,
            "token_accuracy": correct / max(1, supervised),
        }


def collate(batch):
    return {
        "global_features": torch.stack([item["global_feature"] for item in batch]),
        "patch_features": [torch.cat([item["patch_features"][layer] for item in batch], dim=1)
                           for layer in range(4)],
        "text_features": torch.stack([item["text_features"] for item in batch]),
        "masks": torch.stack([item["mask"] for item in batch]),
        "texts": [item["text"] for item in batch],
        "class_names": [item["class_name"] for item in batch],
    }


def make_schedule(files, seed, batch_size, steps):
    generator = torch.Generator().manual_seed(seed)
    schedule = []
    while len(schedule) < steps:
        indices = torch.randperm(len(files), generator=generator).tolist()
        for start in range(0, len(indices), batch_size):
            schedule.append([files[index].name for index in indices[start:start + batch_size]])
            if len(schedule) == steps:
                break
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--vicuna-ckpt", required=True)
    parser.add_argument("--init-ckpt", required=True)
    parser.add_argument("--projection-init-checkpoint")
    parser.add_argument("--sample-order", type=Path,
                        help="JSON list of batches, or object with a 'batches' list; entries are cache filenames")
    parser.add_argument("--code-dir")
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.max_tgt_len < 1:
        parser.error("steps, batch-size and max-tgt-len must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The NF4 matched training requires CUDA")
    torch.cuda.set_device(device)
    if args.save_path.exists() and any(args.save_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.save_path}")
    reset_rng(args.seed)
    args.save_path.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cache.glob("*.pt"))
    if not files:
        raise FileNotFoundError(args.cache)
    if args.sample_order:
        schedule = json.loads(args.sample_order.read_text())
        if isinstance(schedule, dict):
            schedule = schedule["batches"]
        schedule = schedule[:args.max_steps]
        if len(schedule) != args.max_steps:
            raise ValueError("Sample schedule contains fewer batches than max-steps")
    else:
        schedule = make_schedule(files, args.seed, args.batch_size, args.max_steps)
    names = {path.name for path in files}
    for batch in schedule:
        if not isinstance(batch, list) or not 1 <= len(batch) <= args.batch_size:
            raise ValueError(f"Invalid scheduled batch: {batch!r}")
        if any(name not in names for name in batch):
            raise ValueError("Scheduled filename is absent from cache or is not a simple filename")
    (args.save_path / "sample_order.json").write_text(json.dumps({"batches": schedule}, indent=2))
    model = VicunaCachedModel(args.vicuna_ckpt, args.init_ckpt, args.code_dir,
                              args.max_tgt_len, args.device, args.projection_init_checkpoint)
    model.train()
    model.llama_model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name in BRIDGE_MODULES:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = True
    model.pixel_weight = args.pixel_weight
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-3)
    initial_projection = {f"llama_proj.{key}": value.detach().cpu()
                          for key, value in model.llama_proj.state_dict().items()}
    torch.save(initial_projection, args.save_path / "initial_projection.pt")
    tokenizer = model.llama_tokenizer
    token_audit = token_budget_audit(tokenizer, args.cache, schedule, args.max_tgt_len)
    (args.save_path / "token_budget.json").write_text(json.dumps(token_audit, indent=2))
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "backend": "Vicuna-7B-v1.1", "imagebind_instantiated": False,
        "source_sha256": sha256(__file__), "initial_checkpoint_sha256": sha256(args.init_ckpt),
        "projection_init_sha256": sha256(args.projection_init_checkpoint) if args.projection_init_checkpoint else None,
        "sample_order_sha256": sha256(args.save_path / "sample_order.json"),
        "initial_projection_sha256": sha256(args.save_path / "initial_projection.pt"),
        "actual_initial_bridge_fingerprint": bridge_fingerprint(model),
        "token_budget_sha256": sha256(args.save_path / "token_budget.json"),
        "actual_unique_source_paths": token_audit["actual_unique_source_paths"],
        "actual_unique_variants": token_audit["unique_variants"],
        "actual_variant_exposures": token_audit["actual_variant_exposures"],
        "tokenizer": {"class": type(tokenizer).__name__, "bos_token_id": tokenizer.bos_token_id,
                      "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                      "use_fast": tokenizer.is_fast, "format": "manual ### Human / ### Assistant; no chat template"},
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "matching": {"shared_projection_supplied": bool(args.projection_init_checkpoint),
                     "shared_schedule_supplied": bool(args.sample_order),
                     "ce_normalization": "sum over supervised assistant tokens / token count in each update",
                     "reference_map": "disabled; cached text-matching map for every arm",
                     "max_tgt_len": args.max_tgt_len,
                     "rng_reset": "after model, optimizer and token audit; immediately before training",
                     "checkpoint_caveat": "available Vicuna-v1.1, not the paper's Vicuna-v0"},
        "cache_manifest_sha256": sha256(args.cache / "manifest.json") if (args.cache / "manifest.json").is_file() else None,
        "steps": 0,
    }
    (args.save_path / "run.json").write_text(json.dumps(metadata, indent=2))
    reset_rng(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    total_tokens, total_correct = 0, 0
    with open(args.save_path / "history.jsonl", "w", buffering=1) as history:
        for step, names_this_step in enumerate(schedule, 1):
            batch = collate([load_tensors(args.cache / name) for name in names_this_step])
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.forward_cached(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_tokens += details["supervised_tokens"]
            total_correct += details["token_correct"]
            entry = {"step": step, "files": names_this_step, "loss": float(loss.detach()),
                     **details, "grad_norm": float(grad_norm), "elapsed_seconds": time.time() - started}
            history.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    torch.save(model.bridge_state(), args.save_path / "pytorch_model.pt")
    metadata.update({"steps": len(schedule), "num_files": len(files),
                     "seconds": time.time() - started,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "total_supervised_tokens": total_tokens,
                     "training_token_accuracy": total_correct / max(1, total_tokens),
                     "checkpoint_sha256": sha256(args.save_path / "pytorch_model.pt")})
    (args.save_path / "run.json").write_text(json.dumps(metadata, indent=2))
    print("TRAINING_COMPLETE", json.dumps({"steps": len(schedule), "out": str(args.save_path)}), flush=True)


if __name__ == "__main__":
    main()
