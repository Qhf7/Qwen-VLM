"""Train frozen-Qwen / Qwen qkvo-LoRA using a shared bridge and batch schedule.

The model and Focal/Dice formulation are inherited from the recorded strict
bridge. New matched runs use 128 target tokens and a saved initialization/order;
the earlier 64-token results remain a separate experiment.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, BitsAndBytesConfig

from train_vicuna_matched import (
    BRIDGE_MODULES, PROMPT_START, VicunaCachedModel, bridge_fingerprint, collate, import_file, load_tensors,
    process_batch_instance, resolve_code_dir, sha256, token_budget_audit,
)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QwenCachedModel(nn.Module):
    """Cache-only Qwen bridge; neither ImageBind nor its package is imported."""

    def __init__(self, qwen_ckpt, init_ckpt=None, code_dir=None, max_tgt_len=128,
                 device="cuda:0", bridge_init=None):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration
        self.device = torch.device(device)
        self.code_dir = resolve_code_dir(code_dir)
        architecture = import_file("qwen_matched_architecture",
                                   self.code_dir / "model" / "AnomalyGPT_models.py")
        losses = import_file("qwen_matched_losses", self.code_dir / "utils" / "loss.py")
        self.image_decoder = architecture.LinearLayer(1280, 1024, 4)
        self.prompt_learner = architecture.PromptLearner(1, 4096)
        self.loss_focal = losses.FocalLoss()
        self.loss_dice = losses.BinaryDiceLoss()
        self.llama_model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_ckpt, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": str(self.device)}, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.llama_model.model.visual = None
        self.llama_model.eval()
        for parameter in self.llama_model.parameters():
            parameter.requires_grad = False
        self.lm = self.llama_model.model.language_model
        self.lm_head = self.llama_model.lm_head
        if self.lm.config.hidden_size != 4096:
            raise ValueError("Retained PromptLearner requires Qwen hidden_size=4096")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(qwen_ckpt, trust_remote_code=True)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        self.llama_proj = nn.Linear(1024, 4096)
        self.max_tgt_len = max_tgt_len
        self.pixel_weight = 1.0
        if init_ckpt:
            self.load_bridge(init_ckpt, modules=("image_decoder", "prompt_learner"))
        if bridge_init:
            self.load_bridge(bridge_init)
        for name in BRIDGE_MODULES:
            getattr(self, name).to(self.device, dtype=torch.bfloat16)
        gc.collect()
        torch.cuda.empty_cache()

    load_bridge = VicunaCachedModel.load_bridge
    bridge_state = VicunaCachedModel.bridge_state
    forward_cached = VicunaCachedModel.forward_cached

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask,
                    anomaly_embedding=None):
        input_ids, target_ids, attention_mask = [
            value.to(self.device) for value in (input_ids, target_ids, attention_mask)]
        batch_size = img_embeds.shape[0]
        # Preserve the existing strict-Qwen experiment's EOS fallback when its
        # tokenizer defines no BOS. Vicuna uses its own genuine BOS token.
        bos_id = self.llama_tokenizer.bos_token_id or self.llama_tokenizer.eos_token_id
        bos = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)

        def text_embedding(text):
            ids = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
            return self.lm.embed_tokens(ids.to(self.device)).expand(batch_size, -1, -1)

        prefix = [self.lm.embed_tokens(bos), text_embedding(PROMPT_START),
                  img_embeds, text_embedding("</Img> ")]
        if anomaly_embedding is not None:
            prefix.append(anomaly_embedding)
        prefix = torch.cat(prefix, dim=1)
        embeddings = torch.cat([prefix, self.lm.embed_tokens(input_ids)], dim=1)
        labels = torch.cat([torch.full(prefix.shape[:2], -100, device=self.device, dtype=torch.long),
                            target_ids], dim=1)
        attention = torch.cat([torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
                               attention_mask], dim=1)
        if embeddings.shape[:2] != labels.shape or attention.shape != labels.shape:
            raise ValueError("Embedding/label/attention shape mismatch")
        return embeddings, labels, attention

    def _forward_lm(self, inputs_embeds, attention_mask, targets):
        batch_size, sequence_length = inputs_embeds.shape[:2]
        positions = torch.arange(sequence_length, device=inputs_embeds.device)
        positions = positions.view(1, 1, -1).expand(3, batch_size, -1)
        output = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         position_ids=positions, use_cache=False, return_dict=True)
        logits = self.lm_head(output.last_hidden_state)
        labels = targets[:, 1:]
        supervised = labels.ne(-100).sum()
        if supervised.item() == 0:
            raise ValueError("No assistant tokens survive max_tgt_len")
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100,
                               reduction="sum") / supervised
        return SimpleNamespace(loss=loss, logits=logits)


def factory(args):
    """Factory for evaluate_models.py --model-factory train_qwen_matched:factory."""
    return QwenCachedModel(
        qwen_ckpt=args.qwen_ckpt, init_ckpt=getattr(args, "init_ckpt", None) or None,
        code_dir=args.code_dir, device=args.device,
        max_tgt_len=getattr(args, "max_tgt_len", 128),
    )


def read_schedule(path, files, steps, batch_size):
    schedule = json.loads(Path(path).read_text())
    if isinstance(schedule, dict):
        schedule = schedule["batches"]
    if not isinstance(schedule, list) or len(schedule) < steps:
        raise ValueError("Sample schedule has fewer batches than requested steps")
    schedule = schedule[:steps]
    names = {item.name for item in files}
    for batch in schedule:
        if not isinstance(batch, list) or not 1 <= len(batch) <= batch_size:
            raise ValueError(f"Invalid batch: {batch!r}")
        if any(name not in names for name in batch):
            raise ValueError("Scheduled filename absent from cache or not a simple filename")
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--qwen-ckpt", required=True)
    parser.add_argument("--imagebind-ckpt", help="Recorded only; cache-only model does not load it")
    parser.add_argument("--bridge-init", required=True, type=Path)
    parser.add_argument("--sample-order", required=True, type=Path)
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--lora", choices=("none", "qkvo"), default="none")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tgt-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.max_tgt_len < 1:
        parser.error("max-steps, batch-size and max-tgt-len must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The NF4 matched training requires CUDA")
    torch.cuda.set_device(device)
    if args.save_path.exists() and any(args.save_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {args.save_path}")
    for path in [args.bridge_init, args.sample_order]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.code_dir = resolve_code_dir(args.code_dir)
    files = sorted(args.cache.glob("*.pt"))
    if not files:
        raise FileNotFoundError(args.cache)
    schedule = read_schedule(args.sample_order, files, args.max_steps, args.batch_size)
    args.save_path.mkdir(parents=True, exist_ok=True)
    (args.save_path / "sample_order.json").write_text(json.dumps({"batches": schedule}, indent=2))
    reset_rng(args.seed)
    model = QwenCachedModel(qwen_ckpt=args.qwen_ckpt, code_dir=args.code_dir,
                            max_tgt_len=args.max_tgt_len, device=args.device,
                            bridge_init=args.bridge_init)
    model.train()
    model.llama_model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name in BRIDGE_MODULES:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = True
    lora_parameters = 0
    if args.lora == "qkvo":
        from peft import LoraConfig, get_peft_model
        if not hasattr(model.lm, "prepare_inputs_for_generation"):
            model.lm.prepare_inputs_for_generation = (
                lambda input_ids=None, past_key_values=None, **kwargs: kwargs)
        configuration = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model.lm = get_peft_model(model.lm, configuration)
        # Enable only the language adapter's dropout. The frozen Qwen backbone
        # has attention_dropout=0, and its parameters remain frozen.
        model.lm.train()
        lora_parameters = sum(p.numel() for p in model.lm.parameters() if p.requires_grad)
        if lora_parameters == 0:
            raise ValueError("LoRA requested but no language parameters are trainable")
    else:
        model.lm.eval()
    model.lm_head.eval()
    model.pixel_weight = args.pixel_weight
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-3)
    token_audit = token_budget_audit(model.llama_tokenizer, args.cache, schedule, args.max_tgt_len)
    (args.save_path / "token_budget.json").write_text(json.dumps(token_audit, indent=2))
    source_paths = {
        "runner": Path(__file__),
        "shared_training": Path(__file__).with_name("train_vicuna_matched.py"),
        "qwen_model": args.code_dir / "model" / "openllama_qwen3vl_strict.py",
        "bridge_architecture": args.code_dir / "model" / "AnomalyGPT_models.py",
        "pixel_losses": args.code_dir / "utils" / "loss.py",
    }
    tokenizer = model.llama_tokenizer
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "backend": "Qwen3-VL-8B language_model", "lora": args.lora,
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "initial_bridge_sha256": sha256(args.bridge_init),
        "actual_initial_bridge_fingerprint": bridge_fingerprint(model),
        "sample_order_input_sha256": sha256(args.sample_order),
        "sample_order_sha256": sha256(args.save_path / "sample_order.json"),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json") if (args.cache / "manifest.json").is_file() else None,
        "token_budget_sha256": sha256(args.save_path / "token_budget.json"),
        "actual_unique_source_paths": token_audit["actual_unique_source_paths"],
        "actual_unique_variants": token_audit["unique_variants"],
        "actual_variant_exposures": token_audit["actual_variant_exposures"],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "language_lora_parameters": lora_parameters,
        "tokenizer": {"class": type(tokenizer).__name__, "bos_token_id": tokenizer.bos_token_id,
                      "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                      "format": "manual ### Human / ### Assistant; no chat template"},
        "matching": {"initial_bridge": "all three bridge modules loaded strictly from shared tensor file",
                     "sample_order": "explicit per-step cache filenames; no implicit RNG sampling",
                     "ce_normalization": "sum over supervised assistant tokens / token count in each update",
                     "reference_map": "disabled; cached text-matching map for every arm",
                     "max_tgt_len": args.max_tgt_len,
                     "rng_reset": "after model, adapter, optimizer and token audit; immediately before training"},
        "imagebind_instantiated": False,
        "vision_modules_released": ["Qwen native visual tower"],
        "steps": 0,
    }
    (args.save_path / "run.json").write_text(json.dumps(metadata, indent=2))
    # Construction consumes RNG differently across language architectures and
    # LoRA insertion. The schedule is explicit; reset controls training dropout.
    reset_rng(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    total_tokens, total_correct = 0, 0
    with open(args.save_path / "history.jsonl", "w", buffering=1) as history:
        for step, filenames in enumerate(schedule, 1):
            batch = collate([load_tensors(args.cache / name) for name in filenames])
            optimizer.zero_grad(set_to_none=True)
            # Same losses and label handling as Vicuna; Qwen retains its own
            # prompt wrapper and three-axis text-position forward implementation.
            loss, details = VicunaCachedModel.forward_cached(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_tokens += details["supervised_tokens"]
            total_correct += details["token_correct"]
            entry = {"step": step, "files": filenames, "loss": float(loss.detach()),
                     **details, "grad_norm": float(grad_norm),
                     "elapsed_seconds": time.time() - started}
            history.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    torch.save(VicunaCachedModel.bridge_state(model), args.save_path / "pytorch_model.pt")
    if args.lora == "qkvo":
        model.lm.save_pretrained(args.save_path / "lora_adapter")
    metadata.update({"steps": len(schedule), "num_files": len(files),
                     "seconds": time.time() - started,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "checkpoint_sha256": sha256(args.save_path / "pytorch_model.pt"),
                     "total_supervised_tokens": total_tokens,
                     "training_token_accuracy": total_correct / max(1, total_tokens)})
    (args.save_path / "run.json").write_text(json.dumps(metadata, indent=2))
    print("TRAINING_COMPLETE", json.dumps({"steps": len(schedule), "out": str(args.save_path)}), flush=True)


if __name__ == "__main__":
    main()
