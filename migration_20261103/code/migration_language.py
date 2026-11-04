"""Matched, frozen BF16 language backends for the first migration experiment.

This module deliberately uses neither model's native vision tower. Cached
ImageBind CLS features and the same 224-square anomaly maps feed fresh FP32
``Linear(1024,4096)`` and ``PromptLearner(1,4096)`` modules. With the same
``init_seed``, both backends receive identical bridge tensors. No PandaGPT,
Vicuna bridge checkpoint, LoRA, quantization, or backend-dependent rescaling is
applied. PromptLearner retains its original initialization, including N(0,1)
base prompts; ``initialization_audit`` records their scale and the sampled
vocabulary embedding scale so this choice remains visible.

The LM parameters stay frozen/eval, but training forwards remain in autograd:
gradients must pass through the language model into the bridge. Both models
use ordinary text positions in this language-only protocol. Qwen's text model
expands these positions to its three identical rotary axes; no native image
tokens, visual-position masks, or DeepStack features are supplied.

Public outputs: ``pack`` returns padded model tensors plus prefix metadata;
``forward_batch`` returns [B,L,V] logits, ``loss`` returns token-mean causal CE,
``score`` returns a [B] Yes-minus-No first-answer-token logit difference, and
``generate`` returns B raw greedy answer strings. EOS is omitted from returned
strings. Empty, malformed, or non-Yes/No generations are preserved and can be
counted as invalid with ``parse_answer``.
"""
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REPO = Path(os.environ.get("ANOMALYGPT_REPO", "/root/autodl-tmp/AnomalyGPT"))
VICUNA_PATH = Path(os.environ.get('VICUNA_BF16_MODEL', str(REPO / "pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1")))
QWEN_PATH = Path(os.environ.get(
    "QWEN3VL_MODEL", "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"
))
HIDDEN_SIZE = 4096
FEATURE_MARKER = "[[ANOMALY_BRIDGE_FEATURE_SLOT]]"
YES = "Yes, there is an anomaly in the image."
NO = "No, there is no anomaly in the image."
DESCRIPTIONS = {
    "candle": "a candle", "capsules": "a capsule", "cashew": "a cashew",
    "chewinggum": "a chewinggum", "fryum": "a fryum",
    "macaroni1": "a macaroni", "macaroni2": "a macaroni",
    "pcb1": "a pcb", "pcb2": "a pcb", "pcb3": "a pcb", "pcb4": "a pcb",
    "pipe_fryum": "a pipe fryum",
}
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def parse_answer(text: str) -> int | None:
    """Parse an anchored Yes/No; retain abstentions and format failures as None."""
    match = re.match(r"^\s*(Yes|No)\b", text, flags=re.IGNORECASE)
    return None if match is None else int(match.group(1).lower() == "yes")


def _prompt_learner_class():
    # Import the standalone definition without model/__init__ side effects.
    path = REPO / "code/code/model/AnomalyGPT_models.py"
    spec = importlib.util.spec_from_file_location("migration_prompt_learner", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load PromptLearner from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PromptLearner


def _as_list(value, batch_size, name):
    if isinstance(value, str):
        return [value] * batch_size
    result = list(value)
    if len(result) != batch_size or any(not isinstance(x, str) for x in result):
        raise ValueError("%s must contain one string per sample" % name)
    return result


class FrozenLanguageBridge(nn.Module):
    def __init__(self, backend, device="cuda", init_seed=42, model_path=None):
        super().__init__()
        name = str(backend).lower().replace("-", "").replace("_", "")
        if name in {"vicuna", "vicuna7b"}:
            self.backend = "vicuna"
        elif name in {"qwen", "qwen3vl", "qwen3vl8b"}:
            self.backend = "qwen"
        else:
            raise ValueError("backend must be vicuna or qwen")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("The matched BF16 experiment requires a BF16-capable CUDA GPU")
        self.init_seed = int(init_seed)
        self.dtype = torch.bfloat16
        self.model_path = Path(model_path or (VICUNA_PATH if self.backend == "vicuna" else QWEN_PATH))

        # Model construction can consume CPU RNG even when all weights load.
        # Preserve the caller's state and initialize the bridge in its own fork.
        with torch.random.fork_rng(devices=[]):
            self._load_language()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.prompt_learner = _prompt_learner_class()(1, HIDDEN_SIZE).float()
            self.cls_proj = nn.Linear(1024, HIDDEN_SIZE).float()
        self.prompt_learner.to(self.device)
        self.cls_proj.to(self.device)
        self.language_model.to(device=self.device, dtype=self.dtype).eval()
        self.lm_head.to(device=self.device, dtype=self.dtype).eval()
        for module in (self.language_model, self.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                if parameter.is_floating_point() and parameter.dtype != self.dtype:
                    raise RuntimeError("Frozen language parameters must all be BF16")
            if any("bitsandbytes" in type(m).__module__ for m in module.modules()):
                raise RuntimeError("Quantization is forbidden in the matched BF16 experiment")

        self.yes_id = self._single_token("Yes")
        self.no_id = self._single_token("No")
        self._check_answer(YES)
        self._check_answer(NO)
        self._prompt_ids = {}
        self.initialization_audit = self._audit_initialization()

    def _load_language(self):
        from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

        kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True,
                      local_files_only=True, attn_implementation="sdpa")
        if self.backend == "vicuna":
            self.tok = LlamaTokenizer(vocab_file=str(self.model_path / "tokenizer.model"))
            outer = LlamaForCausalLM.from_pretrained(str(self.model_path), **kwargs)
            self.language_model = outer.model
        else:
            from transformers import Qwen3VLForConditionalGeneration
            self.tok = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            if not self.tok.chat_template:
                raise ValueError("Qwen tokenizer must provide its native ChatML template")
            # Load the complete checkpoint in BF16 on CPU first. Retain only the
            # text model and head, then release the outer model/vision tower
            # before either retained module is copied onto the GPU.
            outer = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), **kwargs)
            self.language_model = outer.model.language_model
        if getattr(outer, "is_quantized", False):
            raise RuntimeError("Expected an unquantized checkpoint load")
        self.lm_head = outer.lm_head
        if self.language_model.config.hidden_size != HIDDEN_SIZE:
            raise ValueError("Both backends must have hidden_size=4096")
        generation_eos = getattr(outer.generation_config, "eos_token_id", None)
        if self.tok.eos_token_id is None:
            raise ValueError("A tokenizer-specific EOS token is required")
        self.eos_id = int(self.tok.eos_token_id)
        self.stop_ids = {self.eos_id}
        if self.backend == "qwen" and generation_eos is not None:
            values = generation_eos if isinstance(generation_eos, (list, tuple)) else [generation_eos]
            self.stop_ids.update(int(value) for value in values)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = int(self.tok.pad_token_id)
        self.tok.padding_side = "right"
        # Some legacy Vicuna configs disagree with tokenizer.model (0/1 vs
        # BOS=1, EOS=2). Tokenizer IDs govern this experiment, including loss.
        self.language_model.config.bos_token_id = self.tok.bos_token_id
        self.language_model.config.eos_token_id = self.eos_id
        self.language_model.config.pad_token_id = self.pad_id
        self.language_model.config.use_cache = False
        del outer
        gc.collect()

    def train(self, mode=True):
        super().train(mode)
        self.language_model.eval()
        self.lm_head.eval()
        return self

    def _single_token(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("%s must be one token for strict first-token scoring: %s" % (text, ids))
        return int(ids[0])

    def _check_answer(self, answer):
        text = answer.strip()
        parsed = parse_answer(text)
        if parsed is None:
            raise ValueError("Training answers must begin with Yes or No")
        ids = self.tok.encode(text, add_special_tokens=False)
        expected = self.yes_id if parsed else self.no_id
        if not ids or ids[0] != expected:
            raise ValueError("Answer first token disagrees with Yes/No candidate token")
        return ids + [self.eos_id]

    def _template_ids(self, category):
        if category in self._prompt_ids:
            return self._prompt_ids[category]
        if category not in DESCRIPTIONS:
            raise ValueError("Unknown VisA category: %s" % category)
        user = ("Image features: " + FEATURE_MARKER + "\nThis is a photo of "
                + DESCRIPTIONS[category]
                + " for anomaly detection. Is there any anomaly in the image?")
        if self.backend == "qwen":
            rendered = self.tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        else:
            # Vicuna v1.1 conversation: USER/ASSISTANT, separator ' ', EOS '</s>'.
            rendered = VICUNA_SYSTEM + " USER: " + user + " ASSISTANT:"
        if rendered.count(FEATURE_MARKER) != 1:
            raise ValueError("The feature slot must occur exactly once in the user message")
        before, after = rendered.split(FEATURE_MARKER)
        before_ids = self.tok.encode(before, add_special_tokens=False)
        if self.backend == "vicuna":
            if self.tok.bos_token_id is None:
                raise ValueError("Vicuna tokenizer needs a BOS token")
            before_ids = [int(self.tok.bos_token_id)] + before_ids
        after_ids = self.tok.encode(after, add_special_tokens=False)
        self._prompt_ids[category] = (before_ids, after_ids)
        return before_ids, after_ids

    def pack(self, cls_emb, amap, category, answer=None):
        if cls_emb.ndim == 1:
            cls_emb = cls_emb.unsqueeze(0)
        if cls_emb.ndim != 2 or cls_emb.shape[1] != 1024:
            raise ValueError("CLS must have shape [B,1024]")
        batch_size = cls_emb.shape[0]
        if amap.ndim == 2:
            amap = amap.unsqueeze(0).unsqueeze(0)
        elif amap.ndim == 3:
            amap = amap.unsqueeze(1)
        if amap.shape != (batch_size, 1, 224, 224):
            raise ValueError("Anomaly maps must have shape [B,224,224] or [B,1,224,224]")
        categories = _as_list(category, batch_size, "category")
        answers = [None] * batch_size if answer is None else _as_list(answer, batch_size, "answer")
        cls_emb = cls_emb.to(device=self.device, dtype=torch.float32)
        amap = amap.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(cls_emb).all() or not torch.isfinite(amap).all():
            raise ValueError("Cached CLS/maps must be finite")
        # These operations must remain outside no_grad: both modules learn.
        tokens = self.prompt_learner(amap)
        image_token = self.cls_proj(cls_emb).unsqueeze(1)
        if tokens.shape != (batch_size, 18, HIDDEN_SIZE):
            raise ValueError("Unexpected PromptLearner output: %s" % (tuple(tokens.shape),))
        features = torch.cat([image_token, tokens], dim=1).to(self.dtype)
        embedding = self.language_model.get_input_embeddings()
        sequences, labels, prefix_lengths = [], [], []
        for index, (cat, target) in enumerate(zip(categories, answers)):
            before, after = self._template_ids(cat)
            before = torch.tensor(before, dtype=torch.long, device=self.device)
            after = torch.tensor(after, dtype=torch.long, device=self.device)
            prefix = torch.cat([embedding(before), features[index], embedding(after)], dim=0)
            prefix_lengths.append(prefix.shape[0])
            target_ids = [] if target is None else self._check_answer(target)
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=self.device)
            sequences.append(torch.cat([prefix, embedding(target_tensor)], dim=0))
            labels.append(torch.cat([
                torch.full((prefix.shape[0],), -100, dtype=torch.long, device=self.device),
                target_tensor]))
        max_length = max(sequence.shape[0] for sequence in sequences)
        context_limit = int(getattr(self.language_model.config, "max_position_embeddings", max_length))
        if max_length > context_limit:
            raise ValueError("Packed sequence exceeds context limit; targets are never silently truncated")
        inputs_embeds = torch.stack([F.pad(sequence, (0, 0, 0, max_length-sequence.shape[0]))
                                     for sequence in sequences])
        padded_labels = torch.stack([F.pad(label, (0, max_length-label.shape[0]), value=-100)
                                     for label in labels])
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=self.device)
        positions = torch.arange(max_length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = (positions < lengths.unsqueeze(1)).long()
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                "position_ids": positions, "labels": padded_labels,
                "prefix_lengths": torch.tensor(prefix_lengths, device=self.device),
                "categories": categories, "backend": self.backend}

    def forward_batch(self, pack):
        out = self.language_model(
            inputs_embeds=pack["inputs_embeds"], attention_mask=pack["attention_mask"],
            position_ids=pack["position_ids"], use_cache=False, return_dict=True)
        return self.lm_head(out.last_hidden_state)

    def loss(self, pack):
        labels = pack["labels"][:, 1:]
        if not (labels != -100).any():
            raise ValueError("loss requires answer targets")
        logits = self.forward_batch(pack)[:, :-1, :]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               labels.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def score(self, pack):
        logits = self.forward_batch(pack)
        rows = torch.arange(logits.shape[0], device=logits.device)
        first = logits[rows, pack["prefix_lengths"] - 1].float()
        return first[:, self.yes_id] - first[:, self.no_id]

    @torch.no_grad()
    def generate(self, pack, max_new_tokens=32):
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be positive")
        outputs = []
        embedding = self.language_model.get_input_embeddings()
        # Per-sample cached decoding avoids padding/cache-position ambiguity.
        for index, length in enumerate(pack["prefix_lengths"].tolist()):
            embeds = pack["inputs_embeds"][index:index+1, :length]
            mask = torch.ones((1, length), dtype=torch.long, device=self.device)
            positions = torch.arange(length, device=self.device).unsqueeze(0)
            out = self.language_model(inputs_embeds=embeds, attention_mask=mask,
                                      position_ids=positions, use_cache=True, return_dict=True)
            cache = out.past_key_values
            ids = []
            logits = self.lm_head(out.last_hidden_state[:, -1])
            for step in range(int(max_new_tokens)):
                next_id = int(logits.argmax(dim=-1).item())
                if next_id in self.stop_ids:
                    break
                ids.append(next_id)
                if step + 1 == int(max_new_tokens):
                    break
                if cache is None:
                    raise RuntimeError("Language backend did not return an autoregressive cache")
                token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1)
                position = torch.tensor([[length + step]], dtype=torch.long, device=self.device)
                out = self.language_model(inputs_embeds=embedding(token), attention_mask=mask,
                                          position_ids=position, past_key_values=cache,
                                          use_cache=True, return_dict=True)
                cache = out.past_key_values
                logits = self.lm_head(out.last_hidden_state[:, -1])
            # Keep non-EOS special tokens visible; format failures stay evaluable.
            outputs.append(self.tok.decode(ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False))
            del cache, out
        return outputs

    @torch.no_grad()
    def _audit_initialization(self):
        weight = self.language_model.get_input_embeddings().weight
        sample_ids = torch.linspace(0, weight.shape[0]-1, steps=min(1024, weight.shape[0]),
                                    device=weight.device).long()
        norms = weight.index_select(0, sample_ids).float().norm(dim=-1)
        base = self.prompt_learner.base_prompts.detach().float()
        return {
            "backend": self.backend, "model_path": str(self.model_path),
            "init_seed": self.init_seed, "frozen_dtype": str(self.dtype),
            "bridge_dtype": "torch.float32", "bridge_rescaling": "none",
            "bridge_init": "PyTorch defaults; PromptLearner base prompts N(0,1)",
            "base_prompt_norm_mean": float(base.norm(dim=-1).mean()),
            "vocabulary_sample_norm_mean": float(norms.mean()),
            "vocabulary_sample_norm_std": float(norms.std(unbiased=False)),
            "vocabulary_sample_size": int(norms.numel()),
            "prompt_parameters": sum(p.numel() for p in self.prompt_learner.parameters()),
            "cls_projection_parameters": sum(p.numel() for p in self.cls_proj.parameters()),
            "frozen_language_parameters": sum(p.numel() for p in self.language_model.parameters())
                                          + sum(p.numel() for p in self.lm_head.parameters()),
            "bos_id": self.tok.bos_token_id, "training_eos_id": self.eos_id,
            "generation_stop_ids": sorted(self.stop_ids), "pad_id": self.pad_id,
            "yes_id": self.yes_id, "no_id": self.no_id,
            "position_policy": "text-only monotonic positions; Qwen uses equal rotary axes",
            "generation_policy": "greedy; raw answer; EOS omitted; invalid and empty answers retained",
        }
