"""Stage two: native frozen BF16 Qwen3-VL, with optional learned map tokens.

Run only after the independent GPU parity gate has passed. This script never
loads ImageBind: it consumes the common repaired-teacher FP32 map cache and
uses the official native visual tower, merger, three DeepStack streams and
MRoPE through migration_native_parity's audited API. A fresh, original
PromptLearner(1,4096) produces exactly 18 map tokens after vision_end.

The comparison is native Qwen versus native Qwen plus map tokens. It measures
the map increment under the supervised custom split. It cannot identify a
native-vision effect by comparison with the preceding language-only stage.

Examples (the caller decides when GPU work may start):
  python migration_native_train.py --parity-report runs/native_parity/report.json \
      --baseline --out runs/native_baseline
  python migration_native_train.py --parity-report runs/native_parity/report.json \
      --smoke --out runs/native_map_smoke
  python migration_native_train.py --parity-report runs/native_parity/report.json \
      --out runs/native_map_s42

Microbatch size is one because the audited native API accepts one unpadded
still image. Default 1500 updates x accumulation4 gives 6000 image exposures.
Full runs select on all 380 validation images every 250 updates and generate
greedy answers for all 380 after selecting best. Test prediction/selection is
not implemented. --smoke defaults to 4 updates, validation24, generation8.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from migration_language import DESCRIPTIONS, NO, YES, _prompt_learner_class
from migration_metrics import donor_map_indices, evaluate_predictions, spatial_shuffle_maps
from migration_native_parity import (
    DEFAULT_MODEL, PIPELINE_VERSION as PARITY_VERSION, atomic_json, audit_dtypes,
    manual_decode_step, manual_prefill, prepare_native_inputs, source_audit,
)
from visa_pipeline import PIPELINE_VERSION as CACHE_VERSION, load_image_pil


PIPELINE_VERSION = "qwen3vl-native-map-increment-v1"
DEFAULT_CACHE = "/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/teacher_cache.pt"
HIDDEN_SIZE, AUXILIARY_TOKENS = 4096, 18
EXPECTED_COUNTS = {"train": 9985, "val": 380, "test": 456}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def check_parity_report(report, current):
    """Pure gate validation; a source-only report can never authorize loading."""
    if (report.get("pipeline_version") != PARITY_VERSION or report.get("mode") != "run"
            or report.get("passed") is not True
            or report.get("all_cases_exercised_cached_decoding") is not True):
        raise ValueError("A passed GPU native parity report with cached decoding is required")
    bridge_path = Path(inspect.getsourcefile(prepare_native_inputs))
    if report.get("bridge_script_sha256") != sha256_file(bridge_path):
        raise ValueError("GPU parity report must match the current native bridge helper script")
    old = report.get("source_audit", {})
    if current.get("passed") is not True or old.get("passed") is not True:
        raise ValueError("Both current and reported source audits must pass")
    for key in ("transformers_version", "source_sha256", "model_config", "processor_config", "api"):
        if current.get(key) != old.get(key):
            raise ValueError("Parity report differs from current source/model/processor: " + key)
    if report.get("dtype_audit", {}).get("passed") is not True:
        raise ValueError("GPU report must pass the complete BF16 dtype audit")
    if report.get("runtime", {}).get("attention_backend") != "sdpa":
        raise ValueError("Parity report must use the same SDPA backend")
    if report.get("runtime", {}).get("torch_version") != torch.__version__:
        raise ValueError("Run GPU parity again after changing the PyTorch runtime")
    cases = report.get("cases", [])
    if not cases or not any(case.get("requested_image_wh") == [224, 224] for case in cases):
        raise ValueError("GPU parity must include the 224x224 input case")
    for case in cases:
        greedy = case.get("greedy", {})
        if (case.get("passed") is not True or greedy.get("token_ids_equal") is not True
                or len(greedy.get("step_logits", [])) < 2
                or not all(step.get("passed") is True for step in greedy["step_logits"])):
            raise ValueError("Every parity case must pass and exercise a cached decode step")
    return True


def load_cache(path):
    """Validate provenance and train/val maps; no test-map numerical inspection."""
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    digest = sha256_file(path)
    if digest != sidecar.get("cache_sha256"):
        raise ValueError("Common teacher cache file hash does not match its sidecar")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (cache.get("pipeline_version") != CACHE_VERSION
            or cache.get("training_protocol") != "supervised_custom_60_20_20"
            or cache.get("geometry") != "square224" or cache.get("dtype") != "float32"):
        raise ValueError("Expected the common supervised square224 FP32 teacher cache")
    items, maps = cache["items"], cache["maps"]
    if not isinstance(maps, torch.Tensor) or maps.dtype != torch.float32:
        raise ValueError("Cached maps must remain torch.float32")
    if maps.shape != (len(items), 224, 224):
        raise ValueError("Expected cached maps [N,224,224]")
    if cache.get("manifest_sha256") != json_hash(items):
        raise ValueError("Common cache manifest hash does not match its items")
    if not isinstance(cache.get("decoder_sha256"), str) or len(cache["decoder_sha256"]) != 64:
        raise ValueError("Cache must identify its frozen repaired teacher")
    paths = []
    for item in items:
        if (not all(key in item for key in ("path", "category", "label", "split"))
                or not item["path"] or item["category"] not in DESCRIPTIONS
                or item["label"] not in (0, 1) or item["split"] not in EXPECTED_COUNTS):
            raise ValueError("Invalid cache manifest item")
        paths.append(item["path"])
    if len(set(paths)) != len(paths):
        raise ValueError("Cache paths must be unique across splits")
    indices = {split: [i for i, item in enumerate(items) if item["split"] == split]
               for split in EXPECTED_COUNTS}
    counts = {split: len(ids) for split, ids in indices.items()}
    if counts != EXPECTED_COUNTS or cache.get("counts") != counts:
        raise ValueError("Expected the fixed train9985/val380/test456 manifest: " + str(counts))
    for split in ("train", "val"):
        for offset in range(0, len(indices[split]), 128):
            values = maps[indices[split][offset:offset + 128]]
            if not bool(torch.isfinite(values).all()) or bool((values < 0).any() | (values > 1).any()):
                raise ValueError("Train/validation maps must be finite probabilities in [0,1]")
    # CLS features are unused in this experiment and never copied to the GPU.
    cache.pop("cls", None)
    cache["cache_sha256"] = digest
    return cache, indices


def sample_schedule(items, seed, exposures):
    pools = [[i for i, item in enumerate(items)
              if item["split"] == "train" and item["label"] == label] for label in (0, 1)]
    if any(not pool for pool in pools):
        raise ValueError("Balanced sampling requires both training labels")
    rng = random.Random(seed)
    labels = [0] * (exposures // 2) + [1] * (exposures - exposures // 2)
    rng.shuffle(labels)
    return [rng.choice(pools[label]) for label in labels]


def fixed_subset(ids, items, limit, seed=42):
    """Smoke subsets cover category x label pairs before taking second items."""
    if not limit or limit >= len(ids):
        return list(ids)
    rng = random.Random(seed)
    buckets = []
    for category in sorted({items[i]["category"] for i in ids}):
        for label in (0, 1):
            bucket = [i for i in ids if items[i]["category"] == category and items[i]["label"] == label]
            rng.shuffle(bucket)
            buckets.append(bucket)
    chosen = []
    while len(chosen) < limit:
        for bucket in buckets:
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop())
    return sorted(chosen)


def answer_labels(prepared, prefix_length, target_ids):
    """Mask image, auxiliary and question tokens; supervise only answer + EOS."""
    length = prepared.shadow_input_ids.shape[1]
    auxiliary_n = int(prepared.auxiliary_mask.sum())
    if length != prefix_length + auxiliary_n + len(target_ids):
        raise AssertionError("Expanded answer and prefix lengths disagree")
    if (prepared.position_ids.shape != (3, 1, length)
            or prepared.attention_mask.shape != (1, length)
            or prepared.cache_position.shape != (length,)):
        raise AssertionError("Labels, MRoPE, attention and cache positions must share a length")
    target = prepared.shadow_input_ids.new_tensor(target_ids).view(1, -1)
    if not torch.equal(prepared.shadow_input_ids[:, -len(target_ids):], target):
        raise AssertionError("The canonical answer suffix changed during auxiliary insertion")
    labels = torch.full_like(prepared.shadow_input_ids, -100)
    labels[:, -len(target_ids):] = target
    if bool((labels[prepared.auxiliary_mask | prepared.visual_pos_masks] != -100).any()):
        raise AssertionError("Visual and auxiliary tokens must never receive answer labels")
    return labels


class NativeMapBridge(nn.Module):
    def __init__(self, model_path, device, seed, baseline=False):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = torch.device(device)
        self.baseline = bool(baseline)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        with torch.random.fork_rng(devices=[]):
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map={"": str(self.device)},
                low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
            ).eval()
        self.qwen.requires_grad_(False)
        if int(self.qwen.config.text_config.hidden_size) != HIDDEN_SIZE:
            raise ValueError("This experiment requires official Qwen hidden size4096")
        self.dtype_audit = audit_dtypes(self.qwen)
        if not self.dtype_audit["passed"] or getattr(self.qwen, "is_quantized", False):
            raise ValueError("The entire frozen Qwen, visual/merger/DeepStack included, must be BF16")
        self.prompt_learner = None
        self.prompt_class_path = None
        if not self.baseline:
            cls = _prompt_learner_class()
            # The standalone importer intentionally avoids model/__init__ and
            # does not register the temporary module in sys.modules. Inspect
            # the forward function, whose code object still records its path.
            self.prompt_class_path = Path(inspect.getsourcefile(cls.forward))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.prompt_learner = cls(1, HIDDEN_SIZE).float()
            self.prompt_learner.to(self.device)
        self.yes_id, self.no_id = self._single_token("Yes"), self._single_token("No")
        if self.yes_id == self.no_id:
            raise AssertionError("Yes and No must have different exact first tokens")
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("Tokenizer must supply an answer EOS")
        self.answers = {}
        for label, answer in ((0, NO), (1, YES)):
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if not ids or ids[0] != (self.yes_id if label else self.no_id):
                raise ValueError("Canonical answer must start with the exact Yes/No token")
            self.answers[label] = ids + [int(self.eos_id)]
        # Check tokenization in the actual rendered assistant context as well.
        self._templates = {}
        self.placeholder_id = self._single_token("feature")
        special = set(self.tokenizer.all_special_ids)
        special.update(int(getattr(self.qwen.config, name)) for name in
                       ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"))
        if self.placeholder_id in special:
            raise ValueError("Auxiliary positional placeholder must be an ordinary text token")
        stops = self.qwen.generation_config.eos_token_id
        self.stop_ids = {int(self.eos_id)}
        if stops is not None:
            self.stop_ids.update(map(int, stops if isinstance(stops, list) else [stops]))
        self.geometry_audit = None
        self.initialization_audit = self._audit_initialization(seed)

    def train(self, mode=True):
        super().train(mode)
        self.qwen.eval()
        return self

    def _single_token(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("Exact one-token candidate required for %r: %r" % (text, ids))
        return int(ids[0])

    def _template(self, category):
        if category not in self._templates:
            question = ("This is a photo of " + DESCRIPTIONS[category]
                        + " for anomaly detection. Is there any anomaly in the image?")
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
            for label, answer in ((0, NO), (1, YES)):
                combined = self.tokenizer.encode(rendered + answer, add_special_tokens=False)
                if combined != prefix + self.answers[label][:-1]:
                    raise ValueError("Canonical answer tokenization changes at the assistant boundary")
            self._templates[category] = rendered
        return self._templates[category]

    def batch(self, item, answer=False):
        image = load_image_pil(item["path"], size=224)
        batch = self.processor(text=[self._template(item["category"])], images=[image],
                               padding=False, return_tensors="pt")
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        prefix_length = batch["input_ids"].shape[1]
        if answer:
            ids = batch["input_ids"].new_tensor(self.answers[int(item["label"])]).view(1, -1)
            batch["input_ids"] = torch.cat([batch["input_ids"], ids], dim=1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(ids)], dim=1)
        if self.geometry_audit is None:
            patch_size = int(self.qwen.config.vision_config.patch_size)
            grid = batch["image_grid_thw"].detach().cpu().tolist()
            self.geometry_audit = {
                "source_path": item["path"], "input_geometry": "direct RGB bicubic square resize",
                "requested_image_wh": [224, 224], "image_grid_thw": grid, "patch_size": patch_size,
                "actual_processor_image_wh": [[g[2] * patch_size, g[1] * patch_size] for g in grid],
                "processor_shapes": {key: list(value.shape) for key, value in batch.items()
                                     if isinstance(value, torch.Tensor)},
                "image_tokens": int((batch["input_ids"] == self.qwen.config.image_token_id).sum()),
                "processor_image_settings": self.processor.image_processor.to_dict(),
            }
        return batch, prefix_length

    def auxiliary(self, amap):
        if self.baseline:
            return None
        if amap.shape != (224, 224) or amap.dtype != torch.float32:
            raise ValueError("Use the unchanged common FP32 224-square map")
        tokens = self.prompt_learner(amap.to(self.device).view(1, 1, 224, 224))
        if tokens.shape != (1, AUXILIARY_TOKENS, HIDDEN_SIZE) or not bool(torch.isfinite(tokens).all()):
            raise ValueError("PromptLearner must return finite [1,18,4096] auxiliary tokens")
        return tokens

    def prepare(self, batch, auxiliary):
        prepared = prepare_native_inputs(self.qwen, batch, auxiliary_tokens=auxiliary,
                                         auxiliary_placeholder_id=self.placeholder_id, freeze_vision=True)
        expected = 0 if self.baseline else AUXILIARY_TOKENS
        if int(prepared.auxiliary_mask.sum()) != expected:
            raise AssertionError("Only the specified 18 map tokens may be inserted")
        length = prepared.shadow_input_ids.shape[1]
        if (prepared.position_ids.shape != (3, 1, length)
                or prepared.attention_mask.shape != (1, length)
                or prepared.cache_position.shape != (length,)
                or len(prepared.deepstack_visual_embeds) != 3):
            raise AssertionError("Full MRoPE, attention, cache positions and three DeepStack streams are required")
        if expected:
            positions = prepared.auxiliary_mask[0].nonzero().flatten()
            end = (prepared.shadow_input_ids[0] == self.qwen.config.vision_end_token_id).nonzero()[-1].item()
            if positions.tolist() != list(range(end + 1, end + 1 + expected)):
                raise AssertionError("Auxiliary tokens must follow the complete vision_end")
        return prepared

    def loss(self, item, amap):
        if self.baseline:
            raise ValueError("Native baseline has no trainable map branch")
        batch, prefix_length = self.batch(item, answer=True)
        auxiliary = self.auxiliary(amap)
        prepared = self.prepare(batch, auxiliary)
        if not prepared.inputs_embeds.requires_grad:
            raise AssertionError("Auxiliary token insertion severed the training gradient")
        labels = answer_labels(prepared, prefix_length, self.answers[int(item["label"])])
        # The frozen language forward stays in autograd. Project only supervised
        # positions, which is the same token-mean causal CE without prefix logits.
        output = self.qwen.model.language_model(**prepared.lm_kwargs(use_cache=False))
        selected = labels[:, 1:] != -100
        logits = self.qwen.lm_head(output.last_hidden_state[:, :-1][selected])
        return F.cross_entropy(logits.float(), labels[:, 1:][selected])

    @torch.no_grad()
    def predict(self, item, amap=None, generate=False, max_new_tokens=32):
        batch, _ = self.batch(item)
        prepared = self.prepare(batch, self.auxiliary(amap))
        output = manual_prefill(self.qwen, prepared, logits_to_keep=1)
        logits = output.logits[:, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite first-answer logits")
        score = float((logits[0, self.yes_id] - logits[0, self.no_id]).item())
        text = None
        if generate:
            tokens = []
            for step in range(max_new_tokens):
                next_id = int(logits.argmax(dim=-1).item())
                if next_id in self.stop_ids:
                    break
                tokens.append(next_id)
                if step + 1 < max_new_tokens:
                    output = manual_decode_step(self.qwen, batch["input_ids"].new_tensor([[next_id]]), output.state)
                    logits = output.logits[:, -1].float()
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("Nonfinite cached generation logits")
            text = self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return score, text

    @torch.no_grad()
    def cache_self_check(self, item, amap, *, report_path=None, context=None):
        """Gate the current branch against the same-shaped official forward.

        The complete official/manual prefill, cached step and full replay must
        each match exactly. Cached-versus-full replay changes attention shape
        and is retained only as a diagnostic. Persist reports before raising.
        """
        # Keep CLI/source inspection independent of model construction. The
        # helper has no trainer import at module initialization time.
        import migration_aux_official_parity as official_parity

        self.eval()
        helper_path = Path(inspect.getsourcefile(inspect.unwrap(official_parity.run_case))).resolve()
        native_path = Path(inspect.getsourcefile(inspect.unwrap(manual_prefill))).resolve()
        required_checks = (
            "prefill_logits", "cached_step_logits", "prefill_and_cached_language_inputs",
            "prefill_rope_delta", "cached_rope_delta", "full_replay_logits",
            "full_replay_language_inputs", "full_replay_shadow_ids", "full_replay_rope_delta",
            "official_full_vs_cached_last_position", "cache_lengths",
            "cached_step_has_no_visual_inputs",
        )
        report = {
            "passed": False, "status": "running",
            "scope": "current branch official/manual exact same-shape parity",
            "gate_definition": "finite, dtype-equal, exact same-shaped official/manual prefill, cached-step and full-replay logits and language inputs; exact positions/masks/cache lengths",
            "diagnostic_definition": "official and manual cached-versus-full replay differences are diagnostic only; no numerical tolerance or argmax gate",
            "required_exact_checks": list(required_checks),
            "expected_auxiliary_tokens": 0 if self.baseline else AUXILIARY_TOKENS,
            "item": {key: item[key] for key in ("path", "category", "split", "label")},
            "prompt_state_sha256": None if self.baseline else state_hash(self.prompt_learner),
            "input_map": None if amap is None else {
                "shape": list(amap.shape), "dtype": str(amap.dtype),
                "sha256": hashlib.sha256(amap.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            },
            "provenance": {
                "helper_pipeline_version": official_parity.PIPELINE_VERSION,
                "helper_path": str(helper_path), "helper_sha256": sha256_file(helper_path),
                "native_helper_path": str(native_path), "native_helper_sha256": sha256_file(native_path),
                "trainer_sha256": sha256_file(__file__), "torch_version": torch.__version__,
                "decode_token_choice": "yes", "attention_backend": "sdpa",
            },
            "context": dict(context or {}),
        }
        if report_path is not None:
            official_parity.save_report(report, report_path)
        try:
            result = official_parity.run_case(self, item, amap, decode_choice="yes")
            report.update(result)
            checks = result.get("checks", {})
            failed = [name for name in required_checks if checks.get(name, {}).get("passed") is not True]
            failed.extend(name for name, check in checks.items()
                          if name not in required_checks and check.get("passed") is not True)
            if result.get("auxiliary_tokens") != report["expected_auxiliary_tokens"]:
                failed.append("auxiliary_token_count")
            if result.get("baseline") is not self.baseline:
                failed.append("baseline_branch")
            if result.get("passed") is not True and not failed:
                failed.append("helper_exact_gate")
            report["failed_checks"] = failed
            report["passed"] = result.get("passed") is True and not failed
            report["status"] = "complete" if report["passed"] else "failed"
            if not report["passed"]:
                raise RuntimeError("Official/manual exact same-shape cache gate failed: " + ", ".join(failed))
        except Exception as error:
            report.update(passed=False, status="failed",
                          error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            if report_path is not None:
                official_parity.save_report(report, report_path)
        return official_parity.finite_json(report)

    @torch.no_grad()
    def _audit_initialization(self, seed):
        weight = self.qwen.get_input_embeddings().weight
        ids = torch.linspace(0, weight.shape[0] - 1, min(1024, weight.shape[0]), device=weight.device).long()
        norms = weight.index_select(0, ids).float().norm(dim=-1)
        result = {
            "init_seed": seed, "frozen_dtype": "torch.bfloat16", "bridge_dtype": "torch.float32",
            "initialization": ("official pretrained frozen native Qwen" if self.baseline else
                               "fresh original PromptLearner defaults; base prompts N(0,1)"),
            "auxiliary_tokens": 0 if self.baseline else AUXILIARY_TOKENS,
            "auxiliary_position": "immediately after vision_end; official recomputed MRoPE",
            "cls_tokens": 0, "bridge_rescaling": "none", "vision_deepstack": "official native BF16",
            "vocabulary_sample_norm_mean": float(norms.mean()),
            "vocabulary_sample_norm_std": float(norms.std(unbiased=False)),
            "yes_id": self.yes_id, "no_id": self.no_id, "training_eos_id": int(self.eos_id),
            "generation_stop_ids": sorted(self.stop_ids),
            "canonical_answer_ids": {str(label): ids for label, ids in self.answers.items()},
        }
        if self.prompt_learner is not None:
            result["base_prompt_norm_mean"] = float(self.prompt_learner.base_prompts.float().norm(dim=-1).mean())
            result["trainable_parameters"] = sum(p.numel() for p in self.prompt_learner.parameters())
            result["initialization_sha256"] = state_hash(self.prompt_learner)
        else:
            result["trainable_parameters"] = 0
        return result


def source_snapshot(out, model):
    root = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), root / "migration_native_parity.py", root / "migration_metrics.py",
             root / "migration_language.py", root / "visa_pipeline.py",
             root / "migration_aux_official_parity.py"]
    if model.prompt_class_path is not None:
        paths.append(model.prompt_class_path)
    directory = out / "code"
    directory.mkdir()
    result = {}
    for path in paths:
        result[path.name] = {"source": str(path), "sha256": sha256_file(path)}
        shutil.copyfile(path, directory / path.name)
    return result


def model_manifest(model_path):
    root = Path(model_path)
    small = {path.name: sha256_file(path) for path in sorted(root.iterdir())
             if path.is_file() and path.suffix in {".json", ".jinja", ".model"}
             and path.stat().st_size < 32 * 1024 * 1024}
    weights = [{"name": path.name, "bytes": path.stat().st_size} for path in sorted(root.iterdir())
               if path.is_file() and path.suffix in {".safetensors", ".bin"}]
    return {"path": str(root.resolve()), "config_tokenizer_file_sha256": small, "weight_files": weights}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--parity-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--gen-limit", "--generate-limit", dest="gen_limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--baseline", action="store_true", help="Native Qwen without auxiliary tokens or training")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 4 if args.smoke else 1500
    if args.val_limit is None:
        args.val_limit = 24 if args.smoke else 0
    if args.gen_limit is None:
        args.gen_limit = 8 if args.smoke else 0
    if min(args.steps, args.accum, args.val_every, args.max_new_tokens) < 1 or args.lr <= 0:
        parser.error("Steps, accumulation, intervals, token limit and learning rate must be positive")
    if min(args.val_limit, args.gen_limit) < 0:
        parser.error("Limits must be nonnegative")
    if args.val_limit == 1 or args.gen_limit == 1:
        parser.error("Smoke subsets must include at least two images for binary evaluation")
    if not args.smoke and (args.val_limit or args.gen_limit):
        parser.error("Subset validation/generation is allowed only with --smoke; full runs use all380")
    if args.eval_only and not args.ckpt:
        parser.error("--eval-only requires --ckpt")
    if args.ckpt and not args.eval_only:
        parser.error("Checkpoint loading requires --eval-only; training always initializes fresh")
    if not args.device.startswith("cuda"):
        parser.error("The full BF16 native experiment requires CUDA")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Read both gate inputs before any CUDA allocation or checkpoint construction.
    parity_path = Path(args.parity_report)
    report = json.loads(parity_path.read_text())
    current = source_audit(args.model)
    check_parity_report(report, current)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh output directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    gate = {"passed": True, "report_path": str(parity_path.resolve()),
            "report_sha256": sha256_file(parity_path), "current_source_audit": current,
            "bridge_script_sha256": report["bridge_script_sha256"]}
    atomic_json(gate, out / "parity_gate.json")
    cache, indices = load_cache(args.cache)
    items = cache["items"]
    val_ids = fixed_subset(indices["val"], items, args.val_limit)
    if len({items[i]["label"] for i in val_ids}) != 2:
        raise ValueError("Validation must contain both binary labels")
    schedule = [] if args.baseline or args.eval_only else sample_schedule(items, args.seed, args.steps * args.accum)
    atomic_json(schedule, out / "sample_indices.json")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.time()
    model = NativeMapBridge(args.model, args.device, args.seed, args.baseline)
    params = [] if model.baseline else list(model.prompt_learner.parameters())
    if any(p.requires_grad for p in model.qwen.parameters()):
        raise AssertionError("The entire official Qwen must remain frozen")
    model.eval()
    check_id = val_ids[0]
    cache_check = model.cache_self_check(
        items[check_id], None if args.baseline else cache["maps"][check_id],
        report_path=out / "auxiliary_cache_check.json",
        context={"phase": "before_training_or_checkpoint_load", "cache_index": check_id,
                 "cache_sha256": cache["cache_sha256"], "decoder_sha256": cache["decoder_sha256"],
                 "cache_manifest_sha256": cache["manifest_sha256"]},
    )
    sources = source_snapshot(out, model)
    model_files = model_manifest(args.model)
    config = {
        "pipeline_version": PIPELINE_VERSION, "parity_pipeline_version": PARITY_VERSION,
        "cache_pipeline_version": CACHE_VERSION, "training_protocol": "supervised_custom_60_20_20",
        "experiment": "native Qwen baseline" if args.baseline else "native Qwen plus18 learned map tokens",
        "estimand": "increment of cached anomaly-map auxiliary branch within native Qwen",
        "model": model_files, "model_manifest_sha256": json_hash(model_files),
        "decoder_sha256": cache["decoder_sha256"], "cache_sha256": cache["cache_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "counts": {split: len(ids) for split, ids in indices.items()}, "val_n": len(val_ids),
        "validation_ids_sha256": json_hash(val_ids), "selection_split": "val",
        "selection": "maximum validation image AUROC after updates; earliest wins ties",
        "smoke": args.smoke, "microbatch_size": 1, "accumulation": args.accum,
        "training_exposures": len(schedule), "sample_schedule_sha256": json_hash(schedule),
        "precision": "entire frozen native Qwen BF16; PromptLearner/AdamW/cache maps FP32",
        "trainable_parameters": sum(p.numel() for p in params), "auxiliary_tokens": 0 if args.baseline else 18,
        "cls_tokens": 0, "loss": "canonical answer and EOS token-mean causal cross entropy",
        "initialization_audit": model.initialization_audit, "dtype_audit": model.dtype_audit,
        "geometry_audit": model.geometry_audit, "auxiliary_cache_check": cache_check,
        "parity_report_sha256": gate["report_sha256"], "source_sha256": current["source_sha256"],
        "parity_bridge_script_sha256": gate["bridge_script_sha256"],
        "auxiliary_parity_pipeline_version": cache_check["provenance"]["helper_pipeline_version"],
        "auxiliary_parity_helper_sha256": cache_check["provenance"]["helper_sha256"],
        "code": sources, "config": vars(args),
    }
    atomic_json(config, out / "run.json")
    print("[config]", json.dumps(config, ensure_ascii=False), flush=True)

    full_val_items = [items[i] for i in indices["val"]]
    donor_local = donor_map_indices(full_val_items, seed=42, opposite_label=True)
    donor_by_id = {index: indices["val"][int(donor)] for index, donor in zip(indices["val"], donor_local)}

    @torch.no_grad()
    def evaluate(ids, condition="real_map", generate=False):
        model.eval()
        if any(items[i]["split"] != "val" for i in ids):
            raise AssertionError("Only validation predictions are implemented")
        cmaps = None
        if not args.baseline:
            if condition == "donor_opposite_label_map":
                cmaps = cache["maps"][[donor_by_id[i] for i in ids]]
            elif condition == "zero_map":
                cmaps = torch.zeros((len(ids), 224, 224), dtype=torch.float32)
            elif condition == "spatial_shuffle_map":
                cmaps = spatial_shuffle_maps(cache["maps"][ids], seed=42)
            elif condition == "real_map":
                cmaps = cache["maps"][ids]
            else:
                raise ValueError(condition)
        elif condition != "real_map":
            raise ValueError("Native baseline has no map interventions")
        scores, texts, records = [], [], []
        for position, index in enumerate(ids):
            score, text = model.predict(items[index], None if cmaps is None else cmaps[position],
                                        generate=generate, max_new_tokens=args.max_new_tokens)
            scores.append(score)
            record = dict(items[index], score=score)
            if condition == "donor_opposite_label_map":
                record["donor_path"] = items[donor_by_id[index]]["path"]
                record["donor_label"] = items[donor_by_id[index]]["label"]
            if generate:
                texts.append(text)
                record["generated"] = text
            records.append(record)
            if (position + 1) % 50 == 0:
                print("[eval progress]", condition, position + 1, "/", len(ids), flush=True)
        metrics = evaluate_predictions([items[i] for i in ids], scores, texts if generate else None)
        return metrics, records

    def checkpoint(step, val_auroc):
        state = None if model.baseline else {name: value.detach().cpu() for name, value
                                            in model.prompt_learner.state_dict().items()}
        return {"prompt_learner": state, "step": step, "val_auroc": val_auroc, "run": config}

    def load_checkpoint(path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state["run"]
        for key in ("pipeline_version", "cache_pipeline_version", "decoder_sha256", "cache_sha256",
                    "cache_manifest_sha256", "model_manifest_sha256", "auxiliary_tokens", "cls_tokens",
                    "source_sha256", "parity_bridge_script_sha256"):
            if saved.get(key) != config[key]:
                raise ValueError("Checkpoint provenance differs: " + key)
        if model.baseline:
            if state["prompt_learner"] is not None:
                raise ValueError("Baseline cannot load a learned auxiliary branch")
        else:
            if saved.get("code", {}).get(model.prompt_class_path.name, {}).get("sha256") != sources[model.prompt_class_path.name]["sha256"]:
                raise ValueError("PromptLearner definition differs from the checkpoint")
            model.prompt_learner.load_state_dict(state["prompt_learner"], strict=True)
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise ValueError("Nonfinite PromptLearner checkpoint")
        return state

    selected, selected_predictions, selected_real = None, None, None
    if args.eval_only:
        selected = load_checkpoint(args.ckpt)
    elif args.baseline:
        metrics, selected_predictions = evaluate(val_ids)
        selected = checkpoint(0, metrics["overall"]["scores"]["image_auroc"])
        atomic_json(metrics, out / "initial_val.json")
        atomic_json(selected_predictions, out / "best_val_predictions.json")
        atomic_save(selected, out / "best.pt")
        atomic_save(selected, out / "final.pt")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        if args.steps < 40:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
            scheduler_name = "CosineAnnealingLR (short smoke schedule)"
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                            total_steps=args.steps, pct_start=0.05)
            scheduler_name = "OneCycleLR pct_start0.05"
        config["scheduler"] = scheduler_name
        atomic_json(config, out / "run.json")
        initial, _ = evaluate(val_ids)
        atomic_json(initial, out / "initial_val.json")
        best_auc, best_step, last_auc = -1.0, None, None
        with (out / "training.jsonl").open("w") as history:
            for step in range(1, args.steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_total = 0.0
                for offset in range(args.accum):
                    index = schedule[(step - 1) * args.accum + offset]
                    loss = model.loss(items[index], cache["maps"][index])
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite answer loss at update%d" % step)
                    (loss / args.accum).backward()
                    loss_total += float(loss.detach()) / args.accum
                    del loss
                if any(p.grad is not None for p in model.qwen.parameters()):
                    raise AssertionError("Frozen native Qwen unexpectedly accumulated gradients")
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                if step == 1:
                    base = model.prompt_learner.base_prompts
                    base_norm = 0.0 if base.grad is None else float(base.grad.float().norm())
                    adaptive = sum(float(p.grad.float().norm()) for name, p in model.prompt_learner.named_parameters()
                                   if name != "base_prompts" and p.grad is not None)
                    if min(base_norm, adaptive) <= 0 or not np.isfinite([base_norm, adaptive]).all():
                        raise ValueError("Both base and adaptive PromptLearner branches require finite nonzero gradients")
                    atomic_json({"base_prompt_gradient_norm": base_norm, "adaptive_gradient_norm_sum": adaptive,
                                 "frozen_qwen_gradients": 0, "labels_positions_checked": True},
                                out / "gradient_check.json")
                optimizer.step()
                scheduler.step()
                if not all(bool(torch.isfinite(p).all()) for p in params):
                    raise ValueError("Nonfinite PromptLearner parameter at update%d" % step)
                record = {"step": step, "loss": loss_total, "gradient_norm": float(norm),
                          "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.time() - started,
                          "peak_memory_gb": torch.cuda.max_memory_allocated(args.device) / 1e9}
                history.write(json.dumps(record) + "\n")
                history.flush()
                if step == 1 or step % 10 == 0:
                    print("[step]", json.dumps(record), flush=True)
                if step % args.val_every == 0 or step == args.steps:
                    metrics, predictions = evaluate(val_ids)
                    last_auc = metrics["overall"]["scores"]["image_auroc"]
                    if last_auc is None or not np.isfinite(last_auc):
                        raise ValueError("Checkpoint selection requires finite binary validation AUROC")
                    atomic_json(metrics, out / ("val_step_%d.json" % step))
                    history.write(json.dumps({"step": step, "validation": metrics}) + "\n")
                    history.flush()
                    print("[validation]", step, json.dumps(metrics["overall"]), flush=True)
                    if last_auc > best_auc:
                        best_auc, best_step = last_auc, step
                        selected_predictions = predictions
                        atomic_save(checkpoint(step, best_auc), out / "best.pt")
                        atomic_json(predictions, out / "best_val_predictions.json")
        final = checkpoint(args.steps, last_auc)
        final.update(best_step=best_step, best_val_auroc=best_auc)
        atomic_save(final, out / "final.pt")
        del optimizer, scheduler, final
        torch.cuda.empty_cache()

    if not args.eval_only:
        # A real reload is required even for baseline metadata-only checkpoints.
        selected = load_checkpoint(out / "best.pt")
        reloaded_metrics, reloaded = evaluate(val_ids)
        selected_real = (reloaded_metrics, reloaded)
        if [x["path"] for x in reloaded] != [x["path"] for x in selected_predictions]:
            raise AssertionError("Save/reload changed validation ordering")
        error = max(abs(a["score"] - b["score"]) for a, b in zip(reloaded, selected_predictions))
        if error > 1e-5:
            raise AssertionError("Save/reload validation scores disagree: %.8f" % error)
    # Check the branch that will actually produce the final controls/generation:
    # best after training, or the supplied learned checkpoint for eval-only.
    model.eval()
    selected_path = Path(args.ckpt) if args.eval_only else out / "best.pt"
    selected_context = {
        "phase": "selected_checkpoint_reloaded", "cache_index": check_id,
        "cache_sha256": cache["cache_sha256"], "decoder_sha256": cache["decoder_sha256"],
        "cache_manifest_sha256": cache["manifest_sha256"],
        "selected_step": selected["step"], "checkpoint": str(selected_path.resolve()),
        "checkpoint_sha256": sha256_file(selected_path),
    }
    selected_cache_check = model.cache_self_check(
        items[check_id], None if args.baseline else cache["maps"][check_id],
        report_path=out / "selected_auxiliary_cache_check.json", context=selected_context,
    )
    if not args.eval_only:
        atomic_json({"selected_step": selected["step"], "selected_val_auroc": selected["val_auroc"],
                     "reload_max_abs_score_error": error, "elapsed_seconds": time.time() - started,
                     "selected_auxiliary_cache_check_passed": selected_cache_check["passed"],
                     "peak_memory_gb": torch.cuda.max_memory_allocated(args.device) / 1e9},
                    out / "train_complete.json")
    result = {"split": "val", "selected_step": selected["step"], "selected_val_auroc": selected["val_auroc"],
              "smoke": args.smoke, "val_n": len(val_ids), "full_validation_n": len(indices["val"]),
              "conditions": {}, "score": "first-answer logit(Yes)-logit(No), threshold0 predictsNo",
              "selected_auxiliary_cache_check": selected_cache_check,
              "decoder_sha256": cache["decoder_sha256"], "cache_sha256": cache["cache_sha256"],
              "donor_control": "same-category opposite-label donor from validation pool; diagnostic uses val labels"}
    conditions = ("real_map",) if args.baseline else (
        "real_map", "zero_map", "donor_opposite_label_map", "spatial_shuffle_map")
    for condition in conditions:
        metrics, predictions = (selected_real if condition == "real_map" and selected_real is not None
                                else evaluate(val_ids, condition))
        result["conditions"][condition] = metrics
        atomic_json(predictions, out / ("val_%s_predictions.json" % condition))
        atomic_json(result, out / "val_metrics.json")
        print("[evaluation]", condition, json.dumps(metrics["overall"]), flush=True)
    gen_ids = fixed_subset(val_ids, items, args.gen_limit)
    metrics, predictions = evaluate(gen_ids, generate=True)
    result.update(generated_real_map=metrics, generated_n=len(gen_ids), generated_is_subset=len(gen_ids) != len(val_ids),
                  generation="greedy with complete explicit MRoPE/cache; EOS omitted, invalid answers retained")
    atomic_json(predictions, out / "val_generated_predictions.json")
    atomic_json(result, out / "val_metrics.json")
    atomic_json({"passed": True, "split": "val", "selected_step": selected["step"],
                 "elapsed_seconds": time.time() - started,
                 "peak_memory_gb": torch.cuda.max_memory_allocated(args.device) / 1e9}, out / "complete.json")
    print("[done]", json.dumps(metrics["overall"]), flush=True)


if __name__ == "__main__":
    main()
