"""Native Qwen3-VL inputs and single-GPU, language-only NF4 QLoRA.

No custom visual tokens, grids, positions, or coordinate conversion are used.
Coordinates are integer [0, 1000] values supplied by the dataset. Importing this
module does not load a model or contact a model host.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

import torch


LANGUAGE_ATTENTION = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
VISUAL_WORDS = {"visual", "vision_tower", "vision_model", "merger", "deepstack_merger_list"}


def _is_visual(name: str) -> bool:
    return bool(VISUAL_WORDS.intersection(name.split(".")))


def _is_language(name: str) -> bool:
    return "language_model" in name.split(".") and not _is_visual(name)


def input_device(model: Any) -> torch.device:
    """The loader places the entire model on one CUDA device; no CPU offload."""
    return model.get_input_embeddings().weight.device


def move_inputs(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    return {
        key: value.to(input_device(model)) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def audit_trainable_parameters(model: Any, *, require_lora: bool = True) -> dict[str, Any]:
    """Fail closed if anything outside language-attention LoRA is trainable."""
    trainable = []
    total = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        # PEFT inserts .lora_A.default.weight / .lora_B.default.weight.
        parent = re.split(r"\.lora_[AB]\.", name, maxsplit=1)[0]
        if _is_visual(name) or not LANGUAGE_ATTENTION.search(parent) or ".lora_" not in name:
            raise RuntimeError(f"Forbidden trainable parameter: {name}")
        trainable.append({"name": name, "numel": parameter.numel(), "dtype": str(parameter.dtype)})
    if require_lora and not trainable:
        raise RuntimeError("No trainable language-attention LoRA parameters were found")
    return {
        "trainable_parameters": sum(item["numel"] for item in trainable),
        # 4-bit tensors have packed storage; this is intentionally not a claim
        # about original dense-model parameter count or total GPU memory.
        "stored_parameter_elements": total,
        "trainable_tensors": trainable,
    }


def set_training_mode(model: Any) -> None:
    model.train()
    for name, module in model.named_modules():
        if name.split(".")[-1] in {"visual", "vision_tower", "vision_model"}:
            module.eval()


def load_model(
    model_path: str | Path,
    adapter: str | Path | None = None,
    training: bool = False,
    max_pixels: int = 262144,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Return (model, processor), with identical NF4 loading for base/adapter eval.

    Requires transformers >=4.57, CUDA bf16, bitsandbytes, accelerate, PEFT.
    Local weights are the default: this function does not silently download them.
    Only language-model linear layers are quantized; vision/merger/DeepStack,
    embeddings, and lm_head retain their original bf16 storage.
    """
    from packaging.version import Version
    import transformers

    version = Version(transformers.__version__)
    if version < Version("4.57.0"):
        raise RuntimeError(f"Use transformers>=4.57; found {transformers.__version__}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This loader requires one CUDA GPU with bf16 support")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    from accelerate import init_empty_weights
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"Expected Qwen3-VL dense model, found {config.model_type!r}")

    # Enumerate architecture on meta before loading weights. A global suffix
    # such as q_proj would also select vision projections in some VLMs.
    with init_empty_weights():
        skeleton = Qwen3VLForConditionalGeneration(config)
    target_names = [name for name, _ in skeleton.named_modules() if LANGUAGE_ATTENTION.search(name)]
    expected = int(config.text_config.num_hidden_layers) * 4
    if len(target_names) != expected:
        raise RuntimeError(f"Expected {expected} language q/k/v/o modules, found {len(target_names)}")
    skip_names = [
        name for name, module in skeleton.named_modules()
        if isinstance(module, torch.nn.Linear) and not _is_language(name)
    ]
    skip_names += [name for name, _ in skeleton.named_modules() if _is_visual(name) and name.endswith("visual")]
    del skeleton
    gc.collect()

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Despite its historic name, HF's 4-bit converter also uses this list.
        llm_int8_skip_modules=sorted(set(skip_names + ["lm_head"])),
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    quantized_names = [name for name, module in model.named_modules() if isinstance(module, Linear4bit)]
    if not quantized_names or any(not _is_language(name) for name in quantized_names):
        raise RuntimeError("Quantization audit failed: all 4-bit layers must belong to language_model")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        if _is_visual(name) and parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"Visual parameter is not bf16: {name}: {parameter.dtype}")

    # Deliberately avoid prepare_model_for_kbit_training: its global float32
    # promotion can unnecessarily upcast the complete frozen visual encoder.
    model.config.use_cache = not training
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=training)
    elif training:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_names,
        ))

    if training:
        if gradient_checkpointing:
            # Non-reentrant checkpointing works with frozen inputs. Also enable
            # embedding-output gradients for PEFT/checkpoint compatibility;
            # this does not unfreeze the embedding matrix.
            model.enable_input_require_grads()
            language = model.get_base_model().model.language_model
            language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        audit_trainable_parameters(model)
        set_training_mode(model)
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model._industrial_vlm_audit = {
        "quantized_modules": quantized_names,
        "lora_target_modules": target_names,
        "quantization": "NF4, double quantization, bf16 compute; language linear layers only",
        "visual_dtype": "torch.bfloat16",
        "gradient_checkpointing": bool(training and gradient_checkpointing),
    }

    processor = AutoProcessor.from_pretrained(
        model_path, max_pixels=max_pixels, local_files_only=local_files_only,
    )
    # Recent native Qwen processors express area limits in size; keep that
    # representation consistent with max_pixels if it is present.
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    if isinstance(getattr(image_processor, "size", None), dict) and "longest_edge" in image_processor.size:
        image_processor.size = {**image_processor.size, "longest_edge": max_pixels}
    processor._industrial_max_pixels = max_pixels
    return model, processor


def make_inputs(
    processor: Any,
    image_path: str | Path,
    prompt: str,
    answer: str | None = None,
    max_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """Build single-image native inputs; optionally mask labels to assistant only.

    Full and generation-prefix conversations use exactly the same image. The
    processed prefix token IDs must be an exact prefix of the full token IDs.
    Reject overlong inputs rather than truncate through image tokens or targets.
    Native image_grid_thw and other processor fields are returned unchanged.
    """
    from PIL import Image, ImageOps

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError("answer must be a nonempty string or None")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt},
    ]}]
    prefix_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    with Image.open(image_path) as source:
        # Dataset conversion must derive boxes after the same EXIF transform.
        image = ImageOps.exif_transpose(source).convert("RGB")
    prefix = processor(text=[prefix_text], images=[image], return_tensors="pt", padding=False)
    if answer is None:
        full = prefix
    else:
        full_text = processor.apply_chat_template(
            messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        full = processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
        prefix_ids, ids = prefix["input_ids"], full["input_ids"]
        n_prefix = prefix_ids.shape[-1]
        if ids.shape[-1] <= n_prefix or not torch.equal(ids[:, :n_prefix], prefix_ids):
            raise ValueError("Native chat-template prefix mismatch: refusing shifted assistant labels")
        if "image_grid_thw" not in full or not torch.equal(full["image_grid_thw"], prefix["image_grid_thw"]):
            raise ValueError("Native image grids changed between prompt and answer encoding")
        labels = ids.clone()
        labels[:, :n_prefix] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        if not (labels[:, 1:] != -100).any():
            raise ValueError("No assistant target tokens remain")
    if full["input_ids"].shape[-1] > max_length:
        raise ValueError(
            f"Input has {full['input_ids'].shape[-1]} tokens, exceeding max_length={max_length}; "
            "reduce max_pixels or prompt length; no automatic truncation is performed"
        )
    if "image_grid_thw" not in full:
        raise ValueError("Expected native image_grid_thw for exactly one image")
    if full["image_grid_thw"].shape[0] != 1:
        raise ValueError("This first version supports exactly one image per sample")
    return dict(full)
