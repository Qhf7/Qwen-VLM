"""单图 forward/backward 验证 E1 只有 decoder 与 prompt learner 获得梯度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from b0.b0plus_protocol import P0
    from b0.single_view import candidate_ids
    from e1.model import QwenE1
except ModuleNotFoundError:
    from model import QwenE1
    from b0plus_protocol import P0
    from run_dev720 import candidate_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/e1/e1_smoke.json"))
    args = parser.parse_args()
    seed = 20260901
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=448 * 448, max_pixels=448 * 448
    )
    processor.tokenizer.padding_side = "left"
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    conversations = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": P0},
        ],
    }]]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    device = torch.device("cuda:0")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    model = QwenE1(
        base,
        yes_token_ids=candidate_ids(processor.tokenizer, "Yes"),
        no_token_ids=candidate_ids(processor.tokenizer, "No"),
    ).to(device).train()
    labels = torch.zeros(1, device=device)
    masks = torch.zeros(1, 1, 448, 448, device=device)
    regions = torch.full((1,), 9, device=device, dtype=torch.long)
    # 冻结基座是 FP16，新增可训参数保持 FP32；训练与冒烟都走 mixed precision。
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(inputs, sample_labels=labels, masks=masks, region_labels=regions)
    output.loss.backward()
    summary = model.trainable_parameter_summary()
    if summary["base_model"] != 0:
        raise RuntimeError("冻结基座出现可训练参数")
    decoder_grad = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.decoder.parameters()
    )
    prompt_grad = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.prompt_learner.parameters()
    )
    if not decoder_grad or not prompt_grad:
        raise RuntimeError("E1 新模块没有有限梯度")
    result = {
        "seed": seed,
        "model_id": str(args.model_id),
        "image": str(args.image),
        "trainable_parameters": summary,
        "loss": float(output.loss.detach()),
        "classification_loss": float(output.classification_loss.detach()),
        "segmentation_loss": float(output.segmentation_loss.detach()),
        "region_loss": float(output.region_loss.detach()),
        "anomaly_logits_shape": list(output.anomaly_logits.shape),
        "mask_logits_shape": list(output.mask_logits.shape),
        "region_logits_shape": list(output.region_logits.shape),
        "decoder_parameters_with_finite_grad": decoder_grad,
        "prompt_parameters_with_finite_grad": prompt_grad,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "base_dtype": str(next(base.parameters()).dtype),
        "trainable_dtype": str(next(model.decoder.parameters()).dtype),
        "autocast_dtype": "torch.float16",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
