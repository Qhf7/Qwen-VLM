"""在终端中与本地 Qwen-VL 模型进行多轮图片对话。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="本地模型目录")
    parser.add_argument("--image", type=Path, required=True, help="要分析的图片")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用，请改用 --device cpu")
    if not args.image.is_file():
        raise SystemExit(f"图片不存在：{args.image}")

    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "mps" else torch.float32

    print(f"[load] processor: {args.model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.side * args.side,
        max_pixels=args.side * args.side,
    )
    print(f"[load] model: {args.model_id}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)

    with Image.open(args.image) as source:
        image = source.convert("RGB").resize(
            (args.side, args.side), Image.Resampling.LANCZOS
        )

    messages: list[dict] = []
    first_turn = True
    print("\n模型已就绪。直接输入问题；/clear 清空历史；/exit 退出。")
    while True:
        try:
            prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit"}:
            print("退出。")
            break
        if prompt.lower() == "/clear":
            messages.clear()
            first_turn = True
            print("对话历史已清空；下一条问题会重新附带图片。")
            continue

        content: list[dict] = []
        if first_turn:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device=device, dtype=dtype)
            if value.is_floating_point()
            else value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if args.device == "mps":
            torch.mps.synchronize()

        prompt_length = inputs["input_ids"].shape[1]
        answer = processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        print(f"\n模型> {answer}", flush=True)
        messages.append({"role": "assistant", "content": answer})
        first_turn = False


if __name__ == "__main__":
    main()
