"""Inspect one image without ground-truth labels or an evaluation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DECISION_PROMPT, REGION_PROMPT
from .metrics import parse_prediction


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", choices=("region", "decision"), default="region")
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.max_pixels, args.max_length, args.max_new_tokens) <= 0:
        parser.error("pixel and token budgets must be positive")
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.output and args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")

    import torch
    from .modeling import load_model, make_inputs, move_inputs

    model, processor = load_model(args.model, adapter=args.adapter, training=False,
                                  max_pixels=args.max_pixels)
    prompt = REGION_PROMPT if args.task == "region" else DECISION_PROMPT
    inputs = make_inputs(processor, args.image, prompt, max_length=args.max_length)
    input_tokens = inputs["input_ids"].shape[-1]
    if input_tokens + args.max_new_tokens > args.max_length:
        raise ValueError("Input plus generation budget exceeds max-length")
    inputs = move_inputs(inputs, model)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                   max_new_tokens=args.max_new_tokens, use_cache=True)
    response = generated[:, input_tokens:]
    raw_text = processor.batch_decode(response, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
    parsed, error = parse_prediction(raw_text, require_boxes=args.task == "region")
    result = {
        "image": str(args.image), "task": args.task,
        "raw_text": raw_text, "parsed": parsed, "parse_error": error,
        "coordinate_frame": "full image after EXIF orientation; xyxy normalized 0–1000",
        "model": args.model, "adapter": args.adapter,
        "max_pixels": args.max_pixels, "input_tokens": int(input_tokens),
        "generated_tokens": int(response.shape[-1]),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
