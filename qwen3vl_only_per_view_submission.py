#!/usr/bin/env python3
"""Qwen3-VL-only per-view anomaly submission.

Each camera view is queried independently so the model has enough output
budget to return a valid compact JSON localization result.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from qwen3vl_only_submission import (
    MASK_SIZE,
    VIEWS,
    clamp01,
    extract_json,
    polygon_or_box_mask,
    list_samples,
)


def prompt() -> str:
    return """你是工业视觉质检专家。只观察这一个零部件视角。
请重点检查细小划痕、污渍、凹陷、破损、异物、变形和局部颜色/纹理异常。
不要把整个零件、正常边缘或背景当成异常。请先在内部完成判断，最后只输出 JSON。
坐标使用 0 到 1000 的归一化坐标，原点在左上角。
只输出下面格式，不要 Markdown，不要额外解释：
{"object":"...","anomaly_score":0.00,"is_anomaly":true,"defect_type":"scratch","box":[x1,y1,x2,y2],"reason":"..."}
如果完全看不到缺陷，输出 anomaly_score<0.2、is_anomaly=false、box=[]。
如果有缺陷，box 必须紧贴最可疑的小区域，不能框住整个零件。"""


def infer_view(model, processor, image_path: Path, work_dir: Path, max_new_tokens: int):
    # Keep the original image; the processor performs a valid high-resolution
    # resize with explicit pixel limits below.
    resized = work_dir / "view.png"
    with Image.open(image_path) as image:
        image.convert("RGB").save(resized)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(resized)},
            {"type": "text", "text": prompt()},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        min_pixels=512 * 512,
        max_pixels=1024 * 1024,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return raw, extract_json(raw)


def pack(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output / "submission.csv", "submission.csv")
        for path in sorted((output / "predicted_masks").rglob("*.png")):
            zf.write(path, str(path.relative_to(output)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="Test_A")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", dest="zip_path", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    samples = list_samples(args.data_root / args.split)
    if args.limit:
        samples = samples[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predicted_masks").mkdir(parents=True, exist_ok=True)
    audit = args.output / "qwen_outputs_per_view.jsonl"
    done = set()
    if audit.exists():
        for line in audit.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["group_folder"])
            except Exception:
                pass

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True)
    rows = {}
    csv_path = args.output / "submission.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["group_folder"]] = float(row["anomaly_score"])

    with tempfile.TemporaryDirectory(prefix="qwen3vl_per_view_") as tmp:
        tmp_path = Path(tmp)
        with audit.open("a", encoding="utf-8") as audit_fp:
            for n, (category, sample, paths) in enumerate(samples, 1):
                key = f"{category}/{sample}"
                if key in done:
                    continue
                print(f"[{n}/{len(samples)}] {key}", flush=True)
                views = []
                for view_id, image_path in enumerate(paths):
                    try:
                        raw, parsed = infer_view(
                            model, processor, image_path, tmp_path, args.max_new_tokens
                        )
                    except Exception as exc:
                        raw, parsed = f"INFERENCE_ERROR {type(exc).__name__}: {exc}", {}
                    parsed = parsed if isinstance(parsed, dict) else {}
                    if "regions" not in parsed:
                        box = parsed.get("box", [])
                        parsed["regions"] = [{"box": box}] if isinstance(box, list) and len(box) >= 4 else []
                    if (
                        bool(parsed.get("is_anomaly", False))
                        and clamp01(parsed.get("anomaly_score"), 0.0) < 0.05
                        and isinstance(parsed.get("box"), list)
                        and len(parsed.get("box")) >= 4
                    ):
                        parsed["anomaly_score"] = 0.70
                    parsed["view"] = view_id
                    parsed["raw_answer"] = raw
                    parsed.setdefault("anomaly_score", 0.0)
                    parsed.setdefault("is_anomaly", False)
                    parsed.setdefault("regions", [])
                    parsed["anomaly_score"] = clamp01(parsed["anomaly_score"])
                    views.append(parsed)
                    mask = polygon_or_box_mask(parsed)
                    out_path = args.output / "predicted_masks" / category / sample / f"{view_id}_mask.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask, mode="L").save(out_path, compress_level=1)

                sample_score = max([clamp01(v.get("anomaly_score")) for v in views] or [0.0])
                rows[key] = sample_score
                audit_fp.write(json.dumps({
                    "group_folder": key,
                    "sample_anomaly_score": sample_score,
                    "views": views,
                }, ensure_ascii=False) + "\n")
                audit_fp.flush()
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["group_folder", "anomaly_score"])
                    for group, score in sorted(rows.items()):
                        writer.writerow([group, f"{score:.6f}"])

    pack(args.output, args.zip_path)
    print(f"wrote {args.zip_path}", flush=True)


if __name__ == "__main__":
    main()
