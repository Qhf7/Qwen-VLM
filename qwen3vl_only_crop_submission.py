#!/usr/bin/env python3
"""Qwen3-VL-only crop-scan submission.

This version uses Qwen3-VL as the only visual model, but asks it on local
overlapping crops instead of only the full image.  Crop-level JSON boxes are
mapped back to the full image and rasterized into the 448x448 submission mask.
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
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from qwen3vl_only_submission import VIEWS, MASK_SIZE, clamp01, extract_json, list_samples


def crop_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Return center and corner crops first, then edge crops."""
    size = int(round(min(width, height) * 0.56))
    xs = [0, max(0, (width - size) // 2), max(0, width - size)]
    ys = [0, max(0, (height - size) // 2), max(0, height - size)]
    center = int(round(min(width, height) * 0.72))
    center_box = ((width - center) // 2, (height - center) // 2, (width + center) // 2, (height + center) // 2)
    grid = {(x, y): (x, y, x + size, y + size) for y in ys for x in xs}
    ordered = [
        center_box,
        grid[(xs[0], ys[0])],
        grid[(xs[-1], ys[0])],
        grid[(xs[0], ys[-1])],
        grid[(xs[-1], ys[-1])],
        grid[(xs[1], ys[0])],
        grid[(xs[0], ys[1])],
        grid[(xs[-1], ys[1])],
        grid[(xs[1], ys[-1])],
        grid[(xs[1], ys[1])],
    ]
    return ordered


def prompt() -> str:
    return """你正在查看工业零件的一块局部放大图。请寻找这块局部图里的最可疑缺陷，
包括细小黑色划痕、污渍、凹陷、破损、异物、变形、局部颜色或纹理异常。
只输出合法 JSON，不要 Markdown：
{"anomaly_score":0.00,"is_anomaly":true,"defect_type":"scratch","box":[x1,y1,x2,y2],"reason":"..."}
坐标为当前局部图内的 0 到 1000 归一化坐标。
如果完全没有可疑缺陷，输出 anomaly_score=0.0,is_anomaly=false,box=[]。
如果存在疑似缺陷，请给出 0.4 到 1.0 的分数，并用 box 紧贴缺陷小区域。"""


def query(model, processor, crop_path: Path, max_new_tokens: int):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(crop_path)},
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
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = extract_json(raw)
    return raw, parsed if isinstance(parsed, dict) else {}


def map_box_to_mask(box, crop, image_size, score: float) -> Image.Image:
    w, h = image_size
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    canvas = Image.new("L", (MASK_SIZE, MASK_SIZE), 0)
    if not isinstance(box, list) or len(box) < 4 or score <= 0.02:
        return canvas
    try:
        bx1, by1, bx2, by2 = [float(v) / 1000.0 for v in box[:4]]
    except Exception:
        return canvas
    bx1, bx2 = sorted((np.clip(bx1, 0, 1), np.clip(bx2, 0, 1)))
    by1, by2 = sorted((np.clip(by1, 0, 1), np.clip(by2, 0, 1)))
    fx1 = (x0 + bx1 * cw) / w
    fy1 = (y0 + by1 * ch) / h
    fx2 = (x0 + bx2 * cw) / w
    fy2 = (y0 + by2 * ch) / h
    # Reject huge boxes: they usually mean the model described the object,
    # not a local defect.
    if (fx2 - fx1) * (fy2 - fy1) > 0.20:
        return canvas
    draw = ImageDraw.Draw(canvas)
    fill = int(round(255 * max(score, 0.2)))
    draw.rectangle([
        round(fx1 * MASK_SIZE),
        round(fy1 * MASK_SIZE),
        round(fx2 * MASK_SIZE),
        round(fy2 * MASK_SIZE),
    ], fill=fill)
    return canvas.filter(ImageFilter.GaussianBlur(radius=1.2))


def merge_masks(masks: list[Image.Image]) -> np.ndarray:
    if not masks:
        return np.zeros((MASK_SIZE, MASK_SIZE), np.uint8)
    arr = np.maximum.reduce([np.asarray(m, np.uint8) for m in masks])
    return arr.astype(np.uint8)


def pack(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output / "submission.csv", "submission.csv")
        for path in sorted((output / "predicted_masks").rglob("*.png")):
            zf.write(path, str(path.relative_to(output)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="Test_A")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", dest="zip_path", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-crops", type=int, default=10)
    args = ap.parse_args()

    samples = list_samples(args.data_root / args.split)
    if args.limit:
        samples = samples[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predicted_masks").mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "qwen_crop_outputs.jsonl"
    done = set()
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
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
    with tempfile.TemporaryDirectory(prefix="qwen3vl_crop_") as tmp:
        tmp = Path(tmp)
        with audit_path.open("a", encoding="utf-8") as audit:
            for idx, (category, sample, paths) in enumerate(samples, 1):
                key = f"{category}/{sample}"
                if key in done:
                    continue
                print(f"[{idx}/{len(samples)}] {key}", flush=True)
                view_scores = []
                view_records = []
                for view_id, path in enumerate(paths):
                    image = Image.open(path).convert("RGB")
                    masks = []
                    crop_records = []
                    for crop_id, crop in enumerate(crop_boxes(*image.size)[: args.max_crops]):
                        crop_path = tmp / f"crop_{view_id}_{crop_id}.png"
                        image.crop(crop).save(crop_path)
                        try:
                            raw, parsed = query(model, processor, crop_path, args.max_new_tokens)
                        except Exception as exc:
                            raw, parsed = f"ERROR {type(exc).__name__}: {exc}", {}
                        score = clamp01(parsed.get("anomaly_score"), 0.0)
                        is_anomaly = bool(parsed.get("is_anomaly", score > 0.35))
                        if is_anomaly and score < 0.05 and isinstance(parsed.get("box"), list) and len(parsed.get("box")) >= 4:
                            # Qwen sometimes emits a correct anomaly flag and
                            # box while leaving the numeric field at 0.00.
                            # Interpret that structured semantic signal
                            # deterministically instead of discarding it.
                            score = 0.70
                        if not is_anomaly:
                            score = min(score, 0.15)
                        box = parsed.get("box", [])
                        masks.append(map_box_to_mask(box, crop, image.size, score if is_anomaly else 0.0))
                        crop_records.append({
                            "crop": crop,
                            "score": score,
                            "is_anomaly": is_anomaly,
                            "box": box,
                            "raw": raw,
                            "parsed": parsed,
                        })
                    view_score = max([r["score"] for r in crop_records] or [0.0])
                    view_scores.append(view_score)
                    mask = merge_masks(masks)
                    out_path = args.output / "predicted_masks" / category / sample / f"{view_id}_mask.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask, mode="L").save(out_path, compress_level=1)
                    view_records.append({"view": view_id, "score": view_score, "crops": crop_records})
                sample_score = max(view_scores or [0.0])
                rows[key] = sample_score
                audit.write(json.dumps({
                    "group_folder": key,
                    "sample_anomaly_score": sample_score,
                    "views": view_records,
                }, ensure_ascii=False) + "\n")
                audit.flush()
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["group_folder", "anomaly_score"])
                    for group, score in sorted(rows.items()):
                        writer.writerow([group, f"{score:.6f}"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for key, score in sorted(rows.items()):
            writer.writerow([key, f"{score:.6f}"])
    pack(args.output, args.zip_path)
    print(f"wrote {args.zip_path}", flush=True)


if __name__ == "__main__":
    main()
