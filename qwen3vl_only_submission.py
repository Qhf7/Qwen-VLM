#!/usr/bin/env python3
"""Qwen3-VL-only industrial anomaly submission pipeline.

The only learned visual model used here is Qwen3-VL.  It receives all five
views of one sample and is prompted to return JSON containing:

* sample-level anomaly score;
* per-view anomaly score;
* defect polygons / boxes in normalized 0..1000 coordinates.

The polygons/boxes are deterministically rasterized into the competition's
448x448 grayscale masks.  No DINO, CLIP, external API, test labels, or
anomaly-specific weights are used.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


VIEWS = 5
MASK_SIZE = 448


def list_samples(root: Path) -> list[tuple[str, str, list[Path]]]:
    out = []
    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample in sorted(p for p in category.iterdir() if p.is_dir()):
            paths = [sample / f"{i}.png" for i in range(VIEWS)]
            if all(p.exists() for p in paths):
                out.append((category.name, sample.name, paths))
    return out


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return float(np.clip(x, 0.0, 1.0))


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in starts:
        try:
            value = json.loads(cleaned[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def normal_xy(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    # The prompt requests 0..1000 coordinates.
    return float(np.clip(x / 1000.0, 0.0, 1.0)), float(np.clip(y / 1000.0, 0.0, 1.0))


def polygon_or_box_mask(view: dict[str, Any], size: int = MASK_SIZE) -> np.ndarray:
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    score = clamp01(view.get("anomaly_score"), 0.0)
    if not bool(view.get("is_anomaly", score > 0.5)) or score <= 0.02:
        return np.zeros((size, size), np.uint8)
    fill = int(round(255 * max(score, 0.15)))
    regions = view.get("regions", [])
    if not isinstance(regions, list):
        regions = []

    for region in regions[:4]:
        if not isinstance(region, dict):
            continue
        polygon = region.get("polygon", [])
        pts = [normal_xy(p) for p in polygon] if isinstance(polygon, list) else []
        pts = [p for p in pts if p is not None]
        if len(pts) >= 3:
            draw.polygon([(round(x * size), round(y * size)) for x, y in pts], fill=fill)
            continue
        box = region.get("box", [])
        if isinstance(box, list) and len(box) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) / 1000.0 for v in box[:4]]
                x1, x2 = sorted((np.clip(x1, 0, 1), np.clip(x2, 0, 1)))
                y1, y2 = sorted((np.clip(y1, 0, 1), np.clip(y2, 0, 1)))
                draw.rectangle(
                    [round(x1 * size), round(y1 * size), round(x2 * size), round(y2 * size)],
                    fill=fill,
                )
            except (TypeError, ValueError):
                pass
        else:
            point = normal_xy(region.get("point"))
            if point is not None:
                x, y = round(point[0] * size), round(point[1] * size)
                radius = max(8, round(float(region.get("radius", 24)) / 1000 * size))
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)

    # If the model returned a score but no region, leave the mask black rather
    # than lighting the whole part.  This keeps semantic classification from
    # becoming a false pixel mask.
    return np.asarray(canvas, np.uint8)


def make_prompt() -> str:
    return """你是严格、保守的工业视觉质检专家。下面是同一个工业零部件的五个相机视角，
顺序为 view0 到 view4。请只依据图片本身判断，不要编造看不清的缺陷。

任务：
1. 判断零部件大致类别；
2. 判断整个样本是否有可见异常；
3. 对每个视角给出异常概率 0 到 1；
4. 如果某个视角确实存在异常，定位异常区域。坐标使用 0 到 1000 的归一化坐标，
   原点在左上角，x 向右，y 向下。优先输出紧贴缺陷的 polygon，至少 3 个点；
   如果无法给 polygon，再输出紧贴缺陷的 box=[x1,y1,x2,y2]；
5. 禁止把整个零件轮廓、背景或正常表面作为异常区域。若看不到明确缺陷，
   is_anomaly=false，regions=[]。

必须只输出合法 JSON，不要 Markdown 代码块，格式：
{
  "object": "零部件名称或 unknown",
  "sample_anomaly_score": 0.0,
  "sample_is_anomaly": false,
  "views": [
    {
      "view": 0,
      "anomaly_score": 0.0,
      "is_anomaly": false,
      "defect_type": "none",
      "regions": [
        {
          "polygon": [[x,y],[x,y],[x,y]],
          "box": [x1,y1,x2,y2],
          "point": [x,y],
          "radius": 20
        }
      ],
      "reason": "简短原因"
    }
  ],
  "confidence": 0.0
}
每个 region 只填写 polygon、box、point 三者中的一种。"""


def run_one(
    model,
    processor,
    image_paths: list[Path],
    work_dir: Path,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    resized = []
    for idx, source in enumerate(image_paths):
        target = work_dir / f"view{idx}.png"
        with Image.open(source) as image:
            image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC).save(target)
        resized.append(target)

    content = [{"type": "image", "image": str(p)} for p in resized]
    content.append({"type": "text", "text": make_prompt()})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer, extract_json(answer)


def pack(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output / "submission.csv", "submission.csv")
        for path in sorted((output / "predicted_masks").rglob("*.png")):
            zf.write(path, str(path.relative_to(output)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="Test_A")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    samples = list_samples(args.data_root / args.split)
    if args.limit:
        samples = samples[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predicted_masks").mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "qwen_outputs.jsonl"
    done = set()
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["group_folder"])
            except Exception:
                pass

    print(f"samples={len(samples)} already_done={len(done)}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"model_loaded device={device}", flush=True)

    rows = {}
    csv_path = args.output / "submission.csv"
    if csv_path.exists():
        import csv

        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["group_folder"]] = float(row["anomaly_score"])

    import csv

    with tempfile.TemporaryDirectory(prefix="qwen3vl_views_") as tmp:
        tmp_path = Path(tmp)
        with audit_path.open("a", encoding="utf-8") as audit:
            for index, (category, sample, paths) in enumerate(samples, 1):
                key = f"{category}/{sample}"
                if key in done:
                    continue
                print(f"[{index}/{len(samples)}] {key}", flush=True)
                try:
                    raw, parsed = run_one(
                        model, processor, paths, tmp_path, args.max_new_tokens
                    )
                except Exception as exc:
                    raw, parsed = f"INFERENCE_ERROR: {type(exc).__name__}: {exc}", {}
                    print(raw, flush=True)

                views = parsed.get("views", []) if isinstance(parsed, dict) else []
                view_by_id = {
                    int(v.get("view")): v
                    for v in views
                    if isinstance(v, dict) and str(v.get("view", "")).isdigit()
                }
                sample_score = clamp01(parsed.get("sample_anomaly_score"), 0.0)
                if not sample_score:
                    sample_score = max(
                        [clamp01(v.get("anomaly_score"), 0.0) for v in views if isinstance(v, dict)]
                        or [0.0]
                    )
                rows[key] = sample_score

                record = {
                    "group_folder": key,
                    "raw_answer": raw,
                    "parsed": parsed,
                }
                audit.write(json.dumps(record, ensure_ascii=False) + "\n")
                audit.flush()

                for view_id in range(VIEWS):
                    view = view_by_id.get(view_id, {})
                    mask = polygon_or_box_mask(view)
                    out_path = args.output / "predicted_masks" / category / sample / f"{view_id}_mask.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask, mode="L").save(out_path, compress_level=1)

                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["group_folder", "anomaly_score"])
                    for group, score in sorted(rows.items()):
                        writer.writerow([group, f"{score:.6f}"])

    pack(args.output, args.zip_path)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {args.zip_path}", flush=True)


if __name__ == "__main__":
    main()
