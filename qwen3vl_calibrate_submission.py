#!/usr/bin/env python3
"""Calibrate a completed Qwen3-VL crop audit into a new submission.

This is a deterministic, Qwen-only post-inference calibration stage.  It
does not use test labels, external models, external data, or online APIs.
The original Qwen audit and submission are left untouched.

Signals used:
* reason / defect_type wording strength;
* crop-box geometry and area;
* number of anomalous crops in a view;
* agreement of boxes from multiple crops in the same view;
* raw anomaly_score jointly with is_anomaly;
* multi-view support at sample level.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


MASK_SIZE = 448
VIEWS = 5

POSITIVE_TERMS = (
    "划痕", "刮痕", "裂纹", "裂缝", "破损", "缺口", "凹陷", "凹坑",
    "污渍", "污点", "异物", "腐蚀", "锈", "变形", "断裂", "缺陷",
    "scratch", "scuff", "crack", "dent", "damage", "broken", "stain",
    "spot", "foreign", "corrosion", "rust", "deform", "defect",
)
STRONG_TERMS = (
    "明显", "清晰", "明确", "可见", "线性", "黑色", "对比", "边缘清晰",
    "clear", "clearly", "visible", "distinct", "linear", "black",
    "contrast", "sharp", "located",
)
NEGATIVE_TERMS = (
    "正常", "无异常", "没有异常", "不确定", "看不清", "无法确认", "可能是反光",
    "反光", "高光", "阴影", "normal", "uncertain", "unclear", "reflection",
    "highlight", "shadow", "glare",
)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def get_parsed(rec: dict[str, Any]) -> dict[str, Any]:
    value = rec.get("parsed")
    return value if isinstance(value, dict) else {}


def text_strength(rec: dict[str, Any]) -> float:
    parsed = get_parsed(rec)
    text = " ".join(
        str(parsed.get(key, "")) for key in ("reason", "defect_type")
    ).lower()
    pos = sum(term.lower() in text for term in POSITIVE_TERMS)
    strong = sum(term.lower() in text for term in STRONG_TERMS)
    neg = sum(term.lower() in text for term in NEGATIVE_TERMS)
    value = 0.20 * min(pos, 3) + 0.12 * min(strong, 3) - 0.22 * min(neg, 2)
    return float(np.clip(value, 0.0, 1.0))


def raw_score_and_flag(rec: dict[str, Any]) -> tuple[float, bool]:
    parsed = get_parsed(rec)
    raw_score = clamp01(parsed.get("anomaly_score"), clamp01(rec.get("score")))
    flag = bool(parsed.get("is_anomaly", rec.get("is_anomaly", raw_score > 0.35)))
    return raw_score, flag


def mapped_box(
    rec: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float, float, float], float, float] | None:
    """Map a crop-local 0..1000 box to full-image normalized coordinates."""
    parsed = get_parsed(rec)
    box = parsed.get("box", rec.get("box", []))
    crop = rec.get("crop", [])
    if not isinstance(box, list) or len(box) < 4:
        return None
    if not isinstance(crop, list) or len(crop) < 4:
        return None
    try:
        w, h = image_size
        cx0, cy0, cx1, cy1 = [float(v) for v in crop[:4]]
        bx = [float(v) / 1000.0 for v in box[:4]]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if w <= 0 or h <= 0 or cx1 <= cx0 or cy1 <= cy0:
        return None
    bx1, by1, bx2, by2 = bx
    bx1, bx2 = sorted((float(np.clip(bx1, 0.0, 1.0)), float(np.clip(bx2, 0.0, 1.0))))
    by1, by2 = sorted((float(np.clip(by1, 0.0, 1.0)), float(np.clip(by2, 0.0, 1.0))))
    fx1 = float(np.clip((cx0 + bx1 * (cx1 - cx0)) / w, 0.0, 1.0))
    fy1 = float(np.clip((cy0 + by1 * (cy1 - cy0)) / h, 0.0, 1.0))
    fx2 = float(np.clip((cx0 + bx2 * (cx1 - cx0)) / w, 0.0, 1.0))
    fy2 = float(np.clip((cy0 + by2 * (cy1 - cy0)) / h, 0.0, 1.0))
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    area = (fx2 - fx1) * (fy2 - fy1)
    crop_area = (bx2 - bx1) * (by2 - by1)
    # Large boxes are usually object/background descriptions, not defects.
    if area > 0.20 or crop_area > 0.40:
        return None
    edge_touch = sum(v <= 0.01 for v in (bx1, by1)) + sum(
        v >= 0.99 for v in (bx2, by2)
    )
    # A large box touching two crop borders is especially likely to be a
    # hallucinated object region.
    if area > 0.08 and edge_touch >= 2:
        return None
    return (fx1, fy1, fx2, fy2), area, crop_area


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def box_quality(area: float, crop_area: float) -> float:
    # Compact, defect-like regions receive more confidence. Extremely tiny
    # boxes are retained but do not receive an excessive boost.
    if area <= 0.002:
        return 0.62
    if area <= 0.015:
        return 0.95
    if area <= 0.06:
        return 0.82
    if area <= 0.12:
        return 0.60
    return 0.38


def discrete_zero_score(
    reason: float,
    geometry: float,
    consistency: float,
    crop_support: float,
    raw_score: float,
) -> float:
    """Split the former hard-coded 0.70 into evidence-based levels."""
    evidence = (
        0.30 * reason
        + 0.25 * geometry
        + 0.25 * consistency
        + 0.20 * crop_support
    )
    # If Qwen gave a non-zero raw score, preserve it as an additional signal.
    evidence += 0.25 * raw_score
    if evidence >= 0.72:
        return 0.90
    if evidence >= 0.48:
        return 0.75
    if evidence >= 0.26:
        return 0.60
    return 0.45


def prepare_view(
    view: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[float, list[tuple[tuple[float, float, float, float], float]]]:
    crops = view.get("crops", [])
    if not isinstance(crops, list):
        crops = []
    candidates = []
    for rec in crops:
        if not isinstance(rec, dict):
            continue
        raw, flag = raw_score_and_flag(rec)
        box = mapped_box(rec, image_size)
        if not flag:
            continue
        reason = text_strength(rec)
        if box is None:
            # A semantic flag without a usable region contributes weakly to
            # classification, but never creates a full-image mask.
            candidates.append({
                "rec": rec, "raw": raw, "reason": reason, "box": None,
                "geometry": 0.18,
            })
            continue
        coords, area, crop_area = box
        candidates.append({
            "rec": rec, "raw": raw, "reason": reason, "box": coords,
            "area": area, "crop_area": crop_area,
            "geometry": box_quality(area, crop_area),
        })

    valid = [c for c in candidates if c["box"] is not None]
    for c in candidates:
        if c["box"] is None:
            c["consistency"] = 0.0
            continue
        neighbours = [
            d for d in valid if d is not c and
            (iou(c["box"], d["box"]) >= 0.03 or center_distance(c["box"], d["box"]) <= 0.12)
        ]
        c["consistency"] = min(1.0, len(neighbours) / 2.0)

    support = min(1.0, len(candidates) / 4.0)
    rendered: list[tuple[tuple[float, float, float, float], float]] = []
    scores = []
    for c in candidates:
        raw = c["raw"]
        if raw <= 0.05:
            score = discrete_zero_score(
                c["reason"], c["geometry"], c["consistency"], support, raw
            )
        else:
            # Joint raw-score/flag calibration. Strong wording and repeated
            # boxes can lift a moderate score; weak geometry caps it.
            score = 0.65 * raw + 0.20 * c["reason"] + 0.10 * c["consistency"] + 0.05 * c["geometry"]
            if c["geometry"] < 0.45:
                score *= 0.78
            score = float(np.clip(score, 0.0, 1.0))
        scores.append(score)
        if c["box"] is not None:
            rendered.append((c["box"], score))

    if not scores:
        return 0.0, rendered
    scores.sort(reverse=True)
    # Several agreeing crops should matter, but should not make a view score
    # equal to an unbounded sum.
    view_score = scores[0]
    if len(scores) >= 2:
        view_score += 0.10 * scores[1] * (1.0 + 0.5 * max(c["consistency"] for c in candidates))
    if len(scores) >= 3:
        view_score += 0.04 * scores[2]
    return float(np.clip(view_score, 0.0, 1.0)), rendered


def render_mask(
    regions: list[tuple[tuple[float, float, float, float], float]],
) -> np.ndarray:
    canvas = Image.new("L", (MASK_SIZE, MASK_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for (x1, y1, x2, y2), score in regions:
        fill = int(round(255 * float(np.clip(score, 0.0, 1.0))))
        draw.rectangle(
            [round(x1 * MASK_SIZE), round(y1 * MASK_SIZE),
             round(x2 * MASK_SIZE), round(y2 * MASK_SIZE)],
            fill=max(1, fill),
        )
    return np.asarray(canvas, dtype=np.uint8)


def pack(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output / "submission.csv", "submission.csv")
        for path in sorted((output / "predicted_masks").rglob("*.png")):
            zf.write(path, str(path.relative_to(output)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="Test_A")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", dest="zip_path", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists():
        # This is a new versioned output path; do not touch the original v2.
        shutil.rmtree(args.output)
    (args.output / "predicted_masks").mkdir(parents=True, exist_ok=True)

    rows: dict[str, float] = {}
    record_count = 0
    for line in args.audit.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = str(item["group_folder"])
        category, sample = key.split("/", 1)
        view_scores = []
        for view in item.get("views", []):
            view_id = int(view.get("view", len(view_scores)))
            image_path = args.data_root / args.split / category / sample / f"{view_id}.png"
            with Image.open(image_path) as image:
                image_size = image.size
            view_score, regions = prepare_view(view, image_size)
            view_scores.append(view_score)
            out_path = args.output / "predicted_masks" / category / sample / f"{view_id}_mask.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(render_mask(regions), mode="L").save(out_path, compress_level=1)

        anomalous_views = sum(score >= 0.40 for score in view_scores)
        sample_score = max(view_scores or [0.0])
        # Cross-view support is a small, bounded calibration term.
        if anomalous_views >= 2:
            sample_score = min(1.0, sample_score + 0.025)
        if anomalous_views >= 3:
            sample_score = min(1.0, sample_score + 0.025)
        rows[key] = float(sample_score)
        record_count += 1

    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        for key in sorted(rows):
            writer.writerow([key, f"{rows[key]:.6f}"])

    pack(args.output, args.zip_path)
    masks = sum(1 for _ in (args.output / "predicted_masks").rglob("*_mask.png"))
    print(json.dumps({
        "records": record_count,
        "masks": masks,
        "zip": str(args.zip_path),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
