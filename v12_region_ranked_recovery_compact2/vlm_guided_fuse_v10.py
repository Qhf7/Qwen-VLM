#!/usr/bin/env python3
"""Fuse v10 calibration-fixed dense predictions with bounded local VLM evidence.

The dense map must come from the v10 DINOv2 reconstruction, memory and
pseudo-head pipeline. VLM boxes never become masks: they only add a bounded
residual where the calibrated DINO map already has positive support.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from object_prior import compact_small_anomaly, suppress_outline_rings


MASK_SIZE = 448
NEGATIVE_TERMS = (
    "未检测到", "没有发现", "无异常", "正常反光", "正常高光", "看不清", "无法确认",
    "no anomaly", "no defect", "normal reflection", "normal highlight", "uncertain",
    "unclear", "glare only", "shadow only",
)
DEFECT_TERMS = (
    "划痕", "刮痕", "裂纹", "裂缝", "破损", "缺口", "凹陷", "凹坑", "污渍", "污点",
    "异物", "腐蚀", "锈", "变形", "断裂", "变色", "错位", "scratch", "scuff", "crack",
    "dent", "damage", "broken", "stain", "spot", "foreign", "rust", "corrosion",
    "deform", "discolor", "misalign", "chip", "defect",
)
ASSERTIVE_TERMS = (
    "存在", "发现", "可见", "明显", "清晰", "位于", "符合", "呈现", "there is",
    "detected", "visible", "clear", "clearly", "located", "distinct",
)


@dataclass(frozen=True)
class Region:
    box: tuple[float, float, float, float]
    confidence: float
    defect_type: str
    area: float
    crop_index: int
    agreement: int = 0


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def raw_and_flag(record: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
    parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}
    fields = dict(record)
    fields.update(parsed)
    raw = clamp01(fields.get("anomaly_score"), clamp01(record.get("score")))
    flag = bool(fields.get("is_anomaly", False))
    return raw, flag, fields


def score_tier(raw: float) -> float:
    """Split Qwen's former 0.7 default into calibrated evidence tiers."""
    if raw < 0.45:
        return 0.0
    if raw < 0.60:
        return 0.45
    if raw < 0.75:
        return 0.60
    if raw < 0.90:
        return 0.75
    return 0.90


def reason_evidence(parsed: dict[str, Any]) -> tuple[float, str]:
    defect_type = str(parsed.get("defect_type", "")).strip().lower()
    text = f"{parsed.get('reason', '')} {defect_type}".lower()
    if any(term.lower() in text for term in NEGATIVE_TERMS):
        return 0.0, defect_type
    defects = sum(term.lower() in text for term in DEFECT_TERMS)
    assertive = sum(term.lower() in text for term in ASSERTIVE_TERMS)
    if defects == 0:
        return 0.0, defect_type
    confidence = 0.48 + 0.14 * min(defects, 2) + 0.08 * min(assertive, 2)
    return float(min(confidence, 0.90)), defect_type


def _normalised_box(
    record: dict[str, Any], image_size: tuple[int, int], max_area: float
) -> tuple[tuple[float, float, float, float], float] | None:
    _, _, parsed = raw_and_flag(record)
    box = parsed.get("box", record.get("box", []))
    crop = record.get("crop", [])
    if not isinstance(box, list) or len(box) < 4 or not isinstance(crop, list) or len(crop) < 4:
        return None
    try:
        width, height = image_size
        x0, y0, x1, y1 = [float(value) for value in crop[:4]]
        bx0, by0, bx1, by1 = [float(value) / 1000.0 for value in box[:4]]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
        return None
    bx0, bx1 = sorted(np.clip([bx0, bx1], 0.0, 1.0).tolist())
    by0, by1 = sorted(np.clip([by0, by1], 0.0, 1.0).tolist())
    fx0, fx1 = sorted(np.clip([(x0 + bx0 * (x1 - x0)) / width, (x0 + bx1 * (x1 - x0)) / width], 0.0, 1.0).tolist())
    fy0, fy1 = sorted(np.clip([(y0 + by0 * (y1 - y0)) / height, (y0 + by1 * (y1 - y0)) / height], 0.0, 1.0).tolist())
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    area = float((fx1 - fx0) * (fy1 - fy0))
    crop_area = float((bx1 - bx0) * (by1 - by0))
    touches = sum(value <= 0.01 for value in (bx0, by0)) + sum(value >= 0.99 for value in (bx1, by1))
    if area > max_area or crop_area > 0.22 or (area > 0.025 and touches >= 2):
        return None
    return (fx0, fy0, fx1, fy1), area


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def extract_regions(view: dict[str, Any], image_size: tuple[int, int], max_area: float = 0.075) -> list[Region]:
    regions = []
    for index, record in enumerate(view.get("crops", [])):
        if not isinstance(record, dict):
            continue
        raw, flag, parsed = raw_and_flag(record)
        reason, defect_type = reason_evidence(parsed)
        confidence = max(score_tier(raw), reason)
        if not flag or confidence < 0.45:
            continue
        mapped = _normalised_box(record, image_size, max_area)
        if mapped is None:
            continue
        box, area = mapped
        # Compact regions are more credible for this small-defect benchmark.
        area_factor = 1.0 if area <= 0.012 else math.sqrt(0.012 / max(area, 1e-8))
        confidence *= float(np.clip(area_factor, 0.55, 1.0))
        if area > 0.014:
            confidence *= 0.68
        regions.append(Region(box, confidence, defect_type, area, index))
    output = []
    for index, region in enumerate(regions):
        agreement = 0
        for other_index, other in enumerate(regions):
            if index == other_index:
                continue
            same_type = not region.defect_type or not other.defect_type or region.defect_type == other.defect_type
            if same_type and box_iou(region.box, other.box) >= 0.04:
                agreement += 1
        confidence = min(0.95, region.confidence + 0.07 * min(agreement, 2))
        output.append(replace(region, confidence=confidence, agreement=agreement))
    return output


def proposal_map(regions: list[Region], group_factor: float = 1.0) -> np.ndarray:
    canvas = np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
    for region in regions:
        x0, y0, x1, y1 = region.box
        xa, xb = int(round(x0 * MASK_SIZE)), int(round(x1 * MASK_SIZE))
        ya, yb = int(round(y0 * MASK_SIZE)), int(round(y1 * MASK_SIZE))
        if xb <= xa or yb <= ya:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        cx, cy = (xa + xb - 1) / 2.0, (ya + yb - 1) / 2.0
        rx, ry = max((xb - xa) / 2.0, 1.0), max((yb - ya) / 2.0, 1.0)
        radial = np.clip(1.0 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0.0, 1.0)
        value = float(np.clip(region.confidence * group_factor, 0.0, 1.0)) * radial
        canvas[ya:yb, xa:xb] = np.maximum(canvas[ya:yb, xa:xb], value)
    if canvas.max() > 0:
        canvas = np.asarray(
            Image.fromarray(np.uint8(canvas * 255), "L").filter(ImageFilter.GaussianBlur(0.8)),
            np.float32,
        ) / 255.0
    return canvas


def focus_on_peak_regions(proposal: np.ndarray, peak_percentile: float = 88.0) -> np.ndarray:
    """Keep only the tightest semantic peak so the outline does not dominate."""
    x = np.asarray(proposal, np.float32)
    if x.max() <= 0:
        return x
    positive = x[x > 0]
    if positive.size == 0:
        return x
    thr = float(np.percentile(positive, peak_percentile))
    return np.clip((x - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)


def limit_proposal_area(proposal: np.ndarray, target: float, hardness: float) -> np.ndarray:
    if target <= 0 or hardness <= 0 or proposal.max() <= 0:
        return proposal
    active = float((proposal > 0.18).mean())
    if active <= target:
        return proposal
    return np.clip(proposal * (target / max(active, 1e-8)) ** hardness, 0.0, 1.0)


def fuse_dense_with_vlm(
    base: np.ndarray,
    proposal: np.ndarray,
    support_floor: float = 0.012,
    support_high: float = 0.11,
    max_residual: float = 0.12,
) -> tuple[np.ndarray, float]:
    """Add a bounded residual only where DINO already supports anomaly."""
    dense = np.clip(np.asarray(base, np.float32), 0.0, 1.0)
    semantic = np.clip(np.asarray(proposal, np.float32), 0.0, 1.0)
    support = np.clip((dense - support_floor) / max(support_high - support_floor, 1e-6), 0.0, 1.0)
    aligned = np.clip((semantic * support) ** 1.35, 0.0, 1.0)
    residual = max_residual * aligned * (0.35 + 0.65 * dense)
    fused = dense + residual * (1.0 - dense)
    return np.clip(fused, 0.0, 1.0), float(aligned.max())


def group_factor(active_views: int) -> float:
    if active_views <= 1:
        return 0.92
    if active_views == 2:
        return 1.0
    return 1.04


def load_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["group_folder"]: float(row["anomaly_score"]) for row in csv.DictReader(handle)}


def load_audit(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        key = item.get("group_folder")
        if not isinstance(key, str):
            raise ValueError(f"audit line {line_number} has no group_folder")
        output[key] = item
    return output


def pack(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(root / "submission.csv", "submission.csv")
        for path in sorted((root / "predicted_masks").rglob("*_mask.png")):
            archive.write(path, path.relative_to(root).as_posix())


def main(args: argparse.Namespace) -> None:
    if args.output.exists() or args.zip_path.exists():
        raise FileExistsError(f"refusing to overwrite v9 output: {args.output} / {args.zip_path}")
    if not 0.0 <= args.support_floor < args.support_high <= 1.0:
        raise ValueError("support thresholds must satisfy 0 <= floor < high <= 1")
    if not 0.0 <= args.max_residual <= 0.25 or not 0.0 <= args.image_residual <= 0.10:
        raise ValueError("VLM residuals exceed the conservative v9 safety bounds")
    dense_scores = load_scores(args.dino_root / "submission.csv")
    if len(dense_scores) != 750:
        raise RuntimeError(f"expected 750 dense groups, got {len(dense_scores)}")
    audit = load_audit(args.audit)
    if args.require_full_audit:
        missing = sorted(set(dense_scores) - set(audit))
        if missing:
            raise RuntimeError(f"VLM audit misses {len(missing)} groups, first={missing[:5]}")
    (args.output / "predicted_masks").mkdir(parents=True)
    rows, diagnostics, stats = [], [], Counter()
    for group_index, key in enumerate(sorted(dense_scores)):
        category, sample = key.split("/", 1)
        item = audit.get(key, {})
        regions_by_view = []
        for view_id in range(5):
            image_path = args.data_root / args.split / category / sample / f"{view_id}.png"
            with Image.open(image_path) as image:
                image_size = image.size
            view = next((value for value in item.get("views", []) if int(value.get("view", -1)) == view_id), {})
            regions_by_view.append(extract_regions(view, image_size, args.max_box_area))
        active_views = sum(bool(regions) for regions in regions_by_view)
        consistency = group_factor(active_views)
        aligned_scores = []
        view_diagnostics = []
        for view_id, regions in enumerate(regions_by_view):
            image_path = args.data_root / args.split / category / sample / f"{view_id}.png"
            base_path = args.dino_root / "predicted_masks" / category / sample / f"{view_id}_mask.png"
            with Image.open(base_path) as image:
                base = np.asarray(image.convert("L"), np.float32) / 255.0
            with Image.open(image_path) as image_rgb:
                rgb_image = image_rgb.convert("RGB")
            proposal = proposal_map(regions, consistency)
            proposal = focus_on_peak_regions(proposal, peak_percentile=90.0)
            proposal = limit_proposal_area(proposal, args.proposal_area_target, args.proposal_area_hardness)
            fused, aligned = fuse_dense_with_vlm(
                base, proposal, args.support_floor, args.support_high, args.max_residual
            )
            fused = suppress_outline_rings(fused, rgb_image, ring_keep=0.22)
            fused = compact_small_anomaly(fused, target_frac=0.012, softness=0.7)
            aligned_scores.append(aligned)
            if regions:
                stats["proposal_views"] += 1
                stats["proposal_regions"] += len(regions)
            if aligned > 0:
                stats["dino_aligned_views"] += 1
            output = args.output / "predicted_masks" / category / sample / f"{view_id}_mask.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.round(fused * 255)), "L").save(output, compress_level=1)
            view_diagnostics.append({
                "view": view_id, "regions": len(regions), "aligned_support": aligned,
                "base_mean": float(base.mean()), "fused_mean": float(fused.mean()),
            })
        semantic_support = max(aligned_scores or [0.0])
        base_score = dense_scores[key]
        score = float(np.clip(base_score + args.image_residual * semantic_support * (1.0 - base_score), 0.0, 1.0))
        rows.append((key, score))
        diagnostics.append({
            "group_folder": key, "base_score": base_score, "score": score,
            "active_vlm_views": active_views, "semantic_support": semantic_support,
            "views": view_diagnostics,
        })
        if group_index % 25 == 0 or group_index + 1 == len(dense_scores):
            print(f"V9_PROGRESS fusion {group_index + 1}/{len(dense_scores)} score={score:.5f}", flush=True)
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["group_folder", "anomaly_score"])
        writer.writerows((key, f"{score:.10f}") for key, score in rows)
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    metadata = {
        "method": "Hybrid-v2 DINO three-branch dense localisation plus bounded local VLM guidance v9",
        "dino_root": str(args.dino_root), "audit": str(args.audit),
        "groups": len(rows), "masks": len(list((args.output / "predicted_masks").rglob("*_mask.png"))),
        "proposal_views": stats["proposal_views"], "proposal_regions": stats["proposal_regions"],
        "dino_aligned_views": stats["dino_aligned_views"],
        "support_floor": args.support_floor, "support_high": args.support_high,
        "max_residual": args.max_residual, "image_residual": args.image_residual,
        "official_train_only": True, "external_anomaly_data": False,
        "qwen_only_rectangles": False, "test_labels_used": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pack(args.output, args.zip_path)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"V9_COMPLETE zip={args.zip_path}", flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--dino-root", type=Path, required=True)
    value.add_argument("--audit", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--split", default="Test_A")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--zip", dest="zip_path", type=Path, required=True)
    value.add_argument("--support-floor", type=float, default=0.012)
    value.add_argument("--support-high", type=float, default=0.11)
    value.add_argument("--max-residual", type=float, default=0.12)
    value.add_argument("--image-residual", type=float, default=0.025)
    value.add_argument("--max-box-area", type=float, default=0.075)
    value.add_argument("--proposal-area-target", type=float, default=0.012)
    value.add_argument("--proposal-area-hardness", type=float, default=0.90)
    value.add_argument("--require-full-audit", action="store_true")
    return value


if __name__ == "__main__":
    main(parser().parse_args())
