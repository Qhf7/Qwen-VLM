#!/usr/bin/env python3
"""Fuse fresh DINOv2 continuous maps with conservative Qwen3-VL proposals.

This script is the second model stage of ``qwen_dinov2_hybrid_v1``.  It never
uses a prior submission as a source of masks: ``--dino-root`` must be the
fresh DINOv2 inference output generated from Train-normal memory banks,
reconstruction, and the pseudo-anomaly head.  Qwen's audit JSONL is used
only as a *conservative local proposal*.

Unlike the earlier Qwen-only v3 calibration, a crop is trusted only when a
non-contradictory, explicit defect reason (or a non-trivial raw Qwen score)
coexists with a compact box.  ``is_anomaly=true`` plus ``score=0`` by itself
does not activate anything.  DINO supplies all dense pixel ranking; Qwen
can only locally strengthen it and never paints a box by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


MASK_SIZE = 448

NEGATIVE = (
    "未检测到", "没有发现", "无异常", "正常", "不确定", "看不清", "无法确认",
    "反光", "高光", "阴影", "normal", "no anomaly", "no defect", "uncertain",
    "unclear", "reflection", "highlight", "shadow", "glare",
)
DEFECT = (
    "划痕", "刮痕", "裂纹", "裂缝", "破损", "缺口", "凹陷", "凹坑", "污渍",
    "污点", "异物", "腐蚀", "锈", "变形", "断裂", "缺陷", "scratch", "scuff",
    "crack", "dent", "damage", "broken", "stain", "spot", "foreign", "rust",
    "corrosion", "deform", "defect",
)
ASSERTIVE = (
    "存在", "发现", "可见", "明显", "清晰", "位于", "符合", "呈", "对比",
    "there is", "detected", "visible", "clear", "clearly", "located", "distinct",
)


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def raw_and_flag(rec: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
    parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
    raw = clamp01(parsed.get("anomaly_score"), clamp01(rec.get("score")))
    flag = bool(parsed.get("is_anomaly", rec.get("is_anomaly", False)))
    return raw, flag, parsed


def reason_confidence(parsed: dict[str, Any]) -> float:
    text = " ".join(str(parsed.get(k, "")) for k in ("reason", "defect_type")).lower()
    if any(term.lower() in text for term in NEGATIVE):
        return 0.0
    defects = sum(term.lower() in text for term in DEFECT)
    assertive = sum(term.lower() in text for term in ASSERTIVE)
    if defects == 0:
        return 0.0
    return float(np.clip(0.45 + 0.20 * min(defects, 2) + 0.12 * min(assertive, 2), 0.0, 1.0))


def mapped_box(rec: dict[str, Any], size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    raw, _, parsed = raw_and_flag(rec)
    _ = raw
    box = parsed.get("box", rec.get("box", []))
    crop = rec.get("crop", [])
    if not isinstance(box, list) or len(box) < 4 or not isinstance(crop, list) or len(crop) < 4:
        return None
    try:
        w, h = size
        x0, y0, x1, y1 = [float(v) for v in crop[:4]]
        bx1, by1, bx2, by2 = [float(v) / 1000.0 for v in box[:4]]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if x1 <= x0 or y1 <= y0 or w <= 0 or h <= 0:
        return None
    bx1, bx2 = sorted((float(np.clip(bx1, 0, 1)), float(np.clip(bx2, 0, 1))))
    by1, by2 = sorted((float(np.clip(by1, 0, 1)), float(np.clip(by2, 0, 1))))
    fx1, fx2 = (x0 + bx1 * (x1 - x0)) / w, (x0 + bx2 * (x1 - x0)) / w
    fy1, fy2 = (y0 + by1 * (y1 - y0)) / h, (y0 + by2 * (y1 - y0)) / h
    fx1, fx2 = sorted((float(np.clip(fx1, 0, 1)), float(np.clip(fx2, 0, 1))))
    fy1, fy2 = sorted((float(np.clip(fy1, 0, 1)), float(np.clip(fy2, 0, 1))))
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    area = (fx2 - fx1) * (fy2 - fy1)
    crop_area = (bx2 - bx1) * (by2 - by1)
    touched = sum(v <= 0.01 for v in (bx1, by1)) + sum(v >= 0.99 for v in (bx2, by2))
    # Stricter than v2/v3: Qwen proposals must really be compact local regions.
    if area > 0.075 or crop_area > 0.20 or (area > 0.025 and touched >= 2):
        return None
    return fx1, fy1, fx2, fy2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def proposal_regions(view: dict[str, Any], image_size: tuple[int, int]) -> list[tuple[tuple[float, float, float, float], float]]:
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    for rec in view.get("crops", []):
        if not isinstance(rec, dict):
            continue
        raw, flag, parsed = raw_and_flag(rec)
        reason = reason_confidence(parsed)
        # Explicit semantic evidence is required. A valid numeric score can
        # also act as evidence, but the common 0.00 + true inconsistency cannot.
        evidence = max(reason, raw)
        if not flag or evidence < 0.42:
            continue
        box = mapped_box(rec, image_size)
        if box is not None:
            candidates.append((box, evidence))
    if len(candidates) < 2:
        return candidates
    out = []
    for idx, (box, strength) in enumerate(candidates):
        support = sum(iou(box, other) >= 0.05 for j, (other, _) in enumerate(candidates) if j != idx)
        # A single trustworthy textual report remains usable, but repeated
        # spatial agreement gets a bounded boost.
        out.append((box, min(1.0, strength + 0.10 * min(support, 2))))
    return out


def proposal_map(regions: list[tuple[tuple[float, float, float, float], float]]) -> np.ndarray:
    canvas = np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
    for (x1, y1, x2, y2), strength in regions:
        xa, xb = int(round(x1 * MASK_SIZE)), int(round(x2 * MASK_SIZE))
        ya, yb = int(round(y1 * MASK_SIZE)), int(round(y2 * MASK_SIZE))
        if xb <= xa or yb <= ya:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        cx, cy = (xa + xb - 1) / 2.0, (ya + yb - 1) / 2.0
        rx, ry = max((xb - xa) / 2.0, 1.0), max((yb - ya) / 2.0, 1.0)
        radial = np.clip(1.0 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0.0, 1.0)
        canvas[ya:yb, xa:xb] = np.maximum(canvas[ya:yb, xa:xb], strength * radial)
    if canvas.max() > 0:
        canvas = np.asarray(
            Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255), "L").filter(ImageFilter.GaussianBlur(1.2)),
            np.float32,
        ) / 255.0
    return canvas


def area_limited_proposal(prop: np.ndarray, target_frac: float = 0.018, hardness: float = 0.75) -> np.ndarray:
    """Limit the union of all Qwen proposals after they are merged.

    Per-box filtering is not enough: several individually valid boxes can
    cover too much normal structure together.  Qwen is only a local semantic
    booster here, so broad proposal maps should become conservative instead
    of amplifying a large silhouette region.
    """
    if target_frac <= 0 or hardness <= 0:
        return prop
    x = np.asarray(prop, np.float32)
    if x.max() <= 0:
        return x
    frac = float((x > 0.18).mean())
    if frac <= target_frac:
        return x
    penalty = (target_frac / max(frac, 1e-6)) ** hardness
    return np.clip(x * penalty, 0.0, 1.0)


def load_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["group_folder"]: float(row["anomaly_score"]) for row in csv.DictReader(f)}


def pack(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(root / "submission.csv", "submission.csv")
        for path in sorted((root / "predicted_masks").rglob("*_mask.png")):
            zf.write(path, path.relative_to(root).as_posix())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-root", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="Test_A")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--zip", dest="zip_path", type=Path, required=True)
    ap.add_argument("--boost", type=float, default=0.35)
    ap.add_argument("--proposal-area-target", type=float, default=0.018)
    ap.add_argument("--proposal-area-hardness", type=float, default=0.75)
    ap.add_argument("--require-full-audit", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    out_masks = args.output / "predicted_masks"
    dino_scores = load_scores(args.dino_root / "submission.csv")
    audit = {}
    for line in args.audit.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            audit[item["group_folder"]] = item
    if args.require_full_audit:
        missing = sorted(set(dino_scores) - set(audit))
        if missing:
            raise RuntimeError(f"Qwen audit is missing {len(missing)} samples, first={missing[:5]}")

    rows = []
    stats = Counter()
    for key in sorted(dino_scores):
        category, sample = key.split("/", 1)
        item = audit.get(key, {})
        view_qscores = []
        for view_id in range(5):
            base_path = args.dino_root / "predicted_masks" / category / sample / f"{view_id}_mask.png"
            base = np.asarray(Image.open(base_path).convert("L"), np.float32) / 255.0
            image_path = args.data_root / args.split / category / sample / f"{view_id}.png"
            with Image.open(image_path) as original:
                size = original.size
            view = next((v for v in item.get("views", []) if int(v.get("view", -1)) == view_id), {})
            regions = proposal_regions(view, size)
            raw_prop = proposal_map(regions)
            prop = area_limited_proposal(
                raw_prop,
                target_frac=args.proposal_area_target,
                hardness=args.proposal_area_hardness,
            )
            if regions:
                # Dense DINO map remains primary. Proposal boosts local
                # high-DINO response; it cannot turn a zero DINO region into
                # a bright rectangular mask.
                fused = np.clip(base * (1.0 + args.boost * prop), 0.0, 1.0)
                stats["proposal_views"] += 1
                if float(raw_prop.sum()) > float(prop.sum()) + 1e-6:
                    stats["proposal_area_limited_views"] += 1
            else:
                fused = base
            stats["proposal_regions"] += len(regions)
            view_qscores.append(float(prop.max()))
            out = out_masks / category / sample / f"{view_id}_mask.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.round(fused * 255)), "L").save(out, compress_level=1)
        # DINO's Train-calibrated image score dominates; trusted Qwen proposal
        # only breaks ties / reinforces an already DINO-ranked sample.
        qscore = max(view_qscores or [0.0])
        score = float(np.clip(0.96 * dino_scores[key] + 0.04 * qscore, 0.0, 1.0))
        rows.append((key, score))

    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_folder", "anomaly_score"])
        writer.writerows((key, f"{score:.8f}") for key, score in rows)
    meta = {
        "method": "fresh DINOv2 dense anomaly maps + conservative Qwen3-VL local proposal boost",
        "dino_root": str(args.dino_root),
        "audit": str(args.audit),
        "proposal_views": stats["proposal_views"],
        "proposal_regions": stats["proposal_regions"],
        "proposal_area_limited_views": stats["proposal_area_limited_views"],
        "proposal_area_target": args.proposal_area_target,
        "proposal_area_hardness": args.proposal_area_hardness,
        "qwen_default_zero_used_as_positive": False,
        "qwen_only_rectangles": False,
        "external_data": False,
        "test_labels_used": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pack(args.output, args.zip_path)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    print(f"masks={len(list(out_masks.rglob('*_mask.png')))} zip={args.zip_path}", flush=True)


if __name__ == "__main__":
    main()
