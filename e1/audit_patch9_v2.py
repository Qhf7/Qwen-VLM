"""Run the Patch9-v2 geometry audit without loading a model or starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
from PIL import Image

from e1.patch9_v2_geometry import (crop_mask, make_patch_specs, mask_bbox,
                                    mask_center, patch_contains_mask,
                                    project_mask, resize_mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()
    frame = pd.read_parquet(args.parquet)
    if args.limit:
        frame = frame.head(args.limit)
    rows = []
    for row in frame.itertuples(index=False):
        image_path = args.image_root / str(row.image_path)
        mask_value = getattr(row, "mask_path", None)
        if mask_value is None or pd.isna(mask_value):
            continue
        mask_path = args.mask_root / str(mask_value)
        # Audit geometry only: read the image header and mask pixels, without
        # decoding the full RGB image. This keeps a full-dataset audit fast.
        with Image.open(image_path) as image_file:
            image_size = (int(image_file.height), int(image_file.width))
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(mask_path)
        mask = raw_mask > 0
        if tuple(mask.shape) != image_size:
            raise ValueError(f"image/mask shape mismatch: {image_path} {mask_path}")
        specs = make_patch_specs(image_size[1], image_size[0], fraction=args.fraction, overlap=args.overlap)
        full_448 = cv2.resize(mask.astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST) > 0
        positive = [s for s in specs if patch_contains_mask(mask, s)]
        patch_masks = [cv2.resize(mask[s.y0:s.y1, s.x0:s.x1].astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST) > 0 for s in specs]
        roundtrip = np.zeros_like(mask, dtype=bool)
        for spec, patch_mask in zip(specs, patch_masks):
            if patch_contains_mask(mask, spec):
                projected = cv2.resize(patch_mask.astype(np.uint8), (spec.width, spec.height), interpolation=cv2.INTER_NEAREST) > 0
                roundtrip[spec.y0:spec.y1, spec.x0:spec.x1] |= projected
        intersection = int((roundtrip & mask).sum())
        dice = (2.0 * intersection / (int(roundtrip.sum()) + int(mask.sum()))) if (roundtrip.any() or mask.any()) else 1.0
        union = int((roundtrip | mask).sum())
        iou = intersection / union if union else 1.0
        original_center = mask_center(mask)
        roundtrip_center = mask_center(roundtrip)
        center_error = float(np.hypot(roundtrip_center[0] - original_center[0], roundtrip_center[1] - original_center[1]))
        original_bbox = mask_bbox(mask)
        roundtrip_bbox = mask_bbox(roundtrip)
        bbox_error = [int(roundtrip_bbox[i] - original_bbox[i]) for i in range(4)]
        report = {
            "original_size": [int(image_size[1]), int(image_size[0])],
            "output_size": 448,
            "mask_pixels_original": int(mask.sum()),
            "mask_pixels_full_448": int(full_448.sum()),
            "mask_ratio_original": float(mask.mean()),
            "mask_ratio_full_448": float(full_448.mean()),
            "patch_count": 9,
            "positive_patch_count": len(positive),
            "empty_patch_count": 9 - len(positive),
            "positive_patch_ratios_448": [float(patch_masks[s.index].mean()) for s in positive],
            "patch_mask_pixels_448": [int(p.sum()) for p in patch_masks],
            "roundtrip_dice": float(dice),
            "roundtrip_iou": float(iou),
            "roundtrip_center_error_px": center_error,
            "roundtrip_bbox_error_px": bbox_error,
            "all_patch_boxes": [list(s.as_tuple()) for s in specs],
        }
        report["image_path"] = str(row.image_path)
        report["class_name"] = str(getattr(row, "class_name", ""))
        rows.append(report)
    if not rows:
        raise RuntimeError("no mask-bearing rows were audited")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "rows": len(rows), "fraction": args.fraction, "overlap": args.overlap,
        "positive_patch_mean": sum(r["positive_patch_count"] for r in rows) / len(rows),
        "empty_patch_mean": sum(r["empty_patch_count"] for r in rows) / len(rows),
        "full_448_pixels_mean": sum(r["mask_pixels_full_448"] for r in rows) / len(rows),
        "reports": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
    flat.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({"audited": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
