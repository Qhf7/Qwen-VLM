"""Build a lossless direct448 cache for fold1/train images and valid masks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SIDE = 448
EXPECTED_VIEWS = 18_775


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train).sort_values("image_path", kind="mergesort")
    if len(train) != EXPECTED_VIEWS or train.image_path.nunique() != EXPECTED_VIEWS:
        raise RuntimeError("fold1/train is not the frozen 18,775-view split")
    if not train.loc[train.view_label.eq(1), "mask_file_exists"].all():
        raise RuntimeError("a positive training view is missing its mask")

    existing = pd.read_csv(args.manifest) if args.manifest.is_file() else pd.DataFrame()
    done = set(existing.image_path) if len(existing) else set()
    records = existing.to_dict("records") if len(existing) else []
    for index, row in enumerate(train.to_dict("records"), 1):
        if row["image_path"] in done:
            continue
        image_target = args.cache_root / row["image_path"]
        with Image.open(args.root / row["image_path"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        rgb448 = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        image_target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb448).save(image_target, format="PNG", compress_level=3)

        mask_sha256 = ""
        mask_positive_px = 0
        if row["view_label"] == 1:
            mask_target = args.cache_root / row["mask_path"]
            with Image.open(args.root / row["mask_path"]) as source:
                mask = np.asarray(source.convert("L")) > 0
            mask448 = cv2.resize(
                mask.astype(np.uint8), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST
            ) > 0
            mask_positive_px = int(mask448.sum())
            if mask_positive_px == 0:
                raise RuntimeError(f"positive mask vanished at 448: {row['mask_path']}")
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask448.astype(np.uint8) * 255).save(
                mask_target, format="PNG", compress_level=6
            )
            mask_sha256 = hashlib.sha256(mask448.tobytes()).hexdigest()

        records.append({
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "view_label": row["view_label"],
            "image_pixel_sha256": hashlib.sha256(rgb448.tobytes()).hexdigest(),
            "mask_pixel_sha256": mask_sha256,
            "mask_positive_px": mask_positive_px,
            "image_cached_bytes": image_target.stat().st_size,
        })
        if index % 100 == 0:
            pd.DataFrame(records).to_csv(args.manifest, index=False)
            print(f"[train448 cache] {index}/{EXPECTED_VIEWS}", flush=True)

    frame = pd.DataFrame(records).sort_values("image_path", kind="mergesort")
    frame.to_csv(args.manifest, index=False)
    if len(frame) != EXPECTED_VIEWS or frame.image_path.nunique() != EXPECTED_VIEWS:
        raise RuntimeError("train448 cache is incomplete")
    if (frame.loc[frame.view_label.eq(1), "mask_positive_px"] <= 0).any():
        raise RuntimeError("train448 cache contains an empty positive mask")
    image_gib = frame.image_cached_bytes.sum() / 2**30
    print(
        f"[done] {len(frame)} images; {(frame.view_label == 1).sum()} masks; "
        f"image bytes={image_gib:.3f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
