"""为远端推理生成与 direct448 像素完全等价的无损 PNG 缓存。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SIDE = 448


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    subset = pd.read_parquet(args.subset).sort_values("image_path", kind="mergesort")
    if len(subset) != 3600 or subset.image_path.nunique() != 3600:
        raise RuntimeError("不是冻结的 3600 视角 Dev-mini")
    existing = pd.read_csv(args.manifest) if args.manifest.is_file() else pd.DataFrame()
    done = set(existing.image_path) if len(existing) else set()
    records = existing.to_dict("records") if len(existing) else []
    for index, row in enumerate(subset.to_dict("records"), 1):
        if row["image_path"] in done:
            continue
        source_path = args.root / row["image_path"]
        target_path = args.cache_root / row["image_path"]
        with Image.open(source_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        resized = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(resized).save(target_path, format="PNG", compress_level=3)
        with Image.open(target_path) as check:
            decoded = np.asarray(check.convert("RGB"))
        if not np.array_equal(decoded, resized):
            raise RuntimeError(f"PNG 缓存不是无损像素: {target_path}")
        records.append({
            "image_path": row["image_path"],
            "source_width": row["width"],
            "source_height": row["height"],
            "cached_width": SIDE,
            "cached_height": SIDE,
            "pixel_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
            "cached_bytes": target_path.stat().st_size,
        })
        if index % 50 == 0:
            pd.DataFrame(records).to_csv(args.manifest, index=False)
            print(f"[dev448 cache] {index}/3600", flush=True)
    frame = pd.DataFrame(records).sort_values("image_path", kind="mergesort")
    frame.to_csv(args.manifest, index=False)
    if len(frame) != 3600 or frame.image_path.nunique() != 3600:
        raise RuntimeError("Dev448 缓存不完整")
    print(f"[done] {len(frame)} images, {frame.cached_bytes.sum()/2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
