"""Repack scheduled RGB PNGs as pixel-exact lossless WebP to reduce SSH transfer."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def repack_one(task: tuple[str, str, str, str, str]) -> tuple[str, int]:
    path_text, kind, source_root_text, output_root_text, expected_hash = task
    source = Path(source_root_text) / path_text
    target = Path(output_root_text) / path_text
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "mask":
        shutil.copy2(source, target)
        with Image.open(target) as image:
            value = np.asarray(image.convert("L")) > 0
    else:
        with Image.open(source) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.uint8)
        Image.fromarray(value).save(
            target,
            format="WEBP",
            lossless=True,
            quality=100,
            method=4,
            exact=True,
        )
        with Image.open(target) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(value, decoded):
            raise RuntimeError(f"lossless WebP pixel mismatch: {path_text}")
        value = decoded
    digest = hashlib.sha256(value.tobytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"scheduled cache hash mismatch: {path_text}")
    return path_text, target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", type=Path, default=Path("artifacts/e1/fold1_required_files.csv"))
    parser.add_argument("--cache-manifest", type=Path, default=Path("artifacts/e1/train448_cache_manifest.csv"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    required = pd.read_csv(args.required)
    manifest = pd.read_csv(args.cache_manifest)
    image_hashes = dict(zip(manifest.image_path, manifest.image_pixel_sha256))
    mask_hashes = {
        row.mask_path: row.mask_pixel_sha256
        for row in manifest.itertuples(index=False)
        if row.view_label == 1
    }
    tasks = []
    for row in required.itertuples(index=False):
        expected = image_hashes[row.path] if row.kind == "image" else mask_hashes[row.path]
        tasks.append((
            row.path, row.kind, str(args.source_root), str(args.output_root), expected
        ))
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (_, size) in enumerate(executor.map(repack_one, tasks), 1):
            total += size
            if index % 500 == 0:
                print(f"[lossless repack] {index}/{len(tasks)}", flush=True)
    print(f"[done] {len(tasks)} files; {total / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
