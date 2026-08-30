#!/usr/bin/env python3
"""Report coverage statistics that expose broad mask baseline leakage."""
from __future__ import annotations

import argparse
import json
import zipfile

import numpy as np
from PIL import Image


def summarize(path: str) -> dict[str, float | int | str]:
    values = []
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist()
            if name.startswith("predicted_masks/") and name.endswith("_mask.png")
        ]
        for name in names:
            with Image.open(archive.open(name)) as image:
                values.append(np.asarray(image.convert("L"), np.uint8))
    if not values:
        raise RuntimeError(f"no masks found in {path}")
    flat = np.concatenate([x.ravel() for x in values])
    per_mask_mean = np.asarray([x.mean() for x in values], np.float32)
    return {
        "zip": path,
        "masks": len(values),
        "mean_gray": round(float(flat.mean()), 4),
        "nonzero_fraction": round(float((flat >= 1).mean()), 6),
        "ge16_fraction": round(float((flat >= 16).mean()), 6),
        "ge32_fraction": round(float((flat >= 32).mean()), 6),
        "ge64_fraction": round(float((flat >= 64).mean()), 6),
        "p90_mask_mean": round(float(np.percentile(per_mask_mean, 90)), 4),
        "max_mask_mean": round(float(per_mask_mean.max()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_paths", nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.zip_paths], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
