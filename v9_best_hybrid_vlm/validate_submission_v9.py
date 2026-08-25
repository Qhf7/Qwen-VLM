#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import PurePosixPath

from PIL import Image


def main(path: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "submission.csv" not in names:
            raise RuntimeError("submission.csv is missing")
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open("submission.csv"), encoding="utf-8")))
        if len(rows) != 750:
            raise RuntimeError(f"expected 750 CSV rows, got {len(rows)}")
        if len({row["group_folder"] for row in rows}) != 750:
            raise RuntimeError("group_folder values are not unique")
        masks = [name for name in names if name.startswith("predicted_masks/") and name.endswith("_mask.png")]
        if len(masks) != 3750:
            raise RuntimeError(f"expected 3750 masks, got {len(masks)}")
        for index, name in enumerate(masks):
            parts = PurePosixPath(name).parts
            if len(parts) != 4 or parts[-1] not in {f"{view}_mask.png" for view in range(5)}:
                raise RuntimeError(f"invalid mask path: {name}")
            with Image.open(archive.open(name)) as image:
                if image.size != (448, 448) or image.mode != "L":
                    raise RuntimeError(f"invalid mask {name}: size={image.size} mode={image.mode}")
            if index % 500 == 0:
                print(f"validated {index}/{len(masks)}", flush=True)
    print(f"VALID rows={len(rows)} masks={len(masks)} zip={path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("zip_path")
    main(parser.parse_args().zip_path)
