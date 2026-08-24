#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

from PIL import Image


def main(args):
    expected = []
    for category in sorted(p for p in (args.data_root / args.split).iterdir() if p.is_dir()):
        for sample in sorted(p for p in category.iterdir() if p.is_dir()):
            if all((sample / f"{view}.png").exists() for view in range(5)):
                expected.append(f"{category.name}/{sample.name}")
    with zipfile.ZipFile(args.zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate ZIP member names")
        rows = list(csv.DictReader(archive.read("submission.csv").decode("utf-8").splitlines()))
        keys = [row["group_folder"] for row in rows]
        if keys != expected:
            raise RuntimeError("submission.csv keys/order do not match test groups")
        scores = [float(row["anomaly_score"]) for row in rows]
        if not all(0.0 <= score <= 1.0 for score in scores):
            raise RuntimeError("anomaly_score outside [0,1]")
        masks = [name for name in names if name.startswith("predicted_masks/") and name.endswith("_mask.png")]
        expected_masks = {f"predicted_masks/{key}/{view}_mask.png" for key in keys for view in range(5)}
        if set(masks) != expected_masks:
            raise RuntimeError("mask paths do not exactly match test groups")
        nonzero = 0
        for index, name in enumerate(masks):
            with archive.open(name) as handle:
                image = Image.open(handle); image.load()
            if image.mode != "L" or image.size != (448, 448):
                raise RuntimeError(f"invalid mask {name}: {image.mode} {image.size}")
            nonzero += image.getextrema()[1] > 0
            if index % 500 == 0:
                print(f"validated {index}/{len(masks)}", flush=True)
        if nonzero == 0:
            raise RuntimeError("all masks are black")
    print(
        f"VALID: rows={len(rows)} masks={len(masks)} nonzero_masks={nonzero} zip={args.zip_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="Test_A")
    parser.add_argument("--zip-path", type=Path, required=True)
    main(parser.parse_args())
