#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

from PIL import Image


def main(args):
    samples = []
    for category in sorted(path for path in (args.data_root / args.split).iterdir() if path.is_dir()):
        for sample in sorted(path for path in category.iterdir() if path.is_dir()):
            samples.append(f"{category.name}/{sample.name}")
    with zipfile.ZipFile(args.zip_path) as archive:
        names = set(archive.namelist())
        if "submission.csv" not in names:
            raise AssertionError("submission.csv missing at ZIP root")
        rows = list(csv.DictReader(archive.read("submission.csv").decode("utf-8").splitlines()))
        if len(rows) != len(samples):
            raise AssertionError(f"CSV rows={len(rows)}, expected={len(samples)}")
        keys = [row["group_folder"] for row in rows]
        if set(keys) != set(samples):
            raise AssertionError("CSV group_folder set does not match test data")
        for row in rows:
            score = float(row["anomaly_score"])
            if not 0.0 <= score <= 1.0:
                raise AssertionError(f"invalid anomaly_score {score} for {row['group_folder']}")
        expected_masks = len(samples) * 5
        mask_names = sorted(name for name in names if name.startswith("predicted_masks/") and name.endswith("_mask.png"))
        if len(mask_names) != expected_masks:
            raise AssertionError(f"masks={len(mask_names)}, expected={expected_masks}")
        for index, name in enumerate(mask_names):
            with archive.open(name) as handle:
                image = Image.open(handle)
                if image.mode != "L" or image.size != (448, 448):
                    raise AssertionError(f"bad mask {name}: mode={image.mode}, size={image.size}")
            if index % 500 == 0:
                print(f"validated {index}/{expected_masks}", flush=True)
    print(f"VALID: rows={len(rows)} masks={len(mask_names)} zip={args.zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="Test_A")
    parser.add_argument("--zip-path", type=Path, required=True)
    main(parser.parse_args())
