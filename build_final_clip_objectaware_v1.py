#!/usr/bin/env python3
"""Final submission builder: CLIP image scores + object-aware v4 masks.

This is a thin assembly step.  The image-level scores come from the validated
`submission_clip_vlm_v1` branch, and the pixel masks come from
`submission_object_aware_v4`, whose masks are already background-suppressed
and template-calibrated.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path


def read_rows(root: Path):
    with (root / "submission.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="CLIP-based image-score submission dir")
    parser.add_argument("--refined", type=Path, required=True, help="object-aware refined mask submission dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_rows(args.base)
    refined_rows = read_rows(args.refined)
    if [r["group_folder"] for r in base_rows] != [r["group_folder"] for r in refined_rows]:
        raise RuntimeError("base and refined submission.csv rows do not match")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.base / "submission.csv", args.output / "submission.csv")

    src_masks = args.refined / "predicted_masks"
    dst_masks = args.output / "predicted_masks"
    if dst_masks.exists():
        shutil.rmtree(dst_masks)
    shutil.copytree(src_masks, dst_masks)

    mask_count = sum(1 for _ in dst_masks.rglob("*_mask.png"))
    meta = {
        "method": "final CLIP image scores + object-aware v4 masks",
        "base": str(args.base),
        "refined": str(args.refined),
        "rows": len(base_rows),
        "masks": mask_count,
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(dst_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())

    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
