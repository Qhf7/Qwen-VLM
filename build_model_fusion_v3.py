#!/usr/bin/env python3
"""Conservative model fusion for the validated CLIP/ensemble branch and DINOv2 v3.

This is a submission builder, not label tuning: it combines two independently
generated, compliant model outputs.  The validated branch keeps the strong
classification ranking and spatial masks; the DINOv2 v3 branch can add only
calibrated high-confidence local evidence.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def read_scores(root):
    with (root / "submission.csv").open(newline="", encoding="utf-8") as f:
        return {row["group_folder"]: float(row["anomaly_score"]) for row in csv.DictReader(f)}


def rank_scores(scores):
    keys = list(scores)
    values = np.asarray([scores[k] for k in keys], np.float64)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    denom = max(1, len(keys) - 1)
    return {k: float(order[i] / denom) for i, k in enumerate(keys)}


def load_mask(path):
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def save_mask(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(array, 0, 1) * 255)), "L").save(path, "PNG")


def mask_evidence(mask):
    p99 = float(np.percentile(mask, 99.0))
    area = float((mask > 0.45).mean())
    compact = float(np.clip((0.16 - area) / 0.16, 0.0, 1.0))
    return p99 * compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    base_scores = rank_scores(read_scores(args.base))
    cand_scores = rank_scores(read_scores(args.candidate))
    args.output.mkdir(parents=True, exist_ok=True)
    out_masks = args.output / "predicted_masks"

    rows = []
    for key in sorted(base_scores):
        category, sample = key.split("/", 1)
        view_evidence = []
        for view in range(5):
            base_path = args.base / "predicted_masks" / category / sample / f"{view}_mask.png"
            cand_path = args.candidate / "predicted_masks" / category / sample / f"{view}_mask.png"
            base = load_mask(base_path)
            cand = load_mask(cand_path)
            ev = mask_evidence(cand)
            view_evidence.append(ev)
            # Candidate evidence must be both high and compact before it can
            # affect the validated base mask.  This prevents the v2 failure
            # mode where broad bright normal texture lowers pixel AP/F1.
            add = np.clip((cand - 0.38) / 0.62, 0, 1)
            add *= float(np.clip((ev - 0.20) / 0.45, 0, 1))
            fused = np.maximum(base * 0.96, np.maximum(base, add) * 0.18 + base * 0.82)
            save_mask(out_masks / category / sample / f"{view}_mask.png", fused)
        cand_reliable = float(np.clip((max(view_evidence) - 0.20) / 0.45, 0.0, 1.0))
        score = (0.82 * base_scores[key] + 0.18 * cand_scores.get(key, base_scores[key])) * (0.92 + 0.08 * cand_reliable)
        rows.append({"group_folder": key, "anomaly_score": f"{float(np.clip(score, 0, 1)):.8f}"})

    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "method": "validated CLIP/ensemble branch + DINOv2 v3 calibrated local evidence",
        "base": str(args.base),
        "candidate": str(args.candidate),
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_masks.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
