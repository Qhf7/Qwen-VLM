#!/usr/bin/env python3
"""Build a competition-ready v6 submission without touching old outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "v7_vlm_primary_v5"), str(ROOT / "v6_vlm_zero_shot_v4"), str(ROOT / "v5_multiscale_v3")]

from calibration_v6 import quantize_mask
from common import list_samples, load_many, pick_device, set_seed
from pipeline_v6 import calibrated_output, run_group
from runtime_v6 import fit_normal_calibration, load_models, load_or_build_banks
from progress_v6 import ProgressBar


def main(args):
    if args.output.exists() or args.zip_path.exists():
        raise FileExistsError(f"refusing to overwrite v6 output: {args.output} / {args.zip_path}")
    set_seed(args.seed); device = pick_device()
    checkpoint, dino, model = load_models(args, device)
    train = list_samples(args.data_root / "Train")
    test = list_samples(args.data_root / args.split)
    banks = load_or_build_banks(args, checkpoint, model, dino, train, device, args.bank_cache)
    calibration = fit_normal_calibration(args, checkpoint, dino, model, banks, train, device)
    (args.output / "predicted_masks").mkdir(parents=True)
    rows, diagnostics = [], []
    infer_bar = ProgressBar("test_inference", len(test), every=5)
    with torch.inference_mode():
        for index, sample in enumerate(test):
            images = load_many(sample.paths)
            prediction = run_group(
                images, [sample.category] * 5, model, dino, banks, checkpoint["dino_size"],
                checkpoint["clip_size"], checkpoint["layers"], device,
            )
            maps, score = calibrated_output(prediction, sample.category, calibration)
            rows.append((sample.key, score))
            view_stats = []
            for view, score_map in enumerate(maps):
                quantized = quantize_mask(score_map)
                path = args.output / "predicted_masks" / sample.category / sample.name / f"{view}_mask.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(np.round(quantized * 255)), "L").save(path, compress_level=1)
                view_stats.append({
                    "view": view, "mean": float(quantized.mean()),
                    "area_gt_0_5": float((quantized > 0.5).mean()),
                    "max": float(quantized.max()),
                })
            diagnostics.append({"group_folder": sample.key, "anomaly_score": score, "views": view_stats})
            if index % 5 == 0 or index + 1 == len(test):
                infer_bar.update(
                    index + 1,
                    {"group": sample.key, "score": float(score)},
                    force=True,
                )
    infer_bar.close({"groups": len(rows)})
    with (args.output / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["group_folder", "anomaly_score"])
        writer.writerows((key, f"{score:.10f}") for key, score in rows)
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(args.output / "submission.csv", "submission.csv")
        for path in sorted((args.output / "predicted_masks").rglob("*_mask.png")):
            archive.write(path, path.relative_to(args.output).as_posix())
    scores = np.asarray([x[1] for x in rows])
    all_views = [view for item in diagnostics for view in item["views"]]
    metadata = {
        "method": "DINO-primary localisation plus bounded VLM semantic residual v6",
        "official_train_only": True, "external_anomaly_data": False, "online_api": False,
        "groups": len(rows), "masks": len(all_views), "checkpoint": str(args.checkpoint),
        "score_unique": int(len(np.unique(np.round(scores, 10)))),
        "score_saturation_ge_0_99": float(np.mean(scores >= 0.99)),
        "median_mask_mean": float(np.median([x["mean"] for x in all_views])),
        "median_area_gt_0_5": float(np.median([x["area_gt_0_5"] for x in all_views])),
    }
    (args.output / "run_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"INFER_COMPLETE groups={len(rows)} masks={len(all_views)} zip={args.zip_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--weight-dir", type=Path, required=True)
    p.add_argument("--clip-weight", type=Path, required=True); p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--zip-path", type=Path, required=True); p.add_argument("--bank-cache", type=Path)
    p.add_argument("--split", default="Test_A"); p.add_argument("--calib-per-category", type=int, default=10)
    p.add_argument("--reference-budget", type=int, default=128); p.add_argument("--global-reference-budget", type=int, default=2048)
    p.add_argument("--pixel-threshold-quantile", type=float, default=0.995)
    p.add_argument("--pixel-high-quantile", type=float, default=0.9998)
    p.add_argument("--seed", type=int, default=20260826)
    main(p.parse_args())
