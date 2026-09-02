"""Run local WinCLIP-0shot over an S0 manifest.

The model keeps its native 336/24x24 representation; masks are downsampled to
that grid for metric computation. This is intentionally sequential because a
single CLIP model is large; run it on CUDA for the full manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .run import LocalWinCLIP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "holdout", "train"], default="dev")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--manifests", type=Path, default=Path("artifacts/splits"))
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.manifests / f"fold{args.fold}" / f"{args.split}.parquet")
    if args.limit_groups:
        groups = sorted(df.group_path.unique())[:args.limit_groups]
        df = df[df.group_path.isin(groups)].copy()
    model = LocalWinCLIP(args.checkpoint, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    scores, labels, maps, pixel_y, pixel_s = [], [], [], [], []
    rows = []
    for row in df.to_dict("records"):
        path = args.root / row["image_path"]
        with Image.open(path) as im:
            image = im.convert("RGB")
        score, amap = model.predict(image, row["class_name"].replace("_", " "))
        gt = (np.asarray(Image.open(args.root / row["mask_path"]).convert("L")) > 0
              if row["view_label"] else np.zeros((row["height"], row["width"]), bool))
        gt_small = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize(
            (amap.shape[1], amap.shape[0]), Image.Resampling.NEAREST), bool)
        scores.append(score); labels.append(int(row["view_label"]))
        maps.append(amap); pixel_y.extend(gt_small.ravel()); pixel_s.extend(amap.ravel())
        rows.append({"image_path": row["image_path"], "group_path": row["group_path"],
                     "class_name": row["class_name"], "view_label": int(row["view_label"]),
                     "sample_label": int(row["sample_label"]), "score": float(score)})
        if len(rows) % 20 == 0:
            print(f"processed {len(rows)}/{len(df)}")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output / "view_scores.csv", index=False)
    np.save(args.output / "view_maps_native.npy", np.stack(maps).astype(np.float32))
    # Five-view max aggregation, exactly matching S0.
    grouped = out_df.groupby(["class_name", "group_path"], sort=True)
    sample = grouped.agg(score=("score", "max"), label=("sample_label", "first")).reset_index()
    y, s = sample.label.to_numpy(), sample.score.to_numpy()
    summary = {"n_views": len(out_df), "n_groups": len(sample), "n_classes": int(sample.class_name.nunique()),
               "method": "WinCLIP-0shot-local-ViT-L-14@336", "split": args.split}
    if len(np.unique(y)) == 2:
        summary.update({"sample_I_AUROC": float(roc_auc_score(y, s)),
                        "sample_I_AUPR": float(average_precision_score(y, s))})
    py, ps = np.asarray(pixel_y), np.asarray(pixel_s)
    if len(np.unique(py)) == 2:
        summary.update({"view_P_AUROC_native_grid": float(roc_auc_score(py, ps)),
                        "view_P_AUPR_native_grid": float(average_precision_score(py, ps))})
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
