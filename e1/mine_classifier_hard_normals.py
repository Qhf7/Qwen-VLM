"""Mine normal views that the baseline classifier scores as anomalous.

The output is deliberately a small, auditable CSV consumed by build_sampler_pool.py.
It never reads the Dev split; mining is performed on fold1/train normals only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from e1.evaluate_classification_fold1 import build_classification_model, encode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    p.add_argument("--cache-root", type=Path, default=Path("Train448"))
    p.add_argument("--out", type=Path, default=Path("artifacts/e1/classifier_hard_normals_v1_1.csv"))
    p.add_argument("--top-per-class", type=int, default=128)
    p.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    p.add_argument("--feature-layers", default="5,11,17,23")
    args = p.parse_args()
    if args.top_per_class <= 0:
        raise ValueError("top-per-class must be positive")
    train = pd.read_parquet(args.train)
    normals = train[train.view_label.eq(0)].copy()
    device = torch.device("cuda:0")
    layers = tuple(int(value) for value in args.feature_layers.split(","))
    model, processor = build_classification_model(args.model_id, args.checkpoint, device, feature_layers=layers)
    rows = []
    model.eval()
    for i, row in enumerate(normals.itertuples(index=False), 1):
        with Image.open(args.cache_root / row.image_path) as source:
            image = source.convert("RGB")
        inputs = encode(processor, image, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=model.autocast_dtype):
            out = model(inputs)
        rows.append({
            "image_path": row.image_path,
            "class_name": row.class_name,
            "group_path": row.group_path,
            "view_label": int(row.view_label),
            "p_yes": float(torch.sigmoid(out.anomaly_logits[0]).cpu()),
        })
        if i % 500 == 0:
            print(f"[mine hard normals] {i}/{len(normals)}", flush=True)
    scores = pd.DataFrame(rows).sort_values(["class_name", "p_yes"], ascending=[True, False])
    selected = scores.groupby("class_name", sort=True, group_keys=False).head(args.top_per_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False)
    scores.to_csv(args.out.with_name(args.out.stem + "_all.csv"), index=False)
    print({"all_normals": len(scores), "selected": len(selected), "out": str(args.out)})


if __name__ == "__main__":
    main()
