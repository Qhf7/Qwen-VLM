"""N1 negative control: treat the E1 decoder heatmap directly as a mask.

This is deliberately not used for checkpoint selection. It answers a narrow
question: how useful is the decoder's pixel heatmap before the LLM prompt
path or any post-processing? The script reports macro Pixel AUPR/F1max,
Hit@1%, predicted area, and 3600-view latency on the frozen fold1 Dev split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from b0.b0plus_protocol import P0
from e1.evaluate_fold1 import build_model
from s0 import metrics as pixel_metrics


SIDE = 448


def load_gt(row, gt_root: Path) -> np.ndarray:
    if int(row.view_label) == 0:
        return np.zeros((SIDE, SIDE), dtype=bool)
    mask_path = str(row.mask_path) if not pd.isna(row.mask_path) else ""
    if not mask_path or mask_path.lower() == "nan":
        raise FileNotFoundError(f"positive view has no mask_path: {row.image_path}")
    with Image.open(gt_root / mask_path) as source:
        value = np.asarray(source.convert("L")) > 0
    return cv2.resize(value.astype(np.uint8), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/e1_fold1/best.pt"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--cache-root", type=Path, default=Path("Dataset448"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_n1"))
    parser.add_argument("--tag", default="e1_best")
    parser.add_argument("--feature-layers", default="7,15,23,31")
    parser.add_argument("--empty-mask-dice-only-nonempty", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(args.dev).sort_values(
        ["class_name", "condition", "group_path", "view_idx"], kind="mergesort"
    )
    if len(dev) != 3600 or dev.group_path.nunique() != 720:
        raise RuntimeError("N1 requires the frozen 3600-view fold1 Dev split")
    feature_layers = tuple(int(v.strip()) for v in args.feature_layers.split(",") if v.strip())
    device = torch.device("cuda:0")
    model, processor = build_model(
        args.model_id,
        args.checkpoint,
        device,
        dtype=torch.bfloat16,
        dice_nonempty_only=args.empty_mask_dice_only_nonempty,
        feature_layers=feature_layers,
    )
    states = {
        class_name: {
            "pos": np.zeros(pixel_metrics.N_THRESHOLDS, dtype=np.float64),
            "neg": np.zeros(pixel_metrics.N_THRESHOLDS, dtype=np.float64),
            "ratios": [],
            "hits": [],
            "n": 0,
        }
        for class_name in sorted(dev.class_name.unique())
    }
    started = time.perf_counter()
    latencies = []
    for index, row in enumerate(dev.itertuples(index=False), 1):
        with Image.open(args.cache_root / row.image_path) as source:
            image = source.convert("RGB")
            conversation = [[{
                "role": "user",
                "content": [{"type": "image", "image": image.copy()}, {"type": "text", "text": P0}],
            }]]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"text_kwargs": {"padding": True}},
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        torch.cuda.synchronize()
        one_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - one_started)
        heat = torch.sigmoid(output.mask_logits[0, 0]).float().cpu().numpy()
        gt = load_gt(row, args.gt_root)
        state = states[row.class_name]
        state["pos"], state["neg"] = pixel_metrics._accumulate(
            state["pos"], state["neg"], heat, gt, pixel_metrics.N_THRESHOLDS
        )
        state["ratios"].append(float((heat >= 0.5).mean()))
        state["n"] += 1
        if gt.any():
            k = max(1, int(0.01 * heat.size))
            top = np.argpartition(heat.ravel(), -k)[-k:]
            state["hits"].append(float(gt.ravel()[top].mean()))
        if index % 100 == 0:
            print(f"[N1 {args.tag}] {index}/{len(dev)}", flush=True)

    rows = []
    for class_name, state in states.items():
        result = pixel_metrics.finish_pixel_metrics(
            state["pos"], state["neg"], state["ratios"], state["n"], state["hits"]
        )
        rows.append({
            "class_name": class_name,
            "n_views": result.n_images,
            "pixel_auroc": float(result.p_auroc[0]),
            "pixel_aupr": float(result.p_aupr[0]),
            "pixel_f1max": float(result.p_f1max[0]),
            "hit_at_1pct": float(np.nanmean(result.hit_at_1pct)),
            "predicted_area_mean": float(np.mean(result.pred_area_ratio)),
        })
    per_class = pd.DataFrame(rows).sort_values("class_name")
    per_class.to_csv(args.out_dir / f"{args.tag}_per_class.csv", index=False)
    summary = {
        "protocol": "N1-decoder-heatmap-direct-mask-v1",
        "checkpoint": str(args.checkpoint),
        "feature_layers": list(feature_layers),
        "n_views": len(dev),
        "n_classes": len(per_class),
        "macro_pixel_auroc": float(per_class.pixel_auroc.mean()),
        "macro_pixel_aupr": float(per_class.pixel_aupr.mean()),
        "macro_pixel_f1max": float(per_class.pixel_f1max.mean()),
        "macro_hit_at_1pct": float(per_class.hit_at_1pct.mean()),
        "predicted_area_mean": float(per_class.predicted_area_mean.mean()),
        "view_latency_p50_s": float(np.percentile(latencies, 50)),
        "view_latency_p95_s": float(np.percentile(latencies, 95)),
        "elapsed_s": time.perf_counter() - started,
        "checkpoint_selection": "not used; negative control only",
    }
    (args.out_dir / f"{args.tag}_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
