"""Score a segmentation checkpoint along the *submission* inference path.

Why this exists
---------------
``evaluate_patch9_stitch.py`` scored a path that never reaches the zip:

* submission: fuse the full view with the nine patches at **original
  resolution** (arithmetic mean over overlaps), resize to 448, then quantise
  **once** with ``np.rint(p*255)``;
* old local path: quantise every branch to uint8 **before** fusing, then fuse on
  the 448 grid with centre weights.

Both differences move the score, so the old local numbers could not predict the
leaderboard. This script replays the submission path and additionally

* predicts **every** view, including those the gate suppresses, so a gate
  ablation never mistakes a missing cache for an all-black mask;
* sweeps ``--fusion-weights`` x ``--gate-thresholds`` in one pass, making the
  gate and fusion ablations cheap and directly comparable.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import os
import torch

from e1.evaluate_leaderboard_metrics import pooled_pixel_metrics, auroc_ap
from e1.evaluate_two_stage_pipeline import build_segmentation_model
from e1.patch9 import read_mask
from e1.patch9_v2_geometry import (
    crop_image,
    fuse_probabilities,
    make_patch_specs,
    read_rgb,
    resize_image,
    resize_mask,
)
from e1.train_segmentation_patch9_v2_geometry import predictor

# The original training code assumed an NVIDIA host.  Keep CUDA as the default
# for the competition machine, while allowing the same evaluator to run on an
# Apple MPS or CPU workstation for reproducibility.
DEVICE = torch.device(os.environ.get("VLM_DEVICE", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))

BINS = 256
SIDE = 448


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def load_gt(row, gt_root: Path) -> np.ndarray:
    """Binary GT on the model grid; normal views are all-background."""
    if int(getattr(row, "view_label", 1)) == 1 and not pd.isna(getattr(row, "mask_path", None)):
        mask = read_mask(gt_root / str(row.mask_path))
    else:
        mask = None
    return None if mask is None else resize_mask(mask, SIDE).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("artifacts/splits/fold1/dev_unbiased.parquet"))
    parser.add_argument("--image-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-probs", type=Path)
    parser.add_argument("--fusion-weights", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.5, 0.1, 0.0])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    weights = list(args.fusion_weights)
    gates = list(args.gate_thresholds)
    model, processor, _ = build_segmentation_model(args.model_id, args.segmentation_checkpoint)
    predict = predictor(model, processor)

    frame = pd.read_parquet(args.test).reset_index(drop=True)
    if args.limit:
        frame = frame.head(int(args.limit))
    if frame.image_path.duplicated().any():
        raise ValueError("Duplicate evaluation image")
    if "p_yes" not in frame.columns:
        raise SystemExit("test split has no p_yes column; rebuild it with build_dev_unbiased.py")

    if not np.isfinite(frame.p_yes).all():
        raise ValueError("Missing classifier probabilities")
    # Earlier exploratory scan used the first 600 rows. Keep all related views
    # on tuning side; remaining groups provide a separate confirmation set.
    group_col = "visual_group_id" if "visual_group_id" in frame else "group_path"
    keys = frame.class_name.astype(str) + "/" + frame[group_col].astype(str)
    tune_groups = set(keys.iloc[:600])
    partitions = np.where(keys.isin(tune_groups), "tune", "holdout")
    acc = {(gate, weight): [np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64)]
           for gate in gates for weight in weights}
    per_image_hist = []
    image_records = []
    total_defect_pixels = 0
    blocked_defect_pixels = 0
    anomaly_views = 0
    blocked_anomaly_views = 0

    for number, row in enumerate(frame.itertuples(index=False), 1):
        image = read_rgb(args.image_root / str(row.image_path))
        height, width = image.shape[:2]
        gt = load_gt(row, args.gt_root)
        if gt is None:
            gt = np.zeros((SIDE, SIDE), dtype=bool)
        specs = make_patch_specs(width, height)
        full = predict(resize_image(image))
        patches = {spec.index: predict(crop_image(image, spec)) for spec in specs}

        n_pos = int(gt.sum())
        n_neg = int(gt.size - n_pos)
        score = float(row.p_yes)
        if n_pos > 0:
            anomaly_views += 1
            total_defect_pixels += n_pos
            if score < 0.5:
                blocked_anomaly_views += 1
                blocked_defect_pixels += n_pos

        image_records.append(dict(image_path=row.image_path, class_name=row.class_name,
            group=str(keys.iloc[number-1]), partition=str(partitions[number-1]),
            view_label=int(row.view_label), ratio=float(row.defect_ratio), p_yes=score,
            positive_pixels=n_pos, blocked_at_05=score < .5))
        payload = {}
        row_hist = []
        levels = {}
        for weight in weights:
            fused = fuse_probabilities(full, patches, specs, (height, width), patch_weight=weight)
            level = np.rint(np.clip(fused, 0.0, 1.0) * 255).astype(np.int64)
            levels[weight] = level
            payload[f"fused_w{weight}"] = level.astype(np.uint8)
            image_records[-1][f"seg_max_w{weight}"] = float(level.max() / 255.0)
            row_hist.append(np.stack([np.bincount(level[gt], minlength=BINS),
                                      np.bincount(level[~gt], minlength=BINS)]))
            # Per-view segmentation diagnostics at the conventional 0.5 level.
            predicted = level >= 128
            tp = int((predicted & gt).sum()); fp = int((predicted & ~gt).sum())
            fn = int((~predicted & gt).sum()); denom = 2 * tp + fp + fn
            image_records[-1].update({
                f"pred_pixels_w{weight}": int(predicted.sum()),
                f"tp_w{weight}": tp, f"fp_w{weight}": fp, f"fn_w{weight}": fn,
                f"dice05_w{weight}": float(2 * tp / denom) if denom else (1.0 if not gt.any() else 0.0),
                f"recall05_w{weight}": float(tp / max(tp + fn, 1)),
                f"precision05_w{weight}": float(tp / max(tp + fp, 1)),
                f"empty_pred_w{weight}": bool(predicted.sum() == 0),
            })

        per_image_hist.append(np.stack(row_hist))
        for gate in gates:
            passed = gate <= 0.0 or score >= gate
            for weight in weights:
                hist_pos, hist_neg = acc[(gate, weight)]
                if passed:
                    hist_pos += np.bincount(levels[weight][gt], minlength=BINS)
                    hist_neg += np.bincount(levels[weight][~gt], minlength=BINS)
                else:
                    # Suppressed views are scored as all-background, which is what
                    # the submission writes - but the prediction above was still
                    # computed, so lowering the gate uses real probabilities.
                    hist_pos[0] += n_pos
                    hist_neg[0] += n_neg

        if args.save_probs:
            payload["full"] = np.rint(np.clip(full, 0, 1) * 255).astype(np.uint8)
            for index, probability in patches.items():
                payload[f"patch{index}"] = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
            payload["boxes"] = np.asarray([s.as_tuple() for s in specs], dtype=np.int32)
            payload["original_size"] = np.asarray((height, width), dtype=np.int32)
            destination = args.save_probs / (str(row.image_path).replace("/", "__") + ".npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, **payload)

        if number % 25 == 0:
            print(f"[submission-fusion] {number}/{len(frame)}", flush=True)

    results = {}
    for gate in gates:
        for weight in weights:
            hist_pos, hist_neg = acc[(gate, weight)]
            results[f"gate{gate}|w{weight}"] = pooled_pixel_metrics(hist_pos, hist_neg)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.segmentation_checkpoint),
        "test": str(args.test),
        "rows": int(len(frame)),
        "path": "submission: fuse at original resolution, quantise once after fusion",
        "fusion_weights": weights,
        "gate_thresholds": gates,
        "gating_diagnostics": {
            "anomaly_views": anomaly_views,
            "anomaly_views_blocked_at_0.5": blocked_anomaly_views,
            "blocked_anomaly_fraction": float(blocked_anomaly_views / max(anomaly_views, 1)),
            "total_defect_pixels": total_defect_pixels,
            "blocked_defect_pixels": blocked_defect_pixels,
            "blocked_defect_pixel_share": float(blocked_defect_pixels / max(total_defect_pixels, 1)),
        },
        "results": results,
        "note": ("gate<=0 keeps every real prediction (no gating). Each threshold is "
                 "accumulated independently, so no cache is treated as all-black."),
    }
    write_json(args.out / "submission_fusion_metrics.json", report)
    records = pd.DataFrame(image_records)
    records.to_csv(args.out / "per_image.csv", index=False)
    # Explicit image-classification report, independent of segmentation.
    y = records.view_label.to_numpy(int)
    p_yes = records.p_yes.to_numpy(float)
    cls_auroc, cls_ap = auroc_ap(y, p_yes)
    cls_rows = []
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        pred = p_yes >= threshold
        tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
        fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
        cls_rows.append(dict(threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
            accuracy=float((tp + tn) / max(len(y), 1)),
            precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
            f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
            anomaly_block_fraction=float(fn / max((y == 1).sum(), 1)),
            normal_false_positive_fraction=float(fp / max((y == 0).sum(), 1))))
    segmentation_image_metrics = {}
    for weight in weights:
        scores = records[f"seg_max_w{weight}"].to_numpy(float)
        s_auroc, s_ap = auroc_ap(y, scores)
        segmentation_image_metrics[str(weight)] = {
            "image_auroc": s_auroc, "image_ap": s_ap,
            "note": "image-level score is max fused pixel probability; this is separate from classifier p_yes."
        }
    (args.out / "classification_metrics.json").write_text(json.dumps({
        "image_auroc": cls_auroc, "image_ap": cls_ap,
        "rows": len(records), "positive_views": int((y == 1).sum()),
        "negative_views": int((y == 0).sum()), "thresholds": cls_rows,
        "segmentation_image_metrics": segmentation_image_metrics,
        "note": "p_yes is frozen classifier output; this report measures classification only."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(cls_rows).to_csv(args.out / "classification_thresholds.csv", index=False)
    # Ranked bad cases: each row is directly actionable and retains the image path.
    bad = records.copy()
    bad["classification_error"] = np.select(
        [bad.view_label.eq(1) & bad.p_yes.lt(0.5), bad.view_label.eq(0) & bad.p_yes.ge(0.5)],
        ["classification_miss", "classification_false_positive"], default="")
    bad["segmentation_error"] = np.select(
        [bad.view_label.eq(1) & bad["empty_pred_w0.5"],
         bad.view_label.eq(1) & bad["recall05_w0.5"].lt(0.2),
         bad.view_label.eq(0) & bad["pred_pixels_w0.5"].gt(0),
         bad["precision05_w0.5"].lt(0.2)],
        ["empty_prediction", "low_recall", "normal_false_positive", "low_precision"], default="")
    bad["bad_case_score"] = (
        bad["classification_error"].ne("").astype(int) * 2
        + bad["segmentation_error"].ne("").astype(int)
        + bad["fp_w0.5"] / (448 * 448) * 0.5
        + (1 - bad["recall05_w0.5"]) * bad.view_label
    )
    bad.sort_values("bad_case_score", ascending=False).to_csv(args.out / "bad_cases.csv", index=False)
    for label, chosen in {
        "classification_misses": bad.classification_error.eq("classification_miss"),
        "classification_false_positives": bad.classification_error.eq("classification_false_positive"),
        "segmentation_empty": bad.segmentation_error.eq("empty_prediction"),
        "segmentation_low_recall": bad.segmentation_error.eq("low_recall"),
        "segmentation_false_positive": bad.segmentation_error.eq("normal_false_positive"),
    }.items():
        bad[chosen].head(100).to_csv(args.out / f"bad_cases_{label}.csv", index=False)
    hist = np.asarray(per_image_hist, dtype=np.int64)
    np.savez_compressed(args.out / "per_image_histograms.npz", hist=hist,
                        weights=np.asarray(weights), gates=np.asarray(gates))
    selections = {"full": np.ones(len(frame),bool), "tune": partitions == "tune",
                  "holdout": partitions == "holdout"}
    for c in sorted(records.class_name.unique()):
        selections["class/"+c] = records.class_name.eq(c).to_numpy()
    selections["normal"] = records.view_label.eq(0).to_numpy()
    for name,lo,hi in [("tiny",0,.005),("small",.005,.02),("large",.02,1.01)]:
        selections[name] = ((records.ratio > 0) & (records.ratio >= lo) & (records.ratio < hi)).to_numpy()
    breakdown=[]
    for gate in gates:
        gated_hist=hist.copy()
        blocked=records.p_yes.to_numpy() < gate
        gated_hist[blocked]=0
        gated_hist[blocked,:,:,0]=hist[blocked].sum(axis=-1)
        for j,w in enumerate(weights):
            for label,chosen in selections.items():
                if not chosen.any(): continue
                pos,neg=gated_hist[chosen,j].sum(axis=0)
                metrics=pooled_pixel_metrics(pos,neg)
                h=gated_hist[chosen,j]
                tp=h[:,0,128:].sum(axis=1);fp=h[:,1,128:].sum(axis=1)
                positives=h[:,0].sum(axis=1)
                denom=tp+fp+positives
                dice=np.divide(2*tp,denom,out=np.ones_like(tp,dtype=float),where=denom>0)
                recall=np.divide(tp,positives,out=np.full_like(tp,np.nan,dtype=float),where=positives>0)
                breakdown.append(dict(slice=label,gate=gate,weight=w,rows=int(chosen.sum()),
                    **metrics, anomaly_dice05=float(dice[positives>0].mean()) if (positives>0).any() else None,
                    anomaly_recall05=float(recall[positives>0].mean()) if (positives>0).any() else None,
                    empty_prediction_fraction=float(((tp+fp)==0).mean()),
                    prediction_area_p95=float(np.quantile((tp+fp)/(448*448),.95))))
    pd.DataFrame(breakdown).to_csv(args.out / "error_breakdown.csv", index=False)
    tune=[x for x in breakdown if x['slice']=='tune']
    best=max(tune,key=lambda x:sum(x[k] for k in ('pixel_auroc','pixel_ap','pixel_f1max')))
    confirmed=[x for x in breakdown if x['slice']=='holdout' and x['gate']==best['gate'] and x['weight']==best['weight']]
    write_json(args.out / "selection.json",dict(tuning_choice=best,holdout=confirmed,
        rule="maximize mean pixel AUROC/AP/F1max on tune only; holdout does not select",
        group_column=group_col,checkpoint_sha256=hashlib.sha256(args.segmentation_checkpoint.read_bytes()).hexdigest()))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
