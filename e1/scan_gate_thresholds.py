"""Sweep classifier gates over one frozen segmentation prediction cache.

The audit's per-image histograms contain real predictions for every Dev view,
including views that a gate would suppress. This script changes only the
classifier score and gate; it never approximates an ungated mask as black.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pixel_metrics(pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    p, n = float(pos.sum()), float(neg.sum())
    below = np.r_[0.0, np.cumsum(neg)[:-1]]
    auroc = float(np.sum(pos * (below + 0.5 * neg)) / (p * n))
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / p
    ap = float(np.sum((recall - np.r_[recall[1:], 0]) * precision))
    f1 = 2 * tp / np.maximum(2 * tp + fp + (p - tp), 1)
    return {"pixel_auroc": auroc, "pixel_ap": ap, "pixel_f1max": float(f1.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--classifiers", nargs="+", required=True,
                        help="version=path/to/step12000_views.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99])
    args = parser.parse_args()
    audit = args.audit
    records = pd.read_csv(audit / "per_image.csv")
    archive = np.load(audit / "per_image_histograms.npz")
    hist = archive["hist"]
    weights = archive["weights"]
    if len(records) != 3600 or hist.shape != (3600, 1, 2, 256) or not np.isclose(weights[0], .5):
        raise RuntimeError("expected complete Dev3600, mean_w0.5 histograms")
    if records.image_path.duplicated().any():
        raise RuntimeError("duplicate audit image paths")
    base = hist[:, 0]
    labels = records.view_label.to_numpy(int)
    pos_pixels = base[:, 0].sum(axis=1)
    if not np.array_equal(pos_pixels, records.positive_pixels.to_numpy(int)):
        raise RuntimeError("histogram GT counts disagree with per-image audit")
    rows = []
    for specification in args.classifiers:
        version, filename = specification.split("=", 1)
        scores = pd.read_csv(filename)
        if len(scores) != 3600 or scores.image_path.duplicated().any():
            raise RuntimeError(f"{version}: incomplete classifier scores")
        merged = records[["image_path", "view_label"]].merge(
            scores[["image_path", "view_label", "p_yes"]], on="image_path", suffixes=("_audit", "_classifier"),
            validate="one_to_one", sort=False)
        if not merged.image_path.eq(records.image_path).all() or not merged.view_label_audit.eq(merged.view_label_classifier).all():
            raise RuntimeError(f"{version}: score/segmentation image or label mismatch")
        p_yes = merged.p_yes.to_numpy(float)
        image_auroc = float(roc_auc_score(labels, p_yes))
        image_ap = float(average_precision_score(labels, p_yes))
        for threshold in args.thresholds:
            passed = p_yes >= threshold if threshold > 0 else np.ones(len(p_yes), bool)
            pos = base[passed, 0].sum(axis=0).copy()
            neg = base[passed, 1].sum(axis=0).copy()
            pos[0] += int(base[~passed, 0].sum())
            neg[0] += int(base[~passed, 1].sum())
            metrics = pixel_metrics(pos, neg)
            # Fixed pixel threshold 128; separate from global F1max.
            tp = base[:, 0, 128:].sum(axis=1) * passed
            fp = base[:, 1, 128:].sum(axis=1) * passed
            predicted = tp + fp
            an = labels == 1
            dice = np.divide(2 * tp[an], predicted[an] + pos_pixels[an],
                             out=np.zeros(int(an.sum()), dtype=float),
                             where=(predicted[an] + pos_pixels[an]) > 0)
            recall = np.divide(tp[an], pos_pixels[an], out=np.zeros(int(an.sum()), dtype=float),
                               where=pos_pixels[an] > 0)
            rows.append({
                "classifier": version, "gate": threshold, "n_views": len(records),
                "image_auroc": image_auroc, "image_ap": image_ap,
                **metrics,
                "classifier_tp": int((an & passed).sum()),
                "classifier_fp": int(((~an) & passed).sum()),
                "classifier_fn": int((an & ~passed).sum()),
                "classifier_tn": int(((~an) & ~passed).sum()),
                "blocked_defect_pixel_share": float(pos_pixels[~passed].sum() / pos_pixels.sum()),
                "anomaly_dice_0_5": float(dice.mean()),
                "anomaly_recall_0_5": float(recall.mean()),
                "anomaly_empty_views": int((predicted[an] == 0).sum()),
                "normal_fp_views": int((predicted[~an] > 0).sum()),
            })
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "gate_scan.csv", index=False)
    # Select using tuning groups only in a separate analysis; all-Dev maxima
    # are descriptive and must not be treated as an unbiased test estimate.
    summary = {"audit": str(audit), "classifiers": [r.split("=", 1)[0] for r in args.classifiers],
               "thresholds": args.thresholds, "rows": len(frame),
               "test_images": 3600, "fusion_weight": .5}
    (args.out / "gate_scan_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
