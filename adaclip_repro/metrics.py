"""Metrics for faithful and stricter AdaCLIP evaluation.

The paper reports AUROC and maximum F1 at image and pixel level.  The official
repository additionally computes average precision (AP/AUPR).  This module
keeps those metrics and adds diagnostics that are important for sparse defects:
AUPRO@0.3FPR, top-1%-pixel precision, predicted area, Brier score and ECE.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    auroc: float
    aupr: float
    f1max: float
    f1max_threshold: float
    brier: float
    ece_15: float


@dataclass
class PixelMetrics(BinaryMetrics):
    aupro_30: float
    hit_at_1pct: float
    pred_area_at_05: float
    mean_score: float


def _as_1d(values: Iterable[float], dtype=np.float64) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1)


def expected_calibration_error(labels, scores, n_bins: int = 15) -> float:
    labels = _as_1d(labels)
    scores = np.clip(_as_1d(scores), 0.0, 1.0)
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    total = float(labels.size)
    ece = 0.0
    for bin_id in range(n_bins):
        selected = indices == bin_id
        if not selected.any():
            continue
        ece += selected.sum() / total * abs(scores[selected].mean() - labels[selected].mean())
    return float(ece)


def binary_metrics(labels, scores) -> BinaryMetrics:
    labels = _as_1d(labels, dtype=np.int64)
    scores = np.clip(_as_1d(scores), 0.0, 1.0)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have the same number of elements")
    brier = float(np.mean((scores - labels) ** 2)) if labels.size else float("nan")
    ece = expected_calibration_error(labels, scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return BinaryMetrics(float("nan"), float("nan"), float("nan"),
                             float("nan"), brier, ece)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[best]) if best < thresholds.size else float("inf")
    return BinaryMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        aupr=float(average_precision_score(labels, scores)),
        f1max=float(f1[best]),
        f1max_threshold=threshold,
        brier=brier,
        ece_15=ece,
    )


def _aupro(masks: list[np.ndarray], maps: list[np.ndarray], *,
           max_fpr: float = 0.30, n_bins: int = 256) -> float:
    """Compute area under the per-region-overlap curve up to ``max_fpr``.

    Histogram accumulation avoids materializing a threshold-by-pixel matrix.
    The result is normalized by ``max_fpr`` and therefore lies in [0, 1].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    neg_hist = np.zeros(n_bins, dtype=np.float64)
    region_hists: list[np.ndarray] = []

    for gt, pred in zip(masks, maps):
        gt = np.asarray(gt, dtype=bool)
        pred = np.clip(np.nan_to_num(np.asarray(pred, dtype=np.float32)), 0.0, 1.0)
        if gt.shape != pred.shape:
            raise ValueError(f"mask/map shape mismatch: {gt.shape} vs {pred.shape}")
        neg_hist += np.histogram(pred[~gt], bins=edges)[0]
        components, count = ndimage.label(gt)
        for component_id in range(1, count + 1):
            region_hists.append(
                np.histogram(pred[components == component_id], bins=edges)[0].astype(np.float64)
            )

    n_negative = neg_hist.sum()
    if n_negative == 0 or not region_hists:
        return float("nan")

    fpr = np.r_[0.0, np.cumsum(neg_hist[::-1]) / n_negative]
    region_curves = []
    for hist in region_hists:
        region_curves.append(np.r_[0.0, np.cumsum(hist[::-1]) / max(hist.sum(), 1.0)])
    pro = np.mean(np.stack(region_curves), axis=0)

    # Collapse identical FPR points while retaining the best achievable PRO.
    unique_fpr = np.unique(fpr)
    unique_pro = np.array([pro[fpr == x].max() for x in unique_fpr])
    unique_pro = np.maximum.accumulate(unique_pro)
    keep = unique_fpr < max_fpr
    x = unique_fpr[keep]
    y = unique_pro[keep]
    boundary_y = float(np.interp(max_fpr, unique_fpr, unique_pro))
    x = np.r_[x, max_fpr]
    y = np.r_[y, boundary_y]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(y, x) / max_fpr)


def pixel_metrics(masks: Iterable[np.ndarray], maps: Iterable[np.ndarray]) -> PixelMetrics:
    masks = [np.asarray(x, dtype=bool) for x in masks]
    maps = [np.clip(np.nan_to_num(np.asarray(x, dtype=np.float32)), 0.0, 1.0) for x in maps]
    if len(masks) != len(maps):
        raise ValueError("masks and maps must contain the same number of images")
    if not masks:
        base = BinaryMetrics(*(float("nan"),) * 6)
        return PixelMetrics(**asdict(base), aupro_30=float("nan"),
                            hit_at_1pct=float("nan"), pred_area_at_05=float("nan"),
                            mean_score=float("nan"))
    for gt, pred in zip(masks, maps):
        if gt.shape != pred.shape:
            raise ValueError(f"mask/map shape mismatch: {gt.shape} vs {pred.shape}")

    labels = np.concatenate([x.reshape(-1) for x in masks]).astype(np.uint8)
    scores = np.concatenate([x.reshape(-1) for x in maps])
    base = binary_metrics(labels, scores)

    hits = []
    for gt, pred in zip(masks, maps):
        if not gt.any():
            continue
        k = max(1, int(np.ceil(pred.size * 0.01)))
        top = np.argpartition(pred.reshape(-1), -k)[-k:]
        hits.append(float(gt.reshape(-1)[top].mean()))

    return PixelMetrics(
        **asdict(base),
        aupro_30=_aupro(masks, maps),
        hit_at_1pct=float(np.mean(hits)) if hits else float("nan"),
        pred_area_at_05=float(np.mean([np.mean(x >= 0.5) for x in maps])),
        mean_score=float(np.mean([x.mean() for x in maps])),
    )


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        result[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return result


def _self_test() -> None:
    rng = np.random.default_rng(7)
    masks = []
    for _ in range(4):
        gt = np.zeros((64, 64), dtype=bool)
        y, x = rng.integers(8, 48, size=2)
        gt[y:y + 8, x:x + 8] = True
        masks.append(gt)
    oracle = pixel_metrics(masks, [x.astype(np.float32) for x in masks])
    assert oracle.auroc == 1.0 and oracle.aupr == 1.0 and oracle.f1max == 1.0
    assert oracle.aupro_30 > 0.99
    tied = binary_metrics([0, 1], [0.5, 0.5])
    assert abs(tied.auroc - 0.5) < 1e-12
    print("AdaCLIP metric self-test: OK")


if __name__ == "__main__":
    _self_test()
