#!/usr/bin/env python3
"""Official metric implementation and the local proxy validation harness.

Scoring formula, reverse-engineered from three A-board submissions and
confirmed to within 0.002 absolute:

    S_cls = mean(I-AUROC, I-AUPR)                 per category, then averaged
    S_seg = mean(P-AUROC, P-AUPR, P-F1max)        per category, then averaged
    S_zs  = combined image+pixel on unseen classes
    S     = 100 * (0.3*S_cls + 0.5*S_seg + 0.2*S_zs)

On the A board there are no unseen categories, so S_zs collapses onto the
seen-category segmentation performance and the board behaves as

    S_A = 30*S_cls + 70*S_seg

Verified against actual submissions:
    clip_vlm_v1    S_cls 0.8881  S_seg 0.4649  ->  59.190  (actual 59.1921)
    vlm_proto_v1   S_cls 0.7203  S_seg 0.4604  ->  53.838  (actual 53.8373)
    dinov2_vlm_v2  S_cls 0.7470  S_seg 0.3944  ->  50.017  (actual 50.0167)

The proxy validation set is built only from official Train normal images
with synthetic defects, so no test label is ever touched.  Absolute proxy
numbers are optimistic relative to the real board because synthetic defects
are easier than real ones; the harness is for *relative* comparison of
configurations, especially localisation sharpness.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# base metrics
# --------------------------------------------------------------------------


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic)."""
    y = np.asarray(labels).astype(np.int8).ravel()
    s = np.asarray(scores, np.float64).ravel()
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks within tie groups
    su = np.sort(s)
    start = 0
    for i in range(1, len(su) + 1):
        if i == len(su) or su[i] != su[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _pr_curve(labels: np.ndarray, scores: np.ndarray):
    y = np.asarray(labels).astype(np.int8).ravel()
    s = np.asarray(scores, np.float64).ravel()
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # keep only the last index of each group of equal scores
    keep = np.r_[np.diff(s) != 0, True]
    tp = tp[keep]
    fp = fp[keep]
    total_pos = int(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(total_pos, 1)
    return precision, recall, total_pos


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP as the step-wise sum used by sklearn's average_precision_score."""
    precision, recall, total_pos = _pr_curve(labels, scores)
    if total_pos == 0:
        return float("nan")
    prev = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev)
        prev = r
    return float(ap)


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best achievable F1 over all thresholds."""
    precision, recall, total_pos = _pr_curve(labels, scores)
    if total_pos == 0:
        return float("nan")
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return float(f1.max())


# --------------------------------------------------------------------------
# competition aggregation
# --------------------------------------------------------------------------


def category_scores(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: np.ndarray | None,
    pixel_scores: np.ndarray | None,
) -> dict:
    """All five metrics for a single category."""
    out = {
        "i_auroc": auroc(image_labels, image_scores),
        "i_ap": average_precision(image_labels, image_scores),
        "p_auroc": float("nan"),
        "p_ap": float("nan"),
        "p_f1max": float("nan"),
    }
    if pixel_labels is not None and pixel_labels.size and pixel_labels.max() > 0:
        out["p_auroc"] = auroc(pixel_labels, pixel_scores)
        out["p_ap"] = average_precision(pixel_labels, pixel_scores)
        out["p_f1max"] = f1_max(pixel_labels, pixel_scores)
    return out


def aggregate(per_category: dict[str, dict]) -> dict:
    """Macro-average across categories and apply the A-board formula."""

    def mac(field):
        vals = [v[field] for v in per_category.values() if not np.isnan(v[field])]
        return float(np.mean(vals)) if vals else float("nan")

    i_auroc, i_ap = mac("i_auroc"), mac("i_ap")
    p_auroc, p_ap, p_f1 = mac("p_auroc"), mac("p_ap"), mac("p_f1max")
    s_cls = float(np.nanmean([i_auroc, i_ap]))
    s_seg = float(np.nanmean([p_auroc, p_ap, p_f1]))
    return {
        "i_auroc": i_auroc,
        "i_ap": i_ap,
        "p_auroc": p_auroc,
        "p_ap": p_ap,
        "p_f1max": p_f1,
        "s_cls": s_cls,
        "s_seg": s_seg,
        # A board: no unseen categories, S_zs tracks S_seg
        "score_a": float(30.0 * s_cls + 70.0 * s_seg),
        # full-board estimate if the same quality carried to unseen classes
        "score_b_if_zs_equal": float(100.0 * (0.3 * s_cls + 0.5 * s_seg + 0.2 * s_seg)),
    }


def format_report(agg: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(
        "  I-AUROC %.4f  I-AP %.4f  ->  S_cls %.4f  (%.2f / 30)"
        % (agg["i_auroc"], agg["i_ap"], agg["s_cls"], 30 * agg["s_cls"])
    )
    lines.append(
        "  P-AUROC %.4f  P-AP %.4f  P-F1max %.4f  ->  S_seg %.4f  (%.2f / 70)"
        % (agg["p_auroc"], agg["p_ap"], agg["p_f1max"], agg["s_seg"], 70 * agg["s_seg"])
    )
    lines.append("  SCORE_A %.4f" % agg["score_a"])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# pixel metric accumulation with subsampling
# --------------------------------------------------------------------------


class PixelAccumulator:
    """Collects pixel labels/scores for one category with bounded memory.

    750 samples x 5 views x 448^2 is 7.5e8 pixels, too much to hold.  We keep
    every positive pixel and a fixed-stride subsample of negatives, then
    reweight is unnecessary because AUROC/AP/F1 computed on a uniformly
    subsampled negative set are unbiased estimates up to the sampling ratio.

    IMPORTANT: subsampling negatives changes the positive:negative ratio,
    which shifts AP and F1max (they are prevalence dependent).  To keep the
    proxy comparable to the real board we therefore subsample negatives by a
    *fixed deterministic stride* and record the ratio, and we compare
    configurations only against each other under the same stride.
    """

    def __init__(self, neg_stride: int = 17, max_pos: int = 4_000_000):
        self.neg_stride = neg_stride
        self.max_pos = max_pos
        self.pos: list[np.ndarray] = []
        self.neg: list[np.ndarray] = []
        self.n_pos = 0

    def add(self, score_map: np.ndarray, gt_mask: np.ndarray) -> None:
        s = np.asarray(score_map, np.float32).ravel()
        g = np.asarray(gt_mask).ravel() > 0
        if g.any() and self.n_pos < self.max_pos:
            p = s[g]
            self.pos.append(p.astype(np.float32))
            self.n_pos += len(p)
        n = s[~g]
        self.neg.append(n[:: self.neg_stride].astype(np.float32))

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.pos:
            return np.zeros(0, np.int8), np.zeros(0, np.float32)
        pos = np.concatenate(self.pos)
        neg = np.concatenate(self.neg) if self.neg else np.zeros(0, np.float32)
        labels = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)]
        scores = np.r_[pos, neg]
        return labels, scores


# --------------------------------------------------------------------------
# proxy validation split
# --------------------------------------------------------------------------


def split_train(samples, holdout_frac: float = 0.2, seed: int = 20260814):
    """Per-category split of Train into (fit, proxy) groups.

    Split is per category so every category appears in both halves; the
    proxy metrics are macro-averaged over categories exactly like the board.
    """
    by_cat: dict[str, list] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    rng = np.random.default_rng(seed)
    fit, proxy = [], []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda s: s.name)
        idx = rng.permutation(len(items))
        n_hold = max(1, int(round(len(items) * holdout_frac)))
        hold = set(idx[:n_hold].tolist())
        for i, item in enumerate(items):
            (proxy if i in hold else fit).append(item)
    return fit, proxy
