#!/usr/bin/env python3
"""Score-map fusion, sharpening, and image-level aggregation.

This module is where most of the P-AUPR / P-F1max gain comes from, so the
reasoning is spelled out.

Diagnosis of the 59.19 submission
---------------------------------
Measured on its own masks: among the 60 highest-scoring samples, 16.2% of
mask pixels exceed 0.5.  At 448x448 that is ~32000 pixels, while a real
defect is 200-2000 pixels.  Precision is therefore pinned near 0.01-0.06,
which caps P-AUPR at 0.23 and P-F1max at 0.29 even though P-AUROC is 0.877.
AUROC tolerates this because it only needs anomalous pixels to outrank
normal ones on average; AP and F1 do not.

Four mechanisms address it, in order of expected effect:

1. high-pass filtering        removes broad systematic offsets (glare, edges,
                              dark background) that are locally uniform and
                              cannot be a small defect
2. image-level gating         a sample the classifier is confident is normal
                              should not contribute bright pixels at all;
                              with S_cls already 0.888 this signal is strong
3. per-category calibration   metrics are macro-averaged, so only in-category
                              ranking matters and a per-category affine map
                              is free
4. gamma sharpening           compresses the mid-range that produces most
                              false positives while preserving peaks

All parameters are selected on the local proxy set built from Train normals.
"""
from __future__ import annotations

import numpy as np

from common import MASK_SIZE, gaussian_blur, highpass, resize_map, topk_mean


# --------------------------------------------------------------------------
# per-branch map preparation
# --------------------------------------------------------------------------


def prepare_branch_map(
    raw: np.ndarray,
    out_size: int = MASK_SIZE,
    highpass_frac: float = 0.14,
    pre_blur: float = 0.0,
    upsample: str = "bilinear",
) -> np.ndarray:
    """Take a raw patch-grid score map to full resolution.

    High-pass filtering is applied on the *patch grid* rather than after
    upsampling: at 74x74 the kernel covers real spatial context, and doing it
    before interpolation avoids amplifying interpolation ringing.
    """
    x = np.asarray(raw, np.float32)
    if pre_blur > 0:
        x = gaussian_blur(x, pre_blur)
    if highpass_frac > 0:
        x = highpass(x, highpass_frac)
    return resize_map(x, out_size, mode=upsample)


def combine_scales(maps: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Combine per-scale maps that are already at the same output size.

    Multiplicative-leaning combination (weighted geometric-ish mean via
    weighted sum of a fine map and the max of coarse ones) was tested against
    plain weighted sum on the proxy set; the caller chooses via weights.
    """
    if not maps:
        return np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
    if weights is None:
        weights = [1.0 / len(maps)] * len(maps)
    total = float(sum(weights)) or 1.0
    out = np.zeros_like(np.asarray(maps[0], np.float32))
    for m, w in zip(maps, weights):
        out += np.asarray(m, np.float32) * (w / total)
    return out


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def fuse_branches(
    recon: np.ndarray | None,
    memory: np.ndarray | None,
    pseudo: np.ndarray | None,
    weights=(0.40, 0.35, 0.25),
    mode: str = "weighted",
) -> np.ndarray:
    """Fuse the calibrated branch maps.

    'weighted'  linear combination; safest, keeps recall
    'gated'     linear combination multiplied by a soft agreement term, so a
                pixel needs support from more than one branch to stay bright.
                This trades a little recall for precision, which is the right
                direction for P-AUPR and P-F1max.
    """
    parts, ws = [], []
    for m, w in zip((recon, memory, pseudo), weights):
        if m is not None:
            parts.append(np.asarray(m, np.float32))
            ws.append(float(w))
    if not parts:
        return np.zeros((MASK_SIZE, MASK_SIZE), np.float32)
    total = sum(ws) or 1.0
    linear = np.zeros_like(parts[0])
    for m, w in zip(parts, ws):
        linear += m * (w / total)
    if mode == "weighted" or len(parts) == 1:
        return np.clip(linear, 0, 1)
    stack = np.stack(parts, axis=0)
    if mode == "gated":
        # agreement = geometric mean lifted toward 1 so a single strong branch
        # is attenuated but not erased
        geo = np.exp(np.mean(np.log(np.clip(stack, 1e-4, 1.0)), axis=0))
        agree = 0.55 + 0.45 * (geo / np.maximum(linear, 1e-6)).clip(0, 1)
        return np.clip(linear * agree, 0, 1)
    if mode == "max":
        return np.clip(stack.max(axis=0), 0, 1)
    raise ValueError(mode)


# --------------------------------------------------------------------------
# sharpening and gating
# --------------------------------------------------------------------------


def sharpen(
    score: np.ndarray,
    gamma: float = 2.0,
    floor: float = 0.0,
    post_blur: float = 0.6,
) -> np.ndarray:
    """Suppress the mid-range while preserving peaks.

    gamma > 1 on a [0,1] map pushes ambiguous mid values (the bulk of false
    positives) toward zero much faster than it lowers confident peaks.  The
    small post-blur removes isolated single-pixel responses, which cost
    precision without contributing recall.
    """
    x = np.clip(np.asarray(score, np.float32), 0.0, 1.0)
    if floor > 0:
        x = np.clip((x - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        x = x**gamma
    if post_blur > 0:
        x = gaussian_blur(x, post_blur)
    return np.clip(x, 0.0, 1.0)


def apply_image_gate(score: np.ndarray, image_conf: float, strength: float = 0.75) -> np.ndarray:
    """Scale a mask by image-level confidence.

    With S_cls = 0.888 the image-level ranking is far more reliable than the
    pixel maps.  Because every pixel metric is computed over the whole
    category (normal images included), pulling down masks of confidently
    normal samples raises precision at every threshold.

    strength=0 disables gating; strength=1 makes a fully-confident-normal
    sample produce an all-zero mask.  A residual floor is kept so that a
    misclassified anomaly is not zeroed out entirely, which would cost
    recall that no threshold can recover.
    """
    if strength <= 0:
        return score
    conf = float(np.clip(image_conf, 0.0, 1.0))
    gate = (1.0 - strength) + strength * conf
    return np.clip(np.asarray(score, np.float32) * gate, 0.0, 1.0)


def area_prior(score: np.ndarray, target_frac: float = 0.02, hardness: float = 1.0) -> np.ndarray:
    """Down-weight maps whose bright area is implausibly large.

    A view whose "defect" covers 20% of the frame is almost certainly a
    calibration artefact, not a defect.  This is a deterministic function of
    the map itself (no labels, no test statistics), and it directly attacks
    the failure mode measured in the 59.19 submission.
    """
    if hardness <= 0:
        return score
    x = np.asarray(score, np.float32)
    frac = float((x > 0.5).mean())
    if frac <= target_frac:
        return x
    penalty = (target_frac / max(frac, 1e-6)) ** hardness
    return np.clip(x * penalty, 0.0, 1.0)


# --------------------------------------------------------------------------
# image-level aggregation
# --------------------------------------------------------------------------


def view_score_from_map(raw_map: np.ndarray, topk_frac: float = 0.002) -> float:
    """Image-level evidence from one view's raw (uncalibrated) score map."""
    return topk_mean(raw_map, frac=topk_frac)


def aggregate_views(view_scores: list[float], max_w: float = 0.65) -> float:
    """Combine five view scores.

    A defect is usually visible from a subset of views, so max carries the
    signal; the mean stabilises normal groups where one view happens to have
    a noisy peak.  Second-highest is included because a real defect is often
    visible in two adjacent views, and requiring that suppresses single-view
    noise.
    """
    if not view_scores:
        return 0.0
    v = sorted((float(x) for x in view_scores), reverse=True)
    top = v[0]
    second = v[1] if len(v) > 1 else v[0]
    mean = float(np.mean(v))
    return float(max_w * top + (1.0 - max_w) * (0.5 * second + 0.5 * mean))


def combine_image_scores(parts: dict[str, float], weights: dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        if k in parts and np.isfinite(parts[k]):
            num += w * float(parts[k])
            den += w
    return float(num / den) if den > 0 else 0.0
