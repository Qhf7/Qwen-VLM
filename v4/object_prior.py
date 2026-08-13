#!/usr/bin/env python3
"""Foreground/object priors used inside the v4 model pipeline.

These helpers do not read previous submission masks.  They estimate where the
industrial part is from the input image itself, then use that prior while
building memory banks, training the pseudo head, calibrating normal scores,
and producing model masks.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _rgb_array(image: Image.Image, size: int = 448) -> np.ndarray:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), np.uint8)


def foreground_mask(image: Image.Image, size: int = 448) -> np.ndarray:
    """Return a soft foreground mask in [0,1] at `size`.

    The Real-IAD Variety images often have a bright or dark studio background.
    A robust foreground estimate prevents the model from learning "object
    silhouette" as the anomaly mask and keeps non-part pixels near black.
    """
    rgb = _rgb_array(image, size)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    border = np.concatenate(
        [
            lab[:12].reshape(-1, 3),
            lab[-12:].reshape(-1, 3),
            lab[:, :12].reshape(-1, 3),
            lab[:, -12:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)
    color_dist = np.linalg.norm(lab.astype(np.float32) - bg[None, None, :], axis=2)
    sat = hsv[:, :, 1].astype(np.float32)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    score = 0.72 * (color_dist / max(np.percentile(color_dist, 98), 1.0))
    score += 0.18 * (sat / max(np.percentile(sat, 98), 1.0))
    score += 0.10 * edges
    score = np.clip(score, 0, 1)
    thr = max(0.08, min(0.35, float(np.percentile(score, 72))))
    mask = (score > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.argsort(areas)[::-1][:4] + 1
        mask = np.isin(labels, keep).astype(np.uint8)
    if mask.mean() < 0.035:
        edge_mask = cv2.dilate((edges > 0).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=2)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, 8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.argsort(areas)[::-1][:4] + 1
            mask = np.isin(labels, keep).astype(np.uint8)
    if mask.mean() > 0.82:
        strict = ((color_dist > np.percentile(color_dist, 65)) | (edges > 0)).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(strict, 8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.argsort(areas)[::-1][:4] + 1
            mask = np.isin(labels, keep).astype(np.uint8)

    mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    return np.clip(soft, 0.0, 1.0)


def foreground_grid(image: Image.Image, side: int) -> np.ndarray:
    fg = foreground_mask(image, 448)
    return cv2.resize(fg, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)


def descriptor_subset(features: np.ndarray, fg: np.ndarray, fg_stride: int = 2, bg_stride: int = 24) -> np.ndarray:
    """Select more foreground descriptors and only sparse background normals."""
    flat = features.reshape(-1, features.shape[-1])
    f = fg.reshape(-1) > 0.35
    idx_fg = np.nonzero(f)[0][:: max(1, fg_stride)]
    idx_bg = np.nonzero(~f)[0][:: max(1, bg_stride)]
    idx = np.concatenate([idx_fg, idx_bg]) if len(idx_bg) else idx_fg
    if len(idx) == 0:
        idx = np.arange(0, len(flat), max(1, fg_stride))
    return flat[idx]


def apply_prior(score: np.ndarray, image: Image.Image, background_keep: float = 0.08, boundary_keep: float = 0.55) -> np.ndarray:
    """Apply a soft foreground prior to a score map without destroying recall."""
    x = np.asarray(score, np.float32)
    fg = foreground_mask(image, x.shape[0] if x.shape[0] == x.shape[1] else 448)
    if fg.shape != x.shape:
        fg = cv2.resize(fg, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
    inner = cv2.erode((fg > 0.45).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    boundary = np.clip(fg - inner, 0, 1)
    gate = background_keep + (1.0 - background_keep) * inner + (boundary_keep - background_keep) * boundary
    return np.clip(x * gate, 0.0, 1.0)
