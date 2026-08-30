#!/usr/bin/env python3
"""Foreground/object priors used inside the v4 model pipeline.

These helpers do not read previous submission masks.  They estimate where the
industrial part is from the input image itself, then use that prior while
building memory banks, training the pseudo head, calibrating normal scores,
and producing model masks.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _rgb_array(image: Image.Image, size: int = 448) -> np.ndarray:
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), np.uint8)


def _max_filter(x: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    arr = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    for _ in range(iterations):
        img = img.filter(ImageFilter.MaxFilter(k))
    return np.asarray(img, np.float32) / 255.0


def _min_filter(x: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    arr = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    for _ in range(iterations):
        img = img.filter(ImageFilter.MinFilter(k))
    return np.asarray(img, np.float32) / 255.0


def _gaussian(x: np.ndarray, radius: float) -> np.ndarray:
    arr = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, np.float32) / 255.0


def _resize_float(x: np.ndarray, size: tuple[int, int], resample=Image.Resampling.BILINEAR) -> np.ndarray:
    arr = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize(size, resample)
    return np.asarray(img, np.float32) / 255.0


def _open(mask: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    return _max_filter(_min_filter(mask, k, iterations), k, iterations)


def _close(mask: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    return _min_filter(_max_filter(mask, k, iterations), k, iterations)


def _keep_large_components(mask: np.ndarray, max_components: int = 4, min_area: int = 64) -> np.ndarray:
    """Small dependency-free connected-component filtering for 448x448 masks."""
    m = np.asarray(mask > 0, bool)
    h, w = m.shape
    seen = np.zeros_like(m, bool)
    components: list[list[tuple[int, int]]] = []
    ys, xs = np.nonzero(m)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= h:
                    continue
                for xx in (x - 1, x, x + 1):
                    if xx < 0 or xx >= w or seen[yy, xx] or not m[yy, xx]:
                        continue
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(comp) >= min_area:
            components.append(comp)
    if not components:
        return m.astype(np.float32)
    components.sort(key=len, reverse=True)
    out = np.zeros_like(m, np.float32)
    for comp in components[:max_components]:
        yy, xx = zip(*comp)
        out[np.asarray(yy), np.asarray(xx)] = 1.0
    return out


def foreground_mask(image: Image.Image, size: int = 448) -> np.ndarray:
    """Return a soft foreground mask in [0,1] at `size`.

    The Real-IAD Variety images often have a bright or dark studio background.
    A robust foreground estimate prevents the model from learning "object
    silhouette" as the anomaly mask and keeps non-part pixels near black.
    """
    rgb = _rgb_array(image, size).astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0) * 255.0

    border = np.concatenate(
        [
            rgb[:12].reshape(-1, 3),
            rgb[-12:].reshape(-1, 3),
            rgb[:, :12].reshape(-1, 3),
            rgb[:, -12:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    color_dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    gx = np.zeros_like(gray, np.float32)
    gy = np.zeros_like(gray, np.float32)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = gx + gy
    edges = grad / max(float(np.percentile(grad, 99)), 1.0)
    edges = _max_filter((edges > 0.18).astype(np.float32), 5, iterations=1)

    score = 0.72 * (color_dist / max(np.percentile(color_dist, 98), 1.0))
    score += 0.18 * (sat / max(np.percentile(sat, 98), 1.0))
    score += 0.10 * edges
    score = np.clip(score, 0, 1)
    thr = max(0.08, min(0.35, float(np.percentile(score, 72))))
    mask = (score > thr).astype(np.float32)
    mask = _close(mask, 11, iterations=2)
    mask = _open(mask, 5, iterations=1)
    mask = _keep_large_components(mask, max_components=4)
    if mask.mean() < 0.035:
        edge_mask = _max_filter((edges > 0).astype(np.float32), 13, iterations=2)
        edge_mask = _close(edge_mask, 19, iterations=2)
        mask = _keep_large_components(edge_mask, max_components=4)
    if mask.mean() > 0.82:
        strict = ((color_dist > np.percentile(color_dist, 65)) | (edges > 0)).astype(np.float32)
        strict = _close(strict, 9, iterations=1)
        mask = _keep_large_components(strict, max_components=4)

    mask = _max_filter(mask.astype(np.float32), 7, iterations=1)
    soft = _gaussian(mask.astype(np.float32), 2.0)
    return np.clip(soft, 0.0, 1.0)


def foreground_grid(image: Image.Image, side: int) -> np.ndarray:
    fg = foreground_mask(image, 448)
    return _resize_float(fg, (side, side), Image.Resampling.BOX).astype(np.float32)


def hard_normal_grid(image: Image.Image, side: int) -> np.ndarray:
    """Normal edges, highlights and strong texture used as negative supervision."""
    rgb = np.asarray(
        image.resize((side, side), Image.Resampling.BILINEAR).convert("RGB"), np.float32
    )
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gradient = gx + gy
    local_mean = np.asarray(
        Image.fromarray(np.uint8(np.clip(gray, 0, 255)), "L").filter(ImageFilter.GaussianBlur(1.4)),
        np.float32,
    )
    texture = np.abs(gray - local_mean)
    chroma = rgb.max(2) - rgb.min(2)
    foreground = foreground_grid(image, side) > 0.20
    edge = gradient >= np.percentile(gradient[foreground], 86) if foreground.any() else gradient > 0
    textured = texture >= np.percentile(texture[foreground], 90) if foreground.any() else texture > 0
    highlight = (gray >= np.percentile(gray[foreground], 94)) & (chroma < 42) if foreground.any() else gray > 255
    hard = (edge | textured | highlight) & foreground
    return hard.astype(np.float32)


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


def apply_prior(score: np.ndarray, image: Image.Image, background_keep: float = 0.05, boundary_keep: float = 0.38) -> np.ndarray:
    """Apply a soft foreground prior to a score map without destroying recall."""
    x = np.asarray(score, np.float32)
    fg = foreground_mask(image, x.shape[0] if x.shape[0] == x.shape[1] else 448)
    if fg.shape != x.shape:
        fg = _resize_float(fg, (x.shape[1], x.shape[0]), Image.Resampling.BILINEAR)
    inner = _min_filter((fg > 0.58).astype(np.float32), 7, iterations=1)
    boundary = np.clip(fg - inner, 0, 1)
    boundary = _gaussian(boundary, 1.2)
    # Keep the object, but strongly suppress the silhouette ring so the
    # anomaly score cannot spread along the whole part outline.
    gate = background_keep + (1.0 - background_keep) * inner + (boundary_keep - background_keep) * 0.35 * boundary
    return np.clip(x * gate, 0.0, 1.0)


def compact_small_anomaly(score: np.ndarray, target_frac: float = 0.012, softness: float = 0.7) -> np.ndarray:
    """Prefer a compact defect core over a broad part-shaped response.

    This is deliberately conservative: it only compresses maps whose bright
    area is already too large for the benchmark's tiny defect regime.
    """
    x = np.asarray(score, np.float32)
    if target_frac <= 0 or softness <= 0:
        return x
    active = float((x > 0.18).mean())
    if active <= target_frac:
        return x
    core = np.clip((x - 0.12) / 0.88, 0.0, 1.0)
    return np.clip(core ** (1.0 + softness), 0.0, 1.0)


def suppress_outline_rings(score: np.ndarray, image: Image.Image, ring_keep: float = 0.22) -> np.ndarray:
    """Suppress the silhouette-like boundary response around the object."""
    x = np.asarray(score, np.float32)
    fg = foreground_mask(image, x.shape[0] if x.shape[0] == x.shape[1] else 448)
    if fg.shape != x.shape:
        fg = _resize_float(fg, (x.shape[1], x.shape[0]), Image.Resampling.BILINEAR)
    inner = _min_filter((fg > 0.68).astype(np.float32), 9, iterations=1)
    ring = np.clip(fg - inner, 0.0, 1.0)
    gate = inner + ring_keep * ring + 0.03 * (1.0 - fg)
    return np.clip(x * gate, 0.0, 1.0)
