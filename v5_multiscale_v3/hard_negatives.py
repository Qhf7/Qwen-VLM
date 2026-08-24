"""Hard-normal maps derived only from official Train normal images."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def hard_negative_map(image: Image.Image, size: int) -> np.ndarray:
    rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    gray = rgb.mean(-1)
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)

    # Specular normal surfaces: bright, locally low-saturation pixels.
    saturation = rgb.max(-1) - rgb.min(-1)
    spec = ((rgb.max(-1) > np.quantile(rgb.max(-1), 0.91)) & (saturation < 0.16)).astype(np.float32)

    # Object/texture edges are normal hard negatives unless covered by pseudo GT.
    edge_thr = max(float(np.quantile(grad, 0.86)), 1e-4)
    edges = np.clip(grad / edge_thr, 0, 1)

    # Repeated textures produce strong local residuals but should remain normal.
    smooth = np.asarray(
        Image.fromarray(np.uint8(gray * 255), "L").filter(ImageFilter.GaussianBlur(2.0)), np.float32
    ) / 255.0
    residual = np.abs(gray - smooth)
    rep_thr = max(float(np.quantile(residual, 0.88)), 1e-4)
    texture = np.clip(residual / rep_thr, 0, 1)
    hard = np.maximum.reduce([0.75 * edges, 0.85 * spec, 0.65 * texture])
    return np.clip(hard, 0, 1).astype(np.float32)
