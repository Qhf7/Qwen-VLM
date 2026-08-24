"""Sparse normal-excess calibration without foreground amplification."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExcessBounds:
    threshold: float
    scale: float

    def pixel(self, value):
        z = np.maximum((np.asarray(value, np.float32) - self.threshold) / max(self.scale, 1e-6), 0.0)
        return z / (1.0 + z)

    def score(self, value):
        z = (np.asarray(value, np.float64) - self.threshold) / max(self.scale, 1e-6)
        # Strictly monotonic and asymptotic: unlike sigmoid it does not create
        # a large exact-1 plateau after CSV formatting.
        return 0.5 + np.arctan(z) / np.pi


@dataclass
class V6Calibration:
    pixel_by_category: dict[str, ExcessBounds]
    image_by_category: dict[str, ExcessBounds]
    pixel_global: ExcessBounds
    image_global: ExcessBounds

    def pixel(self, category, value):
        return self.pixel_by_category.get(category, self.pixel_global).pixel(value)

    def image(self, category, value):
        return self.image_by_category.get(category, self.image_global).score(value)


def _pixel_bounds(values, threshold_q=0.995, high_q=0.9998):
    x = np.asarray(values, np.float32)
    threshold = float(np.quantile(x, threshold_q))
    high = float(np.quantile(x, high_q))
    return ExcessBounds(threshold, max(high - threshold, 1e-4))


def _image_bounds(values):
    x = np.asarray(values, np.float64)
    centre = float(np.median(x))
    mad = float(np.median(np.abs(x - centre)))
    return ExcessBounds(centre, max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-4))


def fit_calibration(pixel_values, image_values, pixel_threshold_q=0.995, pixel_high_q=0.9998):
    categories = sorted(set(pixel_values) | set(image_values))
    pixel = {
        c: _pixel_bounds(np.concatenate(pixel_values[c]), pixel_threshold_q, pixel_high_q)
        for c in categories if pixel_values.get(c)
    }
    image = {c: _image_bounds(image_values[c]) for c in categories if image_values.get(c)}
    all_pixels = np.concatenate([np.concatenate(pixel_values[c]) for c in sorted(pixel_values)])
    all_images = np.concatenate([np.asarray(image_values[c]) for c in sorted(image_values)])
    return V6Calibration(
        pixel, image,
        _pixel_bounds(all_pixels, pixel_threshold_q, pixel_high_q),
        _image_bounds(all_images),
    )


def quantize_mask(score):
    """Return the exact float values represented by the final uint8 PNG."""
    return np.round(np.clip(np.asarray(score, np.float32), 0, 1) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
