"""Category-balanced normal calibration shared by proxy and final inference."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def stratified_samples(samples, per_category: int, seed: int):
    by_category = defaultdict(list)
    for sample in samples:
        by_category[sample.category].append(sample)
    rng = np.random.default_rng(seed)
    selected = []
    for category in sorted(by_category):
        values = sorted(by_category[category], key=lambda sample: sample.name)
        count = min(len(values), per_category)
        indices = np.sort(rng.choice(len(values), count, replace=False))
        selected.extend(values[int(index)] for index in indices)
    return selected


@dataclass
class RobustBounds:
    centre: float
    high: float

    def apply(self, value):
        scale = max(self.high - self.centre, 1e-5)
        # Strictly monotonic: centre -> 0.10, high -> 0.90.  Unlike a clipped
        # min-max transform this preserves ranking above the normal high tail,
        # which is essential for AUROC/AP and avoids v4's 0/1 saturation.
        z = 4.394449 * (np.asarray(value, np.float32) - self.centre) / scale - 2.1972246
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class V5Calibration:
    pixel_by_category: dict[str, RobustBounds]
    image_by_category: dict[str, RobustBounds]
    pixel_global: RobustBounds
    image_global: RobustBounds
    categories: tuple[str, ...]

    def pixel(self, category: str, value):
        return self.pixel_by_category.get(category, self.pixel_global).apply(value)

    def image(self, category: str, value):
        return self.image_by_category.get(category, self.image_global).apply(value)


def _bounds(values, lo_q: float, hi_q: float) -> RobustBounds:
    values = np.asarray(values, np.float32)
    return RobustBounds(float(np.quantile(values, lo_q)), float(np.quantile(values, hi_q)))


def fit_calibration(pixel_values, image_values, pixel_lo=0.90, pixel_hi=0.9995):
    categories = sorted(set(pixel_values) | set(image_values))
    pixel_category = {
        category: _bounds(np.concatenate(pixel_values[category]), pixel_lo, pixel_hi)
        for category in categories
        if pixel_values.get(category)
    }
    image_category = {
        category: _bounds(image_values[category], 0.50, 0.995)
        for category in categories
        if image_values.get(category)
    }
    pixel_global = _bounds(
        np.concatenate([np.concatenate(pixel_values[key]) for key in sorted(pixel_values)]),
        pixel_lo,
        pixel_hi,
    )
    image_global = _bounds(
        np.concatenate([np.asarray(image_values[key], np.float32) for key in sorted(image_values)]),
        0.50,
        0.995,
    )
    return V5Calibration(
        pixel_by_category=pixel_category,
        image_by_category=image_category,
        pixel_global=pixel_global,
        image_global=image_global,
        categories=tuple(categories),
    )


def assert_category_coverage(calibration: V5Calibration, expected_categories) -> None:
    expected = set(expected_categories)
    missing_pixel = expected - set(calibration.pixel_by_category)
    missing_image = expected - set(calibration.image_by_category)
    if missing_pixel or missing_image:
        raise RuntimeError(
            f"category calibration incomplete: pixel={sorted(missing_pixel)} image={sorted(missing_image)}"
        )
