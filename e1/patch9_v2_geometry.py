"""Independent Patch9 geometry pipeline for offline audit and later training.

This module intentionally does not import the model or loss code.  It keeps the
full image as context, creates nine overlapping crops from the original
resolution image, applies exactly the same crop to the mask, and maps patch
probabilities back to an output grid.  It is a data/geometry change only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image


DEFAULT_OUTPUT = 448


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


@dataclass(frozen=True)
class PatchSpec:
    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _grid_starts(length: int, window: int, overlap: float) -> tuple[int, int, int]:
    if length < 1 or not 0 < window <= length or not 0 <= overlap < 1:
        raise ValueError(f"invalid grid: length={length}, window={window}, overlap={overlap}")
    stride = max(1, int(round(window * (1.0 - overlap))))
    # Force exact image-edge coverage.  The middle start is derived from the
    # two edge starts, rather than accumulating floating-point fractions.
    last = length - window
    middle = min(last, max(0, stride))
    starts = tuple(sorted({0, middle, last}))
    if len(starts) != 3:
        starts = (0, last // 2, last)
    return starts


def make_patch_specs(width: int, height: int, *, fraction: float = 0.4,
                     overlap: float = 0.25) -> list[PatchSpec]:
    """Return row-major nine boxes on the original-resolution image.

    The boxes always touch all four image borders.  Coordinates are half-open
    and are therefore safe for NumPy slicing and exact mask projection.
    """
    if width < 1 or height < 1 or not 0 < fraction <= 1:
        raise ValueError("invalid image dimensions or fraction")
    wx = min(width, max(1, int(round(width * fraction))))
    wy = min(height, max(1, int(round(height * fraction))))
    xs = _grid_starts(width, wx, overlap)
    ys = _grid_starts(height, wy, overlap)
    specs: list[PatchSpec] = []
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            specs.append(PatchSpec(len(specs), row, col, x0, y0, x0 + wx, y0 + wy))
    if len(specs) != 9:
        raise AssertionError("Patch9 must produce exactly nine boxes")
    return specs


def _check_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")


def _check_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    if mask.ndim != 2 or tuple(mask.shape) != shape:
        raise ValueError(f"mask shape {mask.shape} does not match image {shape}")


def resize_image(image: np.ndarray, output: int = DEFAULT_OUTPUT) -> np.ndarray:
    _check_image(image)
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize(
        (output, output), Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_mask(mask: np.ndarray, output: int = DEFAULT_OUTPUT) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (output, output), Image.Resampling.NEAREST), dtype=np.uint8) > 0


def crop_image(image: np.ndarray, spec: PatchSpec, output: int = DEFAULT_OUTPUT) -> np.ndarray:
    _check_image(image)
    crop = image[spec.y0:spec.y1, spec.x0:spec.x1]
    if crop.shape[:2] != (spec.height, spec.width):
        raise ValueError("patch lies outside image")
    return resize_image(crop, output)


def crop_mask(mask: np.ndarray, spec: PatchSpec, output: int = DEFAULT_OUTPUT) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    crop = mask[spec.y0:spec.y1, spec.x0:spec.x1]
    if crop.shape != (spec.height, spec.width):
        raise ValueError("patch lies outside mask")
    return resize_mask(crop, output)


def patch_contains_mask(mask: np.ndarray, spec: PatchSpec) -> bool:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    return bool(mask[spec.y0:spec.y1, spec.x0:spec.x1].any())


def positive_boxes(mask: np.ndarray, specs: Iterable[PatchSpec]) -> list[int]:
    return [spec.index for spec in specs if patch_contains_mask(mask, spec)]


def project_mask(mask_448: np.ndarray, spec: PatchSpec,
                 original_shape: tuple[int, int]) -> np.ndarray:
    """Map one binary 448 patch mask back to original coordinates."""
    if mask_448.ndim != 2:
        raise ValueError("mask_448 must be HxW")
    height, width = original_shape
    if not (0 <= spec.x0 < spec.x1 <= width and 0 <= spec.y0 < spec.y1 <= height):
        raise ValueError("spec does not fit original shape")
    # Use the exact rectangular target so coordinates remain pixel-aligned.
    projected = np.asarray(Image.fromarray(mask_448.astype(np.uint8) * 255).resize(
        (spec.width, spec.height), Image.Resampling.NEAREST), dtype=np.uint8) > 0
    canvas = np.zeros((height, width), dtype=bool)
    canvas[spec.y0:spec.y1, spec.x0:spec.x1] = projected
    return canvas


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def records_for(image: np.ndarray, mask: np.ndarray, *, output: int = DEFAULT_OUTPUT,
                fraction: float = 0.4, overlap: float = 0.25) -> list[dict]:
    """Create nine auditable records, including empty patches.

    Empty patches are deliberately retained in the records.  A later training
    policy may sample them with empty targets; silently dropping them would
    recreate the old train/inference distribution mismatch.
    """
    _check_image(image)
    _check_mask(mask, image.shape[:2])
    specs = make_patch_specs(image.shape[1], image.shape[0], fraction=fraction, overlap=overlap)
    records = []
    for spec in specs:
        patch_target = crop_mask(mask, spec, output)
        records.append({
            "spec": asdict(spec),
            "image": crop_image(image, spec, output),
            "mask": patch_target,
            "contains_original_positive": patch_contains_mask(mask, spec),
            "patch_positive_pixels": int(patch_target.sum()),
            "patch_positive_ratio": float(patch_target.mean()),
        })
    return records


def _resize_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if probability.ndim != 2:
        raise ValueError("probability must be HxW")
    return np.asarray(Image.fromarray(np.clip(probability, 0, 1).astype(np.float32), mode="F").resize(
        (shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def project_probability(probability: np.ndarray, spec: PatchSpec,
                        original_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project one 448 patch probability to original coordinates and support."""
    height, width = original_shape
    if not (0 <= spec.x0 < spec.x1 <= width and 0 <= spec.y0 < spec.y1 <= height):
        raise ValueError("spec does not fit original shape")
    canvas = np.zeros((height, width), dtype=np.float32)
    support = np.zeros((height, width), dtype=np.float32)
    canvas[spec.y0:spec.y1, spec.x0:spec.x1] = _resize_probability(
        probability, (spec.height, spec.width))
    support[spec.y0:spec.y1, spec.x0:spec.x1] = 1.0
    return canvas, support


def fuse_probabilities(full_probability: np.ndarray,
                       patch_probabilities: Mapping[int, np.ndarray],
                       specs: Iterable[PatchSpec],
                       original_shape: tuple[int, int],
                       *, output: int = DEFAULT_OUTPUT,
                       patch_weight: float = 0.5) -> np.ndarray:
    """Conservative full-image + patch weighted-mean fusion.

    Unlike the old max fusion, a single noisy patch cannot dominate a pixel.
    The returned probability is resized to the model's 448 grid.  This is an
    inference utility; it does not alter training loss or model parameters.
    """
    if not 0 <= patch_weight <= 1:
        raise ValueError("patch_weight must be in [0, 1]")
    height, width = original_shape
    full = _resize_probability(full_probability, (height, width))
    accum = np.zeros((height, width), dtype=np.float32)
    support = np.zeros((height, width), dtype=np.float32)
    for spec in specs:
        if spec.index not in patch_probabilities:
            continue
        projected, covered = project_probability(patch_probabilities[spec.index], spec, original_shape)
        accum += projected * covered
        support += covered
    patch_mean = np.divide(accum, np.maximum(support, 1.0), dtype=np.float32)
    fused = (1.0 - patch_weight) * full + patch_weight * patch_mean
    # Outside the patch union (should only be possible for a malformed grid),
    # retain full-image evidence instead of returning zeros.
    fused = np.where(support > 0, fused, full)
    return _resize_probability(fused, (output, output))


def audit_pair(image: np.ndarray, mask: np.ndarray, *, output: int = DEFAULT_OUTPUT,
               fraction: float = 0.4, overlap: float = 0.25) -> dict:
    """Return geometry and small-object preservation statistics."""
    _check_image(image)
    _check_mask(mask, image.shape[:2])
    specs = make_patch_specs(image.shape[1], image.shape[0], fraction=fraction, overlap=overlap)
    resized_full = resize_mask(mask, output)
    records = records_for(image, mask, output=output, fraction=fraction, overlap=overlap)
    positive = [r for r in records if r["contains_original_positive"]]
    return {
        "original_size": [int(image.shape[1]), int(image.shape[0])],
        "output_size": output,
        "mask_pixels_original": int(mask.sum()),
        "mask_pixels_full_448": int(resized_full.sum()),
        "mask_ratio_original": float(mask.mean()),
        "mask_ratio_full_448": float(resized_full.mean()),
        "patch_count": len(specs),
        "positive_patch_count": len(positive),
        "empty_patch_count": len(records) - len(positive),
        "positive_patch_ratios_448": [r["patch_positive_ratio"] for r in positive],
        "all_patch_boxes": [list(s.as_tuple()) for s in specs],
    }


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    _check_image(image_array)
    _check_mask(mask_array, image_array.shape[:2])
    return image_array, mask_array
