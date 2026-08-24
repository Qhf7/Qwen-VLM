"""Type-aware pseudo anomalies and 12-state pixel supervision.

Only official Train images are used.  Every label is known from the local
procedural operation; no online VLM or external anomaly dataset is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from hard_negatives import hard_negative_map
from object_prior import foreground_mask
from pseudo_base import synthesize
from states import (
    BACKGROUND,
    DEFECT_GENERATORS,
    GENERATOR_TO_STATE,
    NORMAL_EDGE,
    NORMAL_REFLECTION,
    NORMAL_SURFACE,
)


@dataclass
class ViewTarget:
    image: Image.Image
    original: Image.Image
    binary_mask: np.ndarray
    state_mask: np.ndarray
    defect_state: int
    is_anomaly: float
    severity: float


def _normal_states(image: Image.Image, size: int = 448) -> np.ndarray:
    rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    gray = rgb.mean(-1)
    fg = foreground_mask(image, size)
    hard = hard_negative_map(image, size)
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    edge_thr = max(float(np.quantile(grad[fg > 0.25], 0.78)) if (fg > 0.25).any() else 0.02, 1e-4)
    edge = (grad >= edge_thr) & (fg > 0.20)
    saturation = rgb.max(-1) - rgb.min(-1)
    brightness = rgb.max(-1)
    bright_thr = float(np.quantile(brightness[fg > 0.25], 0.88)) if (fg > 0.25).any() else 0.85
    reflection = (brightness >= bright_thr) & (saturation < 0.18) & (fg > 0.25)

    state = np.full((size, size), BACKGROUND, np.int64)
    state[fg > 0.22] = NORMAL_SURFACE
    state[edge | ((hard > 0.72) & (fg > 0.20))] = NORMAL_EDGE
    state[reflection] = NORMAL_REFLECTION
    return state


def make_view_target(
    image: Image.Image,
    rng: np.random.Generator,
    anomalous: bool,
    size: int = 448,
    forced_kind: str | None = None,
) -> ViewTarget:
    original = image.convert("RGB")
    normal_state = _normal_states(original, size)
    if not anomalous:
        return ViewTarget(
            image=original.copy(),
            original=original,
            binary_mask=np.zeros((size, size), np.float32),
            state_mask=normal_state,
            defect_state=-1,
            is_anomaly=0.0,
            severity=0.0,
        )

    kind = forced_kind or str(rng.choice(DEFECT_GENERATORS))
    # A hairline can represent either a scratch or a crack.  This creates two
    # language states without requiring an external texture or annotation.
    generator_kind = "hairline" if kind == "crack" else kind
    defective, native_mask = synthesize(original, rng, kind=generator_kind, n_defects=(1, 3))
    mask = np.asarray(native_mask.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    # Keep soft boundary supervision for the binary loss and a conservative
    # core for the discrete state loss.
    core = mask > 0.20
    state = normal_state.copy()
    defect_state = GENERATOR_TO_STATE[kind]
    state[core] = defect_state
    severity = float(np.clip(mask.mean() / 0.025, 0.05, 1.0))
    return ViewTarget(defective, original, mask, state, defect_state, 1.0, severity)


def make_group_targets(
    images: list[Image.Image],
    rng: np.random.Generator,
    size: int = 448,
    force_normal: bool = False,
) -> list[ViewTarget]:
    if len(images) != 5:
        raise ValueError(f"expected five views, got {len(images)}")
    if force_normal:
        selected: set[int] = set()
    else:
        count = int(rng.integers(1, 4))
        selected = set(int(x) for x in rng.choice(5, count, replace=False))
    # Usually use one defect concept across affected views, but sometimes
    # sample independently to make the five-view module robust to mixed flaws.
    shared = str(rng.choice(DEFECT_GENERATORS))
    out = []
    for view, image in enumerate(images):
        kind = shared if rng.random() < 0.75 else str(rng.choice(DEFECT_GENERATORS))
        out.append(make_view_target(image, rng, view in selected, size=size, forced_kind=kind))
    return out
