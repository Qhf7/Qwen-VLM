"""Pseudo-anomaly plans with generator-family holdout support."""
from __future__ import annotations

import numpy as np

from pseudo_states import make_view_target
from states import DEFECT_GENERATORS


DEFAULT_TRAIN_GENERATORS = (
    "hairline", "scratch", "speck", "dent", "chip", "contamination", "cutpaste",
)
DEFAULT_HELDOUT_GENERATORS = (
    "crack", "discolour", "misalign", "hole",
)


def make_group_targets_v5(
    images,
    rng: np.random.Generator,
    size: int = 448,
    force_normal: bool = False,
    generators=DEFECT_GENERATORS,
):
    generators = tuple(generators)
    if not generators:
        raise ValueError("generators must not be empty")
    if len(images) != 5:
        raise ValueError(f"expected five views, got {len(images)}")
    if force_normal:
        selected = set()
    else:
        count = int(rng.integers(1, 4))
        selected = set(int(x) for x in rng.choice(5, count, replace=False))
    shared = str(rng.choice(generators))
    output = []
    for view, image in enumerate(images):
        kind = shared if rng.random() < 0.75 else str(rng.choice(generators))
        output.append(make_view_target(image, rng, view in selected, size=size, forced_kind=kind))
    return output
