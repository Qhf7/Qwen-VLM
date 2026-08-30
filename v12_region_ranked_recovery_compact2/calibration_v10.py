"""Small, independently testable calibration primitives for v10."""
from __future__ import annotations

import numpy as np


def apply_pixel_calibration(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a branch map from the Train-normal quantile band to ``[0, 1]``.

    ``lo`` is the normal-response baseline and must be subtracted before
    range scaling.  This is deliberately separate from model code so the
    calibration offset cannot silently regress in a future inference edit.
    """
    lower = float(lo)
    upper = max(float(hi), lower + 1e-6)
    return np.clip(
        (np.asarray(values, np.float32) - lower) / (upper - lower),
        0.0,
        1.0,
    )
