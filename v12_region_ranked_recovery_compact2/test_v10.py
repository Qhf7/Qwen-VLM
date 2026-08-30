#!/usr/bin/env python3
"""Regression tests for the v10 calibration-only continuation."""
from __future__ import annotations

import numpy as np

from calibration_v10 import apply_pixel_calibration
from vlm_guided_fuse_v10 import Region, fuse_dense_with_vlm, proposal_map


def main() -> None:
    values = np.asarray([0.10, 0.20, 0.30, 0.40, 0.60], np.float32)
    calibrated = apply_pixel_calibration(values, lo=0.20, hi=0.40)
    np.testing.assert_allclose(calibrated, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)
    assert float(calibrated[1]) == 0.0, "normal lower-quantile baseline must map to zero"

    # Keep the v9 safety invariant: VLM evidence cannot paint an empty DINO map.
    base = np.zeros((448, 448), np.float32)
    region = Region((0.40, 0.40, 0.55, 0.55), 0.9, "scratch", 0.0225, 0)
    proposal = proposal_map([region])
    fused, support = fuse_dense_with_vlm(base, proposal)
    assert support == 0.0 and np.array_equal(fused, base)

    base[200:230, 200:235] = 0.10
    fused, support = fuse_dense_with_vlm(base, proposal)
    assert support > 0.0
    assert float(fused.max()) > float(base.max())
    assert float((fused - base).max()) <= 0.12001
    print("V10_CALIBRATION_TESTS_OK")


if __name__ == "__main__":
    main()
