#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from vlm_guided_fuse_v9 import Region, fuse_dense_with_vlm, group_factor, proposal_map, score_tier


def main() -> None:
    assert [score_tier(x) for x in (0.2, 0.5, 0.65, 0.8, 0.95)] == [0.0, 0.45, 0.60, 0.75, 0.90]
    assert group_factor(1) < group_factor(2) <= group_factor(3)

    base = np.zeros((448, 448), np.float32)
    region = Region((0.40, 0.40, 0.55, 0.55), 0.9, "scratch", 0.0225, 0)
    proposal = proposal_map([region])
    fused, support = fuse_dense_with_vlm(base, proposal)
    assert support == 0.0 and np.array_equal(fused, base), "VLM must not paint an empty DINO map"

    base[200:230, 200:235] = 0.10
    fused, support = fuse_dense_with_vlm(base, proposal)
    assert support > 0.0
    assert float(fused.max()) > float(base.max())
    assert np.array_equal(fused[proposal == 0], base[proposal == 0])
    assert float((fused - base).max()) <= 0.12001

    unchanged, support = fuse_dense_with_vlm(base, np.zeros_like(base))
    assert support == 0.0 and np.array_equal(unchanged, base)
    print("V9_TESTS_OK")


if __name__ == "__main__":
    main()
