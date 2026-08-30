#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from PIL import Image

from region_ranker import (
    Candidate, candidate_target, fit_ranker, generate_candidates, score_candidates,
    select_candidates,
)


def candidate(probability: float, y0: int, x0: int, size: int = 12) -> Candidate:
    mask = np.zeros((64, 64), bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return Candidate(
        mask=mask, features=np.zeros(22, np.float32), area=float(mask.mean()),
        peak=probability, box=(x0, y0, x0 + size, y0 + size), probability=probability,
    )


def main() -> None:
    # Normal views must be allowed to return no region: Top-1 is not forced.
    assert select_candidates([candidate(0.49, 5, 5)], reject_threshold=0.55) == []

    # Keep the strongest independent candidates, remove nested duplicates.
    strong = candidate(0.91, 5, 5)
    nested = candidate(0.88, 7, 7)
    other = candidate(0.80, 38, 38, size=8)
    selected = select_candidates([nested, other, strong], reject_threshold=0.55, max_area=0.10)
    assert selected[0] is strong
    assert any(item is other for item in selected)
    assert all(item is not nested for item in selected)

    truth = np.zeros((64, 64), np.float32)
    truth[5:17, 5:17] = 1.0
    assert candidate_target(strong.mask, truth) > 0.99
    assert candidate_target(other.mask, truth) == 0.0

    # A compact peak and a broad outline generate distinct region features.
    dense = np.zeros((64, 64), np.float32)
    dense[25:30, 25:30] = 0.9
    dense[8:56, 8] = 0.25
    dense[8:56, 55] = 0.25
    dense[8, 8:56] = 0.25
    dense[55, 8:56] = 0.25
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    proposals = generate_candidates(dense, [dense, dense * 0.8], image)
    assert proposals and all(value.features.shape == (22,) for value in proposals)
    assert all(np.isfinite(value.features).all() for value in proposals)

    # Full optimiser and inference smoke test, including view-level rejection
    # calibration. This catches feature-dimension and checkpoint regressions.
    rng = np.random.default_rng(7)
    features = rng.normal(size=(64, 22)).astype(np.float32)
    targets = np.zeros(64, np.float32)
    targets[:16] = 0.8
    features[:16, 5:10] += 2.5
    groups = np.repeat(np.arange(16), 4)
    normal = rng.normal(size=(24, 22)).astype(np.float32)
    normal_views = np.repeat(np.arange(8), 3)
    model, metadata = fit_ranker(
        features, targets, groups, normal, normal_view_ids=normal_views, epochs=2
    )
    scored = score_candidates([candidate(0.0, 4, 4)], model, metadata)
    assert 0.0 <= scored[0].probability <= 1.0
    assert 0.48 <= metadata["reject_threshold"] <= 0.995
    print("V12_REGION_RANKER_TESTS_OK")


if __name__ == "__main__":
    main()
