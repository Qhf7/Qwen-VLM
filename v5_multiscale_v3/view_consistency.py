"""Confidence-level consistency for five non-registered camera views."""
from __future__ import annotations

import numpy as np


def consistent_view_gates(view_maps: list[np.ndarray], low=0.18, strong=0.55):
    """Return per-view scalar gates without assuming pixel correspondence.

    Cameras observe different surfaces, so masks must never be aligned pixel by
    pixel. We compare only robust anomaly evidence and area. A lone weak view is
    suppressed; a strong view remains valid; support from another view boosts
    confidence modestly.
    """
    evidence = []
    for m in view_maps:
        x = np.asarray(m, np.float32).reshape(-1)
        k = max(8, int(len(x) * 0.002))
        top = np.partition(x, len(x) - k)[-k:]
        evidence.append(float(top.mean()))
    support = sum(v >= low for v in evidence)
    gates = []
    for v in evidence:
        if v >= strong:
            gates.append(1.0)
        elif support >= 2:
            gates.append(1.05)
        elif v < low:
            gates.append(0.55)
        else:
            gates.append(0.78)
    return gates, evidence


def apply_view_consistency(view_maps: list[np.ndarray]):
    gates, evidence = consistent_view_gates(view_maps)
    return [np.clip(m * g, 0, 1) for m, g in zip(view_maps, gates)], evidence
