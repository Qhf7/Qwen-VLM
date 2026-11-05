"""Exact histogram AUROC/AP for BF16 probability values, implemented in NumPy.

BF16 numbers stored in float32 have zero low 16 mantissa bits. Nonnegative
float32 bit patterns sort in numeric order, so at most 16,257 bins cover [0,1].
The histogram combines every score tie before computing either metric. This
avoids sorting millions of pixel values and retains noninterpolated AP.

``ranking(labels, scores)`` returns the same single-class boundary values as
evaluate_models.ranking: AUROC=None for one class and AP=None without positives.
Unsupported finite scores use sklearn by default; pass unsupported="raise" to
require the histogram path. The supported fast path imports only NumPy.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


LAST_PROBABILITY_BIN = 0x3F80  # float32 1.0 == uint32 0x3f800000


def _fallback(labels, scores, unsupported, reason):
    if unsupported == "raise":
        raise ValueError("BF16 histogram ranking is unsupported: " + reason)
    if unsupported != "sklearn":
        raise ValueError("unsupported must be 'sklearn' or 'raise'")
    from sklearn.metrics import average_precision_score, roc_auc_score
    positives = int(np.count_nonzero(labels))
    return {
        "auroc": float(roc_auc_score(labels, scores)) if 0 < positives < labels.size else None,
        "ap": float(average_precision_score(labels, scores)) if positives else None,
    }


def ranking(labels, scores, *, unsupported="sklearn"):
    """Compute exact binary AUROC and noninterpolated AP with score ties.

    The fast path requires float32 values in [0,1] that are exactly BF16 values.
    Signed negative zero is accepted and merged with positive zero. NaN/Inf,
    nonbinary labels and unequal lengths are errors rather than fallback cases.
    Inputs are flattened, allowing pixel images or already flattened vectors.
    """
    if unsupported not in {"sklearn", "raise"}:
        raise ValueError("unsupported must be 'sklearn' or 'raise'")
    label_values = np.asarray(labels).reshape(-1)
    score_values = np.asarray(scores).reshape(-1)
    if label_values.size != score_values.size:
        raise ValueError("Labels and scores must contain the same number of values")
    if not np.all((label_values == 0) | (label_values == 1)):
        raise ValueError("Labels must be binary 0/1")
    if not np.isfinite(score_values).all():
        raise ValueError("Ranking scores contain non-finite values")
    if not score_values.size:
        return {"auroc": None, "ap": None}
    if score_values.dtype != np.float32:
        return _fallback(label_values, score_values, unsupported, "scores must have dtype float32")
    # Real pipeline map files use native float32; contiguous conversion is a
    # no-op in that case. It also makes uint32 views safe for strided arrays.
    score_values = np.ascontiguousarray(score_values)
    if np.any(score_values < 0) or np.any(score_values > 1):
        return _fallback(label_values, score_values, unsupported, "scores must lie in [0,1]")
    bits = score_values.view(np.uint32)
    if np.any(bits & np.uint32(0x0000FFFF)):
        return _fallback(label_values, score_values, unsupported, "score low 16 float32 bits are nonzero")
    # Remove sign bit after checking nonnegative values, merging signed zeros.
    codes = ((bits & np.uint32(0x7FFFFFFF)) >> np.uint32(16)).astype(np.int32)
    positive = label_values == 1
    counts = np.bincount(codes, minlength=LAST_PROBABILITY_BIN + 1)
    positives = np.bincount(codes[positive], minlength=LAST_PROBABILITY_BIN + 1)
    negatives = counts - positives
    n_positive = int(positive.sum())
    n_negative = int(label_values.size - n_positive)
    auroc = None
    ap = None
    if n_positive and n_negative:
        # Each positive wins against all lower-score negatives and receives
        # half credit against negatives with the same score (Mann–Whitney U).
        negative_below = np.cumsum(negatives, dtype=np.int64) - negatives
        favorable = np.dot(positives.astype(np.float64),
                           negative_below.astype(np.float64) + 0.5 * negatives)
        auroc = float(favorable / float(n_positive) / float(n_negative))
    if n_positive:
        # At each occupied score, recall increases by positives_at_score/P;
        # precision includes all positives and negatives tied at that score.
        selected = counts > 0
        positive_desc = positives[selected][::-1]
        count_desc = counts[selected][::-1]
        cumulative_positive = np.cumsum(positive_desc, dtype=np.int64)
        cumulative_count = np.cumsum(count_desc, dtype=np.int64)
        precision = cumulative_positive.astype(np.float64) / cumulative_count
        ap = float(np.dot(positive_desc.astype(np.float64) / float(n_positive), precision))
    return {"auroc": auroc, "ap": ap}


def _bf16_truncate(values):
    """Test helper: truncate float32 mantissa to BF16 without importing torch."""
    array = np.asarray(values, dtype=np.float32).copy()
    bits = array.view(np.uint32)
    bits &= np.uint32(0xFFFF0000)
    return array


def self_test():
    """Compare histogram results against sklearn across ties and boundaries."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    cases = []
    rng = np.random.default_rng(20261026)
    for size in (1, 2, 17, 999, 100_003):
        for prevalence in (0.01, 0.25, 0.5, 0.99):
            labels = (rng.random(size) < prevalence).astype(np.uint8)
            values = _bf16_truncate(rng.random(size, dtype=np.float32))
            cases.append((labels, values))
    for size in (1, 2, 101, 100_001):
        for value in (0.0, 0.5, 1.0):
            labels = rng.integers(0, 2, size=size, dtype=np.uint8)
            cases.append((labels, np.full(size, value, dtype=np.float32)))
        values = rng.choice(np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32), size=size)
        cases.extend([(np.zeros(size, dtype=np.uint8), values),
                      (np.ones(size, dtype=np.uint8), values)])
    cases.extend([
        (np.array([], dtype=np.uint8), np.array([], dtype=np.float32)),
        (np.array([0, 1, 1, 0]), np.array([-0.0, 0.0, 1.0, 1.0], dtype=np.float32)),
        (np.array([0, 1, 0, 1]), np.array([0, 0.25, 0.5, 1], dtype=np.float32)[::-1]),
    ])
    maximum_error = 0.0
    for labels, values in cases:
        actual = ranking(labels, values, unsupported="raise")
        positives = int(labels.sum())
        expected = {
            "auroc": float(roc_auc_score(labels, values)) if 0 < positives < len(labels) else None,
            "ap": float(average_precision_score(labels, values)) if positives else None,
        }
        for key in expected:
            if expected[key] is None:
                assert actual[key] is None, (key, expected, actual)
            else:
                error = abs(expected[key] - actual[key])
                maximum_error = max(maximum_error, error)
                assert error <= 1e-12, (key, error, expected, actual)
    for unsupported_values in (np.array([0.1, 0.9], dtype=np.float32),
                               np.array([-1, 2], dtype=np.float32),
                               np.array([0, 1], dtype=np.float64)):
        labels = np.array([0, 1], dtype=np.uint8)
        assert ranking(labels, unsupported_values) == {"auroc": 1.0, "ap": 1.0}
        try:
            ranking(labels, unsupported_values, unsupported="raise")
            raise AssertionError("Unsupported values were accepted")
        except ValueError as error:
            assert "unsupported" in str(error)
    for labels, values in (([0, 1], [0, np.nan]), ([0, 1], [0, np.inf]),
                           ([0, 2], [0, 1]), ([0], [0, 1])):
        try:
            ranking(labels, np.asarray(values, dtype=np.float32))
            raise AssertionError("Invalid input was accepted")
        except ValueError:
            pass
    print(f"SELF_TEST_PASSED cases={len(cases)} max_abs_error={maximum_error:.3g} tolerance=1e-12")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", type=int, default=0, metavar="NUM_PIXELS")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.benchmark:
        if args.benchmark < 1:
            parser.error("benchmark count must be positive")
        rng = np.random.default_rng(42)
        labels = (rng.random(args.benchmark) < 0.02).astype(np.uint8)
        values = _bf16_truncate(rng.random(args.benchmark, dtype=np.float32))
        started = time.perf_counter()
        result = ranking(labels, values, unsupported="raise")
        print(f"BENCHMARK pixels={args.benchmark} seconds={time.perf_counter() - started:.6f} metrics={result}")
    if not args.self_test and not args.benchmark:
        parser.print_help()


if __name__ == "__main__":
    main()
