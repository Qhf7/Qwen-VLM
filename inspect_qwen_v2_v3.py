#!/usr/bin/env python3
"""Read-only comparison of Qwen v2 and calibrated v3 submissions."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from qwen3vl_calibrate_submission import (
    discrete_zero_score,
    mapped_box,
    prepare_view,
    text_strength,
)


ROOT = Path("/root/autodl-tmp/qwen3vl_local")
V2 = ROOT / "submission_qwen3vl_crop_v2"
V3 = ROOT / "submission_qwen3vl_crop_v3_calibrated"


def read_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["group_folder"]: float(row["anomaly_score"]) for row in csv.DictReader(f)}


def main() -> None:
    v2_scores = read_scores(V2 / "submission.csv")
    v3_scores = read_scores(V3 / "submission.csv")
    keys = sorted(v2_scores)
    diff = np.array([v3_scores[k] - v2_scores[k] for k in keys], dtype=np.float64)

    mask_total = exact_same = same_support = support_changed = intensity_only = 0
    changed_examples: list[str] = []
    intensity_examples: list[str] = []
    for old in sorted((V2 / "predicted_masks").rglob("*_mask.png")):
        rel = old.relative_to(V2 / "predicted_masks")
        new = V3 / "predicted_masks" / rel
        a = np.asarray(Image.open(old).convert("L"), dtype=np.uint8)
        b = np.asarray(Image.open(new).convert("L"), dtype=np.uint8)
        mask_total += 1
        if np.array_equal(a, b):
            exact_same += 1
        support_a, support_b = a > 0, b > 0
        if np.array_equal(support_a, support_b):
            same_support += 1
            if not np.array_equal(a, b):
                intensity_only += 1
                if len(intensity_examples) < 6:
                    intensity_examples.append(str(rel))
        else:
            support_changed += 1
            if len(changed_examples) < 6:
                changed_examples.append(str(rel))

    # Inspect raw Qwen audit evidence and which default score cases exist.
    crops_total = flagged = flagged_raw_zero = flagged_with_reason = valid_boxes = 0
    default_buckets = Counter()
    geometry_rejected = 0
    cross_view_hist = Counter()
    view_candidate_hist = Counter()
    reason_hist = Counter()
    for line in (V2 / "qwen_crop_outputs.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        prepared_scores = []
        for view in item.get("views", []):
            candidates = 0
            category, sample = item["group_folder"].split("/", 1)
            view_id = int(view.get("view", 0))
            with Image.open(Path("/root/autodl-tmp/Dataset/Test_A") / category / sample / f"{view_id}.png") as image:
                image_size = image.size
            for rec in view.get("crops", []):
                crops_total += 1
                parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}
                flag = bool(parsed.get("is_anomaly", rec.get("is_anomaly", False)))
                raw = float(parsed.get("anomaly_score", 0.0) or 0.0)
                if flag:
                    flagged += 1
                    candidates += 1
                    if raw <= 0.05:
                        flagged_raw_zero += 1
                    reason = str(parsed.get("reason", "")).strip()
                    if reason:
                        flagged_with_reason += 1
                        reason_hist[reason[:80]] += 1
                    box = parsed.get("box", rec.get("box", []))
                    if isinstance(box, list) and len(box) >= 4:
                        valid_boxes += 1
                        if mapped_box(rec, image_size) is None:
                            geometry_rejected += 1
                    if raw <= 0.05:
                        mapped = mapped_box(rec, image_size)
                        geometry = 0.18 if mapped is None else (0.62 if mapped[1] <= 0.002 else 0.95 if mapped[1] <= 0.015 else 0.82 if mapped[1] <= 0.06 else 0.60 if mapped[1] <= 0.12 else 0.38)
                        # This is a per-crop proxy; it intentionally omits
                        # final cross-crop consistency, reported separately.
                        bucket = discrete_zero_score(text_strength(rec), geometry, 0.0, 0.25, raw)
                        default_buckets[bucket] += 1
            view_candidate_hist[candidates] += 1
            score, _ = prepare_view(view, image_size)
            prepared_scores.append(score)
        cross_view_hist[sum(score >= 0.40 for score in prepared_scores)] += 1

    out = {
        "score": {
            "samples": len(keys),
            "unchanged": int(np.sum(np.isclose(diff, 0.0))),
            "changed": int(np.sum(~np.isclose(diff, 0.0))),
            "mean_v2": round(float(np.mean([v2_scores[k] for k in keys])), 6),
            "mean_v3": round(float(np.mean([v3_scores[k] for k in keys])), 6),
            "mean_delta": round(float(np.mean(diff)), 6),
            "min_delta": round(float(np.min(diff)), 6),
            "max_delta": round(float(np.max(diff)), 6),
            "v2_top": Counter(v2_scores.values()).most_common(10),
            "v3_top": Counter(v3_scores.values()).most_common(10),
        },
        "mask": {
            "total": mask_total,
            "pixel_exact_same": exact_same,
            "same_nonblack_support": same_support,
            "support_changed": support_changed,
            "intensity_only_changed": intensity_only,
            "support_change_examples": changed_examples,
            "intensity_change_examples": intensity_examples,
        },
        "audit": {
            "crops_total": crops_total,
            "flagged_anomalous": flagged,
            "flagged_raw_score_zero": flagged_raw_zero,
            "flagged_with_reason": flagged_with_reason,
            "flagged_with_box": valid_boxes,
            "boxes_rejected_by_geometry": geometry_rejected,
            "anomaly_crop_count_per_view": dict(sorted(view_candidate_hist.items())),
            "anomalous_view_count_per_sample": dict(sorted(cross_view_hist.items())),
            "zero_raw_score_default_bucket_proxy": dict(sorted(default_buckets.items())),
            "common_reason_samples": reason_hist.most_common(5),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
