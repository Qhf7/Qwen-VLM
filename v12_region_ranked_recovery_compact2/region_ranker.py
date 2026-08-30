#!/usr/bin/env python3
"""Train-only candidate ranking for compact industrial anomaly masks.

The dense DINO branches are good proposal generators but often respond to an
entire object silhouette.  This module changes the decision unit from a pixel
to a connected candidate region.  Candidates are described by their anomaly
strength, persistence across thresholds, cross-branch agreement, local
contrast, area and contact with normal object boundaries.  A small MLP is
trained on official Train normals and programmatic pseudo anomalies only.

At inference a view may return zero regions.  We never force Top-1, because
doing so would turn every normal view into a false positive.  Candidates must
pass the Train-only rejection threshold and then compete through soft NMS,
relative-score and total-area constraints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from object_prior import foreground_mask


FEATURE_NAMES = (
    "log_area", "sqrt_area", "bbox_fill", "aspect_balance", "compactness",
    "peak", "mean", "q75", "q90", "core_fraction", "threshold_persistence",
    "inner_ring_contrast", "peak_ring_contrast", "branch_min", "branch_mean",
    "branch_max", "branch_agreement", "branch_dispersion", "foreground_mean",
    "outline_overlap", "image_gradient_mean", "image_gradient_peak",
)
THRESHOLDS = (0.055, 0.09, 0.14, 0.22, 0.34, 0.50)


@dataclass
class Candidate:
    mask: np.ndarray
    features: np.ndarray
    area: float
    peak: float
    box: tuple[int, int, int, int]
    probability: float = 0.0


class RegionRanker(nn.Module):
    def __init__(self, dim: int = len(FEATURE_NAMES), hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _resize_map(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    x = np.asarray(values, np.float32)
    if x.shape == shape:
        return x
    im = Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255), "L")
    return np.asarray(im.resize((shape[1], shape[0]), Image.Resampling.BILINEAR), np.float32) / 255.0


def _image_gradient(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    gray = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR).convert("L"), np.float32)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    grad = np.hypot(gx, gy)
    return np.clip(grad / max(float(np.percentile(grad, 99.5)), 1.0), 0.0, 1.0)


def _boundary_ring(image: Image.Image, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    fg = foreground_mask(image, shape[0] if shape[0] == shape[1] else 448)
    fg = _resize_map(fg, shape)
    hard = fg > 0.55
    eroded = ndimage.binary_erosion(hard, iterations=max(1, round(min(shape) / 112)))
    ring = hard ^ eroded
    return fg, ring


def _component_candidates(score: np.ndarray) -> list[np.ndarray]:
    h, w = score.shape
    min_pixels = max(3, int(round(h * w * 0.000015)))
    max_pixels = max(min_pixels + 1, int(round(h * w * 0.075)))
    proposals: list[np.ndarray] = []
    for threshold in THRESHOLDS:
        binary = score >= threshold
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), np.uint8))
        if count == 0:
            continue
        objects = ndimage.find_objects(labels)
        for label_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            local = labels[slc] == label_id
            area = int(local.sum())
            if area < min_pixels or area > max_pixels:
                continue
            mask = np.zeros((h, w), bool)
            mask[slc] = local
            proposals.append(mask)
    # Deduplicate the nested components produced by the threshold tree.
    proposals.sort(key=lambda m: (int(m.sum()), -float(score[m].max())))
    kept: list[np.ndarray] = []
    for mask in proposals:
        duplicate = False
        for previous in kept:
            intersection = int(np.logical_and(mask, previous).sum())
            small = max(min(int(mask.sum()), int(previous.sum())), 1)
            large = max(int(mask.sum()), int(previous.sum()))
            if intersection / small >= 0.88 and large / small <= 2.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
    return kept


def _candidate_features(
    score: np.ndarray,
    branches: list[np.ndarray],
    mask: np.ndarray,
    fg: np.ndarray,
    outline: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    h, w = score.shape
    area_px = int(mask.sum())
    area = area_px / float(h * w)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bbox_area = max((y1 - y0) * (x1 - x0), 1)
    bbox_fill = area_px / bbox_area
    aspect_balance = min(y1 - y0, x1 - x0) / max(y1 - y0, x1 - x0, 1)
    edge = mask ^ ndimage.binary_erosion(mask)
    perimeter = max(int(edge.sum()), 1)
    compactness = float(np.clip(4.0 * math.pi * area_px / (perimeter * perimeter), 0.0, 1.0))
    values = score[mask]
    dilated = ndimage.binary_dilation(mask, iterations=max(2, round(min(h, w) / 112)))
    ring = dilated & ~mask
    ring_values = score[ring] if ring.any() else np.asarray([0.0], np.float32)
    branch_values = [float(np.mean(branch[mask])) for branch in branches]
    branch_values = branch_values or [float(values.mean())]
    branch_array = np.asarray(branch_values, np.float32)
    support = np.asarray([v >= 0.10 for v in branch_values], np.float32)
    threshold_persistence = np.mean([float(np.any(values >= t)) for t in THRESHOLDS])
    core_fraction = float(np.mean(values >= max(0.22, float(values.max()) * 0.65)))
    return np.asarray(
        [
            np.log10(max(area, 1e-7)) / 7.0 + 1.0,
            math.sqrt(area), bbox_fill, aspect_balance, compactness,
            float(values.max()), float(values.mean()), float(np.quantile(values, 0.75)),
            float(np.quantile(values, 0.90)), core_fraction, threshold_persistence,
            float(values.mean() - ring_values.mean()), float(values.max() - np.quantile(ring_values, 0.90)),
            float(branch_array.min()), float(branch_array.mean()), float(branch_array.max()),
            float(support.mean()), float(branch_array.std()), float(fg[mask].mean()),
            float(outline[mask].mean()), float(gradient[mask].mean()), float(gradient[mask].max()),
        ],
        np.float32,
    )


def generate_candidates(
    score: np.ndarray,
    branch_maps: list[np.ndarray],
    image: Image.Image,
) -> list[Candidate]:
    dense = np.clip(np.asarray(score, np.float32), 0.0, 1.0)
    branches = [_resize_map(np.asarray(x, np.float32), dense.shape) for x in branch_maps if x is not None]
    fg, outline = _boundary_ring(image, dense.shape)
    gradient = _image_gradient(image, dense.shape)
    output = []
    for mask in _component_candidates(dense):
        ys, xs = np.nonzero(mask)
        features = _candidate_features(dense, branches, mask, fg, outline, gradient)
        output.append(
            Candidate(
                mask=mask,
                features=features,
                area=float(mask.mean()),
                peak=float(dense[mask].max()),
                box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
        )
    return output


def candidate_target(mask: np.ndarray, truth: np.ndarray) -> float:
    target = np.asarray(truth) > 0.5
    intersection = int(np.logical_and(mask, target).sum())
    if intersection == 0:
        return 0.0
    precision = intersection / max(int(mask.sum()), 1)
    recall = intersection / max(int(target.sum()), 1)
    # Candidate supervision rewards both tightness and defect coverage.
    return float(np.clip(math.sqrt(precision * recall), 0.0, 1.0))


def fit_ranker(
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: np.ndarray,
    normal_features: np.ndarray,
    normal_view_ids: np.ndarray | None = None,
    epochs: int = 80,
    seed: int = 20260828,
) -> tuple[RegionRanker, dict]:
    if len(features) < 32 or np.max(targets, initial=0.0) <= 0:
        raise ValueError("region ranker needs both positive pseudo candidates and hard normal candidates")
    torch.manual_seed(seed)
    mean = features.mean(0).astype(np.float32)
    std = np.maximum(features.std(0), 1e-4).astype(np.float32)
    x = torch.from_numpy((features - mean) / std).float()
    y = torch.from_numpy((targets >= 0.18).astype(np.float32))
    soft = torch.from_numpy(np.clip(targets, 0, 1).astype(np.float32))
    groups = torch.from_numpy(group_ids.astype(np.int64))
    model = RegionRanker(features.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    positives = max(float(y.sum()), 1.0)
    pos_weight = torch.tensor(min(12.0, max(1.0, (len(y) - positives) / positives)))
    for epoch in range(epochs):
        model.train()
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        regression = F.smooth_l1_loss(torch.sigmoid(logits), soft)
        pair_losses = []
        for group in torch.unique(groups):
            idx = groups == group
            pos = logits[idx & (y > 0.5)]
            neg = logits[idx & (y < 0.5)]
            if len(pos) and len(neg):
                pair_losses.append(F.relu(0.45 - pos[:, None] + neg[None, :]).mean())
        pair = torch.stack(pair_losses).mean() if pair_losses else logits.sum() * 0.0
        loss = bce + 0.35 * regression + 0.30 * pair
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimiser.step()
        if epoch % 10 == 0 or epoch + 1 == epochs:
            print(
                f"V12_PROGRESS region_ranker {epoch+1}/{epochs} loss={loss.detach().item():.5f} "
                f"bce={bce.detach().item():.5f} pair={pair.detach().item():.5f}", flush=True,
            )
    model.eval()
    with torch.inference_mode():
        normal_x = torch.from_numpy((normal_features - mean) / std).float()
        normal_prob = torch.sigmoid(model(normal_x)).numpy() if len(normal_features) else np.asarray([0.5])
    # Calibrate on the maximum candidate score per held-out normal view. This
    # controls view-level false positives; a candidate-level quantile would be
    # too permissive when one view contains dozens of outline proposals.
    if normal_view_ids is not None and len(normal_view_ids) == len(normal_prob):
        maxima = []
        for view_id in np.unique(normal_view_ids):
            maxima.append(float(normal_prob[normal_view_ids == view_id].max()))
        normal_maxima = np.asarray(maxima, np.float32)
    else:
        normal_maxima = normal_prob
    reject = float(np.clip(np.quantile(normal_maxima, 0.995), 0.48, 0.995))
    metadata = {
        "feature_names": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(),
        "reject_threshold": reject, "train_candidates": int(len(features)),
        "normal_calibration_candidates": int(len(normal_features)), "epochs": epochs,
        "normal_calibration_views": int(len(normal_maxima)),
        "normal_view_max_q99": float(np.quantile(normal_maxima, 0.99)),
        "official_train_only": True, "test_labels_used": False,
    }
    return model, metadata


def save_ranker(path: Path, model: RegionRanker, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "metadata": metadata}, path)


def load_ranker(path: Path) -> tuple[RegionRanker, dict]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload["metadata"]
    model = RegionRanker(len(metadata["feature_names"]))
    model.load_state_dict(payload["state"])
    model.eval()
    return model, metadata


def score_candidates(candidates: list[Candidate], model: RegionRanker, metadata: dict) -> list[Candidate]:
    if not candidates:
        return []
    features = np.stack([candidate.features for candidate in candidates])
    mean = np.asarray(metadata["mean"], np.float32)
    std = np.asarray(metadata["std"], np.float32)
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy((features - mean) / std).float())).numpy()
    for candidate, value in zip(candidates, probability):
        candidate.probability = float(value)
    return candidates


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return intersection / union if union else 0.0


def select_candidates(
    candidates: list[Candidate],
    reject_threshold: float,
    top_k: int = 3,
    relative_margin: float = 0.16,
    max_area: float = 0.018,
) -> list[Candidate]:
    # Oversized components are outline/foreground proposals, not eligible
    # defects. Filter them before defining the relative-score floor; otherwise
    # one very confident outline can suppress a valid compact second candidate.
    ranked = sorted(
        (candidate for candidate in candidates if candidate.area <= max_area),
        key=lambda x: (x.probability, x.peak), reverse=True,
    )
    if not ranked or ranked[0].probability < reject_threshold:
        return []
    floor = max(reject_threshold, ranked[0].probability - relative_margin)
    selected: list[Candidate] = []
    total_area = 0.0
    for candidate in ranked:
        if candidate.probability < floor or len(selected) >= top_k:
            continue
        if any(_mask_iou(candidate.mask, previous.mask) > 0.35 for previous in selected):
            continue
        if selected and total_area + candidate.area > max_area:
            continue
        selected.append(candidate)
        total_area += candidate.area
    return selected


def region_ranked_mask(
    score: np.ndarray,
    branch_maps: list[np.ndarray],
    image: Image.Image,
    model: RegionRanker,
    metadata: dict,
    reject_scale: float = 1.0,
    relative_margin: float = 0.16,
    max_area: float = 0.018,
) -> tuple[np.ndarray, dict]:
    dense = np.clip(np.asarray(score, np.float32), 0.0, 1.0)
    candidates = score_candidates(generate_candidates(dense, branch_maps, image), model, metadata)
    threshold = float(np.clip(
        float(metadata["reject_threshold"]) * float(reject_scale), 0.48, 0.995
    ))
    selected = select_candidates(
        candidates, threshold, relative_margin=relative_margin, max_area=max_area
    )
    output = np.zeros_like(dense)
    for candidate in selected:
        # Preserve within-region DINO ordering; the ranker decides which region
        # survives and supplies a globally comparable confidence ceiling.
        local = np.where(candidate.mask, dense, 0.0)
        peak = max(float(local.max()), 1e-6)
        local = (local / peak) ** 1.20
        output = np.maximum(output, local * candidate.probability)
    diagnostics = {
        "candidate_count": len(candidates), "selected_count": len(selected),
        "selected_area": float(sum(x.area for x in selected)),
        "top_probability": float(max((x.probability for x in candidates), default=0.0)),
        "reject_threshold": threshold,
        "reject_scale": float(reject_scale),
    }
    return np.clip(output, 0.0, 1.0), diagnostics
