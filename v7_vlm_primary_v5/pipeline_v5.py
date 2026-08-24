"""Single authoritative inference path for proxy and final submission."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from clip_dense import clip_preprocess_tensor
from common import per_layer_features, topk_mean
from object_prior import apply_prior
from view_consistency import apply_view_consistency


@dataclass
class GroupPrediction:
    raw_maps: list[np.ndarray]
    raw_image_score: float
    branch_maps: dict[str, list[np.ndarray]]
    branch_scores: dict[str, float]


def aggregate_views(values) -> float:
    values = sorted(float(value) for value in values)
    return 0.70 * values[-1] + 0.30 * float(np.mean(values[-2:]))


def _upsample(tensor: torch.Tensor, size: int = 448) -> list[np.ndarray]:
    array = F.interpolate(
        tensor[:, None].float(), (size, size), mode="bilinear", align_corners=False
    )[:, 0].cpu().numpy()
    return [np.asarray(value, np.float32) for value in array]


def run_group(
    images,
    categories,
    model,
    dino,
    reference_bank,
    dino_size,
    clip_size,
    layers,
    device,
    reference_scale: float = 0.18,
) -> GroupPrediction:
    """Run the exact same forward/postprocess path in proxy and submission."""
    dino_maps, _ = per_layer_features(dino, images, dino_size, device, tuple(layers))
    clip_x = clip_preprocess_tensor(images, clip_size, device)
    encoded = model.encode_clip(clip_x)
    reference = (
        reference_bank.score(encoded[1], list(categories), device) / max(float(reference_scale), 1e-5)
    ).clamp(0, 1)
    output = model(dino_maps, clip_x, reference_map=reference, groups=1, encoded_clip=encoded)
    branches = {
        "final": _upsample(output.final_map),
        "vlm": _upsample(output.vlm_map),
        "semantic": _upsample(output.semantic_map),
        "reference": _upsample(output.reference_map),
        "dino": _upsample(output.dino_map),
    }
    prepared = {}
    for branch, maps in branches.items():
        values = [
            apply_prior(score, image, background_keep=0.025, boundary_keep=0.38)
            for score, image in zip(maps, images)
        ]
        values, _ = apply_view_consistency(values)
        prepared[branch] = [np.asarray(value, np.float32) for value in values]
    branch_scores = {
        branch: aggregate_views([topk_mean(score, 0.0015, 8) for score in maps])
        for branch, maps in prepared.items()
    }
    sample_probability = float(output.sample_probability[0].cpu())
    # Final score stays VLM-primary.  VLM dense evidence and CLIP global/state
    # evidence dominate; the final map score contributes only a bounded term.
    raw_image_score = float(
        0.58 * branch_scores["vlm"]
        + 0.27 * sample_probability
        + 0.10 * branch_scores["reference"]
        + 0.05 * branch_scores["dino"]
    )
    return GroupPrediction(
        raw_maps=prepared["final"],
        raw_image_score=raw_image_score,
        branch_maps=prepared,
        branch_scores=branch_scores,
    )


def calibrated_output(prediction, category, calibration, zero_floor=0.015):
    maps = [calibration.pixel(category, score).astype(np.float32) for score in prediction.raw_maps]
    maps = [np.where(score < zero_floor, 0.0, score).astype(np.float32) for score in maps]
    image_score = float(calibration.image(category, prediction.raw_image_score))
    return maps, image_score
