"""One inference path shared by proxy evaluation and final submission."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from clip_dense import clip_preprocess_tensor
from common import per_layer_features, topk_mean
from object_prior import apply_prior
from reference_v6 import fused_dino_features


@dataclass
class GroupPrediction:
    raw_maps: list[np.ndarray]
    raw_image_score: float
    branch_maps: dict[str, list[np.ndarray]]
    branch_scores: dict[str, float]


def _upsample(tensor: torch.Tensor, size=448):
    x = F.interpolate(tensor[:, None].float(), (size, size), mode="bilinear", align_corners=False)[:, 0]
    return [np.asarray(item, np.float32) for item in x.cpu().numpy()]


def aggregate_views(values):
    values = sorted(float(value) for value in values)
    return 0.70 * values[-1] + 0.30 * float(np.mean(values[-2:]))


def run_group(images, categories, model, dino, banks, dino_size, clip_size, layers, device):
    dino_maps, _ = per_layer_features(dino, images, dino_size, device, tuple(layers))
    dino_feature = fused_dino_features(dino_maps)
    clip_x = clip_preprocess_tensor(images, clip_size, device)
    encoded = model.encode_clip(clip_x)
    dino_reference = banks.dino.score(dino_feature, categories, device)
    clip_reference = banks.clip.score(encoded[1], categories, device)
    output = model(
        dino_maps, clip_x, dino_reference=dino_reference, clip_reference=clip_reference,
        groups=1, encoded_clip=encoded,
    )
    tensors = {
        "final": torch.sigmoid(output.final_logit),
        "dino": torch.sigmoid(output.dino_logit),
        "semantic": torch.sigmoid(output.semantic_logit),
        "candidate": output.candidate_gate,
        "positive_residual": output.vlm_residual.clamp_min(0) / model.max_vlm_residual,
    }
    branches = {name: _upsample(value) for name, value in tensors.items()}
    # This is a geometric field-of-view constraint, not percentile mask
    # manipulation. Normal edge/reflection suppression is learned in the loss.
    # Only the submitted final map needs the geometric foreground constraint.
    # Keeping diagnostic branches raw avoids recomputing the foreground mask
    # five times per view and makes branch failures directly observable.
    branches["final"] = [
        apply_prior(score, image, background_keep=0.0, boundary_keep=0.32)
        for score, image in zip(branches["final"], images)
    ]
    scores = {
        name: aggregate_views([topk_mean(score, 0.0015, 8) for score in maps])
        for name, maps in branches.items()
    }
    raw_score = float(output.sample_logit[0].float().cpu())
    return GroupPrediction(branches["final"], raw_score, branches, scores)


def calibrated_output(prediction, category, calibration):
    maps = [calibration.pixel(category, value).astype(np.float32) for value in prediction.raw_maps]
    score = float(calibration.image(category, prediction.raw_image_score))
    return maps, score
