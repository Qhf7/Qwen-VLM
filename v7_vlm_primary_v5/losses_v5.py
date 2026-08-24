"""Losses for VLM-primary local text alignment and normal association."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from states import FIRST_ANOMALY, STATE_NAMES


def resize_float(target, shape):
    return F.interpolate(target[:, None].float(), shape, mode="bilinear", align_corners=False)[:, 0]


def resize_state(target, shape):
    return F.interpolate(target[:, None].float(), shape, mode="nearest")[:, 0].long()


def binary_segmentation(probability, target):
    probability = probability.float().clamp(1e-5, 1 - 1e-5)
    target = target.float()
    logit = torch.logit(probability)
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(2.0, 20.0).detach()
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
    tp = (probability * target).sum((1, 2))
    fp = (probability * (1 - target)).sum((1, 2))
    fn = ((1 - probability) * target).sum((1, 2))
    tversky = 1 - ((tp + 1) / (tp + 0.75 * fp + 0.25 * fn + 1)).mean()
    return 0.45 * bce + 0.85 * tversky


def probability_binary_cross_entropy(probability, target):
    """AMP-safe BCE for model outputs represented as probabilities.

    PyTorch intentionally rejects ``binary_cross_entropy`` inside CUDA
    autocast, even when its inputs have been converted to float32.  Convert
    the bounded probability back to a logit and use the numerically stable
    logits implementation instead.
    """
    probability = probability.float().clamp(1e-5, 1 - 1e-5)
    target = target.float()
    return F.binary_cross_entropy_with_logits(torch.logit(probability), target)


class V5Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "state_weights",
            torch.tensor([0.20, 0.35, 0.90, 0.90] + [2.2] * (len(STATE_NAMES) - FIRST_ANOMALY)),
        )

    def forward(self, model, output, binary_target, state_target, view_target, sample_target):
        shape = output.final_map.shape[-2:]
        binary = resize_float(binary_target, shape)
        states = resize_state(state_target, shape)

        # VLM is explicitly supervised as the primary dense prediction.
        vlm_seg = binary_segmentation(output.vlm_map, binary)
        final_seg = binary_segmentation(output.final_map, binary)
        semantic_bce = F.binary_cross_entropy_with_logits(output.semantic_logit.float(), binary.float())
        state_loss = F.cross_entropy(output.state_logits.float(), states, weight=self.state_weights)

        # Patch/text state alignment is kept local instead of global-only CLIP scoring.
        patch_side = output.patch_features.shape[1]
        patch_states = resize_state(state_target, (patch_side, patch_side))
        text = F.normalize(output.static_text, dim=-1)
        patch = F.normalize(output.patch_features, dim=-1)
        patch_logits = torch.einsum("bhwd,cd->bchw", patch, text) * 20.0
        region_text = F.cross_entropy(patch_logits.float(), patch_states, weight=self.state_weights)

        view_loss = probability_binary_cross_entropy(output.view_probability, view_target)
        sample_loss = output.final_map.new_zeros(())
        if output.sample_probability is not None:
            sample_loss = probability_binary_cross_entropy(output.sample_probability, sample_target)

        # The reference branch must be low on normals and high at the known
        # synthetic edit.  This trains WinCLIP+-style association directly.
        reference_loss = binary_segmentation(output.reference_map, binary)
        prompt_anchor = model.prompt.anchor_loss()
        total = (
            0.95 * vlm_seg
            + 0.25 * final_seg
            + 0.30 * semantic_bce
            + 0.30 * state_loss
            + 0.30 * region_text
            + 0.25 * reference_loss
            + 0.15 * view_loss
            + 0.15 * sample_loss
            + 0.10 * prompt_anchor
        )
        parts = {
            "total": float(total.detach()),
            "vlm": float(vlm_seg.detach()),
            "final": float(final_seg.detach()),
            "semantic": float(semantic_bce.detach()),
            "state": float(state_loss.detach()),
            "region": float(region_text.detach()),
            "reference": float(reference_loss.detach()),
            "view": float(view_loss.detach()),
            "sample": float(sample_loss.detach()),
            "anchor": float(prompt_anchor.detach()),
            "dino_fraction": float(output.dino_fraction.detach()),
        }
        return total, parts
