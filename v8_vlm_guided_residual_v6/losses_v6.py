"""Precision-oriented training objectives for v6."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from states import FIRST_ANOMALY, NORMAL_EDGE, NORMAL_REFLECTION, NORMAL_SURFACE, STATE_NAMES


def resize_float(target, shape):
    return F.interpolate(target[:, None].float(), shape, mode="bilinear", align_corners=False)[:, 0]


def resize_state(target, shape):
    return F.interpolate(target[:, None].float(), shape, mode="nearest")[:, 0].long()


def segmentation_loss(logit, target):
    target = target.float()
    probability = torch.sigmoid(logit.float())
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (neg / pos).clamp(2.0, 20.0).detach()
    bce = F.binary_cross_entropy_with_logits(logit.float(), target, pos_weight=pos_weight)
    tp = (probability * target).sum((1, 2))
    fp = (probability * (1 - target)).sum((1, 2))
    fn = ((1 - probability) * target).sum((1, 2))
    tversky = 1 - ((tp + 1) / (tp + 0.78 * fp + 0.22 * fn + 1)).mean()
    return 0.45 * bce + 0.90 * tversky


def masked_softplus(logit, mask):
    weight = mask.float()
    return (F.softplus(logit.float()) * weight).sum() / weight.sum().clamp_min(1.0)


def hard_normal_ranking(logit, target, hard_normal, margin=1.0):
    losses = []
    for score, anomaly, hard in zip(logit.float(), target > 0.2, hard_normal > 0.2):
        positives = score[anomaly]
        negatives = score[hard & ~anomaly]
        if positives.numel() and negatives.numel():
            pos = positives.topk(min(128, positives.numel()), largest=False).values.mean()
            neg = negatives.topk(min(512, negatives.numel()), largest=True).values.mean()
            losses.append(F.relu(score.new_tensor(margin) - pos + neg))
    return torch.stack(losses).mean() if losses else logit.new_zeros(())


class V6Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "state_weights",
            torch.tensor([0.15, 0.45, 1.20, 1.20] + [2.2] * (len(STATE_NAMES) - FIRST_ANOMALY)),
        )

    def forward(
        self, model, output, binary_target, state_target, hard_negative, view_target, sample_target
    ):
        shape = output.final_logit.shape[-2:]
        binary = resize_float(binary_target, shape)
        states = resize_state(state_target, shape)
        hard = resize_float(hard_negative, shape) * (binary < 0.15)

        final_seg = segmentation_loss(output.final_logit, binary)
        dino_seg = segmentation_loss(output.dino_logit, binary)
        state_loss = F.cross_entropy(output.state_logits.float(), states, weight=self.state_weights)

        patch_side = output.patch_features.shape[1]
        patch_states = resize_state(state_target, (patch_side, patch_side))
        patch = F.normalize(output.patch_features, dim=-1)
        text = F.normalize(output.static_text, dim=-1)
        region_logits = torch.einsum("bhwd,cd->bchw", patch, text) * 20.0
        region_text = F.cross_entropy(region_logits.float(), patch_states, weight=self.state_weights)

        normal_surface = states == NORMAL_SURFACE
        boundary_normal = (states == NORMAL_EDGE) | (states == NORMAL_REFLECTION)
        normal_suppression = masked_softplus(output.final_logit, normal_surface)
        boundary_suppression = masked_softplus(output.final_logit, boundary_normal | (hard > 0.45))
        ranking = hard_normal_ranking(output.final_logit, binary, (hard > 0.20) | boundary_normal)

        probability = torch.sigmoid(output.final_logit.float())
        normal_views = (view_target < 0.5).float()[:, None, None]
        normal_area = (probability * normal_views).sum() / (
            normal_views.sum() * probability.shape[-1] * probability.shape[-2]
        ).clamp_min(1.0)
        bright_area = (
            torch.sigmoid((probability - 0.25) * 16.0) * normal_views
        ).sum() / (normal_views.sum() * probability.shape[-1] * probability.shape[-2]).clamp_min(1.0)
        area_penalty = normal_area + 0.50 * bright_area

        outside_candidate = (1.0 - output.candidate_gate.detach())
        positive_residual = output.vlm_residual.clamp_min(0.0)
        residual_leak = (positive_residual * outside_candidate).mean()
        residual_magnitude = output.vlm_residual.abs().mean()
        candidate_on_normal = (output.candidate_gate * normal_views).sum() / (
            normal_views.sum() * output.candidate_gate.shape[-1] * output.candidate_gate.shape[-2]
        ).clamp_min(1.0)

        view_loss = F.binary_cross_entropy_with_logits(output.view_logits.float(), view_target.float())
        sample_loss = output.final_logit.new_zeros(())
        if output.sample_logit is not None:
            sample_loss = F.binary_cross_entropy_with_logits(output.sample_logit.float(), sample_target.float())
        global_loss = F.binary_cross_entropy_with_logits(output.global_logit.float(), view_target.float())
        prompt_anchor = model.prompt.anchor_loss()

        total = (
            1.00 * final_seg
            + 0.65 * dino_seg
            + 0.30 * state_loss
            + 0.25 * region_text
            + 0.40 * normal_suppression
            + 0.55 * boundary_suppression
            + 0.30 * ranking
            + 0.45 * area_penalty
            + 0.30 * residual_leak
            + 0.08 * residual_magnitude
            + 0.18 * candidate_on_normal
            + 0.18 * view_loss
            + 0.12 * sample_loss
            + 0.12 * global_loss
            + 0.08 * prompt_anchor
        )
        parts = {
            "total": float(total.detach()), "final": float(final_seg.detach()),
            "dino": float(dino_seg.detach()), "state": float(state_loss.detach()),
            "region": float(region_text.detach()), "normal": float(normal_suppression.detach()),
            "boundary": float(boundary_suppression.detach()), "ranking": float(ranking.detach()),
            "area": float(area_penalty.detach()), "residual_leak": float(residual_leak.detach()),
            "residual_l1": float(residual_magnitude.detach()), "view": float(view_loss.detach()),
            "candidate_normal": float(candidate_on_normal.detach()),
            "sample": float(sample_loss.detach()), "global": float(global_loss.detach()),
            "anchor": float(prompt_anchor.detach()),
            "reference_weight": float(output.dino_reference_weight.detach()),
        }
        return total, parts
