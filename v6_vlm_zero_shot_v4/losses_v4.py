from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from states import FIRST_ANOMALY, STATE_NAMES


def _resize_float(target: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(target[:, None].float(), shape, mode="bilinear", align_corners=False)[:, 0]


def _resize_state(target: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(target[:, None].float(), shape, mode="nearest")[:, 0].long()


def boundary_map(target: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(target[:, None], 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target[:, None], 3, stride=1, padding=1)
    return (dilated - eroded)[:, 0].clamp(0, 1)


class V4Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # Normal classes remain important, but each rare defect state receives
        # higher weight so the model cannot minimise loss by predicting normal.
        self.register_buffer(
            "state_weights",
            torch.tensor([0.25, 0.35, 0.75, 0.90] + [2.5] * (len(STATE_NAMES) - FIRST_ANOMALY)),
        )

    def forward(
        self,
        model,
        output,
        binary_target: torch.Tensor,
        state_target: torch.Tensor,
        view_target: torch.Tensor,
        sample_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        shape = output.final_map.shape[-2:]
        binary = _resize_float(binary_target, shape)
        states = _resize_state(state_target, shape)
        # The learned gate returns a probability rather than a single raw
        # logit.  Convert it back to a numerically stable equivalent logit so
        # BCEWithLogits can be used safely inside CUDA autocast.  PyTorch 2.3
        # intentionally rejects binary_cross_entropy(probability, target)
        # under autocast because the sigmoid backward may overflow in fp16.
        probability = output.final_map.float().clamp(1e-5, 1 - 1e-5)
        probability_logit = torch.logit(probability)
        binary = binary.float()

        # Precision-oriented dense loss.
        pixel_bce = F.binary_cross_entropy_with_logits(probability_logit, binary)
        tp = (probability * binary).sum((1, 2))
        fp = (probability * (1 - binary)).sum((1, 2))
        fn = ((1 - probability) * binary).sum((1, 2))
        tversky = 1 - ((tp + 1) / (tp + 0.72 * fp + 0.28 * fn + 1)).mean()
        edge = boundary_map(binary)
        edge_bce = (
            F.binary_cross_entropy_with_logits(probability_logit, binary, reduction="none") * (1 + 2 * edge)
        ).mean()
        pixel_loss = 0.40 * pixel_bce + 0.80 * tversky + 0.25 * edge_bce

        state_loss = F.cross_entropy(output.state_logits, states, weight=self.state_weights)

        query_target = []
        for state in range(FIRST_ANOMALY, len(STATE_NAMES)):
            query_target.append((states == state).float())
        query_target_t = torch.stack(query_target, dim=1)
        query_loss = F.binary_cross_entropy_with_logits(output.query_logits, query_target_t)

        # Region-text alignment: the CLIP patch grid must identify the state
        # present at each sampled location, not merely classify the whole image.
        patch_side = output.patch_features.shape[1]
        patch_states = _resize_state(state_target, (patch_side, patch_side))
        text = F.normalize(output.static_text, dim=-1)
        patch = F.normalize(output.patch_features, dim=-1)
        region_logits = torch.einsum("bhwd,cd->bchw", patch, text) * 20.0
        region_text = F.cross_entropy(region_logits, patch_states, weight=self.state_weights)

        global_loss = torch.zeros((), device=probability.device)
        multiview_loss = torch.zeros_like(global_loss)
        if output.view_logits is not None:
            view_labels = view_target.view(output.view_logits.shape).float()
            global_loss = F.binary_cross_entropy_with_logits(output.view_logits, view_labels)
        if output.sample_logits is not None:
            multiview_loss = F.binary_cross_entropy_with_logits(output.sample_logits, sample_target.float())

        # Preserve the original CLIP language neighbourhood and only align the
        # two visual branches where one of them is confident.
        prompt_anchor = model.prompt.anchor_loss()
        confidence = torch.maximum((output.dino_map - 0.5).abs(), (output.vlm_map - 0.5).abs()) * 2
        cross_modal = ((output.dino_map - output.vlm_map).abs() * confidence.detach()).mean()

        total = (
            1.00 * pixel_loss
            + 0.35 * state_loss
            + 0.25 * region_text
            + 0.30 * query_loss
            + 0.15 * global_loss
            + 0.15 * multiview_loss
            + 0.10 * prompt_anchor
            + 0.10 * cross_modal
        )
        parts = {
            "total": float(total.detach()),
            "pixel": float(pixel_loss.detach()),
            "state": float(state_loss.detach()),
            "region": float(region_text.detach()),
            "query": float(query_loss.detach()),
            "global": float(global_loss.detach()),
            "multiview": float(multiview_loss.detach()),
            "anchor": float(prompt_anchor.detach()),
            "cross": float(cross_modal.detach()),
        }
        return total, parts
