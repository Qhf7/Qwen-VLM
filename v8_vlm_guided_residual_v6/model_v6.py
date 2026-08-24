"""DINO-primary anomaly model with a bounded, spatially gated VLM residual.

The architecture deliberately prevents the v5 failure mode where generic
CLIP foreground semantics could paint an entire component as anomalous.
Dense localisation is supplied by multi-layer DINOv2 and a spatially aligned
normal reference. CLIP contributes defect semantics, hard-normal semantics,
and image-level zero-shot evidence, but positive VLM evidence can only modify
locations already proposed by the DINO branch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_dense import DenseCLIPEncoder, PromptLearner
from multiscale_decoder import LayerAwarePixelDecoder
from states import FIRST_ANOMALY, STATE_NAMES


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    count = x.shape[dim]
    return torch.logsumexp(x, dim=dim) - x.new_tensor(float(count)).log()


class ConvBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )


class LocalSemanticResidual(nn.Module):
    """Predict state-logit corrections from local CLIP patch tokens."""

    def __init__(self, clip_dim: int, width: int, states: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(clip_dim, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.context = nn.Sequential(ConvBlock(width), ConvBlock(width))
        self.state = nn.Conv2d(width, states, 1)

    def forward(self, patch: torch.Tensor, side: int) -> torch.Tensor:
        x = patch.permute(0, 3, 1, 2).contiguous()
        x = self.context(self.proj(x))
        x = F.interpolate(x, (side, side), mode="bilinear", align_corners=False)
        return self.state(x)


@dataclass
class V6Output:
    final_logit: torch.Tensor
    dino_logit: torch.Tensor
    dino_decoder_logit: torch.Tensor
    dino_reference_logit: torch.Tensor
    vlm_residual: torch.Tensor
    candidate_gate: torch.Tensor
    semantic_logit: torch.Tensor
    state_logits: torch.Tensor
    global_logit: torch.Tensor
    view_logits: torch.Tensor
    sample_logit: torch.Tensor | None
    static_text: torch.Tensor
    patch_features: torch.Tensor
    dino_reference_weight: torch.Tensor


class VLMGuidedResidualV6(nn.Module):
    """Use VLM semantics to refine, never replace, DINO localisation."""

    def __init__(
        self,
        dino_dims: list[int],
        clip_model,
        decoder_width: int = 192,
        vlm_width: int = 192,
        prompt_ctx: int = 8,
        max_vlm_residual: float = 1.25,
    ):
        super().__init__()
        self.dino_decoder = LayerAwarePixelDecoder(dino_dims, width=decoder_width)
        self.clip_encoder = DenseCLIPEncoder(clip_model, adapter_dim=64)
        self.prompt = PromptLearner(clip_model, n_ctx=prompt_ctx)
        clip_dim = int(clip_model.visual.output_dim)
        self.local_semantic = LocalSemanticResidual(clip_dim, vlm_width, len(STATE_NAMES))
        self.semantic_temperature = nn.Parameter(torch.tensor(2.3))
        self.reference_weight_logit = nn.Parameter(torch.tensor(-0.6))
        self.candidate_threshold = nn.Parameter(torch.tensor(-1.15))
        self.candidate_temperature_log = nn.Parameter(torch.tensor(-0.15))
        self.negative_residual_fraction_logit = nn.Parameter(torch.tensor(0.0))
        self.max_vlm_residual = float(max_vlm_residual)

    @staticmethod
    def _state_margin(logits: torch.Tensor, dim: int) -> torch.Tensor:
        normal = logits.narrow(dim, 1, FIRST_ANOMALY - 1)
        anomaly = logits.narrow(dim, FIRST_ANOMALY, len(STATE_NAMES) - FIRST_ANOMALY)
        return logmeanexp(anomaly, dim) - logmeanexp(normal, dim)

    @staticmethod
    def _topk_logit(logit: torch.Tensor, fraction: float = 0.002) -> torch.Tensor:
        flat = logit.flatten(1)
        k = max(8, int(round(flat.shape[1] * fraction)))
        return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)

    def encode_clip(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_feature, patch = self.clip_encoder(images)
        return global_feature, self.clip_encoder.window_features(patch)

    def load_v4_adaptation(self, checkpoint: dict, reuse_vlm: bool = False) -> dict[str, int]:
        """Initialise the proven DINO decoder; VLM reuse is opt-in only."""
        source = checkpoint.get("model", checkpoint.get("state", checkpoint))
        prefixes = ("dino_decoder.",)
        if reuse_vlm:
            prefixes += ("clip_encoder.adapter.", "clip_encoder.patch_projection.", "prompt.")
        current = self.state_dict()
        selected = {
            key: value for key, value in source.items()
            if key.startswith(prefixes) and key in current and current[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(selected, strict=False)
        return {"loaded": len(selected), "missing": len(missing), "unexpected": len(unexpected)}

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        excluded = ("clip_encoder.visual.", "prompt.clip_model.")
        return {
            key: value.detach().cpu() for key, value in self.state_dict().items()
            if not key.startswith(excluded)
        }

    def load_compact_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = ("clip_encoder.visual.", "prompt.clip_model.")
        invalid = [key for key in missing if not key.startswith(allowed)]
        if invalid or unexpected:
            raise RuntimeError(f"v6 checkpoint mismatch: missing={invalid}, unexpected={unexpected}")

    def forward(
        self,
        dino_maps: list[torch.Tensor],
        clip_images: torch.Tensor,
        dino_reference: torch.Tensor | None = None,
        clip_reference: torch.Tensor | None = None,
        groups: int | None = None,
        encoded_clip: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> V6Output:
        global_clip, patch = encoded_clip if encoded_clip is not None else self.encode_clip(clip_images)
        text, static_text = self.prompt(global_clip)
        decoder_logit, _ = self.dino_decoder(dino_maps)
        side = int(decoder_logit.shape[-1])

        if dino_reference is None:
            reference_logit = torch.full_like(decoder_logit, -4.0)
        else:
            reference = F.interpolate(
                dino_reference[:, None].float(), (side, side), mode="bilinear", align_corners=False
            )[:, 0].clamp(1e-4, 1 - 1e-4)
            reference_logit = torch.logit(reference)
        # The learned decoder always retains at least 55% of the primary logit.
        reference_weight = 0.20 + 0.25 * torch.sigmoid(self.reference_weight_logit)
        dino_logit = (1.0 - reference_weight) * decoder_logit + reference_weight * reference_logit

        temperature = self.semantic_temperature.exp().clamp(1.0, 30.0)
        semantic = torch.einsum("bhwd,bcd->bchw", patch, text) * temperature
        semantic = F.interpolate(semantic, (side, side), mode="bilinear", align_corners=False)
        state_logits = semantic + self.local_semantic(patch, side)
        semantic_logit = self._state_margin(state_logits, 1)
        if clip_reference is not None:
            clip_ref = F.interpolate(
                clip_reference[:, None].float(), (side, side), mode="bilinear", align_corners=False
            )[:, 0].clamp(1e-4, 1 - 1e-4)
            semantic_logit = semantic_logit + 0.20 * torch.logit(clip_ref)

        gate_temperature = self.candidate_temperature_log.exp().clamp(0.45, 1.5)
        threshold = self.candidate_threshold.clamp(-2.5, 0.0)
        candidate_gate = torch.sigmoid((dino_logit - threshold) / gate_temperature)
        signed_semantic = torch.tanh(semantic_logit / 3.0)
        positive = signed_semantic.clamp_min(0.0) * candidate_gate
        negative_fraction = 0.25 + 0.50 * torch.sigmoid(self.negative_residual_fraction_logit)
        negative = signed_semantic.clamp_max(0.0) * negative_fraction
        vlm_residual = self.max_vlm_residual * (positive + negative)
        final_logit = dino_logit + vlm_residual

        global_state = torch.einsum("bd,bcd->bc", global_clip, text) * temperature
        global_logit = self._state_margin(global_state, 1)
        dense_logit = self._topk_logit(final_logit)
        semantic_support = self._topk_logit(semantic_logit * candidate_gate)
        # Keep scores in logit space. No sigmoid here, so ranking cannot collapse
        # into the 0.99/1.00 plateau observed in v5.
        view_logits = 0.68 * dense_logit + 0.22 * global_logit + 0.10 * semantic_support
        sample_logit = None
        if groups is not None:
            if view_logits.shape[0] != groups * 5:
                raise ValueError(f"batch has {view_logits.shape[0]} views, expected {groups * 5}")
            views = view_logits.view(groups, 5)
            top2 = views.topk(2, dim=1).values
            sample_logit = 0.70 * top2[:, 0] + 0.30 * top2.mean(1)

        return V6Output(
            final_logit=final_logit,
            dino_logit=dino_logit,
            dino_decoder_logit=decoder_logit,
            dino_reference_logit=reference_logit,
            vlm_residual=vlm_residual,
            candidate_gate=candidate_gate,
            semantic_logit=semantic_logit,
            state_logits=state_logits,
            global_logit=global_logit,
            view_logits=view_logits,
            sample_logit=sample_logit,
            static_text=static_text,
            patch_features=patch,
            dino_reference_weight=reference_weight,
        )


def load_trusted_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)
