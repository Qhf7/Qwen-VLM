"""VLM-primary anomaly model inspired by WinCLIP and AdaCLIP.

The frozen generic CLIP backbone is the primary predictor.  Multi-scale local
CLIP tokens, hybrid prompts, and CLIP normal-reference association form the
VLM anomaly map.  DINOv2 is deliberately restricted to a bounded residual so
it can improve localisation without replacing the zero-shot semantic ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_dense import DenseCLIPEncoder, PromptLearner
from multiscale_decoder import LayerAwarePixelDecoder
from states import FIRST_ANOMALY, STATE_NAMES


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-mean-exp, which is invariant to the number of prompt states."""
    count = x.shape[dim]
    return torch.logsumexp(x, dim=dim) - x.new_tensor(float(count)).log()


class ConvBlock(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )


class LocalVLMDecoder(nn.Module):
    """CLIP-only local decoder; DINO features never enter this branch."""

    def __init__(self, clip_dim: int, width: int, states: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(clip_dim, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.context = nn.Sequential(ConvBlock(width), ConvBlock(width))
        self.state_residual = nn.Conv2d(width, states, 1)

    def forward(self, patch: torch.Tensor, target_side: int) -> torch.Tensor:
        x = patch.permute(0, 3, 1, 2).contiguous()
        x = self.context(self.proj(x))
        x = F.interpolate(x, (target_side, target_side), mode="bilinear", align_corners=False)
        return self.state_residual(x)


@dataclass
class V5Output:
    final_map: torch.Tensor
    vlm_map: torch.Tensor
    semantic_map: torch.Tensor
    reference_map: torch.Tensor
    dino_map: torch.Tensor
    semantic_logit: torch.Tensor
    state_logits: torch.Tensor
    global_logit: torch.Tensor
    view_probability: torch.Tensor
    sample_probability: torch.Tensor | None
    static_text: torch.Tensor
    patch_features: torch.Tensor
    dino_fraction: torch.Tensor


class VLMPrimaryV5(nn.Module):
    """CLIP/OpenCLIP-style local alignment with a bounded DINO residual."""

    def __init__(
        self,
        dino_dims: list[int],
        clip_model,
        decoder_width: int = 192,
        vlm_width: int = 192,
        prompt_ctx: int = 8,
        max_dino_fraction: float = 0.15,
    ):
        super().__init__()
        self.dino_decoder = LayerAwarePixelDecoder(dino_dims, width=decoder_width)
        self.clip_encoder = DenseCLIPEncoder(clip_model, adapter_dim=64)
        self.prompt = PromptLearner(clip_model, n_ctx=prompt_ctx)
        clip_dim = int(clip_model.visual.output_dim)
        self.local_decoder = LocalVLMDecoder(clip_dim, vlm_width, len(STATE_NAMES))
        self.semantic_temperature = nn.Parameter(torch.tensor(3.0))
        self.reference_weight_logit = nn.Parameter(torch.tensor(-0.85))
        self.dino_fraction_logit = nn.Parameter(torch.tensor(1.35))
        self.max_dino_fraction = float(max_dino_fraction)

    def load_v3_decoder(self, checkpoint: dict) -> None:
        state = checkpoint.get("state", checkpoint)
        missing, unexpected = self.dino_decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"v3 decoder mismatch: missing={missing}, unexpected={unexpected}")

    def load_v4_adaptation(self, checkpoint: dict, reuse_vlm: bool = False) -> dict[str, int]:
        """Reuse v4's useful components while discarding Gate/Query/MIL.

        By default only the trained multi-scale DINO decoder is reused.  The
        v4 VLM adaptation overfit its pseudo generators on the real board, so
        v5 starts Prompt/Adapter from the generic CLIP representation unless
        an ablation explicitly requests ``reuse_vlm=True``.
        """
        source = checkpoint.get("model", checkpoint)
        allowed_prefixes = ("dino_decoder.",)
        if reuse_vlm:
            allowed_prefixes += ("clip_encoder.adapter.", "clip_encoder.patch_projection.", "prompt.")
        current = self.state_dict()
        selected = {
            key: value
            for key, value in source.items()
            if key.startswith(allowed_prefixes) and key in current and current[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(selected, strict=False)
        return {
            "loaded": len(selected),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        excluded = ("clip_encoder.visual.", "prompt.clip_model.")
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith(excluded)
        }

    def load_compact_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = ("clip_encoder.visual.", "prompt.clip_model.")
        invalid = [key for key in missing if not key.startswith(allowed)]
        if invalid or unexpected:
            raise RuntimeError(f"v5 checkpoint mismatch: missing={invalid}, unexpected={unexpected}")

    @staticmethod
    def _topk(x: torch.Tensor, fraction: float = 0.002) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(8, int(round(flat.shape[1] * fraction)))
        return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)

    @staticmethod
    def _state_margin(logits: torch.Tensor, state_dim: int) -> torch.Tensor:
        normal = logits.narrow(state_dim, 1, FIRST_ANOMALY - 1)
        anomaly = logits.narrow(state_dim, FIRST_ANOMALY, len(STATE_NAMES) - FIRST_ANOMALY)
        return logmeanexp(anomaly, state_dim) - logmeanexp(normal, state_dim)

    def encode_clip(self, clip_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_feature, patch = self.clip_encoder(clip_images)
        return global_feature, self.clip_encoder.window_features(patch)

    def forward(
        self,
        dino_maps: list[torch.Tensor],
        clip_images: torch.Tensor,
        reference_map: torch.Tensor | None = None,
        groups: int | None = None,
        encoded_clip: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> V5Output:
        global_clip, patch = encoded_clip if encoded_clip is not None else self.encode_clip(clip_images)
        text, static_text = self.prompt(global_clip)

        dino_logits, _ = self.dino_decoder(dino_maps)
        target_side = int(dino_logits.shape[-1])

        # WinCLIP-style local multi-scale CLIP-to-text alignment.  Log-mean-exp
        # removes the severe class-count bias of summing eight anomaly classes.
        patch_similarity = torch.einsum("bhwd,bcd->bchw", patch, text)
        temperature = self.semantic_temperature.exp().clamp(1.0, 50.0)
        patch_similarity = patch_similarity * temperature
        semantic_logits = F.interpolate(
            patch_similarity, (target_side, target_side), mode="bilinear", align_corners=False
        )
        state_logits = semantic_logits + self.local_decoder(patch, target_side)
        semantic_logit = self._state_margin(state_logits, 1)
        semantic_map = torch.sigmoid(semantic_logit)

        if reference_map is None:
            reference = torch.zeros_like(semantic_map)
        else:
            reference = F.interpolate(
                reference_map[:, None].float(), semantic_map.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0].clamp(0, 1)

        # WinCLIP+ normal-reference association remains inside the VLM branch.
        reference_weight = 0.15 + 0.25 * torch.sigmoid(self.reference_weight_logit)
        vlm_map = ((1.0 - reference_weight) * semantic_map + reference_weight * reference).clamp(0, 1)

        dino_map = torch.sigmoid(dino_logits)
        dino_fraction = self.max_dino_fraction * torch.sigmoid(self.dino_fraction_logit)
        final_map = ((1.0 - dino_fraction) * vlm_map + dino_fraction * dino_map).clamp(0, 1)

        global_similarity = torch.einsum("bd,bcd->bc", global_clip, text) * temperature
        global_logit = self._state_margin(global_similarity, 1)
        semantic_score = self._topk(semantic_map)
        reference_score = self._topk(reference)
        # VLM contributes 95% of image evidence; DINO is a bounded 5% tie-breaker.
        view_probability = (
            0.60 * self._topk(vlm_map)
            + 0.22 * torch.sigmoid(global_logit)
            + 0.13 * reference_score
            + 0.05 * self._topk(dino_map)
        ).clamp(1e-5, 1 - 1e-5)

        sample_probability = None
        if groups is not None:
            if view_probability.shape[0] != groups * 5:
                raise ValueError(f"batch has {view_probability.shape[0]} views, expected {groups * 5}")
            views = view_probability.view(groups, 5)
            top2 = views.topk(2, dim=1).values
            sample_probability = (0.70 * top2[:, 0] + 0.30 * top2.mean(1)).clamp(1e-5, 1 - 1e-5)

        return V5Output(
            final_map=final_map,
            vlm_map=vlm_map,
            semantic_map=semantic_map,
            reference_map=reference,
            dino_map=dino_map,
            semantic_logit=semantic_logit,
            state_logits=state_logits,
            global_logit=global_logit,
            view_probability=view_probability,
            sample_probability=sample_probability,
            static_text=static_text,
            patch_features=patch,
            dino_fraction=dino_fraction,
        )


def load_trusted_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
