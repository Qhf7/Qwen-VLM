"""Cross-modal trainable anomaly model built on frozen DINOv2 and CLIP."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_dense import DenseCLIPEncoder, PromptLearner
from multiscale_decoder import LayerAwarePixelDecoder
from states import ANOMALY_STATES, FIRST_ANOMALY, STATE_NAMES


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )


class CrossModalPixelHead(nn.Module):
    def __init__(self, dino_dims: list[int], clip_dim: int, width: int = 192):
        super().__init__()
        self.dino_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(d, width // 2, 1, bias=False), nn.GroupNorm(8, width // 2), nn.GELU()) for d in dino_dims]
        )
        self.dino_weights = nn.Parameter(torch.zeros(len(dino_dims)))
        self.clip_proj = nn.Sequential(
            nn.Conv2d(clip_dim, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            ConvBlock(width + width // 2, width),
            ConvBlock(width, width),
        )
        self.state_residual = nn.Conv2d(width, len(STATE_NAMES), 1)
        self.query_visual = nn.Conv2d(width, clip_dim, 1, bias=False)

    def forward(self, dino_maps, clip_patch, target_side: int):
        weights = torch.softmax(self.dino_weights, dim=0)
        dino = []
        for weight, proj, feature in zip(weights, self.dino_proj, dino_maps):
            dino.append(weight * proj(feature.permute(0, 3, 1, 2).contiguous()))
        dino = sum(dino)
        dino = F.interpolate(dino, size=(target_side, target_side), mode="bilinear", align_corners=False)
        clip = self.clip_proj(clip_patch.permute(0, 3, 1, 2).contiguous())
        clip = F.interpolate(clip, size=(target_side, target_side), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([dino, clip], dim=1))
        return fused, self.state_residual(fused), F.normalize(self.query_visual(fused), dim=1)


class DynamicGate(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )

    def forward(self, dino, vlm, query, entropy, seen):
        agreement = 1.0 - torch.abs(dino - vlm)
        seen_map = torch.ones_like(dino) * seen[:, None, None]
        x = torch.stack([dino, vlm, query, entropy, agreement, seen_map], dim=1)
        weights = torch.softmax(self.net(x), dim=1)
        maps = torch.stack([dino, vlm, query], dim=1)
        return (weights * maps).sum(1), weights


class FiveViewMIL(nn.Module):
    def __init__(self, feature_dim: int = 5, hidden: int = 48):
        super().__init__()
        self.view_score = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.view_attention = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.sample_head = nn.Sequential(nn.Linear(feature_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor):
        # features [groups, 5, feature_dim]
        view_logits = self.view_score(features).squeeze(-1)
        attention = torch.softmax(self.view_attention(features).squeeze(-1), dim=1)
        pooled = (features * attention[..., None]).sum(1)
        strongest = view_logits.max(1).values[:, None]
        sample_logit = self.sample_head(torch.cat([pooled, strongest], dim=1)).squeeze(-1)
        return sample_logit, view_logits, attention


@dataclass
class V4Output:
    final_map: torch.Tensor
    dino_map: torch.Tensor
    vlm_map: torch.Tensor
    query_map: torch.Tensor
    state_logits: torch.Tensor
    query_logits: torch.Tensor
    gate_weights: torch.Tensor
    sample_logits: torch.Tensor | None
    view_logits: torch.Tensor | None
    static_text: torch.Tensor
    patch_features: torch.Tensor


class VLMAnomalyV4(nn.Module):
    def __init__(
        self,
        dino_dims: list[int],
        clip_model,
        decoder_width: int = 192,
        fusion_width: int = 192,
        prompt_ctx: int = 8,
    ):
        super().__init__()
        self.dino_decoder = LayerAwarePixelDecoder(dino_dims, width=decoder_width)
        self.clip_encoder = DenseCLIPEncoder(clip_model, adapter_dim=64)
        self.prompt = PromptLearner(clip_model, n_ctx=prompt_ctx)
        clip_dim = int(clip_model.visual.output_dim)
        self.pixel_head = CrossModalPixelHead(dino_dims, clip_dim, fusion_width)
        self.query_delta = nn.Parameter(torch.zeros(len(ANOMALY_STATES), clip_dim))
        self.logit_scale = nn.Parameter(torch.tensor(3.0))
        self.query_scale = nn.Parameter(torch.tensor(3.0))
        self.gate = DynamicGate()
        self.multiview = FiveViewMIL()

    def load_v3_decoder(self, checkpoint: dict) -> None:
        state = checkpoint.get("state", checkpoint)
        missing, unexpected = self.dino_decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"v3 decoder mismatch: missing={missing}, unexpected={unexpected}")

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint trainable/adaptation state without duplicating CLIP.

        The frozen generic CLIP checkpoint is loaded separately and recorded
        by path/hash.  Serialising it through two shared module references
        would create multi-gigabyte v4 checkpoints for no benefit.
        """
        excluded = ("clip_encoder.visual.", "prompt.clip_model.")
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith(excluded)
        }

    def load_compact_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = ("clip_encoder.visual.", "prompt.clip_model.")
        invalid_missing = [key for key in missing if not key.startswith(allowed)]
        if invalid_missing or unexpected:
            raise RuntimeError(f"v4 checkpoint mismatch: missing={invalid_missing}, unexpected={unexpected}")

    @staticmethod
    def _topk(x: torch.Tensor, fraction: float = 0.003) -> torch.Tensor:
        flat = x.flatten(1)
        k = max(4, int(round(flat.shape[1] * fraction)))
        return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)

    def forward(
        self,
        dino_maps: list[torch.Tensor],
        clip_images: torch.Tensor,
        seen: torch.Tensor | None = None,
        groups: int | None = None,
        memory_map: torch.Tensor | None = None,
    ) -> V4Output:
        global_clip, clip_patch = self.clip_encoder(clip_images)
        clip_patch = self.clip_encoder.window_features(clip_patch)
        text, static_text = self.prompt(global_clip)

        # Direct CLIP patch-state logits are the semantic anchor.
        semantic = torch.einsum("bhwd,bcd->bchw", clip_patch, text)
        semantic = semantic * self.logit_scale.exp().clamp(max=100.0)

        dino_logits, _ = self.dino_decoder(dino_maps)
        target_side = int(dino_logits.shape[-1])
        fused, state_residual, query_visual = self.pixel_head(dino_maps, clip_patch, target_side)
        semantic = F.interpolate(semantic, size=(target_side, target_side), mode="bilinear", align_corners=False)
        state_logits = semantic + state_residual
        state_probability = torch.softmax(state_logits, dim=1)
        vlm_map = state_probability[:, FIRST_ANOMALY:].sum(1)

        query_text = static_text[FIRST_ANOMALY:] + self.query_delta
        query_text = F.normalize(query_text, dim=-1)
        query_logits = torch.einsum("bdhw,kd->bkhw", query_visual, query_text)
        query_logits = query_logits * self.query_scale.exp().clamp(max=100.0)
        query_map = 1.0 - torch.prod(1.0 - torch.sigmoid(query_logits).clamp(1e-5, 1 - 1e-5), dim=1)
        dino_map = torch.sigmoid(dino_logits)
        if memory_map is not None:
            memory_map = F.interpolate(memory_map[:, None].float(), dino_map.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            dino_map = (0.68 * dino_map + 0.32 * memory_map.clamp(0, 1)).clamp(0, 1)

        entropy = -(state_probability * state_probability.clamp_min(1e-6).log()).sum(1)
        entropy = entropy / float(torch.log(torch.tensor(len(STATE_NAMES), device=entropy.device)))
        if seen is None:
            seen = torch.ones(dino_map.shape[0], device=dino_map.device)
        final_map, gate_weights = self.gate(dino_map, vlm_map, query_map, entropy, seen.float())

        sample_logits = view_logits = None
        if groups is not None:
            if final_map.shape[0] != groups * 5:
                raise ValueError(f"batch has {final_map.shape[0]} views, expected {groups * 5}")
            global_state = torch.einsum("bd,bcd->bc", global_clip, text)
            normal_margin = global_state[:, FIRST_ANOMALY:].logsumexp(1) - global_state[:, 1:FIRST_ANOMALY].logsumexp(1)
            features = torch.stack(
                [
                    self._topk(final_map),
                    self._topk(dino_map),
                    self._topk(vlm_map),
                    self._topk(query_map),
                    normal_margin,
                ],
                dim=1,
            ).view(groups, 5, -1)
            sample_logits, view_logits, _ = self.multiview(features)

        return V4Output(
            final_map=final_map,
            dino_map=dino_map,
            vlm_map=vlm_map,
            query_map=query_map,
            state_logits=state_logits,
            query_logits=query_logits,
            gate_weights=gate_weights,
            sample_logits=sample_logits,
            view_logits=view_logits,
            static_text=static_text,
            patch_features=clip_patch,
        )
