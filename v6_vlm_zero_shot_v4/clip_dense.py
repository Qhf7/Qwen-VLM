"""Frozen OpenAI CLIP ViT with trainable prompt and local visual adapters."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from states import STATE_NAMES


def load_clip(model_name: str, weight: Path | None, device: torch.device):
    """Load a generic CLIP base checkpoint, never an anomaly-specific weight."""
    import clip

    source = str(weight) if weight is not None and weight.exists() else model_name
    model, preprocess = clip.load(source, device=str(device), jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


class PromptLearner(nn.Module):
    """CoOp-like soft context with image-conditioned prototype offsets.

    The CLIP text transformer stays frozen.  Only context embeddings, the
    dynamic MLP and a small state residual are optimised.
    """

    def __init__(self, clip_model, n_ctx: int = 8, dynamic_hidden: int = 256):
        super().__init__()
        import clip

        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.dtype = clip_model.dtype
        width = int(clip_model.ln_final.weight.numel())
        text_dim = int(clip_model.text_projection.shape[1])
        image_dim = int(clip_model.visual.output_dim)
        initial = "industrial component surface inspection"
        tokenised = clip.tokenize([initial])
        with torch.no_grad():
            init_embed = clip_model.token_embedding(tokenised.to(clip_model.token_embedding.weight.device))[0, 1 : 1 + n_ctx]
        if init_embed.shape[0] < n_ctx:
            pad = torch.randn(n_ctx - init_embed.shape[0], width, device=init_embed.device) * 0.02
            init_embed = torch.cat([init_embed, pad], dim=0)
        self.context = nn.Parameter(init_embed.float().clone())
        self.register_buffer("context_anchor", init_embed.float().clone(), persistent=True)
        self.state_residual = nn.Parameter(torch.zeros(len(STATE_NAMES), text_dim))
        self.dynamic = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, dynamic_hidden),
            nn.GELU(),
            nn.Linear(dynamic_hidden, text_dim),
        )

        token_rows = []
        suffixes = []
        for name in STATE_NAMES:
            tokens = clip.tokenize(["X " * n_ctx + name])[0]
            token_rows.append(tokens)
            with torch.no_grad():
                emb = clip_model.token_embedding(tokens.to(clip_model.token_embedding.weight.device))
            suffixes.append(emb[1 + n_ctx :].float())
        tokenised = torch.stack(token_rows).to(clip_model.token_embedding.weight.device)
        self.register_buffer("tokenised", tokenised, persistent=True)
        self.register_buffer("prefix", clip_model.token_embedding(tokenised[:, :1]).float(), persistent=True)
        self.register_buffer("suffix", torch.stack(suffixes), persistent=True)
        self.text_dim = text_dim

    def _encode_prompts(self) -> torch.Tensor:
        model = self.clip_model
        n = len(STATE_NAMES)
        context = self.context[None].expand(n, -1, -1)
        prompts = torch.cat([self.prefix, context, self.suffix], dim=1).to(self.dtype)
        x = prompts + model.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).to(self.dtype)
        eot = self.tokenised.argmax(dim=-1)
        out = x[torch.arange(n, device=x.device), eot] @ model.text_projection
        out = out.float() + self.state_residual
        return F.normalize(out, dim=-1)

    def forward(self, global_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        static = self._encode_prompts()
        dynamic = self.dynamic(global_image.float())
        # The same material/illumination offset is state-conditioned by the
        # static prototype, avoiding a pure translation that cosine-normalise
        # would largely cancel.
        conditioned = dynamic[:, None, :] * static[None, :, :]
        text = F.normalize(static[None, :, :] + 0.20 * conditioned, dim=-1)
        return text, static

    def anchor_loss(self) -> torch.Tensor:
        return F.mse_loss(self.context, self.context_anchor)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        self.scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.scale) * self.net(x)


def _interpolate_positional_embedding(pos: torch.Tensor, side: int) -> torch.Tensor:
    cls, patch = pos[:1], pos[1:]
    old_side = int(round(math.sqrt(patch.shape[0])))
    if old_side == side:
        return pos
    patch = patch.reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)
    patch = F.interpolate(patch.float(), size=(side, side), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(side * side, -1).to(pos.dtype)
    return torch.cat([cls, patch], dim=0)


class DenseCLIPEncoder(nn.Module):
    """Extract global and patch CLIP embeddings from a frozen ViT backbone."""

    def __init__(self, clip_model, adapter_dim: int = 64):
        super().__init__()
        visual = clip_model.visual
        if not hasattr(visual, "conv1") or visual.proj is None:
            raise TypeError("v4 requires a CLIP VisionTransformer, e.g. ViT-B/16")
        self.visual = visual
        width = int(visual.conv1.out_channels)
        out_dim = int(visual.output_dim)
        self.adapter = ResidualAdapter(width, adapter_dim)
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, out_dim, bias=False),
        )
        with torch.no_grad():
            self.patch_projection[1].weight.copy_(visual.proj.T.float())
        self.window_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.visual
        dtype = v.conv1.weight.dtype
        x = v.conv1(x.to(dtype))
        side = x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(dtype)[None, None].expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = _interpolate_positional_embedding(v.positional_embedding, side)
        x = v.ln_pre(x + pos.to(dtype)[None])
        x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        global_feature = v.ln_post(x[:, 0]) @ v.proj
        patch = self.adapter(x[:, 1:].float())
        patch = self.patch_projection(patch)
        patch = F.normalize(patch, dim=-1).view(x.shape[0], side, side, -1)
        return F.normalize(global_feature.float(), dim=-1), patch

    def window_features(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.permute(0, 3, 1, 2)
        scales = [x]
        for kernel in (3, 5):
            scales.append(F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2))
        weights = torch.softmax(self.window_logits, dim=0)
        fused = sum(w * F.normalize(s, dim=1) for w, s in zip(weights, scales))
        return F.normalize(fused, dim=1).permute(0, 2, 3, 1)


def clip_preprocess_tensor(images, resolution: int, device: torch.device) -> torch.Tensor:
    import numpy as np

    arrays = []
    for image in images:
        image = image.resize((resolution, resolution), resample=3)
        arrays.append(torch.from_numpy(np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0))
    x = torch.stack(arrays).to(device, non_blocking=True)
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device)[None, :, None, None]
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)[None, :, None, None]
    return (x - mean) / std
