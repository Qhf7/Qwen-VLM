"""Layer-aware decoder and training losses for tiny industrial defects."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )


class LayerAwarePixelDecoder(nn.Module):
    """FPN-style decoder over shallow/middle/deep DINOv2 patch maps.

    Each depth gets a separate projection. The decoder then builds a top-down
    pyramid at patch, 2x patch and 4x patch resolution. Predicting beyond the
    native token grid does not invent information; it lets local convolutional
    context and high-resolution supervision preserve narrow boundaries that a
    direct bilinear resize would smear.
    """

    def __init__(self, dims: list[int], width: int = 192):
        super().__init__()
        if len(dims) < 3:
            raise ValueError("LayerAwarePixelDecoder requires at least 3 DINO layers")
        self.dims = list(dims)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, width, 1, bias=False), nn.GroupNorm(8, width), nn.GELU())
            for d in dims
        ])
        self.depth_logits = nn.Parameter(torch.zeros(len(dims)))
        self.context = nn.Sequential(ConvNormAct(width, width, 1), ConvNormAct(width, width, 2))
        self.up2 = nn.Sequential(ConvNormAct(width, width // 2), ConvNormAct(width // 2, width // 2))
        self.up4 = nn.Sequential(ConvNormAct(width // 2, width // 4), ConvNormAct(width // 4, width // 4))
        self.aux = nn.Conv2d(width, 1, 1)
        self.out = nn.Conv2d(width // 4, 1, 1)

    def forward(self, maps: list[torch.Tensor]):
        # Input maps are [B,S,S,D]. Preserve depth identity before fusion.
        projected = [p(m.permute(0, 3, 1, 2).contiguous()) for p, m in zip(self.proj, maps)]
        weights = torch.softmax(self.depth_logits, dim=0)
        fused = sum(w * x for w, x in zip(weights, projected))
        fused = self.context(fused)
        aux = self.aux(fused).squeeze(1)
        x = F.interpolate(fused, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)
        return self.out(x).squeeze(1), aux


def boundary_map(target: torch.Tensor) -> torch.Tensor:
    t = target[:, None]
    dil = F.max_pool2d(t, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-t, 3, stride=1, padding=1)
    return (dil - ero)[:, 0].clamp(0, 1)


class MultiScalePixelLoss(nn.Module):
    """Precision-oriented segmentation loss with explicit boundary learning."""

    def forward(self, logits, aux, target, hard_negative=None):
        target = F.interpolate(target[:, None], logits.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        aux_target = F.interpolate(target[:, None], aux.shape[-2:], mode="area")[:, 0]
        pos = target.sum().clamp_min(1.0)
        neg = (1.0 - target).sum().clamp_min(1.0)
        pos_weight = (neg / pos).clamp(2.0, 20.0).detach()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        p = torch.sigmoid(logits)
        tp = (p * target).sum((1, 2))
        fp = (p * (1 - target)).sum((1, 2))
        fn = ((1 - p) * target).sum((1, 2))
        tversky = 1 - ((tp + 1) / (tp + 0.72 * fp + 0.28 * fn + 1)).mean()
        edge = boundary_map(target)
        edge_loss = F.binary_cross_entropy_with_logits(logits, target, weight=1 + 2.0 * edge)
        aux_loss = F.binary_cross_entropy_with_logits(aux, aux_target)
        loss = 0.45 * bce + 0.85 * tversky + 0.25 * edge_loss + 0.15 * aux_loss
        if hard_negative is not None:
            hn = F.interpolate(hard_negative[:, None], logits.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            hn = hn * (target < 0.15)
            if hn.sum() > 0:
                loss = loss + 0.45 * (
                    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * hn
                ).sum() / hn.sum().clamp_min(1)
        return loss
