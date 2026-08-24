#!/usr/bin/env python3
"""The three anomaly branches: reconstruction, coreset memory, pseudo head.

Each branch produces a raw score map on the backbone patch grid.  They are
complementary in failure mode, which is why they are fused:

* reconstruction  - strong on structural/textural deviation, robust to rare
                    but normal appearance because the decoder learns the
                    whole normal manifold rather than memorising exemplars.
* coreset memory  - strong on "never seen this local pattern" evidence,
                    needs no training, transfers to unseen categories via a
                    global bank.
* pseudo head     - explicitly discriminative, sharpens small defects that
                    the other two blur, because it is trained with defect
                    masks at the correct physical scale.

Compliance: all training here uses official Train normal images plus
synthetic defects derived from them.  Backbones stay frozen and generic.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# branch A: feature reconstruction (Dinomaly-style)
# ==========================================================================


class LinearAttention(nn.Module):
    """Linear attention used instead of softmax attention.

    Dinomaly's observation: softmax attention lets the decoder focus sharply
    and reconstruct anomalies too, collapsing to an identity shortcut.  A
    deliberately "weak" linear attention cannot represent that shortcut as
    easily, so reconstruction error stays high on anomalies.
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # The linear-attention accumulators sum over all N tokens.  At the
        # resolutions used here (74x74 = 5476 tokens) the k-sum reaches ~6500
        # and q.ksum overflows fp16's 65504 limit to inf, so 1/denom becomes 0
        # and the backward pass yields NaN.  Run the accumulation in fp32
        # regardless of the surrounding autocast context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = (F.elu(q.float()) + 1.0)
            k = (F.elu(k.float()) + 1.0)
            v = v.float()
            # normalising by the token count keeps the accumulators O(1) and
            # makes the attention output independent of sequence length
            k = k / float(n)
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
            denom = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))
            z = 1.0 / denom.clamp_min(1e-4)
            out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.transpose(1, 2).reshape(b, n, d).to(x.dtype)
        return self.proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.2):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class ReconstructionBranch(nn.Module):
    """Reconstructs frozen encoder features of normal images.

    Trained on official Train normals only.  Anomaly score at test time is
    the cosine distance between encoder features and their reconstruction,
    averaged over supervised layer groups.
    """

    def __init__(self, dims: list[int], depth: int = 4, heads: int = 8, noise: float = 0.25):
        super().__init__()
        self.dims = dims
        self.noise = noise
        # Group encoder layers into two fusion streams (shallow, deep) as in
        # Dinomaly's group-to-group supervision: it avoids the decoder simply
        # copying one layer into the next.
        self.n_groups = 2
        self.in_proj = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        width = min(768, max(dims))
        self.width = width
        for d in dims:
            self.in_proj.append(nn.Linear(d, width))
            self.out_proj.append(nn.Linear(width, d))
        self.blocks = nn.ModuleList([DecoderBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def group_of(self, layer_idx: int) -> int:
        return 0 if layer_idx < len(self.dims) // 2 else 1

    def forward(self, feats: list[torch.Tensor]):
        """feats: list of [B,N,Dl] encoder features (already layer-normalised).

        Returns reconstructions in the same shapes.
        """
        # fuse the encoder layers into a single bottleneck token sequence
        proj = [self.in_proj[i](f) for i, f in enumerate(feats)]
        x = torch.stack(proj, dim=0).mean(dim=0)
        if self.training and self.noise > 0:
            # noisy bottleneck: forces the decoder to model the normal
            # distribution instead of learning an identity map
            x = x + torch.randn_like(x) * self.noise * x.detach().std()
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return [self.out_proj[i](x) for i in range(len(feats))]


def reconstruction_score(feats, recons, side: int) -> torch.Tensor:
    """Per-patch anomaly score from cosine distance, averaged across layers."""
    scores = []
    for f, r in zip(feats, recons):
        d = 1.0 - F.cosine_similarity(F.normalize(f, dim=-1), F.normalize(r, dim=-1), dim=-1)
        scores.append(d)
    s = torch.stack(scores, dim=0).mean(dim=0)
    b = s.shape[0]
    return s.view(b, side, side)


# ==========================================================================
# branch C: pseudo-anomaly discriminative head
# ==========================================================================


class PseudoHead(nn.Module):
    """Small conv head over frozen multi-layer patch features.

    Convolutional rather than per-patch linear: a defect's context (a dark
    line against uniform surroundings) is a local spatial pattern, and 3x3
    context measurably improves small-defect precision.  Kept shallow so the
    per-frame latency budget is respected.
    """

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, x):
        """x: [B,D,S,S] -> logits [B,S,S]"""
        return self.body(x).squeeze(1)


class FocalTverskyLoss(nn.Module):
    """Loss tuned for the metric that actually matters.

    P-F1max and P-AUPR care about the precision/recall balance on a tiny
    positive class (~0.1% of pixels).  Plain BCE with pos_weight pushes
    recall and floods false positives.  Tversky with beta>alpha penalises
    false negatives more than false positives while the focal exponent
    concentrates gradient on hard cases, which empirically gives sharper
    boundaries than BCE at the same recall.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.3333):
        super().__init__()
        self.alpha = alpha  # weight on false positives
        self.beta = beta  # weight on false negatives
        self.gamma = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        p = p.reshape(p.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        return ((1.0 - tversky) ** self.gamma).mean()


class HeadLoss(nn.Module):
    """Combination of BCE (calibration) and focal Tversky (shape)."""

    def __init__(self, pos_weight: float = 8.0, tversky_w: float = 1.0, bce_w: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.tv = FocalTverskyLoss()
        self.tversky_w = tversky_w
        self.bce_w = bce_w

    def forward(self, logits, target):
        return self.bce_w * self.bce(logits, target) + self.tversky_w * self.tv(logits, target)


# ==========================================================================
# branch B: coreset memory bank
# ==========================================================================


class MemoryBank:
    """Per-(category, view) coreset plus a category-agnostic global bank.

    The global bank is what makes zero-shot possible: an unseen category has
    no Train normals, but industrial surfaces share a lot of low/mid-level
    normal structure, so distance to a broad global normal bank still
    separates defects from surfaces reasonably well.
    """

    def __init__(self, budget_per_key: int = 4096, global_budget: int = 16384, knn: int = 3):
        self.budget = budget_per_key
        self.global_budget = global_budget
        self.knn = knn
        self.banks: dict[tuple, np.ndarray] = {}
        self.global_bank: np.ndarray | None = None

    def set(self, key, vectors: np.ndarray) -> None:
        self.banks[key] = np.ascontiguousarray(vectors, np.float32)

    def set_global(self, vectors: np.ndarray) -> None:
        self.global_bank = np.ascontiguousarray(vectors, np.float32)

    def bank_for(self, key) -> np.ndarray:
        b = self.banks.get(key)
        if b is None or len(b) == 0:
            return self.global_bank if self.global_bank is not None else np.zeros((0, 1), np.float32)
        return b

    def score(self, key, patches: np.ndarray, device=None) -> np.ndarray:
        """patches: [N,D] -> distance [N]"""
        bank = self.bank_for(key)
        if len(bank) == 0:
            return np.zeros(len(patches), np.float32)
        from common import knn_distance, knn_distance_torch

        if device is not None and device.type == "cuda":
            q = torch.from_numpy(np.ascontiguousarray(patches, np.float32)).to(device)
            b = torch.from_numpy(bank).to(device)
            return knn_distance_torch(q, b, self.knn).cpu().numpy()
        return knn_distance(patches, bank, self.knn)


def robust_bank_from_samples(
    vectors: np.ndarray, budget: int, contamination_q: float = 97.0, seed: int = 0
) -> np.ndarray:
    """Build a coreset, optionally trimming likely-contaminated vectors.

    Used for the transductive (unlabelled-test) bank: if a minority of test
    samples are anomalous, their patches sit far from the bulk of the
    distribution.  Trimming the top quantile of mean-distance-to-centroid
    removes most of that contamination before coreset selection.
    """
    from common import greedy_coreset

    x = np.ascontiguousarray(vectors, np.float32)
    if len(x) == 0:
        return x
    if contamination_q < 100.0 and len(x) > 64:
        centre = x.mean(0)
        centre /= max(np.linalg.norm(centre), 1e-6)
        d = 1.0 - x @ centre
        keep = d <= np.percentile(d, contamination_q)
        if keep.sum() >= 32:
            x = x[keep]
    return greedy_coreset(x, budget, seed=seed)
