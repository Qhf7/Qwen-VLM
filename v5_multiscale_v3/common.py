#!/usr/bin/env python3
"""Shared infrastructure for the v4 anomaly detection pipeline.

Design notes tied to the A-board scoring formula S = 30*S_cls + 70*S_seg:

* S_seg = (P-AUROC + P-AUPR + P-F1max) / 3 is 70% of the score and is the
  current bottleneck (0.465).  P-AUPR and P-F1max are dominated by false
  positive pixels, so every design choice below favours spatial sharpness
  over smooth-but-broad heatmaps.
* Every metric is a per-category macro average.  Only within-category
  ranking matters, so calibration is done per category (and per view where
  a reference distribution exists).  Cross-category absolute score scale is
  irrelevant to the metrics.
* Patch grid resolution is the hard limit on achievable pixel AP.  A 37x37
  grid maps one patch to 12x12 px at 448, while real defects are 5-15 px.
  We therefore run DINOv2 at high input resolution.

Compliance: official competition data only, plus public generic DINOv2 /
CLIP weights.  No anomaly-dataset-specific weights, prompts or heads.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SEED = 20260814
MASK_SIZE = 448
VIEWS = 5

# DINOv2 patch size for the ViT-*/14 family.
PATCH = 14


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# dataset layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    category: str
    name: str
    paths: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.category}/{self.name}"


def list_samples(root: Path) -> list[Sample]:
    """Enumerate <root>/<category>/<sample>/{0..4}.png groups."""
    out: list[Sample] = []
    if not root.exists():
        return out
    for cat in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"):
        for sample in sorted(p for p in cat.iterdir() if p.is_dir()):
            paths = tuple(sample / f"{i}.png" for i in range(VIEWS))
            if all(p.exists() for p in paths):
                out.append(Sample(cat.name, sample.name, paths))
    return out


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def load_many(paths, workers: int = 8) -> list[Image.Image]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(load_rgb, paths))


# --------------------------------------------------------------------------
# multi-scale DINOv2 features
# --------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


DINOV2_URLS = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}

DINOV2_SPECS = {
    "dinov2_vits14": dict(embed_dim=384, depth=12, num_heads=6),
    "dinov2_vitb14": dict(embed_dim=768, depth=12, num_heads=12),
    "dinov2_vitl14": dict(embed_dim=1024, depth=24, num_heads=16),
}


def _build_dinov2_local(arch: str, weight: Path):
    """Instantiate DINOv2 from vendored architecture code + local weights.

    torch.hub.load needs network access to fetch the repo definition, which
    is not always available on a competition machine.  The architecture is
    reconstructed with timm's VisionTransformer, which matches the official
    DINOv2 ViT definition for the /14 family, and the public pretrained
    weights are loaded from disk.
    """
    import timm

    spec = DINOV2_SPECS[arch]
    model = timm.models.VisionTransformer(
        patch_size=PATCH,
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        init_values=1.0e-5,
        img_size=518,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(weight, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if "mask_token" not in k]
    if critical:
        raise RuntimeError(f"missing weights for {arch}: {critical[:6]}")
    return _TimmDinoWrapper(model)


class _TimmDinoWrapper(torch.nn.Module):
    """Exposes get_intermediate_layers with the DINOv2 hub signature."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False):
        outs = self.model.get_intermediate_layers(
            x, n=n, reshape=False, return_prefix_tokens=True, norm=True
        )
        result = []
        for tokens, prefix in outs:
            cls = prefix[:, 0]
            result.append((tokens, cls) if return_class_token else tokens)
        return result

    def forward(self, x):
        return self.model(x)


TIMM_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
}

# Minimum plausible checkpoint size per arch, used to reject truncated files.
MIN_CKPT_BYTES = {
    "dinov2_vits14": 80_000_000,
    "dinov2_vitb14": 320_000_000,
    "dinov2_vitl14": 1_100_000_000,
}


def _build_dinov2_timm(arch: str, device):
    """Load DINOv2 through timm's own pretrained-weight resolution.

    timm hosts the same public LVD-142M DINOv2 weights on the HF hub, which
    is usually reachable (and mirrorable via HF_ENDPOINT) where the Facebook
    CDN is slow.  Functionally identical weights to the official release.
    """
    import timm

    model = timm.create_model(
        TIMM_NAMES[arch],
        pretrained=True,
        num_classes=0,
        img_size=518,
        dynamic_img_size=True,
    )
    return _TimmDinoWrapper(model)


def load_dinov2(arch: str, device: torch.device, weight_dir: Path | None = None):
    """Load a public generic DINOv2 backbone (no anomaly-specific tuning).

    Resolution order: a complete local checkpoint, then timm/HF pretrained,
    then torch.hub.  Truncated local files are ignored rather than silently
    loaded, because a partial checkpoint yields a randomly-initialised
    backbone that still produces plausible-looking feature maps.
    """
    local = None
    if weight_dir is not None:
        floor = MIN_CKPT_BYTES.get(arch, 10_000_000)
        for name in (f"{arch}_pretrain.pth", f"{arch}_timm.pth"):
            cand = Path(weight_dir) / name
            if cand.exists() and cand.stat().st_size >= floor:
                local = cand
                break
    if local is not None:
        try:
            model = _build_dinov2_local(arch, local)
        except Exception as exc:  # corrupt or mismatched file
            print(f"local checkpoint {local} unusable ({exc}); falling back", flush=True)
            local = None
    if local is None:
        try:
            model = _build_dinov2_timm(arch, device)
        except Exception as exc:
            print(f"timm pretrained failed ({exc}); trying torch.hub", flush=True)
            model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _to_tensor_batch(images, size: int, device: torch.device) -> torch.Tensor:
    arrs = []
    for im in images:
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        arrs.append(torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0))
    x = torch.stack(arrs).to(device, non_blocking=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None]
    return (x - mean) / std


# Which transformer depths to read.  Mid/late blocks carry the most useful
# signal for dense anomaly localisation; the very last block is too
# semantic/global for fine defects.
LAYER_SETS = {
    "dinov2_vits14": (3, 5, 7, 9),
    "dinov2_vitb14": (3, 6, 9, 11),
    "dinov2_vitl14": (6, 12, 18, 23),
}


@torch.inference_mode()
def dino_features(
    model,
    images,
    size: int,
    device: torch.device,
    layers: tuple[int, ...],
    autocast: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (cls[B,D], patch[B,S,S,D], side).

    Patch tokens from the selected depths are L2-normalised per layer, then
    concatenated.  Per-layer normalisation keeps deep layers with larger
    activation norms from dominating the distance metric.
    """
    x = _to_tensor_batch(images, size, device)
    use_amp = autocast and device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        outs = model.get_intermediate_layers(
            x, n=list(layers), reshape=False, return_class_token=True
        )
    patches, cls = [], []
    for tok, c in outs:
        patches.append(F.normalize(tok.float(), dim=-1))
        cls.append(F.normalize(c.float(), dim=-1))
    patch = F.normalize(torch.cat(patches, dim=-1), dim=-1)
    cls_t = F.normalize(torch.cat(cls, dim=-1), dim=-1)
    side = size // PATCH
    b, n, d = patch.shape
    assert n == side * side, (n, side)
    return (
        cls_t.cpu().numpy(),
        patch.view(b, side, side, d).cpu().numpy(),
        side,
    )


@torch.inference_mode()
def dino_features_torch(
    model, images, size, device, layers, autocast: bool = True
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Same as dino_features but keeps tensors on device (for training)."""
    x = _to_tensor_batch(images, size, device)
    use_amp = autocast and device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        outs = model.get_intermediate_layers(
            x, n=list(layers), reshape=False, return_class_token=True
        )
    patches = [F.normalize(t.float(), dim=-1) for t, _ in outs]
    cls = [F.normalize(c.float(), dim=-1) for _, c in outs]
    side = size // PATCH
    patch = F.normalize(torch.cat(patches, dim=-1), dim=-1)
    b, n, d = patch.shape
    return F.normalize(torch.cat(cls, dim=-1), dim=-1), patch.view(b, side, side, d), side


def per_layer_features(model, images, size, device, layers, autocast: bool = True):
    """Return a list of per-layer patch maps [B,S,S,Dl] (no concatenation).

    The reconstruction branch needs layers separately so the decoder can be
    supervised layer by layer, as in Dinomaly-style feature reconstruction.
    """
    x = _to_tensor_batch(images, size, device)
    use_amp = autocast and device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        outs = model.get_intermediate_layers(
            x, n=list(layers), reshape=False, return_class_token=True
        )
    side = size // PATCH
    maps = []
    for tok, _ in outs:
        t = tok.float()
        b, n, d = t.shape
        maps.append(t.view(b, side, side, d))
    return maps, side


# --------------------------------------------------------------------------
# spatial helpers
# --------------------------------------------------------------------------


def resize_map(x: np.ndarray, size: int = MASK_SIZE, mode: str = "bilinear") -> np.ndarray:
    """Resize a 2-D score map to size x size."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    t = F.interpolate(t, size=(size, size), mode=mode, **kw)
    return t[0, 0].numpy()


def neighbourhood_aggregate(patch: np.ndarray, radius: int = 1) -> np.ndarray:
    """PatchCore-style local aggregation over a [S,S,D] map.

    Average pooling over a (2r+1)^2 window makes each descriptor describe a
    small neighbourhood, which suppresses single-token noise without
    blurring the score map itself.
    """
    if radius <= 0:
        return patch
    t = torch.from_numpy(np.ascontiguousarray(patch, np.float32)).permute(2, 0, 1)[None]
    t = F.avg_pool2d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    t = t[0].permute(1, 2, 0).contiguous()
    return F.normalize(t, dim=-1).numpy()


def highpass(score: np.ndarray, sigma_frac: float = 0.14) -> np.ndarray:
    """Subtract a large-scale local mean from a score map.

    Rationale for pixel AP: a category/view often has a broad region that is
    systematically further from the normal manifold (edges, glare, dark
    background).  That offset barely hurts P-AUROC but floods P-AUPR and
    P-F1max with false positives.  Removing the low-frequency component
    keeps only locally-salient peaks, which is what a defect looks like.
    """
    if sigma_frac <= 0:
        return score
    h, w = score.shape
    k = max(3, int(round(min(h, w) * sigma_frac)) | 1)
    t = torch.from_numpy(np.ascontiguousarray(score, np.float32))[None, None]
    pad = k // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    base = F.avg_pool2d(t, kernel_size=k, stride=1)[0, 0].numpy()
    return score - base


def gaussian_blur(score: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return score
    radius = max(1, int(round(3 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    t = torch.from_numpy(np.ascontiguousarray(score, np.float32))[None, None]
    t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    t = F.conv2d(t, kernel.view(1, 1, 1, -1))
    t = F.conv2d(t, kernel.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def topk_mean(x: np.ndarray, frac: float = 0.005, min_k: int = 4) -> float:
    """Mean of the top fraction of values.

    More robust than a single max (one noisy token) and far more sensitive
    to small defects than a plain mean.
    """
    flat = np.asarray(x, np.float32).ravel()
    k = max(min_k, int(round(len(flat) * frac)))
    k = min(k, len(flat))
    if k <= 0:
        return 0.0
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


# --------------------------------------------------------------------------
# per-category calibration
# --------------------------------------------------------------------------

QUANTILES = (50.0, 90.0, 99.0, 99.9)


class Calibrator:
    """Maps raw branch scores to [0,1] using Train-normal distributions.

    A key consequence of macro-averaged metrics: only within-category
    ranking matters, so a per-category affine map is free.  We store
    quantiles of the *normal* score distribution and map the band between a
    low and a high normal quantile onto [0,1].  Anything a normal image
    routinely produces lands near 0; genuine outliers exceed 1 and clip.
    """

    def __init__(self, lo_q: float = 90.0, hi_q: float = 99.9):
        self.lo_q = lo_q
        self.hi_q = hi_q
        self.table: dict[tuple, np.ndarray] = {}
        self.fallback: np.ndarray | None = None

    def fit_key(self, key: tuple, values: np.ndarray) -> None:
        v = np.asarray(values, np.float32).ravel()
        if v.size == 0:
            return
        self.table[key] = np.percentile(v, QUANTILES).astype(np.float32)

    def fit_fallback(self, values: np.ndarray) -> None:
        v = np.asarray(values, np.float32).ravel()
        if v.size:
            self.fallback = np.percentile(v, QUANTILES).astype(np.float32)

    def _bounds(self, key) -> tuple[float, float]:
        q = self.table.get(key)
        if q is None:
            q = self.fallback
        if q is None:
            return 0.0, 1.0
        lo = float(np.interp(self.lo_q, QUANTILES, q))
        hi = float(np.interp(self.hi_q, QUANTILES, q))
        return lo, max(hi, lo + 1e-6)

    def apply(self, key, x):
        lo, hi = self._bounds(key)
        return np.clip((np.asarray(x, np.float32) - lo) / (hi - lo), 0.0, 1.0)

    def apply_unclipped(self, key, x):
        lo, hi = self._bounds(key)
        return (np.asarray(x, np.float32) - lo) / (hi - lo)

    def state(self):
        return {
            "lo_q": self.lo_q,
            "hi_q": self.hi_q,
            "table": {"|".join(map(str, k)): v.tolist() for k, v in self.table.items()},
            "fallback": None if self.fallback is None else self.fallback.tolist(),
        }


def rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] by rank.  Ties get averaged ranks."""
    v = np.asarray(values, np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    # average ties so identical scores do not get an arbitrary ordering
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return ranks / max(1, len(v) - 1)


# --------------------------------------------------------------------------
# coreset selection
# --------------------------------------------------------------------------


def greedy_coreset(x: np.ndarray, budget: int, seed: int = SEED) -> np.ndarray:
    """Greedy k-center (farthest-point) subsampling on unit vectors.

    PatchCore's coreset: keeps maximal coverage of the normal manifold for a
    fixed memory budget, which matters because k-NN recall against a badly
    covered bank produces false positives on rare-but-normal texture.
    """
    n = len(x)
    if n <= budget:
        return x
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n))
    chosen = [start]
    # cosine distance on unit vectors
    min_d = 1.0 - x @ x[start]
    for _ in range(budget - 1):
        idx = int(np.argmax(min_d))
        if min_d[idx] <= 0:
            break
        chosen.append(idx)
        min_d = np.minimum(min_d, 1.0 - x @ x[idx])
    return x[np.asarray(chosen, np.int64)]


def knn_distance(query: np.ndarray, bank: np.ndarray, k: int = 3, chunk: int = 8192) -> np.ndarray:
    """Mean cosine distance to the k nearest bank entries.

    k>1 is deliberate: a single nearest neighbour makes the score sensitive
    to one lucky bank vector, which inflates the false positive rate that
    P-AUPR punishes.
    """
    if len(bank) == 0:
        return np.zeros(len(query), np.float32)
    k = min(k, len(bank))
    out = np.empty(len(query), np.float32)
    bt = torch.from_numpy(np.ascontiguousarray(bank, np.float32))
    for st in range(0, len(query), chunk):
        q = torch.from_numpy(np.ascontiguousarray(query[st : st + chunk], np.float32))
        sim = q @ bt.T
        top = torch.topk(sim, k=k, dim=1).values
        out[st : st + chunk] = (1.0 - top.mean(dim=1)).numpy()
    return out


def knn_distance_torch(
    query: torch.Tensor, bank: torch.Tensor, k: int = 3, chunk: int = 16384
) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.zeros(len(query), device=query.device)
    k = min(k, len(bank))
    outs = []
    for st in range(0, len(query), chunk):
        sim = query[st : st + chunk] @ bank.T
        outs.append(1.0 - torch.topk(sim, k=k, dim=1).values.mean(dim=1))
    return torch.cat(outs)
