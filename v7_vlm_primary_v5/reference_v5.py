"""WinCLIP+-style normal-reference association in CLIP patch space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from common import greedy_coreset, knn_distance_torch, load_many
from object_prior import foreground_grid
from progress_v5 import ProgressReporter


def aligned_reference_map(current_patch: torch.Tensor, normal_patch: torch.Tensor, scale: float = 0.18):
    """Differentiable same-location reference distance for synthetic training.

    Programmatic anomalies preserve the camera geometry, so an official Train
    normal image is its own exact normal reference.  The fixed scale maps a
    cosine distance typical of a local edit into [0,1] without using labels.
    """
    current = F.normalize(current_patch.float(), dim=-1)
    normal = F.normalize(normal_patch.float(), dim=-1)
    distance = 1.0 - (current * normal).sum(-1)
    return (distance / max(float(scale), 1e-5)).clamp(0, 1)


def _reservoir(store, key, values, capacity, rng):
    values = np.asarray(values, np.float16)
    old = store.get(key)
    if old is None:
        if len(values) > capacity:
            values = values[rng.choice(len(values), capacity, replace=False)]
        store[key] = values.copy()
        return
    merged = np.concatenate([old, values], axis=0)
    if len(merged) > capacity:
        merged = merged[rng.choice(len(merged), capacity, replace=False)]
    store[key] = merged


@dataclass
class CLIPReferenceBank:
    banks: dict[tuple[str, int], np.ndarray]
    global_bank: np.ndarray
    knn: int = 3

    def bank_for(self, category: str, view: int) -> np.ndarray:
        bank = self.banks.get((category, view))
        return self.global_bank if bank is None or len(bank) == 0 else bank

    def score(self, features: torch.Tensor, categories: list[str], device: torch.device) -> torch.Tensor:
        """Return CLIP-reference distance maps for [B,H,W,D] features."""
        outputs = []
        for index, feature in enumerate(features):
            side = int(feature.shape[0])
            bank = torch.from_numpy(self.bank_for(categories[index], index % 5)).to(device)
            distance = knn_distance_torch(feature.reshape(-1, feature.shape[-1]), bank, self.knn)
            outputs.append(distance.reshape(side, side))
        return torch.stack(outputs)


def build_reference_bank(
    model,
    clip_preprocess_fn,
    samples,
    clip_size: int,
    device: torch.device,
    budget: int = 1024,
    global_budget: int = 8192,
    seed: int = 20260825,
    progress_path=None,
    stage: str = "clip_reference_bank",
) -> CLIPReferenceBank:
    """Build bounded per-category/view and global CLIP normal references."""
    rng = np.random.default_rng(seed)
    pools: dict[tuple[str, int], np.ndarray] = {}
    global_pool: dict[str, np.ndarray] = {}
    key_cap = max(budget * 3, budget + 512)
    global_cap = max(global_budget * 3, global_budget + 2048)
    progress = ProgressReporter(stage, len(samples), progress_path=progress_path)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            images = load_many(sample.paths)
            clip_x = clip_preprocess_fn(images, clip_size, device)
            _, patch = model.encode_clip(clip_x)
            arrays = patch.cpu().numpy()
            for view, array in enumerate(arrays):
                fg = foreground_grid(images[view], array.shape[0]).reshape(-1) > 0.20
                values = array.reshape(-1, array.shape[-1])
                fg_values = values[fg][::2]
                bg_values = values[~fg][::24]
                selected = np.concatenate([fg_values, bg_values], axis=0) if len(bg_values) else fg_values
                if not len(selected):
                    selected = values[::4]
                _reservoir(pools, (sample.category, view), selected, key_cap, rng)
                _reservoir(global_pool, "global", selected[::12], global_cap, rng)
            if index % 25 == 0 or index + 1 == len(samples):
                progress.update(index + 1, {"category": sample.category})
    banks = {}
    for key, values in pools.items():
        x = values.astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        banks[key] = np.ascontiguousarray(greedy_coreset(x, min(budget, len(x)), seed=seed))
    global_values = global_pool["global"].astype(np.float32)
    global_values /= np.maximum(np.linalg.norm(global_values, axis=1, keepdims=True), 1e-6)
    global_bank = np.ascontiguousarray(
        greedy_coreset(global_values, min(global_budget, len(global_values)), seed=seed)
    )
    return CLIPReferenceBank(banks=banks, global_bank=global_bank)
