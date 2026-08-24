"""Bounded DINOv2 normal memory for seen and held-out categories."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from branches import MemoryBank
from common import greedy_coreset, load_many


def descriptor(maps: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.cat([F.normalize(x, dim=-1) for x in maps], dim=-1), dim=-1)


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


def build_memory(
    feature_fn,
    samples,
    budget: int = 1536,
    global_budget: int = 8192,
    seed: int = 20260822,
):
    """Build fixed-capacity reservoirs before coreset selection.

    This avoids the unbounded 3072-D patch accumulation that previously
    exceeded the 60GB server RAM during v3 inference.
    """
    rng = np.random.default_rng(seed)
    pools = {}
    global_pool = {}
    key_cap = max(budget * 3, budget + 512)
    global_cap = max(global_budget * 3, global_budget + 2048)
    for index, sample in enumerate(samples):
        maps, _ = feature_fn(load_many(sample.paths))
        feature = descriptor(maps).cpu().numpy()
        for view, array in enumerate(feature):
            values = array.reshape(-1, array.shape[-1])[::8]
            _reservoir(pools, (sample.category, view), values, key_cap, rng)
            _reservoir(global_pool, "global", values[::16], global_cap, rng)
        if index % 50 == 0:
            print(f"v4 memory {index}/{len(samples)}", flush=True)
    bank = MemoryBank(budget_per_key=budget, global_budget=global_budget, knn=3)
    for key, values in pools.items():
        x = values.astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        bank.set(key, greedy_coreset(x, min(budget, len(x)), seed=seed))
    values = global_pool["global"].astype(np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
    bank.set_global(greedy_coreset(values, min(global_budget, len(values)), seed=seed))
    return bank


def memory_maps(bank, maps: list[torch.Tensor], categories: list[str], device: torch.device):
    feature = descriptor(maps).cpu().numpy()
    outputs = []
    for view, (array, category) in enumerate(zip(feature, categories)):
        side = array.shape[0]
        score = bank.score((category, view % 5), array.reshape(-1, array.shape[-1]), device)
        outputs.append(score.reshape(side, side))
    return np.stack(outputs).astype(np.float32)
