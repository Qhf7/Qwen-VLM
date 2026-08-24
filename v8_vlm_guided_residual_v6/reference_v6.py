"""Category/view aligned normal references for DINOv2 and CLIP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from common import greedy_coreset, knn_distance_torch, load_many, per_layer_features
from object_prior import foreground_grid


def fused_dino_features(maps: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.cat([F.normalize(x.float(), dim=-1) for x in maps], dim=-1), dim=-1)


def aligned_distance(current: torch.Tensor, normal: torch.Tensor, scale: float) -> torch.Tensor:
    current = F.normalize(current.float(), dim=-1)
    normal = F.normalize(normal.float(), dim=-1)
    return ((1.0 - (current * normal).sum(-1)) / max(float(scale), 1e-5)).clamp(0, 1)


def aligned_dino_reference(current_maps, normal_maps, scale: float = 0.12):
    return aligned_distance(fused_dino_features(current_maps), fused_dino_features(normal_maps), scale)


def aligned_clip_reference(current_patch, normal_patch, scale: float = 0.18):
    return aligned_distance(current_patch, normal_patch, scale)


def _reservoir(store, key, values, capacity, rng):
    values = np.asarray(values, np.float16)
    old = store.get(key)
    merged = values if old is None else np.concatenate([old, values], axis=0)
    if len(merged) > capacity:
        merged = merged[rng.choice(len(merged), capacity, replace=False)]
    store[key] = merged.copy()


@dataclass
class PatchReferenceBank:
    banks: dict[tuple[str, int], np.ndarray]
    global_bank: np.ndarray
    bounds: dict[tuple[str, int], tuple[float, float]]
    global_bounds: tuple[float, float]
    knn: int = 3

    def raw_score(self, features: torch.Tensor, categories: list[str], device: torch.device) -> torch.Tensor:
        output = []
        for index, feature in enumerate(features):
            bank = self.banks.get((categories[index], index % 5), self.global_bank)
            bank_t = torch.from_numpy(bank).to(device=device, dtype=torch.float32)
            distance = knn_distance_torch(feature.reshape(-1, feature.shape[-1]).float(), bank_t, self.knn)
            output.append(distance.reshape(feature.shape[:2]))
        return torch.stack(output)

    def score(self, features: torch.Tensor, categories: list[str], device: torch.device) -> torch.Tensor:
        raw = self.raw_score(features, categories, device)
        output = []
        for index, value in enumerate(raw):
            threshold, scale = self.bounds.get((categories[index], index % 5), self.global_bounds)
            excess = ((value - threshold) / max(scale, 1e-5)).clamp_min(0.0)
            output.append(excess / (1.0 + excess))
        return torch.stack(output)


@dataclass
class ReferenceBanks:
    dino: PatchReferenceBank
    clip: PatchReferenceBank


def _finalise(pools, global_pool, budget, global_budget, seed):
    banks = {}
    for key, value in pools.items():
        x = value.astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        banks[key] = np.ascontiguousarray(greedy_coreset(x, min(budget, len(x)), seed=seed))
    global_x = global_pool["global"].astype(np.float32)
    global_x /= np.maximum(np.linalg.norm(global_x, axis=1, keepdims=True), 1e-6)
    global_bank = np.ascontiguousarray(
        greedy_coreset(global_x, min(global_budget, len(global_x)), seed=seed)
    )
    return PatchReferenceBank(banks, global_bank, {}, (0.0, 1.0))


def _split_bank_calibration(samples, per_category=2):
    by_category = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    bank, calibration = [], []
    for category in sorted(by_category):
        values = sorted(by_category[category], key=lambda item: item.name)
        count = min(per_category, max(1, len(values) // 5)) if len(values) > 1 else 0
        bank.extend(values[:-count] if count else values)
        calibration.extend(values[-count:] if count else values)
    return bank, calibration


def _fit_bounds(values):
    x = np.asarray(values, np.float32)
    threshold = float(np.quantile(x, 0.90))
    high = float(np.quantile(x, 0.995))
    return threshold, max(high - threshold, 1e-4)


def _attach_bounds(bank, values_by_key, global_values):
    bank.bounds = {key: _fit_bounds(values) for key, values in values_by_key.items() if len(values)}
    bank.global_bounds = _fit_bounds(np.concatenate(global_values))


def build_reference_banks(
    model, dino, clip_preprocess, samples, dino_size, clip_size, layers, device,
    budget=768, global_budget=6144, seed=20260826, progress=None,
):
    """Build banks from official Train normals only.

    Category/view banks are used for seen classes. The global bank is the
    strict fallback for B-board categories that never appear in Train.
    """
    rng = np.random.default_rng(seed)
    bank_samples, calibration_samples = _split_bank_calibration(samples)
    dino_pool, clip_pool = {}, {}
    dino_global, clip_global = {}, {}
    key_cap, global_cap = budget * 3, global_budget * 3
    with torch.inference_mode():
        for index, sample in enumerate(bank_samples):
            images = load_many(sample.paths)
            dino_maps, _ = per_layer_features(dino, images, dino_size, device, tuple(layers))
            dino_feature = fused_dino_features(dino_maps)
            _, clip_feature = model.encode_clip(clip_preprocess(images, clip_size, device))
            for view in range(5):
                for feature, pool, global_pool, fg_stride, bg_stride in (
                    (dino_feature[view], dino_pool, dino_global, 2, 32),
                    (clip_feature[view], clip_pool, clip_global, 2, 32),
                ):
                    array = feature.cpu().numpy()
                    fg = foreground_grid(images[view], array.shape[0]).reshape(-1) > 0.25
                    flat = array.reshape(-1, array.shape[-1])
                    selected = np.concatenate([flat[fg][::fg_stride], flat[~fg][::bg_stride]], axis=0)
                    if not len(selected):
                        selected = flat[::4]
                    _reservoir(pool, (sample.category, view), selected, key_cap, rng)
                    _reservoir(global_pool, "global", selected[::12], global_cap, rng)
            if progress is not None and (index % 10 == 0 or index + 1 == len(bank_samples)):
                progress(index + 1, len(bank_samples) + len(calibration_samples), f"bank:{sample.category}")
    result = ReferenceBanks(
        dino=_finalise(dino_pool, dino_global, budget, global_budget, seed),
        clip=_finalise(clip_pool, clip_global, budget, global_budget, seed + 1),
    )

    dino_values, clip_values = {}, {}
    dino_global_values, clip_global_values = [], []
    with torch.inference_mode():
        for index, sample in enumerate(calibration_samples):
            images = load_many(sample.paths)
            dino_maps, _ = per_layer_features(dino, images, dino_size, device, tuple(layers))
            dino_feature = fused_dino_features(dino_maps)
            _, clip_feature = model.encode_clip(clip_preprocess(images, clip_size, device))
            dino_raw = result.dino.raw_score(dino_feature, [sample.category] * 5, device).cpu().numpy()
            clip_raw = result.clip.raw_score(clip_feature, [sample.category] * 5, device).cpu().numpy()
            for view, (dino_map, clip_map) in enumerate(zip(dino_raw, clip_raw)):
                fg_dino = foreground_grid(images[view], dino_map.shape[0]) > 0.25
                fg_clip = foreground_grid(images[view], clip_map.shape[0]) > 0.25
                dino_selected = dino_map[fg_dino] if fg_dino.any() else dino_map.reshape(-1)
                clip_selected = clip_map[fg_clip] if fg_clip.any() else clip_map.reshape(-1)
                dino_values.setdefault((sample.category, view), []).append(dino_selected[::2])
                clip_values.setdefault((sample.category, view), []).append(clip_selected[::2])
                dino_global_values.append(dino_selected[::8])
                clip_global_values.append(clip_selected[::8])
            if progress is not None and (index % 5 == 0 or index + 1 == len(calibration_samples)):
                progress(
                    len(bank_samples) + index + 1,
                    len(bank_samples) + len(calibration_samples),
                    f"calibration:{sample.category}",
                )
    _attach_bounds(
        result.dino,
        {key: np.concatenate(values) for key, values in dino_values.items()},
        dino_global_values,
    )
    _attach_bounds(
        result.clip,
        {key: np.concatenate(values) for key, values in clip_values.items()},
        clip_global_values,
    )
    return result
