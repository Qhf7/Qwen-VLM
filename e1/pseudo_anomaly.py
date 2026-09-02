"""E1 的可复现 CutPaste 与 NSA-like 伪异常，仅用于 sample-level OK 图。"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


def sample_seed(image_path: str, epoch: int, global_seed: int = 20260901) -> int:
    payload = f"{global_seed}|{epoch}|{image_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rectangle(rng: np.random.Generator, height: int, width: int) -> tuple[int, int, int, int]:
    area_ratio = float(rng.uniform(0.005, 0.05))
    aspect = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    patch_area = height * width * area_ratio
    patch_h = int(np.clip(np.sqrt(patch_area / aspect), 4, height // 2))
    patch_w = int(np.clip(np.sqrt(patch_area * aspect), 4, width // 2))
    top = int(rng.integers(0, height - patch_h + 1))
    left = int(rng.integers(0, width - patch_w + 1))
    return top, left, patch_h, patch_w


def cutpaste(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    source_y, source_x, patch_h, patch_w = _rectangle(rng, height, width)
    target_y = int(rng.integers(0, height - patch_h + 1))
    target_x = int(rng.integers(0, width - patch_w + 1))
    patch = image[source_y:source_y + patch_h, source_x:source_x + patch_w].copy()
    gain = float(rng.uniform(0.8, 1.2))
    bias = float(rng.uniform(-20, 20))
    patch = np.clip(patch.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    output = image.copy()
    output[target_y:target_y + patch_h, target_x:target_x + patch_w] = patch
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[target_y:target_y + patch_h, target_x:target_x + patch_w] = 255
    return output, mask


def nsa_like(
    image: np.ndarray,
    donor: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """用另一张正常图的纹理和低频随机形状生成 NSA-like 局部异常。"""
    height, width = image.shape[:2]
    if donor.shape[:2] != (height, width):
        donor = cv2.resize(donor, (width, height), interpolation=cv2.INTER_AREA)
    top, left, patch_h, patch_w = _rectangle(rng, height, width)
    low_h = max(2, patch_h // 16)
    low_w = max(2, patch_w // 16)
    noise = rng.random((low_h, low_w), dtype=np.float32)
    smooth = cv2.resize(noise, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(1.0, min(patch_h, patch_w) / 16))
    local_mask = smooth >= np.quantile(smooth, 0.55)
    if local_mask.sum() < 16:
        local_mask[patch_h // 4:3 * patch_h // 4, patch_w // 4:3 * patch_w // 4] = True
    alpha = float(rng.uniform(0.5, 0.9))
    target_patch = image[top:top + patch_h, left:left + patch_w].astype(np.float32)
    donor_y = int(rng.integers(0, height - patch_h + 1))
    donor_x = int(rng.integers(0, width - patch_w + 1))
    donor_patch = donor[donor_y:donor_y + patch_h, donor_x:donor_x + patch_w].astype(np.float32)
    blended = np.clip((1 - alpha) * target_patch + alpha * donor_patch, 0, 255).astype(np.uint8)
    output = image.copy()
    region = output[top:top + patch_h, left:left + patch_w]
    region[local_mask] = blended[local_mask]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:top + patch_h, left:left + patch_w][local_mask] = 255
    return output, mask


def make_pseudo_anomaly(
    image: np.ndarray,
    donor: np.ndarray,
    method: str,
    image_path: str,
    epoch: int,
    global_seed: int = 20260901,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(sample_seed(image_path, epoch, global_seed))
    if method == "cutpaste":
        return cutpaste(image, rng)
    if method == "nsa":
        return nsa_like(image, donor, rng)
    raise ValueError(method)
