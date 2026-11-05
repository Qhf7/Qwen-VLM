"""Shared VisA geometry, binary masks, and standard anomaly metrics.

Images and masks use the same direct square resize, without a center crop.
Mask values are binarized before resizing so VisA palette indices 1..8 all
remain foreground. This module deliberately does not import ImageBind.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


PIPELINE_VERSION = "visa-square-binary-sklearn-v1"
IMAGEBIND_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGEBIND_STD = (0.26862954, 0.26130258, 0.27577711)


def _size(size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    return size


def _label(label):
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"normal", "good", "0"}:
            return 0
        if normalized in {"anomaly", "abnormal", "1"}:
            return 1
        raise ValueError(f"Unknown VisA label: {label!r}")
    if label is None or label not in (0, 1):
        raise ValueError(f"VisA label must explicitly be normal/0 or anomaly/1: {label!r}")
    return int(label)


def load_image_pil(path, size=224):
    """Return an RGB image directly resized to ``(size, size)``."""
    size = _size(size)
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def load_image_tensor(path, device="cuda", size=224):
    """Return an ImageBind-normalized float32 tensor of shape [1, 3, H, W]."""
    pixels = np.array(load_image_pil(path, size=size), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    mean = tensor.new_tensor(IMAGEBIND_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGEBIND_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def mask_path_for_image(path, label):
    """Resolve an anomaly mask, failing loudly if its annotation is absent.

    Explicitly normal images have no mask. Anomalies map the VisA directory
    ``Data/Images/Anomaly`` to ``Data/Masks/Anomaly`` and use a .png mask.
    An uppercase/mixed-case PNG extension is accepted as a fallback.
    """
    if _label(label) == 0:
        return None
    image_path = Path(path)
    parts = list(image_path.parts)
    for index in range(len(parts) - 2):
        if parts[index:index + 3] == ["Data", "Images", "Anomaly"]:
            parts[index + 1] = "Masks"
            break
    else:
        raise ValueError(f"Anomaly image is outside VisA Data/Images/Anomaly: {image_path}")
    expected = Path(*parts).with_suffix(".png")
    if expected.is_file():
        return str(expected)
    candidates = []
    if expected.parent.is_dir():
        candidates = sorted(
            candidate for candidate in expected.parent.iterdir()
            if candidate.is_file() and candidate.stem == expected.stem
            and candidate.suffix.lower() == ".png"
        )
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous mask extensions for {image_path}: {candidates}")
    raise FileNotFoundError(f"Missing anomaly mask for {image_path}; expected {expected}")


def load_mask_tensor(path_or_None, device="cuda", size=224):
    """Return binary float32 [1, H, W]; ``None`` explicitly means normal."""
    size = _size(size)
    if path_or_None is None:
        return torch.zeros((1, size, size), dtype=torch.float32, device=device)
    with Image.open(path_or_None) as mask:
        raw = np.asarray(mask)
        binary = raw > 0
        if binary.ndim == 3:
            binary = binary.any(axis=2)
        if binary.ndim != 2:
            raise ValueError(f"Expected a 2D mask: {path_or_None}, shape={raw.shape}")
        resized = Image.fromarray(binary.astype(np.uint8)).resize(
            (size, size), Image.Resampling.NEAREST
        )
        pixels = np.array(resized, dtype=np.float32)
    return torch.from_numpy(pixels).unsqueeze(0).to(device)


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _metric_arrays(y, scores):
    y = _array(y).reshape(-1)
    scores = _array(scores).reshape(-1)
    if not len(y) or y.shape != scores.shape:
        raise ValueError("Labels and scores must have equal, nonzero lengths")
    if not np.isfinite(y).all() or not np.isfinite(scores).all():
        raise ValueError("Labels and scores must be finite")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Anomaly metric labels must be binary 0/1")
    return y.astype(np.uint8, copy=False), scores


def auroc(y, s):
    """Standard AUROC, with average ranks for ties and explicit validation."""
    y, s = _metric_arrays(y, s)
    if np.unique(y).size != 2:
        raise ValueError("AUROC requires both positive and negative labels")
    return float(roc_auc_score(y, s))


def average_precision(y, s):
    """Standard threshold-grouped average precision, including tied scores."""
    y, s = _metric_arrays(y, s)
    return float(average_precision_score(y, s))


def pixel_metrics(items):
    """Summarize iterable ``(class_name, label, gt, anomaly_map)`` predictions.

    Pixel scores are pooled within each category. Macro metrics average
    categories containing foreground and background pixels; weighted AUROC
    weights those category AUROCs by foreground pixel count. Image AUROC
    uses the 99th percentile of each map. Undefined AUROCs are ``None``,
    with counts and per-category results retained for auditing.
    """
    categories = {}
    image_labels, image_scores = [], []
    for class_name, label, gt, anomaly_map in items:
        label = _label(label)
        gt, anomaly_map = _array(gt), _array(anomaly_map)
        # Accept singleton channel/batch dimensions, but reject spatial mismatch.
        gt, anomaly_map = np.squeeze(gt), np.squeeze(anomaly_map)
        if gt.shape != anomaly_map.shape or gt.ndim != 2:
            raise ValueError(
                f"Expected aligned 2D ground truth/map for {class_name}: "
                f"{gt.shape} versus {anomaly_map.shape}"
            )
        y, scores = _metric_arrays(gt, anomaly_map)
        if label == 0 and y.any():
            raise ValueError(f"Normal image has foreground mask pixels: {class_name}")
        category = categories.setdefault(str(class_name), {"y": [], "s": [], "n_images": 0})
        category["y"].append(y)
        category["s"].append(scores)
        category["n_images"] += 1
        image_labels.append(label)
        image_scores.append(float(np.percentile(scores, 99)))
    if not image_labels:
        raise ValueError("No predictions supplied to pixel_metrics")

    aucs, aps, weights = [], [], []
    total_pixels = foreground_pixels = 0
    per_category = {}
    for class_name, category in sorted(categories.items()):
        y = np.concatenate(category["y"])
        scores = np.concatenate(category["s"])
        foreground = int(y.sum())
        total_pixels += len(y)
        foreground_pixels += foreground
        valid = 0 < foreground < len(y)
        category_auc = auroc(y, scores) if valid else None
        category_ap = average_precision(y, scores) if foreground else None
        per_category[class_name] = {
            "pixel_auroc": category_auc, "pixel_ap": category_ap,
            "foreground_pixels": foreground, "n_pixels": len(y),
            "n_images": category["n_images"],
        }
        if valid:
            aucs.append(category_auc)
            aps.append(category_ap)
            weights.append(foreground)
    image_auc = auroc(image_labels, image_scores) if len(set(image_labels)) == 2 else None
    return {
        "pipeline_version": PIPELINE_VERSION,
        "pixel_auroc_macro": float(np.mean(aucs)) if aucs else None,
        "pixel_ap_macro": float(np.mean(aps)) if aps else None,
        "pixel_auroc_weighted": float(np.average(aucs, weights=weights)) if aucs else None,
        "image_auroc_from_map99": image_auc,
        "foreground_ratio": foreground_pixels / total_pixels,
        "n_categories": len(aucs),
        "n_categories_total": len(categories),
        "n_images": len(image_labels),
        "n_anomaly_images": sum(image_labels),
        "counts": {"pixels": total_pixels, "foreground_pixels": foreground_pixels,
                   "background_pixels": total_pixels - foreground_pixels,
                   "normal_images": len(image_labels) - sum(image_labels),
                   "anomaly_images": sum(image_labels),
                   "undefined_pixel_auc_categories": len(categories) - len(aucs)},
        "per_category": per_category,
    }
