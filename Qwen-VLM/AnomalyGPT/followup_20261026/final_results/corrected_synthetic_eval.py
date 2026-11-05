#!/usr/bin/env python3
"""Recompute synthetic-cache metrics for the final four-layer averaged map.

Only the four linear image decoders are instantiated. ImageBind, Qwen, Prompt
Learner and LoRA are unnecessary because their inputs are already cached and
the pixel map precedes the language branch. Default arithmetic matches the
historical evaluator: BF16 decoder, normalize, scale 100, bilinear 224 resize
with align_corners=True, per-layer two-class softmax, then average the maps.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decoder_class(source=None):
    if source is None:
        here = Path(__file__).resolve().parent
        candidates = [here / "model/AnomalyGPT_models.py", Path.cwd() / "model/AnomalyGPT_models.py",
                      here / "server_snapshot/code/model/AnomalyGPT_models.py"]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError("Pass --model-source /path/to/code/model/AnomalyGPT_models.py")
    source = Path(source).resolve()
    # Loading the file directly avoids model/__init__.py, which imports the LLM.
    spec = importlib.util.spec_from_file_location("anomalygpt_decoder_only", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LinearLayer, source


def load_decoder(checkpoint, cls, device, dtype):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    weights = {}
    for key, value in state.items():
        for prefix in ("image_decoder.", "module.image_decoder.", "_orig_mod.image_decoder."):
            if key.startswith(prefix):
                weights[key[len(prefix):]] = value
                break
    decoder = cls(1280, 1024, 4)
    if not weights:
        raise ValueError(f"No image_decoder weights in {checkpoint}")
    decoder.load_state_dict(weights, strict=True)
    return decoder.to(device=device, dtype=dtype).eval()


def assistant_text(record):
    text = record.get("text", "")
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "\n".join(str(turn.get("value", turn.get("content", ""))) for turn in text
                         if isinstance(turn, dict) and turn.get("from", turn.get("role")) in {"gpt", "assistant"})
    return ""


def text_label(text):
    text = text.strip().lower()
    if re.match(r"^yes\b", text):
        return 1
    if re.match(r"^no\b", text):
        return 0
    return None


def image_label(record, answer_label, mask_positive):
    """Keep image semantics independent of the pixel-supervision threshold."""
    for key in ("label", "label_int", "image_label", "is_anomalous"):
        if key not in record or record[key] is None:
            continue
        value = record[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Explicit {key} must be a scalar")
            value = value.item()
        if isinstance(value, str):
            names = {"0": 0, "normal": 0, "false": 0, "no": 0,
                     "1": 1, "anomaly": 1, "abnormal": 1, "true": 1, "yes": 1}
            normalized = value.strip().lower()
            if normalized not in names:
                raise ValueError(f"Invalid explicit image label {key}={value!r}")
            return names[normalized], f"explicit:{key}"
        if value in (0, 1):
            return int(value), f"explicit:{key}"
        raise ValueError(f"Invalid explicit image label {key}={value!r}")
    if answer_label is not None:
        return answer_label, "conversation_yes_no"
    return mask_positive, "threshold_mask_fallback"


def auc(y, scores):
    return float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else None


def metrics(scores, masks, image_y):
    pixel_y = masks.reshape(-1).astype(np.uint8)
    pixel_scores = scores.reshape(-1)
    image_y = np.asarray(image_y, dtype=np.uint8)
    if image_y.shape != (len(scores),) or not np.isin(image_y, (0, 1)).all():
        raise ValueError("Image labels must be one binary value per variant")
    image_scores = scores.reshape(len(scores), -1).max(axis=1)
    return {
        "num_variants": len(scores), "normal_variants": int((image_y == 0).sum()),
        "abnormal_variants": int(image_y.sum()), "positive_pixels": int(pixel_y.sum()),
        "total_pixels": int(pixel_y.size), "pixel_auroc": auc(pixel_y, pixel_scores),
        "pixel_ap": float(average_precision_score(pixel_y, pixel_scores)) if pixel_y.sum() else None,
        "image_auroc": auc(image_y, image_scores),
        "image_ap": float(average_precision_score(image_y, image_scores)) if image_y.sum() else None,
    }


def macro_metrics(per_class):
    result = {"num_classes": len(per_class)}
    for key in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
        values = [entry[key] for entry in per_class.values() if entry[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_defined_classes"] = len(values)
    return result


@torch.inference_mode()
def evaluate(args):
    started = time.time()
    checkpoint = args.checkpoint or args.initial_checkpoint
    if checkpoint is None:
        raise ValueError("Pass --checkpoint or --initial-checkpoint; no random initialization is inferred")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly for a CPU diagnostic")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    cls, source = decoder_class(args.model_source)
    decoder = load_decoder(checkpoint, cls, device, dtype)
    files = sorted(Path(args.cache).glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt cache variants in {args.cache}")
    rows, score_batches, mask_batches, classes = [], [], [], []
    for offset in range(0, len(files), args.batch_size):
        paths = files[offset:offset + args.batch_size]
        records = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        patches = [torch.cat([record["patch_features"][layer] for record in records], dim=1).to(device=device, dtype=dtype)
                   for layer in range(4)]
        features = torch.stack([record["text_features"] for record in records]).to(device=device, dtype=dtype)
        if features.shape[1:] != (2, 1024):
            raise ValueError(f"Expected [B,2,1024] text features, got {features.shape}")
        outputs = decoder(patches)
        maps = []
        for layer in outputs:
            z = F.normalize(layer, dim=-1)
            logits = 100.0 * (z @ features.transpose(-2, -1))
            batch, length, channels = logits.shape
            side = math.isqrt(length)
            if side * side != length or channels != 2:
                raise ValueError(f"Non-square patch grid or invalid classes: {logits.shape}")
            logits = F.interpolate(logits.permute(0, 2, 1).reshape(batch, 2, side, side), size=224,
                                   mode="bilinear", align_corners=True)
            maps.append(F.softmax(logits, dim=1)[:, 1])
        scores = torch.stack(maps, dim=0).mean(dim=0).float().cpu().numpy()
        gt = torch.stack([record["mask"] for record in records]).float()
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if tuple(gt.shape) != tuple(scores.shape):
            raise ValueError(f"Mask/map dimensions differ: {gt.shape} vs {scores.shape}")
        masks = (gt.numpy() > args.mask_threshold).astype(np.uint8)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite map scores")
        score_batches.append(scores)
        mask_batches.append(masks)
        for i, (path, record) in enumerate(zip(paths, records)):
            category = str(record.get("class_name", "unknown"))
            classes.append(category)
            variant = record.get("variant")
            if variant is None and path.stem.rsplit("_", 1)[-1] in {"0", "1"}:
                variant = int(path.stem.rsplit("_", 1)[-1])
            answer = assistant_text(record)
            label = text_label(answer)
            positive = int(masks[i].any())
            image_gt, label_source = image_label(record, label, positive)
            rows.append({
                "cache_file": path.name, "class_name": category, "source_path": record.get("source_path"),
                "source_index": record.get("source_index"), "variant": variant,
                "mask_positive": positive, "mask_positive_pixels": int(masks[i].sum()),
                "assistant_text": answer, "text_label": label,
                "image_label": image_gt, "image_label_source": label_source,
                "pixel_foreground_image_label_match": positive == image_gt,
                "threshold_mask_text_match": label == positive if label is not None else None,
                "explicit_label_text_match": image_gt == label if label_source.startswith("explicit:") and label is not None else None,
                "image_score_max": float(scores[i].max()), "map_score_mean": float(scores[i].mean()),
            })
    scores, masks = np.concatenate(score_batches), np.concatenate(mask_batches)
    classes = np.asarray(classes)
    image_y = np.asarray([row["image_label"] for row in rows], dtype=np.uint8)
    per_class = {category: metrics(scores[classes == category], masks[classes == category], image_y[classes == category])
                 for category in sorted(set(classes))}
    disagreement = [row["cache_file"] for row in rows if row["pixel_foreground_image_label_match"] is False]
    text_disagreement = [row["cache_file"] for row in rows if row["threshold_mask_text_match"] is False]
    unknown_text = [row["cache_file"] for row in rows if row["text_label"] is None]
    histogram = Counter(f"variant={row['variant']},mask={row['mask_positive']},text={row['text_label']},image={row['image_label']}" for row in rows)
    out = {
        "evaluation_version": "four_layer_average_image_label_v2", "cache": str(Path(args.cache).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_role": "trained" if args.checkpoint else "initialization",
        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "decoder_source": str(source), "decoder_source_sha256": sha256(source),
        "device": str(device), "dtype": "bfloat16", "batch_size": args.batch_size,
        "map_aggregation": "mean of four per-layer abnormal-class softmax maps; one map per variant",
        "map_size": [224, 224], "interpolation": "bilinear_align_corners_true", "mask_threshold": args.mask_threshold,
        "image_ground_truth": "explicit record label first, otherwise conversation Yes/No, otherwise threshold-mask fallback; independent of pixel threshold when a semantic label is available",
        "image_score": "maximum of final averaged anomaly map", "pooled": metrics(scores, masks, image_y),
        "per_class": per_class, "macro": macro_metrics(per_class),
        "label_audit": {
            "joint_counts": dict(sorted(histogram.items())),
            "image_label_source_counts": dict(Counter(row["image_label_source"] for row in rows)),
            "pixel_foreground_image_label_disagreement_count": len(disagreement),
            "pixel_foreground_image_label_disagreement_files": disagreement,
            "threshold_mask_text_disagreement_count": len(text_disagreement),
            "threshold_mask_text_disagreement_files": text_disagreement,
            "explicit_label_text_disagreement_files": [row["cache_file"] for row in rows if row["explicit_label_text_match"] is False],
            "unparsed_text_files": unknown_text,
            "variant1_empty_threshold_mask_files": [row["cache_file"] for row in rows if row["variant"] == 1 and row["mask_positive"] == 0],
            "variant1_normal_conversation_files": [row["cache_file"] for row in rows if row["variant"] == 1 and row["text_label"] == 0],
            "interpretation": "Thresholded pixel foreground can be empty for a semantically positive synthetic image; disagreement alone does not establish a mislabeled or failed synthesis. A variant1 normal conversation records the dataset's no-centers branch.",
        },
        "records": rows, "seconds": time.time() - started,
        "limitations": ["Synthetic source variants are not real anomaly evaluation.",
                        "No language model is run; image scores and pixel metrics are decoder outputs.",
                        "Checkpoint initialization exposure requires a separate provenance audit."],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: out[key] for key in ("evaluation_version", "pooled", "macro", "label_audit", "seconds")}, indent=2), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mask-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
