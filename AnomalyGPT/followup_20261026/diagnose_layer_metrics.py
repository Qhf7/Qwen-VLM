"""Reproduce historical layer-concatenated metrics beside final-map metrics.

Only the four small linear decoders run on the GPU. This does not instantiate
ImageBind, a language model, or PromptLearner and does not modify old results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


HISTORICAL_FROZEN42_AUROC = 0.8087700682098014


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranking(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Score/GT shapes differ or scores are not finite")
    return {"auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
            "ap": float(average_precision_score(labels, scores)) if labels.sum() else None}


def distribution(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [0, .01, .1, .5, .9, .99, 1])
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            **{f"q{p}": float(value) for p, value in zip((0, 1, 10, 50, 90, 99, 100), quantiles)}}


def evaluate_arrays(layers, final, masks, classes):
    # [4,N,H,W] is equivalent to the old batch/layer concatenation order,
    # because the matching GT block travels with every layer's score block.
    repeated_gt = np.broadcast_to(masks, layers.shape)
    old = ranking(repeated_gt, layers)
    # BF16 mean is computed on-device by the caller, exactly as inference.
    per_class = {}
    for category in sorted(set(classes.tolist())):
        use = classes == category
        per_class[category] = {"num_variants": int(use.sum()),
                               "final_map": ranking(masks[use], final[use]),
                               "concatenated_layers": ranking(np.broadcast_to(masks[use], layers[:, use].shape), layers[:, use])}
    macro = {}
    for family in ("final_map", "concatenated_layers"):
        macro[family] = {}
        for metric in ("auroc", "ap"):
            values = [row[family][metric] for row in per_class.values() if row[family][metric] is not None]
            macro[family][metric] = float(np.mean(values)) if values else None
    foreground = masks.astype(bool)
    flattened_masks = masks.reshape(-1).astype(bool)
    layer_results, positives, sorted_negatives = [], [], []
    for index, layer in enumerate(layers):
        positives.append(layer.reshape(-1)[flattened_masks])
        sorted_negatives.append(np.sort(layer.reshape(-1)[~flattened_masks]))
        layer_results.append({"layer": index, "pixel": ranking(masks, layer),
                              "foreground_scores": distribution(layer[foreground]),
                              "background_scores": distribution(layer[~foreground])})
    pair_auroc = []
    for positive in positives:
        row = []
        for negative in sorted_negatives:
            if not len(positive) or not len(negative):
                row.append(None)
            else:
                left = np.searchsorted(negative, positive, side="left")
                right = np.searchsorted(negative, positive, side="right")
                row.append(float(np.mean((left + right) / (2.0 * len(negative)))))
        pair_auroc.append(row)
    matrix = np.asarray(pair_auroc, dtype=float)
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(4, dtype=bool)]
    return {"historical_layer_concat": old,
            "final_four_layer_mean": ranking(masks, final), "macro": macro,
            "per_class": per_class, "layers": layer_results,
            "pairwise_layer_auroc": pair_auroc,
            "pairwise_definition": "Row i: foreground pixels scored by layer i; column j: background pixels scored by layer j. Ties receive half credit.",
            "same_layer_auroc_mean": float(np.nanmean(diagonal)),
            "cross_layer_auroc_mean": float(np.nanmean(off_diagonal)),
            "pairwise_matrix_mean": float(np.nanmean(matrix)),
            "concat_vs_pairwise_mean_absolute_error": abs(old["auroc"] - float(np.nanmean(matrix))) if old["auroc"] is not None else None,
            "historical_frozen42_auroc_reference": HISTORICAL_FROZEN42_AUROC,
            "difference_from_historical_frozen42_reference": old["auroc"] - HISTORICAL_FROZEN42_AUROC if old["auroc"] is not None else None}


def assistant_label(record):
    import re
    text = record.get("text", "")
    if isinstance(text, list):
        text = "\n".join(str(turn.get("value", "")) for turn in text
                          if isinstance(turn, dict) and turn.get("from") == "gpt")
    match = re.match(r"^\s*(yes|no)\b", str(text), re.I)
    return None if match is None else int(match.group(1).lower() == "yes")


def make_markdown(result):
    lines = ["# Synthetic cache：历史层拼接指标与最终热图指标诊断", "",
             f"Checkpoint SHA256：`{result['checkpoint_sha256']}`。",
             f"Cache：`{result['cache']}`；{result['num_variants']} 个 variant；BF16；batch={result['batch_size']}。", "",
             "| 归一化 | 历史层拼接 AUROC | 最终四层均值 AUROC | 历史层拼接 AP | 最终四层均值 AP | 最终图类别宏 AUROC |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in result["normalizations"].items():
        old, final = value["historical_layer_concat"], value["final_four_layer_mean"]
        lines.append(f"| {name} | {old['auroc']:.9f} | {final['auroc']:.9f} | {old['ap']:.9f} | {final['ap']:.9f} | {value['macro']['final_map']['auroc']:.9f} |")
    lines += ["", "历史实现对每层异常图分别追加一个 score block，并同步追加相同来源的 GT block，因此这些块没有样本错序。它把四层的所有像素视为同一分数空间。最终图实现先对同图四层概率求均值，再只评一次该图。", "",
              "四层混合 AUROC 等于 4×4 层配对矩阵的平均值：4 个同层配对与 12 个跨层配对。跨层概率幅值偏移会影响历史混合排名；它与最终集成图的排名是两个不同测量。数值差异不能当作新训练收益。", ""]
    for name, value in result["normalizations"].items():
        lines += [f"## {name}：层配对 AUROC", "", "行是前景像素的评分层，列是背景像素的评分层。", "",
                  "| | 背景 L0 | 背景 L1 | 背景 L2 | 背景 L3 |", "|---|---:|---:|---:|---:|"]
        for index, row in enumerate(value["pairwise_layer_auroc"]):
            lines.append(f"| 前景 L{index} | " + " | ".join(f"{score:.6f}" for score in row) + " |")
        lines += ["", f"同层平均：{value['same_layer_auroc_mean']:.9f}；跨层平均：{value['cross_layer_auroc_mean']:.9f}；矩阵均值：{value['pairwise_matrix_mean']:.9f}。",
                  f"与旧 frozen42 AUROC 0.8087700682098014 的差：{value['difference_from_historical_frozen42_reference']:.12g}。", "",
                  "| 层 | Pixel AUROC | Pixel AP | 前景均值 | 背景均值 | 前景中位数 | 背景中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for layer in value["layers"]:
            fg, bg = layer["foreground_scores"], layer["background_scores"]
            lines.append(f"| {layer['layer']} | {layer['pixel']['auroc']:.6f} | {layer['pixel']['ap']:.6f} | {fg.get('mean', 0):.6f} | {bg.get('mean', 0):.6f} | {fg.get('q50', 0):.6f} | {bg.get('q50', 0):.6f} |")
        lines.append("")
    lines += ["## 掩码核对", "", f"阈值：>{result['mask_threshold']}；前景像素：{result['mask_stats']['positive_pixels']} / {result['mask_stats']['total_pixels']}。",
              f"normal variant 非空掩码：{result['mask_stats']['variant0_nonempty_files']}。",
              f"synthetic variant 空掩码：{result['mask_stats']['variant1_empty_files']}。",
              f"问答标签与阈值化掩码存在性不一致：{result['mask_stats']['text_mask_mismatch_files']}。", "",
              "合成掩码由 224² 图像上的 Poisson/CutPaste 前后差异经 logistic-intensity 构造，为软值；训练与旧评测用 >0.3 二值化。真实 VisA 掩码是原始类别索引，按 >0 二值化后 nearest 缩放至 224²；两者标签含义和阈值不同，不能把合成指标称作真实缺陷指标。", "",
              f"两个归一化的最终图最大绝对差：{result['normalization_comparison']['final_map_max_abs_difference']:.9g}；所有单层图最大绝对差：{result['normalization_comparison']['layer_map_max_abs_difference']:.9g}。", ""]
    return "\n".join(lines)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="JSON path; an adjacent .md is also written")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mask-threshold", type=float, default=.3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for path in (args.out, args.out.with_suffix(".md")):
        if path.exists():
            raise FileExistsError(path)
    started = time.time()
    spec = importlib.util.spec_from_file_location("layer_diagnostic_decoder", args.model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.LinearLayer(1280, 1024, 4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    weights = {key[len("image_decoder."):]: value for key, value in state.items() if key.startswith("image_decoder.")}
    decoder.load_state_dict(weights, strict=True)
    del state, weights
    decoder = decoder.to(args.device, dtype=torch.bfloat16).eval()
    files = sorted(args.cache.glob("*.pt"))
    if not files:
        raise ValueError("No cache files")
    maps_by_mode = {"F.normalize": [], "manual_norm_division": []}
    means_by_mode = {name: [] for name in maps_by_mode}
    masks, classes, mask_rows, cache_hashes = [], [], [], {}
    raw_soft_count = 0
    raw_nonzero_count = 0
    for offset in range(0, len(files), args.batch_size):
        paths = files[offset:offset + args.batch_size]
        records = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        patches = [torch.cat([record["patch_features"][layer] for record in records], dim=1).to(args.device, dtype=torch.bfloat16)
                   for layer in range(4)]
        features = torch.stack([record["text_features"] for record in records]).to(args.device, dtype=torch.bfloat16)
        outputs = decoder(patches)
        for mode in maps_by_mode:
            maps = []
            for output in outputs:
                normalized = F.normalize(output, dim=-1) if mode == "F.normalize" else output / output.norm(dim=-1, keepdim=True)
                logits = 100.0 * (normalized @ features.transpose(-2, -1))
                batch, length, channels = logits.shape
                side = math.isqrt(length)
                if side * side != length or channels != 2:
                    raise ValueError("Invalid patch lattice")
                logits = F.interpolate(logits.permute(0, 2, 1).reshape(batch, 2, side, side),
                                       size=224, mode="bilinear", align_corners=True)
                maps.append(logits.softmax(dim=1)[:, 1])
            layer_maps = torch.stack(maps)
            maps_by_mode[mode].append(layer_maps.float().cpu().numpy())
            means_by_mode[mode].append(layer_maps.mean(dim=0).float().cpu().numpy())
        gt = torch.stack([record["mask"] for record in records]).float().numpy()
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if gt.shape != (len(records), 224, 224):
            raise ValueError(f"Invalid GT shape {gt.shape}")
        binary = gt > args.mask_threshold
        masks.append(binary.astype(np.uint8))
        raw_soft_count += int(((gt > 0) & (gt < 1)).sum())
        raw_nonzero_count += int((gt > 0).sum())
        for index, (path, record) in enumerate(zip(paths, records)):
            label = assistant_label(record)
            variant = record.get("variant", int(path.stem.rsplit("_", 1)[-1]))
            foreground = int(binary[index].sum())
            classes.append(record["class_name"])
            cache_hashes[path.name] = sha256(path)
            mask_rows.append({"file": path.name, "source_path": record.get("source_path"),
                              "class_name": record["class_name"], "variant": variant,
                              "raw_min": float(gt[index].min()), "raw_max": float(gt[index].max()),
                              "raw_nonzero_pixels": int((gt[index] > 0).sum()),
                              "positive_pixels": foreground, "text_label": label,
                              "text_mask_match": label == int(foreground > 0) if label is not None else None})
    masks = np.concatenate(masks)
    classes = np.asarray(classes)
    final_layers = {name: np.concatenate(batches, axis=1) for name, batches in maps_by_mode.items()}
    final_means = {name: np.concatenate(batches, axis=0) for name, batches in means_by_mode.items()}
    results = {}
    for name in final_layers:
        results[name] = evaluate_arrays(final_layers[name], final_means[name], masks, classes)
    out = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
           "cache": str(args.cache.resolve()), "cache_hashes": cache_hashes,
           "source_sha256": sha256(__file__), "model_source_sha256": sha256(args.model_source),
           "num_variants": len(files), "batch_size": args.batch_size, "device": args.device,
           "dtype": "bfloat16", "mask_threshold": args.mask_threshold,
           "normalizations": results,
           "normalization_comparison": {
               "layer_map_max_abs_difference": float(np.max(np.abs(final_layers['F.normalize'] - final_layers['manual_norm_division']))),
               "final_map_max_abs_difference": float(np.max(np.abs(final_means['F.normalize'] - final_means['manual_norm_division'])))},
           "mask_stats": {"positive_pixels": int(masks.sum()), "total_pixels": int(masks.size),
                          "raw_nonzero_pixels": raw_nonzero_count, "raw_soft_value_pixels": raw_soft_count,
                          "variant0_nonempty_files": [row["file"] for row in mask_rows if row["variant"] == 0 and row["positive_pixels"]],
                          "variant1_empty_files": [row["file"] for row in mask_rows if row["variant"] == 1 and not row["positive_pixels"]],
                          "text_mask_mismatch_files": [row["file"] for row in mask_rows if row["text_mask_match"] is False],
                          "class_variant_counts": dict(Counter(f"{row['class_name']}:{row['variant']}" for row in mask_rows)),
                          "records": mask_rows}, "seconds": time.time() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(make_markdown(out))
    print(json.dumps({name: {key: value[key] for key in ("historical_layer_concat", "final_four_layer_mean", "same_layer_auroc_mean", "cross_layer_auroc_mean", "difference_from_historical_frozen42_reference")}
                      for name, value in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
