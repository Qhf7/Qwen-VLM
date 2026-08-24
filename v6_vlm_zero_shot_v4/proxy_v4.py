#!/usr/bin/env python3
"""Leave-category-out Train-only proxy for VLM v4.

Held-out categories never enter Prompt/Adapter/Query/Gate training.  Synthetic
defects are generated deterministically only for evaluation and do not update
parameters.  Proxy numbers are for relative model selection, not leaderboard
claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v5_multiscale_v3"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(V3))

from clip_dense import clip_preprocess_tensor, load_clip
from common import list_samples, load_dinov2, load_many, per_layer_features, pick_device, set_seed, topk_mean
from metrics import PixelAccumulator, category_scores, aggregate, format_report
from model_v4 import VLMAnomalyV4
from memory_v4 import build_memory, memory_maps
from pseudo_states import make_group_targets
from splits import split_categories


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    model = VLMAnomalyV4(
        ckpt["dims"], clip_model, ckpt["decoder_width"], ckpt["fusion_width"], ckpt["prompt_ctx"]
    ).to(device)
    model.load_compact_state_dict(ckpt["model"])
    model.eval()
    dino = load_dinov2(ckpt["arch"], device, args.weight_dir).eval()
    return ckpt, dino, model


def main(args):
    set_seed(args.seed)
    device = pick_device()
    ckpt, dino, model = load_model(args, device)
    all_samples = list_samples(args.data_root / "Train")
    fit, heldout = split_categories(all_samples, ckpt["heldout_fold"], args.folds)
    if args.limit:
        heldout = heldout[: args.limit]
    rng = np.random.default_rng(args.seed)
    image_labels = defaultdict(list)
    image_scores = defaultdict(list)
    branch_names = ("dino", "vlm", "query", "final")
    pixels = {
        branch: defaultdict(lambda: PixelAccumulator(neg_stride=args.neg_stride))
        for branch in branch_names
    }

    def feature_fn(images):
        return per_layer_features(dino, images, ckpt["dino_size"], device, tuple(ckpt["layers"]))
    bank = build_memory(feature_fn, fit, args.memory_budget, args.global_memory_budget, args.seed)

    with torch.inference_mode():
        for index, sample in enumerate(heldout):
            normal = load_many(sample.paths)
            # Evaluate one normal and one deterministic pseudo-anomalous group.
            for anomalous in (False, True):
                targets = make_group_targets(normal, rng, size=448, force_normal=not anomalous)
                images = [t.image for t in targets]
                maps, _ = per_layer_features(dino, images, ckpt["dino_size"], device, tuple(ckpt["layers"]))
                clip_x = clip_preprocess_tensor(images, ckpt["clip_size"], device)
                mem = torch.from_numpy(memory_maps(bank, maps, [sample.category] * 5, device)).to(device)
                output = model(maps, clip_x, seen=torch.zeros(5, device=device), groups=1, memory_map=mem)
                branch_maps = {}
                for branch, tensor in (
                    ("dino", output.dino_map),
                    ("vlm", output.vlm_map),
                    ("query", output.query_map),
                    ("final", output.final_map),
                ):
                    branch_maps[branch] = torch.nn.functional.interpolate(
                        tensor[:, None], (448, 448), mode="bilinear", align_corners=False
                    )[:, 0].cpu().numpy()
                dense = branch_maps["final"]
                score = float(torch.sigmoid(output.sample_logits)[0].cpu())
                # Keep dense evidence in image ranking; the learned MIL score
                # alone may be poorly calibrated on unseen categories.
                dense_score = max(topk_mean(x, 0.002, 8) for x in dense)
                score = 0.65 * dense_score + 0.35 * score
                image_labels[sample.category].append(int(anomalous))
                image_scores[sample.category].append(score)
                for view, target in enumerate(targets):
                    for branch in branch_names:
                        pixels[branch][sample.category].add(branch_maps[branch][view], target.binary_mask)
            if index % 10 == 0:
                print(f"heldout proxy {index}/{len(heldout)}", flush=True)

    per_category = {}
    for category in sorted(image_labels):
        labels, scores = pixels["final"][category].finish()
        per_category[category] = category_scores(
            np.asarray(image_labels[category]),
            np.asarray(image_scores[category]),
            labels,
            scores,
        )
    result = aggregate(per_category)
    branch_pixel = {}
    from metrics import auroc, average_precision, f1_max
    for branch in branch_names:
        per_branch = []
        for category in sorted(image_labels):
            labels, scores = pixels[branch][category].finish()
            per_branch.append(
                {
                    "p_auroc": auroc(labels, scores),
                    "p_ap": average_precision(labels, scores),
                    "p_f1max": f1_max(labels, scores),
                }
            )
        branch_pixel[branch] = {
            key: float(np.nanmean([row[key] for row in per_branch]))
            for key in ("p_auroc", "p_ap", "p_f1max")
        }
    result["branch_pixel_metrics"] = branch_pixel
    result["heldout_fold"] = ckpt["heldout_fold"]
    result["heldout_categories"] = sorted(image_labels)
    result["per_category"] = per_category
    result["selection_score"] = float(
        0.45 * result["p_ap"] + 0.30 * result["p_f1max"] + 0.15 * result["i_ap"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_report(result, "VLM v4 held-out-category proxy"), flush=True)
    print("branch_pixel_metrics=" + json.dumps(branch_pixel, indent=2), flush=True)
    print(f"selection_score_without_seen_term={result['selection_score']:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--neg-stride", type=int, default=17)
    parser.add_argument("--memory-budget", type=int, default=1536)
    parser.add_argument("--global-memory-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260823)
    main(parser.parse_args())
