#!/usr/bin/env python3
"""Train-only pseudo-defect proxy for selecting a v3 decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import LAYER_SETS, list_samples, load_dinov2, per_layer_features, pick_device, set_seed
from multiscale_decoder import LayerAwarePixelDecoder
from pseudo_v3 import make_training_targets


def binary_metrics(score, target):
    y = target.reshape(-1).astype(np.uint8)
    s = score.reshape(-1).astype(np.float64)
    order = np.argsort(-s, kind="stable")
    yy, ss = y[order], s[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    positives = max(int(y.sum()), 1)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else 0.0
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 101)))
    f1s = []
    for t in thresholds:
        pred = s >= t
        tpv = float(np.sum(pred & (y > 0))); fpv = float(np.sum(pred & (y == 0))); fnv = float(np.sum(~pred & (y > 0)))
        f1s.append(2 * tpv / max(2 * tpv + fpv + fnv, 1.0))
    # Rank AUROC via pairwise comparison against sorted negatives.
    pos = s[y > 0]; neg = s[y == 0]
    if len(pos) and len(neg):
        neg_sorted = np.sort(neg)
        less = np.searchsorted(neg_sorted, pos, side="left")
        less_or_equal = np.searchsorted(neg_sorted, pos, side="right")
        ties = less_or_equal - less
        auc = float((less.sum() + 0.5 * ties.sum()) / (len(pos) * len(neg)))
    else:
        auc = 0.5
    return {"pixel_ap": ap, "pixel_f1max": float(max(f1s or [0.0])), "pixel_auroc": auc}


def main(args):
    set_seed(20260821)
    device = pick_device(); layers = LAYER_SETS[args.arch]
    model = load_dinov2(args.arch, device, args.weight_dir)
    samples = list_samples(args.data_root / "Train")
    samples = samples[int(len(samples) * args.train_fraction):]
    samples = samples[:args.limit] if args.limit else samples
    ckpt = torch.load(args.checkpoint, map_location=device)
    decoder = LayerAwarePixelDecoder(ckpt["dims"], width=ckpt["width"]).to(device)
    decoder.load_state_dict(ckpt["state"]); decoder.eval()
    rng = np.random.default_rng(20260822)
    scores, targets = [], []
    for i, sample in enumerate(samples):
        for path in sample.paths:
            with Image.open(path) as im: normal = im.convert("RGB")
            defective, target, _ = make_training_targets(normal, rng)
            with torch.no_grad():
                maps, _ = per_layer_features(model, [defective], args.size, device, layers)
                logits, _ = decoder(maps)
                pred = torch.sigmoid(logits.float())[0].cpu().numpy()
            pred = np.asarray(Image.fromarray(np.uint8(pred * 255), "L").resize((448, 448), Image.Resampling.BILINEAR), np.float32) / 255.0
            scores.append(pred); targets.append(target)
        if i % 20 == 0: print(f"proxy {i}/{len(samples)}", flush=True)
    result = binary_metrics(np.stack(scores), np.stack(targets))
    result.update({"samples": len(samples), "images": len(scores), "official_train_only": True, "external_data": False})
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--weight-dir", type=Path, required=True); ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="dinov2_vitb14"); ap.add_argument("--size", type=int, default=1036); ap.add_argument("--train-fraction", type=float, default=.70); ap.add_argument("--limit", type=int, default=80)
    main(ap.parse_args())
