"""Shared model, bank, and calibration loading for v6."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from calibration_v6 import fit_calibration
from clip_dense import load_clip
from model_v6 import VLMGuidedResidualV6, load_trusted_checkpoint
from pipeline_v6 import run_group
from reference_v6 import build_reference_banks
from progress_v6 import ProgressBar


def load_models(args, device):
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    clip_model, _ = load_clip(args.clip_model, args.clip_weight, device)
    model = VLMGuidedResidualV6(
        checkpoint["dims"], clip_model, checkpoint["decoder_width"], checkpoint["vlm_width"],
        checkpoint["prompt_ctx"], checkpoint["max_vlm_residual"],
    ).to(device)
    model.load_compact_state_dict(checkpoint["model"]); model.eval()
    from common import load_dinov2
    dino = load_dinov2(checkpoint["arch"], device, args.weight_dir).eval()
    return checkpoint, dino, model


def load_or_build_banks(args, checkpoint, model, dino, samples, device, cache_path):
    if cache_path and cache_path.exists():
        print(f"REFERENCE_CACHE_LOAD path={cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    reference_bar = ProgressBar("build_reference_banks", len(samples), every=10)

    def progress(current, total, category):
        reference_bar.update(current, {"part": category}, force=(current % 10 == 0 or current == total))

    from clip_dense import clip_preprocess_tensor
    banks = build_reference_banks(
        model, dino, clip_preprocess_tensor, samples, checkpoint["dino_size"],
        checkpoint["clip_size"], checkpoint["layers"], device,
        args.reference_budget, args.global_reference_budget, args.seed, progress,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(banks, temporary); temporary.replace(cache_path)
        print(f"REFERENCE_CACHE_SAVED path={cache_path}", flush=True)
    reference_bar.close({"part": "complete"})
    return banks


def fit_normal_calibration(args, checkpoint, dino, model, banks, samples, device):
    from calibration_v5 import stratified_samples
    from common import load_many
    pixels, images = defaultdict(list), defaultdict(list)
    selected = stratified_samples(samples, args.calib_per_category, args.seed)
    calibration_bar = ProgressBar("normal_calibration", len(selected), every=10)
    with torch.inference_mode():
        for index, sample in enumerate(selected):
            normal = load_many(sample.paths)
            pred = run_group(
                normal, [sample.category] * 5, model, dino, banks, checkpoint["dino_size"],
                checkpoint["clip_size"], checkpoint["layers"], device,
            )
            pixels[sample.category].append(
                np.concatenate([value.ravel()[::29] for value in pred.raw_maps]).astype(np.float32)
            )
            images[sample.category].append(pred.raw_image_score)
            if index % 10 == 0 or index + 1 == len(selected):
                calibration_bar.update(index + 1, {"category": sample.category}, force=True)
    calibration_bar.close({"category": "complete"})
    return fit_calibration(
        pixels, images, args.pixel_threshold_quantile, args.pixel_high_quantile
    )
