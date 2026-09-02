"""Load the official AdaCLIP implementation with the paper configuration."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "third_party" / "AdaCLIP"
DEFAULT_CLIP_WEIGHT = ROOT / "models" / "adaclip" / "ViT-L-14-336px.pt"


def ensure_clip_cache(clip_weight: Path = DEFAULT_CLIP_WEIGHT) -> Path:
    clip_weight = clip_weight.expanduser().resolve()
    if not clip_weight.is_file():
        raise FileNotFoundError(f"OpenAI CLIP backbone not found: {clip_weight}")
    cache_dir = Path.home() / ".cache" / "clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / clip_weight.name
    if target.exists() or target.is_symlink():
        return target
    target.symlink_to(clip_weight)
    return target


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return requested


def build_model(checkpoint: Path, *, device: str = "auto", image_size: int = 518,
                clip_weight: Path = DEFAULT_CLIP_WEIGHT, seed: int = 111,
                precision: str = "fp32"):
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AdaCLIP checkpoint not found: {checkpoint}")
    ensure_clip_cache(clip_weight)
    device = resolve_device(device)
    # Matches official test.py. HSF's sklearn KMeans has random_state=None, so
    # inference is otherwise non-deterministic and even dependent on image order.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    old_cwd = Path.cwd()
    try:
        os.chdir(OFFICIAL_ROOT)
        from method import AdaCLIP_Trainer

        config_path = OFFICIAL_ROOT / "model_configs" / "ViT-L-14-336.json"
        model_config = json.loads(config_path.read_text())
        n_layers = model_config["vision_cfg"]["layers"]
        substage = n_layers // 4
        model = AdaCLIP_Trainer(
            backbone="ViT-L-14-336",
            feat_list=[substage, substage * 2, substage * 3, substage * 4],
            input_dim=model_config["vision_cfg"]["width"],
            output_dim=model_config["embed_dim"],
            learning_rate=0.0,
            device=device,
            image_size=image_size,
            prompting_depth=4,
            prompting_length=5,
            prompting_branch="VL",
            prompting_type="SD",
            use_hsf=True,
            k_clusters=20,
        ).to(device)
        model.load(str(checkpoint))
        if precision == "fp16":
            if device == "cpu":
                raise ValueError("fp16 is unsupported in the verified CPU runner")
            model.half()
        elif precision != "fp32":
            raise ValueError(f"unsupported precision: {precision}")
        model.eval()
        return model, device
    finally:
        os.chdir(old_cwd)
