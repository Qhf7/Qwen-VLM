#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "v6_vlm_zero_shot_v4"), str(ROOT / "v5_multiscale_v3")]

from calibration_v5 import RobustBounds, assert_category_coverage, fit_calibration, stratified_samples
from common import list_samples
from losses_v5 import V5Loss, probability_binary_cross_entropy
from model_v5 import VLMPrimaryV5, logmeanexp
from pseudo_v5 import DEFAULT_HELDOUT_GENERATORS, DEFAULT_TRAIN_GENERATORS
from train_v5 import configure_trainable


class Pass(nn.Module):
    def forward(self, x):
        return x


class DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        width, out = 32, 16
        self.conv1 = nn.Conv2d(3, width, 16, 16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width) * 0.01)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Pass()
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, out) * 0.02)
        self.output_dim = out


class DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        width, out = 32, 16
        self.visual = DummyVisual()
        self.token_embedding = nn.Embedding(50000, width)
        self.positional_embedding = nn.Parameter(torch.randn(77, width) * 0.01)
        self.transformer = Pass()
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, out) * 0.02)
        self.dtype = torch.float32


def main():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(logmeanexp(x, 1), logmeanexp(x.repeat_interleave(2, 1), 1), atol=1e-6)
    assert not set(DEFAULT_TRAIN_GENERATORS) & set(DEFAULT_HELDOUT_GENERATORS)

    # Regression test for the server crash seen under mixed precision.  This
    # helper must use BCEWithLogitsLoss internally, not autocast-unsafe BCE.
    probability = torch.tensor([0.2, 0.8], requires_grad=True)
    target = torch.tensor([0.0, 1.0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_loss = probability_binary_cross_entropy(probability, target)
    amp_loss.backward()
    assert torch.isfinite(amp_loss) and probability.grad is not None

    samples = list_samples(ROOT / "Dataset" / "Train")
    categories = sorted({sample.category for sample in samples})
    selected = stratified_samples(samples, 10, 20260825)
    assert len(categories) == 50 and len(selected) == 500
    assert {sample.category for sample in selected} == set(categories)
    pixel = {category: [np.linspace(0, 1, 100, dtype=np.float32)] for category in categories}
    image = {category: [0.1, 0.2, 0.3, 0.4] for category in categories}
    calibration = fit_calibration(pixel, image)
    assert_category_coverage(calibration, categories)
    mapped = RobustBounds(0.2, 0.6).apply(np.linspace(0, 1, 10))
    assert np.all(np.diff(mapped) > 0)

    model = VLMPrimaryV5([16] * 4, DummyCLIP(), 32, 32, 8, 0.15)
    configure_trainable(model)
    batch = 5
    maps = [torch.randn(batch, 4, 4, 16) for _ in range(4)]
    images = torch.randn(batch, 3, 32, 32)
    _, patch = model.encode_clip(images)
    output = model(maps, images, reference_map=torch.rand(batch, patch.shape[1], patch.shape[2]), groups=1)
    assert output.final_map.shape == (5, 16, 16)
    assert float(output.dino_fraction.detach()) <= 0.150001
    binary = torch.zeros(batch, 448, 448); binary[:, 100:110, 100:130] = 1
    states = torch.ones(batch, 448, 448, dtype=torch.long); states[:, 100:110, 100:130] = 4
    loss, _ = V5Loss()(model, output, binary, states, torch.ones(batch), torch.ones(1))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        rows = [
            (0, {"p_ap": 0.4, "p_f1max": 0.5, "p_auroc": 0.8, "i_ap": 0.9, "i_auroc": 0.85}),
            (1, {"p_ap": 0.5, "p_f1max": 0.55, "p_auroc": 0.82, "i_ap": 0.88, "i_auroc": 0.84}),
            (2, {"p_ap": 0.45, "p_f1max": 0.52, "p_auroc": 0.81, "i_ap": 0.87, "i_auroc": 0.83}),
        ]
        for epoch, data in rows:
            (directory / f"proxy_epoch{epoch}.json").write_text(json.dumps(data), encoding="utf-8")
        (directory / "proxy_epoch0.calibration_progress.json").write_text("{}", encoding="utf-8")
        (directory / "proxy_epoch1.progress.json").write_text("{}", encoding="utf-8")
        out = directory / "selection.json"
        subprocess.run(
            [sys.executable, str(HERE / "select_v5_checkpoint.py"), "--proxy-dir", str(directory), "--out", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert json.loads(out.read_text())["best_epoch"] == 1
    print("V5_ALL_TESTS_OK")


if __name__ == "__main__":
    main()
