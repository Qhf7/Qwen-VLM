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

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "v7_vlm_primary_v5"), str(ROOT / "v6_vlm_zero_shot_v4"), str(ROOT / "v5_multiscale_v3")]

from calibration_v6 import ExcessBounds, quantize_mask
from losses_v6 import V6Loss
from model_v6 import VLMGuidedResidualV6
from reference_v6 import PatchReferenceBank
from select_v6_checkpoint import selection_score
from train_v6 import configure_trainable
from progress_v6 import ProgressBar


class Pass(nn.Module):
    def forward(self, x): return x


class DummyVisual(nn.Module):
    def __init__(self):
        super().__init__(); width, out = 32, 16
        self.conv1 = nn.Conv2d(3, width, 16, 16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width) * 0.01)
        self.ln_pre = nn.LayerNorm(width); self.transformer = Pass(); self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, out) * 0.02); self.output_dim = out


class DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__(); width, out = 32, 16
        self.visual = DummyVisual(); self.token_embedding = nn.Embedding(50000, width)
        self.positional_embedding = nn.Parameter(torch.randn(77, width) * 0.01)
        self.transformer = Pass(); self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, out) * 0.02); self.dtype = torch.float32


def main():
    bar = ProgressBar("unit_test", 3, every=1)
    # Sparse calibration must leave all values below the normal-tail threshold black.
    bounds = ExcessBounds(0.3, 0.1)
    mapped = bounds.pixel(np.array([0.0, 0.2, 0.3, 0.4, 0.8]))
    assert np.all(mapped[:3] == 0) and np.all(np.diff(mapped[2:]) > 0)
    assert np.allclose(quantize_mask(np.array([0.0, 0.5, 1.0])), np.array([0, 128, 255]) / 255)
    bar.update(1, {"stage": "calibration"}, force=True)
    bank = PatchReferenceBank(
        {("known", 0): np.eye(4, dtype=np.float32)}, np.eye(4, dtype=np.float32),
        {("known", 0): (0.2, 0.1)}, (0.3, 0.1), knn=1,
    )
    feature = torch.eye(4).reshape(1, 2, 2, 4)
    known = bank.score(feature, ["known"], torch.device("cpu"))
    unseen = bank.score(feature, ["unseen"], torch.device("cpu"))
    assert float(known.max()) == 0.0 and float(unseen.max()) == 0.0

    model = VLMGuidedResidualV6([16] * 4, DummyCLIP(), 32, 32, 8, max_vlm_residual=1.25)
    configure_trainable(model)
    batch = 5; maps = [torch.randn(batch, 4, 4, 16) for _ in range(4)]
    images = torch.randn(batch, 3, 32, 32)
    _, patch = model.encode_clip(images)
    output = model(
        maps, images, dino_reference=torch.rand(batch, 4, 4),
        clip_reference=torch.rand(batch, patch.shape[1], patch.shape[2]), groups=1,
    )
    assert output.final_logit.shape == (5, 16, 16)
    assert torch.allclose(output.final_logit, output.dino_logit + output.vlm_residual)
    bar.update(2, {"stage": "forward_backward"}, force=True)
    assert float(output.vlm_residual.detach().abs().max()) <= 1.25001
    # Positive VLM evidence cannot escape a closed DINO candidate gate.
    signed = output.vlm_residual.clamp_min(0)
    assert float((signed * (output.candidate_gate < 0.05)).detach().max()) < 0.063

    binary = torch.zeros(batch, 64, 64); binary[:, 20:24, 20:28] = 1
    states = torch.ones(batch, 64, 64, dtype=torch.long); states[:, 20:24, 20:28] = 4
    states[:, 5:8] = 2
    hard = torch.zeros(batch, 64, 64); hard[:, 5:8] = 1
    loss, parts = V6Loss()(
        model, output, binary, states, hard, torch.ones(batch), torch.ones(1)
    )
    loss.backward()
    assert torch.isfinite(loss) and parts["area"] >= 0 and parts["boundary"] >= 0

    clean = {"p_ap": .45, "p_f1max": .50, "p_auroc": .85, "i_ap": .94, "i_auroc": .90, "selection_penalty": 0}
    broad = dict(clean, p_ap=.48, selection_penalty=.20)
    assert selection_score(clean) > selection_score(broad)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory); (path / "proxy_epoch0.json").write_text(json.dumps(broad))
        (path / "proxy_epoch1.json").write_text(json.dumps(clean))
        (path / "proxy_epoch1.progress.json").write_text("{}")
        out = path / "selection.json"
        subprocess.run([sys.executable, str(HERE / "select_v6_checkpoint.py"), "--proxy-dir", str(path), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
        assert json.loads(out.read_text())["best_epoch"] == 1
    bar.close({"stage": "complete"})
    print("V6_ALL_TESTS_OK")


if __name__ == "__main__": main()
