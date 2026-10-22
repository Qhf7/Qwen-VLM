"""抽取 Qwen3-VL 视觉塔特征并缓存（A3 用）。

为什么要缓存：视觉塔 576M，单张 448px 前向约 70–100 ms，加上读图解码约 0.25 s/张。
训练要反复读同一批图，在线算会拖到不可接受，所以先一次性抽取成 npy。

存储：每张图 28×28 网格 × 1152 维 → 用一个固定随机投影降到 512 维，fp16 存
（28×28×512×2 ≈ 800 KB/张，2 万张约 16 GB）。

用法：
    python -m seg_cross_part.extract_vlm_feats --protocol protocol_c_unseen_category --fold 0 \
        --splits train val test --train-limit 8000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .common import ROOT, load_config, resolve_splits, with_defaults

VLM_MEAN = (0.5, 0.5, 0.5)
VLM_STD = (0.5, 0.5, 0.5)
OUT_DIM = 512
GRID = 28          # 448 / 16


def build_projection(in_dim: int = 1152, out_dim: int = OUT_DIM, seed: int = 0) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(in_dim, out_dim, generator=g) / (in_dim ** 0.5)
    return w.numpy().astype(np.float32)


class VLMFeats:
    def __init__(self, model_dir: str, size: int = 448, device: str = "cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_dir)
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device).eval()
        self.vis = m.model.visual
        self.size = size
        self.grid = size // 16
        self.device = device
        self.W = torch.from_numpy(build_projection()).to(device)

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> np.ndarray:
        im = img.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        px = self.proc.image_processor(images=[im], return_tensors="pt")["pixel_values"]
        px = px.to(self.device, torch.bfloat16)
        grid = torch.tensor([[1, self.grid, self.grid]], device=self.device)
        out = self.vis(px, grid_thw=grid)
        hs = out.last_hidden_state                      # (grid*grid, 1152)
        hs = hs.reshape(self.grid, self.grid, -1).float()
        hs = hs @ self.W                                # (28,28,512)
        return hs.half().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取 Qwen3-VL 视觉特征")
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--vlm", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--train-limit", type=int, default=8000, help="训练集只抽前 N 张（控制缓存体积）")
    ap.add_argument("--out", default="artifacts/seg_cross_part/vlm_feats")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    src = ROOT / cfg["data"]["source_root"]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "projection.npy").write_bytes(b"")   # 占位，见下
    np.save(out_dir / "projection.npy", build_projection())

    model = VLMFeats(args.vlm)
    for sp in args.splits:
        df = pd.read_parquet(splits[sp])
        if sp == "train" and args.train_limit:
            df = df.sample(min(args.train_limit, len(df)), random_state=0).reset_index(drop=True)
        n = len(df)
        arr = np.lib.format.open_memmap(out_dir / f"{sp}.npy", mode="w+",
                                        dtype=np.float16, shape=(n, GRID, GRID, OUT_DIM))
        t0 = time.time()
        for i, p in enumerate(df["image_path"]):
            arr[i] = model(Image.open(src / p))
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"[{sp}] {i+1}/{n}  {el/(i+1):.2f}s/张  预计剩余 {el/(i+1)*(n-i-1)/60:.0f} 分钟",
                      flush=True)
        arr.flush()
        df[["image_path"]].to_parquet(out_dir / f"{sp}_index.parquet")
        print(f"[{sp}] 完成 {n} 张，{(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
