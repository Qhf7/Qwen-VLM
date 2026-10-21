"""离线像素级评测：从已保存的 checkpoint 复评，产出可复算的统计量。

为什么要单独一个脚本：训练时只留了聚合指标，没法事后做「按面积档分层」和
「按 group 的 bootstrap」。这里把每张图的充分统计量（直方图 + 区域覆盖曲线）
落盘，之后分层/重采样都不需要再过模型。

用法：
    python -m seg_cross_part.evaluate_pixel \
      --run artifacts/seg_cross_part/b_a0_fix_s20261021 --split test

    # 冒烟（不加载真主干，验证链路）
    python -m seg_cross_part.evaluate_pixel --run <dir> --split val --stub --limit 24

产物：
    <run>/stats_<split>.npz        每图统计量（bootstrap 用）
    <run>/eval_<split>.json        overall / by_class / by_area
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import MANIFEST_PIXEL, ROOT, resolve_splits, with_defaults
from .stats_pixel import PixelStats, image_hist, image_pro
from .train_seg import DinoFeatures, SegCollate, SegDataset, SegDecoder, StubFeatures


def build_model(arm: str, backbone: str, hidden: int, size: int, device, stub: bool):
    if stub:
        feats = StubFeatures()
        dims = [feats.dim] * 4
        up = 2
    else:
        feats = DinoFeatures(backbone)
        dims = [768] * 4
        up = 1
    dec = SegDecoder(dims, hidden=hidden, out_size=size).to(device)
    feats.to(device)

    class M(torch.nn.Module):
        def forward(self, x):
            f = self.feats(x)
            if self.up != 1:
                f = [torch.nn.functional.interpolate(t, scale_factor=self.up, mode="nearest") for t in f]
            return self.dec(f)

    m = M()
    m.feats, m.dec, m.up = feats, dec, up
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="离线像素级评测（含分层统计量）")
    ap.add_argument("--run", type=Path, required=True, help="训练产物目录（含 best.pt）")
    ap.add_argument("--split", choices=["val", "test", "train"], default="test")
    ap.add_argument("--weights", type=str, default="best.pt")
    ap.add_argument("--stub", action="store_true", help="用 stub 特征，验证链路（结果无效）")
    ap.add_argument("--limit", type=int, default=None, help="只评测 N 张（试跑）")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else ("mps" if torch.backends.mps.is_available() else "cpu"))
    frozen = json.loads((args.run / "frozen_protocol.json").read_text(encoding="utf-8"))
    proto = frozen["protocol"]
    cfg = with_defaults({"protocol": proto,
                         "model": {"hidden": frozen.get("model", {}).get("hidden", 256)},
                         "input": {"image_size": frozen.get("input", {}).get("image_size", 448)}})
    size = cfg["input"]["image_size"]
    backbone = frozen.get("model", {}).get("backbone", "models/dinov2-base")
    arm = frozen.get("arm", "a0")
    print(f"[eval] run={args.run.name} arm={arm} split={args.split} device={device} stub={args.stub}")

    splits = resolve_splits(cfg, None if proto.get("type") != "c" else proto.get("fold"))
    pix = pd.read_parquet(MANIFEST_PIXEL)[
        ["image_path", "group_path", "class_name", "area_bucket", "mask_path", "mask_valid", "view_label"]]
    df = pd.read_parquet(splits[args.split]).merge(pix, on="image_path", how="left", suffixes=("", "_pix"))
    for col in ("group_path", "class_name"):
        if f"{col}_pix" in df.columns:
            df[col] = df[f"{col}_pix"]
            df = df.drop(columns=[f"{col}_pix"])
    df = df[df["mask_valid"].notna()]
    if args.limit:
        df = df.sample(min(len(df), args.limit), random_state=0)
    df = df.reset_index(drop=True)
    print(f"[eval] n={len(df)} 类={df.class_name.nunique()} 组={df.group_path.nunique()}")

    ds = SegDataset(df, ROOT / cfg["data"]["source_root"], size, train=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4,
                        collate_fn=SegCollate())
    meta = {r["image_path"]: r for _, r in df.iterrows()}

    model = build_model(arm, backbone, cfg["model"]["hidden"], size, device, args.stub)
    ck = args.run / args.weights
    if ck.exists() and not args.stub:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.dec.load_state_dict(state["decoder"])
        print(f"[eval] 载入 {ck.name}（step={state.get('step')}）")
    elif not args.stub:
        raise SystemExit(f"找不到权重 {ck}")
    model.eval().to(device)

    st = PixelStats()
    with torch.no_grad():
        for bi, (x, gt, paths, cls, lbl) in enumerate(loader):
            prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
            gts = gt.numpy()
            for i, p in enumerate(paths):
                r = meta[p]
                score, g = prob[i], gts[i] > 0.5
                neg, pos = image_hist(score, g)
                pro, nreg = image_pro(score, g)
                st.add(group=r["group_path"], cls=r["class_name"], area=r["area_bucket"],
                       label=int(r["view_label"]), img_score=float(score.max()),
                       neg=neg, pos=pos, pro=pro, n_regions=nreg)
            if (bi + 1) % 50 == 0:
                print(f"  {min((bi+1)*args.batch, len(df))}/{len(df)}", flush=True)

    out = {"overall": st.metrics(), "by_class": st.by_class(), "by_area": st.by_area(),
           "run": args.run.name, "split": args.split, "stub": bool(args.stub)}
    (args.run / f"stats_{args.split}.npz").parent.mkdir(parents=True, exist_ok=True)
    st.save(args.run / f"stats_{args.split}.npz")
    (args.run / f"eval_{args.split}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    o = out["overall"]
    print(f"\n[overall] AUPRO={o['apro_0.3']:.4f}  P-AUPR={o['aupr']:.4f}  "
          f"P-AUROC={o['auroc']:.4f}  I-AUROC={o['i_auroc']:.4f}")
    print("[by_area] " + "  ".join(f"{k}:{v['apro_0.3']:.3f}" for k, v in out["by_area"].items()))
    print(f"已写出 {args.run}/eval_{args.split}.json 与 stats_{args.split}.npz")


if __name__ == "__main__":
    main()
