"""A3：冻结 DINOv2 + 冻结 Qwen3-VL 视觉塔 → 联合解码器。

这是项目主张「VLM + DINOv2」的直接检验：
    DINOv2 负责密集的纹理/结构对应（它很强）；
    VLM 视觉塔提供「更语义化」的表征（它是否带来跨类可迁移的信息？）。

两者都冻结，只训练解码器，所以在同一条 Protocol C 上可与 A0/B1 直接对比。

VLM 特征已由 extract_vlm_feats.py 缓存（28×28×512，fp16）。

用法：
    python -m seg_cross_part.train_a3 --fold 0 --steps 6000 --exp-id c_f0_a3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import MANIFEST_PIXEL, ROOT, load_config, resolve_splits, with_defaults
from .stats_pixel import PixelStats, image_hist, image_pro
from .train_seg import DinoFeatures, SegDataset, focal_dice


class VLMFeatStore:
    def __init__(self, root: Path, split: str):
        self.arr = np.load(root / f"{split}.npy", mmap_mode="r")
        idx = pd.read_parquet(root / f"{split}_index.parquet")
        self.pos = {p: i for i, p in enumerate(idx["image_path"])}

    def get(self, path: str):
        i = self.pos.get(path)
        return None if i is None else np.asarray(self.arr[i], dtype=np.float32)


class FusionDataset(Dataset):
    """在 SegDataset 之上挂一份缓存的 VLM 特征。"""

    def __init__(self, df, source_root, size, train, store: VLMFeatStore, drop_vlm: bool = False):
        self.df = df.reset_index(drop=True)
        self.drop_vlm = drop_vlm
        self.base = SegDataset(self.df, source_root, size, train)
        self.store = store

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, gt, path, cls, lbl = self.base[i]
        v = self.store.get(path)
        v = torch.zeros(28, 28, 512) if v is None else torch.from_numpy(v)
        if self.drop_vlm:
            v = torch.zeros_like(v)
        return x, v.permute(2, 0, 1).half(), gt, path, cls, lbl


class FusionDecoder(torch.nn.Module):
    def __init__(self, hidden: int = 256, out_size: int = 448):
        super().__init__()
        self.out_size = out_size
        self.p = torch.nn.ModuleList([torch.nn.Conv2d(768, hidden, 1) for _ in range(4)]
                                     + [torch.nn.Conv2d(512, hidden, 1)])
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 5, hidden, 3, padding=1), torch.nn.GroupNorm(8, hidden), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden // 2, 3, padding=1), torch.nn.GroupNorm(8, hidden // 2), torch.nn.GELU(),
        )
        self.head = torch.nn.Conv2d(hidden // 2, 1, 1)

    def forward(self, dino_feats, vlm_feat):
        base = dino_feats[0].shape[-2:]
        vs = F.interpolate(vlm_feat.float(), size=base, mode="bilinear", align_corners=False)
        xs = [p(f) for p, f in zip(self.p[:4], dino_feats)] + [self.p[4](vs)]
        x = self.fuse(torch.cat(xs, 1))
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.head(x).squeeze(1)


@torch.no_grad()
def evaluate(model, dino, loader, device, size):
    model.eval()
    st = PixelStats()
    for x, v, gt, path, cls, lbl in loader:
        x = x.to(device); v = v.to(device)
        f = dino(x)
        prob = torch.sigmoid(model(f, v)).float().cpu().numpy()
        gtn = gt.numpy() > 0.5
        for b in range(prob.shape[0]):
            neg, pos = image_hist(prob[b], gtn[b])
            pro, nr = image_pro(prob[b], gtn[b])
            st.add(group=path[b], cls=cls[b], area="none", label=int(lbl[b]),
                   img_score=float(prob[b].max()), neg=neg, pos=pos, pro=pro, n_regions=nr)
    m = st.metrics()
    out = {"macro": {k: float(np.mean([v[k] for v in st.by_class().values()]))
                     for k in ("apro_0.3", "aupr", "auroc", "i_auroc")}, "n": len(st)}
    return out, st


def main() -> None:
    ap = argparse.ArgumentParser(description="A3：DINOv2 + Qwen3-VL 视觉特征")
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="models/dinov2-base")
    ap.add_argument("--feats", default="artifacts/seg_cross_part/vlm_feats")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[1000, 2000, 3000, 4000, 5000, 6000])
    ap.add_argument("--limit", type=int, default=0, help="冒烟：每 split 只用 N 条")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--drop-vlm", action="store_true",
                    help="对照：把 VLM 特征置零（其余完全相同），用来隔离 VLM 的净贡献")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    src = ROOT / cfg["data"]["source_root"]
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "mask_path", "mask_valid"])
    feat_root = ROOT / args.feats

    def load(sp):
        df = pd.read_parquet(splits[sp]).merge(pix, on="image_path", how="left")
        store = VLMFeatStore(feat_root, sp)
        df = df[df["image_path"].isin(store.pos)].reset_index(drop=True)   # 只保留有特征的
        if args.limit:
            df = df.head(args.limit)
        return df, store

    tr_df, tr_store = load("train")
    va_df, va_store = load("val")
    te_df, te_store = load("test")
    print(f"[data] train={len(tr_df)} val={len(va_df)} test={len(te_df)} (已限制为有 VLM 特征的)")

    tr = FusionDataset(tr_df, src, args.size, True, tr_store, args.drop_vlm)
    va = FusionDataset(va_df, src, args.size, False, va_store, args.drop_vlm)
    te = FusionDataset(te_df, src, args.size, False, te_store, args.drop_vlm)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=6)
    te_loader = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=6)

    dino = DinoFeatures(args.backbone).to(device).eval()
    model = FusionDecoder(out_size=args.size).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable={n_train:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))

    exp = args.exp_id or f"a3_f{args.fold}"
    out_dir = ROOT / "artifacts" / "seg_cross_part" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_step = -1.0, 0
    it = iter(tr_loader)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        try:
            x, v, gt, *_ = next(it)
        except StopIteration:
            it = iter(tr_loader); x, v, gt, *_ = next(it)
        x = x.to(device); v = v.to(device); gt = gt.to(device)
        with torch.no_grad():
            f = dino(x)
        loss = focal_dice(model(f, v), gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"  step {step}/{args.steps} loss={loss.item():.4f} ({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step in args.eval_steps:
            m, _ = evaluate(model, dino, va_loader, device, args.size)
            print(f"[eval@{step}] val macro AUPRO={m['macro']['apro_0.3']:.4f} "
                  f"P-AUPR={m['macro']['aupr']:.4f}", flush=True)
            if m["macro"]["apro_0.3"] > best:
                best, best_step = m["macro"]["apro_0.3"], step
                torch.save({"decoder": model.state_dict(), "step": step}, out_dir / "best.pt")
    torch.save({"decoder": model.state_dict(), "step": args.steps}, out_dir / "last.pt")

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["decoder"])
    m_val, _ = evaluate(model, dino, va_loader, device, args.size)
    m_te, st_te = evaluate(model, dino, te_loader, device, args.size)
    st_te.save(out_dir / "stats_test.npz")
    res = {"exp_id": exp, "best_step": best_step, "val": m_val, "test": m_te,
           "by_class": st_te.by_class(), "n_train": len(tr_df)}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[val ] macro AUPRO={m_val['macro']['apro_0.3']:.4f}")
    print(f"[test] macro AUPRO={m_te['macro']['apro_0.3']:.4f} P-AUPR={m_te['macro']['aupr']:.4f} "
          f"P-AUROC={m_te['macro']['auroc']:.4f}")
    print(f"已写出 {out_dir}/metrics.json 与 stats_test.npz")


if __name__ == "__main__":
    main()
