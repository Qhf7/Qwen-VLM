"""跨零件异常分割训练器（A0：冻结 DINOv2 + 解码器）。

这是新方向的第一把尺子：**纯视觉**的跨类分割基线。
后续 A1（+VLM 视觉）、A2（+VLM 语言）、A3（两者）都复用同一个训练器，只换 --arm。

用法：
    # 1) 离线冒烟（stub 特征，不下载权重）
    python -m seg_cross_part.train_seg --smoke

    # 2) 真跑（服务器）
    python -m seg_cross_part.train_seg \
      --protocol protocol_b_seen_category \
      --backbone models/dinov2-base --arm a0 \
      --exp-id b_s0_dino_s20261020

    # 3) Protocol C 四折
    ... --protocol protocol_c_unseen_category --fold 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .common import (MANIFEST_PIXEL, ROOT, check_protocol, load_config,
                     resolve_splits, sha256_of, with_defaults)
from .metrics_pixel import PixelEvaluator, load_mask

OUT_ROOT = ROOT / "artifacts" / "seg_cross_part"


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
class SegDataset(Dataset):
    """图像 + 像素 mask。

    **几何变换必须同时作用在图像和 mask 上**（这是分割训练的命门）：
    随机裁剪/翻转的同一组参数同时裁剪两者，否则监督信号与输入错位，
    模型学到的是"错的 mask"，评测却用对齐的 mask，训练与评测脱节。
    颜色/归一化只作用在图像上。
    """

    def __init__(self, df: pd.DataFrame, source_root: Path, size: int, train: bool,
                 rrc_scale=(0.85, 1.0), hflip: float = 0.5):
        self.df = df.reset_index(drop=True)
        self.source_root = source_root
        self.size = size
        self.train = train
        self.rrc_scale = tuple(rrc_scale)
        self.hflip = float(hflip)
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    # ---------------- 几何：图像与 mask 共用 ----------------
    def _load_mask_pil(self, path: Path, size_wh):
        from PIL import Image
        m = Image.open(path).convert("L")
        if m.size != tuple(size_wh):
            m = m.resize(tuple(size_wh), Image.NEAREST)
        return m.point(lambda v: 255 if v > 127 else 0)

    def _geom(self, img, mask):
        from PIL import Image
        s = self.size
        if not self.train:
            return (img.resize((s, s), Image.BILINEAR),
                    mask.resize((s, s), Image.NEAREST))
        W, H = img.size
        area = float(W * H)
        box = None
        for _ in range(10):
            target = area * float(torch.empty(1).uniform_(*self.rrc_scale))
            ar = float(torch.empty(1).uniform_(0.9, 1.1))
            w = int(round(math.sqrt(target * ar)))
            h = int(round(math.sqrt(target / ar)))
            if 0 < w <= W and 0 < h <= H:
                i = int(torch.randint(0, H - h + 1, (1,)))
                j = int(torch.randint(0, W - w + 1, (1,)))
                box = (j, i, j + w, i + h)
                break
        if box is None:
            box = (0, 0, W, H)
        img = img.crop(box).resize((s, s), Image.BILINEAR)
        mask = mask.crop(box).resize((s, s), Image.NEAREST)
        if float(torch.rand(1)) < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = default_loader(str(self.source_root / r["image_path"]))
        mask = (self._load_mask_pil(self.source_root / r["mask_path"], img.size)
                if r["mask_valid"] else Image.new("L", img.size, 0))
        img, mask = self._geom(img, mask)
        x = self.norm(self.to_tensor(img))
        gt = torch.from_numpy((np.array(mask) > 127).astype(np.float32))
        return (x, gt, str(r["image_path"]), str(r["class_name"]), int(r["view_label"]))


def check_alignment(n: int = 20, iou_thr: float = 0.90) -> None:
    """自检：图像里的缺陷区域与 mask 必须始终重合。

    做法：造一张「亮块 + 黑底」的图，mask 标的就是这个亮块；
    几何变换后，阈值化图像 与 变换后的 mask 的 IoU 必须仍接近 1。
    若图像被裁剪/翻转而 mask 没跟上，IoU 会掉到缺陷面积占比量级（<0.1）。
    注意：图像用双线性、mask 用最近邻，边缘会有细小差异，所以判据是 IoU>=0.90。
    """
    size = 448
    df = pd.DataFrame([{
        "image_path": "synthetic.png", "mask_path": "synthetic_mask.png",
        "class_name": "synthetic", "view_label": 1, "mask_valid": True,
    }])
    ds = SegDataset(df, ROOT, size, train=True)
    rng = np.random.default_rng(0)
    ious = []
    for _ in range(n):
        arr = np.zeros((600, 640), np.uint8)
        y, x = rng.integers(50, 500), rng.integers(50, 550)
        arr[y:y + 60, x:x + 40] = 255
        img = Image.fromarray(np.stack([arr] * 3, -1))
        mask = Image.fromarray(arr)
        img2, mask2 = ds._geom(img, mask)
        a = np.array(img2.convert("L")) > 127
        b = np.array(mask2) > 127
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append(inter / max(union, 1))
    ious = np.asarray(ious)
    print(f"对齐自检：{n} 次变换，IoU 均值 {ious.mean():.4f} 最小 {ious.min():.4f}")
    if ious.min() < iou_thr:
        raise SystemExit(f"几何变换错位（最小 IoU {ious.min():.4f} < {iou_thr}），禁止训练")
    print("对齐自检通过：图像与 mask 使用同一套裁剪/翻转")


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class StubFeatures(nn.Module):
    """冒烟用：随机卷积产生 4 路「特征」，接口与真主干一致。"""

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.stem = nn.Conv2d(3, dim, 3, stride=1, padding=1)

    def forward(self, x):
        f = torch.relu(self.stem(x))
        f = F.avg_pool2d(f, 2)                      # /2
        return [f, f, f, f]


class DinoFeatures(nn.Module):
    """冻结 DINOv2，取多层 patch 特征（去掉 CLS），拼成 4×768。"""

    def __init__(self, backbone: str, layers=(3, 6, 9, 12)):
        super().__init__()
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(backbone)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = list(layers)

    @torch.no_grad()
    def forward(self, x):
        out = self.model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states                      # (embed + 12 层)
        grid = int(round(math.sqrt(x.shape[-1] * x.shape[-2] / (14 ** 2))))
        feats = []
        for L in self.layers:
            f = hs[L][:, 1:, :]                     # 去掉 CLS
            b, n, c = f.shape
            g = int(round(math.sqrt(n)))
            feats.append(f.transpose(1, 2).reshape(b, c, g, g))
        return feats


class SegDecoder(nn.Module):
    """4 层融合 → 逐级上采样到输入分辨率。"""

    def __init__(self, in_dims, hidden=256, out_size=448):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.ModuleList([nn.Conv2d(d, hidden, 1) for d in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * len(in_dims), hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GroupNorm(8, hidden // 2), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden // 2, 1, 1)

    def forward(self, feats):
        base = feats[0].shape[-2:]
        x = torch.cat([p(f) if p(f).shape[-1] == base[0] else
                       F.interpolate(p(f), size=base, mode="bilinear", align_corners=False)
                       for p, f in zip(self.proj, feats)], dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear",
                          align_corners=False)
        return self.head(x).squeeze(1)



class SegModel(nn.Module):
    """冻结视觉特征 + 解码器 (+ 可选语言分支)。训练与离线复评共用。"""

    def __init__(self, feats, decoder, upscale=1, lang=None, out_size=448):
        super().__init__()
        self.feats, self.decoder, self.upscale, self.lang = feats, decoder, upscale, lang
        self.out_size = out_size

    def forward(self, xb):
        f = self.feats(xb)
        # 语言分支在原始 patch 网格（32×32）上算；放到 448 会慢 8 倍以上
        aux = None
        if self.lang is not None:
            aux = self.lang(f, out_size=self.out_size)
        if self.upscale != 1:
            f = [F.interpolate(t, scale_factor=self.upscale, mode="nearest") for t in f]
        logits = self.decoder(f)
        if aux is None:
            return logits, None
        return logits + aux, aux


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------
def focal_dice(logits, gt, gamma=2.0, eps=1e-6):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pt = p * gt + (1 - p) * (1 - gt)
    focal = ((1 - pt) ** gamma * ce).mean()
    inter = (p * gt).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + gt.sum() + eps)
    return 0.5 * focal + 0.5 * dice


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, size, n_classes_log=0):
    model.eval()
    per_class: dict[str, PixelEvaluator] = {}
    for batch in loader:
        x, gt, paths, cls, lbl = batch
        x = x.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits).cpu().numpy()
        gts = gt.numpy()
        for i in range(len(paths)):
            ev = per_class.setdefault(cls[i], PixelEvaluator())
            ev.add(prob[i], gts[i] > 0.5, lbl[i])
    out = {"per_class": {}, "macro": {}}
    for c, ev in per_class.items():
        out["per_class"][c] = ev.compute()
    if out["per_class"]:
        for k in ("auroc", "aupr", "f1max", "apro_0.3", "i_auroc", "i_aupr"):
            vals = [v[k] for v in out["per_class"].values() if np.isfinite(v[k])]
            out["macro"][k] = float(np.mean(vals)) if vals else float("nan")
    return out


class SegCollate:
    def __call__(self, batch):
        xs = torch.stack([b[0] for b in batch])
        gts = torch.stack([b[1] for b in batch])
        return xs, gts, [b[2] for b in batch], [b[3] for b in batch], [b[4] for b in batch]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def train(cfg, args):
    seed = cfg["train"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else ("mps" if torch.backends.mps.is_available() else "cpu"))
    size = cfg["input"]["image_size"]
    print(f"[seg] device={device} size={size} arm={args.arm}")

    splits = resolve_splits(cfg, args.fold)
    info = check_protocol(cfg, splits)
    print(f"[protocol] type={info['type']} fold={info['fold']} | "
          f"train {info['train_classes']} 类 / test {info['test_classes']} 类 | "
          f"类别交集 {info['train_test_class_overlap']} | "
          f"train {info['train_views']} 视图 / test {info['test_views']} 视图")
    print(f"[protocol] test 示例类: {info['test_sample_classes']}")
    split_sha = {s: sha256_of(p) for s, p in splits.items()}
    pix = pd.read_parquet(MANIFEST_PIXEL)

    def load(name):
        df = pd.read_parquet(splits[name]).merge(
            pix[["image_path", "mask_path", "mask_valid", "pixel_eval_valid"]],
            on="image_path", how="left")
        df = df[df["pixel_eval_valid"]]
        if args.limit_per_split:
            df = df.sample(min(len(df), args.limit_per_split), random_state=seed)
        return df.reset_index(drop=True)

    train_df, val_df, test_df = load("train"), load("val"), load("test")
    print(f"[splits] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ---- 模型 ----
    if args.smoke:
        feats = StubFeatures()
        in_dims = [feats.dim] * 4
        dummy_scale = 2
    else:
        feats = DinoFeatures(args.backbone)
        in_dims = [768] * 4
        dummy_scale = 1
    decoder = SegDecoder(in_dims, hidden=cfg["model"]["hidden"], out_size=size)
    feats.to(device); decoder.to(device)

    lang, anchor_meta = None, None
    if args.arm == "a2":
        from .a2_language import build_language_branch
        lang, anchor_meta = build_language_branch(sum(in_dims), args, hidden=cfg["model"]["hidden"])
        lang.to(device)
        print(f"[A2] 语言分支：mode={anchor_meta['mode']} K={anchor_meta['n_prompts']} "
              f"d_text={anchor_meta['d_text']} shuffled={anchor_meta['shuffled']}")
    model = SegModel(feats, decoder, dummy_scale, lang, out_size=size).to(device)
    forward = model

    params = [p for p in decoder.parameters() if p.requires_grad]
    if lang is not None:
        params += [p for p in lang.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[model] trainable={n_train:,}")

    src = ROOT / cfg["data"]["source_root"]
    ds_tr = SegDataset(train_df, src, size, train=True)
    ds_va = SegDataset(val_df, src, size, train=False)
    ds_te = SegDataset(test_df, src, size, train=False)
    nw = 0 if args.smoke else cfg["input"]["num_workers"]
    tr_loader = DataLoader(ds_tr, batch_size=cfg["train"]["batch"], shuffle=True,
                           num_workers=nw, collate_fn=SegCollate(), drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=cfg["train"]["eval_batch"], shuffle=False,
                           num_workers=nw, collate_fn=SegCollate())
    te_loader = DataLoader(ds_te, batch_size=cfg["train"]["eval_batch"], shuffle=False,
                           num_workers=nw, collate_fn=SegCollate())

    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    total = cfg["train"]["max_steps"]; warm = cfg["train"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(s + 1, warm) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min((s - warm) / max(total - warm, 1), 1.0))))

    exp_id = args.exp_id or f"seg_{cfg['protocol']['name']}_{args.arm}_s{seed}"
    out_dir = OUT_ROOT / exp_id
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"产物目录已存在: {out_dir}（换 --exp-id 或加 --overwrite）")
    out_dir.mkdir(parents=True, exist_ok=True)
    if lang is not None:
        torch.save({"anchors": lang.anchors.detach().cpu(), "n_normal": lang.n_normal,
                    "meta": anchor_meta}, out_dir / "anchors.pt")

    history, best = [], -1.0
    step = 0
    t0 = time.time()
    while step < total:
        for x, gt, *_ in tr_loader:
            x, gt = x.to(device), gt.to(device)
            logits, aux = forward(x)
            loss = focal_dice(logits, gt)
            if aux is not None:                     # 辅助监督，帮 projector 对齐
                loss = loss + args.aux_weight * focal_dice(aux, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            opt.step(); sched.step(); step += 1
            if step % cfg["train"]["log_every"] == 0:
                print(f"  step {step}/{total} loss={loss.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step in cfg["train"]["eval_steps"] or step >= total:
                res = evaluate(model, va_loader, device, size)
                macro = res["macro"].get("apro_0.3", float("nan"))
                history.append({"step": step, "val_macro_apro": macro,
                                "val_macro_aupr": res["macro"].get("aupr", float("nan")),
                                "loss": float(loss)})
                print(f"[eval@{step}] val macro AUPRO={macro:.4f} "
                      f"P-AUPR={res['macro'].get('aupr', float('nan')):.4f} "
                      f"P-AUROC={res['macro'].get('auroc', float('nan')):.4f} "
                      f"I-AUROC={res['macro'].get('i_auroc', float('nan')):.4f}", flush=True)
                if macro == macro and macro > best:
                    best = macro
                    ck = {"decoder": decoder.state_dict(), "step": step, "arm": args.arm}
                    if lang is not None:
                        ck["lang"] = lang.state_dict()
                    torch.save(ck, out_dir / "best.pt")
                if step >= total:
                    break
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    ck = out_dir / "best.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        decoder.load_state_dict(state["decoder"])
        if lang is not None and "lang" in state:
            lang.load_state_dict(state["lang"])
    else:
        last = {"decoder": decoder.state_dict(), "step": step, "arm": args.arm}
        if lang is not None:
            last["lang"] = lang.state_dict()
        torch.save(last, out_dir / "last.pt")

    te = evaluate(model, te_loader, device, size)
    va = evaluate(model, va_loader, device, size)
    for name, r in (("val", va), ("test", te)):
        (out_dir / f"{name}_metrics.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{name}] macro AUPRO={r['macro'].get('apro_0.3', float('nan')):.4f} "
              f"AUPR={r['macro'].get('aupr', float('nan')):.4f}")

    (out_dir / "frozen_protocol.json").write_text(json.dumps({
        "exp_id": exp_id, "arm": args.arm, "protocol": cfg["protocol"],
        "model": {"backbone": args.backbone, "hidden": cfg["model"]["hidden"]},
        "train": cfg["train"], "split_sha256": split_sha,
        "best_val_macro_apro": best, "device": str(device),
        "anchors": anchor_meta,
        "note": "frozen backbone + trainable decoder; checkpoint by val macro AUPRO@0.3",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[seg] 完成 {out_dir}（{(time.time()-t0)/60:.1f} min, best AUPRO={best:.4f}）")


def main():
    ap = argparse.ArgumentParser(description="跨零件异常分割（A0: DINOv2 + 解码器）")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "seg_cross_part" / "protocols" / "seg_default.yaml")
    ap.add_argument("--protocol", type=str, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--arm", choices=["a0", "a2"], default="a0",
                    help="a0=纯 DINOv2；a2=+文本锚点语言分支")
    ap.add_argument("--text-encoder", choices=["clip", "random"], default="clip",
                    help="文本锚点来源：clip=CLIP 文本塔；random=随机向量（消融/冒烟）")
    ap.add_argument("--text-model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--shuffle-anchors", action="store_true",
                    help="打乱锚点与语义的对应（随机锚点消融）")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="语言分支辅助损失权重")
    ap.add_argument("--backbone", type=str, default="models/dinov2-base")
    ap.add_argument("--image-size", type=int, default=None,
                    help="覆盖输入分辨率（需为 14 的倍数：448 / 518 / 616）；默认取配置")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--exp-id", type=str, default=None)
    ap.add_argument("--limit-per-split", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-align", action="store_true",
                    help="只做几何对齐自检（不训练）")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="跳过开跑前复核（默认强制跑；跳过请在版本记录里写明理由）")
    args = ap.parse_args()

    if args.check_align:
        check_alignment()
        return

    # 没显式给 --config 时，按 --protocol 自动挑配置文件（避免"改了名字没换划分"）
    default_cfg = ROOT / "seg_cross_part" / "protocols" / "seg_default.yaml"
    if args.protocol and args.config == default_cfg:
        cand = ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"
        if cand.exists():
            args.config = cand
            print(f"[config] 按 --protocol 自动使用 {cand.name}")
    cfg = with_defaults(load_config(args.config))
    if args.protocol:
        cfg["protocol"]["name"] = args.protocol
    if args.image_size:
        sz = int(args.image_size)
        if sz % 14 != 0:
            raise SystemExit(f"--image-size 必须是 14 的倍数（DINOv2 patch=14），收到 {sz}")
        cfg["input"]["image_size"] = sz
        print(f"[config] 输入分辨率覆盖为 {sz}（patch 网格 {sz//14}×{sz//14}）")

    if args.smoke:
        cfg["train"]["max_steps"] = 4
        cfg["train"]["eval_steps"] = [2, 4]
        cfg["train"]["batch"] = 2
        cfg["train"]["eval_batch"] = 2
        cfg["train"]["log_every"] = 1
        cfg["input"]["num_workers"] = 0
        args.limit_per_split = args.limit_per_split or 8
        args.overwrite = True
        args.exp_id = args.exp_id or "smoke"
    # ---- 强制复核：不通过就不开跑 ----
    if not args.smoke and not args.skip_preflight:
        from .preflight import run_preflight
        print("=" * 60)
        print("开跑前复核（preflight）")
        print("=" * 60)
        if not run_preflight(cfg["protocol"]["name"], args.fold, args.config,
                             args.exp_id, sample=10):
            raise SystemExit("复核未通过，已阻止训练。修好后重跑；确需跳过用 --skip-preflight")
    elif args.smoke:
        print("[preflight] 冒烟模式跳过复核（不影响真跑）")

    train(cfg, args)


if __name__ == "__main__":
    main()
