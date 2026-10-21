"""B2 基线：WinCLIP（语言）+ DINOv2（视觉参考关联）= WinCLIP+ 的 DINOv2 版。

论文：Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023.

为什么这样结合
--------------
WinCLIP 的核心是「窗口 CLS 特征 + 文本原型」——它成立的前提是 **CLIP 的 CLS 与文本对齐**。
DINOv2 是纯自监督，没有语言空间，**不能**直接和文本向量做余弦。
所以 B2 的分工是：
    语言侧：CLIP 滑窗 CLS（2×2 / 3×3 / 图像级）→ 调和平均 → M_lang
    视觉侧：DINOv2 patch 特征 + K 张正常参考图 → 最近邻关联（式 4）→ M_ref
最后按 WinCLIP+ 的方式融合。

与论文的差异（必须写进论文的 Implementation Details）
----------------------------------------------------
1. 论文的视觉记忆来自 **CLIP** 的 small/mid 窗口特征与 patch token；
   这里换成 **DINOv2** 的 patch 特征（这正是我们要测的"换主干"）。
2. 论文用 ViT-B/16+（LAION），我们用 OpenAI CLIP ViT-L/14@336（本地权重）。
3. 状态词/模板是我们按论文 4.1 的思路自行整理的（论文附录未给出完整列表）。

用法
----
    python -m seg_cross_part.winclip_b2 --protocol protocol_c_unseen_category --fold 0 \
        --clip models/clip-vit-large-patch14-336 --shots 4 --limit 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .common import MANIFEST_PIXEL, ROOT, load_config, resolve_splits, with_defaults
from .metrics_pixel import load_mask
from .stats_pixel import PixelStats, image_hist, image_pro

# --------------------------------------------------------------------------
# 4.1 Compositional Prompt Ensemble（状态词 × 模板）
# --------------------------------------------------------------------------
STATE_NORMAL = ["normal", "flawless", "perfect", "intact", "undamaged"]
STATE_ANOMALY = ["anomalous", "damaged", "defective", "broken", "cracked",
                 "scratched", "deformed", "contaminated"]
TEMPLATES = [
    "a photo of a {c}.",
    "a photo of a {c} for visual inspection.",
    "a close-up photo of a {c}.",
    "a bright photo of a {c}.",
    "a photo of the {c}.",
]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def _to_tensor(img: Image.Image, size: int, mean, std) -> torch.Tensor:
    im = img.convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1))


def _resize_map(m: np.ndarray, size: int) -> np.ndarray:
    """把 (h,w) 的概率图双线性放大到 size×size。"""
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _harmonic(*arrays: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    inv = sum(1.0 / np.maximum(a, eps) for a in arrays)
    return len(arrays) / np.maximum(inv, eps)


class ClipWindowScorer:
    """窗口 CLS 特征 → 文本原型相似度 → 多尺度异常图。"""

    def __init__(self, model_dir: str, device="cuda"):
        from transformers import CLIPModel, CLIPTokenizer
        self.device = device
        self.clip = CLIPModel.from_pretrained(model_dir).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(model_dir)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.vm = self.clip.vision_model
        self.size = 336
        self.grid = 24                      # 336 / 14
        self.dim = self.clip.config.projection_dim

    # ---- 文本原型 ----
    @torch.no_grad()
    def text_prototypes(self, obj_word: str = "object") -> torch.Tensor:
        texts = [t.format(c=f"{w} {obj_word}") for w in STATE_NORMAL + STATE_ANOMALY
                 for t in TEMPLATES]
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # transformers 5.x 的 get_text_features 可能返回 ModelOutput，这里直接走 text_model
        txt = self.clip.text_model(**batch)
        emb = self.clip.text_projection(txt.pooler_output)
        emb = F.normalize(emb, dim=-1)
        n_norm = len(STATE_NORMAL) * len(TEMPLATES)
        proto = torch.stack([emb[:n_norm].mean(0), emb[n_norm:].mean(0)])
        return F.normalize(proto, dim=-1)

    # ---- 窗口 CLS ----
    def _window_tokens(self, x: torch.Tensor):
        emb = self.vm.embeddings
        patch = emb.patch_embedding(x)                       # (B,d,24,24)
        B, C, H, W = patch.shape
        flat = patch.flatten(2).transpose(1, 2)              # (B,576,d)
        pos = emb.position_embedding.weight
        flat = flat + pos[1:].unsqueeze(0)
        cls = (emb.class_embedding + pos[0]).expand(B, 1, -1)
        return torch.cat([cls, flat], 1), H, W               # (B,1+HW,d)

    @torch.no_grad()
    def maps(self, x: torch.Tensor, proto: torch.Tensor, stride: int = 2) -> np.ndarray:
        """返回 24×24 的语言分支异常概率图。"""
        tokens, H, W = self._window_tokens(x)
        B = tokens.shape[0]
        out = []
        for k in (2, 3):
            idx = []
            for i in range(0, H - k + 1, stride):
                for j in range(0, W - k + 1, stride):
                    win = [(i + di) * W + (j + dj) for di in range(k) for dj in range(k)]
                    idx.append([0] + [1 + t for t in win])
            idx = torch.tensor(idx, device=x.device)          # (N, k*k+1)
            seq = tokens[:, idx]                              # (B,N,k*k+1,d)
            seq = seq.reshape(B * idx.shape[0], idx.shape[1], -1)
            hs = self.vm.encoder(inputs_embeds=seq).last_hidden_state[:, 0]
            hs = self.vm.post_layernorm(hs)
            hs = self.clip.visual_projection(hs)
            hs = F.normalize(hs, dim=-1)
            sim = hs @ proto.T                                # (B*N, 2)
            p_anom = torch.softmax(sim * self.clip.logit_scale.exp(), dim=-1)[:, 1]
            p_anom = p_anom.reshape(B, -1).float().cpu().numpy()
            out.append(self._agg(p_anom, H, W, k, stride))
        # 图像级：整图的 CLS
        hs = self.vm(x).pooler_output
        hs = self.clip.visual_projection(hs)
        hs = F.normalize(hs, dim=-1)
        p_img = torch.softmax((hs @ proto.T) * self.clip.logit_scale.exp(), -1)[:, 1]
        p_img = p_img.float().cpu().numpy()[:, None, None]
        out.append(np.broadcast_to(p_img.squeeze(), (H, W)).copy())
        return _harmonic(*out)

    @staticmethod
    def _agg(scores: np.ndarray, H: int, W: int, k: int, stride: int = 1) -> np.ndarray:
        """重叠窗口的调和平均（论文式 3）。"""
        inv = np.zeros((H, W))
        cnt = np.zeros((H, W))
        n = 0
        for i in range(0, H - k + 1, stride):
            for j in range(0, W - k + 1, stride):
                inv[i:i + k, j:j + k] += 1.0 / np.maximum(scores[:, n], 1e-6)
                cnt[i:i + k, j:j + k] += 1
                n += 1
        m = cnt / np.maximum(inv, 1e-6)
        m[cnt == 0] = 1.0          # 未被窗口覆盖的格子先置"非异常"，避免除零放大
        return m


class DinoRef:
    """DINOv2 patch 特征 + K 张正常参考图的最近邻关联（论文式 4）。"""

    def __init__(self, backbone: str, device="cuda", size: int = 448):
        from transformers import Dinov2Model
        self.device = device
        self.size = size
        self.model = Dinov2Model.from_pretrained(backbone).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def feats(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        f = hs[12][:, 1:, :]                       # (B,1024,768) 去 CLS
        return F.normalize(f, dim=-1)

    def build_memory(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (K,1024,d) → (M,d) 展平的参考记忆。"""
        return feats.reshape(-1, feats.shape[-1])

    @torch.no_grad()
    def score(self, q: torch.Tensor, mem: torch.Tensor) -> np.ndarray:
        """q: (1024,d) → 32×32 异常图（0.5*(1-cos) 的最近邻）。"""
        sim = q @ mem.T                            # (1024, M)
        best = sim.max(dim=1).values
        m = 0.5 * (1.0 - best)
        g = int(round(math.sqrt(q.shape[0])))
        return m.reshape(g, g).float().cpu().numpy()


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="B2：WinCLIP(语言) + DINOv2(视觉参考)")
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--clip", default="models/clip-vit-large-patch14-336")
    ap.add_argument("--backbone", default="models/dinov2-base")
    ap.add_argument("--shots", type=int, default=4, help="每类正常参考图数量（0=不用视觉侧）")
    ap.add_argument("--fuse", choices=["harmonic", "mean", "max", "ref"], default="harmonic",
                    help="ref=只用 DINOv2 参考关联（视觉下界 B1，不加载 CLIP，快很多）")
    ap.add_argument("--obj-word", default="object", choices=["object", "class"])
    ap.add_argument("--win-stride", type=int, default=2,
                    help="窗口滑动步长（1=论文设定但很慢；2 约快 4 倍）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    df = pd.read_parquet(splits[args.split])
    pix = pd.read_parquet(MANIFEST_PIXEL,
                          columns=["image_path", "mask_path", "mask_valid", "area_bucket"])
    df = df.merge(pix, on="image_path", how="left")
    src = ROOT / cfg["data"]["source_root"]
    if args.limit:
        df = df.groupby("class_name", group_keys=False).head(max(2, args.limit // 6))

    clip_sc = None if args.fuse == "ref" else ClipWindowScorer(args.clip, device)
    dino = DinoRef(args.backbone, device)
    rng = np.random.default_rng(args.seed)
    classes = sorted(df["class_name"].unique())
    t0 = time.time()
    texts = {}

    stats_by_cls = {}
    for ci, cls in enumerate(classes):
        sub = df[df["class_name"] == cls].reset_index(drop=True)
        normals = sub[sub["view_label"] == 0].reset_index(drop=True)
        ref_idx = []
        if args.shots > 0 and len(normals):
            ref_idx = rng.choice(len(normals), size=min(args.shots, len(normals)), replace=False)
        # 参考记忆
        mem = None
        if args.shots > 0 and len(ref_idx):
            feats = []
            for i in ref_idx:
                im = Image.open(src / normals.iloc[i]["image_path"])
                x = _to_tensor(im, dino.size, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
                feats.append(dino.feats(x)[0])
            mem = dino.build_memory(torch.stack(feats))
        ref_paths = {normals.iloc[i]["image_path"] for i in ref_idx}

        proto = None if clip_sc is None else clip_sc.text_prototypes(
            sub["class_name"].iloc[0].replace("_", " ") if args.obj_word == "class" else "object")

        st = PixelStats()
        for _, r in sub.iterrows():
            if r["image_path"] in ref_paths:
                continue                                     # 参考图不参与评测
            im = Image.open(src / r["image_path"])
            if clip_sc is not None:
                xc = _to_tensor(im, clip_sc.size, CLIP_MEAN, CLIP_STD).unsqueeze(0).to(device)
                m_lang = _resize_map(clip_sc.maps(xc, proto, stride=args.win_stride), dino.size)
            else:
                m_lang = None
            if mem is not None:
                xd = _to_tensor(im, dino.size, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
                m_ref = dino.score(dino.feats(xd)[0], mem)
                m_ref = _resize_map(m_ref, dino.size)
                if m_lang is None:
                    score = m_ref
                elif args.fuse == "harmonic":
                    score = _harmonic(m_lang, m_ref)
                elif args.fuse == "mean":
                    score = 0.5 * (m_lang + m_ref)
                else:
                    score = np.maximum(m_lang, m_ref)
            else:
                score = m_lang
            score = np.clip(score, 1e-6, 1.0).astype(np.float32)

            lbl = int(r["view_label"])
            gt = load_mask(src / r["mask_path"], score.shape[0]) if lbl else np.zeros_like(score, bool)
            neg, pos = image_hist(score, gt)
            pro, nreg = image_pro(score, gt)
            st.add(group=str(r["group_path"]), cls=cls, area=str(r["area_bucket"]),
                   label=lbl, img_score=float(score.max()), neg=neg, pos=pos, pro=pro, n_regions=nreg)
        stats_by_cls[cls] = st
        texts[cls] = st.metrics()
        print(f"[{ci+1}/{len(classes)}] {cls:28s} AUPRO={texts[cls]['apro_0.3']:.4f} "
              f"n={len(st)} shots={len(ref_idx)} ({time.time()-t0:.0f}s)", flush=True)

    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / "seg_cross_part" / \
        f"b2_{args.protocol.split('_')[1]}_f{args.fold}_k{args.shots}_{args.fuse}"
    out_dir.mkdir(parents=True, exist_ok=True)
    allst = PixelStats()
    for st in stats_by_cls.values():
        for i in range(len(st)):
            allst.add(group=st.group[i], cls=st.cls[i], area=st.area[i], label=st.label[i],
                      img_score=st.img_score[i], neg=st.neg[i], pos=st.pos[i], pro=st.pro[i],
                      n_regions=st.n_regions[i])
    allst.save(out_dir / f"stats_{args.split}.npz")
    res = {"overall": allst.metrics(), "by_class": {c: st.metrics() for c, st in stats_by_cls.items()},
           "by_area": {a: allst.metrics(idx=np.where(np.array(allst.area) == a)[0])
                       for a in sorted(set(allst.area))},
           "macro": {k: float(np.mean([m[k] for m in texts.values()]))
                     for k in ("apro_0.3", "aupr", "auroc", "i_auroc")},
           "config": {k: v for k, v in vars(args).items()}}
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[macro] " + " ".join(f"{k}={v:.4f}" for k, v in res["macro"].items()))
    print(f"已写出 {out_dir}/eval_{args.split}.json 与 stats_{args.split}.npz")


if __name__ == "__main__":
    main()
