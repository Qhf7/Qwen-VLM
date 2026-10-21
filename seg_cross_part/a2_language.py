"""A2 语言分支：把「异常/正常」的文本锚点投射到 DINOv2 的 patch 特征上。

核心思路（《项目说明》§5.2）：
    DINOv2 patch 特征  --projector-->  文本空间  --cos-->  锚点
    score(patch) = logsumexp(异常锚点) − logsumexp(正常锚点)

文本锚点有两种来源：
    clip   ：用 CLIP 文本塔编码一批"对象无关"的短语（A2-a，链路先打通）
    random ：随机向量（等价于随机锚点消融，也用于本地无权重冒烟）

注意：DINOv2 的特征空间与 CLIP 文本空间 **本来不对齐**，projector 就是在
seen classes 的 mask 监督下学这个对齐。这一点是 A2 的全部技术含量所在。
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 对象无关的锚点短语（不写具体零件类别；给类名的版本留给消融）
PROMPTS_NORMAL = [
    "a photo of a normal industrial part",
    "a photo of a clean manufactured component",
    "a photo of a part without any defect",
    "a photo of a flawless metal product",
    "a photo of a perfect industrial component",
]
PROMPTS_ANOMALY = [
    "a photo of a damaged industrial part",
    "a photo of a part with a scratch",
    "a photo of a part with a crack or a break",
    "a photo of a deformed or discolored component",
    "a photo of a defective manufactured part",
]


def prompts_sha256() -> str:
    text = "\n".join(PROMPTS_NORMAL + ["--"] + PROMPTS_ANOMALY)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
def build_anchors(mode: str = "random", model_id: str = "openai/clip-vit-large-patch14-336",
                  seed: int = 0, shuffle: bool = False) -> tuple[torch.Tensor, dict]:
    """返回 (anchors [K,d], meta)。前 len(PROMPTS_NORMAL) 个是正常锚点。"""
    prompts = PROMPTS_NORMAL + PROMPTS_ANOMALY
    if shuffle:                                   # 消融：词与语义错配
        g = torch.Generator().manual_seed(seed)
        prompts = [prompts[i] for i in torch.randperm(len(prompts), generator=g).tolist()]

    if mode == "clip":
        from transformers import CLIPTextModel, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(model_id)
        enc = CLIPTextModel.from_pretrained(model_id).eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            batch = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            out = enc(**batch)
            # 取 EOS 位置的隐状态（CLIP 文本塔的标准池化）
            eos = batch["input_ids"].argmax(dim=-1)
            emb = out.last_hidden_state[torch.arange(len(prompts)), eos]
        meta = {"mode": "clip", "model_id": model_id, "prompts_sha256": prompts_sha256(),
                "n_prompts": len(prompts), "shuffled": bool(shuffle),
                "d_text": int(emb.shape[1])}
        return F.normalize(emb.float(), dim=-1), meta

    # random：随机方向，同时作为本地冒烟与"随机锚点消融"的基线
    g = torch.Generator().manual_seed(seed)
    d = 768
    emb = torch.randn(len(prompts), d, generator=g)
    meta = {"mode": "random", "seed": seed, "prompts_sha256": prompts_sha256(),
            "n_prompts": len(prompts), "shuffled": bool(shuffle), "d_text": d}
    return F.normalize(emb, dim=-1), meta


# --------------------------------------------------------------------------
class LanguageBranch(nn.Module):
    """DINOv2 多层 patch 特征 → 文本空间 → 与锚点做余弦 → 异常 logit 图。"""

    def __init__(self, in_dim: int, anchors: torch.Tensor, hidden: int = 256,
                 n_normal: int = len(PROMPTS_NORMAL), scale: float = 10.0,
                 fuse: str = "add"):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, int(anchors.shape[1]), 1),
        )
        self.register_buffer("anchors", anchors.clone().float())
        self.n_normal = int(n_normal)
        self.scale = float(scale)
        self.fuse = str(fuse)
        if self.fuse == "gate":
            # 逐 patch 调制：门控由「特征 + 文本上下文」共同决定（输入相关，不会退化）
            self.gate = nn.Sequential(
                nn.Conv2d(in_dim + int(anchors.shape[1]), hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, in_dim, 1), nn.Sigmoid(),
            )

    def _sim(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(feats, dim=1)                       # (B, 4*768, h, w)
        z = F.normalize(self.proj(x), dim=1)              # (B, d, h, w)
        a = F.normalize(self.anchors, dim=1)              # (K, d)
        return torch.einsum("bdhw,kd->bkhw", z, a) * self.scale

    def modulate(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """用文本语义逐位置改写特征：f' = f ⊙ (1 + gate)。"""
        sim = self._sim(feats)                            # (B, K, h, w)
        attn = torch.softmax(sim, dim=1)
        a = F.normalize(self.anchors, dim=1)
        ctx = torch.einsum("bkhw,kd->bdhw", attn, a)      # 每个 patch 查回来的"语义"
        x = torch.cat(feats, dim=1)
        g = self.gate(torch.cat([x, ctx], dim=1))         # (B, 4*768, h, w)
        parts = torch.split(g, [f.shape[1] for f in feats], dim=1)
        return [f * (1.0 + gi) for f, gi in zip(feats, parts)]

    def forward(self, feats: list[torch.Tensor], out_size: int) -> torch.Tensor:
        sim = self._sim(feats)                            # (B, K, h, w) 余弦相似度
        norm = torch.logsumexp(sim[:, :self.n_normal], dim=1)
        anom = torch.logsumexp(sim[:, self.n_normal:], dim=1)
        logit = anom - norm
        logit = F.interpolate(logit.unsqueeze(1), size=(out_size, out_size),
                              mode="bilinear", align_corners=False)
        return logit.squeeze(1)


def build_language_branch(in_dim: int, args, hidden: int = 256) -> tuple[LanguageBranch, dict]:
    anchors, meta = build_anchors(mode=args.text_encoder, model_id=args.text_model,
                                  seed=args.seed if hasattr(args, "seed") else 0,
                                  shuffle=args.shuffle_anchors)
    fuse = getattr(args, "fuse", "add")
    meta["fuse"] = fuse
    return LanguageBranch(in_dim, anchors, hidden=hidden, fuse=fuse), meta
