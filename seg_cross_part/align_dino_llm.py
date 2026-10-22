"""阶段一：把 DINOv2 的 patch 特征对齐到 Qwen3-VL 的 LLM 词向量空间（LLaVA 式）。

为什么要这一步：I-27 的 SFT 让模型学会了"格式"，但缺陷类型准确率 12.8%（低于 4 选 1 的 25%），
说明**语言与局部视觉证据从未被绑定**。LLaVA 的做法正是先训一个 projector 把视觉特征
映射进 LLM 的 embedding 空间，再在此基础上做 SFT。

本脚本：
    冻结 DINOv2（32×32×768）与 Qwen3-VL 的 LLM（4096 维），只训 projector：
        Conv3×3(768→1024, s=2) → GELU → Conv3×3(1024→4096, s=2) → 8×8=64 个视觉 token
    输入序列 = [64 个视觉 token] + [文本 token]，损失只算在答案上。

用法：
    python -m seg_cross_part.align_dino_llm train --data data/instructions_fold0.jsonl --steps 500
    python -m seg_cross_part.align_dino_llm eval  --ckpt artifacts/align_dino/proj.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .common import ROOT
from .train_seg import DinoFeatures

INSTR = "Question: {q}\nAnswer:"
FULL = "Question: {q}\nAnswer: {a}\n"


class Projector(nn.Module):
    def __init__(self, in_dim=768, mid=1024, out_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(mid, out_dim, 3, stride=2, padding=1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1].float()        # 用 DINOv2 最后一层（语义最强）
        x = self.net(x.float())                       # (B,4096,8,8)
        return x.flatten(2).transpose(1, 2)           # (B,64,4096)


class Aligner:
    def __init__(self, model_dir: str, device="cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.tok = self.proc.tokenizer
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.m = m
        self.lm = m.model.language_model
        self.emb = m.get_input_embeddings()
        self.head = m.lm_head
        self.device = device

    # ---------- 组装序列 ----------
    def build(self, dino_feats, q: str, a: str | None = None):
        vis = None  # 由外部传入 projector 输出
        prompt = self.tok(INSTR.format(q=q), return_tensors="pt").input_ids.to(self.device)
        if a is None:
            return prompt
        full = self.tok(FULL.format(q=q, a=a), return_tensors="pt").input_ids.to(self.device)
        labels = full.clone()
        labels[:, :prompt.shape[1] - 1] = -100     # 只对答案（含 ':' 之后）计损失
        return full, labels, prompt.shape[1] - 1

    @torch.no_grad()
    def generate(self, proj, dino_feats, q: str, n_new: int = 12) -> str:
        vis = proj(dino_feats).to(torch.bfloat16)
        prompt = self.tok(INSTR.format(q=q), return_tensors="pt").input_ids.to(self.device)
        e = self.emb(prompt)
        ids = prompt
        for _ in range(n_new):
            inp = torch.cat([vis, self.emb(ids).to(torch.bfloat16)], dim=1)
            h = self.lm(inputs_embeds=inp).last_hidden_state
            nxt = self.head(h[:, -1:]).argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == self.tok.eos_token_id:
                break
        return self.tok.decode(ids[0, prompt.shape[1]:], skip_special_tokens=True).strip()


def load_imgs(rows, src: Path, size=448):
    out = []
    for r in rows:
        im = Image.open(src / r["image"]).convert("RGB").resize((size, size), Image.BICUBIC)
        a = np.asarray(im, np.float32) / 255.0
        a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        out.append(torch.from_numpy(a.transpose(2, 0, 1)))
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "evalq"])
    ap.add_argument("--data", default="data/instructions_fold0.jsonl")
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--backbone", default="models/dinov2-base")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default="artifacts/align_dino/proj.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = ROOT / "Dataset"
    rows = [json.loads(l) for l in (ROOT / args.data).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    dino = DinoFeatures(args.backbone).to("cuda").eval()
    al = Aligner(args.model)
    proj = Projector().to("cuda")

    if args.cmd == "train":
        opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.0)
        n = sum(p.numel() for p in proj.parameters())
        print(f"[align] projector trainable={n:,}  样本={len(rows)}")
        step, t0 = 0, time.time()
        while step < args.steps:
            idx = np.random.permutation(len(rows))[: args.batch]
            batch = [rows[i] for i in idx]
            x = load_imgs(batch, src).to("cuda")
            with torch.no_grad():
                feats = dino(x)
            vis = proj(feats).to(torch.bfloat16)
            losses = []
            for r in batch:
                conv = r["conversations"]
                q, a = conv[0]["value"], conv[1]["value"]
                full, labels, cut = al.build(feats[:1], q, a)
                e = torch.cat([vis[:1], al.emb(full).to(torch.bfloat16)], dim=1)
                lab = torch.cat([torch.full((1, vis.shape[1]), -100, device="cuda"), labels], dim=1)
                h = al.lm(inputs_embeds=e).last_hidden_state
                logits = al.head(h)
                shift_l, shift_g = logits[:, :-1], lab[:, 1:]
                losses.append(F.cross_entropy(
                    shift_l.reshape(-1, shift_l.shape[-1]).float(),
                    shift_g.reshape(-1), ignore_index=-100))
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step(); step += 1
            if step % 20 == 0:
                print(f"  step {step}/{args.steps} loss={loss.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        out = ROOT / args.ckpt
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(proj.state_dict(), out)
        print(f"已保存 projector -> {out}")
    elif args.cmd == "eval":
        proj.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cuda"))
        proj.eval()
        for r in rows[:5]:
            x = load_imgs([r], src).to("cuda")
            with torch.no_grad():
                f = dino(x)
            print(f"问题: {r['conversations'][0]['value'][:70]}...")
            print(f"标准: {r['conversations'][1]['value']}")
            print(f"预测: {al.generate(proj, f, r['conversations'][0]['value'])}\n", flush=True)

    else:   # evalq：在 Protocol C 测试类上评 yes/no 与缺陷类型
        import pandas as pd
        from .common import MANIFEST_PIXEL, load_config, resolve_splits, with_defaults
        proj.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cuda")); proj.eval()
        cfg = with_defaults(load_config(ROOT / "seg_cross_part/protocols/protocol_c_unseen_category.yaml"))
        df = pd.read_parquet(resolve_splits(cfg, 0)["test"])
        df = df.merge(pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "defect_norm"]),
                      on="image_path", how="left")
        QC = "Does this part have any manufacturing defect? Answer yes or no."
        QT = "What type of defect is visible on this part? Answer with the defect name."
        rows_t = []
        for cls, g in df.groupby("class_name"):
            a = g[g.view_label == 1].head(20); nz = g[g.view_label == 0].head(10)
            rows_t.extend(a.to_dict("records")); rows_t.extend(nz.to_dict("records"))
        ok = n = tok_ok = tok_n = 0
        for r in rows_t:
            x = load_imgs([{"image": r["image_path"]}], src).to("cuda")
            with torch.no_grad():
                f = dino(x)
            ans = al.generate(proj, f, QC, 6).split("\n")[0].lower()
            pred = 1 if ans.startswith("yes") else (0 if ans.startswith("no") else None)
            ok += (pred == int(r["view_label"])); n += 1
            if int(r["view_label"]) == 1 and isinstance(r["defect_norm"], str):
                at = al.generate(proj, f, QT, 8).split("\n")[0].lower()
                tok_ok += (r["defect_norm"].lower().replace("or", " or ") in at
                           or at.startswith(r["defect_norm"].lower()[:6]))
                tok_n += 1
        print(json.dumps({"cls_acc": ok / max(n, 1), "n": n,
                          "type_acc": tok_ok / max(tok_n, 1), "n_type": tok_n}, indent=2))


if __name__ == "__main__":
    main()
