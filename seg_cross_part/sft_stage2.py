"""阶段二：projector（已对齐）+ LoRA 联合微调。

阶段一（I-28）证明：冻结 LLM、只训 44.8M projector 做模态对齐，就能把
图像级判断从 60.4% 提到 66.7%，且缺陷类型从 0% 提到 16.7%。

阶段二把两者接起来：
    冻结 DINOv2 / 冻结 LLM 主干
    ├─ projector（阶段一权重，可训）
    └─ LoRA（挂在 Qwen3VLTextModel 上，可训）
    → 联合优化，让"视觉已对齐"的表征在 SFT 中继续被利用。

用法：
    python -m seg_cross_part.sft_stage2 --projector artifacts/align_dino/proj.pt \
        --data data/instructions_fold0.jsonl --steps 300 --out artifacts/sft_stage2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .align_dino_llm import FULL, INSTR, Projector
from .common import ROOT
from .train_seg import DinoFeatures

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_x(path: Path, size=448) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, np.float32) / 255.0
    a = (a - IMG_MEAN) / IMG_STD
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--backbone", default="models/dinov2-base")
    ap.add_argument("--projector", default="artifacts/align_dino/proj.pt")
    ap.add_argument("--data", default="data/instructions_fold0.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr-proj", type=float, default=2e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=150)
    ap.add_argument("--out", default="artifacts/sft_stage2")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    for p in m.parameters():
        p.requires_grad_(False)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()

    proj = Projector().to("cuda")
    proj.load_state_dict(torch.load(ROOT / args.projector, map_location="cuda"))
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                     bias="none", task_type=None,   # 直接包 text model，不走 generate 包装
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    lm = get_peft_model(m.model.language_model, cfg)
    lm.train()
    emb = m.get_input_embeddings()
    head = m.lm_head
    dino = DinoFeatures(args.backbone).to("cuda").eval()

    n_proj = sum(p.numel() for p in proj.parameters())
    n_lora = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(f"[stage2] projector={n_proj:,}  LoRA={n_lora:,}  合计={n_proj+n_lora:,}")

    rows = [json.loads(l) for l in (ROOT / args.data).open(encoding="utf-8")]
    opt = torch.optim.AdamW([{"params": proj.parameters(), "lr": args.lr_proj},
                             {"params": [p for p in lm.parameters() if p.requires_grad],
                              "lr": args.lr_lora}])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    src = ROOT / "Dataset"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    step, t0, run = 0, time.time(), 0.0
    while step < args.steps:
        perm = np.random.permutation(len(rows))
        for i in perm:
            r = rows[i]
            q, a = r["conversations"][0]["value"], r["conversations"][1]["value"]
            x = load_x(src / r["image"]).to("cuda")
            with torch.no_grad():
                feats = dino(x)
            vis = proj(feats).to(torch.bfloat16)                 # (1,64,4096)
            full = tok(FULL.format(q=q, a=a), return_tensors="pt").input_ids.to("cuda")
            prompt = tok(INSTR.format(q=q), return_tensors="pt").input_ids.to("cuda")
            labels = full.clone()
            labels[:, :prompt.shape[1] - 1] = -100
            e = emb(full).to(torch.bfloat16)
            inp = torch.cat([vis, e], dim=1)
            lab = torch.cat([torch.full((1, vis.shape[1]), -100, device="cuda"), labels], dim=1)
            h = lm(inputs_embeds=inp).last_hidden_state
            logits = head(h).float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   lab[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(proj.parameters()) + list(lm.parameters()) if p.requires_grad], 1.0)
            opt.step(); sched.step()
            step += 1; run += loss.item()
            if step % 10 == 0:
                print(f"  step {step}/{args.steps} loss={run/10:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                run = 0.0
            if step % args.save_every == 0:
                torch.save(proj.state_dict(), out / f"proj-{step}.pt")
                lm.save_pretrained(out / f"lora-{step}")
                print(f"  saved {out}/*-{step}", flush=True)
            if step >= args.steps:
                break
    torch.save(proj.state_dict(), out / "proj.pt")
    lm.save_pretrained(out / "lora")
    print(f"完成：{out}/proj.pt 与 {out}/lora")


if __name__ == "__main__":
    main()
