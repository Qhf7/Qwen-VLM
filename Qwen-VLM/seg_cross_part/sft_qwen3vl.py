"""Qwen3-VL-8B 的 LoRA SFT：工业异常检测指令微调。

数据：build_instructions.py 产出的 JSONL（每条 = 一张图 + 一问一答）。
训练：冻结基座，只在语言模型的注意力/MLP 投影上挂 LoRA；损失只算在**答案**上（prompt 部分 mask 掉）。

用法：
    python -m seg_cross_part.sft_qwen3vl --data data/instructions_fold0.jsonl \
        --model models/Qwen3-VL-8B-Instruct --steps 500 --out artifacts/sft_fold0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import ROOT


class InsDataset(Dataset):
    def __init__(self, path: Path, source_root: Path, max_px: int = 448):
        self.rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        self.src = source_root
        self.max_px = max_px

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.src / r["image"]).convert("RGB")
        w, h = img.size
        s = self.max_px / max(w, h)
        if s < 1:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        img2 = None
        if r.get("image2"):
            img2 = Image.open(self.src / r["image2"]).convert("RGB")
            w2, h2 = img2.size
            s2 = self.max_px / max(w2, h2)
            if s2 < 1:
                img2 = img2.resize((max(1, int(w2 * s2)), max(1, int(h2 * s2))), Image.BICUBIC)
        return img, img2, r["conversations"]


def make_batch(img, img2, conv, proc):
    """单样本：直接返回 processor 输出 + 只对答案计损失的 labels。
    img2 不为空时放两张图（pair 任务）。"""
    content = [{"type": "image", "image": img}]
    if img2 is not None:
        content.append({"type": "image", "image": img2})
    content.append({"type": "text", "text": conv[0]["value"]})
    msgs_full = [{"role": "user", "content": list(content)},
                 {"role": "assistant", "content": [{"type": "text", "text": conv[1]["value"]}]}]
    msgs_prompt = [{"role": "user", "content": list(content)}]
    full = proc.apply_chat_template(msgs_full, tokenize=True, return_dict=True,
                                    return_tensors="pt")
    prompt = proc.apply_chat_template(msgs_prompt, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, :prompt["input_ids"].shape[1]] = -100
    out = {k: v for k, v in full.items()}
    out["labels"] = labels
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-VL LoRA SFT")
    ap.add_argument("--data", default="data/instructions_fold0.jsonl")
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-px", type=int, default=448)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", default="artifacts/sft_fold0")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                     bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = InsDataset(ROOT / args.data, ROOT / "Dataset", args.max_px)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, drop_last=True,
                    collate_fn=lambda b: b[0])          # 逐样本，绕过 PIL 的默认 collate
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = args.steps * args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model.train()
    step, micro, t0, run = 0, 0, time.time(), 0.0
    while step < args.steps:
        for img, img2, conv in dl:
            batch = make_batch(img, img2, conv, proc)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / args.accum).backward()
            run += loss.item()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"  step {step}/{args.steps} loss={run/args.accum:.4f} "
                          f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                run = 0.0
                if step % args.save_every == 0:
                    model.save_pretrained(out / f"ckpt-{step}")
                    print(f"  saved {out}/ckpt-{step}", flush=True)
                if step >= args.steps:
                    break
    model.save_pretrained(out / "final")
    proc.save_pretrained(out / "final")
    print(f"完成，adapter 在 {out}/final")


if __name__ == "__main__":
    main()
