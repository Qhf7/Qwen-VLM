"""大规模二选一（pair）评测：6 个测试类 × N 组，带 Wilson CI 与 bootstrap。

为什么只测这个任务：I-29 证明 VLM 的"是/否"绝对判断退化为恒答（准确率=异常占比），
而二选一不可能靠常数策略得分，是唯一被验证的真实信号。

用法：
    python -m seg_cross_part.eval_pair_big --adapter artifacts/sft_choice/final --n-per-class 200
    python -m seg_cross_part.eval_pair_big --adapter "" --n-per-class 50
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .common import ROOT, load_config, resolve_splits, with_defaults

Q = ("The first image and the second image are two parts of the same type. Exactly one of them "
     "has a manufacturing defect. Which one? Answer strictly: first or second.")


class Judge:
    def __init__(self, model_dir, adapter="", max_px=448):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.m = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
        if adapter:
            from peft import PeftModel
            self.m = PeftModel.from_pretrained(self.m, adapter).eval()
        self.max_px = max_px

    def _prep(self, im):
        w, h = im.size
        s = self.max_px / max(w, h)
        return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC) if s < 1 else im

    @torch.no_grad()
    def ask(self, imgs, q, n_new=6):
        content = [{"type": "image", "image": self._prep(im)} for im in imgs]
        content.append({"type": "text", "text": q})
        inp = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to("cuda")
        out = self.m.generate(**inp, max_new_tokens=n_new, do_sample=False)
        return self.proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-px", type=int, default=448)
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/eval_pair_big.json")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    df = pd.read_parquet(resolve_splits(cfg, args.fold)["test"])
    src = ROOT / cfg["data"]["source_root"]
    rng = random.Random(args.seed)
    judge = Judge(args.model, args.adapter, args.max_px)

    recs, t0 = [], time.time()
    for cls, g in df.groupby("class_name"):
        a = g[g.view_label == 1]
        nz = g[g.view_label == 0]
        k = args.n_per_class
        for i in range(k):
            ai = a.sample(1, random_state=rng.randint(0, 10 ** 9)).iloc[0]
            ni = nz.sample(1, random_state=rng.randint(0, 10 ** 9)).iloc[0]
            anom_first = rng.random() < 0.5
            p1, p2 = ((ai.image_path, ni.image_path) if anom_first
                      else (ni.image_path, ai.image_path))
            ans = judge.ask([Image.open(src / p1).convert("RGB"),
                             Image.open(src / p2).convert("RGB")], Q).lower()
            pf = ans.startswith("first")
            ok = (pf == anom_first) if (ans.startswith("first") or ans.startswith("second")) else False
            recs.append({"class": cls, "anom_first": anom_first, "ans": ans[:20], "ok": bool(ok)})
            if len(recs) % 100 == 0:
                print(f"  {len(recs)}  {(time.time()-t0)/len(recs):.2f}s/组  "
                      f"acc={np.mean([r['ok'] for r in recs]):.4f}", flush=True)

    oks = np.array([r["ok"] for r in recs])
    n = len(oks); k = int(oks.sum())
    lo, hi = wilson(k, n)
    prng = np.random.default_rng(0)
    boot = [prng.choice(oks, n, replace=True).mean() for _ in range(2000)]
    res = {"adapter": args.adapter or None, "max_px": args.max_px,
           "n_pairs": n, "acc": float(k / n),
           "wilson_ci": [float(lo), float(hi)],
           "boot_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "p_vs_chance": float((np.array(boot) <= 0.5).mean()),
           "per_class": {c: float(np.mean([r["ok"] for r in recs if r["class"] == c]))
                         for c in sorted({r["class"] for r in recs})},
           "ans_first_rate": float(np.mean([r["anom_first"] for r in recs]))}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "records": recs}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=2, ensure_ascii=False))
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
