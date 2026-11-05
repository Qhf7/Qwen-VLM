"""评测二选一判别能力（pair / type2），可对比零样本与 LoRA-SFT。

pair ：两张同类图（一正常一异常）→ 问"哪张有缺陷"，答案 first/second，位置随机。
type2：一张异常图 + 两种缺陷名（一对一错）→ 问"是 A 还是 B"。

用法：
    python -m seg_cross_part.eval_choice --adapter ""            # 零样本
    python -m seg_cross_part.eval_choice --adapter artifacts/sft_choice/final
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

from .common import MANIFEST_PIXEL, ROOT, load_config, resolve_splits, with_defaults

DEFECT_WORDS = {
    "OilStain": "oil stain", "GelatinousContaminant": "gelatinous contaminant",
    "FiberorBurrs": "fiber or burrs", "CharacterOmission": "character omission",
    "UnevenCoating": "uneven coating", "BrokenWire": "broken wire",
    "TwistedWire": "twisted wire", "MetallicContaminant": "metallic contaminant",
    "ImpactDamage": "impact damage", "MissingPart": "missing part",
    "BurnMark": "burn mark", "Discoloration": "discoloration", "Deformation": "deformation",
    "Scratch": "scratch", "Damage": "damage", "Fracture": "fracture",
    "Pit": "pit", "Warpage": "warpage", "Porosity": "porosity", "Abrasion": "abrasion",
}
Q_PAIR = ("The first image and the second image are two parts of the same type. Exactly one of "
          "them has a manufacturing defect. Which one? Answer strictly: first or second.")
Q_TYPE2 = ("Look at this part. Is the defect a {a} or a {b}? "
           "Answer with one of the two names only.")


class Judge:
    def __init__(self, model_dir: str, adapter: str = "", max_px: int = 448):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.max_px = max_px

    def _prep(self, img):
        w, h = img.size
        s = self.max_px / max(w, h)
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC) if s < 1 else img

    @torch.no_grad()
    def ask(self, imgs: list[Image.Image], q: str, n_new: int = 6) -> str:
        content = [{"type": "image", "image": self._prep(im)} for im in imgs]
        content.append({"type": "text", "text": q})
        msgs = [{"role": "user", "content": content}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to("cuda")
        out = self.model.generate(**inp, max_new_tokens=n_new, do_sample=False)
        return self.proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-px", type=int, default=448)
    ap.add_argument("--n", type=int, default=180, help="每类抽多少张异常图（pair/type2 各用一次）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="artifacts/eval_choice.json")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    df = pd.read_parquet(resolve_splits(cfg, args.fold)["test"])
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "defect_norm"])
    df = df.merge(pix, on="image_path", how="left")
    src = ROOT / cfg["data"]["source_root"]
    rng = random.Random(args.seed)

    judge = Judge(args.model, args.adapter, args.max_px)
    pair_ok = pair_n = type_ok = type_n = 0
    t0 = time.time()
    for cls, g in df.groupby("class_name"):
        a = g[g.view_label == 1].head(args.n // max(1, df.class_name.nunique()))
        nz = g[g.view_label == 0]
        own = sorted({str(x) for x in g.defect_norm.dropna()} - {""})
        for _, r in a.iterrows():
            # ---- pair ----
            if len(nz):
                ni = nz.sample(1, random_state=rng.randint(0, 10 ** 9)).iloc[0]
                anom_first = rng.random() < 0.5
                p1, p2 = ((r.image_path, ni.image_path) if anom_first
                          else (ni.image_path, r.image_path))
                ims = [Image.open(src / p1).convert("RGB"), Image.open(src / p2).convert("RGB")]
                ans = judge.ask(ims, Q_PAIR).lower()
                pick_first = ans.startswith("first")
                ok = (pick_first == anom_first) if (ans.startswith("first") or ans.startswith("second")) else False
                pair_ok += ok; pair_n += 1
            # ---- type2 ----
            if len(own) >= 2 and isinstance(r.defect_norm, str) and r.defect_norm in own:
                others = [x for x in own if x != r.defect_norm]
                wrong = rng.choice(others)
                cands = [r.defect_norm, wrong]; rng.shuffle(cands)
                q = Q_TYPE2.format(a=DEFECT_WORDS.get(cands[0], cands[0].lower()),
                                   b=DEFECT_WORDS.get(cands[1], cands[1].lower()))
                im = Image.open(src / r.image_path).convert("RGB")
                ans = judge.ask([im], q).lower()
                hit = [c for c in cands if DEFECT_WORDS.get(c, c.lower()) in ans]
                ok = bool(hit) and hit[0] == r.defect_norm
                type_ok += ok; type_n += 1
        print(f"  {cls:26s} pair {pair_ok}/{pair_n}  type2 {type_ok}/{type_n}  "
              f"({(time.time()-t0):.0f}s)", flush=True)

    res = {"adapter": args.adapter or None,
           "pair_acc": pair_ok / max(pair_n, 1), "pair_n": pair_n,
           "type2_acc": type_ok / max(type_n, 1), "type2_n": type_n}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
