"""SFT 后的 Qwen3-VL 评测：留出类上的"有没有缺陷"与"是什么缺陷"。

与训练时的指令模板一致（同一套问法），只在 Protocol C 的测试类上评，
所以能直接反映**跨零件泛化**，也能和零样本基线对比。

用法：
    # 零样本基线
    python -m seg_cross_part.eval_sft --adapter "" --n-per-class 30 --out artifacts/eval_zeroshot.json
    # SFT 后
    python -m seg_cross_part.eval_sft --adapter artifacts/sft_fold0/final --n-per-class 30 \
        --out artifacts/eval_sft.json
"""
from __future__ import annotations

import argparse
import json
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
Q_CLS = "Does this part have any manufacturing defect? Answer yes or no."
Q_TYPE = "What type of defect is visible on this part? Answer with the defect name."


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

    @torch.no_grad()
    def ask(self, img: Image.Image, q: str, n_new: int = 16) -> str:
        w, h = img.size
        s = self.max_px / max(w, h)
        if s < 1:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": q}]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to("cuda")
        out = self.model.generate(**inp, max_new_tokens=n_new, do_sample=False)
        return self.proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()


def parse_yes(t: str):
    s = t.strip().lower()
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return None


def parse_type(t: str, cands: list[str] | None = None):
    """用全 20 类词表解析（不限本类），返回命中的官方缺陷名。"""
    s = t.strip().lower()
    pool = cands if cands else list(DEFECT_WORDS)
    for c in pool:
        if DEFECT_WORDS.get(c, c.lower()) in s:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/eval_sft.json")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    df = pd.read_parquet(resolve_splits(cfg, args.fold)["test"])
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "defect_norm"])
    df = df.merge(pix, on="image_path", how="left")
    src = ROOT / cfg["data"]["source_root"]

    rows = []
    for cls, sub in df.groupby("class_name"):
        a = sub[sub.view_label == 1].head(args.n_per_class)
        n = sub[sub.view_label == 0].head(max(5, args.n_per_class // 2))
        rows.append(pd.concat([a, n]))
    df = pd.concat(rows).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"[eval] {len(df)} 张，{df.class_name.nunique()} 类，adapter={args.adapter or '无(零样本)'}")

    judge = Judge(args.model, args.adapter)
    recs, t0 = [], time.time()
    for i, r in df.iterrows():
        img = Image.open(src / r["image_path"]).convert("RGB")
        ans_cls = judge.ask(img, Q_CLS, 8)
        pred = parse_yes(ans_cls)
        gt = int(r["view_label"])
        rec = {"image": r["image_path"], "class": r["class_name"], "label": gt,
               "ans_cls": ans_cls, "pred_cls": pred, "ok_cls": (pred == gt) if pred is not None else False}
        if gt == 1 and isinstance(r["defect_norm"], str):
            own = sorted({str(x) for x in df[df.class_name == r["class_name"]]["defect_norm"].dropna()} - {""})
            ans_t = judge.ask(img, Q_TYPE, 12)
            pt = parse_type(ans_t)                      # 全词表解析
            rec.update({"ans_type": ans_t, "pred_type": pt,
                        "own_types": own,
                        "in_own": (pt in own) if pt else False,
                        "ok_type": (pt == r["defect_norm"]) if pt else False})
        recs.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)}  {(time.time()-t0)/(i+1):.2f}s/张", flush=True)

    ok = [r["ok_cls"] for r in recs]
    parsed = sum(r["pred_cls"] is not None for r in recs)
    types = [r for r in recs if "ok_type" in r]
    res = {
        "n": len(recs), "adapter": args.adapter or None,
        "cls_accuracy": float(np.mean(ok)),
        "cls_parse_rate": parsed / len(recs),
        "cls_accuracy_on_parsed": float(np.mean([r["ok_cls"] for r in recs if r["pred_cls"] is not None]))
        if parsed else None,
        "type_accuracy": float(np.mean([r["ok_type"] for r in types])) if types else None,
        "type_in_own_class_rate": float(np.mean([r["in_own"] for r in types])) if types else None,
        "type_parse_rate": sum(r["pred_type"] is not None for r in types) / max(len(types), 1),
        "pred_type_top": {k: int(v) for k, v in
                          pd.Series([r["pred_type"] for r in types]).value_counts().head(6).items()},
        "n_type": len(types),
        "per_class": {c: float(np.mean([r["ok_cls"] for r in recs if r["class"] == c]))
                      for c in sorted({r["class"] for r in recs})},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "samples": recs}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2, ensure_ascii=False))
    print("per_class:", {k: round(v, 3) for k, v in res["per_class"].items()})
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
