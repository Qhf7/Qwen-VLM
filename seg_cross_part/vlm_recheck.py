"""VLM 候选区域复核：DINOv2 定位 + Qwen3-VL 判断。

动机：前面 8 次实验证明 CLIP 文本先验无效，但 VLM（Qwen3-VL）**从未被测过**。
CLIP 的用法是"文本向量 × 视觉特征做余弦"，要求两个空间预先对齐；VLM 的正确用法是
**当推理器**——给它一块候选区域，让它判断"这里有没有缺陷"。

流程：
    1. DINOv2 少样本参考（= B1）产生 32×32 异常图；
    2. 取图上 top-K 个非极大值峰，每个峰扩成 112×112 的候选框；
    3. 把候选框裁出来喂给 Qwen3-VL，问 yes/no；
    4. 用 VLM 的判定去复核 DINOv2 的候选（图像级重排 / 区域级否决）。

评测：
    - 图像级：B1 的 max 分数 vs VLM P(yes) vs 二者融合（AUROC/AUPR）
    - 像素级：被 VLM 否决的区域置零后的 AUPRO（只在子集上跑，作趋势参考）

用法：
    python -m seg_cross_part.vlm_recheck --fold 0 --per-class 50 --topk 3 --limit 0
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
from .metrics import auroc, aupr
from .winclip_b2 import IMG_MEAN, IMG_STD, DinoRef, _to_tensor

PROMPT = (
    "You are an industrial quality inspector. This is a close-up crop of a manufactured part "
    "(metal or plastic component). Decide whether this crop shows a manufacturing DEFECT, "
    "such as scratch, crack, deformation, discoloration, oil stain, contamination, "
    "missing part, broken wire, burr, pit or burn mark.\n"
    "Ignore normal texture, reflections, machining marks and shadows.\n"
    "Answer strictly with one word: yes or no."
)


PROMPT_CMP = (
    "You are an industrial quality inspector. The first two images are crops of a NORMAL part. "
    "The last image is a candidate crop from the same type of part.\n"
    "Compare them: does the candidate show a manufacturing defect that is absent in the normal crops "
    "(scratch, crack, deformation, discoloration, stain, missing part, broken wire, burr, pit, burn mark)?\n"
    "Answer strictly with one word: normal or defective."
)


def peaks_from_map(m: np.ndarray, k: int = 3, min_dist: int = 3) -> list[tuple[int, int, float]]:
    """在 32×32 异常图上取 top-k 非极大值峰，返回 (row, col, score)。"""
    work = m.copy()
    out = []
    for _ in range(k):
        idx = int(np.argmax(work))
        r, c = divmod(idx, work.shape[1])
        out.append((r, c, float(work[r, c])))
        r0, r1 = max(0, r - min_dist), min(work.shape[0], r + min_dist + 1)
        c0, c1 = max(0, c - min_dist), min(work.shape[1], c + min_dist + 1)
        work[r0:r1, c0:c1] = -1.0
    return out


def crop_around(img: Image.Image, r: int, c: int, grid: int, side_ratio: float = 0.25,
                out_size: int = 448) -> Image.Image:
    W, H = img.size
    cell = 1.0 / grid
    cx, cy = (c + 0.5) * cell * W, (r + 0.5) * cell * H
    side = side_ratio * max(W, H)
    x0 = int(max(0, min(W - side, cx - side / 2)))
    y0 = int(max(0, min(H - side, cy - side / 2)))
    return img.crop((x0, y0, int(x0 + side), int(y0 + side))).resize((out_size, out_size), Image.BICUBIC)


class VLMJudge:
    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_dir)
        dt = torch.bfloat16 if "8B" in model_dir else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=dt, device_map=device).eval()
        self.device = device

    @torch.no_grad()
    def ask_compare(self, refs: list[Image.Image], cand: Image.Image) -> float:
        """对照式提问：给 2 张正常参考裁剪，问候选是否 defective。返回 1/0/nan。"""
        content = []
        for r in refs:
            content.append({"type": "image", "image": r})
        content.append({"type": "image", "image": cand})
        content.append({"type": "text", "text": PROMPT_CMP})
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True, return_dict=True,
                                               return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        text = self.proc.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip().lower()
        if "defective" in text:
            return 1.0
        if "normal" in text:
            return 0.0
        return float("nan")

    @torch.no_grad()
    def ask_yes(self, img: Image.Image) -> float:
        """返回 P(yes)；解析失败返回 nan。"""
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": PROMPT}]}]
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True, return_dict=True,
                                               return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.proc.tokenizer.decode(gen, skip_special_tokens=True).strip().lower()
        if text.startswith("yes"):
            return 1.0
        if text.startswith("no"):
            return 0.0
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM 候选区域复核")
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--backbone", default="models/dinov2-base")
    ap.add_argument("--vlm", default="models/Qwen3-VL-2B-Instruct")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=50, help="每类抽多少张（0=全量）")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 张（冒烟）")
    ap.add_argument("--mode", choices=["yesno", "compare"], default="yesno")
    ap.add_argument("--vlm-size", type=int, default=1024,
                    help="候选裁剪送进 VLM 的分辨率（448 会抹掉小缺陷，见 I-24）")
    ap.add_argument("--out", default="artifacts/seg_cross_part/vlm_recheck")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    df = pd.read_parquet(splits[args.split])
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "mask_path"])
    df = df.merge(pix, on="image_path", how="left")
    src = ROOT / cfg["data"]["source_root"]

    rng = np.random.default_rng(0)
    if args.per_class:
        keep = []
        for cls, sub in df.groupby("class_name"):
            # 每类按 6:4 抽异常/正常，保证两类都有
            anom = sub[sub.view_label == 1]
            norm = sub[sub.view_label == 0]
            na = min(len(anom), int(args.per_class * 0.6))
            nn = min(len(norm), args.per_class - na)
            keep.append(anom.sample(na, random_state=0))
            keep.append(norm.sample(nn, random_state=0))
        df = pd.concat(keep).sample(frac=1.0, random_state=0).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    dino = DinoRef(args.backbone, device)
    judge = VLMJudge(args.vlm, device)

    rows = []
    t0 = time.time()
    for cls, sub in df.groupby("class_name"):
        refs = df[(df.class_name == cls) & (df.view_label == 0)].head(args.shots)
        ref_paths = set(refs["image_path"])
        mem = []
        for _, r in refs.iterrows():
            im = Image.open(src / r["image_path"])
            mem.append(dino.feats(_to_tensor(im, dino.size, IMG_MEAN, IMG_STD).unsqueeze(0).to(device))[0])
        mem = dino.build_memory(torch.stack(mem)) if mem else None

        for _, r in sub.iterrows():
            if r["image_path"] in ref_paths:
                continue                              # 参考图不参与评测
            im = Image.open(src / r["image_path"])
            x = _to_tensor(im, dino.size, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
            m = dino.score(dino.feats(x)[0], mem) if mem is not None else np.zeros((32, 32), np.float32)
            pk = peaks_from_map(m, k=args.topk)
            yes = []
            # 对照式提问需要的正常参考裁剪（取该类参考图的中心区域）
            ref_crops = []
            if args.mode == "compare":
                for _, rr in refs.head(2).iterrows():
                    rim = Image.open(src / rr["image_path"])
                    ref_crops.append(crop_around(rim, m.shape[0] // 2, m.shape[1] // 2,
                                                 m.shape[0], out_size=args.vlm_size))
            for (pr, pc, ps) in pk:
                crop = crop_around(im, pr, pc, m.shape[0], out_size=args.vlm_size)
                yes.append(judge.ask_compare(ref_crops, crop) if args.mode == "compare"
                           else judge.ask_yes(crop))
            rows.append({"image_path": r["image_path"], "class": cls,
                         "label": int(r["view_label"]), "b1_max": float(m.max()),
                         "b1_top": [round(p[2], 4) for p in pk],
                         "vlm": yes})
            if len(rows) % 20 == 0:
                print(f"  {len(rows)}/{len(df)}  {(time.time()-t0)/len(rows):.2f}s/张", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "preds.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    y = np.array([r["label"] for r in rows])
    b1 = np.array([r["b1_max"] for r in rows])
    vlm_max = np.array([np.nanmax(r["vlm"]) if len(r["vlm"]) else np.nan for r in rows])
    vlm_mean = np.array([np.nanmean(r["vlm"]) if len(r["vlm"]) else np.nan for r in rows])
    vlm_any = np.array([1.0 if np.nansum(r["vlm"]) > 0 else 0.0 for r in rows])
    combo = b1 * (0.5 + 0.5 * vlm_any)     # 低分被 VLM 拉高，高分被否决则打折

    def rep(name, s):
        ok = ~np.isnan(s)
        print(f"  {name:14s} AUROC={auroc(y[ok], s[ok]):.4f}  AUPR={aupr(y[ok], s[ok]):.4f}  n={ok.sum()}")

    print(f"\n[图像级] n={len(rows)} 异常={y.sum()}")
    rep("B1 max", b1)
    rep("VLM max", vlm_max)
    rep("VLM mean", vlm_mean)
    rep("VLM any", vlm_any)
    rep("B1×VLM", combo)
    res = {"n": len(rows), "n_anom": int(y.sum()),
           "auroc": {"b1": float(auroc(y, b1)), "vlm_max": float(auroc(y[~np.isnan(vlm_max)], vlm_max[~np.isnan(vlm_max)])),
                     "vlm_any": float(auroc(y, vlm_any)), "combo": float(auroc(y, combo))},
           "aupr": {"b1": float(aupr(y, b1)), "vlm_any": float(aupr(y, vlm_any)), "combo": float(aupr(y, combo))}}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(f"\n已写出 {out}/preds.json 与 summary.json")


if __name__ == "__main__":
    main()
