"""指令数据构造：把 Real-IAD 的标注转成 Qwen3-VL 的 SFT 指令样本。

设计目标（对应"大模型基座算法"的工程要点）：
    1. 多任务指令：判断 / 分类 / 定位 / 描述 / 推理链，同一张图可派生多条；
    2. 指令多样性：每个任务准备多个问法模板，避免模型只记住一种句式；
    3. 类别平衡：20 种缺陷 × 正常，按任务分层采样；
    4. 可审计：每条样本带 meta（来源 split、类别、缺陷名、面积档），并输出统计报告；
    5. 防泄漏：训练样本只从 Protocol C 的训练类里取，测试类不进训练集。

输出格式：JSONL，每行
    {"id": "...", "image": "<相对路径>", "task": "cls|type|loc|desc|cot",
     "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
     "meta": {...}}

用法：
    python -m seg_cross_part.build_instructions --fold 0 --n 5000 --out data/instructions_fold0.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MANIFEST_PIXEL, ROOT, load_config, resolve_splits, with_defaults

# --------------------------------------------------------------------------
# 缺陷名 → 自然语言
# --------------------------------------------------------------------------
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
AREA_WORD = {"small": "a small area", "medium": "a moderate area", "large": "a large area"}
GRID_NAME = {(0, 0): "upper-left", (0, 1): "top center", (0, 2): "upper-right",
             (1, 0): "middle-left", (1, 1): "center", (1, 2): "middle-right",
             (2, 0): "lower-left", (2, 1): "bottom center", (2, 2): "lower-right"}

# --------------------------------------------------------------------------
# 指令模板（每个任务 4–6 种问法）
# --------------------------------------------------------------------------
T_CLS = [
    "Does this part have any manufacturing defect? Answer yes or no.",
    "Is there a defect on this component? Please answer yes or no.",
    "Examine the image. Should this part be rejected by quality control? Answer yes or no.",
    "Look at this industrial part. Is its surface and structure free of defects? Answer yes or no.",
]
T_TYPE = [
    "What type of defect is visible on this part? Answer with the defect name.",
    "Identify the defect in this image. Reply with a single defect type.",
    "Which kind of manufacturing defect is present here?",
    "Name the defect shown in the image, e.g. scratch, crack, deformation, contamination.",
]
T_LOC = [
    "Where is the defect located? Describe its position using a 3x3 grid (upper-left ... lower-right).",
    "In which region of the part does the defect appear? Answer with one grid position.",
    "Localize the defect coarsely: which of the nine regions (top/middle/bottom x left/center/right)?",
    "Give the approximate location of the defect in the image.",
]
T_DESC = [
    "Describe the condition of this part for an inspection report.",
    "Write a one-sentence inspection note about this component.",
    "Summarize what you observe on this part.",
    "Briefly describe the defect and its extent.",
]
T_COT = [
    "Inspect this part step by step: first judge whether it is normal, then locate and name the "
    "defect if any, and finally give your conclusion.",
    "Reason about the quality of this component: describe what you look at, what you find, "
    "and what you conclude.",
]

ANS_CLS = {"yes": "Yes.", "no": "No."}


def grid_of(mask_path: Path) -> str | None:
    """mask 外接框中心落在 3×3 的哪一格。"""
    from PIL import Image
    a = np.array(Image.open(mask_path).convert("L")) > 127
    if not a.any():
        return None
    ys, xs = np.nonzero(a)
    cy = ys.mean() / a.shape[0]
    cx = xs.mean() / a.shape[1]
    r = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    c = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    return GRID_NAME[(r, c)]


def capable_tasks(row) -> list[str]:
    """该图能支撑哪些任务（正常图没有缺陷类型/位置）。"""
    if int(row["view_label"]) == 1:
        t = ["cls", "type", "desc", "cot"]
        if row.get("mask_valid") and row.get("mask_path"):
            t.append("loc")
        return t
    return ["cls", "desc", "cot"]


def build_one(row, src: Path, rng: random.Random, task: str | None = None) -> list[dict]:
    """一张图派生一条指令样本（task 指定）；task=None 时随机挑一个可做的任务。"""
    out = []
    label = int(row["view_label"])
    defect = DEFECT_WORDS.get(str(row.get("defect_norm") or ""), None)
    area = AREA_WORD.get(str(row.get("area_bucket") or ""), "some area")
    base = {"image": row["image_path"], "class": row["class_name"], "label": label,
            "defect": defect, "area": row.get("area_bucket"),
            "split": row.get("split", "?")}

    # 1) 图像级判断
    q = rng.choice(T_CLS)
    out.append({"task": "cls", "conversations": [{"from": "human", "value": q},
                                                 {"from": "gpt", "value": ANS_CLS["yes" if label else "no"]}],
                "meta": dict(base)})

    if label and defect:
        # 2) 缺陷类型
        q = rng.choice(T_TYPE)
        out.append({"task": "type", "conversations": [{"from": "human", "value": q},
                                                      {"from": "gpt", "value": defect.capitalize() + "."}],
                    "meta": dict(base)})
        # 3) 位置（需要 mask）
        if row.get("mask_valid") and row.get("mask_path"):
            g = grid_of(src / row["mask_path"])
            if g:
                q = rng.choice(T_LOC)
                out.append({"task": "loc", "conversations": [{"from": "human", "value": q},
                                                             {"from": "gpt", "value": f"The defect is in the {g} region."}],
                            "meta": {**base, "grid": g}})
        # 4) 描述
        q = rng.choice(T_DESC)
        ans = f"The part shows a {defect} affecting {area} of the component."
        out.append({"task": "desc", "conversations": [{"from": "human", "value": q},
                                                      {"from": "gpt", "value": ans}], "meta": dict(base)})
        # 5) 推理链
        if rng.random() < 0.5:
            q = rng.choice(T_COT)
            loc = ""
            if row.get("mask_valid") and row.get("mask_path"):
                g = grid_of(src / row["mask_path"])
                if g:
                    loc = f" in the {g} region"
            ans = (f"1) The part is not intact: there is a visible irregularity{loc}. "
                   f"2) The irregularity looks like {defect} covering {area}. "
                   f"3) Conclusion: the part is defective and should be flagged as NG.")
            out.append({"task": "cot", "conversations": [{"from": "human", "value": q},
                                                         {"from": "gpt", "value": ans}],
                        "meta": {**base, "grid": loc or None}})
    else:
        # 正常图：描述 + 推理
        q = rng.choice(T_DESC)
        ans = "The part is intact: no scratch, crack, deformation, stain or missing part is visible."
        out.append({"task": "desc", "conversations": [{"from": "human", "value": q},
                                                      {"from": "gpt", "value": ans}], "meta": dict(base)})
        if rng.random() < 0.5:
            q = rng.choice(T_COT)
            ans = ("1) The surface is uniform and the structure is complete. "
                   "2) No irregularity, discoloration or missing region is found. "
                   "3) Conclusion: the part is normal (OK).")
            out.append({"task": "cot", "conversations": [{"from": "human", "value": q},
                                                         {"from": "gpt", "value": ans}], "meta": dict(base)})
    if task is not None:
        out = [x for x in out if x["task"] == task] or out[:1]
    else:
        out = [rng.choice(out)]
    for _s in out:
        _s["image"] = row["image_path"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="构造 Qwen3-VL 指令数据")
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="只从这些 split 取（test 绝不进训练集）")
    ap.add_argument("--n", type=int, default=5000, help="目标样本条数（不是图片数）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/instructions_fold0.jsonl")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    src = ROOT / cfg["data"]["source_root"]
    pix = pd.read_parquet(MANIFEST_PIXEL,
                          columns=["image_path", "mask_path", "mask_valid", "area_bucket",
                                   "defect_norm", "group_path"])
    dfs = []
    for sp in args.splits:
        d = pd.read_parquet(splits[sp])
        d["split"] = sp
        dfs.append(d)
    df = pd.concat(dfs).merge(pix.drop(columns=["group_path"]), on="image_path", how="left")
    print(f"[data] 候选图片 {len(df)}（split={args.splits}，测试类不入）")

    rng = random.Random(args.seed)
    # 任务配额（总和 = n），保证五种任务均衡、且每条样本对应一张不同的图
    w = {"cls": 0.24, "type": 0.20, "loc": 0.20, "desc": 0.22, "cot": 0.14}
    quota = {k: int(args.n * v) for k, v in w.items()}
    quota["cls"] += args.n - sum(quota.values())

    idx_by_task = defaultdict(list)
    for i in range(len(df)):
        for t in capable_tasks(df.iloc[i]):
            idx_by_task[t].append(i)
    for t in idx_by_task:
        rng.shuffle(idx_by_task[t])

    used, samples = set(), []
    for task, n_t in quota.items():
        cand = [i for i in idx_by_task.get(task, []) if i not in used]
        for i in cand[:n_t]:
            s = build_one(df.iloc[i], src, rng, task=task)
            if not s:
                continue
            used.add(i)
            samples.append({**s[0], "id": f"f{args.fold}_{len(samples):06d}"})
            if len(samples) >= args.n:
                break
        if len(samples) >= args.n:
            break
    rng.shuffle(samples)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- 统计报告 ----
    tasks = Counter(s["task"] for s in samples)
    labels = Counter(s["meta"]["label"] for s in samples)
    defects = Counter(s["meta"]["defect"] for s in samples if s["meta"]["defect"])
    answers = [s["conversations"][1]["value"] for s in samples]
    lens = [len(a) for a in answers]
    rep = {
        "n_samples": len(samples),
        "n_unique_images": len({s["image"] for s in samples}),
        "task_distribution": dict(tasks),
        "label_distribution": {"anomaly": labels[1], "normal": labels[0]},
        "n_defect_types": len(defects),
        "defect_top": dict(defects.most_common(8)),
        "answer_len": {"mean": float(np.mean(lens)), "p50": float(np.percentile(lens, 50)),
                       "max": int(np.max(lens))},
        "splits": dict(Counter(s["meta"].get("split", "?") for s in samples)),
    }
    (out.parent / (out.stem + "_report.json")).write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n已写出 {out} 与 {out.parent}/{out.stem}_report.json")


if __name__ == "__main__":
    main()
