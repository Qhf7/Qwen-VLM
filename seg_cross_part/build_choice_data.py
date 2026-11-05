"""构造"二选一判别"指令数据（用来检验模型是否真的在看图）。

背景：SFT 后缺陷类型准确率只有 12.8%，低于"类内 4 选 1"的 25%，说明模型在背训练集的
全局先验（Deformation / Scratch 最多），而不是看图像判别。二选一是最简单的判别形式，
如果连它都做不了，就不是"数据没教好"而是"能力不够"。

两类任务：
    pair  ：两张同类图（一正常一异常），问"哪一张有缺陷？答 first 或 second"
    type2 ：一张异常图 + 该类的两种缺陷名（一对一错），问"是 A 还是 B？"

输出与 SFT 相同格式，额外字段 image2（pair 任务用）、meta。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

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

Q_PAIR = [
    "The first image and the second image are two parts of the same type. Exactly one of them "
    "has a manufacturing defect. Which one? Answer strictly: first or second.",
    "Two parts of the same type are shown, first and second. One is defective. "
    "Which image shows the defect? Answer strictly: first or second.",
]
Q_TYPE2 = [
    "Look at this part. Is the defect a {a} or a {b}? Answer with one of the two names only.",
    "This part has a defect. Is it more likely a {a} or a {b}? Answer with one of the two.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000, help="总数（pair 与 type2 各一半）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/choice_fold0.jsonl")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "defect_norm", "mask_valid"])
    dfs = []
    for sp in ("train", "val"):
        d = pd.read_parquet(splits[sp]); d["split"] = sp; dfs.append(d)
    df = pd.concat(dfs).merge(pix, on="image_path", how="left")
    rng = random.Random(args.seed)

    by_cls = {c: g for c, g in df.groupby("class_name")}
    classes = sorted(by_cls)
    out, n_each = [], args.n // 2

    # ---- pair：同类两张图，一张异常一张正常 ----
    tries = 0
    while sum(1 for s in out if s["task"] == "pair") < n_each and tries < n_each * 50:
        tries += 1
        c = rng.choice(classes)
        g = by_cls[c]
        a = g[g.view_label == 1]
        nz = g[g.view_label == 0]
        if len(a) == 0 or len(nz) == 0:
            continue
        ai = a.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        ni = nz.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        anom_first = rng.random() < 0.5
        img1, img2 = (ai.image_path, ni.image_path) if anom_first else (ni.image_path, ai.image_path)
        ans = "first" if anom_first else "second"
        out.append({"task": "pair", "image": img1, "image2": img2,
                    "conversations": [{"from": "human", "value": rng.choice(Q_PAIR)},
                                      {"from": "gpt", "value": ans.capitalize() + "."}],
                    "meta": {"class": c, "label": 1, "anom_first": anom_first}})

    # ---- type2：一张异常图，二选一缺陷名 ----
    tries = 0
    while sum(1 for s in out if s["task"] == "type2") < n_each and tries < n_each * 50:
        tries += 1
        c = rng.choice(classes)
        g = by_cls[c]
        a = g[g.view_label == 1].dropna(subset=["defect_norm"])
        a = a[a.defect_norm != ""]
        if len(a) == 0:
            continue
        own = sorted(set(a.defect_norm) - {""})
        if len(own) < 2:
            continue
        r = a.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
        others = [x for x in own if x != r.defect_norm]
        wrong = rng.choice(others)
        cands = [r.defect_norm, wrong]
        rng.shuffle(cands)
        a_word = DEFECT_WORDS.get(cands[0], cands[0].lower())
        b_word = DEFECT_WORDS.get(cands[1], cands[1].lower())
        q = rng.choice(Q_TYPE2).format(a=a_word, b=b_word)
        out.append({"task": "type2", "image": r.image_path,
                    "conversations": [{"from": "human", "value": q},
                                      {"from": "gpt", "value": DEFECT_WORDS.get(r.defect_norm, r.defect_norm).capitalize() + "."}],
                    "meta": {"class": c, "label": 1, "gt": r.defect_norm, "cands": cands}})

    rng.shuffle(out)
    for i, s in enumerate(out):
        s["id"] = f"c{args.fold}_{i:05d}"
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    rep = {"n": len(out), "tasks": dict(Counter(s["task"] for s in out)),
           "classes": len({s["meta"]["class"] for s in out}),
           "pair_answer_first_ratio": round(sum(s["meta"].get("anom_first") is True
                                                for s in out if s["task"] == "pair") /
                                            max(1, sum(s["task"] == "pair" for s in out)), 3)}
    (p.parent / (p.stem + "_report.json")).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"已写出 {p}")


if __name__ == "__main__":
    main()
