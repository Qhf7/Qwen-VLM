"""评测编排：跑一个 predictor 过一折，产出完整指标报告。

数据流
------
    fold{k}/holdout.parquet (或 dev)  ─┐
    每行 = 一个视角图像                   │
            │ 按 group_path 归组（5视角样本）▼
        读原图 + GT → ResizePolicy → Predictor
            ▼
    项目级分数（view→sample 聚合）  + 像素级热图
            ▼
    metrics.py 按类算 → 宏平均 → 报告

高效点（S0 的性能约束）
----------------------
  3669px 图 decode 就要 ~0.27s，且数据集里不少。所以：
  1. **单趟**：每个视角 decode 一次、predict 一次，同时产出 score + heatmap。
     之前分图像级/像素级两趟跑，等于 decode 两遍，浪费一倍。
  2. **进程池**：decode+resize+推理是 CPU 密集，用多进程摊平。
     注意 predictor 必须可 pickle（平凡基线都可；torch 模型要放在 __main__ 或
     用 spawn 启动器，见 predictor 文档）。
  3. **--limit**：冒烟测试用，快速验证 I/O 和数据流。

两纪律在此强制
--------------
  1. P-AUROC 只输出，不参与决策（P-AUPR / P-F1max 才是决策依据）。
  2. "平均预测面积占比"必报——识破"把整张图涂亮"最快的单一数字。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s0 import metrics  # noqa: E402
from s0.predictor import (  # noqa: E402
    ConstantPredictor,
    NativeResizePolicy,
    Predictor,
    RandomPredictor,
    Resize448Policy,
    ResizePolicy,
)

Image.MAX_IMAGE_PIXELS = None

# 进程池里执行的函数。predictor/policy 必须可 pickle（平凡基线都行）。
# torch 模型如果在 __main__ 里构造，用 spawn 启动器时无法从子进程导入，
# 这是 S0 之后接真实模型时要处理的问题（见 predictor.py 文档）。
def _process_view(args):
    row, predictor, policy, dataset_root = args
    img = np.asarray(Image.open(dataset_root / row["image_path"]).convert("RGB"))
    gt_binary = (
        np.asarray(Image.open(dataset_root / row["mask_path"])).astype(bool)
        if row["view_label"]
        else np.zeros((row["height"], row["width"]), dtype=bool)
    )
    model_in = policy.model_input(img)
    score, heat = predictor.predict(model_in)
    pred_al, gt_al = policy.align(heat, gt_binary)
    area_ratio = float((pred_al > 0.5).mean())
    return {
        "group_path": row["group_path"],
        "class_name": row["class_name"],
        "image_path": row["image_path"],
        "area_bucket": row["area_bucket"],
        "sample_label": row["sample_label"],
        "view_label": int(row["view_label"]),
        "score": float(score),
        "heat_digit": pred_al,
        "gt": gt_al,
        "area_ratio": area_ratio,
    }


def _run_views(
    rows: list[dict], predictor, policy, workers: int, dataset_root: Path
) -> Iterator[dict]:
    if workers <= 1:
        for row in rows:
            yield _process_view((row, predictor, policy, dataset_root))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        args = ((r, predictor, policy, dataset_root) for r in rows)
        yield from pool.map(_process_view, args, chunksize=16)


def _macro_mean(values: list[float]) -> float:
    v = np.array([x for x in values if not np.isnan(x)])
    return float(v.mean()) if v.size else float("nan")


def run(manifest: pd.DataFrame, policy: ResizePolicy, predictor: Predictor,
        data_split: str, class_names: list[str] | None = None,
        workers: int = 1, limit: int = 0,
        dataset_root: Path = Path("Dataset")) -> str:
    if class_names is None:
        class_names = sorted(manifest.class_name.unique())
    if limit:
        # 冒烟测试按完整 5 视角样本抽，不能把评测单位切碎。
        groups = manifest[["group_path", "class_name", "sample_label"]].drop_duplicates()
        groups = groups.sort_values(
            ["class_name", "sample_label", "group_path"], kind="mergesort"
        )
        per_stratum = max(1, int(np.ceil(limit / max(1, 2 * len(class_names)))))
        groups = (
            groups.groupby(["class_name", "sample_label"], sort=True, group_keys=False)
            .head(per_stratum)
            .head(limit)
        )
        manifest = manifest[manifest.group_path.isin(groups.group_path)].reset_index(drop=True)

    rows = manifest.to_dict("records")
    # 流式累计：任一时刻只持有一个视角的 pred/GT，避免 3600×448² 数 GB 内存。
    pixel_state = {
        cl: {
            "pos": np.zeros(metrics.N_THRESHOLDS),
            "neg": np.zeros(metrics.N_THRESHOLDS),
            "ratios": [],
            "hits": [],
            "n": 0,
        }
        for cl in class_names
    }
    group_scores: dict[str, dict[str, float]] = defaultdict(dict)
    group_labels: dict[str, dict[str, int]] = defaultdict(dict)
    bucket_recall: dict[str, list[float]] = defaultdict(list)
    n_results = 0

    for view in _run_views(rows, predictor, policy, workers, dataset_root):
        n_results += 1
        cl = view["class_name"]
        group_path = view["group_path"]
        old_score = group_scores[cl].get(group_path, float("-inf"))
        group_scores[cl][group_path] = max(old_score, view["score"])
        old_label = group_labels[cl].setdefault(group_path, view["sample_label"])
        if old_label != view["sample_label"]:
            raise AssertionError(f"同一 group_path 标签不一致: {group_path}")

        state = pixel_state[cl]
        pred, gt = view["heat_digit"], view["gt"]
        state["pos"], state["neg"] = metrics._accumulate(
            state["pos"], state["neg"], pred, gt, metrics.N_THRESHOLDS
        )
        state["ratios"].append(view["area_ratio"])
        state["n"] += 1
        if view["view_label"] and gt.sum() > 0:
            flat = pred.ravel()
            k = max(1, int(0.01 * len(flat)))
            idx = np.argpartition(flat, -k)[-k:]
            state["hits"].append(float(gt.ravel()[idx].mean()))
            if view["area_bucket"]:
                bucket_recall[view["area_bucket"]].append(
                    float((pred > 0.5)[gt].mean())
                )

    im_by_class = {
        cl: metrics.compute_image_metrics_per_class(group_scores[cl], group_labels[cl])
        for cl in class_names
        if group_scores[cl]
    }
    pm_by_class = {
        cl: metrics.finish_pixel_metrics(
            state["pos"], state["neg"], state["ratios"], state["n"], state["hits"]
        )
        for cl, state in pixel_state.items()
        if state["n"]
    }
    rec = [
        (
            bucket,
            float(np.mean(bucket_recall[bucket])) if bucket_recall[bucket] else float("nan"),
            len(bucket_recall[bucket]),
        )
        for bucket in metrics.AREA_LABELS
    ]

    # ---- 报告 ----
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"S0 评测报告  |  数据{data_split:16s} |  策略 {policy.name} |  workers {workers}")
    add("=" * 78)
    add(f"  类别数 {len(im_by_class)}   视角数 {n_results}")
    add("")
    add("【图像级 ·宏平均】")
    add(f"  I-AUROC = {_macro_mean([m.i_auroc[0] for m in im_by_class.values()]):.4f}")
    add(f"  I-AUPR  = {_macro_mean([m.i_aupr[0] for m in im_by_class.values()]):.4f}")
    add(f"  F1max   = {_macro_mean([m.f1max[0] for m in im_by_class.values()]):.4f}")
    add("")
    add("【像素级 ·宏平均】")
    add(f"  P-AUROC = {_macro_mean([m.p_auroc[0] for m in pm_by_class.values()]):.4f}  ← 只输出，不决策")
    add(f"  P-AUPR  = {_macro_mean([m.p_aupr[0] for m in pm_by_class.values()]):.4f}  ← 决策")
    add(f"  P-F1max = {_macro_mean([m.p_f1max[0] for m in pm_by_class.values()]):.4f}  ← 决策")
    add(f"  Hit@1%  = {_macro_mean([x for m in pm_by_class.values() for x in m.hit_at_1pct]):.4f}")
    add(f"  平均预测面积占比 = {_macro_mean([x for m in pm_by_class.values() for x in m.pred_area_ratio]):.4%}")
    add("    ↑ 固定阈值 0.5；若远大于 GT 面积占比，说明在涂亮整图")
    add("")
    add("【按缺陷面积分档召回】(阈值 0.5)")
    for bucket, r, n in rec:
        add(f"  {bucket:>10s}  recall={r if np.isnan(r) else f'{r:.3f}'}   N={n}")
    add("")
    add("【按类明细】(宏平均来源，类别间方差可见)")
    add("  类别        I-AUROC  I-AUPR  F1max   P-AUROC  P-AUPR  P-F1max  面积占比")
    for cl in class_names:
        m = im_by_class.get(cl)
        pmc = pm_by_class.get(cl)
        if m is None or pmc is None:
            continue
        pa = _macro_mean(pmc.pred_area_ratio)
        add(f"  {cl[:20]:20s}"
            f" {_macro_mean(m.i_auroc):.3f}  {_macro_mean(m.i_aupr):.3f}  {_macro_mean(m.f1max):.3f} "
            f" {_macro_mean(pmc.p_auroc):.3f}  {_macro_mean(pmc.p_aupr):.3f}  {_macro_mean(pmc.p_f1max):.3f}   {pa:.3%}")
    add("=" * 78)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="S0 评测编排")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "holdout", "train"], default="dev")
    ap.add_argument("--predictor", choices=["constant", "random"], default="constant")
    ap.add_argument("--policy", choices=["r448", "rnative"], default="r448")
    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试取前 N 个完整样本；0=全部")
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--out", type=Path, default=None, help="可选：同时把报告写入文件")
    ap.add_argument("--manifests", type=Path, default=Path("artifacts/splits"))
    args = ap.parse_args()

    p: Predictor = (
        RandomPredictor(seed=0) if args.predictor == "random"
        else ConstantPredictor(score=0.5, heat=0.5)
    )
    policy: ResizePolicy = (
        NativeResizePolicy(max_side=args.max_side or 0) if args.policy == "rnative"
        else Resize448Policy()
    )

    manifest = pd.read_parquet(args.manifests / f"fold{args.fold}" / f"{args.split}.parquet")
    class_names = sorted(manifest.class_name.unique())
    report = run(
        manifest, policy, p, args.split, class_names, args.workers, args.limit,
        args.root,
    )
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
