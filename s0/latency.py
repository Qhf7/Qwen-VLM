"""端到端延迟测量：按完整 5 视角样本、分辨率类别和视角分别报告。

计时拆成 decode / preprocess / inference，并同时报告三者之和。旧实现每个
``part_uid`` 只取一行却把每五行 reshape 成一个样本，会把五个不同样本误拼在一起；
这里始终以 ``group_path`` 为样本单位。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s0.predictor import (  # noqa: E402
    ConstantPredictor,
    NativeResizePolicy,
    Predictor,
    RandomPredictor,
    Resize448Policy,
    ResizePolicy,
)

Image.MAX_IMAGE_PIXELS = None


def _percentiles(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _time_view(
    row: pd.Series,
    predictor: Predictor,
    policy: ResizePolicy,
    dataset_root: Path,
) -> dict:
    t0 = time.perf_counter()
    image = np.asarray(Image.open(dataset_root / row.image_path).convert("RGB"))
    t1 = time.perf_counter()
    model_input = policy.model_input(image)
    t2 = time.perf_counter()
    predictor.predict(model_input)
    t3 = time.perf_counter()
    return {
        "group_path": row.group_path,
        "class_name": row.class_name,
        "view_idx": int(row.view_idx),
        "width": int(row.width),
        "height": int(row.height),
        "decode": t1 - t0,
        "preprocess": t2 - t1,
        "inference": t3 - t2,
        "end_to_end": t3 - t0,
    }


def select_complete_groups(
    manifest: pd.DataFrame, class_names: list[str], n_per_class: int
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for class_name in class_names:
        sub = manifest[manifest.class_name == class_name]
        complete = sub.groupby("group_path").filter(
            lambda group: len(group) == 5 and group.view_idx.nunique() == 5
        )
        group_names = sorted(complete.group_path.unique())[:n_per_class]
        selected.append(complete[complete.group_path.isin(group_names)])
    rows = pd.concat(selected, ignore_index=True)
    return rows.sort_values(["class_name", "group_path", "view_idx"], kind="mergesort")


def report(
    predictor: Predictor,
    policy: ResizePolicy,
    manifest: pd.DataFrame,
    class_names: list[str],
    dataset_root: Path,
    n_per_class: int = 5,
) -> str:
    rows = select_complete_groups(manifest, class_names, n_per_class)
    if rows.empty:
        raise ValueError("没有可用于延迟测量的完整 5 视角样本")

    # 不计时预热，避免真实模型首次 kernel/allocator 初始化污染 p95。
    warm = rows.iloc[0]
    image = np.asarray(Image.open(dataset_root / warm.image_path).convert("RGB"))
    predictor.predict(policy.model_input(image))

    records = [
        _time_view(row, predictor, policy, dataset_root)
        for _, row in rows.iterrows()
    ]
    times = pd.DataFrame(records)
    sample_times = times.groupby(["class_name", "group_path"], sort=True).agg(
        decode=("decode", "sum"),
        preprocess=("preprocess", "sum"),
        inference=("inference", "sum"),
        end_to_end=("end_to_end", "sum"),
    )

    view_p50, view_p95 = _percentiles(times.end_to_end.tolist())
    sample_p50, sample_p95 = _percentiles(sample_times.end_to_end.tolist())
    decode_p50, decode_p95 = _percentiles(times.decode.tolist())
    pre_p50, pre_p95 = _percentiles(times.preprocess.tolist())
    infer_p50, infer_p95 = _percentiles(times.inference.tolist())

    lines = [
        "=" * 78,
        f"延迟报告 | policy={policy.name}",
        "=" * 78,
        f"完整样本 {len(sample_times)}，视角 {len(times)}，类别 {class_names}",
        f"每视角端到端  p50={view_p50 * 1000:.1f}ms  p95={view_p95 * 1000:.1f}ms",
        f"每样本5视角   p50={sample_p50 * 1000:.1f}ms  p95={sample_p95 * 1000:.1f}ms",
        f"每视角 decode p50/p95={decode_p50 * 1000:.1f}/{decode_p95 * 1000:.1f}ms",
        f"每视角 pre    p50/p95={pre_p50 * 1000:.1f}/{pre_p95 * 1000:.1f}ms",
        f"每视角 infer  p50/p95={infer_p50 * 1000:.1f}/{infer_p95 * 1000:.1f}ms",
        "",
        "按类别（5视角样本端到端）",
        "类别                         N      p50ms      p95ms",
    ]
    for class_name in class_names:
        values = sample_times.loc[class_name].end_to_end.tolist()
        p50, p95 = _percentiles(values)
        lines.append(f"{class_name[:28]:28s} {len(values):3d} {p50*1000:10.1f} {p95*1000:10.1f}")

    lines.extend(["", "按视角（端到端）", "视角   N      p50ms      p95ms"])
    for view_idx in range(5):
        values = times.loc[times.view_idx == view_idx, "end_to_end"].tolist()
        p50, p95 = _percentiles(values)
        lines.append(f"C{view_idx + 1}   {len(values):3d} {p50*1000:10.1f} {p95*1000:10.1f}")

    lines.extend(
        [
            "",
            f"约束（5视角样本 p95 ≤ 1s）：{'达成' if sample_p95 <= 1.0 else '未达成'}",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="S0 端到端延迟测量")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "holdout", "train"], default="holdout")
    ap.add_argument("--predictor", choices=["constant", "random"], default="constant")
    ap.add_argument("--policy", choices=["r448", "rnative"], default="r448")
    ap.add_argument("--max-side", type=int, default=0)
    ap.add_argument("--groups-per-class", type=int, default=5)
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--manifests", type=Path, default=Path("artifacts/splits"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    predictor: Predictor = (
        ConstantPredictor(score=0.5, heat=0.5)
        if args.predictor == "constant"
        else RandomPredictor(seed=0)
    )
    policy: ResizePolicy = (
        Resize448Policy()
        if args.policy == "r448"
        else NativeResizePolicy(max_side=args.max_side)
    )
    manifest = pd.read_parquet(
        args.manifests / f"fold{args.fold}" / f"{args.split}.parquet"
    )
    widths = manifest.groupby("class_name").width.median()
    class_names = [str(widths.idxmin()), str(widths.idxmax())]
    text = report(
        predictor, policy, manifest, class_names, args.root, args.groups_per_class
    )
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
