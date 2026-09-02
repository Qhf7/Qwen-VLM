"""构建 E1 sampler pool，并验证四路等额 batch 的可复现性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from e1.balanced_sampler import BalancedBatchSampler, SOURCES, default_steps_per_epoch
except ModuleNotFoundError:
    from balanced_sampler import BalancedBatchSampler, SOURCES, default_steps_per_epoch


def base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "class_name", "group_path", "image_path", "mask_path", "view_idx",
        "sample_label", "view_label", "defect_norm", "defect_ratio", "area_bucket",
    ]
    return frame[columns].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--hard-normals", type=Path, default=Path("artifacts/e1/hard_normals.csv"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train)
    hard = pd.read_csv(args.hard_normals)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    overlap = sorted(set(train.class_name) & holdout)
    if overlap:
        raise RuntimeError(f"train 混入 holdout 类: {overlap}")

    real = base_columns(train[train.view_label.eq(1) & train.mask_file_exists]).copy()
    real["source_type"] = "real_anomaly"
    real["augmentation"] = "none"
    real["area_train_bin"] = pd.cut(
        real.defect_ratio,
        [-np.inf, 0.005, 0.02, np.inf],
        labels=["tiny<=0.5%", "small0.5-2%", "large>2%"],
    ).astype(str)

    hard_paths = set(hard.image_path)
    normal = base_columns(train[train.view_label.eq(0) & ~train.image_path.isin(hard_paths)]).copy()
    normal["source_type"] = "normal"
    normal["augmentation"] = "none"
    normal["area_train_bin"] = "none"

    hard_rows = base_columns(train[train.image_path.isin(hard_paths)]).copy()
    if len(hard_rows) != len(hard_paths):
        raise RuntimeError("hard-normal 无法与 train manifest 一一对应")
    hard_rows["source_type"] = "hard_normal"
    hard_rows["augmentation"] = "none"
    hard_rows["area_train_bin"] = "none"

    pseudo = base_columns(train[train.sample_label.eq(0)]).copy()
    pseudo["source_type"] = "pseudo_anomaly"
    pseudo["augmentation"] = pseudo.image_path.map(
        lambda value: "cutpaste" if sum(value.encode("utf-8")) % 2 == 0 else "nsa"
    )
    pseudo["area_train_bin"] = "synthetic"

    pool = pd.concat([real, normal, hard_rows, pseudo], ignore_index=True)
    pool["sampler_cell"] = np.where(
        pool.source_type.eq("real_anomaly"),
        pool.source_type + "|" + pool.class_name + "|" + pool.area_train_bin,
        pool.source_type + "|" + pool.class_name,
    )
    cell_counts = pool.groupby(["source_type", "sampler_cell"]).size().rename("cell_count")
    source_cells = pool.groupby("source_type").sampler_cell.nunique().rename("source_n_cells")
    pool = pool.join(cell_counts, on=["source_type", "sampler_cell"])
    pool = pool.join(source_cells, on="source_type")
    pool["sample_weight"] = (
        1.0 / len(SOURCES) / pool.source_n_cells / pool.cell_count
    )
    pool.insert(0, "pool_index", np.arange(len(pool)))
    pool.to_parquet(args.out_dir / "sampler_pool.parquet", index=False)
    pool.to_csv(args.out_dir / "sampler_pool.csv", index=False)

    steps = default_steps_per_epoch(pool, args.batch_size)
    sampler = BalancedBatchSampler(pool, args.batch_size, steps_per_epoch=min(steps, 100))
    observed = []
    first_batches = []
    for batch_index, indices in enumerate(sampler):
        batch = pool.iloc[indices]
        counts = batch.source_type.value_counts().to_dict()
        expected = args.batch_size // len(SOURCES)
        if any(counts.get(source, 0) != expected for source in SOURCES):
            raise RuntimeError(f"batch {batch_index} 不等额: {counts}")
        observed.extend(batch.source_type)
        if batch_index < 3:
            first_batches.append(indices)
    sampler_again = BalancedBatchSampler(pool, args.batch_size, steps_per_epoch=min(steps, 100))
    if first_batches != [batch for _, batch in zip(range(3), sampler_again)]:
        raise RuntimeError("BalancedBatchSampler 固定 seed 不可复现")

    summary = (
        pool.groupby(["source_type", "class_name", "area_train_bin"], as_index=False)
        .agg(n_rows=("image_path", "size"), weight_sum=("sample_weight", "sum"))
    )
    summary.to_csv(args.out_dir / "sampler_pool_summary.csv", index=False)
    config = {
        "protocol": "E1-balanced-sampler-v1",
        "sources": list(SOURCES),
        "source_probability": {source: 0.25 for source in SOURCES},
        "batch_size": args.batch_size,
        "quota_per_source": args.batch_size // len(SOURCES),
        "default_steps_per_epoch": steps,
        "area_bins": ["tiny<=0.5%", "small0.5-2%", "large>2%"],
        "real_anomaly_requires_nonempty_mask": True,
        "pseudo_pool_requires_sample_level_ok": True,
        "holdout_intersection": overlap,
        "counts": pool.source_type.value_counts().sort_index().to_dict(),
        "self_test_batches": min(steps, 100),
        "self_test_passed": True,
    }
    (args.out_dir / "sampler_protocol.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
