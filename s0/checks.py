"""快速验证 S0 产物不变量；任何下游实验开始前都应通过。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def batch_of(group_path: str) -> str:
    parts = Path(group_path).parts
    return "OK" if parts[1] == "OK" else parts[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.artifacts / "manifest_with_groups.parquet")
    groups = pd.read_parquet(args.artifacts / "part_clusters.parquet")
    assert len(manifest) == 29855
    assert manifest.class_name.nunique() == 24
    assert manifest.group_path.nunique() == 5971
    assert len(groups) == groups.group_path.nunique() == 5971
    assert manifest.visual_group_id.notna().all()
    assert int(manifest.mask_file_exists.sum()) == 23378
    assert int(manifest.view_label.sum()) == 23369

    group_batches = groups.assign(batch=groups.group_path.map(batch_of))
    assert not group_batches.duplicated(["visual_group_id", "batch"]).any()

    for fold in range(4):
        fold_dir = args.artifacts / "splits" / f"fold{fold}"
        train = pd.read_parquet(fold_dir / "train.parquet")
        dev = pd.read_parquet(fold_dir / "dev.parquet")
        holdout = pd.read_parquet(fold_dir / "holdout.parquet")
        assert not (set(train.visual_group_id) & set(dev.visual_group_id))
        assert not (set(train.group_path) & set(dev.group_path))
        assert set(train.class_name).isdisjoint(set(holdout.class_name))
        assert set(dev.class_name).isdisjoint(set(holdout.class_name))
        counts = (
            dev.drop_duplicates("group_path")
            .groupby(["class_name", "condition"])
            .size()
            .unstack(fill_value=0)
        )
        assert len(counts) == 18
        assert (counts["OK"] == 20).all() and (counts["NG"] == 20).all()
        print(
            f"fold{fold}: train/dev/holdout 视角 "
            f"{len(train)}/{len(dev)}/{len(holdout)} ✓"
        )
    print("S0 artifact invariants passed ✓")


if __name__ == "__main__":
    main()
