"""从冻结的 fold1 dev 构建 20 个完整样本的模型对照子集。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/splits/fold1/dev.parquet"),
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("artifacts/b0/smoke20_manifest.parquet"),
    )
    ap.add_argument("--classes", type=int, default=10)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    class_width = manifest.groupby("class_name").width.median().sort_values()
    indices = np.unique(
        np.linspace(0, len(class_width) - 1, args.classes, dtype=int)
    )
    selected_classes = class_width.index[indices].tolist()

    selected_groups: list[str] = []
    selection_rows: list[dict] = []
    for class_name in selected_classes:
        class_rows = manifest[manifest.class_name == class_name]
        target_width = float(class_rows.width.median())
        group_meta = (
            class_rows.groupby("group_path", as_index=False)
            .agg(
                condition=("condition", "first"),
                sample_label=("sample_label", "first"),
                median_width=("width", "median"),
                n_views=("view_idx", "nunique"),
            )
        )
        for condition in ("OK", "NG"):
            candidates = group_meta[
                (group_meta.condition == condition) & (group_meta.n_views == 5)
            ].copy()
            candidates["distance_to_class_median"] = (
                candidates.median_width - target_width
            ).abs()
            chosen = candidates.sort_values(
                ["distance_to_class_median", "group_path"], kind="mergesort"
            ).iloc[0]
            selected_groups.append(str(chosen.group_path))
            selection_rows.append(
                {
                    "class_name": class_name,
                    "condition": condition,
                    "group_path": chosen.group_path,
                    "median_width": int(chosen.median_width),
                }
            )

    subset = manifest[manifest.group_path.isin(selected_groups)].copy()
    subset = subset.sort_values(
        ["class_name", "condition", "group_path", "view_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    assert subset.group_path.nunique() == 2 * len(selected_classes)
    assert len(subset) == 5 * subset.group_path.nunique()
    counts = (
        subset.drop_duplicates("group_path")
        .groupby("condition")
        .size()
        .to_dict()
    )
    assert counts == {"NG": len(selected_classes), "OK": len(selected_classes)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(args.out, index=False)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(args.out.with_suffix(".csv"), index=False)
    print(selection.to_string(index=False))
    print(
        f"[write] {args.out}: {subset.group_path.nunique()} 个样本 / "
        f"{len(subset)} 个视角 / {subset.class_name.nunique()} 类"
    )


if __name__ == "__main__":
    main()
