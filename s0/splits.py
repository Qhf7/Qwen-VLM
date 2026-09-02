"""冻结四折 holdout，并按视觉近重复组构建无已知近重复泄漏的 dev-mini。

三个不同概念不能混用：

* ``group_path``：一次 5 视角拍摄，是图像级评测样本；
* ``source_part_key``：数据源的 S 号，只用于追溯，跨批身份不可靠；
* ``visual_group_id``：C1 内容近重复连通分量，只用于 train/dev 防泄漏。

已经写出的 ``holdout_classes.txt`` 默认视为冻结契约，本脚本会复用它们而不会重新
抽签。只有显式传 ``--rebuild-holdout`` 才会重建整类四折。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s0 import vocab  # noqa: E402


FOLDS = 4
DEV_OK_TARGET = 20
DEV_NG_TARGET = 20
DEV_MIN_PER_LABEL = 5
DEV_OK_MAX = 25
DEV_NG_MAX = 40


def _score_fold(classes: set[str], manifest: pd.DataFrame) -> dict:
    group_rows = manifest[manifest.class_name.isin(classes)].drop_duplicates("group_path")
    widths = group_rows.groupby("class_name").width.median()
    words = set(group_rows.query('condition == "NG"').defect_norm.dropna())
    unique_classes = set(
        group_rows.query("defect_norm in @vocab.UNIQUE_DEFECTS").class_name
    )
    return {
        "n_hf": len(words & set(vocab.HIGH_FREQ_DEFECTS)),
        "unique_classes": sorted(classes & unique_classes),
        "w_min": int(widths.min()),
        "w_med": int(widths.median()),
        "w_max": int(widths.max()),
    }


def make_folds(manifest: pd.DataFrame) -> list[list[str]]:
    """首次运行时确定四折；之后由磁盘上的 holdout 类表冻结。"""
    classes = sorted(manifest.class_name.unique())
    if len(classes) != 24:
        raise AssertionError(f"期望 24 类，实际 {len(classes)}")
    medians = manifest.groupby("class_name").width.median()
    unique_classes = sorted(
        set(manifest.query("defect_norm in @vocab.UNIQUE_DEFECTS").class_name),
        key=lambda c: (-medians[c], c),
    )

    folds: list[list[str]] = [[] for _ in range(FOLDS)]
    for i, class_name in enumerate(unique_classes):
        folds[i].append(class_name)
    remaining = [c for c in sorted(classes, key=lambda c: (medians[c], c)) if c not in unique_classes]
    for class_name in remaining:
        # 先平衡类数，再让各折分辨率中位附近的总 rank 尽量接近。
        candidate = min(
            range(FOLDS),
            key=lambda i: (len(folds[i]), sum(medians[c] for c in folds[i]), i),
        )
        folds[candidate].append(class_name)
    if sorted(c for fold in folds for c in fold) != classes or any(len(f) != 6 for f in folds):
        raise AssertionError("四折没有形成 24 类的 6×4 完整分区")
    return [sorted(fold) for fold in folds]


def load_or_make_folds(
    manifest: pd.DataFrame, out_dir: Path, rebuild: bool
) -> tuple[list[list[str]], bool]:
    paths = [out_dir / f"fold{i}" / "holdout_classes.txt" for i in range(FOLDS)]
    if not rebuild and all(path.is_file() for path in paths):
        folds = [
            sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            for path in paths
        ]
        expected = sorted(manifest.class_name.unique())
        actual = sorted(c for fold in folds for c in fold)
        if actual != expected or any(len(fold) != 6 for fold in folds):
            raise AssertionError("已冻结的 holdout_classes.txt 不是 24 类的 6×4 完整分区")
        return folds, True
    return make_folds(manifest), False


def _visual_group_counts(class_df: pd.DataFrame) -> list[tuple[str, int, int]]:
    groups = class_df.drop_duplicates("group_path")
    rows: list[tuple[str, int, int]] = []
    for group_id, sub in groups.groupby("visual_group_id", sort=True):
        rows.append(
            (
                str(group_id),
                int((sub.condition == "OK").sum()),
                int((sub.condition == "NG").sum()),
            )
        )
    return rows


def choose_dev_groups(
    class_df: pd.DataFrame,
    target_ok: int = DEV_OK_TARGET,
    target_ng: int = DEV_NG_TARGET,
) -> tuple[set[str], tuple[int, int]]:
    """二维 subset-sum：选完整视觉组，使 dev 接近 20 OK + 20 NG。

    状态空间最多 ``50×200``。候选必须让 train 和 dev 都至少保留一个 OK、一个
    NG；若做不到则返回空集，调用方把该类标为 dev-unsplittable。
    """
    components = _visual_group_counts(class_df)
    total_ok = sum(ok for _, ok, _ in components)
    total_ng = sum(ng for _, _, ng in components)
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}

    for group_id, n_ok, n_ng in components:
        additions: dict[tuple[int, int], tuple[str, ...]] = {}
        for (cur_ok, cur_ng), selected in states.items():
            next_ok, next_ng = cur_ok + n_ok, cur_ng + n_ng
            # 限制 dev 规模并给 train 留足两类样本；否则一个巨型 component 会
            # 产生 1 OK + 62 NG 之类统计上不可用的 dev。
            if (
                next_ok > DEV_OK_MAX
                or next_ng > DEV_NG_MAX
                or total_ok - next_ok < DEV_MIN_PER_LABEL
                or total_ng - next_ng < DEV_MIN_PER_LABEL
            ):
                continue
            key = (next_ok, next_ng)
            if key not in states and key not in additions:
                additions[key] = selected + (group_id,)
        states.update(additions)

    valid = [
        (counts, selected)
        for counts, selected in states.items()
        if counts[0] >= DEV_MIN_PER_LABEL and counts[1] >= DEV_MIN_PER_LABEL
    ]
    if not valid:
        return set(), (0, 0)

    def objective(item: tuple[tuple[int, int], tuple[str, ...]]) -> tuple:
        (n_ok, n_ng), selected = item
        return (
            abs(n_ok - target_ok) + abs(n_ng - target_ng),
            abs(n_ok - n_ng),
            n_ok + n_ng,
            selected,
        )

    counts, selected = min(valid, key=objective)
    return set(selected), counts


def main() -> None:
    ap = argparse.ArgumentParser(description="冻结四折 + 按视觉组切 train/dev")
    ap.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/manifest_with_groups.parquet"),
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/splits"))
    ap.add_argument(
        "--rebuild-holdout", action="store_true",
        help="显式重建整类四折；默认复用已冻结的 holdout_classes.txt",
    )
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    required = {"class_name", "group_path", "condition", "visual_group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"manifest 缺少列 {sorted(missing)}；请先运行 s0/part_clusters.py")
    args.out.mkdir(parents=True, exist_ok=True)

    folds, reused = load_or_make_folds(manifest, args.out, args.rebuild_holdout)
    report: list[str] = [
        "=" * 78,
        "三集冻结：4 折整类 holdout + visual_group 原子化 train/dev",
        "=" * 78,
        f"holdout 类表：{'复用既有冻结结果' if reused else '首次生成/显式重建'}",
        "评测样本单位：group_path（5 视角）；防泄漏代理：visual_group_id",
    ]

    all_classes = set(manifest.class_name.unique())
    for fold_index, holdout_list in enumerate(folds):
        holdout = set(holdout_list)
        train_classes = sorted(all_classes - holdout)
        fold_score = _score_fold(holdout, manifest)
        dev_group_ids: set[str] = set()
        class_dev_counts: list[tuple[str, int, int, int]] = []
        unsplittable: list[str] = []

        for class_name in train_classes:
            class_df = manifest[manifest.class_name == class_name]
            selected, (n_ok, n_ng) = choose_dev_groups(class_df)
            if not selected:
                unsplittable.append(class_name)
                class_dev_counts.append((class_name, 0, 0, 0))
                continue
            dev_group_ids |= selected
            class_dev_counts.append((class_name, len(selected), n_ok, n_ng))

        holdout_mask = manifest.class_name.isin(holdout)
        dev_mask = manifest.visual_group_id.isin(dev_group_ids) & ~holdout_mask
        train_mask = ~holdout_mask & ~dev_mask
        train_df = manifest[train_mask].copy()
        dev_df = manifest[dev_mask].copy()
        holdout_df = manifest[holdout_mask].copy()

        train_groups = set(train_df.visual_group_id)
        dev_groups = set(dev_df.visual_group_id)
        if train_groups & dev_groups:
            raise AssertionError("train/dev visual_group_id 相交")
        if set(train_df.group_path) & set(dev_df.group_path):
            raise AssertionError("train/dev group_path 相交")
        if train_df.class_name.isin(holdout).any() or dev_df.class_name.isin(holdout).any():
            raise AssertionError("holdout 类进入 train/dev")

        fold_dir = args.out / f"fold{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(fold_dir / "train.parquet", index=False)
        dev_df.to_parquet(fold_dir / "dev.parquet", index=False)
        holdout_df.to_parquet(fold_dir / "holdout.parquet", index=False)
        (fold_dir / "holdout_classes.txt").write_text(
            "\n".join(sorted(holdout)) + "\n", encoding="utf-8"
        )

        report.extend(
            [
                "",
                f"── 折 {fold_index} ──────────────────────────────────────────────",
                f"  holdout 类（6）：{','.join(sorted(holdout))}",
                f"  高频词 {fold_score['n_hf']}/8 | 独有词类 {fold_score['unique_classes']} | "
                f"分辨率 {fold_score['w_min']}~{fold_score['w_max']} 中位 {fold_score['w_med']}",
                f"  视角 train={len(train_df)} dev={len(dev_df)} holdout={len(holdout_df)}",
                f"  样本目录 train={train_df.group_path.nunique()} "
                f"dev={dev_df.group_path.nunique()} holdout={holdout_df.group_path.nunique()}",
                f"  dev-unsplittable（不按 S 号强拆）：{unsplittable or '无'}",
                "  每类 dev：visual组 / OK样本 / NG样本",
            ]
        )
        report.extend(
            f"    {class_name[:28]:28s} {n_groups:3d} / {n_ok:3d} / {n_ng:3d}"
            for class_name, n_groups, n_ok, n_ng in class_dev_counts
        )

    report.extend(
        [
            "",
            "【纪律】",
            "  holdout 按整类冻结；默认运行不会重新抽签。",
            "  train/dev 的 visual_group_id 严格互斥。",
            "  visual_group_id 是视觉近重复代理，不被夸大为无误的物理工件 ID。",
            "  图像级指标始终按 group_path 聚合，不能按 visual_group_id 聚合。",
            "=" * 78,
        ]
    )
    report_text = "\n".join(report)
    (args.out / "split_report.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
