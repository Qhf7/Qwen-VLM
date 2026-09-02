"""扫描 Dataset/，产出视角级 manifest。

manifest 是 S0 的核心产物：一行 = 一个视角图像。所有下游代码只读 manifest，
不得再直接遍历 Dataset 目录——目录结构里的坑（ID 复用、mask 缺失、视角分辨率
不一致）在这里一次性解决，不要让它们泄漏到训练和评测代码里。

三条在这里被强制执行的口径
--------------------------
1. **视角级标签以 mask 是否非空为准，不继承样本级标签。**
   NG 图 23860 张，mask 文件 23378 张，其中 9 张为空；因此 23369 个视角
   确有缺陷，491 个视角按视角级正常处理（482 无文件 + 9 空文件）。
   继承样本标签会让 decoder 学成物体性而非缺陷性。

2. **S 号不是可靠的跨批物理零件 ID。**
   `<类>/OK/S0001` 与 `<类>/NG/*/S0001` 可能相同，也可能完全不同；因此这里只
   保留 `source_part_key=<类>/<Sxxxx>` 作为源数据键，绝不拿它防泄漏。防泄漏用
   `part_clusters.py` 生成的 `visual_group_id`，评测样本单位则是 `group_path`。

3. **分辨率按视角而非按类别变化。**
   C1 是高分辨率 RGB 视角，C2–C5 是较低分辨率的 W 光照视角，
   23/24 个类别里 C1 都是最大的。resize 策略必须按视角看，不能按类别看。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s0 import vocab  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # 数据集里有 3669px 的图，别触发解压炸弹告警

#: 文件名里的视角编号，形如 `..._C3_20240724094525379.png`。
#: 类别名本身含下划线和数字（如 `3_adapter`），所以取**最后一个**匹配。
_VIEW_RE = re.compile(r"_C(\d)_")

#: 缺陷面积分档（占图像面积的比例）。总分会被大缺陷主导，
#: 分档才能看出小缺陷上到底行不行。
AREA_BINS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
AREA_LABELS = ("<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%")


@dataclass(slots=True)
class ViewRecord:
    """一个视角图像的全部元信息。"""

    class_name: str
    source_part_key: str   # "<类>/<Sxxxx>" —— 仅为源数据键，不是物理身份
    group_path: str        # "<类>/OK/<Sxxxx>" 或 "<类>/NG/<类型>/<Sxxxx>"
    condition: str         # "OK" | "NG"
    defect_raw: str        # OK 为 ""
    defect_norm: str       # OK 为 ""
    view_idx: int          # 0..4（由文件名 C1..C5 减一得到）
    image_path: str        # 相对 Dataset/ 的路径
    mask_file_exists: bool # 区分无文件与有文件但空 mask
    mask_path: str         # 无 mask 为 ""
    sample_label: int      # 样本级：任一视角有缺陷 → 1
    view_label: int        # 视角级：该视角有非空 mask → 1
    width: int
    height: int
    defect_px: int         # 原分辨率下的缺陷像素数
    defect_ratio: float    # defect_px / (w*h)
    area_bucket: str       # 见 AREA_LABELS；正常视角为 ""


def _view_index(filename: str) -> int:
    matches = _VIEW_RE.findall(filename)
    if not matches:
        raise ValueError(f"文件名里找不到视角编号 C1..C5: {filename}")
    return int(matches[-1]) - 1


def _area_bucket(ratio: float) -> str:
    if ratio <= 0:
        return ""
    idx = int(np.digitize(ratio, AREA_BINS[1:-1], right=True))
    return AREA_LABELS[min(idx, len(AREA_LABELS) - 1)]


def _scan_group(job: tuple[str, str, str, str, str]) -> list[dict]:
    """扫描一个样本目录（5 个视角）。在子进程里跑。"""
    root, class_name, condition, defect_raw, group_rel = job
    group_dir = Path(root) / group_rel
    records: list[dict] = []

    images = sorted(
        p for p in group_dir.glob("*.png") if not p.name.endswith("_mask.png")
    )
    for img in images:
        mask = img.with_name(img.stem + "_mask.png")
        mask_file_exists = mask.exists()
        has_mask = mask_file_exists

        with Image.open(img) as im:
            width, height = im.size

        defect_px = 0
        if has_mask:
            with Image.open(mask) as mk:
                arr = np.asarray(mk)
            defect_px = int(np.count_nonzero(arr))
            if defect_px == 0:
                # 抽样时未见过，但不做假设：空 mask 一律按视角级正常处理，
                # 并在汇总里单独计数，便于发现数据集变更。
                has_mask = False

        ratio = defect_px / float(width * height) if width and height else 0.0
        records.append(
            asdict(
                ViewRecord(
                    class_name=class_name,
                    source_part_key=f"{class_name}/{group_dir.name}",
                    group_path=group_rel,
                    condition=condition,
                    defect_raw=defect_raw,
                    defect_norm=vocab.normalize(defect_raw) if defect_raw else "",
                    view_idx=_view_index(img.name),
                    image_path=str(img.relative_to(root)),
                    mask_file_exists=mask_file_exists,
                    mask_path=str(mask.relative_to(root)) if has_mask else "",
                    sample_label=1 if condition == "NG" else 0,
                    view_label=1 if has_mask else 0,
                    width=width,
                    height=height,
                    defect_px=defect_px,
                    defect_ratio=ratio,
                    area_bucket=_area_bucket(ratio),
                )
            )
        )
    return records


def _enumerate_groups(root: Path) -> list[tuple[str, str, str, str, str]]:
    """列出所有样本目录，不读图。"""
    jobs: list[tuple[str, str, str, str, str]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = class_dir.name

        ok_dir = class_dir / "OK"
        if ok_dir.is_dir():
            for grp in sorted(p for p in ok_dir.iterdir() if p.is_dir()):
                jobs.append((str(root), cls, "OK", "", str(grp.relative_to(root))))

        ng_dir = class_dir / "NG"
        if ng_dir.is_dir():
            for tdir in sorted(p for p in ng_dir.iterdir() if p.is_dir()):
                for grp in sorted(p for p in tdir.iterdir() if p.is_dir()):
                    jobs.append(
                        (str(root), cls, "NG", tdir.name, str(grp.relative_to(root)))
                    )
    return jobs


def build(root: Path, workers: int) -> pd.DataFrame:
    jobs = _enumerate_groups(root)
    print(f"[manifest] 样本目录 {len(jobs)} 个，{workers} 进程扫描中…", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, recs in enumerate(pool.map(_scan_group, jobs, chunksize=8), 1):
            rows.extend(recs)
            if i % 500 == 0:
                print(f"[manifest]   {i}/{len(jobs)} 目录", flush=True)

    df = pd.DataFrame(rows)
    # 稳定排序，保证 manifest 逐位可复现
    df = df.sort_values(
        ["class_name", "condition", "defect_raw", "group_path", "view_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> str:
    """汇总报告。刻意把"最容易犯错"的几个量放在最前面。"""
    ng = df[df.condition == "NG"]
    ng_views = len(ng)
    ng_with_mask = int(ng.view_label.sum())
    mask_files = int(ng.mask_file_exists.sum())
    empty_masks = mask_files - ng_with_mask
    no_mask_file = ng_views - mask_files
    blind = ng_views - ng_with_mask

    lines: list[str] = []
    add = lines.append

    add("=" * 74)
    add("manifest 汇总")
    add("=" * 74)
    add("")
    add("【口径 1】样本级 vs 视角级标签")
    add(f"  NG 样本目录            {ng.group_path.nunique():>7d}")
    add(f"  NG 视角图              {ng_views:>7d}")
    add(f"  mask 文件              {mask_files:>7d}")
    add(f"  其中非空 mask          {ng_with_mask:>7d}   → 视角级异常")
    add(f"  空 mask                    {empty_masks:>4d}   → 视角级正常")
    add(f"  无 mask 文件              {no_mask_file:>4d}   → 视角级正常")
    add(f"  视角级正常合计          {blind:>7d}")
    add(f"  占 NG 视角比例         {blind / max(ng_views, 1):>7.2%}")
    add("  ↑ 这批图若继承样本标签当异常训，decoder 会学成物体性而非缺陷性")
    add("")

    add("【口径 2】源 S 键 vs 样本目录")
    add(f"  源键 source_part_key   {df.source_part_key.nunique():>7d}")
    add(f"  样本目录 group_path    {df.group_path.nunique():>7d}")
    per_key = df.groupby("source_part_key").group_path.nunique()
    add(f"  每个源键的目录数       中位 {int(per_key.median())}，"
        f"最多 {int(per_key.max())}")
    add("  ↑ S 号跨批身份不可靠；本表只记录，不以它划分 train/dev")
    add("")

    add("【口径 3】分辨率按视角变化")
    piv = df.groupby("view_idx").width.median()
    add("  各视角中位边长：" + "  ".join(f"C{v + 1}={int(w)}" for v, w in piv.items()))
    add(f"  全局范围 {df.width.min()} ~ {df.width.max()}，"
        f"{df.groupby(['width', 'height']).ngroups} 种尺寸")
    nonsq = int((df.width != df.height).sum())
    add(f"  非正方形 {nonsq} 张（{nonsq / len(df):.2%}）")
    add("")

    add("【口径 4】类别不平衡")
    n_ok = int((df.sample_label == 0).sum())
    n_ng = int((df.sample_label == 1).sum())
    add(f"  视角图 正常 {n_ok} : 异常 {n_ng}  =  1 : {n_ng / max(n_ok, 1):.2f}")
    add(f"  视角级 正常 {int((df.view_label == 0).sum())} : "
        f"异常 {int((df.view_label == 1).sum())}  =  1 : "
        f"{df.view_label.sum() / max((df.view_label == 0).sum(), 1):.2f}")
    add("  ↑ 与工业现场相反；采样比例必须作为显式超参记录")
    add("")

    add("【缺陷面积分档】（仅视角级异常）")
    pos = df[df.view_label == 1]
    for label in AREA_LABELS:
        n = int((pos.area_bucket == label).sum())
        if n:
            add(f"  {label:>10s}  {n:>6d}  ({n / len(pos):>6.2%})")
    add(f"  缺陷面积占比 中位 {pos.defect_ratio.median():.4%}，"
        f"均值 {pos.defect_ratio.mean():.4%}")
    add("")

    add("【规模】")
    add(f"  类别 {df.class_name.nunique()}   视角图 {len(df)}   "
        f"mask {int(df.view_label.sum())}")
    add("=" * 74)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="构建视角级 manifest")
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/manifest.parquet"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"找不到数据集目录: {args.root}")

    df = build(args.root, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    report = summarize(df)
    print(report)
    args.out.with_name("manifest_report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n[manifest] 已写出 {args.out} 及 .csv / manifest_report.txt")


if __name__ == "__main__":
    main()
