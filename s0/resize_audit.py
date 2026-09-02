"""S0 resize 策略审计：量化 448 压缩的信息损失与滑窗成本。

这不是模型实验。它只回答两个在开训前必须冻结的问题：

1. 原图/GT 压到 448 后，细缺陷在目标网格上是否仍有可见核心；
2. 若改用 448 原尺度滑窗（默认 stride=336），每个 5 视角样本要付出多少窗口。

全量 manifest 用于精确统计缩放倍率和窗口数；实际 mask 重采样采用按
``class_name × area_bucket`` 分层的确定性样本，避免读取全部 17 GB 数据。
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _n_windows(length: int, side: int, stride: int) -> int:
    return max(1, math.ceil(max(0, length - side) / stride) + 1)


def _sample_strata(positive: pd.DataFrame, n_per_stratum: int) -> pd.DataFrame:
    rows = []
    for _, sub in positive.groupby(["class_name", "area_bucket"], sort=True):
        sub = sub.sort_values(["defect_ratio", "image_path"], kind="mergesort")
        n = min(n_per_stratum, len(sub))
        idx = np.unique(np.linspace(0, len(sub) - 1, n, dtype=int))
        rows.append(sub.iloc[idx])
    return pd.concat(rows, ignore_index=True)


def _mask_job(job: tuple[str, dict, int]) -> dict:
    root_s, row, side = job
    mask = cv2.imread(str(Path(root_s) / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"无法读取 mask: {row['mask_path']}")
    mask = (mask > 0).astype(np.float32)
    h, w = mask.shape

    direct = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)
    scale = min(side / w, side / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    letterbox = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def stats(array: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_support_px": int((array > 0).sum()),
            f"{prefix}_core_px": int((array >= 0.5).sum()),
            f"{prefix}_mass_px": float(array.sum()),
        }

    return {
        "class_name": row["class_name"],
        "area_bucket": row["area_bucket"],
        "image_path": row["image_path"],
        "width": w,
        "height": h,
        "native_defect_px": int(mask.sum()),
        **stats(direct, "direct"),
        **stats(letterbox, "letterbox"),
    }


def audit(
    manifest: pd.DataFrame,
    dataset_root: Path,
    side: int,
    stride: int,
    n_per_stratum: int,
    workers: int,
) -> tuple[str, pd.DataFrame]:
    views = manifest.copy()
    views["scale_448"] = np.minimum(side / views.width, side / views.height)
    views["tile_windows"] = [
        _n_windows(int(w), side, stride) * _n_windows(int(h), side, stride)
        for w, h in zip(views.width, views.height)
    ]
    sample_windows = views.groupby("group_path").tile_windows.sum()

    positive = views[views.view_label == 1]
    sampled = _sample_strata(positive, n_per_stratum)
    jobs = [(str(dataset_root), row, side) for row in sampled.to_dict("records")]
    if workers <= 1:
        mask_rows = [_mask_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            mask_rows = list(pool.map(_mask_job, jobs, chunksize=8))
    mask_audit = pd.DataFrame(mask_rows)

    direct_lost = float((mask_audit.direct_core_px == 0).mean())
    letter_lost = float((mask_audit.letterbox_core_px == 0).mean())
    direct_tiny = float((mask_audit.direct_core_px < 4).mean())
    letter_tiny = float((mask_audit.letterbox_core_px < 4).mean())

    lines = [
        "=" * 86,
        "S0 resize 策略审计",
        "=" * 86,
        f"全量视角 {len(views)}；异常视角 {len(positive)}；实际 mask 分层样本 {len(mask_audit)}",
        f"目标边长 {side}；滑窗 stride {stride}（重叠 {1 - stride / side:.0%}）",
        "",
        "【全量缩放倍率】",
        f"  线性 scale p05/p50/p95 = {np.percentile(views.scale_448, 5):.3f} / "
        f"{np.percentile(views.scale_448, 50):.3f} / {np.percentile(views.scale_448, 95):.3f}",
        f"  下采样 ≥4×（scale≤0.25） {float((views.scale_448 <= .25).mean()):.2%}",
        f"  下采样 ≥8×（scale≤0.125） {float((views.scale_448 <= .125).mean()):.2%}",
        f"  非正方形 {int((views.width != views.height).sum())} 张；仅这些图 direct 与 letterbox 几何不同",
        "",
        "【mask 分层样本：448 网格可见核心（AREA 重采样后覆盖率≥0.5）】",
        f"  direct resize：核心完全消失 {direct_lost:.2%}；核心 <4 px {direct_tiny:.2%}",
        f"  letterbox：    核心完全消失 {letter_lost:.2%}；核心 <4 px {letter_tiny:.2%}",
        "  注：正式 GT 对齐仍用 nearest-neighbor；这里的 AREA+0.5 专门诊断图像压缩后",
        "      缺陷是否有足够像素覆盖，不用于篡改评测标签。",
        "",
        "【448 原尺度滑窗成本】",
        f"  每视角窗口数 p50/p95/max = {np.percentile(views.tile_windows, 50):.0f} / "
        f"{np.percentile(views.tile_windows, 95):.0f} / {views.tile_windows.max():.0f}",
        f"  每个5视角样本窗口数 p50/p95/max = {np.percentile(sample_windows, 50):.0f} / "
        f"{np.percentile(sample_windows, 95):.0f} / {sample_windows.max():.0f}",
        "",
        "【按类审计】",
        "类别                         scale中位  窗口/视角中位  direct核心消失  样本N",
    ]
    for class_name in sorted(views.class_name.unique()):
        class_views = views[views.class_name == class_name]
        class_masks = mask_audit[mask_audit.class_name == class_name]
        lost = float((class_masks.direct_core_px == 0).mean()) if len(class_masks) else float("nan")
        lines.append(
            f"{class_name[:28]:28s} {class_views.scale_448.median():9.3f} "
            f"{class_views.tile_windows.median():14.0f} {lost:15.2%} {len(class_masks):7d}"
        )

    lines.extend(
        [
            "",
            "【冻结建议】",
            "  主评测网格固定 448，GT 用 nearest-neighbor 对齐；这是跨实验统一口径。",
            "  模型输入先用 direct 448 作为 B0/E1 可归因基线；letterbox 只对 0.55% 非方图有意义。",
            "  原尺度滑窗保留为 B0w 独立消融，不能与 prompt/模型改动叠加；其窗口数直接进入延迟报告。",
            "  同一预测另报原分辨率对照，用来区分模型失败与 448 压缩损失。",
            "=" * 86,
        ]
    )
    return "\n".join(lines), mask_audit


def main() -> None:
    ap = argparse.ArgumentParser(description="审计 resize 信息损失与滑窗成本")
    ap.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/manifest_with_groups.parquet"),
    )
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--stride", type=int, default=336)
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out", type=Path, default=Path("artifacts/resize_audit.txt"))
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    text, mask_audit = audit(
        manifest, args.root, args.side, args.stride, args.per_stratum, args.workers
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    mask_audit.to_parquet(args.out.with_suffix(".parquet"), index=False)
    print(text)
    print(f"[write] {args.out} / {args.out.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
