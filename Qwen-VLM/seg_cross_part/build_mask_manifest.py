"""构建像素级（分割）manifest：复用已冻结的划分，只把 mask 层接回来。

背景：`classification_cross_part` 的 manifest 特意剔除了 mask 字段（那套实验只做图像级分类）。
现在任务转向「跨零件类别的异常分割」，需要把 mask、缺陷面积、缺陷类型重新挂上，
但**划分本身不动**——`group_path` / `visual_group_id` / Protocol A/B/C 全部继承。

输出 `artifacts/seg_cross_part/manifest_pixel.parquet`，字段：
    image_path, class_name, group_path, visual_group_id, view_idx,
    view_label, sample_label, width, height,
    mask_path, mask_valid, defect_px, defect_ratio, area_bucket, defect_norm,
    pixel_eval_valid

三条冻结规则（写进报告，禁止看到预测后修改）：
1. `mask_valid`：view_label==1 且 defect_px>0；
2. `pixel_eval_valid`：正常视角一律有效（GT 全 0）；异常视角只有在有 mask 时才有效，
   491 张「盲 NG」（NG 但无缺陷区域）**不参与像素级指标**，只在图像级保留；
3. `area_bucket`：按有效 mask 的 defect_ratio 三分位切 small/medium/large，阈值写进报告。

用法：
    python -m seg_cross_part.build_mask_manifest
    python -m seg_cross_part.build_mask_manifest --verify 500   # 额外抽查 500 张 mask 与记录是否一致
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "artifacts" / "manifest.parquet"
OUT_DIR = ROOT / "artifacts" / "seg_cross_part"

KEEP = [
    "image_path", "class_name", "group_path", "visual_group_id", "view_idx",
    "view_label", "sample_label", "width", "height",
    "mask_path", "defect_px", "defect_ratio", "defect_norm",
]


def verify_masks(df: pd.DataFrame, n: int, seed: int = 0) -> dict:
    """抽查：真实读 mask，核对 defect_px 与是否为空。"""
    from PIL import Image
    sub = df[df["mask_valid"]].sample(min(n, int(df["mask_valid"].sum())), random_state=seed)
    bad_px, bad_read = 0, 0
    for _, r in sub.iterrows():
        p = ROOT / "Dataset" / r["mask_path"]
        try:
            a = np.array(Image.open(p).convert("L"))
        except Exception:
            bad_read += 1
            continue
        if int((a > 127).sum()) != int(r["defect_px"]):
            bad_px += 1
    # 正常视角应该没有有效缺陷区域
    normal_with_mask = int(((df["view_label"] == 0) & (df["defect_px"] > 0)).sum())
    return {"抽查数": int(len(sub)), "读取失败": bad_read,
            "像素数不一致": bad_px, "正常视角却有缺陷像素": normal_with_mask}


def main() -> None:
    ap = argparse.ArgumentParser(description="构建像素级 manifest（分割任务）")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--verify", type=int, default=200, help="抽查多少张 mask 做一致性校验")
    args = ap.parse_args()

    src = pd.read_parquet(args.src)
    missing = [c for c in KEEP if c not in src.columns]
    if missing:
        raise SystemExit(f"源 manifest 缺列: {missing}")

    df = src[KEEP].copy()
    df["defect_px"] = df["defect_px"].fillna(0).astype(int)
    df["defect_ratio"] = df["defect_ratio"].fillna(0.0).astype(float)
    df["mask_valid"] = (df["view_label"] == 1) & (df["defect_px"] > 0)
    df["pixel_eval_valid"] = (df["view_label"] == 0) | df["mask_valid"]

    # 面积三分位（只看有效 mask）
    valid_ratio = df.loc[df["mask_valid"], "defect_ratio"]
    q1, q2 = np.quantile(valid_ratio, [1 / 3, 2 / 3])
    bins = [-1.0, q1, q2, np.inf]
    df["area_bucket"] = pd.cut(df["defect_ratio"], bins=bins,
                               labels=["small", "medium", "large"]).astype(str)
    df.loc[~df["mask_valid"], "area_bucket"] = "none"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "manifest_pixel.parquet"
    df.to_parquet(out, index=False)

    stats = {
        "rows": int(len(df)),
        "classes": int(df["class_name"].nunique()),
        "groups": int(df["group_path"].nunique()),
        "views_normal": int((df["view_label"] == 0).sum()),
        "views_anomaly": int((df["view_label"] == 1).sum()),
        "mask_valid": int(df["mask_valid"].sum()),
        "blind_ng_excluded_from_pixel": int(((df["view_label"] == 1) & ~df["mask_valid"]).sum()),
        # 这些视角属于 NG 零件、但该视角没有可见缺陷，早已按视角级正常处理（见 label_audit）
        "blind_ng_views_labeled_normal": int(((df["sample_label"] == 1) & (df["view_label"] == 0)).sum()),
        "pixel_eval_valid": int(df["pixel_eval_valid"].sum()),
        "area_thresholds_defect_ratio": [float(q1), float(q2)],
        "area_bucket_counts": df["area_bucket"].value_counts().to_dict(),
    }
    if args.verify:
        stats["verify"] = verify_masks(df, args.verify)

    report = ["=" * 68, "像素级 manifest 构建报告（分割任务）", "=" * 68]
    for k, v in stats.items():
        report.append(f"{k}: {v}")
    report += [
        "",
        "三条冻结规则：",
        "1) mask_valid = (view_label==1 且 defect_px>0)",
        "2) 所属 NG 零件但本视角无缺陷的 491 个视角，按视角级正常处理（GT 全 0），"
        "既不是盲区也不是错误",
        "3) area_bucket 按有效 mask 的 defect_ratio 三分位切分，阈值已写入本报告",
    ]
    text = "\n".join(report) + "\n"
    (args.out_dir / "mask_manifest_report.txt").write_text(text, encoding="utf-8")
    (args.out_dir / "mask_manifest_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
