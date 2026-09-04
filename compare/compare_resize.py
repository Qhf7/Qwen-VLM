"""Create before/after resize panels for one OK and one NG view per class.

The selection is restricted to fold1 Dev so this diagnostic does not read
holdout-zs data.  It uses the production preprocessing: INTER_AREA for RGB
and INTER_NEAREST for the binary GT mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIDE = 448


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    positive = mask.astype(bool)
    # Red overlay makes fine defects visible without hiding the component.
    result[positive, 0] = np.maximum(result[positive, 0], 235)
    result[positive, 1] *= 0.40
    result[positive, 2] *= 0.40
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_panel(row: pd.Series, out_path: Path, split_path: Path, dataset_root: Path) -> dict:
    image_path = Path(row.image_path)
    if not image_path.is_file():
        image_path = dataset_root / image_path
    mask_path = Path(row.mask_path) if bool(row.mask_file_exists) else None
    if mask_path is not None and not mask_path.is_file():
        mask_path = dataset_root / mask_path
    rgb = read_rgb(image_path)
    mask = read_mask(mask_path, rgb.shape[:2])
    rgb448 = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    mask448 = cv2.resize(mask, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)

    original_ratio = float(mask.mean())
    resized_ratio = float(mask448.mean())
    ratio_delta = resized_ratio - original_ratio
    relative_delta = ratio_delta / original_ratio if original_ratio > 0 else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [
        (rgb, f"Original image\n{rgb.shape[1]}×{rgb.shape[0]}"),
        (rgb448, "Model image\n448×448, INTER_AREA"),
        (mask, f"Original mask\narea={original_ratio:.4%}"),
        (mask448, f"Model mask\n448×448, INTER_NEAREST\narea={resized_ratio:.4%}"),
        (overlay(rgb, mask), "Original overlay\nmask in red"),
        (overlay(rgb448, mask448), "Model overlay\nmask in red"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    class_name = str(row.class_name)
    condition = str(row.condition)
    defect = str(row.defect_norm) if condition == "NG" else "normal"
    stem = out_path.stem.replace("_resize_compare", "")
    component_dir = out_path.parent
    cv2.imwrite(str(component_dir / f"{stem}_original.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(component_dir / f"{stem}_448.png"), cv2.cvtColor(rgb448, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(component_dir / f"{stem}_mask_original.png"), mask * 255)
    cv2.imwrite(str(component_dir / f"{stem}_mask_448.png"), mask448 * 255)
    fig.suptitle(
        f"{class_name} | {condition} | {defect} | view C{int(row.view_idx) + 1}\n"
        f"source: {image_path}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "class_name": class_name,
        "condition": condition,
        "defect_norm": str(row.defect_norm),
        "source_part_key": str(row.source_part_key),
        "view_idx": int(row.view_idx),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path else "",
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "original_mask_area_ratio": original_ratio,
        "resized_mask_area_ratio": resized_ratio,
        "mask_area_ratio_delta": ratio_delta,
        "mask_area_relative_delta": relative_delta,
        "resize_interpolation": "cv2.INTER_AREA",
        "mask_resize_interpolation": "cv2.INTER_NEAREST",
        "panel_path": str(out_path),
    }


def choose_one_per_class(df: pd.DataFrame) -> pd.DataFrame:
    # Use the first complete C1 view of the first group per class/condition.
    df = df[df.view_idx == 0].copy()
    choices = []
    for class_name in sorted(df.class_name.unique()):
        for condition in ("OK", "NG"):
            part = df[(df.class_name == class_name) & (df.condition == condition)]
            if part.empty:
                raise ValueError(f"missing {condition} example for {class_name}")
            choices.append(part.sort_values(["source_part_key", "image_path"]).iloc[0])
    return pd.DataFrame(choices).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("compare"))
    args = parser.parse_args()

    df = pd.read_parquet(args.split)
    chosen = choose_one_per_class(df)
    records = []
    for row in chosen.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        condition = str(row_series.condition)
        class_name = str(row_series.class_name)
        out_path = args.out_dir / class_name / f"{condition.lower()}_c1_resize_compare.png"
        records.append(draw_panel(row_series, out_path, args.split, args.dataset_root))

    summary = pd.DataFrame(records)
    summary.to_csv(args.out_dir / "resize_mask_summary.csv", index=False)
    chosen[["class_name", "condition", "defect_norm", "source_part_key", "view_idx", "image_path", "mask_path"]].to_csv(
        args.out_dir / "selected_samples.csv", index=False
    )
    readme = args.out_dir / "README.md"
    readme.write_text(
        "# Resize / mask comparison\n\n"
        "每个 fold1 Dev 类别选取一个 OK 和一个 NG 的 C1（view_idx=0）视角。\n\n"
        "- 原图：原始 PNG，不改变尺寸。\n"
        "- 模型图：`cv2.INTER_AREA` 直接缩放到 448×448，与当前 B0/E1 输入协议一致。\n"
        "- GT mask：原始 mask 与 `cv2.INTER_NEAREST` 缩放到 448×448，与训练/评测对齐代码一致。\n"
        "- 红色 overlay：帮助观察异常区域在缩放后是否仍可见。\n"
        "- `resize_mask_summary.csv`：记录原始与 448 mask 面积比例及变化；正常样本没有 GT mask 时按全零 mask 处理。\n\n"
        "本目录仅用于缩放诊断，不改变任何训练、评测或冻结配置。\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} panels to {args.out_dir}")
    print(summary[["class_name", "condition", "original_mask_area_ratio", "resized_mask_area_ratio", "mask_area_relative_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
