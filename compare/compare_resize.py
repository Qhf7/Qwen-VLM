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


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def mask_bbox_edges(mask: np.ndarray) -> np.ndarray | None:
    """Return left/top/right/bottom pixel-edge coordinates; right/bottom are exclusive."""
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float64)


def overlap_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = int(np.logical_and(a, b).sum())
    total = int(a.sum() + b.sum())
    union = int(np.logical_or(a, b).sum())
    dice = 1.0 if total == 0 else 2.0 * intersection / total
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (mask > 0) & (eroded == 0)


def boundary_distances(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Symmetric Hausdorff and mean boundary distance in source-image pixels."""
    boundary_a = mask_boundary(a)
    boundary_b = mask_boundary(b)
    if not boundary_a.any() and not boundary_b.any():
        return 0.0, 0.0
    if not boundary_a.any() or not boundary_b.any():
        return float("inf"), float("inf")
    distance_to_b = cv2.distanceTransform(
        (~boundary_b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_to_a = cv2.distanceTransform(
        (~boundary_a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    a_to_b = distance_to_b[boundary_a]
    b_to_a = distance_to_a[boundary_b]
    hausdorff = max(float(a_to_b.max()), float(b_to_a.max()))
    mean_symmetric = (float(a_to_b.mean()) + float(b_to_a.mean())) / 2.0
    return hausdorff, mean_symmetric


def geometry_diagnostics(mask: np.ndarray, mask448: np.ndarray) -> tuple[dict, np.ndarray]:
    height, width = mask.shape
    roundtrip = cv2.resize(mask448, (width, height), interpolation=cv2.INTER_NEAREST)
    original_center = mask_centroid(mask)
    resized_center = mask_centroid(mask448)
    original_bbox = mask_bbox_edges(mask)
    resized_bbox = mask_bbox_edges(mask448)

    empty = original_center is None
    if empty:
        centroid_dx_448 = centroid_dy_448 = centroid_error_448 = np.nan
        centroid_dx_original = centroid_dy_original = centroid_error_original = np.nan
        bbox_errors_448 = np.full(4, np.nan)
        bbox_errors_original = np.full(4, np.nan)
    else:
        scale = np.array([SIDE / width, SIDE / height], dtype=np.float64)
        # OpenCV maps pixel centers as (dst + 0.5) / scale - 0.5.
        expected_center_448 = (original_center + 0.5) * scale - 0.5
        center_delta_448 = resized_center - expected_center_448
        centroid_dx_448, centroid_dy_448 = center_delta_448
        centroid_error_448 = float(np.linalg.norm(center_delta_448))

        roundtrip_center = mask_centroid(roundtrip)
        center_delta_original = roundtrip_center - original_center
        centroid_dx_original, centroid_dy_original = center_delta_original
        centroid_error_original = float(np.linalg.norm(center_delta_original))

        expected_bbox_448 = original_bbox * np.array(
            [SIDE / width, SIDE / height, SIDE / width, SIDE / height], dtype=np.float64
        )
        bbox_errors_448 = resized_bbox - expected_bbox_448
        bbox_errors_original = mask_bbox_edges(roundtrip) - original_bbox

    original_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    resized_components, _, _, _ = cv2.connectedComponentsWithStats(
        mask448.astype(np.uint8), connectivity=8
    )
    original_components -= 1
    resized_components -= 1
    if original_components:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        smallest_label = int(np.argmin(component_areas)) + 1
        resized_labels = cv2.resize(
            labels.astype(np.float32), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        surviving_labels = set(np.unique(resized_labels).tolist()) - {0}
        vanished_labels = set(range(1, original_components + 1)) - surviving_labels
        smallest_area = int(component_areas.min())
        smallest_survives = smallest_label in surviving_labels
        vanished_count = len(vanished_labels)
        smallest_vanished_area = (
            min(int(stats[label, cv2.CC_STAT_AREA]) for label in vanished_labels)
            if vanished_labels
            else 0
        )
    else:
        smallest_area = 0
        smallest_survives = True
        vanished_count = 0
        smallest_vanished_area = 0

    dice, iou = overlap_metrics(mask, roundtrip)
    hausdorff, mean_boundary = boundary_distances(mask, roundtrip)
    diagonal = float(np.hypot(width, height))
    metrics = {
        "centroid_dx_448_px": centroid_dx_448,
        "centroid_dy_448_px": centroid_dy_448,
        "centroid_error_448_px": centroid_error_448,
        "centroid_dx_original_px": centroid_dx_original,
        "centroid_dy_original_px": centroid_dy_original,
        "centroid_error_original_px": centroid_error_original,
        "bbox_left_error_448_px": bbox_errors_448[0],
        "bbox_top_error_448_px": bbox_errors_448[1],
        "bbox_right_error_448_px": bbox_errors_448[2],
        "bbox_bottom_error_448_px": bbox_errors_448[3],
        "bbox_max_abs_error_448_px": float(np.nanmax(np.abs(bbox_errors_448))) if not empty else np.nan,
        "bbox_left_error_original_px": bbox_errors_original[0],
        "bbox_top_error_original_px": bbox_errors_original[1],
        "bbox_right_error_original_px": bbox_errors_original[2],
        "bbox_bottom_error_original_px": bbox_errors_original[3],
        "bbox_max_abs_error_original_px": float(np.nanmax(np.abs(bbox_errors_original))) if not empty else np.nan,
        "original_connected_components": original_components,
        "resized_connected_components": resized_components,
        "connected_component_count_delta": resized_components - original_components,
        "vanished_original_components": vanished_count,
        "smallest_original_component_area_px": smallest_area,
        "smallest_component_survives": bool(smallest_survives),
        "smallest_vanished_component_area_px": smallest_vanished_area,
        "roundtrip_dice": dice,
        "roundtrip_iou": iou,
        "boundary_hausdorff_original_px": hausdorff,
        "boundary_mean_symmetric_original_px": mean_boundary,
        "boundary_hausdorff_normalized": hausdorff / diagonal,
        "boundary_mean_symmetric_normalized": mean_boundary / diagonal,
    }
    return metrics, roundtrip


def draw_panel(row: pd.Series, out_path: Path, dataset_root: Path) -> dict:
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
    geometry, roundtrip = geometry_diagnostics(mask, mask448)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    panels = [
        (rgb, f"Original image\n{rgb.shape[1]}×{rgb.shape[0]}"),
        (rgb448, "Model image\n448×448, INTER_AREA"),
        (mask, f"Original mask\narea={original_ratio:.4%}"),
        (mask448, f"Model mask\n448×448, INTER_NEAREST\narea={resized_ratio:.4%}"),
        (overlay(rgb, mask), "Original overlay\nmask in red"),
        (overlay(rgb448, mask448), "Model overlay\nmask in red"),
        (roundtrip, f"448 mask mapped back\n{rgb.shape[1]}×{rgb.shape[0]}"),
        (overlay(rgb, roundtrip), "Roundtrip overlay\nmask in red"),
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
    cv2.imwrite(str(component_dir / f"{stem}_mask_448_back_to_original.png"), roundtrip * 255)
    geometry_title = ""
    if condition == "NG":
        geometry_title = (
            f" | center error={geometry['centroid_error_448_px']:.3f}px@448"
            f" | Dice={geometry['roundtrip_dice']:.4f}"
        )
    fig.suptitle(
        f"{class_name} | {condition} | {defect} | view C{int(row.view_idx) + 1}\n"
        f"source: {image_path}{geometry_title}",
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
        **geometry,
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
        records.append(draw_panel(row_series, out_path, args.dataset_root))

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
        "- `resize_mask_summary.csv`：记录面积、中心点、边界框、连通域、反向映射 Dice/IoU 和边界距离；正常样本没有 GT mask 时按全零 mask 处理。\n\n"
        "本目录仅用于缩放诊断，不改变任何训练、评测或冻结配置。\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} panels to {args.out_dir}")
    print(summary[["class_name", "condition", "original_mask_area_ratio", "resized_mask_area_ratio", "mask_area_relative_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
