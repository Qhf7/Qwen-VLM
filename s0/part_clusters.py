"""构建防泄漏用的视觉近重复组，并回填到视角级 manifest。

数据里 ``Sxxxx`` 只在批内有意义，不能作为跨 OK/NG 批次的物理工件 ID。
另一方面，仅凭图片内容也无法证明两个外观相同的工业件就是同一个实体。因此本模块
输出的是可审计的 ``visual_group_id``（视觉近重复组），不是宣称无误的物理身份。

安全方向
--------
拆分时必须让整个 visual group 进入同一个集合。身份匹配只允许发生在不同采集批
（OK / 各缺陷批）之间，并要求互为最近邻且距离低于阈值；一个分量内同一批最多
出现一次。这个约束避免“同批多个外观相似产品→跨批 chaining→整类并成一组”。

性能
----
旧实现显式构造 ``(n,n,4096)`` 广播差分，249 张图就会产生约 1 GiB 临时数组。
这里用 ``||x-y||² = ||x||² + ||y||² - 2 x·y``，只保留 ``(n,n)`` 距离矩阵；
同一轮聚类结果直接用于 parquet 与 manifest 回填，不再计算两遍。

产物
----
``artifacts/part_clusters.parquet`` 一行对应一个 C1 样本目录，包含：
``class_name, group_path, c1_path, visual_group_id, component_size, threshold``。

``artifacts/manifest_with_groups.parquet`` 在原 manifest 上增加
``visual_group_id``；后续拆分只读这一列，评测样本单位仍是 ``group_path``。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


THUMB_SIDE = 64
DEFAULT_THRESHOLD = 25.0
Image.MAX_IMAGE_PIXELS = None


def class_c1_keys(base: Path, class_name: str) -> list[tuple[str, str]]:
    """返回该类所有 ``(C1 路径, group_path)``，顺序稳定。"""
    class_dir = base / class_name
    groups: list[Path] = []
    ok_dir = class_dir / "OK"
    if ok_dir.is_dir():
        groups.extend(sorted(p for p in ok_dir.glob("S*") if p.is_dir()))
    ng_dir = class_dir / "NG"
    if ng_dir.is_dir():
        for defect_dir in sorted(p for p in ng_dir.iterdir() if p.is_dir()):
            groups.extend(sorted(p for p in defect_dir.glob("S*") if p.is_dir()))

    rows: list[tuple[str, str]] = []
    for group_dir in groups:
        candidates = sorted(
            p for p in group_dir.glob("*C1*.png") if "_mask" not in p.name
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{group_dir}: 期望恰好 1 张 C1 原图，实际 {len(candidates)}"
            )
        rows.append(
            (
                str(candidates[0].relative_to(base)),
                str(group_dir.relative_to(base)),
            )
        )
    return rows


def _decode_class(job: tuple[str, str]) -> tuple[str, list[tuple[str, str]], np.ndarray]:
    """子进程任务：解码一个类的 C1 灰度缩略图。"""
    base_s, class_name = job
    base = Path(base_s)
    keys = class_c1_keys(base, class_name)
    thumbnails: list[np.ndarray] = []
    for image_path, _ in keys:
        # 阈值 25 的历史审计全部基于 PIL convert("L")；OpenCV 灰度系数略有
        # 差异，单链连通分量会把微差放大，不能为了速度偷偷改变距离口径。
        with Image.open(base / image_path) as source:
            image = np.asarray(source.convert("L"))
        # 历史阈值实验使用 cv2.resize 的默认 INTER_LINEAR；必须保持同一标度。
        thumb = cv2.resize(image, (THUMB_SIDE, THUMB_SIDE), interpolation=cv2.INTER_LINEAR)
        thumbnails.append(thumb.astype(np.float32) / 255.0)
    xs = (
        np.stack(thumbnails)
        if thumbnails
        else np.empty((0, THUMB_SIDE, THUMB_SIDE), np.float32)
    )
    return class_name, keys, xs


def load_thumbnails(
    base: Path, workers: int
) -> dict[str, tuple[list[tuple[str, str]], np.ndarray]]:
    """并行读取各类缩略图；主进程按类接收并报告进度。"""
    classes = sorted(p.name for p in base.iterdir() if p.is_dir())
    result: dict[str, tuple[list[tuple[str, str]], np.ndarray]] = {}
    jobs = [(str(base), class_name) for class_name in classes]
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            class_name, keys, xs = _decode_class(job)
            result[class_name] = (keys, xs)
            print(f"[decode] {i:>2}/{len(jobs)} {class_name}: {len(keys)} 张", flush=True)
        return result

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_decode_class, job): job[1] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            class_name, keys, xs = future.result()
            result[class_name] = (keys, xs)
            print(f"[decode] {i:>2}/{len(jobs)} {class_name}: {len(keys)} 张", flush=True)
    return result


def pairwise_mse(thumbnails: np.ndarray) -> np.ndarray:
    """计算 64×64 灰度图的成对 MSE，返回旧实验使用的 ``×1000`` 标度。"""
    n = len(thumbnails)
    if n == 0:
        return np.empty((0, 0), np.float32)
    flat = np.ascontiguousarray(thumbnails.reshape(n, -1), dtype=np.float32)
    d = flat.shape[1]
    sq = np.einsum("ij,ij->i", flat, flat) / d
    distances = sq[:, None] + sq[None, :] - (2.0 / d) * (flat @ flat.T)
    np.maximum(distances, 0.0, out=distances)
    distances *= 1000.0
    np.fill_diagonal(distances, 0.0)
    return distances


def cluster_class(
    class_name: str,
    keys: list[tuple[str, str]],
    thumbnails: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """用跨批互为最近邻匹配构建视觉组。

    每个类约有 5 个采集批（OK + 4 种缺陷）。同一实体在一个批内最多出现一次，
    因此同批边不合法；跨批也只接受 mutual nearest neighbor，避免一对多误并。
    """
    distances = pairwise_mse(thumbnails)
    n = len(keys)

    def batch_of(group_path: str) -> str:
        parts = Path(group_path).parts
        return "OK" if parts[1] == "OK" else parts[2]

    batches = [batch_of(group_path) for _, group_path in keys]
    batch_indices: dict[str, list[int]] = {}
    for i, batch in enumerate(batches):
        batch_indices.setdefault(batch, []).append(i)

    # 候选边：每对批次中互为最近邻且低于阈值。
    candidates: list[tuple[float, int, int]] = []
    batch_names = sorted(batch_indices)
    for left_pos, left_name in enumerate(batch_names):
        left = batch_indices[left_name]
        for right_name in batch_names[left_pos + 1:]:
            right = batch_indices[right_name]
            block = distances[np.ix_(left, right)]
            left_to_right = np.argmin(block, axis=1)
            right_to_left = np.argmin(block, axis=0)
            for local_left, local_right in enumerate(left_to_right.tolist()):
                if int(right_to_left[local_right]) != local_left:
                    continue
                i, j = left[local_left], right[local_right]
                distance = float(distances[i, j])
                if distance <= threshold:
                    candidates.append((distance, i, j))

    parent = list(range(n))
    component_batches = [{batches[i]} for i in range(n)]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, i, j in sorted(candidates):
        ri, rj = find(i), find(j)
        if ri == rj or component_batches[ri] & component_batches[rj]:
            continue
        # 确定性 union：较小 root 作为根。
        if ri > rj:
            ri, rj = rj, ri
        parent[rj] = ri
        component_batches[ri] |= component_batches[rj]

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    ordered = sorted(members.values(), key=lambda idxs: min(keys[i][1] for i in idxs))
    stable_label = {old_i: new_i for new_i, idxs in enumerate(ordered) for old_i in idxs}
    if any(len({batches[i] for i in idxs}) != len(idxs) for idxs in ordered):
        raise AssertionError(f"{class_name}: 视觉组内出现重复采集批")

    rows: list[dict] = []
    for old_i, (c1_path, group_path) in enumerate(keys):
        component_id = stable_label[old_i]
        rows.append(
            {
                "class_name": class_name,
                "group_path": group_path,
                "c1_path": c1_path,
                "visual_group_id": f"{class_name}_vg{component_id:04d}",
                "component_size": len(ordered[component_id]),
                "threshold": float(threshold),
            }
        )
    print(
        f"[cluster] {class_name}: {len(ordered)} 组 / {len(keys)} 个目录，"
        f"最大组 {max((len(x) for x in ordered), default=0)}，"
        f"互近邻边 {len(candidates)}，阈值 {threshold:g}",
        flush=True,
    )
    return pd.DataFrame(rows)


def build(base: Path, threshold: float, workers: int) -> pd.DataFrame:
    decoded = load_thumbnails(base, workers)
    frames = [
        cluster_class(class_name, *decoded[class_name], threshold)
        for class_name in sorted(decoded)
    ]
    groups = pd.concat(frames, ignore_index=True)
    if groups.group_path.duplicated().any():
        duplicate = groups.loc[groups.group_path.duplicated(), "group_path"].iloc[0]
        raise AssertionError(f"group_path 重复: {duplicate}")
    return groups.sort_values(
        ["class_name", "group_path"], kind="mergesort"
    ).reset_index(drop=True)


def attach_visual_groups(manifest: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """一次 join 回填视觉组，并强校验全覆盖。"""
    manifest = manifest.copy()
    # 旧 manifest 的 part_uid 是由 S 号拼出的原始键，保留信息但纠正命名。
    if "part_uid" in manifest.columns and "source_part_key" not in manifest.columns:
        manifest = manifest.rename(columns={"part_uid": "source_part_key"})
    manifest = manifest.drop(columns=["visual_group_id"], errors="ignore")
    mapping = groups[["group_path", "visual_group_id"]]
    manifest = manifest.merge(mapping, on="group_path", how="left", validate="many_to_one")
    unknown = int(manifest.visual_group_id.isna().sum())
    if unknown:
        examples = (
            manifest.loc[manifest.visual_group_id.isna(), "group_path"]
            .drop_duplicates()
            .head(5)
        )
        raise AssertionError(
            f"{unknown} 行无法匹配 visual_group_id，例如 {examples.tolist()}"
        )
    return manifest


def write_audit(groups: pd.DataFrame, out_path: Path) -> Path:
    report_path = out_path.with_name("part_clusters_report.txt")
    lines = [
        "=" * 78,
        "视觉近重复组审计（防泄漏代理，不宣称是真实物理 ID）",
        "=" * 78,
        f"样本目录 {groups.group_path.nunique()}，视觉组 {groups.visual_group_id.nunique()}",
        "",
        "类别                         目录数  视觉组  最大组  单例组",
    ]
    for class_name, sub in groups.groupby("class_name", sort=True):
        sizes = sub.groupby("visual_group_id").size()
        lines.append(
            f"{class_name[:28]:28s} {len(sub):6d} {len(sizes):7d} "
            f"{int(sizes.max()):7d} {int((sizes == 1).sum()):7d}"
        )
    lines.extend(
        [
            "",
            "解释：视觉组必须整体进入 train/dev/holdout 中的一个集合。",
            "每组同一采集批最多一个成员；组大小不应超过该类采集批数量。",
            "拆分脚本只移动完整视觉组，并对 dev 的 OK/NG 数量做最低门槛校验。",
            "=" * 78,
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="构建视觉近重复组并回填 manifest")
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--workers", type=int,
        default=min(8, max(1, (os.cpu_count() or 4) - 2)),
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/part_clusters.parquet"))
    ap.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.parquet"))
    ap.add_argument(
        "--manifest-out", type=Path,
        default=Path("artifacts/manifest_with_groups.parquet"),
    )
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"找不到数据集目录: {args.root}")
    if not args.manifest.is_file():
        raise SystemExit(f"找不到 manifest: {args.manifest}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    groups = build(args.root, args.threshold, args.workers)
    groups.to_parquet(args.out, index=False)
    manifest = attach_visual_groups(pd.read_parquet(args.manifest), groups)
    manifest.to_parquet(args.manifest_out, index=False)
    report_path = write_audit(groups, args.out)

    print(f"[write] {args.out}（{len(groups)} 行）")
    print(f"[write] {args.manifest_out}（{len(manifest)} 行）")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
