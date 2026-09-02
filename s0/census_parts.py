"""【历史探索脚本，禁止用于正式拆分】普查批内视觉簇数量。

正式实现与口径见 ``part_clusters.py`` / ``FINDINGS.md``。本文件保留早期调查过程，
其中“真实物理零件”措辞并不成立，且无约束单链会 chaining。

Sxxxx 不是物理零件 id。用 C1 视角 64x64 灰度缩略图的邻接聚类
(union-find, MSE 阈值) 把每个批压缩成"真实物理零件组"。

输出：每个 (类, 批) 的 distinct 零件数, 以及全数据集的分布。
这决定 split 的防泄漏单位——不是 S，而是聚类出来的物理零件。
"""
import glob
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
THRESHOLD = 30.0   # within~2-8, across~877, 100x 间隙里安全


def arr(p):
    a = np.asarray(Image.open(p).convert("L"))
    return cv2.resize(a, (64, 64)).astype(np.float32) / 255


def cluster_images(img_list):
    """img_list: list of (label, image_array)。返回 {label: group_id}。"""
    xs = np.array([b for _, b in img_list])
    n = len(xs)
    if n == 0:
        return {}, 0
    m = ((xs[:, None, :, :] - xs[None, :, :, :]) ** 2).mean(axis=(-1, -2)) * 1000
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if float(m[i, j]) < THRESHOLD:
                union(i, j)
    groups = {}
    for i in range(n):
        groups[img_list[i][0]] = find(i)
    return groups, len(set(groups.values()))


def batch_imgs(base, cls, condition, defect=""):
    """取一个批的 C1 缩略图 {(cls,cond,defect,S): arr}。"""
    out = []
    if condition == "OK":
        pat = f"{base}/{cls}/{condition}/S*/*C1*.png"
        for p in glob.glob(pat):
            if "_mask" in p:
                continue
            s = os.path.basename(os.path.dirname(p))
            out.append(((cls, "OK", "", s), arr(p)))
    else:
        pat = f"{base}/{cls}/NG/{defect}/S*/*C1*.png"
        for p in glob.glob(pat):
            if "_mask" in p:
                continue
            s = os.path.basename(os.path.dirname(p))
            out.append(((cls, "NG", defect, s), arr(p)))
    return out


def main():
    base = "Dataset"
    classes = sorted(d for d in os.listdir(base) if os.path.isdir(f"{base}/{d}"))
    per_batch = []
    ok_batches = []
    glass = {}
    for cls in classes:
        # OK
        imgs = batch_imgs(base, cls, "OK")
        groups, k = cluster_images(imgs)
        ok_batches.append((cls, k, len(imgs)))
        # NG 各缺陷
        for defect in sorted(os.listdir(f"{base}/{cls}/NG")):
            if defect.startswith("."):
                continue
            imgs = batch_imgs(base, cls, "NG", defect)
            if not imgs:
                continue
            groups, k = cluster_images(imgs)
            per_batch.append((cls, defect, k, len(imgs)))

    print("== NG 批: 每个批 S 数 -> 真实物理零件数 ==")
    ratio = []
    for cls, d, k, n in per_batch:
        ratio.append(k / n)
    print(f"  批总数 {len(per_batch)}（24类×4缺陷={24*4} 期望）")
    print(f"  每个批的 distinct 零件数: min={min(x[2] for x in per_batch)} "
          f"max={max(x[2] for x in per_batch)} 中位={sorted(x[2] for x in per_batch)[len(per_batch)//2]}")
    print(f"  压缩比: 平均每个批只有 {np.mean([x[2] for x in per_batch]):.1f} 个真实零件 "
          f"(S 数 {np.mean([x[3] for x in per_batch]):.1f})")
    print()
    print("== OK 批 ==")
    print("  每类 OK 的 distinct 零件数:", sorted(x[1] for x in ok_batches))
    print(f"  中位 {sorted(x[1] for x in ok_batches)[len(ok_batches)//2]}")
    print()
    # 每类真实 distinct 数（OK + 各NG批 去重 —— 但跨批可能重复，这里粗略）
    print("== 全数据集真实物理零件的上界 ≈ ==")
    from collections import defaultdict
    class_batches = defaultdict(int)
    for cls, d, k, n in per_batch:
        class_batches[cls] += k
    for cls, k, n in ok_batches:
        class_batches[cls] += k
    print("  各类的批级 distinct 总和:", dict(sorted(class_batches.items())))
    print(f"  总和 {sum(class_batches.values())}（跨批未去重，是上界）")


if __name__ == "__main__":
    main()
