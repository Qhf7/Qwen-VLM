"""【历史探索脚本，禁止用于正式拆分】无约束全局单链聚类。

它会发生 chaining，不能得到“真实物理零件数”。正式实现见 ``part_clusters.py``。

关键：跨批的 S0001 可能同part也可能不同part（已确认 3_adapter 里
OK≈OilStain≈FiberorBurrs 但 ≠Fracture≈UnevenCoating）。所以不能按批单独聚，
要把整个类内的所有图像放进同一个聚类，看真正的 distinct 物理零件数。

只在 C1 上看，因为 C1 是最高分辨率 RGB 视角。
"""
import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
THRESHOLD = 30.0


def arr(p):
    a = np.asarray(Image.open(p).convert("L"))
    return cv2.resize(a, (64, 64)).astype(np.float32) / 255


def cluster_global(base, cls):
    """返回 (groups, sample_map)。sample_map: (batch,S) -> group_id。"""
    imgs = []  # (key, arr)
    for batch in ["OK"] + sorted(os.listdir(f"{base}/{cls}/NG")):
        if batch.startswith("."):
            continue
        bpath = f"{base}/{cls}/{'OK' if batch == 'OK' else 'NG'}"
        dpath = bpath if batch == "OK" else f"{bpath}/{batch}"
        for p in sorted(glob.glob(f"{dpath}/S*/*C1*.png")):
            if "_mask" in p:
                continue
            s = os.path.basename(os.path.dirname(p))
            key = (batch, s)
            # 一个批内同 S 只取一个（C1 有很多？）—— 这里每个 S 目录取第一个 C1
            if any(k == key for k, _ in imgs):
                continue
            imgs.append((key, arr(p)))
    xs = np.array([b for _, b in imgs])
    n = len(xs)
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
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(imgs[i][0])
    return len(groups), groups


if __name__ == "__main__":
    base = "Dataset"
    classes = sorted(d for d in os.listdir(base) if os.path.isdir(f"{base}/{d}"))
    totals = []
    for cls in classes:
        k, groups = cluster_global(base, cls)
        totals.append((cls, k))
        sizes = sorted(len(g) for g in groups.values())
        print(f"{cls}: {k:3d} 个真实物理零件  (组大小 {sizes})")
    print(f"\n全部 {len(classes)} 类, 真实零件数 均值 {np.mean([t[1] for t in totals]):.1f}, "
          f"总和 {sum(t[1] for t in totals)}")
    print("S0 现阶段按 S 号(1200) 划是错的；真实 distinct 远少于这个数。")
