"""【历史探索脚本，禁止用于正式拆分】批内视觉相似度探测。

仅凭内容不能断言“真实物理零件数”；正式实现与限制见 ``part_clusters.py``。

用 C1 视角缩略图的 MSE 做 union-find 近邻聚类，找出一个缺陷批里究竟
有几个"真实的物理零件"。这是 S0 的一个关键测量——它决定 split 的防泄漏单位。
"""
import glob
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def arr(p):
    a = np.asarray(Image.open(p).convert("L"))
    return cv2.resize(a, (64, 64)).astype(np.float32) / 255


def cluster_batch(base, defect, cls=1, threshold=30, max_s=50):
    ps = []
    for s in range(1, max_s + 1):
        g = glob.glob(f"{base}/NG/{defect}/S{s:04d}/*C1*.png")
        if g:
            ps.append((s, arr(g[0])))
    xs = np.array([b for _, b in ps])
    n = len(xs)
    # (n,n) MSE：对每对图像的全像素求平均
    m = ((xs[:, None, :, :] - xs[None, :, :, :]) ** 2).mean(axis=(-1, -2)) * 1000

    # union-find
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
            if float(m[i, j]) < threshold:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(ps[i][0])
    return len(groups), sorted(groups.values(), key=len, reverse=True), ps


if __name__ == "__main__":
    for defect in ["Fracture", "OilStain", "UnevenCoating", "FiberorBurrs"]:
        k, groups, _ = cluster_batch("Dataset/3_adapter", defect)
        print(f"3_adapter/{defect}: 50个S -> {k} 个物理零件组")
        sizes = [len(g) for g in groups]
        print(f"   组大小: {sizes}")
        print(f"   前5组: {groups[:5]}")
