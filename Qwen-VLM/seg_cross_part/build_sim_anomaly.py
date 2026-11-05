"""模拟异常生成：cut-paste + Poisson 融合（AnomalyGPT 的做法）。

为什么需要：真实异常样本少且标注贵。把一张图的局部贴到另一张正常图上，
**mask 免费得到**（贴上去的区域），而且能控制缺陷的形状、大小、位置，
正好补上我们最缺的"小缺陷"和"长尾缺陷类型"。

关键细节（照 AnomalyGPT）：
    1. **随机形状**：不是直接贴矩形，而是在补丁上再用随机 blob 裁一次，
       避免模型学到"矩形拼接缝"；
    2. **Poisson 融合**（`cv2.seamlessClone`）：让补丁边缘颜色与背景连续，
       否则模型学的是边界不连续，而不是缺陷本身；
    3. **同类别取样**：补丁来自同一零件的其它图，保证纹理/颜色分布接近。

输出：
    data/sim/{fold}/images/*.png   合成图
    data/sim/{fold}/masks/*.png    GT（贴上去的区域）
    data/sim/{fold}/manifest.parquet

用法：
    python -m seg_cross_part.build_sim_anomaly --fold 0 --per-class 60
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import MANIFEST_PIXEL, ROOT, load_config, resolve_splits, with_defaults

try:
    import cv2
except Exception:                                    # 没有 cv2 就用 alpha 混合兜底
    cv2 = None


def blob_mask(h: int, w: int, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
    """在 h×w 内生成随机形状的布尔 mask（椭圆 / 多边形 / 条带）。"""
    kind = kind or rng.choice(["ellipse", "polygon", "stripe", "irregular"])
    m = np.zeros((h, w), np.uint8)
    if kind == "ellipse":
        cy, cx = h // 2, w // 2
        ry = int(h * rng.uniform(0.25, 0.45)); rx = int(w * rng.uniform(0.25, 0.45))
        cv2.ellipse(m, (cx, cy), (rx, ry), float(rng.uniform(0, 180)), 0, 360, 255, -1)
    elif kind == "polygon":
        k = int(rng.integers(4, 7))
        ang = np.sort(rng.uniform(0, 2 * math.pi, k))
        rad = rng.uniform(0.25, 0.48, k)
        pts = np.stack([w / 2 + np.cos(ang) * rad * w, h / 2 + np.sin(ang) * rad * h], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
    elif kind == "stripe":
        th = int(h * rng.uniform(0.08, 0.22))
        y0 = int(h * rng.uniform(0.2, 0.7))
        m[y0:y0 + th, :] = 255
    else:  # irregular：几个随机圆叠起来
        for _ in range(int(rng.integers(2, 5))):
            cy, cx = int(rng.integers(h // 4, 3 * h // 4)), int(rng.integers(w // 4, 3 * w // 4))
            r = int(min(h, w) * rng.uniform(0.12, 0.3))
            cv2.circle(m, (cx, cy), r, 255, -1)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))      # 收一点边，避免贴到裁片边界
    if m.sum() == 0:
        m[h // 3:2 * h // 3, w // 3:2 * w // 3] = 255
    return m > 0


def foreground(img: np.ndarray, thr: int = 235) -> np.ndarray:
    """粗略前景：不是接近纯白的像素（这批图都是白底）。"""
    return (img.astype(np.int16).max(axis=2) < thr) | (img.astype(np.int16).min(axis=2) < thr - 15)


def make_one(dst_img: np.ndarray, src_img: np.ndarray, rng: np.random.Generator,
             target_ratio: float | None = None):
    """从 src 取一小块贴到 dst 的**前景**上，返回 (合成图, mask)。

    target_ratio：目标缺陷面积占比（按真实数据分布采样），据此反推补丁大小。
    """
    H, W = dst_img.shape[:2]
    Sh, Sw = src_img.shape[:2]
    # 由目标占比反推补丁边长：ratio ≈ (patch/image)² × blob 覆盖率
    if target_ratio is None:
        target_ratio = float(rng.uniform(0.004, 0.04))
    f = float(np.clip(1.5 * math.sqrt(target_ratio / 0.7), 0.04, 0.5))
    ph, pw = max(12, int(H * f)), max(12, int(W * f))
    ph, pw = min(ph, Sh), min(pw, Sw)
    y0 = int(rng.integers(0, Sh - ph + 1)); x0 = int(rng.integers(0, Sw - pw + 1))
    patch = src_img[y0:y0 + ph, x0:x0 + pw].copy()
    m = blob_mask(ph, pw, rng)
    if m.sum() < 20:
        return None
    ys, xs = np.nonzero(m)
    patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ph, pw = patch.shape[:2]
    # 贴到前景上：随机试几次，要求候选区域 ≥80% 是零件本身
    fg = foreground(dst_img)
    dy = dx = None
    for _ in range(40):
        ty = int(rng.integers(0, H - ph + 1)); tx = int(rng.integers(0, W - pw + 1))
        if fg[ty:ty + ph, tx:tx + pw].mean() >= 0.8:
            dy, dx = ty, tx
            break
    if dy is None:                                   # 找不到就退回中心附近
        dy = int(H / 2 - ph / 2 + rng.integers(-H // 6, H // 6))
        dx = int(W / 2 - pw / 2 + rng.integers(-W // 6, W // 6))
        dy = max(0, min(H - ph, dy)); dx = max(0, min(W - pw, dx))
    canvas = dst_img.copy()
    mask = np.zeros((H, W), np.uint8)
    mask[dy:dy + ph, dx:dx + pw] = (m * 255).astype(np.uint8)
    if cv2 is not None:
        center = (dx + pw // 2, dy + ph // 2)
        try:
            # seamlessClone 会整体调整色彩平衡（背景会变灰），
            # 所以只把它在 mask 内的结果取回来，背景保持原样
            cloned = cv2.seamlessClone(patch, canvas, (m * 255).astype(np.uint8), center,
                                       cv2.NORMAL_CLONE)
            roi = canvas[dy:dy + ph, dx:dx + pw]
            roi[m] = cloned[dy:dy + ph, dx:dx + pw][m]
            canvas[dy:dy + ph, dx:dx + pw] = roi
        except Exception:
            canvas[dy:dy + ph, dx:dx + pw][m] = patch[m]
    else:                                            # 兜底：带羽化的 alpha 混合
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2)[..., None] if cv2 else m[..., None].astype(np.float32)
        roi = canvas[dy:dy + ph, dx:dx + pw].astype(np.float32)
        canvas[dy:dy + ph, dx:dx + pw] = (patch * alpha + roi * (1 - alpha)).astype(np.uint8)
    return canvas, mask > 127


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="protocol_c_unseen_category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sim")
    args = ap.parse_args()

    cfg = with_defaults(load_config(ROOT / "seg_cross_part" / "protocols" / f"{args.protocol}.yaml"))
    splits = resolve_splits(cfg, args.fold)
    src_root = ROOT / cfg["data"]["source_root"]
    df = pd.concat([pd.read_parquet(splits[s]).assign(split=s) for s in ("train", "val")])
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out / f"f{args.fold}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    # 真实缺陷面积分布，用于给合成缺陷定"该多大"
    real_ratio = pd.read_parquet(MANIFEST_PIXEL, columns=["defect_ratio", "view_label"])
    real_ratio = real_ratio[(real_ratio.view_label == 1) & (real_ratio.defect_ratio > 0)]
    real_ratio = real_ratio.defect_ratio.to_numpy()
    print(f"[area] 真实缺陷占比：n={len(real_ratio)} 中位数={np.median(real_ratio):.4f}")

    rows = []
    for cls, g in df.groupby("class_name"):
        normals = g[g.view_label == 0]
        if len(normals) < 4:
            continue
        n_made = 0
        tries = 0
        while n_made < args.per_class and tries < args.per_class * 10:
            tries += 1
            a = normals.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            # 补丁来源：同类、不同 group（避免拿同一张图自己贴自己）
            pool = normals[normals.group_path != a.group_path]
            if len(pool) == 0:
                continue
            b = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            dst = np.array(Image.open(src_root / a.image_path).convert("RGB"))
            src = np.array(Image.open(src_root / b.image_path).convert("RGB"))
            tgt = float(rng.choice(real_ratio))
            r = make_one(dst, src, rng, target_ratio=tgt)
            if r is None:
                continue
            img, mask = r
            name = f"{cls}_{n_made:04d}"
            Image.fromarray(img).save(out / "images" / f"{name}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
            ratio = float(mask.mean())
            rows.append({"class_name": cls, "image_path": f"sim/f{args.fold}/images/{name}.png",
                         "mask_path": f"sim/f{args.fold}/masks/{name}.png",
                         "defect_norm": "Synthetic", "view_label": 1, "mask_valid": True,
                         "area_bucket": ("small" if ratio < 0.0085 else
                                         "medium" if ratio < 0.0236 else "large"),
                         "defect_ratio": ratio, "split": "sim"})
            n_made += 1
        print(f"  {cls:28s} {n_made} 张", flush=True)

    man = pd.DataFrame(rows)
    man.to_parquet(out / "manifest.parquet")
    print(f"\n共 {len(man)} 张，按面积档：{man.area_bucket.value_counts().to_dict()}")
    print(f"已写出 {out}/manifest.parquet（图片路径相对 {cfg['data']['source_root']}）")


if __name__ == "__main__":
    main()
