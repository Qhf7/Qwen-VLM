"""两套像素指标实现的一致性检查。

存在两套口径：
    PixelEvaluator（逐图累加，训练期用）
    PixelStats    （每图充分统计量，离线复评/bootstrap 用）
它们必须对同一批数据给出**逐位相同**的结果，否则论文表会出现"同一模型两个数"。

用法：
    python -m seg_cross_part.check_metrics
返回 {指标: 绝对差}，全部 < 1e-6 视为通过。
"""
from __future__ import annotations

import numpy as np

from .metrics_pixel import PixelEvaluator
from .stats_pixel import PixelStats, image_hist, image_pro


def compare_implementations(n_images: int = 12, size: int = 96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ev, st = PixelEvaluator(), PixelStats()
    for i in range(n_images):
        gt = np.zeros((size, size), bool)
        if i % 4 != 0:                                  # 3/4 是缺陷图
            r, c = rng.integers(5, size - 25, 2)
            gt[r:r + 16, c:c + 14] = True
        score = (gt * 0.45 + rng.random((size, size)) * 0.55).astype(np.float32)
        label = int(gt.any())
        ev.add(score, gt, label)
        neg, pos = image_hist(score, gt)
        pro, nreg = image_pro(score, gt)
        st.add(group=f"g{i // 4}", cls="c0", area="small", label=label,
               img_score=float(score.max()), neg=neg, pos=pos, pro=pro, n_regions=nreg)
    a, b = ev.compute(), st.metrics()
    return {k: abs(float(a[k]) - float(b[k])) for k in ("auroc", "aupr", "f1max", "apro_0.3")}


def main() -> None:
    d = compare_implementations()
    print("PixelEvaluator vs PixelStats：")
    for k, v in d.items():
        print(f"  {k:10s} Δ = {v:.2e}")
    ok = max(d.values()) < 1e-6
    print("通过" if ok else "不通过：两套实现已经漂移，先统一口径再跑实验")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
