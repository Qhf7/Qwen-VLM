"""像素级与图像级指标计算（像素指标采用定宽直方图法）。

所有指标都在"类别宏平均"口径下算：先按类算，再对类取平均。
P-AUROC / P-AUPR / P-F1max 用 256 桶直方图阈值法高精度近似，在 3669px 图上
不会爆内存。自检要求其与 sklearn 逐像素精确结果的误差小于 1e-3。

关键设计（回应 S0 的两条纪律）
----------------------------
1. **P-AUROC 不参与任何决策** —— 它只是输出，决策看 P-AUPR / P-F1max。
2. **面积占比 vs 小缺陷召回** —— 提供分档召回，用于判断小缺陷上到底行不行。
   面积分档用的是"缺陷占该视角图像面积"的比例，见 AREA_BINS。

"平均预测面积占比"也在这里算：它是识破"把整张图涂亮"最快的单一数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# 缺陷面积分档（占视角图像面积）。与 manifest.AREA_BINS 保持一致。
AREA_BINS = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
AREA_LABELS = ("<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%")

# 阈值个数。量化到 256 级足够精确，且远快于逐像素排序。
N_THRESHOLDS = 256


@dataclass
class PixelMetrics:
    """单个类别在**像素级**上的指标（宏平均会跨类别再平均）。

    注意：这些是 per-class 的，跨类别汇总时按类别取平均，不按像素数加权。
    """

    n_images: int = 0          # 该类像素级用到的视角图数
    n_pos_px: int = 0          # 该类的正例像素总数
    n_neg_px: int = 0          # 该类反例像素总数
    pred_pos_px: float = 0.0   # 平均每个视角预测为异常的面积（归一化）
    # 每个指标存 "raw / list"，因此可以跨列取平均
    p_auroc: list = field(default_factory=list)
    p_aupr: list = field(default_factory=list)
    p_f1max: list = field(default_factory=list)
    hit_at_1pct: list = field(default_factory=list)
    pred_area_ratio: list = field(default_factory=list)


@dataclass
class ImageMetrics:
    """单个类别在**图像级**上的指标（宏平均会跨类别再平均）。"""

    n_groups: int = 0
    i_auroc: list = field(default_factory=list)
    i_aupr: list = field(default_factory=list)
    f1max: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 直方图法阈值扫描
# ---------------------------------------------------------------------------

def _histogram_scan(pos_hist: np.ndarray, neg_hist: np.ndarray):
    """从最严到最松的阈值扫描，返回 (F1max, AUROC, AUPR, tpr, ppv)。

    pos_hist[i] / neg_hist[i] = 分数落在第 i 个 bin 的正例/负例数，
    bin 0 对应最低分、bin n-1 对应最高分。

    阈值方向：从最严（只看最高分 bin）逐步放宽（包含更低分）。
    放宽时 recall/FPR 单调升，precision 通常下降，因此：
      - AUROC 用梯形积分（TPR vs FPR）
      - AUPR 用矩形法则（与 sklearn.average_precision_score 一致）
    """
    n_bins = len(pos_hist)
    pos_cum = np.cumsum(pos_hist[::-1])[::-1]   # pos_cum[i] = 分数 ≥ bin i 的正例数
    neg_cum = np.cumsum(neg_hist[::-1])[::-1]
    total_pos = pos_cum[0]
    total_neg = neg_cum[0]
    eps = 1e-10

    tpr = pos_cum / (total_pos + eps)
    fpr = neg_cum / (total_neg + eps)
    ppv = pos_cum / (pos_cum + neg_cum + eps)
    f1 = 2 * tpr * ppv / (tpr + ppv + eps)
    f1max = float(np.max(f1))

    # AUROC：按阈值从高到低扫描，并显式补 (0,0)。这样常数分数的一次跳变
    # 正确得到 0.5，也不会依赖 argsort 对大量相同 FPR 的不稳定次序。
    roc_tpr = np.r_[0.0, tpr[::-1]]
    roc_fpr = np.r_[0.0, fpr[::-1]]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(integrate(roc_tpr, roc_fpr))

    # AUPR：从最严（只看最高分 bin）到最松，recall 单调升。
    # 每个阈值产生一个 (precision, recall) 点；AUPR = Σ (recall 增量 × 该点 precision)。
    # 必须把"最严阈值下首次发现的 recall"也算进去（否则全正例集中在最高 bin 时 AUPR=0）。
    aupr = 0.0
    prev_recall = 0.0
    for i in range(n_bins - 1, -1, -1):
        recall_step = max(0.0, tpr[i] - prev_recall)
        aupr += recall_step * ppv[i]
        prev_recall = max(prev_recall, tpr[i])
    return f1max, auroc, aupr, tpr, ppv


def _accumulate(pos_hist: np.ndarray, neg_hist: np.ndarray,
                pred: np.ndarray, gt: np.ndarray, n_bins: int):
    """把一个视角的 (预测热图, GT) 累加进已有的直方图。单趟遍历用。"""
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    pred = np.clip(pred, 0.0, 1.0)
    gt = gt.astype(bool)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    digit = np.clip(np.digitize(pred.ravel(), bins) - 1, 0, n_bins - 1)
    pos_hist += np.bincount(digit[gt.ravel()], minlength=n_bins).astype(np.float64)
    neg_hist += np.bincount(digit[~gt.ravel()], minlength=n_bins).astype(np.float64)
    return pos_hist, neg_hist


def finish_pixel_metrics(
    pos_hist: np.ndarray, neg_hist: np.ndarray,
    pred_ratios: list[float],
    n_images: int, hit_list: list[float],
) -> PixelMetrics:
    """由累加好的直方图收尾出 PixelMetrics。与 compute_pixel_metrics_per_class 结果一致。

    实现细节：histogram 法在整类合并时与逐视角平均不同（更稳，MVTec 做法），
    收尾时用同一套 _histogram_scan；hit 与面积占比是逐视角累积的 list。
    """
    m = PixelMetrics()
    m.n_images = n_images
    m.pred_area_ratio = pred_ratios
    if pred_ratios:
        m.pred_pos_px = float(np.mean(pred_ratios))
    m.n_pos_px = int(pos_hist.sum())
    m.n_neg_px = int(neg_hist.sum())
    m.p_auroc = []
    m.p_aupr = []
    m.p_f1max = []
    m.hit_at_1pct = list(hit_list)

    if pos_hist.sum() > 0:
        f1max, auroc, aupr, *_ = _histogram_scan(pos_hist.astype(np.float64),
                                                 neg_hist.astype(np.float64))
        m.p_auroc.append(float(auroc))
        m.p_aupr.append(float(aupr))
        m.p_f1max.append(f1max)
    else:
        m.p_auroc.append(np.nan)
        m.p_aupr.append(np.nan)
        m.p_f1max.append(np.nan)
    return m


def compute_pixel_metrics_per_class(
    preds: list[np.ndarray],  # 该类别所有异常视角的预测热图（float32，0..1）
    gts: list[np.ndarray],    # 对应 GT binary（bool），同网格
    n_bins: int = N_THRESHOLDS,
) -> PixelMetrics:
    """对一个类别计算像素级指标。调用方负责把预测与 GT 对齐（ResizePolicy.align）。

    gts 可能全 False（视角级正常），此时该视角不贡献正例，但会贡献反例，
    且作为"预测面积占比"分母的一部分 —— 这正是"平均预测面积占比"要的。
    """
    m = PixelMetrics()
    m.n_images = len(preds)
    pos_hists, neg_hists = [], []
    total_area_ratios: list[float] = []

    for pred, gt in zip(preds, gts):
        pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        pred = np.clip(pred, 0.0, 1.0)
        gt = gt.astype(bool)
        total = gt.size
        m.n_pos_px += int(gt.sum())
        m.n_neg_px += int(total - gt.sum())
        total_area_ratios.append(pred.mean())

        # 直方图法：量化 pred 到 [0, n_bins-1]，必须 clamp，否则 x=1.0 会落到第 n_bins 桶
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        digit = np.clip(np.digitize(pred.ravel(), bins) - 1, 0, n_bins - 1)
        pos_hist = np.bincount(digit[gt.ravel()], minlength=n_bins).astype(np.float64)
        neg_hist = np.bincount(digit[~gt.ravel()], minlength=n_bins).astype(np.float64)
        pos_hists.append(pos_hist)
        neg_hists.append(neg_hist)

    # 合并所有视角的直方图（同一个类别）算一次。
    pos_hist = np.sum(pos_hists, axis=0)
    neg_hist = np.sum(neg_hists, axis=0)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    if total_pos > 0:
        f1max, auroc, aupr, *_ = _histogram_scan(pos_hist, neg_hist)
        m.p_auroc.append(float(auroc))
        m.p_aupr.append(float(aupr))
        m.p_f1max.append(f1max)
    else:
        m.p_auroc.append(np.nan)
        m.p_aupr.append(np.nan)
        m.p_f1max.append(np.nan)

    # Hit@1%：预测分数最高的 1% 像素里，有多少是正例
    for pred, gt in zip(preds, gts):
        if gt.sum() == 0:
            m.hit_at_1pct.append(np.nan)
            continue
        flat = pred.ravel()
        k = max(1, int(0.01 * len(flat)))
        idx = np.argpartition(flat, -k)[-k:]
        m.hit_at_1pct.append(float(gt.ravel()[idx].mean()))

    m.pred_area_ratio = total_area_ratios
    m.pred_pos_px = float(np.mean(total_area_ratios))
    return m


__all__ = [
    "PixelMetrics",
    "ImageMetrics",
    "AREA_BINS",
    "AREA_LABELS",
    "N_THRESHOLDS",
    "compute_pixel_metrics_per_class",
    "compute_image_metrics_per_class",
    "finish_pixel_metrics",
    "_accumulate",
    "_histogram_scan",
]


def compute_image_metrics_per_class(
    group_scores: dict[str, float],   # 样本路径 -> 聚合后的 anomaly_score
    group_labels: dict[str, int],     # 样本路径 -> 0/1（样本级标签）
) -> ImageMetrics:
    """对一个类别计算图像级指标。

    score 是按样本聚合的（当前用 max 视角），label 是样本级（任一视角有缺陷 = 1）。
    """
    m = ImageMetrics()
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    if set(group_scores) != set(group_labels):
        raise ValueError("group_scores 与 group_labels 的键集合不一致")
    keys = sorted(group_scores)
    m.n_groups = len(keys)
    scores = np.array([group_scores[k] for k in keys], dtype=np.float64)
    labels = np.array([group_labels[k] for k in keys], dtype=np.int64)
    if len(scores) == 0 or labels.sum() == 0 or (labels == 0).sum() == 0:
        m.i_auroc.append(np.nan)
        m.i_aupr.append(np.nan)
        m.f1max.append(np.nan)
        return m

    m.i_auroc.append(float(roc_auc_score(labels, scores)))
    m.i_aupr.append(float(average_precision_score(labels, scores)))
    # sklearn 按唯一阈值处理 ties，避免把相同分数人为拆开得到虚高 F1。
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    m.f1max.append(float(f1.max()))
    return m


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _self_test():
    """用已知构造验证直方图法与 sklearn 一致，且 Oracle 得到满分。"""
    from sklearn.metrics import roc_auc_score, average_precision_score

    rng = np.random.default_rng(0)
    ok = True
    for trial in range(10):
        h, w = rng.integers(32, 512), rng.integers(32, 512)
        pred = rng.random((h, w)).astype(np.float32)
        gt = rng.random((h, w)) < 0.05

        m = compute_pixel_metrics_per_class([pred], [gt])
        f1h, auroch, auprh = m.p_f1max[0], m.p_auroc[0], m.p_aupr[0]
        aurocs = roc_auc_score(gt.ravel(), pred.ravel())
        auprs = average_precision_score(gt.ravel(), pred.ravel())
        if not (abs(auroch - aurocs) < 1e-3 and abs(auprh - auprs) < 1e-3):
            ok = False
            print(f"[self-test] 直方图 vs sklearn 不一致 trial={trial}: "
                  f"AUROC {auroch:.4f} vs {aurocs:.4f}, AUPR {auprh:.4f} vs {auprs:.4f}")
    print(f"[self-test] 直方图法与 sklearn 在 10 个随机样本上 "
          f"{'一致 ✓' if ok else '不一致 ✗'}")

    # Oracle：pred = gt，应得 ≈1
    gt = (rng.random((256, 256)) < 0.03).astype(float)
    mo = compute_pixel_metrics_per_class([gt.copy()], [gt.astype(bool)])
    print(f"[self-test] Oracle(perfect) P-AUROC={mo.p_auroc[0]:.4f} "
          f"P-AUPR={mo.p_aupr[0]:.4f} P-F1max={mo.p_f1max[0]:.4f} "
          f"(应≈1)")
    # Constant：pred = 0.5 恒定，应得随机水平 ≈ 0.5
    random_gt = (rng.random((256, 256)) < 0.03)
    mc = compute_pixel_metrics_per_class([np.full((256, 256), 0.5, np.float32)], [random_gt])
    print(f"[self-test] Constant P-AUROC={mc.p_auroc[0]:.4f} "
          f"(应≈0.5)  P-AUPR={mc.p_aupr[0]:.4f}")

    # 图像级：完美排序 / 反向排序 / ties。
    perfect = compute_image_metrics_per_class(
        {"ok": 0.0, "ng": 1.0}, {"ok": 0, "ng": 1}
    )
    reverse = compute_image_metrics_per_class(
        {"ok": 1.0, "ng": 0.0}, {"ok": 0, "ng": 1}
    )
    tied = compute_image_metrics_per_class(
        {"ok": 0.5, "ng": 0.5}, {"ok": 0, "ng": 1}
    )
    image_ok = (
        abs(perfect.i_auroc[0] - 1.0) < 1e-12
        and abs(perfect.i_aupr[0] - 1.0) < 1e-12
        and abs(reverse.i_auroc[0]) < 1e-12
        and abs(tied.i_auroc[0] - 0.5) < 1e-12
        and abs(tied.i_aupr[0] - 0.5) < 1e-12
    )
    print(f"[self-test] 图像级 perfect/reverse/ties {'一致 ✓' if image_ok else '不一致 ✗'}")
    ok &= image_ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _self_test()
