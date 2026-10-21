"""像素级异常分割指标（基准定义 v1.0，2026-10-21 冻结）。

设计原则：**每个数字都必须能追到一条明确的定义**，且与工业异常检测文献可比。

--------------------------------------------------------------------------
指标定义
--------------------------------------------------------------------------
1. P-AUROC / P-AUPR / P-F1max —— 池化像素
   把所有图的像素混在一起算（每类单独算，再宏平均），是 MVTec / Real-IAD
   评测里像素级指标的通行口径。
   * 关键：抽样必须**与标签无关**（统一比例抽正负像素），否则 AUPR 会被抽样
     比例直接改变——AUPR 对阳性率敏感，AUROC 不敏感。
   * 随机基线：AUROC=0.5；AUPR≈像素阳性率；F1max 由阳性率决定。

2. AUPRO@0.3 —— 每区域重叠（per-region overlap）
   * 阈值网格：256 个等距阈值，直接对**原始概率**（sigmoid 输出）切；
   * 每张有缺陷的图：标出 GT 连通域，对每个阈值算「每个区域被覆盖的比例」，
     再对该图的区域取平均 → 该图在当前阈值的 PRO；
   * FPR：**所有被评测图像**（含正常图）的误报像素数 / 正常像素总数；
   * 曲线 = 所有缺陷图的 PRO 平均 vs 全局 FPR，在 FPR∈[0,0.3] 上积分再除以 0.3。
   * 随机基线 **≈0.15**（PRO≈FPR 的对角线在 [0,0.3] 上积分除以 0.3），不是 0.5。

3. I-AUROC / I-AUPR —— 图像级诊断
   图像分数 = 该图概率图的 **max**（与分类线的 group 聚合口径一致）；按类算再宏平均。
   只作诊断，不作主指标。

--------------------------------------------------------------------------
写作时必须注明的口径
--------------------------------------------------------------------------
* 报的是 **类内宏平均**（每类先算再平均），不是全数据集一次性池化；
* AUPRO 区域等权、图像等权，与 Bergmann et al. (IJCV 2021) 一致；
* 阈值网格 256 等距（不是每图自适应），保证跨图可比；
* 池化像素在超预算时做**标签无关**的均匀抽样（见 add()），不改变阳性率。

用法：
    ev = PixelEvaluator()
    ev.add(prob_map, gt_mask, image_label)     # HxW float / bool / 0|1
    res = ev.compute()
"""
from __future__ import annotations

import numpy as np

from . import metrics

# NumPy 2.x 把 trapz 改名
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

N_BINS = 256
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def _label_regions(gt: np.ndarray):
    """连通域标注；无 scipy 时整张 GT 记作一个区域。"""
    try:
        from scipy import ndimage
        rid, n = ndimage.label(gt)
        if n == 0:
            return rid, np.zeros(0, dtype=np.int64)
        return rid, np.bincount(rid.ravel())[1:]
    except Exception:
        n = int(gt.sum())
        return gt.astype(np.int64), (np.array([n], dtype=np.int64) if n else np.zeros(0, np.int64))


def _hist_ge(x: np.ndarray) -> np.ndarray:
    """返回长度 N_BINS 的「分数 >= 各阈值」计数（阈值 = EDGES[:-1] 升序）。"""
    h, _ = np.histogram(x, bins=EDGES)
    return np.cumsum(h[::-1])[::-1].astype(np.float64)


def pooled_metrics_from_hist(neg: np.ndarray, pos: np.ndarray) -> dict:
    """由正/负像素的分数直方图直接算池化 AUROC / AUPR / F1max。

    训练期评测与离线复评共用这一份实现，避免两套口径。
    （曾经的实现按「剩余预算」抽样，前 ~10 张图就把预算吃完，后面的图 rate=0 被跳过，
      导致 P-AUPR/P-AUROC 只在前 10 张图上统计——已废弃。）
    """
    neg = np.asarray(neg, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    tot_neg, tot_pos = float(neg.sum()), float(pos.sum())
    out = {"pos_rate": tot_pos / max(tot_pos + tot_neg, 1)}
    if tot_pos == 0 or tot_neg == 0:
        out.update({k: float("nan") for k in ("auroc", "aupr", "aupr_minus_prevalence", "f1max")})
        return out
    cneg = np.cumsum(neg[::-1])[::-1]
    cpos = np.cumsum(pos[::-1])[::-1]
    tpr = cpos / tot_pos
    fpr = cneg / tot_neg
    order = np.argsort(fpr)
    out["auroc"] = float(_trapz(tpr[order], fpr[order]))
    prec = cpos / np.maximum(cpos + cneg, 1.0)
    out["aupr"] = float(np.sum(prec * (pos / tot_pos)))
    out["aupr_minus_prevalence"] = out["aupr"] - out["pos_rate"]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + tpr) > 0, 2 * prec * tpr / (prec + tpr), 0.0)
    out["f1max"] = float(np.nanmax(f1))
    return out


class PixelEvaluator:
    def __init__(self, max_fpr: float = 0.30, max_pixels: int | None = None,
                 seed: int = 20261020):
        self.max_fpr = max_fpr
        self.max_pixels = max_pixels      # 已废弃：现在用全量直方图，不抽样
        # 池化指标：累加正/负像素的分数直方图（全量，不抽样）
        self._pos_hist = np.zeros(N_BINS, dtype=np.float64)
        self._neg_hist = np.zeros(N_BINS, dtype=np.float64)
        # AUPRO 累加器
        self._fp = np.zeros(N_BINS, dtype=np.float64)
        self._neg_total = 0
        self._pro_sum = np.zeros(N_BINS, dtype=np.float64)
        self._n_anomalous = 0
        # 图像级
        self._img_scores: list[float] = []
        self._img_labels: list[int] = []
        # 统计
        self.n_images = 0
        self.n_regions = 0
        self.n_pos_px = 0
        self.n_neg_px = 0

    # ------------------------------------------------------------------
    def add(self, score: np.ndarray, gt: np.ndarray, image_label: int = 0) -> None:
        score = np.asarray(score, dtype=np.float32)
        gt = np.asarray(gt).astype(bool)
        if score.shape != gt.shape:
            raise ValueError(f"score {score.shape} 与 gt {gt.shape} 不一致")
        self.n_images += 1
        self._img_scores.append(float(score.max()))
        self._img_labels.append(int(image_label))

        pos = int(gt.sum())
        neg = int(gt.size) - pos
        self.n_pos_px += pos
        self.n_neg_px += neg
        flat_s, flat_y = score.ravel(), gt.ravel()

        # ---- 池化指标：全量像素的直方图累加（不抽样，无偏）----
        if pos:
            self._pos_hist += np.histogram(flat_s[flat_y], bins=EDGES)[0]
        if neg:
            self._neg_hist += np.histogram(flat_s[~flat_y], bins=EDGES)[0]

        # ---- AUPRO：全图阈值计数 ----
        if neg > 0:
            self._fp += _hist_ge(flat_s[~flat_y])
            self._neg_total += neg
        if pos == 0:
            return
        rid, sizes = _label_regions(gt)
        if len(sizes) == 0:
            return
        self.n_regions += len(sizes)
        rid_flat = rid.ravel()
        per_region = []
        for r, size in enumerate(sizes, start=1):
            sel = rid_flat == r
            if sel.any():
                per_region.append(_hist_ge(flat_s[sel]) / float(size))
        if per_region:
            self._pro_sum += np.mean(np.stack(per_region), axis=0)
            self._n_anomalous += 1

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        out = {
            "n_images": int(self.n_images),
            "n_regions": int(self.n_regions),
            "n_pos_px": int(self.n_pos_px),
            "n_neg_px": int(self.n_neg_px),
            "pos_rate_px": float(self.n_pos_px / max(self.n_pos_px + self.n_neg_px, 1)),
            "pooled_pixels_used": int(self.n_pos_px + self.n_neg_px),
        }
        pooled = pooled_metrics_from_hist(self._neg_hist, self._pos_hist)
        out.update({k: pooled[k] for k in ("auroc", "aupr", "aupr_minus_prevalence", "f1max")})

        if self._n_anomalous > 0 and self._neg_total > 0:
            fpr = self._fp / float(self._neg_total)
            pro = self._pro_sum / float(self._n_anomalous)
            # 补两个端点：全判正常 (0,0)、全判异常 (1,1)，否则曲线在 [0,0.3] 上没有左端
            fpr = np.concatenate([[0.0], fpr, [1.0]])
            pro = np.concatenate([[0.0], pro, [1.0]])
            order = np.argsort(fpr)
            fpr, pro = fpr[order], pro[order]
            # 同一 FPR 取可达的最大 PRO（保证单调）
            uf = np.unique(fpr)
            up = np.array([pro[fpr == v].max() for v in uf])
            fpr, pro = uf, up
            # 插值到公共 FPR 网格再积分（端点已补，np.interp 会自动钳位）
            grid = np.linspace(0.0, self.max_fpr, 101)
            pro_grid = np.interp(grid, fpr, pro)
            out["apro_0.3"] = float(_trapz(pro_grid, grid) / self.max_fpr)
            out["apro_fpr_max_used"] = float(self.max_fpr)
        else:
            out["apro_0.3"] = float("nan")

        if self._img_labels and len(set(self._img_labels)) == 2:
            ys = np.asarray(self._img_labels)
            ss = np.asarray(self._img_scores, dtype=float)
            out.update({"i_auroc": metrics.auroc(ys, ss),
                        "i_aupr": metrics.aupr(ys, ss),
                        "i_f1max": metrics.f1max(ys, ss)})
        else:
            out.update({k: float("nan") for k in ("i_auroc", "i_aupr", "i_f1max")})
        return out


# --------------------------------------------------------------------------
def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """最近邻缩放到 size×size，保持二值。"""
    if mask.shape == (size, size):
        return mask.astype(bool)
    from PIL import Image
    im = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.array(im.resize((size, size), Image.NEAREST)) > 127


def load_mask(path, size: int) -> np.ndarray:
    from PIL import Image
    a = np.array(Image.open(path).convert("L"))
    return resize_mask(a > 127, size)
