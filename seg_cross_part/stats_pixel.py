"""像素级统计量容器：把「每张图」压成可复算的充分统计量。

目的：评估一次、统计多次。推理时每张图只留
    - 负/正像素的 256 桶分数直方图（用于池化 AUROC/AUPR/F1max 的任意重采样）
    - 每区域的覆盖曲线（256 个阈值上的 PRO，用于 AUPRO 的任意重采样）
    - 元信息（group_path / 类别 / 面积档 / 图像分数 / 标签）
这样 **分层评测**（按类、按面积档）和 **bootstrap**（按 group 重采样）都不需要再过一遍模型。

代价：每图约 3×256×8B ≈ 6 KB；一个 split（约 4.5k 图）≈ 27 MB，可接受。

口径与 `metrics_pixel.PixelEvaluator` 一致：
- 阈值网格 256 等距（EDGES[:-1]）；
- FPR = 累计负像素误报 / 全部负像素；
- PRO = 有缺陷图的每区域覆盖度平均。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics_pixel import EDGES, N_BINS, _trapz, pooled_metrics_from_hist

MAX_FPR = 0.30


# --------------------------------------------------------------------------
# 单图统计
# --------------------------------------------------------------------------
def image_hist(score: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=np.float32).ravel()
    g = np.asarray(gt).astype(bool).ravel()
    neg, _ = np.histogram(s[~g], bins=EDGES)
    pos, _ = np.histogram(s[g], bins=EDGES)
    return neg.astype(np.float64), pos.astype(np.float64)


def image_pro(score: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, int]:
    """返回 (每区域覆盖度曲线在 256 个阈值上的平均, 区域数)。"""
    from .metrics_pixel import _hist_ge, _label_regions
    gt = np.asarray(gt).astype(bool)
    if not gt.any():
        return np.zeros(N_BINS, dtype=np.float64), 0
    rid, sizes = _label_regions(gt)
    if len(sizes) == 0:
        return np.zeros(N_BINS, dtype=np.float64), 0
    rid_flat = rid.ravel()
    s = np.asarray(score, dtype=np.float32).ravel()
    curves = []
    for r, size in enumerate(sizes, start=1):
        sel = rid_flat == r
        if sel.any():
            curves.append(_hist_ge(s[sel]) / float(size))
    if not curves:
        return np.zeros(N_BINS, dtype=np.float64), 0
    return np.mean(np.stack(curves), axis=0), len(sizes)


# --------------------------------------------------------------------------
class PixelStats:
    def __init__(self):
        self.group: list[str] = []
        self.cls: list[str] = []
        self.area: list[str] = []
        self.label: list[int] = []
        self.img_score: list[float] = []
        self.neg: list[np.ndarray] = []
        self.pos: list[np.ndarray] = []
        self.pro: list[np.ndarray] = []
        self.n_regions: list[int] = []

    def add(self, *, group: str, cls: str, area: str, label: int, img_score: float,
            neg: np.ndarray, pos: np.ndarray, pro: np.ndarray, n_regions: int) -> None:
        self.group.append(str(group))
        self.cls.append(str(cls))
        self.area.append(str(area))
        self.label.append(int(label))
        self.img_score.append(float(img_score))
        self.neg.append(neg)
        self.pos.append(pos)
        self.pro.append(pro)
        self.n_regions.append(int(n_regions))

    def __len__(self) -> int:
        return len(self.label)

    # ---------------- 存取 ----------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            group=np.array(self.group), cls=np.array(self.cls), area=np.array(self.area),
            label=np.array(self.label, dtype=np.int8),
            img_score=np.array(self.img_score, dtype=np.float64),
            neg=np.stack(self.neg) if self.neg else np.zeros((0, N_BINS)),
            pos=np.stack(self.pos) if self.pos else np.zeros((0, N_BINS)),
            pro=np.stack(self.pro) if self.pro else np.zeros((0, N_BINS)),
            n_regions=np.array(self.n_regions, dtype=np.int32),
        )

    @classmethod
    def load(cls, path) -> "PixelStats":
        z = np.load(Path(path), allow_pickle=False)
        st = cls()
        st.group = [str(x) for x in z["group"]]
        st.cls = [str(x) for x in z["cls"]]
        st.area = [str(x) for x in z["area"]]
        st.label = [int(x) for x in z["label"]]
        st.img_score = [float(x) for x in z["img_score"]]
        # 关键：先把三个大数组**各解压一次**，再逐行切；不能写成 z["neg"][i]
        neg, pos, pro = z["neg"], z["pos"], z["pro"]
        n = len(st.label)
        st.neg = [neg[i] for i in range(n)]
        st.pos = [pos[i] for i in range(n)]
        st.pro = [pro[i] for i in range(n)]
        st.n_regions = [int(x) for x in z["n_regions"]]
        return st

    # ---------------- 子集 ----------------
    def subset(self, idx: np.ndarray | list[int]) -> "PixelStats":
        out = PixelStats()
        for i in idx:
            out.add(group=self.group[i], cls=self.cls[i], area=self.area[i],
                    label=self.label[i], img_score=self.img_score[i],
                    neg=self.neg[i], pos=self.pos[i], pro=self.pro[i],
                    n_regions=self.n_regions[i])
        return out

    # ---------------- 指标 ----------------
    def _matrices(self):
        if getattr(self, "_cache", None) is None:
            self._cache = (
                np.stack(self.neg), np.stack(self.pos), np.stack(self.pro),
                np.asarray(self.n_regions), np.asarray(self.label),
                np.asarray(self.img_score, dtype=float),
            )
        return self._cache

    def metrics(self, idx: np.ndarray | None = None, max_fpr: float = MAX_FPR) -> dict:
        """idx 给定时只统计这些行（bootstrap 用，避免拷贝对象）。"""
        n = len(self)
        if n == 0:
            return {"n_images": 0}
        NEG, POS, PRO, NREG, LBL, IMG = self._matrices()
        if idx is None:
            neg = NEG.sum(0); pos = POS.sum(0); pro_sum = PRO.sum(0)
            n_anom = int((NREG > 0).sum()); lbl = LBL; img = IMG; n = len(NREG)
        else:
            # 用重采样计数做加权求和：不拷贝 (n_images × 256) 的大矩阵
            cnt = np.bincount(idx, minlength=len(NREG)).astype(np.float64)
            neg = NEG.T @ cnt; pos = POS.T @ cnt; pro_sum = PRO.T @ cnt
            n_anom = int(np.dot((NREG > 0).astype(np.float64), cnt))
            lbl = LBL[idx]; img = IMG[idx]; n = len(idx)
        tot_neg, tot_pos = neg.sum(), pos.sum()
        # 阈值从高到低 → 累计（分数 >= t）
        cneg = np.cumsum(neg[::-1])[::-1]
        fpr = cneg / max(tot_neg, 1)
        order = np.argsort(fpr)
        out = {"n_images": int(n), "pos_rate_px": float(tot_pos / max(tot_pos + tot_neg, 1))}
        out.update(pooled_metrics_from_hist(neg, pos))
        # AUPRO
        out["n_anomalous_images"] = n_anom
        if n_anom > 0 and tot_neg > 0:
            pro = pro_sum / n_anom
            # 与 PixelEvaluator 一致：补端点 (0,0)/(1,1)，同一 FPR 取最大 PRO
            f2 = np.concatenate([[0.0], fpr, [1.0]])
            p2 = np.concatenate([[0.0], pro, [1.0]])
            o2 = np.argsort(f2)
            f2, p2 = f2[o2], p2[o2]
            uf = np.unique(f2)
            up = np.array([p2[f2 == v].max() for v in uf])
            grid = np.linspace(0.0, max_fpr, 101)
            pro_g = np.interp(grid, uf, up)
            out["apro_0.3"] = float(_trapz(pro_g, grid) / max_fpr)
        else:
            out["apro_0.3"] = float("nan")
        # 图像级
        ys = np.asarray(lbl)
        ss = np.asarray(img, dtype=float)
        if len(set(ys.tolist())) == 2:
            from . import metrics as M
            out["i_auroc"] = M.auroc(ys, ss)
            out["i_aupr"] = M.aupr(ys, ss)
        else:
            out["i_auroc"] = out["i_aupr"] = float("nan")
        return out

    def by_class(self) -> dict[str, dict]:
        arr = np.asarray(self.cls)
        return {c: self.metrics(np.flatnonzero(arr == c)) for c in sorted(set(self.cls))}

    def by_area(self) -> dict[str, dict]:
        arr = np.asarray(self.area)
        return {a: self.metrics(np.flatnonzero(arr == a)) for a in sorted(set(self.area))}


# --------------------------------------------------------------------------
# bootstrap：以 group_path 为重采样单位
# --------------------------------------------------------------------------
def bootstrap(st: PixelStats, n_boot: int = 1000, seed: int = 20261020,
              keys=("apro_0.3", "aupr", "auroc", "i_auroc")) -> dict:
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(set(st.group)))
    idx_by_group = {g: np.flatnonzero(np.array(st.group) == g) for g in groups}
    point = st.metrics()
    draws = {k: [] for k in keys}
    n = len(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([idx_by_group[groups[i]] for i in pick])
        m = st.metrics(idx)
        for k in keys:
            draws[k].append(m.get(k, float("nan")))
    out = {"n_groups": int(n), "n_boot": n_boot, "point": {k: point.get(k) for k in keys},
           "ci": {}}
    for k in keys:
        a = np.asarray(draws[k], dtype=float)
        a = a[np.isfinite(a)]
        out["ci"][k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [float("nan")] * 2
    return out
