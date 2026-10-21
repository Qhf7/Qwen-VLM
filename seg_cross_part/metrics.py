"""基础指标（AUROC / AUPR / F1max / 阈值 / ECE / Brier），分类与像素级评测共用。

（原文件：视角级异常分类指标（无第三方依赖，纯 numpy；对应主文档 §8.1）。

指标：AUROC / AUPR / F1max / Balanced Accuracy / Recall@FPR=1%,5% /
正常 FPR / 异常 Recall / ECE / Brier。全部输入连续 anomaly_score 与 0/1 标签。
"""
from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    s = y_score[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:  # 平均秩处理并列
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_r = ranks[y_true == 1].sum()
    return float((sum_r - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / n_pos)


def f1max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(f1.max())


def _roc_points(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(y_score, kind="mergesort")
    y = y_true[order]
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:          # 只有一个类别，ROC 未定义
        return np.array([]), np.array([]), n_pos
    tp = 0
    fp = 0
    fpr = [0.0]
    tpr = [0.0]
    # 按唯一分数升序，越过每个阈值累加
    s = y_score[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        tp += int(y[i:j + 1].sum())
        fp += int((1 - y[i:j + 1]).sum())
        fpr.append(fp / (n - n_pos))
        tpr.append(tp / n_pos)
        i = j + 1
    return np.array(fpr), np.array(tpr), n_pos


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, n_pos = _roc_points(y_true, y_score)
    if len(tpr) == 0 or n_pos == 0:       # 退化输入：返回 nan 而不是崩
        return float("nan")
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def balanced_accuracy_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return float("nan")
    tpr = pred[y_true == 1].mean()
    tnr = 1 - pred[y_true == 0].mean()
    return float((tpr + tnr) / 2.0)


def fpr_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    neg = (y_true == 0)
    if neg.sum() == 0:
        return float("nan")
    return float(pred[neg].mean())


def recall_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    pos = (y_true == 1)
    if pos.sum() == 0:
        return float("nan")
    return float(pred[pos].mean())


def ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    conf = np.clip(y_score, 1e-6, 1 - 1e-6)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    e = 0.0
    total = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        avg_conf = conf[m].mean()
        e += m.sum() * abs(avg_conf - acc)
    return float(e / total)


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score, dtype=float)
    return float(((y_score - y_true) ** 2).mean())


# --------------------------------------------------------------------------
# 阈值：在 val 上冻结、再原样用到 test（§9.2）
# --------------------------------------------------------------------------
def threshold_for_f1max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """返回使 F1 最大的阈值 τ（判定规则 score >= τ 为正）。

    只允许在 val 上调用；把返回的 τ 写进 frozen_protocol.json 后再套到 test。
    并列分数取中点，避免"同分样本跨阈值"带来的边界抖动。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return 0.5
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp)
    rec = tp / n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    i = int(np.argmax(f1))
    if i + 1 < len(s) and s[i + 1] < s[i]:
        return float((s[i] + s[i + 1]) / 2.0)
    return float(s[i])


def f1_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int(((1 - pred) & (y_true == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else float("nan")


def precision_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")
