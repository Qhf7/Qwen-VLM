"""Audit saved single-image predictions and diagnose small-defect score separation.

No GPU inference. Thresholds selected on Dev are exploratory, not deployment estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

STEPS = (250, 500, 750, 1000)
BINS = (
    ("<=0.01%", 0, .0001), ("0.01-0.1%", .0001, .001),
    ("0.1-1%", .001, .01), ("1-10%", .01, .1), (">10%", .1, 1),
)


def positive_sets(frame):
    pos = frame.view_label.eq(1)
    ratio = frame.defect_ratio
    sets = {"all defects": pos, "small <=1%": pos & ratio.le(.01)}
    sets.update({name: pos & ratio.gt(lo) & ratio.le(hi) for name, lo, hi in BINS})
    return sets


def operating_point(frame, threshold):
    y = frame.view_label.to_numpy(dtype=int)
    pred = frame.p_yes.to_numpy() >= threshold
    tp, fp = int(pred[y == 1].sum()), int(pred[y == 0].sum())
    fn, tn = int((~pred[y == 1]).sum()), int((~pred[y == 0]).sum())
    result = dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                  accuracy=(tp + tn) / len(y), precision=tp / max(tp + fp, 1),
                  recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                  fpr=fp / max(fp + tn, 1), predicted_positive_rate=float(pred.mean()))
    for name, mask in positive_sets(frame).items():
        result["recall_" + name] = float(pred[mask].mean()) if mask.any() else np.nan
    return result


def binary_ranking(labels, scores):
    if len(np.unique(labels)) < 2:
        return dict(auroc=np.nan, aupr=np.nan, f1max=np.nan, f1max_threshold=np.nan)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    values = np.divide(2 * precision[:-1] * recall[:-1], denom,
                       out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(values))
    return dict(auroc=float(roc_auc_score(labels, scores)),
                aupr=float(average_precision_score(labels, scores)),
                f1max=float(values[best]), f1max_threshold=float(thresholds[best]))


def threshold_at_fpr(normal_scores, budget):
    """Lowest threshold meeting empirical FPR budget, respecting tied scores and >=."""
    scores = np.sort(np.asarray(normal_scores, dtype=float))
    allowed = int(np.floor(len(scores) * budget + 1e-10))
    if allowed >= len(scores):
        return 0.0
    return float(np.nextafter(scores[len(scores) - allowed - 1], np.inf))


def threshold_at_recall(positive_scores, target):
    """Highest observed threshold retaining at least target empirical recall."""
    scores = np.sort(np.asarray(positive_scores, dtype=float))[::-1]
    if not len(scores):
        return np.nan
    return float(scores[int(np.ceil(target * len(scores))) - 1])


def area_ranking(frame, mask):
    """Compare selected positives with normal images from the same represented classes."""
    positives = frame[mask]
    normal = frame[frame.view_label.eq(0) & frame.class_name.isin(positives.class_name.unique())]
    combined = pd.concat([normal, positives])
    rank = binary_ranking(combined.view_label, combined.p_yes)
    class_ranks = [binary_ranking(g.view_label, g.p_yes) for _, g in combined.groupby("class_name")
                   if g.view_label.nunique() == 2]
    return dict(n_positive=len(positives), n_normal_reference=len(normal),
                n_classes=len(class_ranks), auroc=rank["auroc"], aupr=rank["aupr"],
                ap_chance=len(positives) / len(combined) if len(combined) else np.nan,
                macro_auroc=float(np.mean([x["auroc"] for x in class_ranks])) if class_ranks else np.nan,
                score_median=float(positives.p_yes.median()) if len(positives) else np.nan)


def read_inputs(run_dir, dev_path):
    paths = [run_dir / f"step{s:04d}_{suffix}" for s in STEPS
             for suffix in ("views.csv", "metrics.json", "per_class.csv")]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing current-run inputs; no analysis written:\n" + "\n".join(missing))
    dev = pd.read_parquet(dev_path).set_index("image_path")
    assert dev.index.is_unique
    frames, sources = {}, {}
    for step in STEPS:
        prefix = run_dir / f"step{step:04d}"
        meta = json.loads(Path(str(prefix) + "_metrics.json").read_text())
        if meta.get("evaluation_unit") != "single image" or meta.get("target") != "view_label":
            raise ValueError(f"step{step}: requires explicit single-image evaluation metadata")
        frame = pd.read_csv(Path(str(prefix) + "_views.csv"))
        assert frame.image_path.is_unique and set(frame.image_path) == set(dev.index)
        aligned = dev.loc[frame.image_path]
        assert np.array_equal(frame.view_label, aligned.view_label)
        assert np.array_equal(frame.class_name, aligned.class_name)
        assert np.allclose(frame.defect_ratio, aligned.defect_ratio, atol=1e-12, rtol=1e-9)
        assert frame.view_label.isin([0, 1]).all()
        assert np.isfinite(frame.p_yes).all() and frame.p_yes.between(0, 1).all()
        assert np.isfinite(frame.defect_ratio).all() and frame.defect_ratio.between(0, 1).all()
        assert frame.loc[frame.view_label.eq(1), "defect_ratio"].gt(0).all()
        pc = pd.read_csv(Path(str(prefix) + "_per_class.csv")).set_index("class_name")
        assert set(pc.index) == set(frame.class_name)
        for cls, group in frame.groupby("class_name"):
            rank = binary_ranking(group.view_label, group.p_yes)
            assert np.isclose(rank["aupr"], pc.loc[cls, "aupr"], atol=1e-8)
            assert int(pc.loc[cls, "n_views"]) == len(group)
        op = operating_point(frame, .5)
        assert np.isclose(op["recall_small <=1%"], meta["small_defect_recall"], atol=1e-8)
        assert np.isclose(op["recall"], meta["overall"]["recall"], atol=1e-8)
        frames[step] = frame
    for path in paths + [dev_path]:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frames, sources


def markdown_table(frame):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{value:.4f}"
        return str(value)
    rows = [[str(c) for c in frame.columns], ["---"] * len(frame.columns)]
    rows += [[fmt(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def analyze(frames, sources, out):
    out.mkdir(parents=True, exist_ok=True)
    ops, classes, areas, curves, class_ops = [], [], [], [], []
    for step, frame in frames.items():
        rank = binary_ranking(frame.view_label, frame.p_yes)
        points = {"fixed_0.5": .5, "dev_f1max": rank["f1max_threshold"]}
        normal = frame.loc[frame.view_label.eq(0), "p_yes"].to_numpy()
        small = frame.loc[positive_sets(frame)["small <=1%"], "p_yes"].to_numpy()
        for budget in (.01, .05, .10):
            points[f"FPR<={budget:.0%}"] = threshold_at_fpr(normal, budget)
        for target in (.80, .90, .95):
            points[f"small_recall>={target:.0%}"] = threshold_at_recall(small, target)
        for name, threshold in points.items():
            ops.append(dict(step=step, policy=name, **rank, **operating_point(frame, threshold)))
        for cls, group in frame.groupby("class_name"):
            small_mask = positive_sets(group)["small <=1%"]
            classes.append(dict(step=step, class_name=cls, n_views=len(group),
                                n_normal=int(group.view_label.eq(0).sum()),
                                n_anomaly=int(group.view_label.eq(1).sum()), n_small=int(small_mask.sum()),
                                **binary_ranking(group.view_label, group.p_yes),
                                **operating_point(group, .5),
                                small_vs_normal_auroc=area_ranking(group, small_mask)["auroc"]))
            for budget in (.01, .05, .10):
                threshold = threshold_at_fpr(group.loc[group.view_label.eq(0), "p_yes"], budget)
                point = operating_point(group, threshold)
                class_ops.append(dict(step=step, class_name=cls, fpr_budget=budget,
                                      n_small=int(small_mask.sum()),
                                      small_tp=int(group.loc[small_mask, "p_yes"].ge(threshold).sum()),
                                      **point))
        for name, mask in positive_sets(frame).items():
            areas.append(dict(step=step, area_bin=name, **area_ranking(frame, mask),
                              recall_at_05=float(frame.loc[mask, "p_yes"].ge(.5).mean())))
        # Keep every score transition, not just a coarse grid: scores may crowd near 0 or 1.
        thresholds = np.unique(np.r_[0., .5, 1., frame.p_yes, np.nextafter(frame.p_yes.to_numpy(), np.inf)])
        for threshold in thresholds:
            curves.append(dict(step=step, **operating_point(frame, threshold)))
    op_df, class_df, area_df, curve_df = map(pd.DataFrame, (ops, classes, areas, curves))
    class_op_df = pd.DataFrame(class_ops)
    for name, df in (("operating_points", op_df), ("per_class", class_df),
                     ("area_strata", area_df), ("recall_thresholds", curve_df),
                     ("per_class_operating_points", class_op_df)):
        df.to_csv(out / (name + ".csv"), index=False)
    (out / "input_sha256.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    for ax, step in zip(axes.flat, STEPS):
        df = curve_df[curve_df.step.eq(step)]
        for name, _, _ in BINS:
            count = int(area_df.loc[area_df.step.eq(step) & area_df.area_bin.eq(name), "n_positive"].iloc[0])
            if count:
                ax.step(df.threshold, df["recall_" + name], where="post", label=f"{name} (n={count})")
        ax.plot(df.threshold, df.fpr, color="black", ls=":", label="Normal false-positive rate")
        ax.axvline(.5, color="grey", ls="--", lw=1)
        ax.set(title=f"Step {step}", xlabel="Anomaly-score threshold", ylabel="Recall / false-positive rate",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(out / "recall_threshold_curves.png", dpi=180)
    fig.savefig(out / "recall_threshold_curves.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for step in STEPS:
        df = curve_df[curve_df.step.eq(step)].sort_values("threshold", ascending=False)
        ax.plot(df.fpr, df["recall_small <=1%"], label=f"Step {step}")
    ax.set(xlabel="Normal false-positive rate", ylabel="Small-defect recall (area <=1%)",
           xlim=(0, 1), ylim=(0, 1.02), title="Small-defect recall at comparable false-positive cost")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / "small_recall_vs_fpr.png", dpi=180)
    plt.close(fig)
    names = sorted(class_df.class_name.unique())
    fig, axes = plt.subplots(6, 3, figsize=(15, 19), sharex=True, sharey=True, constrained_layout=True)
    for ax, cls in zip(axes.flat, names):
        for step, frame in frames.items():
            group = frame[frame.class_name.eq(cls)]
            scores = np.sort(group.loc[positive_sets(group)["small <=1%"], "p_yes"].to_numpy())
            if len(scores):
                ts = np.unique(np.r_[0., scores, np.nextafter(scores, np.inf), 1.])
                ax.step(ts, (len(scores) - np.searchsorted(scores, ts, side="left")) / len(scores),
                        where="post", label=str(step))
        ax.set(title=cls, xlabel="Threshold", ylabel="Small-defect recall", xlim=(0, 1), ylim=(0, 1.02))
        ax.axvline(.5, color="grey", ls="--", lw=.7)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(out / "per_class_small_recall.png", dpi=160)
    plt.close(fig)
    columns = ["step", "policy", "threshold", "recall_small <=1%", "recall", "fpr", "precision", "f1"]
    final_ops = op_df[op_df.step.eq(1000)].set_index("policy")
    fixed, calibrated, high = [final_ops.loc[p] for p in ("fixed_0.5", "dev_f1max", "small_recall>=90%")]
    final_area = area_df[area_df.step.eq(1000)].set_index("area_bin")
    final_class_ops = class_op_df[class_op_df.step.eq(1000) & class_op_df.fpr_budget.eq(.10)]
    class_recall = final_class_ops.small_tp.sum() / final_class_ops.n_small.sum()
    conclusion = [
        "## 本次结果与结论", "",
        "现有证据支持两种因素同时存在：固定阈值影响漏检数量，而当前模型对小缺陷的分数区分能力仍是限制高召回、低误报的瓶颈。仅凭最终分数无法进一步确定问题发生在视觉特征、分类头还是输入细节保留环节。", "",
        f"- step1000：阈值 0.5 时小缺陷召回 {fixed['recall_small <=1%']:.2%}、正常误报率 {fixed.fpr:.2%}。取 Dev 全体图片 F1max 阈值 {calibrated.threshold:.6f}，召回提高到 {calibrated['recall_small <=1%']:.2%}，误报率也提高到 {calibrated.fpr:.2%}。",
        f"- 要达到至少 90% 小缺陷召回，阈值需降到 {high.threshold:.6f}；实际召回 {high['recall_small <=1%']:.2%}、正常误报率 {high.fpr:.2%}，即 {int(high.fp)}/1825 张正常图被报异常。",
        f"- 全局阈值控制误报率 ≤10% 时，小缺陷召回为 {final_ops.loc['FPR<=10%', 'recall_small <=1%']:.2%}。即使对各类别分别在 Dev 选阈值，并逐类控制误报率 ≤10%，汇总小缺陷召回也只有 {class_recall:.2%}。类别间分数偏移不能完整解释问题。",
        f"- step1000 小缺陷 AUROC 为 {final_area.loc['small <=1%', 'auroc']:.4f}；1–10% 面积缺陷为 {final_area.loc['1-10%', 'auroc']:.4f}。类别内部等权 AUROC 分别为 {final_area.loc['small <=1%', 'macro_auroc']:.4f} 和 {final_area.loc['1-10%', 'macro_auroc']:.4f}，同类内部仍存在小缺陷区分困难。",
        "- step500 的固定阈值召回下降需结合分数排序看待：四个 step 的小缺陷 AUROC、受控误报下召回见下表；固定 0.5 的召回下降本身不等同于学习坍缩。", "",
        "下一步建议继续分类诊断：在独立校准划分上选阈值，先明确可接受误报率；对低 AUROC 类别复核漏检图片和标签，再设计单变量输入分辨率或特征探针对照，以定位区分能力不足的原因。", "",
    ]
    lines = ["# 小缺陷 Recall–阈值诊断", "",
             "使用四个 checkpoint 保存的单图分数重新统计；每张图按 view_label 判定。缺陷面积采用 Dev manifest 的 defect_ratio，小缺陷定义为 0 < 面积占比 ≤ 1%。", "",
             *conclusion,
             "## 阈值与误报代价", "", markdown_table(op_df[columns]), "",
             "![面积分层曲线](recall_threshold_curves.png)", "",
             "![召回与误报](small_recall_vs_fpr.png)", "",
             "## 面积分层与分数区分能力", "", markdown_table(area_df), "",
             "每个面积层的 AUROC/AUPR 以该层异常图片和同类正常图片计算；macro_auroc 在有效类别内等权平均。不同层的正类占比不同，不能直接把 AUPR 高低解释为缺陷大小效应。极小层样本少时，单个样本会明显改变召回。", "",
             "## 逐类别", "", markdown_table(class_df[["step", "class_name", "n_small", "aupr", "auroc", "recall_small <=1%", "small_vs_normal_auroc", "fpr"]]), "",
             "![逐类小缺陷召回](per_class_small_recall.png)", "",
             "## 如何判断原因", "",
             "降低阈值后，小缺陷召回提高且正常误报仍满足应用要求，说明阈值有可改进空间。若达到高召回需要大量正常图误报，说明当前模型分数区分能力仍不足，单改阈值无法兼顾召回与误报。", "",
             "AUROC 与受控误报率下召回用于衡量分数排序；它们不能单独定位到视觉特征、分类头、损失、标签或缩放中的具体根因。确认视觉特征学习问题还需冻结特征探针或受控对照。", "",
             "本报告 F1max、召回目标阈值和 FPR 阈值均在同一 Dev 上搜索，仅作诊断；应用允许的误报上限尚未指定。正式阈值需在独立校准集选择，并在未参与选择的评测集报告表现。", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(op_df[columns].to_string(index=False))
    print(out / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/e1_real_classification_qwen3_verified"))
    parser.add_argument("--dev", type=Path, default=Path("artifacts/splits/fold1/dev.parquet"))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    frames, sources = read_inputs(args.run_dir, args.dev)
    analyze(frames, sources, args.out_dir or args.run_dir / "recall_threshold_analysis")


if __name__ == "__main__":
    main()
