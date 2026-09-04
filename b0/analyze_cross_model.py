"""Compare two models on the same frozen multi-view manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

AGGREGATIONS = ("max", "top2_mean", "mean")
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260901


def f1max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(2 * p * r / (p + r + 1e-12)))


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if np.unique(y).size < 2:
        return {"auroc": float("nan"), "aupr": float("nan"), "f1max": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "aupr": float(average_precision_score(y, s)), "f1max": f1max(y, s)}


def aggregate(x, method):
    x = np.sort(x.to_numpy(float))
    return float(x[-1] if method == "max" else x[-2:].mean() if method == "top2_mean" else x.mean())


def validate_views(v, expected_groups, expected_views, expected_classes, score_column):
    missing = {"class_name", "group_path", "sample_label", score_column} - set(v.columns)
    if missing:
        raise ValueError(f"输入缺少列: {sorted(missing)}")
    if len(v) != expected_views or v.group_path.nunique() != expected_groups or v.class_name.nunique() != expected_classes:
        raise ValueError(f"输入规模不符: views={len(v)}, groups={v.group_path.nunique()}, classes={v.class_name.nunique()}")
    sizes = v.groupby(["class_name", "group_path"]).size()
    if not sizes.eq(5).all():
        raise ValueError(f"每个 group 必须恰好 5 个视角；异常示例 {sizes[~sizes.eq(5)].head().to_dict()}")
    if not v.groupby("group_path").sample_label.nunique().eq(1).all():
        raise ValueError("同一 group 内 sample_label 不一致")
    if "image_path" in v and v.image_path.duplicated().any():
        raise ValueError("输入存在重复 image_path")


def validate_same_manifest(a, b):
    if "image_path" not in a or "image_path" not in b:
        raise ValueError("两份输入都必须包含 image_path 以验证同一 manifest")
    keys = ["class_name", "group_path", "image_path", "sample_label"]
    left = a[keys].sort_values(keys).reset_index(drop=True)
    right = b[keys].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("两模型输入不是同一冻结 manifest")


def build_samples(v, score_column):
    v = v.copy()
    if "width" not in v: v["width"] = 0
    if "height" not in v: v["height"] = v.width
    if "defect_ratio" not in v: v["defect_ratio"] = 0.0
    v["long_side"] = v[["width", "height"]].max(axis=1)
    keys = ["class_name", "group_path"]
    out = v.groupby(keys, as_index=False, sort=True).agg(sample_label=("sample_label", "first"), resolution=("long_side", "median"), area=("defect_ratio", "max"))
    for method in AGGREGATIONS:
        score = v.groupby(keys, sort=True)[score_column].apply(lambda x, m=method: aggregate(x, m)).rename(method).reset_index()
        out = out.merge(score, on=keys, validate="one_to_one")
    return out


def summarize(x):
    rows = []
    for method in AGGREGATIONS:
        y, s = x.sample_label.to_numpy(), x[method].to_numpy()
        overall = metrics(y, s)
        per_class = [metrics(g.sample_label, g[method]) for _, g in x.groupby("class_name")]
        rows.append({"aggregation": method, **{f"overall_{k}": v for k, v in overall.items()}, "macro_auroc": float(np.nanmean([z["auroc"] for z in per_class])), "macro_aupr": float(np.nanmean([z["aupr"] for z in per_class])), "macro_f1max": float(np.nanmean([z["f1max"] for z in per_class])), "ok_median": float(np.median(s[y == 0])), "ng_median": float(np.median(s[y == 1]))})
    return pd.DataFrame(rows)


def per_class_metrics(x):
    rows = []
    for method in AGGREGATIONS:
        for class_name, part in x.groupby("class_name", sort=True):
            rows.append({"aggregation": method, "class_name": class_name, **metrics(part.sample_label, part[method])})
    return pd.DataFrame(rows)


def score_distributions(x):
    rows = []
    for method in AGGREGATIONS:
        for label, part in x.groupby("sample_label", sort=True):
            values = part[method].to_numpy(float)
            rows.append({
                "aggregation": method, "label": "defect" if label else "normal", "n": len(part),
                "mean": values.mean(), "median": np.median(values),
                "p05": np.quantile(values, .05), "p95": np.quantile(values, .95),
            })
    return pd.DataFrame(rows)


def bootstrap(x, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy() for _, g in x.groupby(["class_name", "sample_label"], sort=True)]
    frames = []
    for repeat in range(repeats):
        ix = np.concatenate([rng.choice(z, len(z), replace=True) for z in strata])
        frame = summarize(x.loc[ix]); frame.insert(0, "repeat", repeat); frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for method, part in raw.groupby("aggregation", sort=False):
        for name in ["overall_auroc", "overall_aupr", "overall_f1max", "macro_auroc", "macro_aupr", "macro_f1max"]:
            z = part[name].dropna(); rows.append({"aggregation": method, "metric": name, "ci_low": z.quantile(.025), "median": z.quantile(.5), "ci_high": z.quantile(.975)})
    return raw, pd.DataFrame(rows)


def split_calibration(x):
    x = x.copy(); x["_hash"] = x.group_path.map(lambda z: hashlib.sha256(str(z).encode()).hexdigest()); ids = []
    for key, g in x.groupby(["class_name", "sample_label"], sort=True):
        if len(g) < 2: raise ValueError(f"calibration split 要求 class×label 至少 2 组: {key}")
        ids.extend(g.sort_values("_hash").index[:len(g)//2])
    return x.loc[ids].drop(columns="_hash"), x.drop(ids).drop(columns="_hash")


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12); i = int(np.argmax(f)); return float(t[i]), float(f[i])


def ece(y, p, bins=10):
    ids = np.minimum(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), bins - 1); value = 0.0
    for i in range(bins):
        m = ids == i
        if m.any(): value += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(value)


def calibration(x):
    try:
        cal, ev = split_calibration(x)
    except ValueError as error:
        return pd.DataFrame([{"aggregation": method, "status": "skipped", "reason": str(error)} for method in AGGREGATIONS])
    rows = []; grid = np.geomspace(.05, 20, 301)
    for method in AGGREGATIONS:
        cp = np.clip(cal[method].to_numpy(float), 1e-7, 1 - 1e-7); cy = cal.sample_label.to_numpy(float); cl = np.log(cp / (1 - cp)); losses = []
        for t in grid:
            q = 1 / (1 + np.exp(-np.clip(cl / t, -50, 50))); losses.append(-np.mean(cy*np.log(q) + (1-cy)*np.log(1-q)))
        temperature = float(grid[int(np.argmin(losses))]); threshold, cal_f1 = best_threshold(cy, cp)
        y = ev.sample_label.to_numpy(int); raw = np.clip(ev[method].to_numpy(float), 1e-7, 1 - 1e-7); logits = np.log(raw / (1 - raw)); tp = 1 / (1 + np.exp(-np.clip(logits / temperature, -50, 50))); m = metrics(y, raw)
        rows.append({"aggregation": method, "status": "ok", "temperature": temperature, "threshold": threshold, "calibration_f1_at_threshold": cal_f1, "cal_n": len(cal), "eval_n": len(ev), "eval_auroc_raw": m["auroc"], "eval_aupr_raw": m["aupr"], "eval_f1max_raw": m["f1max"], "eval_accuracy_raw_0.5": np.mean((raw >= .5) == y), "eval_accuracy_temperature_0.5": np.mean((tp >= .5) == y), "eval_accuracy_threshold": np.mean((raw >= threshold) == y), "eval_brier_raw": brier_score_loss(y, raw), "eval_brier_temperature": brier_score_loss(y, tp), "eval_ece_raw": ece(y, raw), "eval_ece_temperature": ece(y, tp)})
    return pd.DataFrame(rows)


def paired_deltas(first, second, first_name, second_name):
    rows = []
    for method in AGGREGATIONS:
        left = first[first.aggregation.eq(method)].set_index("repeat")
        right = second[second.aggregation.eq(method)].set_index("repeat")
        for metric_name in ["overall_auroc", "overall_aupr", "overall_f1max", "macro_auroc", "macro_aupr", "macro_f1max"]:
            delta = left[metric_name] - right[metric_name]
            rows.append({"aggregation": method, "contrast": f"{first_name}-{second_name}", "metric": metric_name, "ci_low": delta.quantile(.025), "median": delta.quantile(.5), "ci_high": delta.quantile(.975), "probability_positive": (delta > 0).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", nargs="+", type=Path, required=True); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--score-column", default="p_yes"); ap.add_argument("--expected-groups", type=int, default=720); ap.add_argument("--expected-views", type=int, default=3600); ap.add_argument("--expected-classes", type=int, default=18); ap.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS); args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(args.input) != len(args.names) or not 1 <= len(args.input) <= 2:
        raise ValueError("--input and --names must contain one or two matching entries")
    views = [pd.read_csv(p) for p in args.input]
    for v in views: validate_views(v, args.expected_groups, args.expected_views, args.expected_classes, args.score_column)
    if len(views) == 2:
        validate_same_manifest(*views)
    reports = []; raws = []
    for v, name in zip(views, args.names):
        x = build_samples(v, args.score_column); x.to_csv(args.out_dir / f"{name}_sample_scores.csv", index=False); sm = summarize(x); sm.insert(0, "model", name); sm.to_csv(args.out_dir / f"{name}_aggregation_metrics.csv", index=False); pc = per_class_metrics(x); pc.insert(0, "model", name); pc.to_csv(args.out_dir / f"{name}_per_class_metrics.csv", index=False); dist = score_distributions(x); dist.insert(0, "model", name); dist.to_csv(args.out_dir / f"{name}_score_distributions.csv", index=False); raw, ci = bootstrap(x, args.bootstrap_repeats); raws.append(raw); ci.insert(0, "model", name); ci.to_csv(args.out_dir / f"{name}_bootstrap_ci.csv", index=False); ca = calibration(x); ca.insert(0, "model", name); ca.to_csv(args.out_dir / f"{name}_calibration.csv", index=False); reports.append((name, len(v), len(x), sm, pc, dist, ci, ca))
    lines = ["# B0f Dev 720 report", "", "All inputs passed the 18-class/720-group/3600-view gate. Max is the primary aggregation; calibration is diagnostic only.", ""]
    for name, nv, ng, sm, pc, dist, ci, ca in reports: lines += [f"## {name}", f"{ng} groups / {nv} views", "", "Aggregate metrics:", sm.to_csv(index=False), "Per-class metrics:", pc.to_csv(index=False), "Score distributions:", dist.to_csv(index=False), "Bootstrap 95% intervals (group-stratified):", ci.to_csv(index=False), "Calibration split:", ca.to_csv(index=False), ""]
    if len(raws) == 2:
        deltas = paired_deltas(raws[0], raws[1], args.names[0], args.names[1])
        deltas.to_csv(args.out_dir / "paired_model_delta_ci.csv", index=False)
        lines += ["## Paired model deltas (same bootstrap draws)", "", deltas.to_csv(index=False)]
    (args.out_dir / "CROSS_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
