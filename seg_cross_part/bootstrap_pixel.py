"""像素级 bootstrap 置信区间（以 group_path 为重采样单位）。

输入是 `evaluate_pixel.py` 产出的 `stats_<split>.npz`（每图充分统计量）。
重采样单位是**零件组**，不是图像——同一组五视角不能当独立样本。

用法：
    # 单模型：给 AUPRO / P-AUPR / P-AUROC / I-AUROC 的 95% CI
    python -m seg_cross_part.bootstrap_pixel \
      --stats artifacts/seg_cross_part/c_f0_a0/stats_test.npz --n-boot 1000

    # 两模型配对：同一批重采样上算差值 CI（A2 vs A0 用这个）
    python -m seg_cross_part.bootstrap_pixel \
      --stats a0/stats_test.npz a2/stats_test.npz --names A0 A2 --out ci.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stats_pixel import PixelStats, bootstrap


def paired(a: PixelStats, b: PixelStats, n_boot: int, seed: int,
           keys=("apro_0.3", "aupr", "auroc")) -> dict:
    """在同一批 group 重采样上算两模型的差值（数据必须来自同一个 split）。"""
    rng = np.random.default_rng(seed)
    ga = np.array(a.group); gb = np.array(b.group)
    common = np.array(sorted(set(a.group) & set(b.group)))
    if len(common) == 0:
        raise SystemExit("两个 stats 没有共同的 group，检查是否同一个 split")
    idx_a = {g: np.flatnonzero(ga == g) for g in common}
    idx_b = {g: np.flatnonzero(gb == g) for g in common}
    pa, pb = a.metrics(), b.metrics()
    draws = {k: [] for k in keys}
    n = len(common)
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        ia = np.concatenate([idx_a[common[i]] for i in pick])
        ib = np.concatenate([idx_b[common[i]] for i in pick])
        ma, mb = a.metrics(ia), b.metrics(ib)
        for k in keys:
            draws[k].append(ma.get(k, np.nan) - mb.get(k, np.nan))
    out = {"n_groups": int(n), "n_boot": n_boot,
           "points": {k: [pa.get(k), pb.get(k)] for k in keys}, "delta_ci": {}}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out["delta_ci"][k] = {"delta": float(pa[k] - pb[k]), "ci95": [float(lo), float(hi)],
                                  "p_delta_le_0": float(np.mean(arr <= 0))}
        else:
            out["delta_ci"][k] = {"delta": float("nan"), "ci95": [float("nan")] * 2}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="像素级 bootstrap（group_path 为单位）")
    ap.add_argument("--stats", type=Path, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = args.names or [p.parent.name if p.parent.name else p.stem for p in args.stats]
    if len(names) != len(args.stats):
        raise SystemExit("--names 数量与 --stats 不一致")
    sts = [PixelStats.load(p) for p in args.stats]
    result = {"models": {}, "paired": []}

    for name, st in zip(names, sts):
        r = bootstrap(st, n_boot=args.n_boot, seed=args.seed)
        result["models"][name] = r
        ci = r["ci"]
        print(f"[{name}] n_groups={r['n_groups']}")
        for k in ("apro_0.3", "aupr", "auroc", "i_auroc"):
            print(f"   {k:10s} {r['point'][k]:.4f}  95%CI [{ci[k][0]:.4f}, {ci[k][1]:.4f}]")

    for i in range(len(sts)):
        for j in range(i + 1, len(sts)):
            pr = paired(sts[i], sts[j], args.n_boot, args.seed)
            pr.update({"a": names[i], "b": names[j]})
            result["paired"].append(pr)
            print(f"[{names[i]} − {names[j]}]（配对）")
            for k, v in pr["delta_ci"].items():
                print(f"   Δ{k:9s} {v['delta']:+.4f}  95%CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                      f"  P(Δ<=0)={v.get('p_delta_le_0', float('nan')):.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"已写出 {args.out}")


if __name__ == "__main__":
    main()
