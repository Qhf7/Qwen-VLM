"""Summarize classification-only checkpoints and flag late/constant-score collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/e1_classification_fold1"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("step*_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "step": int(metrics["tag"].replace("step", "")),
            "macro_auroc": metrics["macro_auroc"],
            "macro_aupr": metrics["macro_aupr"],
            "macro_f1max": metrics["macro_f1max"],
            "small_defect_recall": metrics["small_defect_recall"],
            "view_score_std": metrics["score_diagnostics"]["view_std"],
            "ok_median": metrics["score_diagnostics"]["ok_median"],
            "ng_median": metrics["score_diagnostics"]["ng_median"],
            "median_separation": metrics["score_diagnostics"]["median_separation_ng_minus_ok"],
        })
    frame = pd.DataFrame(rows).sort_values("step")
    if frame.empty or frame.step.duplicated().any() or not frame.step.is_monotonic_increasing:
        raise RuntimeError("expected one or more ordered classification evaluations")
    frame["constant_score_collapse"] = frame.view_score_std.lt(1e-3)
    frame["near_random_ranking"] = frame.macro_aupr.le(0.55)
    best_index = frame.macro_aupr.idxmax()
    best_step = int(frame.loc[best_index, "step"])
    best_aupr = float(frame.loc[best_index, "macro_aupr"])
    final_aupr = float(frame.iloc[-1].macro_aupr)
    final_step = int(frame.iloc[-1].step)
    late_drop = best_aupr - final_aupr
    verdict = {
        "best_step": best_step,
        "best_macro_aupr": best_aupr,
        "final_macro_aupr": final_aupr,
        "best_to_final_drop": late_drop,
        "final_step": final_step,
        "late_metric_collapse": bool(best_step < final_step and late_drop >= 0.05),
        "constant_score_collapse_any_checkpoint": bool(frame.constant_score_collapse.any()),
        "near_random_ranking_any_checkpoint": bool(frame.near_random_ranking.any()),
        "collapse_rules_preregistered": {
            "constant_score": "view score std < 0.001 (diagnostic flag)",
            "near_random_ranking": "macro AUPR <= 0.55 (heuristic warning; not a statistical test)",
            "late_metric": "best checkpoint is before the final checkpoint and macro AUPR drops >= 0.05 by the final checkpoint",
        },
    }
    frame.to_csv(args.run_dir / "classification_convergence.csv", index=False)
    (args.run_dir / "classification_convergence.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Classification-only convergence",
        "",
        "| step | macro AUROC | macro AUPR | macro F1max | small recall | score std | OK median | NG median | NG-OK median | constant collapse | near random |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.step} | {row.macro_auroc:.4f} | {row.macro_aupr:.4f} | "
            f"{row.macro_f1max:.4f} | {row.small_defect_recall:.4f} | "
            f"{row.view_score_std:.6f} | {row.ok_median:.6f} | {row.ng_median:.6f} | "
            f"{row.median_separation:.6f} | {row.constant_score_collapse} | {row.near_random_ranking} |"
        )
    lines += ["", "```json", json.dumps(verdict, ensure_ascii=False, indent=2), "```", ""]
    (args.run_dir / "CLASSIFICATION_CONVERGENCE.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
