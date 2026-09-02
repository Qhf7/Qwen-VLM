"""B0+ prompt、多视角聚合和 5crop 的离线分析。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

try:
    from b0.b0plus_protocol import PROMPTS, SELECTION_KEYS, SELECTION_RULE
except ModuleNotFoundError:
    from b0plus_protocol import PROMPTS, SELECTION_KEYS, SELECTION_RULE


def f1max(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(np.max(2 * precision * recall / (precision + recall + 1e-10)))


def aggregate_scores(values: pd.Series, method: str) -> float:
    scores = np.sort(values.to_numpy(dtype=float))
    if method == "max":
        return float(scores[-1])
    if method == "top2_mean":
        return float(scores[-2:].mean())
    if method == "mean":
        return float(scores.mean())
    raise ValueError(method)


def summarize_views(views: pd.DataFrame, method: str, prompt_id: str) -> dict:
    samples = (
        views.groupby(["class_name", "group_path"], as_index=False)
        .agg(
            sample_label=("sample_label", "first"),
            score=("p_yes", lambda values: aggregate_scores(values, method)),
        )
    )
    labels = samples.sample_label.to_numpy()
    scores = samples.score.to_numpy()
    per_class = [
        roc_auc_score(part.sample_label, part.score)
        for _, part in samples.groupby("class_name")
        if part.sample_label.nunique() == 2
    ]
    class_directions = [
        float(part.loc[part.sample_label == 1, "score"].mean())
        > float(part.loc[part.sample_label == 0, "score"].mean())
        for _, part in samples.groupby("class_name")
        if part.sample_label.nunique() == 2
    ]
    return {
        "prompt_id": prompt_id,
        "aggregation": method,
        "sample_auroc": float(roc_auc_score(labels, scores)),
        "sample_aupr": float(average_precision_score(labels, scores)),
        "sample_f1max": f1max(labels, scores),
        "macro_class_auroc": float(np.mean(per_class)),
        "accuracy_at_0.5": float(((scores >= 0.5).astype(int) == labels).mean()),
        "hard_yes_rate_at_0.5": float((scores >= 0.5).mean()),
        "ng_gt_ok_classes": int(sum(class_directions)),
        "n_evaluable_classes": len(class_directions),
        "ok_score_median": float(np.median(scores[labels == 0])),
        "ng_score_median": float(np.median(scores[labels == 1])),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_dynamic_range": float(scores.max() - scores.min()),
        "n_samples": len(samples),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def choose_best(prompt_rows: pd.DataFrame) -> str:
    direct = prompt_rows[prompt_rows.aggregation == "max"].copy()
    direct = direct.sort_values(
        [*SELECTION_KEYS, "prompt_id"],
        ascending=[False] * len(SELECTION_KEYS) + [True],
        kind="mergesort",
    )
    return str(direct.iloc[0].prompt_id)


def analyze_prompts(out_dir: Path) -> str:
    rows = []
    image_sets = []
    for prompt_id in PROMPTS:
        views = pd.read_csv(out_dir / f"{prompt_id}_views.csv")
        image_sets.append(set(views.image_path))
        for method in ("max", "top2_mean", "mean"):
            rows.append(summarize_views(views, method, prompt_id))
    assert all(paths == image_sets[0] for paths in image_sets[1:])
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "aggregation_ablation.csv", index=False)
    prompt_table = table[table.aggregation == "max"].reset_index(drop=True)
    prompt_table.to_csv(out_dir / "prompt_ablation.csv", index=False)
    best = choose_best(table)
    selection = {
        "best_prompt": best,
        "selection_rule": SELECTION_RULE,
        "selection_keys": list(SELECTION_KEYS),
    }
    (out_dir / "best_prompt.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display = table.copy()
    numeric = [column for column in display.columns if column not in {"prompt_id", "aggregation"}]
    for column in numeric:
        display[column] = display[column].map(
            lambda value: str(int(value))
            if column in {"n_samples", "ng_gt_ok_classes", "n_evaluable_classes"}
            else f"{value:.6f}"
        )
    report = (
        "# B0+ prompt 与聚合消融\n\n"
        f"预注册选择规则：{SELECTION_RULE}\n\n"
        f"按该规则选中：**{best}**\n\n"
        + markdown_table(display)
        + "\n"
    )
    (out_dir / "prompt_aggregation_report.md").write_text(report, encoding="utf-8")
    return best


def analyze_five_crop(out_dir: Path, best_prompt: str) -> pd.DataFrame:
    direct = pd.read_csv(out_dir / f"{best_prompt}_views.csv")
    crops = pd.read_csv(out_dir / f"{best_prompt}_5crop_scores.csv")
    view_scores = (
        crops.groupby(
            ["class_name", "group_path", "image_path", "sample_label"],
            as_index=False,
        )
        .agg(p_yes=("p_yes", "max"))
    )
    direct_row = summarize_views(direct, "max", f"{best_prompt}_direct")
    crop_row = summarize_views(view_scores, "max", f"{best_prompt}_5crop")
    table = pd.DataFrame([direct_row, crop_row])
    batches = pd.read_csv(out_dir / f"{best_prompt}_5crop_batches.csv")
    group_latency = batches.groupby("group_path").total_s.sum()
    crop_index = table.prompt_id == f"{best_prompt}_5crop"
    table.loc[crop_index, "five_view_latency_p50_s"] = np.percentile(group_latency, 50)
    table.loc[crop_index, "five_view_latency_p95_s"] = np.percentile(group_latency, 95)
    table.to_csv(out_dir / "five_crop_ablation.csv", index=False)
    display = table.copy().fillna("")
    for column in display.columns:
        if column not in {"prompt_id", "aggregation"}:
            display[column] = display[column].map(
                lambda value: "" if value == "" else f"{float(value):.6f}"
            )
    report = "# B0w-5crop 消融\n\n" + markdown_table(display) + "\n"
    (out_dir / "five_crop_report.md").write_text(report, encoding="utf-8")
    return table


def main() -> None:
    out_dir = Path("artifacts/b0plus")
    best = analyze_prompts(out_dir)
    if (out_dir / f"{best}_5crop_scores.csv").is_file():
        analyze_five_crop(out_dir, best)
    print((out_dir / "prompt_aggregation_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
