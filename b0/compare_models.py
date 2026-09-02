"""汇总两个 B0 视觉模型在同一冻结子集上的结果。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


OUT = Path("artifacts/b0")
SLUGS = ("qwen3_vl_2b", "qwen25_vl_3b")


def summarize(slug: str) -> dict:
    views = pd.read_csv(OUT / f"{slug}_views.csv")
    config = json.loads((OUT / f"{slug}_config.json").read_text(encoding="utf-8"))
    samples = (
        views.groupby(["class_name", "group_path"], as_index=False)
        .agg(sample_label=("sample_label", "first"), score=("p_yes", "max"),
             latency_s=("total_s", "sum"))
    )
    per_class = [
        roc_auc_score(part.sample_label, part.score)
        for _, part in samples.groupby("class_name")
        if part.sample_label.nunique() == 2
    ]
    cold_group = views.iloc[0].group_path
    warm = views[views.group_path != cold_group]
    warm_samples = samples[samples.group_path != cold_group]
    positive = views[(views.view_label == 1) & views.json_ok]
    grids = sorted(
        set(zip(views.grid_t.astype(int), views.grid_h.astype(int), views.grid_w.astype(int)))
    )
    labels = samples.sample_label.to_numpy()
    scores = samples.score.to_numpy()
    return {
        "slug": slug,
        "model": config["model_id"],
        "parameters_B": config["n_parameters"] / 1e9,
        "views": len(views),
        "samples": len(samples),
        "visual_grids": ";".join("x".join(map(str, grid)) for grid in grids),
        "answer_parse_rate": views.answer_ok.mean(),
        "json_parse_rate": views.json_ok.mean(),
        "yes_output_rate": (views.parsed_answer == "Yes").mean(),
        "view_accuracy": ((views.p_yes >= .5).astype(int) == views.view_label).mean(),
        "sample_accuracy": ((scores >= .5).astype(int) == labels).mean(),
        "sample_auroc": roc_auc_score(labels, scores),
        "sample_aupr": average_precision_score(labels, scores),
        "macro_class_auroc": float(np.mean(per_class)),
        "region_accuracy": (positive.parsed_region == positive.gt_region).mean(),
        "cold_view_s": views.iloc[0].total_s,
        "warm_view_p50_s": np.percentile(warm.total_s, 50),
        "warm_view_p95_s": np.percentile(warm.total_s, 95),
        "five_view_p50_s": np.percentile(warm_samples.latency_s, 50),
        "five_view_p95_s": np.percentile(warm_samples.latency_s, 95),
    }


def main() -> None:
    first = pd.read_csv(OUT / f"{SLUGS[0]}_views.csv")
    second = pd.read_csv(OUT / f"{SLUGS[1]}_views.csv")
    assert set(first.image_path) == set(second.image_path), "两模型不是同一视觉子集"
    table = pd.DataFrame(summarize(slug) for slug in SLUGS)
    table.to_csv(OUT / "comparison.csv", index=False)

    percent = {
        "answer_parse_rate", "json_parse_rate", "yes_output_rate", "view_accuracy",
        "sample_accuracy", "region_accuracy",
    }
    markdown = table.copy()
    for column in percent:
        markdown[column] = markdown[column].map(lambda value: f"{value:.1%}")
    for column in set(markdown.columns) - percent - {"slug", "model", "visual_grids"}:
        markdown[column] = markdown[column].map(lambda value: f"{value:.4f}")
    headers = markdown.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in markdown.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    text = "# B0 同子集对照\n\n" + "\n".join(lines) + "\n"
    (OUT / "comparison.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
