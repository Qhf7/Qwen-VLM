#!/usr/bin/env python3
"""Analyze Qwen/Jev disagreement without new API calls or GPU forward."""
from __future__ import annotations

from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import socket

import numpy as np


HOST = "autodl-container-132444827f-9c594038"
ROOT = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_hardcase_integration_20261111")
PILOT = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111/online_route_pilot")
SCORES = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111/stage09_qwen_matched/scores.jsonl")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    assert socket.gethostname() == HOST
    labels_doc = json.loads((PILOT / "approved_labels24.json").read_text())
    targets_doc = json.loads((PILOT / "approved_targets24.json").read_text())
    labels = labels_doc["labels"]
    targets = {r["source_id"]: r for r in targets_doc["records"]}
    assert len(labels) == len(targets) == 24

    scores = {}
    for row in load_jsonl(SCORES):
        if row.get("arm") == "decision_decision":
            scores[row["source_id"]] = row
    assert set(labels).issubset(scores)
    duplicate = {r["source_id"]: r for r in load_jsonl(PILOT / "final/predictions.jsonl") if r["arm"] == "duplicate"}
    roi = {r["source_id"]: r for r in load_jsonl(PILOT / "final/predictions.jsonl") if r["arm"] == "roi"}
    gate = {r["source_id"]: r for r in load_jsonl(PILOT / "gate/predictions.jsonl")}
    assert set(labels).issubset(duplicate) and set(gate).issubset(labels)

    rows = []
    for sid in labels:
        q = float(scores[sid]["anomaly_log_odds"])
        q_pred = int(q > 0)
        d = float(duplicate[sid]["probability"])
        d_pred = int(d >= 0.5)
        r = float(roi[sid]["probability"]) if sid in roi else None
        r_pred = int(r >= 0.5) if r is not None else None
        g = float(gate[sid]["probability"])
        area = float(targets[sid].get("candidate_iou", 0.0))  # retained only as viewed diagnostic, not queue input
        # Queue priority uses only pre-label model/geometry signals. The target
        # IoU is retained in the private analysis output for post-hoc grouping,
        # but never used to select a human-review item.
        priority = 0
        reasons = []
        if q_pred != d_pred:
            priority += 4
            reasons.append("qwen_duplicate_disagreement")
        if r_pred is not None and q_pred != r_pred:
            priority += 3
            reasons.append("qwen_roi_disagreement")
        if abs(q) < 1.0:
            priority += 2
            reasons.append("qwen_near_threshold")
        if abs(g - 0.5) < 0.2:
            priority += 1
            reasons.append("jev_gate_uncertain")
        rows.append({
            "source_id": sid,
            "category": duplicate[sid].get("category"),
            "qwen_log_odds": q,
            "qwen_pred": q_pred,
            "duplicate_probability": d,
            "duplicate_pred": d_pred,
            "roi_probability": r,
            "roi_pred": r_pred,
            "jev_gate_probability": g,
            "candidate_iou_viewed_diagnostic": area,
            "label_viewed_diagnostic": int(labels[sid]),
            "priority": priority,
            "reasons": reasons,
        })

    public_queue = sorted(rows, key=lambda x: (-x["priority"], abs(x["qwen_log_odds"]), x["source_id"]))
    qwen_errors = sum(r["qwen_pred"] != r["label_viewed_diagnostic"] for r in rows)
    duplicate_errors = sum(r["duplicate_pred"] != r["label_viewed_diagnostic"] for r in rows)
    disagreement = [r for r in rows if r["qwen_pred"] != r["duplicate_pred"]]
    roi_disagreement = [r for r in rows if r["roi_pred"] is not None and r["qwen_pred"] != r["roi_pred"]]
    result = {
        "schema": "jev-hardcase-disagreement-analysis-v1",
        "status": "complete",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "source_files_sha256": {
            "scores": sha(SCORES),
            "approved_labels24": sha(PILOT / "approved_labels24.json"),
            "approved_targets24": sha(PILOT / "approved_targets24.json"),
            "duplicate_predictions": sha(PILOT / "final/predictions.jsonl"),
            "gate_predictions": sha(PILOT / "gate/predictions.jsonl"),
        },
        "n": 24,
        "qwen_duplicate_disagreement_count": len(disagreement),
        "qwen_roi_disagreement_count": len(roi_disagreement),
        "qwen_error_count_viewed_diagnostic": qwen_errors,
        "duplicate_error_count_viewed_diagnostic": duplicate_errors,
        "priority_queue_selection": "model disagreement, score margin, and gate uncertainty only; viewed target IoU/labels never used",
        "api_requests": 0,
        "gpu_forward": False,
        "stage13_data_read": False,
        "rows_private_analysis": rows,
        "public_queue": [
            {k: r[k] for k in ["source_id", "category", "qwen_log_odds", "duplicate_probability", "roi_probability", "jev_gate_probability", "priority", "reasons"]}
            for r in public_queue
        ],
        "limitations": [
            "All 24 sources and predictions were previously viewed development artifacts",
            "Target labels and candidate IoU are retained only for post-hoc diagnostic grouping",
            "Priority queue itself excludes labels and target IoU",
            "No new API, image opening, GPU forward or confirmation read",
        ],
    }
    (ROOT / "hardcase_disagreement_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 困难样本分歧分析",
        "",
        "本分析只读取已有 Qwen DD、Jev gate、duplicate/ROI 结果和 24 图 pilot 元数据，不发 API、不做 GPU forward。公开复核顺序只使用模型分歧、分数边界和 gate 不确定性，不使用标签或 GT IoU 选样。",
        "",
        f"- Qwen 与 duplicate 预测分歧：{len(disagreement)}/24",
        f"- Qwen 与 ROI 预测分歧：{len(roi_disagreement)}/{len(roi)} 个有 ROI 结果的来源",
        f"- Qwen viewed-diagnostic errors：{qwen_errors}/24；duplicate viewed-diagnostic errors：{duplicate_errors}/24",
        "",
        "## 人工复核优先顺序（匿名来源）",
        "",
        "| 排序 | source_id | Qwen log-odds | duplicate p | ROI p | gate p | 原因 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(public_queue, 1):
        roi_text = "" if r["roi_probability"] is None else f"{r['roi_probability']:.3f}"
        lines.append(f"| {i} | `{r['source_id']}` | {r['qwen_log_odds']:.4f} | {r['duplicate_probability']:.3f} | {roi_text} | {r['jev_gate_probability']:.3f} | {', '.join(r['reasons'])} |")
    lines += [
        "",
        "该排序用于人工复核安排，不构成新的测试集或泛化结论。填写时仍应使用现有匿名 review packet，并保留 `uncertain`。",
    ]
    (ROOT / "HARDCASE_DISAGREEMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "qwen_duplicate_disagreement": len(disagreement), "qwen_roi_disagreement": len(roi_disagreement), "api_requests": 0, "gpu_forward": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
