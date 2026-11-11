#!/usr/bin/env python3
"""Retrospective Qwen-DD gate replay on the frozen 24-image online pilot."""
from __future__ import annotations

from pathlib import Path
import datetime as dt
import hashlib
import json
import socket

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HOST = "autodl-container-132444827f-9c594038"
BASE = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_decision_round2_20261111/online_route_pilot")
SCORES = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111/stage09_qwen_matched/scores.jsonl")
OUT = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_hardcase_integration_20261111")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(rows):
    y = np.asarray([r["label"] for r in rows], dtype=int)
    p = np.asarray([r["probability"] for r in rows], dtype=float)
    pred = p >= 0.5
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    return {
        "n": len(rows),
        "normal": int((y == 0).sum()),
        "anomalous": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / int((y == 1).sum()),
        "fpr": fp / int((y == 0).sum()),
    }


def main():
    assert socket.gethostname() == HOST
    labels_doc = json.loads((BASE / "approved_labels24.json").read_text())
    targets_doc = json.loads((BASE / "approved_targets24.json").read_text())
    routing = json.loads((BASE / "routing.json").read_text())
    labels = labels_doc["labels"]
    source_ids = list(labels)
    assert len(source_ids) == 24 and sum(labels.values()) == 12
    assert routing["status"] == "route_frozen_before_final_requests" and routing["K"] == 8
    eligible = set(routing["union_roi"])
    assert len(eligible) == 13 and eligible.issubset(source_ids)

    score_rows = {}
    for line in SCORES.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm") == "decision_decision":
            score_rows[row["source_id"]] = row
    assert set(source_ids).issubset(score_rows)

    final_rows = {}
    for line in (BASE / "final/predictions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        final_rows.setdefault(row["source_id"], {})[row["arm"]] = float(row["probability"])
    assert set(source_ids).issubset(final_rows)
    assert all("duplicate" in final_rows[sid] for sid in source_ids)
    assert all(sid in final_rows and "roi" in final_rows[sid] for sid in eligible)

    qwen_scores = {sid: float(score_rows[sid]["anomaly_log_odds"]) for sid in source_ids}
    qwen_selected = {sid for sid in eligible if qwen_scores[sid] > 0.0}

    def replay(selected):
        rows = []
        for sid in source_ids:
            prob = final_rows[sid]["roi"] if sid in selected else final_rows[sid]["duplicate"]
            rows.append({"source_id": sid, "label": int(labels[sid]), "probability": prob, "used_roi": sid in selected})
        return rows

    policy_rows = {"qwen_dd_gate": replay(qwen_selected)}
    for name, ids in routing["routes"].items():
        policy_rows[name] = replay(set(ids))
    policy_rows["new_duplicate"] = replay(set())

    result = {
        "schema": "jev-qwen-dd-gate-retro-v1",
        "status": "complete",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "source_files_sha256": {
            "scores": sha(SCORES),
            "approved_labels24": sha(BASE / "approved_labels24.json"),
            "approved_targets24": sha(BASE / "approved_targets24.json"),
            "routing": sha(BASE / "routing.json"),
            "final_predictions": sha(BASE / "final/predictions.jsonl"),
        },
        "n": 24,
        "normal": 12,
        "anomalous": 12,
        "eligible": 13,
        "K": 8,
        "qwen_threshold": "anomaly_log_odds > 0",
        "qwen_semantics": "whole-image anomaly score; not crop-needed probability",
        "qwen_selected_eligible": sorted(qwen_selected),
        "qwen_selected_count": len(qwen_selected),
        "api_requests": 0,
        "gpu_forward": False,
        "stage13_data_read": False,
        "label_semantics": "image-label proxy for candidate route replay",
        "results": {name: metrics(rows) | {"roi_calls": sum(r["used_roi"] for r in rows), "rows": rows} for name, rows in policy_rows.items()},
        "limitations": [
            "Qwen DD score was trained/evaluated as whole-image anomaly score, not crop-needed semantics",
            "The 24 sources and final duplicate/ROI predictions are already-viewed development pilot artifacts",
            "No threshold was fit on these 24 labels; fixed anomaly_log_odds>0 was reused",
            "No new API request or GPU forward was made",
        ],
    }
    (OUT / "qwen_dd_gate_retro_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Qwen DD gate 回顾结果",
        "",
        "本报告重放已冻结的 24 张在线 pilot 开发图，不新增 API、不做 GPU forward、不在 24 张标签上调阈值。Qwen DD 的 `anomaly_log_odds>0` 是整图异常分数门控，不是 crop-needed 语义；所有路线指标均为 image-label proxy。",
        "",
        "| 策略 | AUROC | AP | Recall | FPR | ROI调用数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in result["results"].items():
        lines.append(f"| {name} | {d['auroc']:.4%} | {d['ap']:.4%} | {d['recall']:.2%} | {d['fpr']:.2%} | {d['roi_calls']} |")
    lines += [
        "",
        f"Qwen DD 固定阈值在 13 个 eligible 候选中选择 {len(qwen_selected)} 个进入 ROI 回放。该结果只能用于低成本策略诊断；若 Qwen DD gate 不劣于 Jev gate，则没有为 gate 单独支付 Jev 调用费用的理由。",
    ]
    (OUT / "QWEN_DD_GATE_RETRO_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "selected": len(qwen_selected), "results": {k: {m: v for m, v in d.items() if m != "rows"} for k, d in result["results"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
