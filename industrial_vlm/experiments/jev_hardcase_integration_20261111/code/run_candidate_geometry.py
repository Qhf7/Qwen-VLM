#!/usr/bin/env python3
"""Generate and evaluate pre-registered candidate geometry rules.

This script runs only on the approved AutoDL container.  Generation never reads
ground-truth masks or labels.  Evaluation happens only after prediction_freeze
exists and uses the already approved stage05 target boxes for the viewed
development set; it never touches confirmation data.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image, ImageOps
import torch


HOST = "autodl-container-132444827f-9c594038"
MAP_SIZE = 224
PEAK_SIZE = 56


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        y, x = np.unravel_index(int(np.argmax(mask.astype(np.float32))), mask.shape)
        return int(x), int(y), int(x + 1), int(y + 1)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def union95(prob: np.ndarray) -> dict:
    threshold = float(np.quantile(prob, 0.95, method="linear"))
    mask = prob >= threshold
    return {"bbox_map": list(bbox_from_mask(mask)), "threshold": threshold, "rule_params": {"quantile": 0.95}}


def connected95(prob: np.ndarray) -> dict:
    threshold = float(np.quantile(prob, 0.95, method="linear"))
    mask = prob >= threshold
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))])
        seen[y, x] = True
        pixels = []
        while q:
            cy, cx = q.popleft()
            pixels.append((cy, cx))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < MAP_SIZE and 0 <= nx < MAP_SIZE and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        arr = np.asarray(pixels, dtype=np.int32)
        score = float(prob[arr[:, 0], arr[:, 1]].mean())
        area = int(len(pixels))
        min_y, min_x = map(int, arr.min(axis=0))
        max_y, max_x = map(int, arr.max(axis=0))
        components.append((score, area, min_y, min_x, (min_x, min_y, max_x + 1, max_y + 1)))
    if not components:
        return {"bbox_map": list(bbox_from_mask(mask)), "threshold": threshold, "rule_params": {"quantile": 0.95, "connectivity": 8}}
    # Highest mean score; deterministic tie-break by smaller area then top-left.
    components.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    score, area, min_y, min_x, bbox = components[0]
    return {
        "bbox_map": list(bbox),
        "threshold": threshold,
        "rule_params": {"quantile": 0.95, "connectivity": 8, "component_mean_score": score, "component_area": area},
    }


def peak56(prob: np.ndarray) -> dict:
    y, x = map(int, np.unravel_index(int(np.argmax(prob)), prob.shape))
    half = PEAK_SIZE // 2
    x0 = min(max(x - half, 0), MAP_SIZE - PEAK_SIZE)
    y0 = min(max(y - half, 0), MAP_SIZE - PEAK_SIZE)
    return {
        "bbox_map": [x0, y0, x0 + PEAK_SIZE, y0 + PEAK_SIZE],
        "threshold": None,
        "rule_params": {"window": PEAK_SIZE, "argmax_xy": [x, y]},
    }


RULES = {"union95": union95, "connected95": connected95, "peak56": peak56}


def original_bbox(map_bbox, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = map_bbox
    return [x0 / MAP_SIZE, y0 / MAP_SIZE, x1 / MAP_SIZE, y1 / MAP_SIZE]


def bbox_to_pixels(norm, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = norm
    return [
        max(0, int(np.floor(x0 * width))),
        max(0, int(np.floor(y0 * height))),
        min(width, int(np.ceil(x1 * width))),
        min(height, int(np.ceil(y1 * height))),
    ]


def load_context(root: Path):
    source_root = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111")
    cache_path = Path("/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/teacher_cache.pt")
    image_root = Path("/root/autodl-tmp/industrial_vlm/datasets/VisA")
    plan_path = source_root / "source_baseline/frozen/plan.json"
    stage3_path = source_root / "stages/03_candidates.json"
    assert socket.gethostname() == HOST
    assert plan_path.exists() and stage3_path.exists() and cache_path.exists()
    plan = json.loads(plan_path.read_text())
    stage3 = json.loads(stage3_path.read_text())
    assert stage3["status"] == "complete" and stage3["confirmation_opened"] is False
    assert len(plan["queries"]) == len(stage3["records"]) == 120
    query_by_id = {row["source_id"]: row for row in plan["queries"]}
    stage3_by_id = {row["source_id"]: row for row in stage3["records"]}
    assert set(query_by_id) == set(stage3_by_id)
    sidecar = json.loads(cache_path.with_suffix(".json").read_text())
    cache = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=False)
    assert cache["geometry"] == "square224" and cache["pipeline_version"] == "visa-square-binary-sklearn-v1"
    return source_root, cache_path, image_root, plan_path, stage3_path, plan, stage3_by_id, query_by_id, cache, sidecar


def generate(root: Path) -> None:
    if (root / "prediction_freeze.json").exists() or (root / "predictions.jsonl").exists():
        raise RuntimeError("generation outputs already exist; refusing to overwrite")
    source_root, cache_path, image_root, plan_path, stage3_path, plan, stage3_by_id, query_by_id, cache, sidecar = load_context(root)
    # The code directory may already exist because the caller uploads the script
    # before generation. Only result files are protected against overwrite.
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "jev-hardcase-candidate-geometry-v1",
        "status": "frozen_before_generation",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "query_count": 120,
        "rules": {"union95": {"quantile": 0.95}, "connected95": {"quantile": 0.95, "connectivity": 8}, "peak56": {"window_square224": 56}},
        "coordinate_chain": "square224 -> original image -> 448",
        "inputs": {
            "plan_sha256": sha256(plan_path),
            "stage03_candidates_sha256": sha256(stage3_path),
            "teacher_cache_sha256_from_sidecar": sidecar["cache_sha256"],
        },
        "ground_truth_masks_opened": False,
        "confirmation_opened": False,
        "api_requests": 0,
        "gpu_forward": False,
    }
    write_json(root / "protocol.json", protocol)
    rows = []
    for sid in sorted(query_by_id):
        q = query_by_id[sid]
        old = stage3_by_id[sid]
        idx = int(old["cache_index"])
        prob = cache["maps"][idx].detach().cpu().numpy()
        assert prob.shape == (224, 224) and np.isfinite(prob).all()
        image_path = (image_root / q["relative_image"]).resolve()
        with Image.open(image_path) as raw:
            orientation = int(raw.getexif().get(274, 1))
            assert orientation == 1
            image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = image.size
        rules = {}
        for name, fn in RULES.items():
            out = fn(prob)
            norm = original_bbox(out["bbox_map"], width, height)
            pixels = bbox_to_pixels(norm, width, height)
            assert pixels[2] > pixels[0] and pixels[3] > pixels[1]
            rules[name] = {
                "bbox_map_xyxy": out["bbox_map"],
                "bbox_original_norm": norm,
                "bbox_original_pixels": pixels,
                "area_fraction": float((norm[2] - norm[0]) * (norm[3] - norm[1])),
                "full_image": bool(norm == [0.0, 0.0, 1.0, 1.0]),
                "threshold": out["threshold"],
                "rule_params": out["rule_params"],
            }
        rows.append({
            "source_id": sid,
            "category": q["category"],
            "relative_image": q["relative_image"],
            "original_sha256": q["sha256"],
            "teacher_split": old["teacher_split"],
            "cache_index": idx,
            "map_sha256": hashlib.sha256(prob.tobytes()).hexdigest(),
            "original_size": [width, height],
            "exif_orientation": orientation,
            "rules": rules,
        })
    with (root / "predictions.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    freeze = {
        "schema": "jev-hardcase-candidate-prediction-freeze-v1",
        "status": "complete",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / "protocol.json"),
        "predictions_sha256": sha256(root / "predictions.jsonl"),
        "n_queries": len(rows),
        "ground_truth_masks_opened": False,
        "confirmation_opened": False,
        "api_requests": 0,
        "gpu_forward": False,
    }
    write_json(root / "prediction_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + ba - inter
    return inter / union if union else 0.0


def evaluate(root: Path) -> None:
    freeze = json.loads((root / "prediction_freeze.json").read_text())
    assert freeze["status"] == "complete" and freeze["ground_truth_masks_opened"] is False
    assert not freeze["confirmation_opened"] and freeze["api_requests"] == 0
    source_root = Path("/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111")
    stage5_path = source_root / "stages/05_evaluation.json"
    stage5 = json.loads(stage5_path.read_text())
    assert stage5["status"] == "complete" and stage5["confirmation_opened"] is False
    pred_rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines() if line.strip()]
    old_rows = {r["source_id"]: r for r in stage5["records"]}
    assert len(pred_rows) == len(old_rows) == 120 and set(r["source_id"] for r in pred_rows) == set(old_rows)
    labels = {sid: int(old_rows[sid]["label"]) for sid in old_rows}
    assert sum(labels.values()) == 60
    rules = ["union95", "connected95", "peak56"]
    metrics = {name: {"all": [], "anomaly": [], "normal": []} for name in rules}
    detailed = []
    for row in pred_rows:
        old = old_rows[row["source_id"]]
        gt = old["gt_bbox_original_norm"]
        label = labels[row["source_id"]]
        for name in rules:
            pred = row["rules"][name]
            score = iou(pred["bbox_original_norm"], gt) if label else None
            item = {"source_id": row["source_id"], "category": row["category"], "label": label, "rule": name, "iou": score, "area_fraction": pred["area_fraction"], "full_image": pred["full_image"]}
            metrics[name]["all"].append(item)
            metrics[name]["anomaly" if label else "normal"].append(item)
            detailed.append(item)
    def summarize(items):
        anomalies = [x for x in items if x["label"] == 1]
        normals = [x for x in items if x["label"] == 0]
        ious = [x["iou"] for x in anomalies]
        return {
            "n": len(items),
            "anomalous": len(anomalies),
            "normal": len(normals),
            "mean_iou_all_anomaly_denominator": float(np.mean(ious)) if ious else None,
            "iou_ge_0_5_anomaly_rate": float(np.mean([x["iou"] >= 0.5 for x in anomalies])) if anomalies else None,
            "normal_box_rate": float(np.mean([x["area_fraction"] > 0 for x in normals])) if normals else None,
            "normal_full_image_rate": float(np.mean([x["full_image"] for x in normals])) if normals else None,
            "full_image_rate": float(np.mean([x["full_image"] for x in items])) if items else None,
            "anomaly_area_coverage_proxy": float(np.mean([x["area_fraction"] for x in anomalies])) if anomalies else None,
        }
    result = {
        "schema": "jev-hardcase-candidate-geometry-results-v1",
        "status": "complete",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "prediction_freeze_sha256": sha256(root / "prediction_freeze.json"),
        "stage05_targets_sha256": sha256(stage5_path),
        "rules": {name: summarize(metrics[name]["all"]) for name in rules},
        "denominators": {"all": 120, "anomaly": 60, "normal": 60},
        "ground_truth_source": "already-approved stage05 target boxes; no new mask file read",
        "confirmation_opened": False,
        "api_requests": 0,
        "gpu_forward": False,
        "details": detailed,
    }
    write_json(root / "metrics.json", result)
    lines = [
        "# 候选几何审计结果",
        "",
        "本报告仅评价已冻结的 120 张开发 query；候选生成阶段未读取 GT。评价阶段读取既有 stage05 已批准目标框，不打开 confirmation，不调用 Jev API，不做 GPU forward。",
        "",
        "| 规则 | 异常 mean IoU | IoU≥0.5 | 正常出框率 | 正常整图框率 | 全部整图框率 | 异常候选平均面积 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in rules:
        s = result["rules"][name]
        lines.append(f"| {name} | {s['mean_iou_all_anomaly_denominator']:.6f} | {s['iou_ge_0_5_anomaly_rate']:.2%} | {s['normal_box_rate']:.2%} | {s['normal_full_image_rate']:.2%} | {s['full_image_rate']:.2%} | {s['anomaly_area_coverage_proxy']:.4f} |")
    lines += [
        "",
        "所有异常指标以 60 张异常图为分母，正常出框率以 60 张正常图为分母。面积分层和逐图结果保存在 metrics.json；本轮结果属于已看开发集工程诊断，不是新盲测。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "rules": result["rules"], "confirmation_opened": False, "api_requests": 0}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["generate", "evaluate"], required=True)
    args = parser.parse_args()
    if args.phase == "generate":
        generate(args.root)
    else:
        evaluate(args.root)


if __name__ == "__main__":
    main()
