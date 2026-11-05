"""Audit completed v2 training and summarize five validation evaluations.

Run locally or on the server with ordinary Python; no torch/CUDA is imported.
All required results must be complete before any report is replaced. This
script reads neither confirmation examples nor confirmation predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from industrial_vlm.compare import (  # noqa: E402
    OPTIONAL_CONDITIONS, REQUIRED_CONDITIONS, _load_run, paired_comparison,
)
from industrial_vlm.metrics import _summary  # noqa: E402

ARMS = ("region_region", "decision_decision", "region_decision")
EVALUATIONS = (
    ("region_region", "region"), ("region_region", "decision"),
    ("decision_decision", "decision"),
    ("region_decision", "region"), ("region_decision", "decision"),
)
PAIRS = (
    ("region_decision_decision", "region_region_decision", "common_decision_RD_minus_RR"),
    ("region_decision_decision", "decision_decision_decision", "common_decision_RD_minus_DD"),
    ("region_region_decision", "decision_decision_decision", "common_decision_RR_minus_DD"),
    ("region_decision_region", "region_region_region", "common_region_RD_minus_RR"),
)
MATCHED_TRAINING = (
    "model", "adapter", "max_steps", "seed", "sample_order", "max_pixels", "max_length",
    "micro_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
    "max_grad_norm", "warmup_steps", "lora_r", "lora_alpha", "lora_dropout",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row["source_id"] + "\n").encode())
    return digest.hexdigest()


def training_audit(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    updates = plan["optimizer_updates_per_arm"]
    n_exposures = plan["source_exposures_per_arm"]
    n_sources = plan["unique_training_sources_per_arm"]
    accumulation = plan["gradient_accumulation_steps"]
    if updates * accumulation != n_exposures or n_exposures != n_sources * 2:
        raise ValueError("Plan must specify two exposures per source and a consistent update budget")
    summaries, configurations, initial_hashes, sequence_hashes = {}, {}, set(), set()
    for arm in ARMS:
        directory = root / "runs/multitask_v2" / arm
        summary = read_json(directory / "summary.json")
        config = read_json(directory / "run_config.json")
        exposure_path = directory / "exposures.jsonl"
        exposures = read_jsonl(exposure_path)
        planned_path = root / "prepared/multitask_v2" / f"train_{arm}.jsonl"
        planned = read_jsonl(planned_path)
        if summary.get("status") != "complete" or summary.get("reload_check", {}).get("passed") is not True:
            raise ValueError(f"{arm}: training and reload check must be complete")
        if (summary.get("optimizer_steps"), summary.get("microsteps"), len(exposures)) != (updates, n_exposures, n_exposures):
            raise ValueError(f"{arm}: update/exposure budget differs from the plan")
        if config.get("sample_order") != "sequential" or config.get("adapter") is not None:
            raise ValueError(f"{arm}: expected sequential schedule and fresh initialization")
        if config.get("max_steps") != updates:
            raise ValueError(f"{arm}: configured update count differs from the plan")
        if config.get("train_sha256") != sha(planned_path) or len(planned) != n_exposures:
            raise ValueError(f"{arm}: training-file hash or count differs from prepared schedule")
        if config.get("seed") != plan["seed"] or config.get("gradient_accumulation_steps") != accumulation:
            raise ValueError(f"{arm}: seed or accumulation differs from the plan")
        counts = Counter(row["source_id"] for row in exposures)
        if len(counts) != n_sources or set(counts.values()) != {2}:
            raise ValueError(f"{arm}: each of {n_sources} sources must occur exactly twice")
        by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"exposures": 0, "target_tokens": 0, "input_tokens": 0})
        source_tasks: dict[str, Counter] = defaultdict(Counter)
        for index, (actual, expected) in enumerate(zip(exposures, planned)):
            task = "region" if "boxes" in json.loads(expected["answer"]) else "decision"
            expected_fields = (index // accumulation + 1, index % accumulation, expected["id"], expected["source_id"], task)
            actual_fields = tuple(actual.get(key) for key in ("optimizer_step", "microstep", "id", "source_id", "task"))
            if actual_fields != expected_fields:
                raise ValueError(f"{arm}: actual exposure {index} differs from the prepared sequence")
            if any(type(actual.get(key)) is not int or actual[key] <= 0 for key in ("target_tokens", "input_tokens")):
                raise ValueError(f"{arm}: invalid token counts at exposure {index}")
            by_task[task]["exposures"] += 1
            by_task[task]["target_tokens"] += actual["target_tokens"]
            by_task[task]["input_tokens"] += actual["input_tokens"]
            source_tasks[actual["source_id"]][task] += 1
        expected_tasks = ({"region": 2} if arm == "region_region" else
                          {"decision": 2} if arm == "decision_decision" else {"region": 1, "decision": 1})
        if any(dict(tasks) != expected_tasks for tasks in source_tasks.values()):
            raise ValueError(f"{arm}: per-source task allocation differs from the arm definition")
        for key, total_key in (("target_tokens", "total_target_tokens"), ("input_tokens", "total_input_tokens")):
            if sum(task[key] for task in by_task.values()) != summary.get(total_key):
                raise ValueError(f"{arm}: exposure {key} sum disagrees with training summary")
        digest = source_digest(exposures)
        if summary.get("source_exposure_sha256") != digest or summary.get("exposure_trace_sha256") != sha(exposure_path):
            raise ValueError(f"{arm}: exposure digest disagrees with training summary")
        initial = summary.get("initial_trainable_sha256")
        if not isinstance(initial, str) or len(initial) != 64:
            raise ValueError(f"{arm}: missing initial trainable-weight digest")
        parameter_audit = read_json(directory / "parameter_audit.json")
        if parameter_audit.get("initial_trainable_sha256") != initial:
            raise ValueError(f"{arm}: initial-weight digest disagrees with parameter audit")
        initial_hashes.add(initial)
        sequence_hashes.add(digest)
        configurations[arm] = config
        summaries[arm] = {**summary, "unique_sources": len(counts), "by_task": dict(sorted(by_task.items())),
                          "run_config": config, "exposure_file": str(exposure_path.resolve())}
    reference = configurations[ARMS[0]]
    for arm, config in configurations.items():
        for key in MATCHED_TRAINING:
            if config.get(key) != reference.get(key):
                raise ValueError(f"{arm}: unmatched training condition {key}")
    if len(initial_hashes) != 1 or len(sequence_hashes) != 1:
        raise ValueError("Training arms differ in fresh initialization or actual source order")
    return {"status": "passed", "source_exposure_sha256": next(iter(sequence_hashes)),
            "initial_trainable_sha256": next(iter(initial_hashes)), "matched_training_fields": list(MATCHED_TRAINING),
            "arms": summaries}


def prompt_consistency(region: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Align different task IDs by source; invalid outputs never count as agreements."""
    for key in REQUIRED_CONDITIONS + OPTIONAL_CONDITIONS + ("adapter_files_sha256",):
        if region["config"].get(key) != decision["config"].get(key):
            raise ValueError(f"Cross-prompt comparison differs in {key}")
    maps = [{row["source_id"]: row for row in run["rows"]} for run in (region, decision)]
    if any(len(rows) != len(run["rows"]) for rows, run in zip(maps, (region, decision))):
        raise ValueError("Prompt consistency requires exactly one row per source per prompt")
    if set(maps[0]) != set(maps[1]):
        raise ValueError("Cross-prompt source sets differ")
    counts = Counter()
    by_label: dict[str, Counter] = {"normal": Counter(), "anomalous": Counter()}
    transitions = []
    for source in sorted(maps[0]):
        a, b = maps[0][source], maps[1][source]
        if any(a.get(key) != b.get(key) for key in ("label", "gt_boxes", "category", "anomaly_area_fraction")):
            raise ValueError(f"Cross-prompt ground truth differs: {source}")
        if a["localization_eligible"] is not True or b["localization_eligible"] is not False:
            raise ValueError("Expected region and decision tasks in that order")
        av = a["parsed"] is not None
        bv = b["parsed"] is not None
        ap = a["parsed"]["is_anomalous"] if av else None
        bp = b["parsed"]["is_anomalous"] if bv else None
        ac, bc = av and ap == a["label"], bv and bp == b["label"]
        events = {"n_sources": 1, "both_valid": av and bv,
                  "invalid_region": not av, "invalid_decision": not bv,
                  "invalid_any": not (av and bv), "invalid_both": not av and not bv,
                  "agreement": av and bv and ap == bp,
                  "region_true_to_decision_false": av and bv and ap is True and bp is False,
                  "region_false_to_decision_true": av and bv and ap is False and bp is True,
                  "both_correct": ac and bc, "region_correct": ac, "decision_correct": bc,
                  "region_correct_to_decision_incorrect": ac and not bc,
                  "region_incorrect_to_decision_correct": not ac and bc}
        label = "anomalous" if a["label"] else "normal"
        counts.update({key: int(value) for key, value in events.items()})
        by_label[label].update({key: int(value) for key, value in events.items()})
        if not (av and bv and ap == bp):
            transitions.append({"source_id": source, "label": a["label"], "category": a["category"],
                                "region_prediction": ap, "decision_prediction": bp,
                                "region_valid": av, "decision_valid": bv})
    n = counts["n_sources"]
    return {"region_run": region["name"], "decision_run": decision["name"], "counts": dict(counts),
            "agreement_rate_all_sources": counts["agreement"] / n,
            "agreement_rate_among_both_valid": counts["agreement"] / counts["both_valid"] if counts["both_valid"] else None,
            "both_correct_rate_all_sources": counts["both_correct"] / n,
            "by_label": {key: dict(value) for key, value in by_label.items()}, "flips_or_invalid_sources": transitions,
            "definition": "Region prompt -> decision prompt; boolean agreement requires both parsed outputs valid. Invalid outputs remain in the all-source denominator; both-correct rate also uses all sources."}


def artifact_hashes(root: Path) -> dict[str, str]:
    files = [root / "prepared/multitask_v2/plan.json", Path(__file__).resolve(),
             root / "src/industrial_vlm/compare.py", root / "src/industrial_vlm/metrics.py",
             root / "prepared/multitask_v2/val_region_120.jsonl",
             root / "prepared/multitask_v2/val_decision_120.jsonl"]
    for arm in ARMS:
        files.extend(root / "runs/multitask_v2" / arm / name for name in
                     ("summary.json", "run_config.json", "parameter_audit.json", "exposures.jsonl"))
        files.append(root / "prepared/multitask_v2" / f"train_{arm}.jsonl")
    for arm, task in EVALUATIONS:
        files.extend(root / "evaluations/multitask_v2" / f"{arm}_{task}" / name for name in
                     ("meta.json", "metrics.json", "predictions.jsonl"))
    for name in ("training_source_manifest.json", "input_hashes.json"):
        path = root / "experiments/multitask_v2" / name
        if path.is_file():
            files.append(path)
    return {str(path.relative_to(root) if path.is_relative_to(root) else path): sha(path) for path in files}


def build(root: Path, *, samples: int = 2000, seed: int = 17) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = read_json(root / "prepared/multitask_v2/plan.json")
    training = training_audit(root, plan)
    loaded, runs = {}, []
    for arm, task in EVALUATIONS:
        name = f"{arm}_{task}"
        run, error = _load_run(root / "evaluations/multitask_v2", name)
        if run is None:
            raise ValueError(f"{name}: {error}")
        if run["config"].get("branch") != "multitask_v2_validation":
            raise ValueError(f"{name}: expected validation branch")
        validation_path = root / "prepared/multitask_v2" / f"val_{task}_120.jsonl"
        expected = {row["id"]: row for row in read_jsonl(validation_path)}
        if len(expected) != 120 or len(run["rows"]) != 120 or set(expected) != {row["id"] for row in run["rows"]}:
            raise ValueError(f"{name}: expected the frozen120 validation task IDs")
        if run["config"].get("data_sha256") != sha(validation_path):
            raise ValueError(f"{name}: evaluated data hash differs from frozen validation inputs")
        for row in run["rows"]:
            expected_row = expected[row["id"]]
            for prediction_key, prepared_key in (("source_id", "source_id"), ("image", "image"),
                                                  ("label", "is_anomalous"), ("gt_boxes", "boxes"),
                                                  ("category", "category"), ("prompt", "prompt"),
                                                  ("anomaly_area_fraction", "anomaly_area_fraction")):
                if row.get(prediction_key) != expected_row.get(prepared_key):
                    raise ValueError(f"{name}: prediction metadata differs from validation inputs: {prediction_key}")
        if any(row["localization_eligible"] != (task == "region") for row in run["rows"]):
            raise ValueError(f"{name}: unexpected task type")
        # Use prediction-derived values; retained metadata includes the exact
        # data/config and adapter hashes already validated by compare._load_run.
        overall = _summary(run["rows"])
        stored = read_json(root / "evaluations/multitask_v2" / name / "metrics.json")["overall"]
        if overall != stored:
            raise ValueError(f"{name}: stored overall metrics differ from predictions")
        loaded[name] = run
        runs.append({"name": name, "arm": arm, "prompt_task": task, "overall": overall,
                     "directory": run["directory"], "config": run["config"]})
    comparisons = [{"branch": branch, **paired_comparison(loaded[left], loaded[right], samples=samples, seed=seed)}
                   for left, right, branch in PAIRS]
    consistency = {arm: prompt_consistency(loaded[f"{arm}_region"], loaded[f"{arm}_decision"])
                   for arm in ("region_region", "region_decision")}
    provenance = artifact_hashes(root)
    created = datetime.now(timezone.utc).isoformat()
    limitations = [
        "本轮结果来自同一批固定120来源的验证集，用于开发比较；冻结确认集未由本脚本读取或评估。",
        "三臂来源及曝光序列、更新数和初始化匹配；输入token、监督token、任务难度及计算耗时不相等。",
        "混合臂按监督token归一CE，两个任务的曝光数各半不意味着梯度权重各半。",
        "RD与RR/DD的比较反映固定曝光预算下的任务分配取舍，不能孤立解释为纯坐标信号的因果作用。",
        "固定最终192更新；不同目标任务的验证CE不可用作三臂共同性能排名；旧96更新结果仅为历史参考。",
        "配对bootstrap区间仅描述当前模型在验证来源重采样下的差异，不包含训练seed变化；四组多指标比较未做多重校正。",
        "Macro F1平均正常/异常两个标签；框指标为包围矩形的端到端召回，分母包含漏检与格式错误。",
        "格式有效性沿用现有parser，允许JSON围栏、浮点坐标和多框，不等于严格整数单框合规。",
    ]
    matrix = {"schema_version": "industrial-vlm-multitask-v2-validation-v1", "created_at": created,
              "status": "complete", "scope": "validation_development_comparison", "runs": runs,
              "training_audit": training, "prompt_consistency": consistency,
              "artifact_sha256": provenance, "limitations": limitations}
    paired = {"schema_version": "industrial-vlm-multitask-v2-paired-v1", "created_at": created,
              "status": "complete", "scope": "validation_development_comparison", "n_resamples": samples,
              "seed": seed, "method": "paired source_id cluster bootstrap, percentile95% intervals with linear interpolation",
              "difference_direction": "left minus right; JSON stores fractions, Markdown reports percentage points",
              "comparisons": comparisons, "artifact_sha256": provenance, "limitations": limitations}
    return matrix, paired


def percent(value: Any) -> str:
    return "—" if value is None else f"{value * 100:.2f}"


def number(value: Any, decimals: int = 2) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def matrix_markdown(result: dict[str, Any]) -> str:
    lines = ["# 多任务SFT v2：验证集开发比较", "",
             "RR = region+region；DD = decision+decision；RD = region+decision。下表指标来自相同验证来源，模型固定为最终第192次更新。", "",
             "| 训练臂 / 推理提示 | N | Macro F1 % | 异常召回 % | 正常FPR % | 框R@IoU.5 % | 格式有效 % | 延迟秒 | 推理峰值GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        m = run["overall"]
        peak = m.get("max_peak_memory_allocated_bytes")
        lines.append(f"| {run['name']} | {m['n']} | {percent(m['macro_f1'])} | {percent(m['anomaly_recall'])} | "
                     f"{percent(m['fpr'])} | {percent(m['union_box_recall_iou_0.5'])} | {percent(m['format_valid_rate'])} | "
                     f"{number(m.get('mean_latency_seconds'))} | {number(peak / 2**30 if peak is not None else None)} |")
    lines.extend(["", "## 同一adapter的提示一致性", "",
                  "方向均为 region提示 → decision提示；双有效一致率的分母为两次输出均有效的来源数，全来源一致率与两次均正确率的分母为全部来源。双无效不算一致。", "",
                  "| 臂 | 双有效/N | 全来源一致 % | 双有效一致 % | 两次均正确 % | true→false | false→true | region无效 | decision无效 | 任一无效 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, item in result["prompt_consistency"].items():
        c = item["counts"]
        lines.append(f"| {arm} | {c['both_valid']}/{c['n_sources']} | {percent(item['agreement_rate_all_sources'])} | "
                     f"{percent(item['agreement_rate_among_both_valid'])} | {percent(item['both_correct_rate_all_sources'])} | "
                     f"{c['region_true_to_decision_false']} | {c['region_false_to_decision_true']} | {c['invalid_region']} | "
                     f"{c['invalid_decision']} | {c['invalid_any']} |")
    lines.extend(["", "## 训练预算与实际监督", "",
                  "| 臂 | 来源 | 更新/曝光 | 训练秒* | 峰值allocated GiB | 峰值reserved GiB | 输入token | 监督token |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, summary in result["training_audit"]["arms"].items():
        lines.append(f"| {arm} | {summary['unique_sources']} | {summary['optimizer_steps']}/{summary['microsteps']} | "
                     f"{number(summary.get('train_wall_seconds_including_validation'))} | {number(summary.get('peak_allocated_gib'))} | "
                     f"{number(summary.get('peak_reserved_gib'))} | {summary['total_input_tokens']} | {summary['total_target_tokens']} |")
    lines.extend(["", "*训练秒包含训练循环内验证CE，不含初始模型加载及最终adapter重载。", "",
                  "| 臂 / 目标任务 | 曝光数 | 输入token | 监督token |", "|---|---:|---:|---:|"])
    for arm, summary in result["training_audit"]["arms"].items():
        for task, totals in summary["by_task"].items():
            lines.append(f"| {arm} / {task} | {totals['exposures']} | {totals['input_tokens']} | {totals['target_tokens']} |")
    lines.extend(["", "真实来源曝光序列与初始可训练权重哈希在三臂间一致；逐条曝光均已与冻结训练行序核对。详见JSON内审计与文件哈希。", "",
                  "## 解释边界", ""])
    lines.extend(f"- {text}" for text in result["limitations"])
    return "\n".join(lines) + "\n"


def paired_markdown(result: dict[str, Any]) -> str:
    lines = ["# 多任务SFT v2：验证集配对差异", "",
             f"对相同来源做{result['n_resamples']}次配对bootstrap，seed={result['seed']}，报告百分位95%区间。差值为左臂减右臂，单位为百分点；FPR/FNR越低越好。", "",
             "| 左 − 右（同提示） | 指标 | 左 % | 右 % | 差值pp | 95%区间pp | 有定义重采样数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for comparison in result["comparisons"]:
        for metric, estimate in comparison["metrics"].items():
            interval = ("—" if estimate["ci95_low"] is None else
                        f"[{estimate['ci95_low'] * 100:+.2f}, {estimate['ci95_high'] * 100:+.2f}]")
            difference = "—" if estimate["difference"] is None else f"{estimate['difference'] * 100:+.2f}"
            lines.append(f"| {comparison['left']} − {comparison['right']} | {metric} | {percent(estimate['left'])} | "
                         f"{percent(estimate['right'])} | {difference} | {interval} | {estimate['n_defined_resamples']} |")
    lines.extend(["", "区间不包含零仅描述本次验证对比在指定重采样方法下的差异；任务预算取舍与开发选择仍需结合定位、判别及提示一致性共同解释。", "",
                  "## 解释边界", ""])
    lines.extend(f"- {text}" for text in result["limitations"])
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be at least20")
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "reports/multitask_v2"
    matrix, paired = build(root, samples=args.bootstrap_samples, seed=args.seed)
    # Construct every artifact before replacing any existing report.
    documents = {
        "validation_matrix.json": json.dumps(matrix, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "validation_matrix.md": matrix_markdown(matrix),
        "paired_differences.json": json.dumps(paired, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "paired_differences.md": paired_markdown(paired),
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, text in documents.items():
        atomic_write(output / filename, text)
    print(json.dumps({"status": "complete", "evaluations": len(matrix["runs"]),
                      "paired_comparisons": len(paired["comparisons"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
