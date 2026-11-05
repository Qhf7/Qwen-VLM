"""Read six phase-one JSON result sets and write a paired, validation-only report.

Uses only the Python standard library. It never imports torch, opens model
checkpoints, reads predictions/training logs, or starts GPU work. Missing or
partially written results remain ongoing; only three complete, matched seed
pairs contribute to the final mean and sample standard deviation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


DEFAULT_ROOT = "/root/autodl-tmp/AnomalyGPT/runs/migration_20261103"
SEEDS = (42, 43, 44)
BACKENDS = ("vicuna", "qwen")
CONDITIONS = ("real_map", "zero_map", "donor_opposite_label_map", "spatial_shuffle_map")
FILES = ("run.json", "train_complete.json", "val_metrics.json")
COUNTS = {"train": 9985, "val": 380, "test": 456}
SCORE_KEYS = ("image_auroc", "image_ap", "accuracy", "macro_f1", "recall", "fpr", "invalid_rate")
GENERATED_KEYS = ("accuracy", "macro_f1", "recall", "fpr", "invalid_rate")
MAIN_KEYS = ("image_auroc", "image_ap", "generated_macro_f1", "generated_recall",
             "generated_fpr", "generated_invalid_rate")
PAIR_FIELDS = (
    "initialization_sha256", "decoder_sha256", "cache_sha256", "cache_manifest_sha256",
    "sample_schedule_sha256", "trainable_parameters", "pipeline_version", "training_protocol",
    "counts", "val_n", "precision", "loss", "selection", "config.seed", "config.steps",
    "config.batch_size", "config.accum", "config.val_every", "config.lr", "config.val_limit",
    "config.eval_batch_size", "config.eval_split", "config.generate_limit", "config.skip_map_controls",
)
CROSS_SEED_FIELDS = tuple(key for key in PAIR_FIELDS
                          if key not in {"initialization_sha256", "sample_schedule_sha256", "config.seed"})
MISSING = object()


def nested(value, path, default=MISSING):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_text(text, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    if not path.is_file():
        return None, {"state": "missing", "path": str(path)}
    try:
        raw = path.read_bytes()
        def reject_constant(text):
            raise ValueError("Nonfinite JSON constant: " + text)
        value = json.loads(raw, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, {"state": "unreadable", "path": str(path), "error": str(error)}
    except ValueError as error:
        return None, {"state": "invalid", "path": str(path), "error": str(error)}
    if not isinstance(value, dict):
        return None, {"state": "invalid", "path": str(path), "error": "JSON root must be an object"}
    return value, {"state": "read", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def audit_run_contract(run, backend, seed):
    errors = []
    for field in PAIR_FIELDS:
        if nested(run, field) is MISSING or nested(run, field) is None:
            errors.append("run.json 缺少字段 " + field)
    for field in ("initialization_sha256", "decoder_sha256", "cache_sha256",
                  "cache_manifest_sha256", "sample_schedule_sha256"):
        value = run.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            errors.append(field + " 不是有效 SHA256")
    expected = {"backend": backend, "config.backend": backend, "config.seed": seed,
                "training_protocol": "supervised_custom_60_20_20", "counts": COUNTS,
                "val_n": 380, "config.steps": 1500, "config.val_every": 250,
                "config.val_limit": 0, "config.generate_limit": 0, "config.eval_split": "val",
                "config.skip_map_controls": False, "config.eval_only": False}
    for field, value in expected.items():
        actual = nested(run, field)
        if actual is MISSING or actual != value:
            errors.append("%s 应为 %r，实际 %r" % (field, value, None if actual is MISSING else actual))
    for field in ("trainable_parameters", "config.steps", "config.batch_size", "config.accum",
                  "config.val_every", "config.eval_batch_size"):
        value = nested(run, field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(field + " 必须是正整数")
    batch, accum = nested(run, "config.batch_size"), nested(run, "config.accum")
    if isinstance(batch, int) and isinstance(accum, int) and batch * accum != 4:
        errors.append("batch_size × accum 必须为 4（1500 更新、6000 图片曝光）")
    if not numeric(nested(run, "config.lr")) or nested(run, "config.lr") <= 0:
        errors.append("config.lr 必须为有限正数")
    if nested(run, "initialization_audit.init_seed", seed) != seed:
        errors.append("initialization_audit.init_seed 与目录 seed 不同")
    return errors


def extract_channel(metrics, channel, keys, location, errors, pending):
    values = nested(metrics, "overall." + channel, None)
    if values is None:
        pending.append(location + " 尚无 overall." + channel)
        return None
    if not isinstance(values, dict):
        errors.append(location + " overall." + channel + " 格式错误")
        return None
    for key in ("n", "n_normal", "n_anomaly"):
        value = values.get(key)
        expected = {"n": 380, "n_normal": 172, "n_anomaly": 208}[key]
        if value is None:
            pending.append(location + " 缺少 " + key)
        elif value != expected:
            errors.append("%s %s 应为 %d，实际 %r" % (location, key, expected, value))
    result = {key: values.get(key) for key in keys}
    for key, value in result.items():
        if value is None:
            pending.append(location + " 缺少指标 " + key)
        elif not numeric(value) or not 0 <= value <= 1:
            errors.append(location + " 指标 " + key + " 必须是 [0,1] 有限数值")
            result[key] = None
    return result


def inspect_run(root, backend, seed):
    directory = root / ("lm_%s_s%d" % (backend, seed))
    documents, files = {}, {}
    for name in FILES:
        documents[name], files[name] = read_json(directory / name)
    run, trained, metrics = (documents[name] for name in FILES)
    errors = [name + ": " + info["error"] for name, info in files.items() if info["state"] == "invalid"]
    pending = [name + " " + info["state"] for name, info in files.items()
               if info["state"] in {"missing", "unreadable"}]
    if run is not None:
        errors.extend(audit_run_contract(run, backend, seed))
    completion = {}
    if trained is not None:
        for key in ("best_step", "best_val_auroc", "reload_max_abs_score_error"):
            value = trained.get(key)
            if value is None:
                pending.append("train_complete.json 缺少 " + key)
            elif not numeric(value):
                errors.append("train_complete.json " + key + " 必须为有限数值")
            else:
                completion[key] = value
        best = completion.get("best_step")
        if best is not None and (not isinstance(best, int) or not 1 <= best <= 1500):
            errors.append("best_step 必须为 1..1500 的整数")
        auc = completion.get("best_val_auroc")
        if auc is not None and not 0 <= auc <= 1:
            errors.append("best_val_auroc 必须在 [0,1]")
        reload_error = completion.get("reload_max_abs_score_error")
        if reload_error is not None and not 0 <= reload_error < 1e-5:
            errors.append("save/reload 最大分数误差应小于 1e-5")
        for key in ("elapsed_seconds", "peak_memory_gb"):
            if numeric(trained.get(key)):
                completion[key] = trained[key]
    conditions, generated = {}, None
    if metrics is not None:
        if metrics.get("split") != "val":
            errors.append("val_metrics.json split 必须是 val")
        if run is not None and metrics.get("decoder_sha256") != run.get("decoder_sha256"):
            errors.append("val_metrics.json decoder SHA 与 run.json 不同")
        selected = metrics.get("selected_step")
        if selected is None:
            pending.append("val_metrics.json 缺少 selected_step")
        elif completion.get("best_step") is not None and selected != completion["best_step"]:
            errors.append("val_metrics.json selected_step 与 train_complete best_step 不同")
        source_conditions = metrics.get("conditions", {})
        if not isinstance(source_conditions, dict):
            errors.append("conditions 必须是对象")
            source_conditions = {}
        for condition in CONDITIONS:
            if condition not in source_conditions:
                pending.append("尚未写入控制条件 " + condition)
                continue
            result = extract_channel(source_conditions[condition], "scores", SCORE_KEYS,
                                     condition, errors, pending)
            if result is not None:
                conditions[condition] = result
        source_generated = metrics.get("generated_real_map")
        if source_generated is None:
            pending.append("自由生成尚未完成（generated_real_map 缺失）")
        else:
            generated = extract_channel(source_generated, "generated", GENERATED_KEYS,
                                        "generated_real_map", errors, pending)
        if "generated_n" not in metrics or "generated_is_subset" not in metrics:
            pending.append("自由生成完成计数尚未写入")
        elif metrics["generated_n"] != 380 or metrics["generated_is_subset"] is not False:
            errors.append("自由生成必须覆盖完整 380 张 val，不能用子集代替")
        auc = nested(conditions, "real_map.image_auroc", None)
        if numeric(auc) and numeric(completion.get("best_val_auroc")):
            if abs(auc - completion["best_val_auroc"]) > 1e-10:
                errors.append("real_map AUROC 与 train_complete best_val_auroc 不同")
    main = {key: nested(conditions, "real_map." + key, None) for key in ("image_auroc", "image_ap")}
    main.update({"generated_" + key: None if generated is None else generated.get(key)
                 for key in ("macro_f1", "recall", "fpr", "invalid_rate")})
    status = "invalid" if errors else "ongoing" if pending else "complete"
    return {"seed": seed, "backend": backend, "directory": str(directory), "status": status,
            "files": files, "errors": errors, "pending": pending, "completion": completion,
            "contract": None if run is None else {key: nested(run, key, None) for key in PAIR_FIELDS},
            "main_metrics": main, "conditions": conditions, "generated_real_map": generated}


def compare_contracts(left, right, fields):
    checks = {}
    for field in fields:
        a = None if left is None else left.get(field)
        b = None if right is None else right.get(field)
        checks[field] = {"vicuna": a, "qwen": b,
                         "passed": None if a is None or b is None else a == b}
    return checks


def differences(vicuna, qwen):
    return {key: qwen.get(key) - vicuna.get(key)
            if numeric(vicuna.get(key)) and numeric(qwen.get(key)) else None
            for key in set(vicuna) | set(qwen)}


def moments(values):
    if len(values) != 3 or not all(numeric(value) for value in values):
        raise ValueError("Final statistics require exactly three finite paired seed values")
    return {"n": 3, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def summarize(root):
    runs = {backend: {str(seed): inspect_run(root, backend, seed) for seed in SEEDS} for backend in BACKENDS}
    pairs = []
    for seed in SEEDS:
        vicuna, qwen = (runs[backend][str(seed)] for backend in BACKENDS)
        checks = compare_contracts(vicuna["contract"], qwen["contract"], PAIR_FIELDS)
        mismatches = [field for field, check in checks.items() if check["passed"] is False]
        all_checks = all(check["passed"] is True for check in checks.values())
        status = ("invalid" if mismatches or "invalid" in {vicuna["status"], qwen["status"]}
                  else "complete" if all_checks and vicuna["status"] == qwen["status"] == "complete" else "ongoing")
        pairs.append({"seed": seed, "status": status, "contract_checks": checks,
                      "contract_mismatches": mismatches,
                      "paired_delta_qwen_minus_vicuna": differences(vicuna["main_metrics"], qwen["main_metrics"])
                      if status == "complete" else None,
                      "condition_delta_qwen_minus_vicuna": {
                          condition: differences(vicuna["conditions"][condition], qwen["conditions"][condition])
                          for condition in CONDITIONS} if status == "complete" else None})
    available = [run for backend in BACKENDS for run in runs[backend].values() if run["contract"] is not None]
    cross = []
    if available:
        anchor = available[0]
        for other in available[1:]:
            for field in CROSS_SEED_FIELDS:
                a, b = anchor["contract"].get(field), other["contract"].get(field)
                if a is not None and b is not None and a != b:
                    cross.append({"field": field, "left": "%s_s%d" % (anchor["backend"], anchor["seed"]),
                                  "right": "%s_s%d" % (other["backend"], other["seed"]), "left_value": a, "right_value": b})
    complete_seeds = [pair["seed"] for pair in pairs if pair["status"] == "complete"]
    status = "invalid" if cross or any(pair["status"] == "invalid" for pair in pairs) else (
        "complete" if len(complete_seeds) == 3 else "ongoing")
    aggregate = None
    if status == "complete":
        aggregate = {"by_backend": {}, "paired_delta_qwen_minus_vicuna": {}, "conditions": {}}
        for backend in BACKENDS:
            aggregate["by_backend"][backend] = {key: moments([runs[backend][str(seed)]["main_metrics"][key] for seed in SEEDS])
                                                   for key in MAIN_KEYS}
        aggregate["paired_delta_qwen_minus_vicuna"] = {
            key: moments([pair["paired_delta_qwen_minus_vicuna"][key] for pair in pairs]) for key in MAIN_KEYS}
        for condition in CONDITIONS:
            aggregate["conditions"][condition] = {
                backend: {key: moments([runs[backend][str(seed)]["conditions"][condition][key] for seed in SEEDS])
                          for key in SCORE_KEYS} for backend in BACKENDS}
            aggregate["conditions"][condition]["paired_delta_qwen_minus_vicuna"] = {
                key: moments([pair["condition_delta_qwen_minus_vicuna"][condition][key] for pair in pairs])
                for key in SCORE_KEYS}
    return {"schema_version": "phase1-paired-val-summary-v1", "root": str(root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "required_seeds": list(SEEDS), "completed_paired_seeds": complete_seeds,
            "input_scope": "only six run.json/train_complete.json/val_metrics.json result sets; no checkpoints/GPU/test metrics",
            "runs": runs, "pairs": pairs, "cross_seed_contract_mismatches": cross, "aggregate": aggregate,
            "aggregation_rule": "all three matched seed pairs complete; arithmetic mean and sample SD (ddof=1); no best-seed selection",
            "interpretation": ["有监督自定义 60/20/20 划分；train9985、val380，使用真实异常训练。",
                               "本阶段仅比较冻结语言骨干；两臂均不使用 Qwen 原生视觉，桥接模块重新初始化。",
                               "checkpoint 仅由 val AUROC 选择，均值覆盖预定 seeds42/43/44，不能挑选最好 seed。",
                               "本报告只汇总 val；历史456张 test 已用于旧开发，不能称为未查看确认集。",
                               "候选打分与自由生成分别报告；invalid 保留在生成指标分母，FPR 须结合 invalid 解读。",
                               "donor 条件使用 val 标签构造同类别相反标签诊断，不是可部署推理算法。",
                               "相同图片曝光与更新预算不代表相同 token 数、计算量或两骨干能力的单独因果效应。"]}


def fmt(value):
    return "—" if not numeric(value) else "%.6f" % value


def format_moment(value):
    return "%.6f ± %.6f" % (value["mean"], value["sample_sd"])


def markdown(summary):
    lines = ["# 第一阶段语言骨干配对结果", "", "状态：**%s**；已完成配对：%s / 3。" % (
        summary["status"], len(summary["completed_paired_seeds"])), "",
        "有监督 custom 60/20/20；train 9,985、val 380（正常172/异常208）。两臂冻结语言骨干，以同一 teacher cache 训练重新初始化的 PromptLearner 与 CLS 投影，均不使用 Qwen 原生视觉。",
        "", "全部数值按 0–1 报告。候选分数是首答案 token 的 logit(Yes)−logit(No)；生成指标独立计算。缺失值以 — 标示，ongoing 不计为 0。", "",
        "| Seed | 骨干 | 状态 | Best step | Real AUROC | Real AP | 生成 macro F1 | 生成 recall | 生成 FPR | 生成 invalid |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for backend in BACKENDS:
            run = summary["runs"][backend][str(seed)]
            values = [fmt(run["main_metrics"].get(key)) for key in MAIN_KEYS]
            lines.append("| %d | %s | %s | %s | %s |" % (
                seed, backend, run["status"], run["completion"].get("best_step", "—"), " | ".join(values)))
    lines.extend(["", "正在写入的运行可显示已有数值；只有状态 complete 且配置匹配的配对进入 Δ。",
                  "", "| Seed | 配对状态 | Δ AUROC | Δ AP | Δ 生成 F1 | Δ recall | Δ FPR | Δ invalid |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for pair in summary["pairs"]:
        delta = pair["paired_delta_qwen_minus_vicuna"] or {}
        lines.append("| %d | %s | %s |" % (pair["seed"], pair["status"], " | ".join(fmt(delta.get(key)) for key in MAIN_KEYS)))
    lines.extend(["", "Δ 均为 Qwen−Vicuna；FPR/invalid 越低越好。每个 seed 内核对初始化、decoder/cache/manifest、采样顺序哈希、可训练参数、步数、batch、accum、验证间隔和完整 val380，并核对其余共同训练配置。", ""])
    if summary["aggregate"] is None:
        lines.append("三组配对尚未全部完成并通过一致性检查，暂不计算 mean±sampleSD，也不挑选最好 seed。")
    else:
        aggregate = summary["aggregate"]
        lines.extend(["三 seed mean ± sampleSD（样本标准差，ddof=1；n=3）：", "",
                      "| 指标 | Vicuna | Qwen | 配对 Δ |", "|---|---:|---:|---:|"])
        for key in MAIN_KEYS:
            lines.append("| %s | %s | %s | %s |" % (
                key, format_moment(aggregate["by_backend"]["vicuna"][key]),
                format_moment(aggregate["by_backend"]["qwen"][key]),
                format_moment(aggregate["paired_delta_qwen_minus_vicuna"][key])))
    lines.extend(["", "各控制条件的候选分数：", "",
                  "| Seed | 骨干 | 条件 | AUROC | AP |", "|---|---|---|---:|---:|"])
    for seed in SEEDS:
        for backend in BACKENDS:
            run = summary["runs"][backend][str(seed)]
            for condition in CONDITIONS:
                values = run["conditions"].get(condition, {})
                lines.append("| %d | %s | %s | %s | %s |" % (seed, backend, condition,
                             fmt(values.get("image_auroc")), fmt(values.get("image_ap"))))
    if summary["aggregate"] is not None:
        lines.extend(["", "控制条件的三 seed mean ± sampleSD：", "",
                      "| 条件 | 指标 | Vicuna | Qwen | 配对 Δ |", "|---|---|---:|---:|---:|"])
        for condition in CONDITIONS:
            entry = summary["aggregate"]["conditions"][condition]
            for key in ("image_auroc", "image_ap"):
                lines.append("| %s | %s | %s | %s | %s |" % (condition, key,
                             format_moment(entry["vicuna"][key]), format_moment(entry["qwen"][key]),
                             format_moment(entry["paired_delta_qwen_minus_vicuna"][key])))
    lines.extend(["", "一致性与完成情况：", ""])
    for seed in SEEDS:
        for backend in BACKENDS:
            run = summary["runs"][backend][str(seed)]
            reasons = run["errors"] + run["pending"]
            lines.append("- %s_s%d：%s。%s" % (backend, seed, run["status"], "；".join(reasons) if reasons else "输入完整、协议检查通过。"))
    for pair in summary["pairs"]:
        if pair["contract_mismatches"]:
            lines.append("- seed%d 配对不一致：%s。" % (pair["seed"], "、".join(pair["contract_mismatches"])))
    for mismatch in summary["cross_seed_contract_mismatches"]:
        lines.append("- 跨 seed 不一致：%s 与 %s 的 %s。" % (mismatch["left"], mismatch["right"], mismatch["field"]))
    lines.extend(["", "解释边界：", ""])
    lines.extend("- " + text for text in summary["interpretation"])
    lines.extend(["", "仅读取六组 run.json、train_complete.json 和 val_metrics.json；文件路径与 SHA256 见 phase1_summary.json。未读取 checkpoint、预测文件、训练日志或 test 指标。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Directory containing lm_{vicuna,qwen}_s{42,43,44}")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error("Results root must already exist: " + str(root))
    summary = summarize(root)
    atomic_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", root / "phase1_summary.json")
    atomic_text(markdown(summary), root / "PHASE1_RESULTS.md")
    print(json.dumps({"status": summary["status"], "completed_paired_seeds": summary["completed_paired_seeds"],
                      "summary": str(root / "phase1_summary.json"), "report": str(root / "PHASE1_RESULTS.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
