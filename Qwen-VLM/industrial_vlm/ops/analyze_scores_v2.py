"""Audit eight completed boolean-score runs; summarize validation/regression.

Ordinary Python only; never loads torch, a model or CUDA. Confirmation source
IDs are read for isolation checks; confirmation images/predictions are not read.
All checks must pass before replacing the report. Optional --plots emits ROC/PR.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from industrial_vlm.evaluate import _atomic_json, _sha256  # noqa: E402
from industrial_vlm.score import (  # noqa: E402
    SCHEMA_VERSION, _canonical_hash, _read_scores, calibrate_normal_threshold,
    ranking_metrics, threshold_metrics,
)

ARMS = ("base", "region_region", "decision_decision", "region_decision")
ROLES = ("validation", "regression")
MANIFESTS = {"validation": "prepared/multitask_v2/val_decision_120.jsonl",
             "regression": "prepared/visa/test_balanced_120_decision.jsonl"}
BOUNDARY_AUDIT = "native full token IDs == prompt IDs + offset-verified continuation IDs"


def obj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same(actual, expected):
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same(actual[k], v) for k, v in expected.items())
    return actual == expected


def plot_curves(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    colors = ("#4c4c4c", "#0072b2", "#d55e00", "#009e73")
    names = ("Base", "Region + region", "Decision + decision", "Region + decision")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for role in ROLES:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), layout="constrained")
        for arm, color, name in zip(ARMS, colors, names):
            groups = defaultdict(list)
            for row in runs[arm][role]["records"]:
                groups[row["anomaly_log_odds"]].append(row["label"])
            tp = fp = 0
            fpr, recall, precision = [0.0], [0.0], [1.0]
            for threshold in sorted(groups, reverse=True):
                labels = groups[threshold]
                tp += sum(labels)
                fp += len(labels) - sum(labels)
                fpr.append(fp / 60)
                recall.append(tp / 60)
                precision.append(tp / (tp + fp))
            metric = runs[arm][role]["ranking"]
            axes[0].plot(fpr, recall, color=color, lw=1.8, label=f"{name} (AUC {metric['auroc']:.3f})")
            axes[1].step(recall, precision, where="pre", color=color, lw=1.8, label=f"{name} (AP {metric['average_precision']:.3f})")
        axes[0].plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", lw=1)
        axes[1].axhline(0.5, color="#aaaaaa", linestyle="--", lw=1)
        axes[0].set(xlabel="False positive rate", ylabel="Anomaly recall", title="ROC")
        axes[1].set(xlabel="Anomaly recall", ylabel="Anomaly precision", title="Precision-recall")
        for axis in axes:
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
            axis.legend(loc="lower right" if axis is axes[0] else "lower left", fontsize=8)
        fig.suptitle("Validation development set (120 images)" if role == "validation"
                     else "Previously evaluated test: development regression (120 images)", fontsize=12)
        for extension in ("pdf", "png", "svg"):
            path = output / f"scoring_{role}_curves.{extension}"
            fig.savefig(path, dpi=220)
            artifacts.append(str(path))
        plt.close(fig)
    return artifacts


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(report):
    text = ["# 连续异常分数：验证与开发回归", "",
            "8 个评分运行均通过审计，每组 120 张图像（60 正常、60 异常）。使用相同 decision prompt，"
            "只计算 JSON 布尔值 token 的条件似然，再在 true/false 两候选之间归一化。", "",
            "| 模型 | 验证 AUROC | 验证 AP | 旧测试开发回归 AUROC | 旧测试开发回归 AP |",
            "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        value = report["models"][arm]
        val, reg = value["validation"]["ranking"], value["regression"]["ranking"]
        text.append(f"| {arm} | {pct(val['auroc'])} | {pct(val['average_precision'])} | {pct(reg['auroc'])} | {pct(reg['average_precision'])} |")
    text += ["", "阈值只使用该模型验证集正常样本确定，随后固定应用于旧测试开发回归。", "",
             "| 模型 | 回归阈值规则 | 异常召回 | 正常误报率 | Macro F1 | 异常 F1 | TP / FP |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for label, key in (("验证正常样本阈值", "validation_calibrated_threshold"),
                           ("两候选等似然阈值", "default_equal_candidate_threshold")):
            metric = report["models"][arm]["regression"][key]
            text.append(f"| {arm} | {label} | {pct(metric['recall'])} | {pct(metric['fpr'])} | "
                        f"{pct(metric['macro_f1'])} | {pct(metric['anomaly_f1'])} | {metric['tp']} / {metric['fp']} |")
    text += ["", "| 模型 | 固定 log-odds 阈值 | 校准正常样本 | 校准误报 / 正常 | 校准经验误报率 |",
             "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        cal = report["models"][arm]["calibration"]
        text.append(f"| {arm} | {cal['threshold_log_odds']:.8g} | {cal['n_calibration_normals']} | "
                    f"{cal['actual_calibration_false_positives']} / {cal['n_calibration_normals']} | {pct(cal['empirical_calibration_fpr'])} |")
    text += ["",
             "阈值规则为 anomaly_log_odds > threshold，边界相等判正常；等似然阈值是 log-odds 0（归一分数 0.5）。"
             "排序使用 log-odds，避免 sigmoid 饱和制造额外相同分数。AUROC 按相同分数组处理；"
             "AP 为非插值 average precision，不采用 PR 梯形面积。", "",
             "60 张校准正常图像的误报率步长为 1/60 ≈ 1.67%；5% 目标最多允许 3 次校准误报，相同分数可能使实际误报更少。"
             "这只约束校准集经验值，不保证未来或跨域误报率 ≤5%，也不使分数成为校准置信度。", "",
             "验证集用于开发与阈值选择；旧 120 张测试图像已用于前轮评价，此处属于开发回归。"
             "这些结果不能替代最终独立确认。冻结 240 张确认集只读取 source ID 元数据核验隔离，未打开其图像或读取确认预测。", "",
             f"审计通过 {len(report['checks'])} 项，包括文件 SHA256、原始似然与派生分数、指标重算、模型与输入约定、"
             "校准来源及阈值、split/source/hash 隔离。完整输入 hash 清单见同名 JSON。"
             "基础权重完整 hash 采用评分时记录，汇总时核验跨运行一致性；本脚本不重复读取基础权重或数据图像。", ""]
    if report["figures"]:
        text += ["曲线分别标明验证开发集和旧测试开发回归，导出 PNG/PDF/SVG。", ""]
        text.extend(f"- [{Path(path).name}](figures/{Path(path).name})" for path in report["figures"])
        text.append("")
    return "\n".join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    directory, output = root / "evaluations/multitask_v2_scores", root / "reports/multitask_v2"
    checks, inputs = {}, {Path(__file__).resolve(), Path(sys.modules["industrial_vlm.score"].__file__).resolve()}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError("scoring audit failed: " + name)

    check("no_torch_import", "torch" not in sys.modules)
    manifests = {}
    for role, relative in MANIFESTS.items():
        path = root / relative
        inputs.add(path)
        manifests[role] = rows(path)
        check(role + ":manifest_unique_120", len(manifests[role]) == len({r["source_id"] for r in manifests[role]}) == 120)
    confirmation = root / "prepared/multitask_v2/confirmation_decision_240.jsonl"
    inputs.add(confirmation)
    confirmation_ids = {row["source_id"] for row in rows(confirmation)}
    check("confirmation_unique_240", len(confirmation_ids) == 240)
    provenance = root / "experiments/multitask_v2/scoring_source_manifest.json"
    inputs.add(provenance)
    frozen = obj(provenance)
    for relative, digest in frozen["files"].items():
        path = root / "experiments/multitask_v2/scoring_runtime_src" / relative
        inputs.add(path)
        check("frozen_source:" + relative, _sha256(path) == digest)
    check("exactly_eight_scoring_runs", {p.parent.name for p in directory.glob("*/meta.json")} ==
          {a + "_" + r for a in ARMS for r in ROLES})
    for path in (root / "evaluations").rglob("meta.json"):
        config = obj(path).get("config", {})
        check("no_confirmation_metadata:" + str(path.relative_to(root)),
              config.get("data_role") != "confirmation" and "confirmation" not in str(config.get("data", "")).lower())
        inputs.add(path)
    runs, shared_policy, token_policy = {}, None, None
    for arm in ARMS:
        runs[arm] = {}
        for role in ROLES:
            name = arm + "_" + role
            folder = directory / name
            inputs.update(folder / filename for filename in ("scores.jsonl", "meta.json", "metrics.json"))
            records, meta = _read_scores(folder)
            config, contract, metrics = meta["config"], meta["config"]["score_model_contract"], obj(folder / "metrics.json")
            check(name + ":complete_balanced_120", meta.get("n_completed") == config["n_expected"] == len(records) == 120
                  and sum(r["label"] for r in records) == 60)
            check(name + ":schema_role_budget", meta["schema_version"] == metrics["schema_version"] == SCHEMA_VERSION
                  and config["data_role"] == role and config["limit"] is None
                  and contract["max_pixels"] == 524288 and contract["max_length"] == 2048)
            check(name + ":no_confirmation_sources", not confirmation_ids.intersection(r["source_id"] for r in records))
            check(name + ":contract_hash", contract["contract_sha256"] ==
                  _canonical_hash({k: v for k, v in contract.items() if k != "contract_sha256"}))
            check(name + ":frozen_code", all(frozen["files"].get("industrial_vlm/" + filename) == digest
                  for filename, digest in contract["code_sha256"].items()))
            shared = {k: v for k, v in contract.items() if k not in {"adapter_files_sha256", "contract_sha256"}}
            if shared_policy is None:
                shared_policy = shared
            check(name + ":shared_base_and_policy", shared == shared_policy)
            check(name + ":adapter_policy", (contract["adapter_files_sha256"] is None) == (arm == "base")
                  and (config["adapter"] is None) == (arm == "base"))
            expected = {row["id"]: row for row in manifests[role]}
            check(name + ":manifest_ids", len(expected) == len({r["id"] for r in records}) == 120 and set(expected) == {r["id"] for r in records})
            check(name + ":manifest_hash", config["data_sha256"] == _sha256(root / MANIFESTS[role]))
            check(name + ":subset_hash", config["subset_sha256"] ==
                  _canonical_hash([(r["id"], r["source_id"], r["prompt"]) for r in manifests[role]]))
            check(name + ":image_hash_inventory", set(config["image_sha256_by_id"]) == set(expected))
            valid_sources = valid_hashes = valid_tokens = valid_budgets = True
            for record in records:
                source = expected[record["id"]]
                valid_sources &= all(record.get(k) == source[k] for k in ("source_id", "image", "category", "prompt"))
                valid_sources &= record["label"] == source["is_anomalous"] and record["prompt"] == contract["prompt"]
                match = re.search(r"_([a-f0-9]{64})$", record["source_id"])
                valid_hashes &= match is not None and match.group(1) == record["image_sha256"] == config["image_sha256_by_id"][record["id"]]
                signature = {k: record[k] for k in ("value_token_ids", "n_value_tokens", "common_json_prefix_token_ids", "common_suffix_token_ids")}
                if token_policy is None:
                    token_policy = signature
                valid_tokens &= signature == token_policy and record["token_boundary_audit"] == BOUNDARY_AUDIT
                valid_budgets &= all(record["n_value_tokens"][v] == len(record["value_token_ids"][v]) > 0 and
                    record["candidate_input_tokens"][v] == record["prompt_tokens"] + len(record["common_json_prefix_token_ids"])
                    + record["n_value_tokens"][v] + len(record["common_suffix_token_ids"]) <= contract["max_length"] for v in ("true", "false"))
            check(name + ":source_records", valid_sources)
            check(name + ":source_sha256", valid_hashes)
            check(name + ":token_boundaries", valid_tokens)
            check(name + ":token_budgets", valid_budgets)
            ranking, equal = ranking_metrics(records), threshold_metrics(records, 0.0)
            check(name + ":ranking_recomputed", same(metrics["overall"], ranking))
            check(name + ":equal_threshold_recomputed", same(metrics["default_equal_candidate_threshold"], equal))
            check(name + ":metrics_metadata", same(metrics["metadata"], {**config, "complete": True}))
            checks[name + ":raw_likelihood_score_and_file_hash"] = True
            runs[arm][role] = {"records": records, "config": config, "stored_metrics": metrics,
                               "ranking": ranking, "default_equal_candidate_threshold": equal}
        val, reg = runs[arm]["validation"], runs[arm]["regression"]
        check(arm + ":same_model_across_splits", val["config"]["score_model_contract"] == reg["config"]["score_model_contract"])
        check(arm + ":split_sources_disjoint", not {r["source_id"] for r in val["records"]}.intersection(r["source_id"] for r in reg["records"]))
        check(arm + ":split_hashes_disjoint", not {r["image_sha256"] for r in val["records"]}.intersection(r["image_sha256"] for r in reg["records"]))
        path = directory / (arm + "_threshold.json")
        inputs.add(path)
        cal = obj(path)
        expected = calibrate_normal_threshold(val["records"], 0.05)
        check(arm + ":threshold_recomputed", all(same(cal.get(k), v) for k, v in expected.items()))
        check(arm + ":calibration_schema", cal["schema_version"] == SCHEMA_VERSION and cal["kind"] == "validation_normal_threshold")
        check(arm + ":calibration_contract", cal["score_model_contract"] == val["config"]["score_model_contract"])
        check(arm + ":calibration_score_hash", cal["source_scores_sha256"] == _sha256(directory / (arm + "_validation") / "scores.jsonl"))
        check(arm + ":calibration_data_hash", cal["source_data_sha256"] == val["config"]["data_sha256"] and
              cal["source_subset_sha256"] == val["config"]["subset_sha256"])
        check(arm + ":calibration_sources", cal["calibration_source_ids"] == sorted(r["source_id"] for r in val["records"]))
        check(arm + ":frozen_threshold_hash", reg["config"]["calibration_sha256"] == _sha256(path) and val["config"]["calibration_sha256"] is None)
        check(arm + ":calibration_metrics", same(cal["calibration_set_metrics_not_independent_test"], threshold_metrics(val["records"], cal["threshold_log_odds"])))
        calibrated = threshold_metrics(reg["records"], cal["threshold_log_odds"])
        check(arm + ":regression_threshold_metrics", same(reg["stored_metrics"]["validation_calibrated_threshold"], calibrated))
        reg["validation_calibrated_threshold"], runs[arm]["calibration"] = calibrated, cal
        if arm != "base":
            for filename, digest in val["config"]["score_model_contract"]["adapter_files_sha256"].items():
                path = root / "runs/multitask_v2" / arm / "adapter" / filename
                inputs.add(path)
                check(arm + ":adapter_bytes:" + filename, _sha256(path) == digest)
    models = {a: {r: {k: v for k, v in runs[a][r].items() if k not in {"records", "config", "stored_metrics"}} for r in ROLES} for a in ARMS}
    for arm in ARMS:
        models[arm]["calibration"] = runs[arm]["calibration"]
    report = {"schema_version": "industrial-vlm-scoring-matrix-v2-v1", "status": "complete", "audit_status": "passed",
              "created_at": datetime.now(timezone.utc).isoformat(), "checks": checks, "models": models,
              "token_contract": token_policy, "shared_base_and_scoring_policy": shared_policy,
              "confirmation": {"source_ids_checked": 240, "images_opened": False, "predictions_read": False,
                               "source_overlap_with_eight_scoring_runs": 0,
                               "invocation_evidence": "no confirmation role/path in existing evaluation metadata"},
              "limits": ["Validation is used for development and threshold selection.",
                         "The old test subset is development regression, not final independent confirmation.",
                         "Only 60 validation normals; empirical FPR<=5% is not a future or cross-domain guarantee.",
                         "Two-option likelihood normalization is not calibrated confidence.",
                         "Recorded base hashes are checked across runs; base weights/images are not reread."],
              "input_sha256": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): _sha256(p) for p in sorted(inputs)},
              "figures": []}
    if args.plots:
        report["figures"] = plot_curves(runs, output / "figures")
    check("no_torch_import_after_analysis", "torch" not in sys.modules)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "scoring_matrix.json", report)
    temporary = output / "scoring_matrix.md.tmp"
    temporary.write_text(markdown(report), encoding="utf-8")
    temporary.replace(output / "scoring_matrix.md")
    print(json.dumps({"status": "complete", "audit_status": "passed", "n_checks": len(checks), "n_runs": 8,
                      "json": str(output / "scoring_matrix.json"), "markdown": str(output / "scoring_matrix.md"),
                      "confirmation_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
