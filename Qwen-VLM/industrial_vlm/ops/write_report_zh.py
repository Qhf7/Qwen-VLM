"""Render the Chinese pilot report only from completed, verified artifacts.

Run after compare, plot_results and verify_completion. This command uses only
the Python standard library and the project's dependency-free metric reader;
it never loads a model or starts CUDA. Missing/stale inputs fail before writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("quality_region", "random_region", "quality_decision")
PANELS = (
    ("VisA 区域任务", ("base_region", "quality_region", "random_region"), 120, True),
    ("VisA 共同判断提示", ("base_decision", "quality_region_decision_prompt", "quality_decision"), 120, False),
    ("MVTec 外部区域任务 pilot", ("external_base_region", "external_quality_region"), 150, True),
)
PAIRS = (
    ("quality_region", "base_region", "域内区域 SFT 相对底座"),
    ("quality_region", "random_region", "面积覆盖采样相对随机采样"),
    ("quality_region_decision_prompt", "quality_decision", "区域训练相对判断训练（共同判断提示）"),
    ("external_quality_region", "external_base_region", "外部 MVTec 区域迁移"),
)
NAMES = {
    "base_region": "底座 · 区域提示",
    "quality_region": "面积覆盖 SFT · 区域提示",
    "random_region": "随机采样 SFT · 区域提示",
    "base_decision": "底座 · 判断提示",
    "quality_region_decision_prompt": "区域 SFT · 判断提示",
    "quality_decision": "判断 SFT · 判断提示",
    "external_base_region": "底座 · 外部区域提示",
    "external_quality_region": "面积覆盖 SFT · 外部区域提示",
    "overfit32_training_diagnostic": "32 图训练拟合诊断",
}
METRICS = {
    "macro_f1": "二类 macro F1",
    "balanced_accuracy": "平衡准确率",
    "fpr": "正常 FPR",
    "fnr": "异常 FNR",
    "union_box_recall_iou_0.5": "Union-box R@IoU 0.5",
}
AREA_GROUPS = (
    ("anomalous_lt_1pct", "mask 面积 <1%"),
    ("anomalous_1_to_5pct", "mask 面积 1%–<5%"),
    ("anomalous_ge_5pct", "mask 面积 ≥5%"),
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"Expected an object: {path}:{index}")
            result.append(value)
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, context: str, *, rate: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Missing/nonfinite number: {context}")
    require(not rate or 0 <= value <= 1, f"Rate outside [0,1]: {context}")
    return float(value)


def same_number(left: Any, right: Any, context: str) -> None:
    if left is None or right is None:
        require(left is right, f"Mismatched null value: {context}")
    else:
        require(math.isclose(number(left, context), number(right, context), abs_tol=1e-12, rel_tol=1e-12),
                f"Mismatched observed values: {context}")


def load_bundle(root: Path) -> dict[str, Any]:
    """Validate all prerequisites before producing any report text."""
    sys.path.insert(0, str(root / "src"))
    from industrial_vlm.compare import _load_run
    from industrial_vlm.metrics import compute_metrics

    verification = read(root / "ops/completion_verification.json")
    require(verification.get("status") == "passed" and verification.get("checks"),
            "Run ops/verify_completion.py successfully before writing the report")
    require(read(root / "pipeline_state.json").get("status") == "complete", "Pipeline is not complete")
    visa = read(root / "prepared/visa/audit_report.json")
    external = read(root / "prepared/mvtec/audit_report.json")
    prepared = read(root / "ops/prepared_validation.json")
    split = read(root / "prepared/visa/split.json")
    isolation = read(root / "ops/cross_dataset_isolation.json")
    require(prepared.get("status") == "passed", "Prepared-data validation did not pass")
    require(visa["valid_unique_images"] == prepared["n_sources"] == 10821, "Expected the fixed 10,821-image VisA inventory")
    split_sets = {key: set(split[key]) for key in ("train", "val", "test")}
    require([len(split_sets[k]) for k in ("train", "val", "test")] == [7575, 1624, 1622], "Unexpected VisA split sizes")
    require(not (split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or
                 split_sets["val"] & split_sets["test"]), "Overlapping VisA source splits")
    require(sum(len(v) for v in split_sets.values()) == visa["valid_unique_images"], "Split inventory differs from VisA audit")
    require(external["external_test_n"] == 150 and isolation["external_sources"] == 150 and
            isolation["visa_sources"] == visa["valid_unique_images"] and isolation["image_sha256_overlap"] == [],
            "External size/isolation audit differs from the fixed pilot")
    sampling = visa["sampling"]
    require(sampling["normal_source_ids_equal"] and sampling["category_label_sequence_equal"], "Sampling controls are not matched")
    require(all(v.get("computed_from") == "train_anomalies_only" for v in sampling["area_thresholds"].values()),
            "Area thresholds were not computed from training anomalies only")

    training, training_rows = {}, {}
    for name in ARMS:
        directory = root / "runs" / name
        summary, config = read(directory / "summary.json"), read(directory / "run_config.json")
        audit, first = read(directory / "parameter_audit.json"), read(directory / "first_sample_audit.json")
        events = rows(directory / "metrics.jsonl")
        train_events = [e for e in events if e.get("event") == "train"]
        val_events = [e for e in events if e.get("event") == "validation"]
        load_events = [e for e in events if e.get("event") == "loaded"]
        require(summary.get("status") == "complete" and summary["optimizer_steps"] == 96 and summary["microsteps"] == 768,
                f"{name}: expected completed 96-update/768-exposure training")
        require(verification["training"].get(name) == summary, f"{name}: completion verification is stale")
        require(summary["reload_check"].get("passed") is True, f"{name}: fresh-base adapter reload check failed")
        require(config["seed"] == 42 and config["max_steps"] == 96 and config["gradient_accumulation_steps"] == 8 and
                config["micro_batch_size"] == 1 and config["max_pixels"] == 524288 and config["max_length"] == 2048,
                f"{name}: changed training budget")
        require(config["lora_r"] == 16 and config["lora_alpha"] == 32 and config["val_max_samples"] == 24,
                f"{name}: changed LoRA/validation settings")
        require(summary["trainable_parameters"] == audit["trainable_parameters"] == 15335424,
                f"{name}: unexpected trainable parameter count")
        require(sum(p["numel"] for p in audit["trainable_tensors"]) == audit["trainable_parameters"],
                f"{name}: trainable tensor counts disagree")
        require(all(".language_model.layers." in p["name"] and
                    re.search(r"\.self_attn\.[qkvo]_proj\.lora_[AB]\.", p["name"])
                    for p in audit["trainable_tensors"]), f"{name}: trainable tensors outside language-attention LoRA")
        require(audit["visual_dtype"] == "torch.bfloat16" and audit["gradient_checkpointing"] is True and
                audit["quantization"] == "NF4, double quantization, bf16 compute; language linear layers only",
                f"{name}: changed precision/quantization policy")
        require(audit["quantized_modules"] and all(".language_model.layers." in p for p in audit["quantized_modules"]),
                f"{name}: unexpected quantized modules")
        require({"pixel_values", "image_grid_thw", "input_ids", "labels"} <= set(first["native_fields"]) and
                first["masked_prefix_tokens"] > 0 and first["target_tokens"] > 0,
                f"{name}: missing native image/assistant-target audit")
        require([e["optimizer_step"] for e in train_events] == list(range(1, 97)), f"{name}: incomplete training event history")
        require([e["optimizer_step"] for e in val_events] == [32, 64, 96] and
                all(e["validation_samples"] == 24 for e in val_events), f"{name}: unexpected validation history")
        require(len(load_events) == 1 and load_events[0]["train_samples"] == 768 and load_events[0]["val_samples"] == 120,
                f"{name}: mixed/incomplete model-load history")
        for total_key, event_key in (("total_target_tokens", "target_tokens"), ("total_input_tokens", "input_tokens")):
            require(sum(e[event_key] for e in train_events) == summary[total_key], f"{name}: {total_key} disagrees with events")
        for key in ("last_train_loss", "train_wall_seconds_including_validation", "peak_allocated_gib", "peak_reserved_gib"):
            number(summary[key], f"{name}:{key}")
        for e in train_events:
            number(e["loss"], f"{name}: train loss")
            number(e["seconds"], f"{name}: update time")
        manifest = root / "prepared/visa" / f"train_{name}.jsonl"
        require(sha(manifest) == config["train_sha256"], f"{name}: training manifest hash differs")
        selected = rows(manifest)
        sources = {r["source_id"] for r in selected}
        require(len(selected) == len(sources) == 768 and sources <= split_sets["train"], f"{name}: wrong/overlapping training sources")
        val_name = "val_balanced_120_decision.jsonl" if name == "quality_decision" else "val_balanced_120.jsonl"
        require(sha(root / "prepared/visa" / val_name) == config["val_sha256"], f"{name}: validation manifest hash differs")
        training_rows[name] = selected
        training[name] = {"summary": summary, "config": config, "audit": audit, "first": first,
                          "last_validation": val_events[-1], "load_seconds": load_events[0]["load_seconds"],
                          "update_seconds": sum(e["seconds"] for e in train_events)}
    for key in ("model", "seed", "max_steps", "max_pixels", "max_length", "micro_batch_size",
                "gradient_accumulation_steps", "learning_rate", "weight_decay", "max_grad_norm",
                "warmup_steps", "save_steps", "eval_steps", "val_max_samples", "lora_r", "lora_alpha", "lora_dropout"):
        require(len({json.dumps(training[name]["config"][key], sort_keys=True) for name in ARMS}) == 1,
                f"Training arms have unmatched {key}")
    a, b, c = (training_rows[name] for name in ARMS)
    normal_ids = lambda data: {r["source_id"] for r in data if not r["is_anomalous"]}
    require(normal_ids(a) == normal_ids(b), "Coverage and random normal sources differ")
    require([(r["category"], r["is_anomalous"]) for r in a] == [(r["category"], r["is_anomalous"]) for r in b],
            "Coverage and random category/label order differs")
    require([r["source_id"] for r in a] == [r["source_id"] for r in c], "Region and decision source order differs")
    overlap = len({r["source_id"] for r in a} & {r["source_id"] for r in b})
    require(overlap == sampling["quality_random_overlap"], "Sampling overlap audit differs")

    evaluations = {}
    all_train_sources = {r["source_id"] for selected in training_rows.values() for r in selected}
    all_names = [name for _, names, _, _ in PANELS for name in names] + ["overfit32_training_diagnostic"]
    for name in all_names:
        run, error = _load_run(root / "evaluations", name)
        require(run is not None, f"{name}: {error}")
        directory = root / "evaluations" / name
        metrics, meta = read(directory / "metrics.json"), read(directory / "meta.json")
        recomputed = compute_metrics(run["rows"])
        require(all(recomputed[key] == metrics[key] for key in
                    ("overall", "by_category", "by_anomaly_area", "area_stratification_sources")), f"{name}: stale metrics")
        config, overall = run["config"], metrics["overall"]
        expected_n = 32 if name == "overfit32_training_diagnostic" else 150 if name.startswith("external_") else 120
        require(overall["n"] == overall["n_sources"] == expected_n, f"{name}: unexpected source count")
        if name != "overfit32_training_diagnostic":
            groups: dict[tuple[str, bool], int] = {}
            for row in run["rows"]:
                key = (row["category"], row["label"])
                groups[key] = groups.get(key, 0) + 1
            expected_categories = 15 if name.startswith("external_") else 12
            require(len(groups) == expected_categories * 2 and set(groups.values()) == {5},
                    f"{name}: expected five sources per category/label stratum")
        require(config["max_pixels"] == 524288 and config["max_length"] == 2048 and config["max_new_tokens"] == 128 and
                config["generation"] == {"do_sample": False, "num_beams": 1}, f"{name}: changed evaluation budget")
        for key in ("format_valid_rate", "macro_f1", "balanced_accuracy", "anomaly_recall", "fpr", "fnr", "normal_failure_rate"):
            number(overall[key], f"{name}:{key}", rate=True)
        require(number(overall["mean_latency_seconds"], f"{name}: latency") >= 0 and
                number(overall["max_peak_memory_allocated_bytes"], f"{name}: memory") > 0, f"{name}: invalid cost metrics")
        require(all(number(r["input_tokens"], f"{name}: input tokens") > 0 and
                    number(r["generated_tokens"], f"{name}: generated tokens") >= 0 for r in run["rows"]),
                f"{name}: invalid token counts")
        sources = {r["source_id"] for r in run["rows"]}
        if name != "overfit32_training_diagnostic":
            require(not sources & all_train_sources, f"{name}: SFT/test source overlap")
            require(verification["evaluations"][name]["predictions_sha256"] == sha(directory / "predictions.jsonl"),
                    f"{name}: predictions changed after completion verification")
            if not name.startswith("external_"):
                require(sources <= split_sets["test"], f"{name}: sources outside VisA test split")
        else:
            diagnostic_sources = {r["source_id"] for r in rows(root / "prepared/visa/overfit32.jsonl")}
            require(sources <= split_sets["train"] and sources == diagnostic_sources,
                    "Training-fit diagnostic differs from its independent train-only source manifest")
        evaluations[name] = {**run, "metrics": metrics, "meta": meta}
    matched_keys = ("model", "base_config_sha256", "data_sha256", "subset_sha256", "max_pixels", "max_length",
                    "max_new_tokens", "generation", "quantization", "limit")
    for title, names, _, is_region in PANELS:
        for key in matched_keys:
            require(len({json.dumps(evaluations[name]["config"].get(key), sort_keys=True) for name in names}) == 1,
                    f"Unmatched {key} in {title}")
        for name in names:
            run = evaluations[name]
            require(run["config"]["n_region_task"] == (run["metrics"]["overall"]["n"] if is_region else 0),
                    f"{name}: unexpected task type")
            require(run["metrics"]["overall"]["union_box_recall_iou_0.5"] is not None if is_region else
                    run["metrics"]["overall"]["union_box_recall_iou_0.5"] is None, f"{name}: localization applicability differs")
    for key, _ in AREA_GROUPS:
        left = evaluations["base_region"]["metrics"]["by_anomaly_area"][key]
        right = evaluations["quality_region"]["metrics"]["by_anomaly_area"][key]
        require(left["n"] == right["n"] and left["n"] > 0, f"Missing/mismatched area stratum: {key}")
        require(left["n_anomalous"] == left["n"] == right["n_anomalous"], f"Non-anomalous area stratum: {key}")
        for metric in ("anomaly_recall", "union_box_recall_iou_0.5"):
            number(left[metric], f"base_region:{key}:{metric}", rate=True)
            number(right[metric], f"quality_region:{key}:{metric}", rate=True)
    require(sum(evaluations["base_region"]["metrics"]["by_anomaly_area"][key]["n"] for key, _ in AREA_GROUPS) ==
            evaluations["base_region"]["metrics"]["overall"]["n_anomalous"], "Area groups do not cover all primary-test anomalies")
    require(evaluations["base_region"]["metrics"]["area_stratification_sources"] == ["mask"] and
            evaluations["quality_region"]["metrics"]["area_stratification_sources"] == ["mask"], "Area groups are not based on original masks")

    paired = read(root / "reports/paired_difference.json")
    require(paired.get("status") == "complete" and paired.get("missing") == [] and paired["n_resamples"] == 2000,
            "Expected all four completed 2,000-resample paired comparisons")
    comparisons = {(item["left"], item["right"]): item for item in paired["comparisons"]}
    require(set(comparisons) == {(a, b) for a, b, _ in PAIRS} and len(paired["comparisons"]) == 4, "Unexpected paired comparisons")
    for left, right, _ in PAIRS:
        item = comparisons[(left, right)]
        require(item["n_resamples"] == 2000 and item["seed"] == paired["seed"], "Paired bootstrap budget/seed differs")
        require(item["n_rows"] == item["n_sources"] == evaluations[left]["metrics"]["overall"]["n"], "Paired source counts differ")
        require(set(item["metrics"]) == set(METRICS), "Paired metric set differs")
        for side, name in (("left", left), ("right", right)):
            require(item[f"{side}_adapter_files_sha256"] == evaluations[name]["config"]["adapter_files_sha256"],
                    f"{left}/{right}: stale paired adapter identity")
        for key, estimate in item["metrics"].items():
            x, y = (evaluations[name]["metrics"]["overall"][key] for name in (left, right))
            same_number(estimate["left"], x, f"{left}:{key}")
            same_number(estimate["right"], y, f"{right}:{key}")
            same_number(estimate["difference"], x - y if x is not None and y is not None else None, f"paired {left}/{right}:{key}")
            if x is None or y is None:
                require(estimate["ci95_low"] is None and estimate["ci95_high"] is None and estimate["n_defined_resamples"] == 0,
                        "Decision-only localization must remain undefined")
            else:
                low, high = number(estimate["ci95_low"], key), number(estimate["ci95_high"], key)
                require(-1 <= low <= high <= 1 and 0 < estimate["n_defined_resamples"] <= 2000, f"Invalid paired interval: {key}")
    figure_data = read(root / "reports/figures/figure_data.json")
    require(figure_data.get("warnings") == [], "Result figures contain incomplete/unmatched input warnings")
    for name in ARMS:
        require(figure_data["training"][name]["sha256"] == sha(root / "runs" / name / "metrics.jsonl"), "Training figure input is stale")
    for _, names, _, _ in PANELS:
        for name in names:
            require(figure_data["evaluations"][name]["sha256"] == sha(root / "evaluations" / name / "metrics.json"),
                    f"Evaluation figure input is stale: {name}")
    for name in ("method", "training_curves", "evaluation_bars", "latency_macro_f1"):
        for suffix in ("png", "pdf"):
            require((root / "reports/figures" / f"{name}.{suffix}").is_file(), f"Missing figure {name}.{suffix}")
    require((root / "reports/matrix.md").is_file(), "Missing completed experiment matrix")
    runtime, cleanup = read(root / "ops/runtime_audit.json"), read(root / "ops/cleanup_manifest.json")
    require(cleanup.get("completed_unix") and cleanup["delete"], "Cleanup manifest is not complete")
    removed_bytes = sum(int(item["bytes"]) for item in cleanup["delete"])
    require(removed_bytes > 0 and runtime["bf16_supported"] is True, "Missing cleanup/BF16 runtime evidence")
    provenance = read(root / "ops/mvtec_subset_provenance.json")
    require(provenance["n_images"] == external["external_test_n"] and provenance["per_category_per_label"] == 5 and
            provenance["revision"] and provenance["seed"] == 31415, "External provenance differs from the fixed pilot")
    return {"root": root, "verification": verification, "visa": visa, "external": external, "prepared": prepared,
            "runtime": runtime, "cleanup": cleanup, "removed_bytes": removed_bytes, "provenance": provenance,
            "training": training, "training_rows": training_rows, "evaluations": evaluations,
            "paired": paired, "comparisons": comparisons, "figure_data": figure_data,
            "normal_sources": len(normal_ids(a)), "overlap": overlap}


def pct(value: Any) -> str:
    return f"{number(value, 'percentage', rate=True) * 100:.2f}"


def signed(value: Any) -> str:
    return f"{number(value, 'difference') * 100:+.2f}"


def interval(estimate: dict[str, Any]) -> str:
    return f"[{signed(estimate['ci95_low'])}, {signed(estimate['ci95_high'])}]"


def interpretation(estimate: dict[str, Any]) -> str:
    delta, low, high = (estimate[key] for key in ("difference", "ci95_low", "ci95_high"))
    preferred = estimate.get("preferred_direction", "positive")
    if delta == 0:
        point = "点估计持平"
    else:
        point = f"点估计{'上升' if delta > 0 else '下降'}"
        if abs(delta) < 0.02:
            point += "，幅度不足 2 个百分点"
    if low <= 0 <= high:
        return point + "；95% 区间覆盖 0，本次 pilot 未确定差异方向"
    favorable = low > 0 if preferred == "positive" else high < 0
    return point + ("；区间支持本次 pilot 上的正向差异" if favorable else "；区间支持本次 pilot 上的不利差异")


def render(bundle: dict[str, Any], output: Path) -> str:
    root, ev, tr = bundle["root"], bundle["evaluations"], bundle["training"]
    visa, ext, runtime = bundle["visa"], bundle["external"], bundle["runtime"]
    qcfg = tr["quality_region"]["config"]

    def link(relative: str, label: str | None = None) -> str:
        target = os.path.relpath(root / relative, output.parent).replace(os.sep, "/")
        return f"[{label or relative}]({target})"

    def overall(name: str) -> dict:
        return ev[name]["metrics"]["overall"]

    lines = ["# 工业异常检测 VLM 领域后训练：实验报告", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}。本报告读取已完成的 9 项评估、3 组训练、数据审计、配对差值与完成验证产物；不补齐缺失指标。", "",
             "## 实测结论", ""]
    for left, right, label in PAIRS:
        item = bundle["comparisons"][(left, right)]
        estimate = item["metrics"]["macro_f1"]
        text = (f"- **{label}**：二类 macro F1 从 {pct(estimate['right'])}% 到 {pct(estimate['left'])}%，"
                f"差值 {signed(estimate['difference'])} pp，95% CI {interval(estimate)} pp；{interpretation(estimate)}。")
        box = item["metrics"]["union_box_recall_iou_0.5"]
        if box["difference"] is not None:
            text += (f" Union-box R@0.5 差值 {signed(box['difference'])} pp，95% CI {interval(box)} pp；"
                     f"{interpretation(box)}。")
        errors = []
        for key, label_name in (("fpr", "FPR"), ("fnr", "FNR")):
            estimate_error = item["metrics"][key]
            errors.append(f"{label_name} {signed(estimate_error['difference'])} pp（95% CI {interval(estimate_error)} pp）")
        text += " 误报/漏报差值：" + "，".join(errors) + "；两项均以负值为改善。"
        lines.append(text)
    lines += ["", "上述差异均以左侧方法减右侧方法计算。正向区间仅支持这轮已训练模型在固定 pilot 上的对比，不能外推为跨随机种子、跨场景的稳定优势或 SOTA。", "",
              "## 部署、环境与模型实现", "",
              f"实验使用单张 **{runtime['gpu']}**，显存 {runtime['gpu_total_bytes'] / 2**30:.2f} GiB；"
              f"Python {runtime['python']}，CUDA {runtime['torch_cuda']}，驱动 {runtime['driver']}，BF16 支持状态为 `{runtime['bf16_supported']}`。", "",
              "| 组件 | 实际版本 |", "|---|---|"]
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "Pillow", "safetensors"):
        lines.append(f"| {name} | {runtime['packages'][name]} |")
    removed = bundle["removed_bytes"]
    lines += ["", f"部署时按已完成清单清理 {len(bundle['cleanup']['delete'])} 个服务器路径，"
              f"清单字节数合计 **{removed:,} bytes = {removed / 2**30:.2f} GiB（{removed / 1e9:.2f} GB）**；"
              "该值是清单中的路径大小之和，文件系统 `df` 前后结果一并留档。保留 Qwen3-VL-8B-Instruct 权重及运行虚拟环境。"
              f"证据：{link('ops/cleanup_manifest.json')}、{link('ops/runtime_audit.json')}、{link('ops/environment_full_freeze.txt')}。", "",
              "输入为一张完整原始图像及统一短英文指令，沿用 Qwen3-VL 原生 processor、视觉 token、image grid 与位置编码。"
              "视觉编码器、merger 和 DeepStack 保持冻结 BF16；语言 Linear 使用 NF4 双重量化与 BF16 计算；"
              "仅语言 attention 的 q/k/v/o 投影加入 LoRA。", "",
              f"LoRA r={qcfg['lora_r']}、alpha={qcfg['lora_alpha']}、dropout={qcfg['lora_dropout']}；"
              f"每组可训练参数 **{tr['quality_region']['summary']['trainable_parameters']:,}**，"
              f"参数审计记录 {len(tr['quality_region']['audit']['quantized_modules'])} 个量化模块、"
              f"{len(tr['quality_region']['audit']['lora_target_modules'])} 个 LoRA 目标模块。"
              "使用梯度 checkpointing、单卡单进程 AdamW 与 assistant-only causal CE；图像、用户指令及前缀标签均屏蔽，"
              "累计梯度按实际监督 token 总数归一化。", "",
              f"原生输入与监督位置见 {link('runs/quality_region/first_sample_audit.json')}，"
              f"参数边界见 {link('runs/quality_region/parameter_audit.json')}，实现见 {link('src/industrial_vlm/modeling.py')}、"
              f"{link('src/industrial_vlm/train.py')}。", "",
              f"![方法与实验流程]({os.path.relpath(root / 'reports/figures/method.png', output.parent)})", "",
              "## 数据、标注和受控对照", "",
              f"VisA 官方压缩包产生 **{visa['valid_unique_images']:,}** 张有效唯一图像；"
              f"拒绝 {visa['rejected_rows']} 条、移除重复图 {visa['duplicates_removed']} 张。"
              f"按类别×正常/异常、源图 SHA256 隔离的 70/15/15 自定义有监督划分为 "
              f"**{visa['splits']['train']['n']:,}/{visa['splits']['val']['n']:,}/{visa['splits']['test']['n']:,}**。"
              "训练分区真实异常及 mask 用于构造训练目标；验证、测试也有真实异常与 mask，供验证和评分使用。"
              "该信息条件不同于 AnomalyGPT 原始 normal-only 合成异常训练协议，论文分数不宜直接并表比较。", "",
              "VisA 原始 mask 的 0 为背景、所有非零类别索引为异常前景；MVTec mask 按二值语义验证。"
              "每张 mask 转为覆盖全部前景的一个外接矩形，xyxy 坐标以全图归一化至整数 [0,1000]。"
              "同时检查图像解码、mask 尺寸、EXIF、标签与框边界；先划分源图，再构造指令。推理仅接收图像与 prompt，"
              "不读取答案、类别、标签、mask 或 GT 框。", "",
              "| 训练组 | 实际源图数 | 正常/异常 | 抽样与监督 |", "|---|---:|---:|---|"]
    for name in ARMS:
        selected = bundle["training_rows"][name]
        normal = sum(not r["is_anomalous"] for r in selected)
        description = {"quality_region": "训练异常 mask 面积三分位覆盖；判断＋区域框",
                       "random_region": "同类别×标签配额内均匀随机；判断＋区域框",
                       "quality_decision": "与面积覆盖区域组相同源图及顺序；只监督判断"}[name]
        lines.append(f"| {name} | {len(selected)} | {normal}/{len(selected)-normal} | {description} |")
    lines += ["", f"面积覆盖与随机组固定全部 **{bundle['normal_sources']}** 张正常源图、类别×标签配额、"
              f"每个训练位置的类别标签序列及下游随机种子；两组共重叠 **{bundle['overlap']}** 张源图。"
              "面积阈值只由训练异常计算。这里检验的是 **area coverage（缺陷面积覆盖采样）**；"
              "两组标签均有效，不应把该对照解释成标注清洗或推理链质量提升。", "",
              f"主测试固定为 **{overall('base_region')['n']}** 张 VisA 留出图（"
              f"{overall('base_region')['n_normal']} 正常＋{overall('base_region')['n_anomalous']} 异常），每类别每标签 5 张；"
              "验证另有独立120张均衡子集，完整 holdout manifest 仍保留。共同判断提示面板与区域面板使用相同源图；"
              "前者固定判断 prompt，比较底座、区域训练适配器、判断训练适配器。", "",
              f"外部评估是 **{ext['external_test_n']}** 张 MVTec 官方 test 图像的公开镜像选择，15 类×2 标签×"
              f"{bundle['provenance']['per_category_per_label']} 张；抽样 seed={bundle['provenance']['seed']}，"
              f"镜像 `{bundle['provenance']['repo']}`，revision `{bundle['provenance']['revision']}`。"
              "抽样在模型评测前固定，不按图像内容或模型结果选图；这只是外部 pilot，并非完整1,725张官方测试。"
              f"VisA 与外部图像原始字节 SHA256 无交集。证据：{link('prepared/visa/audit_report.json')}、"
              f"{link('prepared/mvtec/audit_report.json')}、{link('ops/mvtec_subset_provenance.json')}、"
              f"{link('ops/cross_dataset_isolation.json')}、{link('DATA_PROTOCOL.md')}。", "",
              "## 训练预算、损失与成本", "",
              f"每组 micro-batch={qcfg['micro_batch_size']}、梯度累计={qcfg['gradient_accumulation_steps']}，"
              "96 次 optimizer update 共768次源图曝光，每张训练图恰好一遍。"
              f"学习率 {qcfg['learning_rate']}，warmup {qcfg['warmup_steps']} 步，weight decay={qcfg['weight_decay']}，"
              f"梯度范数裁剪 {qcfg['max_grad_norm']}；max_pixels={qcfg['max_pixels']:,}、max_length={qcfg['max_length']:,}。"
              "固定使用最后96步适配器，验证损失只作审计，不据测试选择 checkpoint。"
              "每32步的 teacher-forced assistant CE 仅计算验证 manifest 前24条，不是完整120张验证评分。", "",
              "| 训练组 | 更新/曝光 | 输入 token 总数 | 监督 token 总数 | 训练循环秒 | 加载秒 | 峰值 allocated/reserved GiB | 最后训练 CE | 最后验证 CE（n=24） |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ARMS:
        t, s = tr[name], tr[name]["summary"]
        lines.append(f"| {name} | {s['optimizer_steps']}/{s['microsteps']} | {s['total_input_tokens']:,} | "
                     f"{s['total_target_tokens']:,} | {s['train_wall_seconds_including_validation']:.2f} | {t['load_seconds']:.2f} | "
                     f"{s['peak_allocated_gib']:.2f}/{s['peak_reserved_gib']:.2f} | {s['last_train_loss']:.6f} | "
                     f"{t['last_validation']['validation_loss']:.6f} |")
    total_seconds = sum(t["summary"]["train_wall_seconds_including_validation"] for t in tr.values())
    lines += ["", f"三组训练循环合计 **{total_seconds:.2f} 秒（{total_seconds / 60:.2f} 分钟）**。"
              "循环时间包含验证及中间 checkpoint 保存，排除初始模型加载、最终保存与重载核验；"
              "输入 token 计数含视觉 token 和文本目标，监督 token 仅计 assistant 目标（含结束标记）。"
              "相同更新数和图像曝光不等于相同监督 token 或 FLOP 预算；判断目标较短，因此不能用不同任务 CE 高低证明方法优劣。"
              "本次未记录 GPU 小时单价、电费或 FLOP 测量，不推算货币成本。", "",
              "三组保存后的适配器均以新底座重载，第一样本 prompt 最后一个位置 logits 校验通过；"
              f"最大绝对差分别为 {', '.join(f'{name}={tr[name]["summary"]["reload_check"]["max_absolute_logit_difference"]:.6g}' for name in ARMS)}。"
              "该检查范围是单个提示位置，不能替代任意输入下的完整数值等价证明。", "",
              f"![原始训练与验证损失]({os.path.relpath(root / 'reports/figures/training_curves.png', output.parent)})", "",
              "## 留出评估：三个匹配面板", "",
              "表中比率单位均为%。二类 macro F1 为正常、异常两类 F1 的平均，不是12个产品类别的宏平均。"
              "格式失败保留在分母：异常无效输出计入漏报，正常无效输出降低正常类 F1 与 recall；"
              "FPR 仅统计明确异常预测/全部正常图，正常失败率额外计入格式失败。"
              "Union-box recall 要求异常判断正确且外接矩形 IoU 达阈值，分母包含所有异常区域任务图；"
              "格式失败、空框和错误判断均为定位失败。", ""]
    for title, names, _, is_region in PANELS:
        lines += [f"### {title}（n={overall(names[0])['n']}）", "",
                  "| 方法 | n | 格式有效 | 二类 macro F1 | 平衡准确率 | 异常召回 | 正常 FPR | 异常 FNR | 正常失败率 | Union R@0.5 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in names:
            m = overall(name)
            values = [pct(m[k]) for k in ("format_valid_rate", "macro_f1", "balanced_accuracy", "anomaly_recall", "fpr", "fnr", "normal_failure_rate")]
            box = pct(m["union_box_recall_iou_0.5"]) if is_region else "不适用"
            lines.append(f"| {NAMES[name]} | {m['n']} | " + " | ".join(values) + f" | {box} |")
        lines.append("")
    lines += ["共同判断提示比较控制了测试图片、prompt 与解码任务；区域训练和判断训练仍有目标格式及监督长度差别，"
              "应解释为加入区域任务的整体效果。判断任务不要求框，定位指标不适用。", "",
              f"![留出评估结果]({os.path.relpath(root / 'reports/figures/evaluation_bars.png', output.parent)})", "",
              "### VisA 主测试异常面积分层", "",
              "面积来自原始前景 mask 像素占比，以下仅统计异常图；base 与面积覆盖区域 SFT 使用相同测试源图。", "",
              "| 异常面积组 | n | 底座异常召回 % | SFT 异常召回 % | 底座 Union R@0.5 % | SFT Union R@0.5 % |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, label in AREA_GROUPS:
        x, y = (ev[name]["metrics"]["by_anomaly_area"][key] for name in ("base_region", "quality_region"))
        lines.append(f"| {label} | {x['n']} | {pct(x['anomaly_recall'])} | {pct(y['anomaly_recall'])} | "
                     f"{pct(x['union_box_recall_iou_0.5'])} | {pct(y['union_box_recall_iou_0.5'])} |")
    lines += ["", "各组尤其较大面积组样本很少，分层结果仅用于描述失败模式，不能据此确认面积相关的稳定优势。"
              "本次训练参数、提示和最终步数按预定方案冻结，不依据这张测试分层表改参。", "",
              "## 四组配对差值与95%区间", "",
              f"按 `source_id` 做配对 cluster bootstrap，**{bundle['paired']['n_resamples']:,} 次**重采样，"
              f"seed={bundle['paired']['seed']}；同一轮两模型使用相同抽中源图及任务次数。"
              "报告左减右的百分点差值及 percentile 95% 区间（线性插值）。FPR/FNR 越低越好，其余指标越高越好；"
              "单模型图上的区间来自各评估的源图 bootstrap，不能用两个单模型区间是否重叠替代配对差值检验。", "",
              "| 对比（左−右） | 指标 | 差值 pp | 95% CI pp | 有定义的重采样次数 |", "|---|---|---:|---:|---:|"]
    for left, right, label in PAIRS:
        item = bundle["comparisons"][(left, right)]
        for key, metric_label in METRICS.items():
            estimate = item["metrics"][key]
            if estimate["difference"] is None:
                lines.append(f"| {label} | {metric_label} | 不适用（判断任务） | 不适用 | {estimate['n_defined_resamples']} |")
            else:
                lines.append(f"| {label} | {metric_label} | {signed(estimate['difference'])} | {interval(estimate)} | {estimate['n_defined_resamples']} |")
    lines += ["", "区间条件于本轮固定训练模型和小规模 pilot，未覆盖训练种子波动、域偏移或预训练污染。"
              "四组预定比较和多个指标均为描述性分析，未做多重比较校正；区间覆盖0时应保留不确定性。"
              f"完整结果见 {link('reports/paired_difference.json')} 与 {link('reports/paired_difference.md')}。", "",
              "## 推理成本与训练拟合诊断", "",
              "所有测试均使用相同 NF4 加载策略、全图 max_pixels=524,288、max_length=2,048、"
              "max_new_tokens=128、贪心解码（do_sample=false、num_beams=1）。", "",
              "| 评估 | 平均生成延迟 秒/图 | 峰值 allocated GiB | 平均输入 token | 平均生成 token |", "|---|---:|---:|---:|---:|"]
    for _, names, _, _ in PANELS:
        for name in names:
            data, m = ev[name]["rows"], overall(name)
            lines.append(f"| {NAMES[name]} | {m['mean_latency_seconds']:.4f} | {m['max_peak_memory_allocated_bytes'] / 2**30:.2f} | "
                         f"{statistics.mean(r['input_tokens'] for r in data):.2f} | {statistics.mean(r['generated_tokens'] for r in data):.2f} |")
    d = overall("overfit32_training_diagnostic")
    lines += ["", "延迟测量排除图片预处理与首次 warm-up，包含张量迁移、模型生成、同步和解码；"
              "它不等于端到端服务延迟、并发吞吐或可直接外推的线上 QPS。显存为 PyTorch allocated 峰值，不是整卡占用。", "",
              f"![推理延迟与分类效果]({os.path.relpath(root / 'reports/figures/latency_macro_f1.png', output.parent)})", "",
              f"单独的32张训练源图诊断：实际 n={d['n']}，格式有效率 {pct(d['format_valid_rate'])}%，"
              f"二类 macro F1 {pct(d['macro_f1'])}%，异常召回 {pct(d['anomaly_recall'])}%，"
              f"Union R@0.5 {pct(d['union_box_recall_iou_0.5'])}%。这些图用于训练拟合检查，"
              "与留出测试面板分开，不能当作泛化或领域适配收益证据。", "",
              "## 结论适用范围", "",
              "- 仅一个训练随机种子、每组768张训练图；主测试120张、外部150张，每类别只有5正常＋5异常，类别级指标不稳定。均衡标签比例不同于生产异常率，precision/F1 不能直接作为实际产线估计。",
              "- VisA 为本项目的真实异常有监督重新划分；与原始 normal-only/zero-shot/few-shot 协议的信息条件不同。外部 MVTec 只验证一个固定 mirror pilot，不能代替完整官方测试或广泛跨域结论。",
              "- SHA256 隔离排除完全相同原始文件，缺少物理零件身份信息，不能证明同一零件、相邻拍摄或近重复内容完全隔离；公开基准是否进入底座预训练未知。",
              "- 面积覆盖与随机组可能共享异常源图，配额和正常成员已匹配；此对照只检验抽样策略。共同判断提示控制了评估任务，但训练 prompt、答案长度和监督 token 数不同，FLOP 也未测量。",
              "- 格式有效率接受可选 JSON code fence、有限数值坐标和多个框；多个框先归并为外接矩形。因此该指标不代表严格整数、单框输出遵从率。正常标签却给出非空框等语义一致性另在 metrics 中记录。",
              "- 定位是包含所有缺陷的外接矩形 IoU；无法证明像素级分割、每个缺陷实例识别或精细边界质量。离散布尔生成未转换为异常分数，未计算 AUROC。",
              "- 本次目标仅为判断与框，没有逐图专家解释、教师推理链或解释质量验证；单图训练也不能支持多视角、参考正常图条件化的能力结论。",
              "- 固定最终96步，验证 CE 仅来自前24条；32图诊断与测试分开。重载检查是一个提示位置的 logits 检查；完成验证核对产物、身份及指标一致性，并非独立重新训练。训练只保存适配器及配置，未实现优化器状态断点恢复。",
              "- 报告依据四组配对差值，不从较小的正向点估计推导稳定方法优势；跨0区间保留不确定性。结果不构成 SOTA 或跨训练种子结论。", "",
              "## 对应的大模型后训练工程能力", "",
              "这份实验体现的是可核查的多模态领域后训练工程：将工业图像、真实 mask 和判断标签转成可追溯的结构化 SFT 数据；"
              "控制源图隔离、面积配额与实验信息条件；继承原生 VLM 的图文连接结构，核对视觉 token、坐标、assistant 标签 mask 的一致性；"
              "在24GB单卡上完成语言侧 QLoRA、变长目标 token 加权、梯度累计及适配器保存重载；"
              "用保留格式失败的留出评估、共同提示消融、外部 pilot、配对区间和逐图输出约束结论。"
              "视觉连接模块没有更新，相关经验应表述为基于原生 VLM 的多模态 SFT 与空间 grounding 领域适配。", "",
              "## 可核查产物与复现", "",
              f"完成核验时间 `{bundle['verification']['verified_at']}`，通过 **{len(bundle['verification']['checks'])}** 项检查。"
              f"核验范围：产物、数据、身份及指标一致性；详见 {link('ops/completion_verification.json')}。", "",
              "| 评估 | 指标 | 元数据 | 逐图输出 |", "|---|---|---|---|"]
    for name in NAMES:
        lines.append(f"| {name} | {link(f'evaluations/{name}/metrics.json', 'metrics')} | "
                     f"{link(f'evaluations/{name}/meta.json', 'meta')} | {link(f'evaluations/{name}/predictions.jsonl', 'predictions')} |")
    lines += ["", "| 训练组 | 摘要 | 配置 | 日志 | 参数审计 |", "|---|---|---|---|---|"]
    for name in ARMS:
        lines.append(f"| {name} | {link(f'runs/{name}/summary.json', 'summary')} | {link(f'runs/{name}/run_config.json', 'config')} | "
                     f"{link(f'runs/{name}/metrics.jsonl', 'events')} | {link(f'runs/{name}/parameter_audit.json', 'parameters')} |")
    lines += ["", f"其他入口：{link('configs/experiment_plan.json', '预定方案')}、{link('PROJECT_CARD_ZH.md', '项目卡')}、"
              f"{link('README.md', 'README')}、{link('reports/matrix.md', '实验矩阵')}、"
              f"{link('reports/figures/figure_data.json', '绘图源数据')}、{link('ops/prepared_validation.json', '数据核验')}、"
              f"{link('ops/visa_download_provenance.json', 'VisA 来源')}。", ""]
    for name, label in (("method", "方法图"), ("training_curves", "训练曲线"),
                        ("evaluation_bars", "评估柱状图"), ("latency_macro_f1", "延迟图")):
        lines.append(f"- {label}：{link(f'reports/figures/{name}.png', 'PNG')} / {link(f'reports/figures/{name}.pdf', 'PDF')}。")
    server_root = Path(runtime["model"]).parent.parent
    virtualenvs = [Path(p) for p in bundle["cleanup"]["preserve"] if Path(p).name == ".venv"]
    require(len(virtualenvs) == 1, "Cannot identify recorded runtime interpreter")
    python = virtualenvs[0] / "bin/python"
    lines += ["", "在已准备相同模型权重、原始数据与固定配置的服务器项目目录执行：", "", "```bash",
              f"cd {shlex.quote(str(server_root))}", 'export PYTHONPATH="$PWD/src"',
              "export OMP_NUM_THREADS=8", "export TOKENIZERS_PARALLELISM=false",
              f"PY={shlex.quote(str(python))}",
              "$PY ops/run_pipeline.py",
              "$PY -m industrial_vlm.compare --evaluations evaluations --output reports --bootstrap-samples 2000 --seed " + str(bundle["paired"]["seed"]),
              '$PY ops/plot_results.py --root "$PWD" --output reports/figures',
              '$PY ops/verify_completion.py --root "$PWD"',
              '$PY ops/write_report_zh.py --root "$PWD"', "```", "",
              "流水线会复用完成的训练标记，并用原数据、配置和适配器身份检查恢复评估。"
              "重新训练或改变预算应使用独立实验目录及重新生成的配置；未完成训练不能用适配器冒充优化器 checkpoint 续接。"
              "单图推理示例：", "", "```bash",
              "$PY -m industrial_vlm.infer --model models/Qwen3-VL-8B-Instruct \\",
              "  --adapter runs/quality_region/adapter --image /absolute/path/to/image.png \\",
              "  --task region --output prediction.json", "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/实验报告.md"))
    parser.add_argument("--check-only", action="store_true", help="Validate and render in memory without writing the report")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    bundle = load_bundle(root)
    document = render(bundle, output)
    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({"status": "validated" if args.check_only else "written", "output": str(output),
                      "evaluations": len(bundle["evaluations"]), "training_arms": len(bundle["training"]),
                      "paired_comparisons": len(bundle["comparisons"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
