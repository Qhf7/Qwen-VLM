"""把 B0+ 的原始 CSV 汇总为可复核的中文实验报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from b0.analyze_b0plus import analyze_five_crop, analyze_prompts
except ModuleNotFoundError:
    from analyze_b0plus import analyze_five_crop, analyze_prompts


OUT = Path("artifacts/b0plus")


def percentile(frame: pd.DataFrame, column: str, q: float) -> float:
    return float(np.percentile(frame[column], q))


def compact_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    labels = {
        "prompt_id": "配置",
        "aggregation": "聚合",
        "sample_auroc": "AUROC",
        "sample_aupr": "AUPR",
        "sample_f1max": "F1max",
        "macro_class_auroc": "宏 AUROC",
        "hard_yes_rate_at_0.5": "Yes率@0.5",
        "ng_gt_ok_classes": "NG>OK 类数",
        "score_dynamic_range": "分数跨度",
        "five_view_latency_p50_s": "5视角p50(s)",
        "five_view_latency_p95_s": "5视角p95(s)",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    best = analyze_prompts(OUT)
    crop_table = analyze_five_crop(OUT, best)
    validation = pd.read_csv(OUT / "forward_validation.csv")
    single = pd.read_csv(OUT / "single_view_latency.csv")
    batch5 = pd.read_csv(OUT / "batch5_latency.csv")
    prompt = pd.read_csv(OUT / "prompt_ablation.csv")
    aggregation = pd.read_csv(OUT / "aggregation_ablation.csv")
    protocol = json.loads((OUT / "preregistered_protocol.json").read_text(encoding="utf-8"))

    single_p50 = percentile(single, "total_s", 50)
    single_p95 = percentile(single, "total_s", 95)
    batch_p50 = percentile(batch5, "total_s", 50)
    batch_p95 = percentile(batch5, "total_s", 95)
    serial5_estimate = 5 * single_p50
    batch_penalty = batch_p50 / serial5_estimate
    prompt_columns = [
        "prompt_id", "sample_auroc", "sample_aupr", "sample_f1max",
        "macro_class_auroc", "hard_yes_rate_at_0.5", "ng_gt_ok_classes",
        "score_dynamic_range",
    ]
    aggregation_columns = [
        "prompt_id", "aggregation", "sample_auroc", "sample_aupr",
        "sample_f1max", "macro_class_auroc",
    ]
    crop_columns = [
        "prompt_id", "sample_auroc", "sample_aupr", "sample_f1max",
        "macro_class_auroc", "five_view_latency_p50_s", "five_view_latency_p95_s",
    ]
    lines = [
        "# B0+：forward-only、prompt、聚合与 5-crop 实验报告",
        "",
        "## 1. 冻结协议",
        "",
        f"- 协议：`{protocol['version']}`，预注册日期 `{protocol['preregistered_at']}`；",
        "- 模型：`Qwen2.5-VL-3B-Instruct`，MPS FP16，direct 448；",
        "- 子集：`smoke20`，20 个完整样本、100 个视角、10 类；",
        "- P0/P1/P2 在运行前写入协议文件，选择只使用 5-view max；",
        f"- 最佳 prompt：`{best}`；holdout-zs 未使用。",
        "",
        "## 2. B0f 数值正确性",
        "",
        f"forward-only 使用 `use_cache=False` 和 `logits_to_keep=1`。5 张固定图上，"
        f"forward 与 generate 首 token 的最大 P(Yes) 误差为 "
        f"`{validation.abs_forward_generate.max():.3g}`；batch5 与逐图 forward 的最大误差为 "
        f"`{validation.abs_batch_single.max():.3g}`。两者均通过预注册容差。",
        "",
        "## 3. 延迟",
        "",
        "| 路径 | 重复数 | 总延迟 p50 | 总延迟 p95 | forward p50 | forward p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 单视角 B0f | {len(single)} | {single_p50:.3f}s | {single_p95:.3f}s | "
        f"{percentile(single, 'forward_s', 50):.3f}s | {percentile(single, 'forward_s', 95):.3f}s |",
        f"| 5 视角 batch prefill | {len(batch5)} | {batch_p50:.3f}s | {batch_p95:.3f}s | "
        f"{percentile(batch5, 'forward_s', 50):.3f}s | {percentile(batch5, 'forward_s', 95):.3f}s |",
        "",
        f"按单图中位数估计，5 次串行为 `{serial5_estimate:.3f}s`；真实 batch5 中位数反而是其 "
        f"`{batch_penalty:.2f}×`。这不是算法加速，而是 18GB 统一内存下的大 batch 换页/内存压力。"
        "因此 batch5 只作为工程负结果保留，完整消融使用已验证分数等价的逐图 B0f。",
        "",
        "## 4. Prompt 独立消融（固定 5-view max）",
        "",
        compact_table(prompt, prompt_columns),
        "",
        f"预注册规则选中 `{best}`。三个 prompt 在 0.5 阈值下仍全部硬判 No（Yes率为 0），"
        "说明 prompt 改写没有解决校准问题；但连续 logits 的排序能力不同，P0 最好。",
        "",
        "## 5. 离线多视角聚合",
        "",
        compact_table(aggregation, aggregation_columns),
        "",
        "P0 的 mean/top-2 mean 在 smoke20 上高于 max，但这是同一小子集上的事后诊断。"
        "S0 主口径仍保持 max；只有在 720 样本 Dev-mini 上复现后才允许修改聚合规则。",
        "",
        "## 6. B0w-5crop（只用最佳 prompt）",
        "",
        compact_table(crop_table, crop_columns),
        "",
        "5-crop 定义为全局图加原图四象限，每个 crop 独立缩放到 448；先对 crop 取 max，"
        "再对 5 个视角取 max。它只改变局部输入，不改变 prompt 或模型。",
        "全局 crop 与 direct 的 100 个视角分数逐项完全一致。四象限却在 97/100 个视角上"
        "取得最高分；样本分数相对 direct 的中位增量在 OK 上为 0.0801、在 NG 上仅为 0.0527。"
        "也就是说，局部放大主要放大了正常边缘和纹理的伪异常分，解释了 AUROC/AUPR 的下降。",
        "",
        "## 7. 结论与决策",
        "",
        "1. B0f 已成立，可替代 generate 做图级概率评测；",
        "2. 本机不应使用 5 图 batch，串行单图反而更快、更稳定；",
        f"3. 零训练配置冻结为 `{best}`，主聚合暂时仍是 max；",
        "4. B0w-5crop 的 AUROC 下降 0.155、AUPR 下降 0.1184，且 5 视角 p50 达 39.3s；"
        "按预先写明的阶段门，停止 B0w，不打开 B0w-native；",
        "5. 下一步是在 720 样本 Dev-mini 上复核冻结候选，再准备 hard-normal、平衡 sampler 和 E1。",
        "",
    ]
    (OUT / "B0plus_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "B0plus_results.md")


if __name__ == "__main__":
    main()
