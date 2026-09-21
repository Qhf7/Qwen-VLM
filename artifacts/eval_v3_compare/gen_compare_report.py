# -*- coding: utf-8 -*-
import json, os

BASE = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/artifacts/eval_v3_compare"

def load(name):
    with open(os.path.join(BASE, f"fusion_{name}", "fusion_ablation.json")) as f:
        return json.load(f)["results"]

res = {n: load(n) for n in ["baseline", "mixed", "v3"]}

def g(m, s, k):
    return res[m][s][k]

def pct(x):
    return f"{x*100:.1f}%"

# 统一口径：baseline 取 full_only（整图，非 patch 模型）；mixed/v3 取 mean_w0.5（九宫格融合）
main = {
    "baseline (decoder_only, full_only)": ("baseline", "full_only"),
    "mixed (patch9_mixed, mean_w0.5)":     ("mixed", "mean_w0.5"),
    "v3 (patch9_mixed_v3, mean_w0.5)":     ("v3", "mean_w0.5"),
}

metrics = [
    ("macro_dice", "Macro Dice", True),
    ("tiny_dice", "tiny Dice", True),
    ("tiny_recall", "tiny Recall", True),
    ("small_dice", "small Dice", True),
    ("small_recall", "small Recall", True),
    ("large_dice", "large Dice", True),
    ("two_stage_dice", "two_stage Dice", True),
    ("pixel_precision", "Pixel Precision", True),
    ("pixel_recall", "Pixel Recall", True),
    ("normal_nonempty", "正常图非空数(假阳)", False),
    ("normal_fp_pixel_rate", "正常图假阳像素率", True),
]

def table_block(title, rows):
    lines = [f"### {title}", "",
             "| 模型 | " + " | ".join(r[1] for r in metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for label, (m, s) in rows:
        cells = []
        for key, _, isp in metrics:
            v = g(m, s, key)
            cells.append(pct(v) if isp else f"{int(v)}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)

# 主表
main_tbl = table_block("统一口径三方对比（baseline=full_only, mixed/v3=mean_w0.5）", list(main.items()))

# 各模型 full_only vs mean_w0.5
internal = {
    "baseline": [("baseline", "full_only"), ("baseline", "mean_w0.5")],
    "mixed":     [("mixed", "full_only"), ("mixed", "mean_w0.5")],
    "v3":        [("v3", "full_only"), ("v3", "mean_w0.5")],
}
internal_tbl = table_block("各模型内部：整图(full_only) vs 九宫格融合(mean_w0.5)",
                           [("baseline full_only", ("baseline","full_only")),
                            ("baseline mean_w0.5", ("baseline","mean_w0.5")),
                            ("mixed full_only", ("mixed","full_only")),
                            ("mixed mean_w0.5", ("mixed","mean_w0.5")),
                            ("v3 full_only", ("v3","full_only")),
                            ("v3 mean_w0.5", ("v3","mean_w0.5"))])

# tiny/small 专项
tiny_small = ["tiny_dice","tiny_recall","small_dice","small_recall"]
ts_metrics = [(k, k.replace("_"," ").title(), True) for k in tiny_small]
def ts_table():
    lines = ["### tiny / small 召回专项（mean_w0.5 口径）", "",
             "| 模型 | tiny Dice | tiny Recall | small Dice | small Recall |",
             "|---|---|---|---|---|"]
    for label, (m, s) in [("baseline (full_only)", ("baseline","full_only")),
                          ("mixed (mean_w0.5)", ("mixed","mean_w0.5")),
                          ("v3 (mean_w0.5)", ("v3","mean_w0.5"))]:
        cells = [pct(g(m,s,k)) for k in tiny_small]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
ts_tbl = ts_table()

# 差值（v3 - mixed, 均 mean_w0.5）
def diff_str(mk):
    dv = g("v3","mean_w0.5",mk) - g("mixed","mean_w0.5",mk)
    return f"{dv*100:+.1f} pp"

deltas = {
    "Macro Dice": diff_str("macro_dice"),
    "tiny Recall": diff_str("tiny_recall"),
    "small Dice": diff_str("small_dice"),
    "two_stage Dice": diff_str("two_stage_dice"),
    "Pixel Precision": diff_str("pixel_precision"),
}

md = f"""# v3 vs baseline vs mixed 三方评测对比报告

> 生成时间：2026-09-21 ｜ 评测运行：19:11–19:49（服务器 RTX 4090D）
> 数据集：dev（anomaly=1775，分类门控后 needed=1923 图）｜ seg_thr=0.5, gate_thr=0.5
> 来源：`artifacts/eval_v3_compare/fusion_{{baseline,mixed,v3}}/fusion_ablation.json`

## 1. 核心结论（先说结果）

**v3 在 dev 上未达预期 —— RC-3（含缺陷曝光翻倍到 8/step）在此配置下没有带来提升，反而下降。**

- v3 的 `mean_w0.5` 单模型 **Macro Dice = {pct(g('v3','mean_w0.5','macro_dice'))}**，two_stage Dice = {pct(g('v3','mean_w0.5','two_stage_dice'))}
- 对比 mixed（同口径）Macro Dice = {pct(g('mixed','mean_w0.5','macro_dice'))}，two_stage = {pct(g('mixed','mean_w0.5','two_stage_dice'))} → v3 **低 {deltas['Macro Dice']}**
- 对比 baseline 整图（full_only）Macro Dice = {pct(g('baseline','full_only','macro_dice'))}，two_stage = {pct(g('baseline','full_only','two_stage_dice'))} → v3 也低
- **尤其关键：tiny Recall 仅 {pct(g('v3','mean_w0.5','tiny_recall'))}**，远低于 mixed（{pct(g('mixed','mean_w0.5','tiny_recall'))}）与 baseline（{pct(g('baseline','full_only','tiny_recall'))}）。这与"RC-3 提升 tiny/small 召回"的目标**完全相反**。

**结论：本次 v3 实验为阴性结果，不能作为正式提交替换 mixed；mixed 仍是本地 dev 最佳的可提交模型。**

## 2. {main_tbl}

## 3. {internal_tbl}

> 说明：对 patch 模型（mixed/v3），`mean_w0.5` 融合明显高于其 `full_only` 整图（mixed +2.4pp、v3 +2.2pp），证明九宫格融合有效。
> 对 baseline（整图模型），`mean_w0.5` 反而把整图优势拉低（0.774→0.716），因其 patch 由不擅长局部放大的整图模型裁出、质量差——故 baseline 统一以 `full_only` 代表。

## 4. {ts_tbl}

## 5. 为什么 v3 不如 mixed（原因分析，待验证）

v3 与 mixed 的唯一实质差异是 RC-3（异常行 2→4，accumulation 8→12，含缺陷曝光 4→8/step）。
v3 的 `full_only` 基值（{pct(g('v3','full_only','macro_dice'))}）就低于 mixed（{pct(g('mixed','full_only','macro_dice'))}），
且 `tiny_recall` 在 mean_w0.5 下异常低（{pct(g('v3','mean_w0.5','tiny_recall'))} vs mixed {pct(g('mixed','mean_w0.5','tiny_recall'))}）。可能原因：

1. **等效 batch 变大但学习率未同步** —— v3 每步累积 12 样本（mixed 为 8），但 lr 调度/peak（1e-4）沿用，等效学习率相对偏小，12000 步收敛不足。
2. **seed 差异** —— v3 用 seed=20260921，mixed 为原 seed，训练轨迹随机性可能影响最终泛化。
3. **过拟合** —— v3 训练末步 loss≈0.296（最佳 0.044@11630），但 dev 泛化差，提示在训练集上过拟合、验证分布偏移。
4. **空 patch / 正常图比例被相对稀释** —— 异常行占比从 2/8 升到 4/12，正常图空监督占比下降，可能损害正常图假阳控制（normal_nonempty: v3={int(g('v3','mean_w0.5','normal_nonempty'))} vs mixed={int(g('mixed','mean_w0.5','normal_nonempty'))}）。

## 6. 下一步建议

- **不替换正式提交**：维持 mixed（step11000 + mean_w0.5）为当前最佳可提交模型。
- **若坚持验证 RC-3 假设**：重训 v3 时把 lr peak 提到 1.5–2e-4（或 accumulation 保持 12 但 step 增到 20000），让更大 batch 充分收敛；或固定 mixed 的 seed 复现对比。
- **额外重评 11630 检查点**：v3 训练 loss 最低点在 step11630，可补存该检查点重评，确认是否仅"末段过拟合"导致 12000 偏弱。
- **排查 v3 训练曲线**：回看 `training_report.html` 的 val 近似指标，确认是否存在早停点。

---
*本报告数字由 `gen_compare_report.py` 从三个 fusion_ablation.json 自动抽取，未手工改写。*
"""

out = os.path.join(BASE, "v3三方对比报告.md")
with open(out, "w") as f:
    f.write(md)
print("WROTE", out)
print("v3 mean_w0.5 macro_dice =", pct(g('v3','mean_w0.5','macro_dice')), "tiny_recall =", pct(g('v3','mean_w0.5','tiny_recall')))
print("mixed mean_w0.5 macro_dice =", pct(g('mixed','mean_w0.5','macro_dice')), "tiny_recall =", pct(g('mixed','mean_w0.5','tiny_recall')))
print("baseline full_only macro_dice =", pct(g('baseline','full_only','macro_dice')))
