# Qwen2.5-VL-3B 与 Qwen3-VL-2B 差异分析

## 当前结论

目前不能把两者的差距解释为纯粹的视觉能力差距。已完成的 36 个完整样本（180 视角）扩展测试中，两模型使用相同的英文 P0、direct 448、5-view max；Qwen2.5 的样本级 AUROC/AUPR 为 0.7454/0.7913，Qwen3 为 0.5401/0.5670。但两者的硬准确率都接近随机（50%），且 Qwen3 的 P(Yes) 几乎全部饱和在 1，Qwen2.5 则集中在 0.2 左右。这首先是输出先验和校准差异。

更重要的是，已有的 Qwen3 Dev720 文件只有 27 行，不是 3600 视角正式结果；它不能与 Qwen2.5 的 3600 视角 forward-only 基线直接比较。严格的 4090D 同协议 Qwen3 测试已排队，需使用同一 `fold1/dev.parquet`、同一 `Dataset448` 缓存、同一 P0、`use_cache=False`、`logits_to_keep=1` 和相同的 3600 视角聚合。

## 为什么会出现饱和

1. 聊天模型没有二分类头。这里把首个自然语言 token 的 Yes/No 相对 logit 当分类器，结果同时受指令微调的回答先验、语言模型词频和模板末尾状态影响。
2. Qwen3 与 Qwen2.5 不是同一 checkpoint 的小版本。参数量约 2.13B 对 3.75B，视觉网格约 28x28 对 32x32，视觉投影、指令微调和 tokenizer/chat-template 均不同；同一 prompt 不保证同一 logit 标尺。
3. 两个 tokenizer 的 `Yes`/`No` 候选词 ID 已核对，不能把差距归因于误用固定 token ID；但应始终对每个模型动态读取 `word` 和带前导空格的候选 token。
4. 两个模型的 chat-template SHA256 不同（Qwen3: `3636d0f0...a489e4`，Qwen2.5: `a0bc6f6f...f2fddb`）。即使可见 prompt 文本相同，序列化后的 system/user/assistant 边界和视觉占位符也不同，首 token hidden state 也会不同。
5. 旧的 36 样本测试使用 `generate` 读取首 token；Qwen2.5 的正式 Dev720 使用 CUDA forward-only。generate 与 forward-only 已在 Qwen2.5 上验证误差很小，但 Qwen3 仍需在 4090D 重新验证，不能假设完全等价。
6. 5 个视角属于同一物体，不能视为 5 个独立样本；max 聚合会放大偶然高分。36 个完整样本只能说明饱和现象不是 smoke20 单个样本造成的，最终判断要以 720 组 bootstrap 为准。

## 当前使用的提示词

### P0（准确性主线，已冻结）

```text
You are inspecting one view of an industrial component for visible defects.
A defect means a visible scratch, crack, deformation, discoloration, stain, missing or broken part, pit, burn mark, coating problem, contaminant, or another localized abnormality. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

The first generated token must be exactly Yes or No. Then output exactly one JSON object with a region selected from: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
Format: Yes {"region":"center"}
For a normal image use: No {"region":"none"}
Do not output Markdown or an explanation.
```

### P1（已冻结的提示词消融）

```text
Inspect this image of an industrial component and classify only what is visibly present.

Normal: the component is intact and its edges, holes, reflections, shadows, text, color, and repeated texture are internally consistent.
Anomalous: a localized region visibly breaks that consistency through damage, missing material, contamination, deformation, or an irregular surface or color change.

Treat the normal and anomalous hypotheses symmetrically. The first generated token must be exactly Yes for anomalous or No for normal. Then output exactly one JSON object whose region is one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
Use No {"region":"none"} when the normal hypothesis is stronger. Do not output Markdown or an explanation.
```

### P2（已冻结的提示词消融）

```text
Compare two hypotheses for the pictured <object> using only visible image evidence.

H0: this is a normal instance; visible structure and appearance are mutually consistent.
H1: this instance contains a localized anomaly; some visible region is inconsistent with the rest of the object.

Choose the better-supported hypothesis. The first generated token must be exactly No for H0 or Yes for H1. Then output exactly one JSON object whose region is one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
For H0 use No {"region":"none"}. Do not output Markdown or an explanation.
```

## 下一步实验

- 在 4090D 以 forward-only 同协议完成 Qwen3 3600 视角，并保存首 token raw logits、`yes_logit-no_logit`、P(Yes)、解析率、视觉网格和模板哈希。
- 在同一 36/180 扩展子集上额外运行 P1/P2，比较 prompt 是否只改变先验饱和而不改变排序。
- 对完整 720 组报告 max/top2/mean、18 类宏指标和按 group 分层 bootstrap；在独立 calibration split 报温度与阈值，禁止用评测集调参。

在上述同协议结果完成前，Qwen2.5-VL-3B 继续作为 B0+ 准确性主线，Qwen3 仅作对照，不替换主线，也不开启 Visual Adapter、LoRA 或 holdout-zs。
