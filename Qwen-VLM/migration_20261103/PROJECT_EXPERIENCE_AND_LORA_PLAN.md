# 已完成项目经历与下一阶段 LoRA 方案

更新：2026-11-04，Asia/Shanghai。本文区分已经完成并有实验产物支持的工作，以及尚未实施的 LoRA 计划。没有在本次整理中启动训练或新增测试集查询。

后续执行记录：用户随后授权实施B/C方案，目前六次正式训练与独立重载检查均已完成。结果见 [LoRA三seed分析](lora_20261104/ANALYSIS.md) 和 [固定协议](lora_20261104/PROTOCOL.md)。下文第1—6节保留方案提出时的事实状态；已实现LoRA的新简历条目见第7节。

## 1. 当前项目定位

项目名称建议：基于 AnomalyGPT 与 Qwen3-VL 的工业缺陷检测及多模态适配。

当前成果属于多模态模型迁移、跨模态接口适配、监督定位、知识蒸馏和可复现评测。现有语言模型保持冻结，实际训练的是 CLS 投影、异常图 PromptLearner 或独立定位头。尚未开展 LLM LoRA、全参数微调、偏好优化或大模型预训练。

## 2. 已完成工作

| 工作 | 已完成内容 | 可引用证据与边界 |
|---|---|---|
| 数据与评测管线 | 检查并修正 mask 二值化、图像/掩码几何、patch 布局、定位器权重加载与图像/像素指标口径；保存数据和模型来源 SHA | 自定义监督划分 train9985、val380；真实异常参与训练；不能写成官方 normal-only 无监督结果 |
| 语言骨干迁移 | 固定 ImageBind 与 BigDec，接入 Qwen3-VL-8B 语言部分；训练相同结构的 CLS 投影与 18-token 异常图 PromptLearner | 三 seed 配对开发验证 Image AUROC：Vicuna 0.889684 → Qwen 0.925779，+3.6095 个百分点；生成 macro F1 0.816122 → 0.845672 |
| 原生多模态接口 | 接入 Qwen 原生 RGB 与异常图 token；检查 MRoPE、DeepStack、attention mask 与 KV cache；进行官方/manual 同形状输出核验 | native+map 验证 AUROC0.866391；与路线1输入、模板、训练接口不同，不作为编码器单变量排名 |
| 定位与蒸馏 | 比较正常记忆库、pre/post-merge、224/448、GT 与 GT+KD 定位头；核验匹配训练计划与最佳权重重载 | Qwen pre448 GT Pixel AP0.336404；同特征 KD 为0.327643，不能写蒸馏稳定提高 |
| 推理时移除外部 teacher | Qwen视觉→SpatialHead→预测ROI→原图+ROI→原生Qwen；完成GT/KD各val380和不需类别/标签/cache的单图入口 | GT ROI Image AUROC0.836161，对照teacher ROI0.837265；生成F1低于teacher，不能写性能等价 |
| 消融与失败分析 | 三seed、置零/空间打乱/反标签donor map、同预算中心/随机/teacher/student ROI；检查逐图漏检与GT覆盖 | GT ROI90个FN分解为零覆盖30、部分17、全部43；60个仍保留GT不能证明细节可见或模型理解 |
| 工程交付 | 单卡串行GPU实验、配置/采样/源码哈希、strict权重重载、JSON/JSONL/CSV报告、可执行单图CLI | 两例student单图与批量地图/ROI/logits/生成精确一致；包括保留的真实异常漏检，不是额外泛化评测 |

主要证据：

- [语言替换三seed](PHASE1_RESULTS.md)
- [原生多模态接口](PHASE2_RESULTS.md)
- [定位与蒸馏](PHASE3_5_RESULTS.md)
- [ROI对照](PHASE4_RESULTS.md)
- [学生ROI集成](student_roi_20261104/STUDENT_ROI_RESULTS.md)
- [逐图失败诊断](student_roi_20261104/failure_analysis_224/REPORT.md)
- [单图入口](student_roi_20261104/INFERENCE_USAGE.md)

## 3. 可直接用于简历的版本

**基于 AnomalyGPT 与 Qwen3-VL 的工业缺陷检测及多模态适配**

技术栈：Python、PyTorch、Transformers、Qwen3-VL-8B、ImageBind、跨模态适配、知识蒸馏。

- 实现 AnomalyGPT 语言骨干向 Qwen3-VL-8B 的迁移，构建全局视觉投影与18-token异常图提示适配接口；固定视觉前端并开展3个随机种子的配对实验，在自定义监督验证集上将平均 Image AUROC 从88.97%提升至92.58%（+3.61个百分点），生成 Macro-F1 从81.61%提升至84.57%。
- 适配 Qwen 原生多模态推理接口，核验 MRoPE、DeepStack 与 KV cache 的输出一致性；对 pre/post-merge 特征、输入分辨率及GT/KD定位头开展对照，完成 Qwen视觉定位头驱动ROI与回答的推理链，移除推理阶段的 ImageBind/BigDec 依赖。
- 构建覆盖图像检测、像素定位和文本生成的统一评测流程，修正数据几何与特征布局问题，设计异常图干预及同预算ROI消融；结合逐图缺陷覆盖分析拆解漏检，建立配置/数据/权重哈希追踪与最佳权重重载核验。
- 在 RTX4090D 24GB 单卡环境完成训练及单图推理验证，交付异常评分、原始概率图、热图及结构化输出；Qwen学生ROI链路的固定样例通过单图/批量地图、ROI、logits及生成token精确对照。

篇幅不足时保留前三条。技术栈中的“知识蒸馏”代表已实施与评估，不代表蒸馏取得提升。项目周期、职位和单位请使用真实信息，不根据实验目录日期推断。

面试介绍建议：

> 我研究的是如何把更强的语言骨干迁移到工业异常检测系统。先固定视觉前端，通过相同初始化与采样计划比较 Vicuna 和 Qwen，再探索 Qwen 原生视觉与定位接口。当前三seed验证支持语言替换路线的排序性能改善；同时做了推理移除外部teacher的完整链路和逐图失败分析。下一阶段准备在已对齐接口上引入语言LoRA，区分继续训练、低秩适配和任务损失各自的收益。

## 4. 为什么下一步可以做语言 LoRA

现有系统把视觉信息适配到冻结的 Qwen 输入空间。下一步在语言注意力线性层增加低秩更新，使语言内部计算也能适配这些视觉embedding及工业二分类任务。

标准LoRA形式为 W' = W + (alpha/r)BA；原权重W保持冻结，更新低秩矩阵A/B。这属于参数高效领域后训练。它有明确技术可行性，但是否提高AUROC、召回或生成表现，必须由当前项目的匹配实验回答。

当前监督答案主要是两种固定Yes/No模板。这样的LoRA可评测异常判别适配，尚不足以证明复杂缺陷解释或推理能力增强。若后续目标包括缺陷类型、位置和原因描述，需要另建经核验的相应训练与评测数据。

## 5. 建议执行的实验顺序（尚未启动）

### 5.1 从已有桥接权重出发

每个seed加载自身已经完成的Qwen桥接checkpoint，保留同一ImageBind/BigDec、模板、cache、评分和生成接口。先在seed42做工程预检，正式结论补齐42/43/44；不可只拿新seed42与旧三seed均值比较。

| Arm | 原语言权重 | 已训练桥接 | 新LoRA | 作用 |
|---|---|---|---|---|
| A | 冻结 | 冻结 | 无 | 已有模型的起点锚点，不增加训练 |
| B | 冻结 | 继续训练 | 无 | 控制额外数据曝光/训练预算 |
| C | 冻结 | 冻结 | 训练 | 隔离语言LoRA的作用 |
| D | 冻结 | 小学习率继续训练 | 训练 | 检查桥接与语言联合适配，第二步再做 |

B/C先采用相同新增500updates、有效batch4（2000次新增图像曝光）与相同采样计划，每100步完整val380并记录step0。预算为拟议起点，不是已验证最优配置。D视B/C结果追加。

### 5.2 最小LoRA起点

- 只对实际保留的language_model内部目标层注入adapter；按named_modules检查真实名称。
- 初始候选：q_proj/v_proj，r=8、alpha=16、bias=none；先固定这些设置。
- LoRA学习率初始候选1e-4；联合训练时桥接使用更小学习率并独立分组。不得直接把桥接旧学习率5e-4当作已验证的LoRA学习率。
- 先完成小样本拟合、有限loss/梯度、仅目标参数更新、未训练adapter不改变起点输出、adapter和桥接保存重载核验。
- rank16或q/k/v/o扩展仅在最小方案完成后作为单独对照，不同时改变loss、模板、精度与采样。

### 5.3 当前代码的必要改动

- 原实现会冻结语言模型全部参数；adapter注入后要核验LoRA参数进入optimizer且原权重仍冻结。
- 原train()会强制语言模型eval。eval不禁止梯度，但会影响LoRA dropout的行为，需明确处理adapter训练态。
- 原checkpoint仅保存PromptLearner/CLS投影；新checkpoint必须同时保存LoRA权重、配置和所绑定bridge/base/cache身份。
- 沿用自定义inputs_embeds接口；若开启梯度检查点，特别检查冻结桥接的LoRA-only分支是否仍有有效梯度，不能仅以loss下降或进程退出0作为接入成功。
- 后续若增加排序损失，应从训练forward提取可导的Yes/No logits；现有score()带有no_grad，仅适合评测，不能直接作为训练loss的分数来源。
- 微批次/累积、梯度检查点或精度变化应对B/C同步；不把数值/资源配置变化混入算法效应。

### 5.4 单卡资源策略

历史Qwen桥接训练seed42峰值20,057,506,816bytes，约20.06GB（十进制）/18.68GiB。该值不是新LoRA显存测量。先尝试BF16、microbatch1×accum4并实测显存和吞吐；根据实际情况使用梯度检查点。

若BF16确实无法满足资源预算，再考虑4-bit NF4 QLoRA。量化会改变基座数值，必须增加相同量化精度但无LoRA的对照；不能把BF16旧结果与QLoRA直接差值归因于LoRA。

### 5.5 验收与下一层算法

主指标：同seed配对Image AUROC/AP；同时报告生成Macro-F1、召回、FPR/invalid、类别等权结果、可训练参数、峰值显存与吞吐。

val仍属于已经用于多轮选型的开发集；旧test456也不能重新称为未见盲测。若要支持泛化，应另设未参与调参的评估来源。

在LoRA基线与继续训练对照完成后，可单独尝试：

1. 首答案token加权，检验共同答案后缀是否稀释判别监督。
2. 同类别正负样本的Yes−No分数排序损失，与原答案CE组合，直接对齐图像异常排序目标。
3. 仅使用train数据构建困难样本采样/权重；不把val的错误样本移进训练。

以上均为待验证提议。只有完成后才能在简历增加“LoRA/SFT、梯度检查点、量化适配或任务损失优化”等实际实现的条目及真实增量，不填预期数字。

## 6. 方法资料

- Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, arXiv:2106.09685。
- Hugging Face PEFT官方LoRA配置、初始化、Custom models及adapter保存文档（本次查阅main文档；落地时以服务器安装版本API为准）。
- Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, arXiv:2305.14314；PEFT官方Quantization文档。

## 7. LoRA完成后的项目更新

B/C均完成seed42/43/44各500updates。C冻结桥接，仅更新Qwen语言q/v LoRA的3,833,856个参数；候选r8、alpha16、学习率1e-4，BF16基座与FP32 adapter，未使用量化或梯度检查点。三seed的AUROC/AP均高于自身起点；按val AUROC选择时，均值AUROC 0.925779→0.936983，生成Macro-F1 0.845672→0.861732。异常召回均值0.719551→0.764423，同时FPR由0升至0.019380。固定500步终点AUROC均值0.936452。D联合训练尚未实施。

技术栈现在可以增加 **PEFT、LoRA、参数高效领域SFT**。可在原简历第一条之后增加：

- 在Qwen3-VL-8B语言迁移基线上实现q/v LoRA领域SFT，冻结视觉前端及桥接、仅更新383万参数；设计已有起点/桥接延续/语言LoRA三seed配对对照，将自定义监督开发验证Image AUROC由92.58%提高至93.70%、生成Macro-F1由84.57%提高至86.17%，并分析召回与误报取舍；完成梯度与冻结审计、PEFT导出、独立进程重载和关闭adapter恢复起点核验。

面试时可说明：主要接入难点包括自定义inputs_embeds与冻结桥接下的梯度流、PEFT旧版本adapter精度继承、仅保存增量权重时绑定bridge/base/cache身份，以及区分额外训练预算与语言低秩适配的收益。约16.37GiB为本轮实测峰值allocated；没有测得吞吐加速。仍不能写成复杂推理增强、无监督官方成绩、未见测试泛化或已完成大模型预训练。
