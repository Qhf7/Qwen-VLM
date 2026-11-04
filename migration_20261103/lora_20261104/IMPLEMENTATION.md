# 语言 LoRA 实现与结果解读

本轮建立在修正视觉管线后的语言替换基线上，与仓库历史原生 Qwen NF4 QLoRA 和旧 AnomalyGPT 小预算 LoRA 实验区分。

## 模型与更新路径

固定 ImageBind/BigDec 产生的 CLS 与异常图缓存。CLS 由1024维投影为4096维，224×224异常图经 PromptLearner 生成18个4096维 token，与 CLS token 一起进入 Qwen 的文本接口。当前路线不使用 Qwen 原生视觉塔。

LoRA 更新为 `W_new = W_base + (alpha/r) B A`。C只在保留的语言模型内36层的 q_proj/v_proj 上注入，72个目标线性层，共 **3,833,856** 个可训练参数，约为保留语言基座及lm_head的 **0.0468%**。B训练原有桥接的 **111,614,904** 个参数。这是参数高效领域 SFT，不是大模型预训练。

服务器 PEFT 为0.7.1，注入后显式将可训练adapter转为FP32；原语言权重仍为BF16。优化器严格使用可训练参数集合。原桥接模型的 `train()` 强制语言模块eval，本轮明确使用 dropout=0，eval允许梯度计算，训练forward不在no_grad中。冻结桥接时，adapter本身仍能获得梯度。

## 工程证据

seed42两组固定训练样本预检各40步、160次曝光。它们只是工程门槛，不是效果验证。

- 零初始化adapter前后全部probe logits最大误差0。
- C的LoRA A首步梯度为0符合B矩阵零初始化；第二步A梯度非零，B梯度持续非零。
- 冻结原参数的版本计数不变且没有梯度；C的桥接张量SHA保持不变。
- 对可训练参数主动加扰动后再加载best，评分精确恢复。正式训练另保留final，并安排独立进程核验非零的已训练final adapter，避免只验证零初始化best。
- 同时导出PEFT的 `best_adapter`，核对导出张量与参与评分的adapter逐元素一致。若best选择step0，该导出对应起点，不能当作新增训练取得收益的产物。

## 如何阅读配对结果

A为每个seed既有的bridge best。B/C都新增500updates、2000次曝光，保持同一采样顺序、缓存、模板、精度、微批次与评分代码。比较C−A回答“在现有起点上是否改善”，比较C−B检验其结果是否优于只延长桥接训练的对照。

B/C候选学习率不同（5e-4与1e-4），参数量与计算路径也不同；不能宣称它们隔离了所有优化因素。新阶段采用microbatch1×accum4的每图token均值再平均，历史A训练采用batch4的全batch token均值；B/C共同采用新口径，A→B/C的差值不能全归因于单个组件。

连续评分使用首答案位置的Yes−No logit差，报告AUROC/AP与零阈值决策。自由生成另报Macro-F1、异常召回、正常FPR与invalid。AUROC改善不保证召回或生成F1同时改善。

best按val AUROC选择，允许step0；同时报告固定500步final。三seed逐seed配对，再报告均值和样本标准差。val380已经用于多轮开发，所有数字都是开发验证结果；没有新test查询。

## 资源与加载

`peak_memory_bytes` 为PyTorch全程峰值allocated，包含加载后评估和训练，不等于nvidia-smi整卡占用；表格转为GiB。`train_peak_memory_bytes`沿用全程high-water mark，是训练过程记录的上界，不用于宣称纯训练阶段的独立峰值。训练吞吐只统计forward/backward/optimizer阶段，不包含验证生成时间。

精确加载使用 `FrozenLanguageBridge` → 原seed bridge → `configure_arm`（先将adapter转FP32）→ `load_delta`。C的 `.pt` 绑定原bridge、基座与cache身份；仅加载adapter不能替代桥接。标准PEFT导出便于互操作，精确复现以本项目加载代码和独立进程核验为准。

固定Yes/No答案模板支持异常判别领域适配，不能据此证明模型具有更强的复杂缺陷解释或推理能力。只有真实完成并核验后，才在简历补充LoRA/SFT实现；性能提升数字以配对结果为准。
