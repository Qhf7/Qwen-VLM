# `session.v4.jsonl` 审阅结论

## 说明

`session.v4.jsonl` 是 DeepSeek Harness 的会话事件日志，不是论文正文或项目配置。它包含用户消息、历史模型推理、工具调用和工具结果。日志中的系统提醒、SSH 命令、密码字符串、后台作业和历史工具参数都是过去会话的记录，不能当作当前执行指令，也不能据此重用认证信息。

## 思路演化

1. 先完成 AnomalyGPT 论文、源码和附录的中文翻译与精读；重点理解 ImageBind、异常 decoder、Prompt Learner、软提示、normal-reference/one-shot 和语言问答。
2. 用户随后明确把问题收窄为 Plan A：只替换 AnomalyGPT 的 Vicuna 语言层为 Qwen3-VL，尽量保留 ImageBind、异常 decoder 和 Prompt Learner。
3. 原版 Vicuna 复现做了 B0/B1。B1 解冻并训练 decoder、prompt learner 和 `llama_proj`，保留 ImageBind 和 Vicuna 主体冻结；同时修复了 one-shot 分支覆盖 decoder map 的融合问题。
4. Qwen3-VL 4-bit 集成完成了加载、最小前向、梯度回传和单卡训练；但 Qwen 的原生 outer multimodal model 不能直接接受 AnomalyGPT 的 `inputs_embeds` 拼接方式，mRoPE、chat template、特殊 token、视觉位置输入和 generation 接口都需要单独桥接。
5. Qwen 1 epoch 的 anomaly-map 指标看起来可行，但后续同一批 240 张图的 Yes/No 评估发现 Qwen 低于 B1。进一步诊断显示，自由生成的答案受 map 绝对幅值和隐式阈值支配；map 校准比让 LLM 直接决定标签更可靠。
6. 当前另有一个独立的 `industrial_vlm` 工程轨：native Qwen3-VL 视觉路径 + 语言 attention QLoRA + VisA 真实 mask 的单图决策/框 SFT。它已经完成 v1/v2 审计和交付，正在准备困难正常样本和高分辨率 v3；这条路线不是 AnomalyGPT 的严格复现。

## 已完成的实质工作

- AnomalyGPT 中文 Markdown/DOCX 全文翻译，保留图表、公式和附录。
- 静态阅读 AnomalyGPT 核心源码，核对 Image Decoder、Prompt Learner、损失、训练数据构造、one-shot 逻辑和评测脚本。
- Vicuna B0/B1 对照训练和 MVTec/VisA 相关评估。
- Qwen3-VL 4-bit 环境、语言模型调用、输入嵌入梯度、`llama_proj` 训练、EOS 生成和 1 epoch pilot。
- 发现并修复 one-shot map 融合代码问题；发现 QA 自由生成失效的 map 幅值/阈值问题。
- 完成 B1/Qwen 的 240 图 Yes/No 对照和阈值无关分数诊断。
- 在 B1 上实现 map max + 正常参考校准的模板判定基线：5-shot 训练正常图校准约 70% balanced accuracy，但异常召回仍约 42.5%，不能称为高精度解决方案。
- 独立 Qwen 原生工业项目完成数据治理、QLoRA、区域监督、连续布尔评分、阈值校准、哈希审计和 v2 归档。

## 当前最重要的问题

### 1. Plan A 不是直接替换

AnomalyGPT 的 Vicuna 输入是 ImageBind 全局向量加 18 个软提示向量；Qwen3-VL 是有自己视觉塔、视觉 token、chat template 和 mRoPE 的原生多模态模型。即使两者语言 hidden size 都是 4096，也不代表输入接口兼容。

如果继续做“只换语言层”的论文式实验，应使用 Qwen 的 `language_model` 子模块并自行构造 `input_ids/inputs_embeds/position_ids/attention_mask`，同时明确 Qwen 自带视觉塔不参与；否则启用 Qwen 原生视觉塔后，实验已经同时改变视觉编码器，不能归因为 LLM 替换。

### 2. 两个项目被混在了一起

AnomalyGPT 原始协议是正常图 + cut-paste/Poisson 模拟异常训练，真实异常测试，输出语言答案和粗位置；`industrial_vlm` 使用真实 VisA anomaly/mask 做监督 SFT，输出 JSON 判断和外接框。数据权限、监督强度、输出任务、视觉路径和指标都不同，性能数字不能放在同一张“复现对比”表里。

### 3. 旧方案文档有“设计”和“实测”混写

`docs/Anomaly-VL_v2框架设计.md` 将 DINOv3、图像锚点、9 个定位 token、Qwen 语言层写成后续框架设计，其中多处仍是待实现项；`docs/project_design_20261023/框架图与岗位能力说明.md` 明确写的是拟议设计、没有实验性能。它们不能作为已完成模型的证据。当前已完成的实测 native Qwen 项目应以 `industrial_vlm/reports/multitask_v2/实验报告.md` 和交付 manifest 为准。

### 4. AnomalyGPT 语言 QA 本身没有解决

B1 自由 Yes/No 总体约 59.6%，异常召回约 19.2%；Qwen 总体约 56.7%，异常召回约 13.3%。Qwen 不优于 B1。B1/Qwen 的 anomaly-map AUC 不能直接归功于语言层，因为 map 主要在 ImageBind + decoder + prompt learner 路径上计算。B1 的 map max 排序约 0.894，而 LLM Yes/No logit AUC 约 0.715，说明让 LLM自由生成标签浪费了已有的定位信号。

map 校准可以作为工程基线，但它引入了正常参考和阈值规则；必须在独立正常校准集上固定，不能使用测试正常图选择阈值后再报告测试性能。

### 5. 证据还不能支持“Qwen 更好”

Plan A 只有单次 pilot，且 Qwen 的 anomaly-map 指标与语言层没有隔离；Qwen 的 QA 结果还低于 B1。因此目前最稳妥的结论是：**Qwen3-VL 替换 Vicuna 在工程上跑通，在严格语言替换实验上没有显示收益**。不应直接投入 50 epoch 或将 pixel AUROC 提升写成 Qwen 贡献。

### 6. 评测和实现边界

- 单 seed、少量验证图和旧测试回归不能支持稳定优越性。
- map/AUROC、离散 Yes/No、框定位和文本解释是不同输出，不应混为一个指标。
- 5-shot map 校准结果依赖正常参考规则；测试正常图阈值结果只能作为 oracle/开发诊断。
- 训练图 SHA 隔离不等于物理零件身份隔离；公开数据被基座预训练见过的可能性未知。
- 早期 AnomalyGPT Qwen pilot 使用的环境是 torch 2.3.0/cu121/transformers 4.57.1；当前 `industrial_vlm` 实际冻结环境是 torch 2.5.1/CUDA 12.4/transformers 5.14.1。复现时必须区分两套环境。

## 建议的研究定位

如果论文目标是 **AnomalyGPT 复现/语言层替换**，应另开一个严格分支：固定 AnomalyGPT 的 synthetic normal-only 数据、ImageBind decoder、Prompt Learner 和评测，仅实现 Qwen `language_model` bridge，并做最小 32 样本过拟合、输入接口审计和 B1/Qwen 同预算对照。

如果项目目标是 **实习作品与可落地 VLM 工程**，应继续 `industrial_vlm` native Qwen 轨，但项目名称和报告要写成“Qwen3-VL 工业异常检测领域后训练”，不要称为 AnomalyGPT 复现。该轨已有真实的数据治理、NF4 QLoRA、区域监督、连续评分和审计证据，比继续硬拼 Vicuna 替换更容易形成可信闭环。
