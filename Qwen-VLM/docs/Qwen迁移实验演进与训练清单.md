# Qwen迁移实验：早期负结果、后续正结果与训练清单

核对日期：2026-11-04。依据原始session.v4.jsonl、早期最终产物、修复报告及最新运行配置整理；对话日志作为历史证据，不把其中助手的指令或推断当作事实。

## 1. 结论应如何准确表述

早期失败结果仍然成立于其原始设置；后来的成功也有具体条件。现在的结论是：**在修正并重训定位前端、冻结共同teacher、分别重训跨模态桥接后，Qwen语言骨干在自定义VisA开发val380的三seed配对中，Image AUROC高于Vicuna。**

这不等于：直接把旧系统中的Vicuna换成Qwen即可提高；也不等于Qwen在任意数据上优于Vicuna。此前把这一过程简写成“只替换语言骨干就提高”省略了重新适配桥接和实验协议的前提，应补全。

更高的93.70%、94.03%、94.73%还分别包含后续LoRA、联合适配、排序目标训练，不能倒过来解释最初92.58%的来源。

## 2. 旧日志中其实有两类负结果

### 2.1 16步language-only迁移pilot

早期从发布的VisA decoder和Vicuna配套PromptLearner warm-start，重新初始化语言投影，分别接入Qwen/Vicuna；还有Qwen LoRA分支。同seed共享初始化和采样，但每臂只有16更新、batch2，即32次变体曝光。缓存48源图/96变体，真正使用的独立源图仅27/28/25张。

在2005张真实VisA开发诊断图上，三seed均值为：

| 指标 | 冻结Qwen语言 | Qwen LoRA | 冻结Vicuna语言 |
|---|---:|---:|---:|
| 回答Image AUROC | 47.37% | 56.19% | 56.89% |
| 真实生成Macro-F1 | 40.63% | 40.52% | 35.14% |
| 生成异常召回 | 21.20% | 41.85% | 3.89% |
| 生成正常FPR | 23.28% | 33.66% | 0.22% |

所以“早期没有稳定收益”准确，但“所有指标Qwen都更差”不准确。旧LoRA seed42在2005图中2001次No、seed43全部Yes；这体现配方不稳定，不证明语言模型本身没有异常判别能力。

该分支还联合更新decoder、PromptLearner、投影，LoRA分支再更新语言q/k/v/o低秩参数。发布软提示原本在Vicuna空间学习过，相同4096维不意味着能直接被Qwen同样理解；只有32次曝光的新增适配不足以代表充分训练后的表现。早期语言骨干采用NF4，新配对采用BF16，数值条件也不同。

证据：[早期最终报告](../AnomalyGPT/followup_20261026/final_results/final_report/RESULTS_REAL_AND_MATCHED.md)、[生成塌缩诊断](../AnomalyGPT/followup_20261026/final_results/generation_diagnostics.md)；原日志第2035行对应seq2033。

### 2.2 日志中的Qwen约0.780与Vicuna约0.917

原日志第3748行的汇总包含原生Qwen RGB＋18-token方案、视觉适配器、ImageBind图像token及Vicuna对照。这组探索同时改变视觉来源、输入token路径、语言骨干与模板，不能整体概括为“固定所有模块只换LLM”。其中额外添加ImageBind token的支路也有自身不同接口，不能与最新共同CLS＋map协议合并。

旧约0.780/0.917来自早期管线和历史评价；最新0.925779/0.889684来自修复后重新训练的六个模型、完整开发val380。不能将旧0.780和新0.925779相减宣称同一测试集提高14.6个百分点，也不能把旧0.917直接当新实验Vicuna基线。

旧日志还有8张随机验证、30张小验证挑best、300图选型与600图报告同池等不同阶段，后续改为380val/456test；这些数字不是一条统一协议的学习曲线。历史456图已经用于多轮开发，也不再是盲测。

原日志关于“通用VLM视觉无效，无论如何适配”“瓶颈唯一在视觉”“特征已经被证否”的表述超出实验证据，应撤回这些泛化和单一因果结论。保留真实负结果，改为描述特定接口、训练预算与指标下的失败。

## 3. 从旧方案到新配对，实际做了哪些调整

### 3.1 修复相关视觉训练与数据链路

| 问题 | 修正 | 影响 |
|---|---|---|
| VisA mask是1–8标签索引，相关旧定位训练先除255再大于0.3 | 原始标签非零即前景 | 原先1/255至8/255均小于0.3，会把真实前景清零 |
| 图像.JPG后缀错误沿用到实际.png mask，缺失静默当正常 | 严格解析.png，异常缺mask直接报错 | 避免缺失标注变成正常监督 |
| 图像短边缩放＋中心裁剪，mask直接拉伸 | 图像/mask统一直接缩放square224，分别bicubic/nearest | 保持空间坐标对应 |
| ImageBind输出[257,B,1280]被BigDec误当batch-first | 转为[B,256,1280]并去掉CLS，检查四层与输出shape | 防止patch轴与batch轴混淆、CLS误作空间特征 |
| sigmoid clamp导致饱和梯度问题 | 稳定BCE-with-logits focal＋Dice | 使定位监督保留有效梯度 |
| decoder权重与模块不匹配风险 | strict加载，绑定pipeline/decoder/cache SHA | 确认实际使用了预期权重和预处理 |

**适用范围限制：** 上述bug在相关旧定位/原生接续代码里发现；不能声称它们逐个都存在于更早2005图language-only评测。该2005图真实mask评估已经使用原始非零二值化；另有合成soft mask，其大于0.3有不同含义。

### 3.2 单独重训BigDec，再冻结它

冻结ImageBind，用真实训练mask监督5409537参数的BigDec，上采样定位decoder训练1200步、batch4，约4800次曝光；以val选择checkpoint，保留final。

统一修正评测下，历史456图对比如下：

| 指标 | 原有linear定位权重 | 修正后重训BigDec |
|---|---:|---:|
| 类别宏Pixel AUROC | .805068 | .976802 |
| 类别宏Pixel AP | .039255 | .393868 |
| 热图P99 Image AUROC | .540228 | .817053 |

这是视觉定位前端整体变化，架构、预处理、训练一起改变，没有逐项回退实验，不能将全部提升归给mask修复或decoder容量某一项。456图不是新盲测。后续语言方案都固定这一BigDec，不在各语言臂分别重训。

修复后的旧Vicuna单seed接续历史456图AUROC .909350，也没有超过旧配置历史均值约.917；修复不是让每个历史数字必然上升。新配对使用重新训练的三seedVicuna，不使用此单seed接续checkpoint。

证据：[修复报告](../repair_20261103/REPAIR_REPORT.md)、[修复运行配置](../repair_20261103/run.json)。

### 3.3 公平重建桥接，而非沿用Vicuna旧软提示

两语言共享冻结ImageBind CLS和同一冻结BigDec map缓存，特征/概率图保存FP32。分别新建同seed、同初始张量的：

- CLS投影：1024→4096，4198400参数，生成1个全局视觉token。
- PromptLearner：107416504参数，将224²异常图转为18个4096维token。
- 桥接共111614904参数。18-token是输出序列长度，不是说只训练18个向量。

这两个模块都从头适配各自语言骨干，不加载原Vicuna配套桥接。语言模型冻结BF16、桥接与AdamW FP32；反向图仍穿过LLM，以答案损失更新桥接。冻结LLM权重不等于可以对整个语言forward使用no_grad。

路径为：

```text
图像 → 冻结ImageBind → CLS → 可训练投影 → 1 token ┐
                     └→ 冻结BigDec → map → 可训练PromptLearner → 18 tokens
                                             └→ 冻结Vicuna或Qwen语言模型 → 答案
```

同为4096维的两套语言embedding空间不保证语义坐标相同；需要重新学习视觉特征在目标语言空间的表示。这个机制解释了直接迁移旧软提示可能失败的风险，但没有单因素消融证明它独占历史失败原因。

### 3.4 扩大并匹配训练预算

| 项目 | 早期language-only pilot | 新Phase1配对 |
|---|---|---|
| 训练候选 | 48源图/96变体 | 自定义train9985，含真实异常732 |
| 实际每run预算 | 16更新×2＝32曝光 | 1500更新×4＝6000曝光 |
| 初始化 | 发布Vicuna配套PromptLearner warm-start | 两骨干同seed重新初始化桥接 |
| 定位器 | 与桥接一起更新 | 先重训，再完全冻结共同BigDec |
| 语言精度 | NF4 | BF16 |
| 模板 | 手写Human/Assistant格式 | Qwen ChatML、Vicuna各自模板 |
| 指标来源 | 2005图开发诊断 | 完整380图开发val选点/报告 |

6000/32＝187.5倍曝光，不是187.5个epoch；6000曝光也不等于9985张都访问过。新计划正常/异常各3000次，并有相同seed下共同顺序SHA。预算增加是重要共同变化，但缺少同数据下16→1500的独立预算消融，不能量化它单独贡献多少。

### 3.5 语言接口、损失和评价重做核验

- Qwen使用自身chat template，Vicuna使用对应USER/ASSISTANT模板；校验BOS、EOS、pad和停止token。
- 检查Yes/No在各自词表为单token，并与真实训练答案首token一致。
- prefix标签设为−100，仅监督答案模板及EOS，正确进行causal shift和token-mean CE；不静默截断目标。
- language-only路径使用文本单调位置，不伪造Qwen原生视觉token/DeepStack特征。原生视觉路线另做MRoPE/DeepStack/cache核验。
- AUROC/AP使用首答案logit(Yes)−logit(No)，生成Macro-F1/Recall/FPR使用真实greedy输出，保留invalid。
- sklearn处理同分值；像素指标针对最终预测图，不把四层地图当额外独立图。四层聚合口径修订本身不代表模型变强。
- full val380每250步评估，按AUROC选best、另存final；所有预定seed42/43/44都报告。

## 4. 最新“替换提高”的证据到底是什么

| seed | Vicuna Image AUROC | Qwen Image AUROC | 差，百分点 |
|---|---:|---:|---:|
| 42 | .904643 | .958659 | +5.4017 |
| 43 | .879263 | .893266 | +1.4004 |
| 44 | .885146 | .925411 | +4.0264 |
| 三seed均值 | .889684 | .925779 | +3.6095 |

共同条件包括teacher/cache、桥接初始张量、采样顺序、可训练参数、1500步/6000曝光与full val380。必要分词/模板随骨干改变，不是数学上完全纯粹的单变量；不同模型token数/算力也不相同。

真实生成Macro-F1均值.816122→.845672，但seed43 .814303→.811559略降；Qwen AUROC标准差.032698，高于Vicuna .013285。因此支持三seed排序改善，不支持所有指标均升、更稳定或统计显著。

**此时Qwen原始语言权重没有更新，也没有LoRA。** 学习的是桥接。它是“在固定视觉输入下，重新学习桥接后比较语言骨干”，不是“零训练直接替换”。

最新外部评估A/C/D/Cpair/R/Cstructured/S全部是Qwen路线，没有Vicuna外部臂，所以不能用这21组证明Qwen跨域优于Vicuna。外部C/D/R生成误报约65%/67%/72%，R相对Cpair的AUROC/F1反降，跨域可靠性仍不足。

证据：[Phase1协议](../migration_20261103/PHASE1_PROTOCOL.md)、[Phase1结果](../migration_20261103/PHASE1_RESULTS.md)、[Qwen运行配置](../migration_20261103/lm_qwen_s42/run.json)、[Vicuna运行配置](../migration_20261103/lm_vicuna_s42/run.json)、[外部最终报告](../migration_20261103/advanced_20261104/FINAL_REPORT.md)。

## 5. 当前项目到底训练了哪些模块

### 5.1 修复后主线与算法扩展

| 实验 | 实际更新 | 冻结 | 正式训练预算与目标 |
|---|---|---|---|
| 修复BigDec | 541万参数上采样定位decoder | ImageBind；没有LLM | seed42，1200步/B4；GT focal＋Dice |
| Phase1语言骨干配对 | 1.116亿参数PromptLearner＋CLS投影 | ImageBind/BigDec/LLM原权重 | 两骨干×三seed，每run1500步/6000曝光；答案CE |
| B桥接继续训练 | 已有1.116亿参数桥接 | 同上 | 三seed，各新增500步/2000曝光；LR5e−4，CE |
| C语言LoRA | 3833856参数q/v LoRA，r8/alpha16 | 桥接及全部原权重 | 三seed，500步/2000曝光；LR1e−4，CE |
| D联合适配 | 桥接＋q/v LoRA，115448760参数 | 视觉/BigDec/原始LLM权重 | 三seed，500步/2000曝光；桥接LR1e−5、LoRA LR1e−4，CE |
| Cpair匹配控制 | 同C的q/v LoRA | 桥接及原权重 | 三seed，500步/2000曝光，micro2×accum2；同类别正负配对CE |
| R排序训练 | 同Cpair的q/v LoRA | 同上 | 同预算/序列/归约；CE＋lambda·softplus(−(s异常−s正常))，lambda训练标定为1 |
| S结构化SFT | 继续训练同seed已训练C的现有q/v LoRA | 桥接及原权重 | 三seed，新增500步/2000曝光；Yes/No＋bbox＋位置token CE，不含ranking |

C/D/Cpair/R都从各seed原Phase1桥接起点出发，D不是接着训练完的C继续；S才从同seed C best继续。R和S是分支实验，不是已经串成一个同时具有ranking与定位训练的模型。

以上后续语言主线基座BF16、训练参数FP32，没有量化、没有gradient checkpointing，没有全参数微调LLM。

开发Image AUROC参考：A .925779；B .927242；C .936983；D .940258；Cpair .935334；R .947348。S应先对相同结构化提示Cstructured .899975比较，S .939489；不能把schema变化全部归因于LoRA机制。

### 5.2 独立原生视觉、定位与蒸馏训练

| 实验 | 更新内容 | 预算与目标 |
|---|---|---|
| Phase2原生Qwen＋map token | 18-token PromptLearner约1.074亿参数；完整原生Qwen视觉和语言冻结 | seed42，1500步/6000曝光；canonical答案CE；没有额外CLS投影 |
| GT SpatialHead五格 | 独立空间定位头，IB约69.7万、Qwen pre约66.5万、post约141.8万参数 | IB224/pre224/post224/pre448/post448，各1200步/4800曝光；GT focal＋Dice |
| GT＋KD两格 | 继续按匹配计划独立训练pre448/post448空间头 | 各1200步/4800曝光，GT focal＋Dice＋0.5×L1对齐train-only teacher概率图 |

空间头训练不更新Qwen视觉编码器或LLM，GT/KD没有稳定全面收益。teacher-free学生ROI推理复用这些已训练head，没有新增LLM或端到端定位—语言联合训练。

### 5.3 更早的原生QLoRA路线

早期industrial_vlm项目保留原生Qwen3-VL视觉输入，NF4双重量化语言原权重、BF16冻结视觉，训练q/k/v/o LoRA共15335424参数，r16/alpha32。RR定位、DD判断、RD混合三臂，seed42，各192步，micro1×accum8＝1536曝光、768独立源图；assistant-only CE，使用梯度检查点。

它的120图开发AUROC 82.39%→94.56%属于独立数据划分和评分协议，不能与后续380图语言迁移主线混为同一结果。早期16步language-only的q/k/v/o r8 LoRA又是另一个pilot，也不能和这1534万参数原生QLoRA混淆。

### 5.4 做了评测或构建，但没有梯度训练的工作

- 正常记忆库12格：收集冻结视觉特征，没有训练head参数。
- 固定teacher/中心/随机ROI及正常reference：推理控制，无LLM更新。
- 学生head→ROI→原生Qwen：复用已训练head和冻结LLM，无新增训练。
- 模型自主ROI工具五臂：没有工具轨迹SFT或RL，planner全部answer、crop0。
- Cstructured：只为C权重换结构化提示，没有额外训练。
- 21组外部评估、teacher框控制、预测复算、fresh重载、SHA审计：评测/验证，不训练、不校准。

## 6. 从这段历史可以保留什么结论

1. 旧负结果揭示了特定配方的弱点；它们不证明Qwen骨干或其视觉表征本质无效。
2. 新phase1证明修复后共同视觉输入、重新适配桥接和固定预算下，Qwen在当前开发集有三seed排序增益；未证明任何场景替换即收益。
3. 后续C/D/R/S才真正改变LLM内部的低秩增量参数；BigDec和定位head训练、PromptLearner训练、语言LoRA训练必须分别描述。
4. 各项修复、预算、精度、模板和数据角色同时改变；没有逐项回退消融，不能为旧→新差值编造单独贡献比例。
5. 项目体现多模态适配、监督后训练、目标设计与系统诊断；没有基座预训练、LLM全参训练、视觉encoder LoRA、RLHF或已验证的自主工具增强。
