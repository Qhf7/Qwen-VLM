# Jev 能力分析与项目应用研究

更新日期：2026-11-10。对象：**TypeSafe AI 发布的 Jev**。

> **第一阶段已完成：360/360 个正式配对判断，独立最终审计通过。** 固定 120 张开发图上，0/1/4 正常参考的 pooled AUROC 为 **82.54% / 87.19% / 88.85%**，4−0 为 +6.31pp；类别宏 AUROC 为 90.50% / 93.00% / 93.83%，类别宏配对区间均跨零。Brier 改善但 ECE 不单调；这是单支持集同域开发结果，不是新盲测、定位或 Agent 收益。详见 [正式实验报告](Jev_图像判断基线实验报告.md)。

完整任务见：[Jev 研究任务清单](Jev_研究任务清单.md)。

## 1. Jev 是什么，为什么能分类

Jev 根据输入内容、问题和判断标准，返回供程序使用的选择、概率、评分或矩形位置。分类并不要求先生成一段长文本；模型可以直接对有限候选进行评估。官方将其称为 System One，强调范围明确的判断。[S1][S4]

可以用以下**行为公式**理解：

\[
p(y\mid x,q,c)
\]

其中 x 是文本或图片证据，q 是问题，c 是类别描述或判断规则，y 是候选答案。**这解释输入输出关系，不是 Jev 已公开的内部架构。**

例如，同一张零件图片，配合“是否异常”“更像裂纹还是污染”“严重程度如何”，分别构成二分类、多类分类和有序评分。Choice 的选项及描述由请求定义，因而可测试新任务；这不代表所有任务都可靠，也不表示预训练时没有接触过相关知识。[S3]

TypeSafe 将训练方法称为 **RLCD（Reinforcement Learning for Calibrated Decisions，面向校准决策的强化学习）**。[S5] 当前已查阅的公开资料没有给出可完整复现的网络结构、参数量、训练数据和 RLCD 算法，因此不能断言 Jev 是“某种 Transformer 加分类头”，不能推断它一定单次前向，也不能假设可下载权重做 LoRA。厂商对架构和校准的说明不代替本域验证。

## 2. 四种判断类型

| 类型 | 输出与作用 | 文本例子 | 本项目候选用法 | 边界 |
|---|---|---|---|---|
| Choice | 单选、各选项概率、confidence | 工单属于退款、物流还是售后 | 缺陷类型；有限动作选择 | 一题单选；没有合适选项时仍会选择 |
| Noul | 是非问题，返回“是”的概率 | 维修记录是否提到划痕 | 原图或裁片是否存在可见异常 | 无独立 confidence 字段，概率仍可能错误 |
| Score | 按文字定义的有序等级评分，返回等级分布 | 报告证据是否充分 | 严重程度或可疑程度 | 不等于精确数值测量或通用回归 |
| Box | 一个目标的归一化矩形位置 | 不适用纯文本空间定位 | 一个缺陷候选框 | 每问题一个框，目标不存在也返回框；不是像素分割 |

官方定义见 [S3][S6][S7][S8]。

多标签应拆成多个 Noul，例如分别问是否存在裂纹、是否存在污染。单个 Choice 不适合表达两个缺陷同时存在。未知类别也不能只靠一个“其他”选项解决：官方建议额外问一个 Noul，判断是否有任何候选类别适用，再读取 Choice。[S3][S9]

Box 没有“未找到”答案，因此应配存在性 Noul，并由代码决定是否采用框。Jev 的 top/bottom/left/right 为 0–1，本项目使用 0–1000 整数 xyxy；需明确转换顺序、尺度并验证几何关系，不能静默修补。[S8]

## 3. 是否涵盖所有分类和判断，包括图片与文本

**支持很多任务，不等于能可靠解决所有任务。** “能写成选择题”与“模型具备充分证据和能力”不同。

| 任务 | 关键条件 |
|---|---|
| 判断记录是否提到裂纹 | 语言证据、否定、引用和歧义 |
| 判断明显表面缺损 | 图片清晰、缺陷定义明确 |
| 区分反光、阴影与细小裂纹 | 细节分辨与领域适应，需验证误报和漏报 |
| 判断内部不可见裂纹 | 图片没有必要证据，应引入其他检测信息 |
| 推断完整故障因果链 | 通常需要额外证据与多步验证 |

截至本次核查，官方当前版本 **jev-1.14.0** 发布于 **2026-10-14**，支持自然语言、JSON 对象/数组与 PNG/JPEG 图片。不原生接受音频或视频；转写、抽帧后输入属于额外流程，不等于原生时序理解。[S1]

官方图片与 Box 仍是实验功能，尚未达到厂商生产标准。需显式开启图片 header；每边 280–1568 像素，单文件上限 16 MiB。[S2] 本项目的 224×224 异常图不能原尺寸直接提交；放大只能满足接口尺寸，不能恢复已丢失细节。原图判断与热图判断也不是同一实验。

### 概率、confidence 与校准

概率 0.9 不保证某张图正确。校准是群体性质：大量预测中，预测约为 0.9 的事件发生频率应接近 90%；在本项目域中是否成立需实测。[S4][S10]

Choice/Score 的 confidence 概括分布集中程度，不是额外测得的正确率。模型可以高 confidence 地答错；Noul 没有独立 confidence 字段，不能伪造。AUROC、校准和阈值下误报/召回分别衡量不同性质，不能互相替代。

## 4. 在大模型与 Agent 系统中的应用

Jev 适合作为局部判别组件。主模型或程序组织整体任务，Jev 回答窄问题，代码执行工具并处理预算和异常。Jev 本身不执行工具，不等于完整 Agent。[S11]

| 环节 | 可以交给 Jev 的问题 | 外部系统责任 | 评测重点 |
|---|---|---|---|
| 模型路由 | 当前输入是否需要更强模型 | 调用、预算控制 | 质量净收益与成本 |
| 工具或技能选择 | 有限候选动作中哪一个适用 | 参数检查与执行 | 任务成功率，不能只看合法率 |
| RAG 筛选 | 资料是否相关、是否支持断言 | 检索与证据组织 | 相关性与事实支持分开评测 |
| 输出复核 | 结论是否有输入证据 | 重试、拒答或人工复核 | 误接受与误拒绝 |
| 记忆筛选 | 信息是否值得长期保留 | 保存、去重与更新 | 污染率与遗漏率 |
| 质量分级 | 输出是否达到明确标准 | 通过或修改 | 与标注的一致性 |

官方提供工具选择与编码 Agent 辅助示例，但不证明 Jev 在这些任务上都优于其他模型。[S12][S13] 同一请求中的问题独立评估，不会读取彼此答案；需要依赖上一轮结果时由代码串联。[S11]

## 5. 与现有 Qwen 判断分数的联系

现有 [score.py](../../industrial_vlm/src/industrial_vlm/score.py) 的 _score_image 在同一图像、提示与共同 JSON 前缀下计算 true/false 值 token 的条件对数似然：

\[
s(x)=\log P(\mathrm{true}\mid x,\mathrm{prefix})-\log P(\mathrm{false}\mid x,\mathrm{prefix})
\]

normalized_score 在两个候选间归一化。代码核对图像网格、共同前后缀与 token 边界，真实标签只用于事后评价。这说明生成模型也可以提供连续候选判断，但**不能据此推断 Jev 内部采用同一机制**。

相同 120 张同域开发验证图上的既有结果：

| 模型 | 验证 AUROC | 验证 AP |
|---|---:|---:|
| 原始量化 Qwen 底座 | 82.39% | 85.58% |
| RR | 91.49% | 93.01% |
| DD | 94.56% | 95.26% |
| RD | 92.01% | 93.54% |

来源：[多任务实验报告](../../industrial_vlm/reports/multitask_v2/实验报告.md)。DD 的 AUROC 点估计相对底座增加 12.17 个百分点，表示该开发集上的排序改善，不是准确率提高，也不证明跨域稳定收益。现有归一分数不是已校准的真实异常概率，以上也不是 Jev 结果。

## 6. 项目方向一：0/1/4-shot 正常参考图基线

| 条件 | 输入 | 参数更新 |
|---|---|---|
| 0-shot | 查询图与冻结的正常/异常定义 | 无 |
| 1-shot | 同一查询图与 1 张同类正常图 | 无 |
| 4-shot | 同一查询图与 4 张同类正常图 | 无 |

这是**正常参考上下文推理**，不是 4-shot LoRA。0-shot 仍有任务定义、类别等先验，不等于模型没有预训练知识。

现有 [winclip_b2.py](../../seg_cross_part/winclip_b2.py) 的 DinoRef.feats 使用冻结 DINO 提取 patch；build_memory 将正常图 patch 展平，score 用最近正常 patch 的余弦距离构造 32×32 异常图。

| 比较项 | DINO 正常 memory | 拟议 Jev 正常参考 |
|---|---|---|
| 支持形式 | patch embedding | 正常参考图片 |
| 机制 | 最近邻距离 | 模型结合问题进行判断 |
| 输出 | 空间异常图 | Noul 概率等 typed 输出 |
| 权重更新 | 当前代码无 | 本实验无 |

现有 DINO 从所评 split 的正常图抽参考并排除参考自身；本轮 Jev 使用训练正常支持图，协议不同需明确。参数 --prompts defect 会从所评 split 异常记录读取具体缺陷词，该配置不能称严格未知缺陷 zero-shot。跨 backbone 的分数差也不能全归因于 shots。

## 7. 项目方向二：候选区域语义复核

**提出候选区域与判断是否应该报告异常，是不同能力。** 现有外部 1183 异常图结果如下：

| 方法 | 完整异常分母平均 IoU | IoU≥0.5 比例 | 正常图出正框率 |
|---|---:|---:|---:|
| 无条件 teacher union-box | 0.224989 | 0.182587 | 1.000000 |
| Cstructured score 门控 teacher 框，三 seed 均值 | 0.210527 | 0.173852 | 0.635204 |
| S 生成框，三 seed 均值 | 0.069815 | 0.015779 | 见原报告逐 seed 结果 |

证据：[FINAL_REPORT.md](../../migration_20261103/advanced_20261104/FINAL_REPORT.md)。较好的 teacher 候选不等于可部署检测，无条件出框与误报仍待解决。

可研究：DINO/teacher 提候选 → 原图、裁片及正常参考图 → Jev Noul 判断是否有真实缺陷 → 代码保留、丢弃或人工复核。

必须共同评价完整异常召回、全异常定位、正常误报和成本。不能只计算复核通过子集的 IoU。当前证据尚不能说明 Jev 会带来收益。

## 8. 项目方向三：ROI 与模型升级路由

[tool_workflow.py](../../migration_20261103/advanced_20261104/tool_workflow.py) 已有有限动作 planner、parse_request、crop_bounds 与 build_final_inputs，支持实际 RGB 裁剪，固定裁剪面积为原图的 1/9。

但当前 380 张开发图，**380/380 次 planner 均选择 answer，真实 crop=0**；工具臂与重复原图控制逐图 logits、文本和 token 一致，漏检 202/208 张异常图。已完成的是工具协议与控制评测，不能声称获得自主裁剪收益。[结果证据](../../migration_20261103/advanced_20261104/FINAL_REPORT.md)

可让 Jev 在直接回答、检查候选裁片、升级模型、人工复核之间判断。Python 仍控制执行与预算。保留原图、固定裁剪、随机裁剪及其他路由控制。调用率提高不是收益，应比较固定成本下的任务结果或相近质量下的成本。预测“裁剪是否有用”也需要实际验证。

## 9. 项目方向四：结构正确之外的语义正确

[structured_geometry.py](../../migration_20261103/advanced_20261104/structured_geometry.py) 的 parse_structured 已检查结构、重复 key、字段、正常图空框、整数坐标和九宫格位置词一致性；localization_metrics 将无效、No、缺框计入完整异常分母的零分。

| 外部三 seed 均值 | Cstructured | S |
|---|---:|---:|
| 严格结构有效率 | 64.13% | 91.96% |
| 全异常框平均 IoU | 0.071108 | 0.069815 |

结构有效率提高没有转化为定位提高。格式、结构一致性与视觉事实应分别评价。[报告](../../migration_20261103/advanced_20261104/FINAL_REPORT.md)

Jev 可尝试复核“描述中的裂纹是否可见”“解释与区域是否一致”。JSON、几何和 IoU 继续由程序计算；模型复核器自身也要标注评测。

## 10. 视觉桥接 soft tokens 不能直接发给 Jev

[migration_language.py](../../migration_20261103/code/migration_language.py) 的 FrozenLanguageBridge.pack 将 CLS 与 224×224 异常图送入投影和 prompt learner，构造投影 CLS 加 18 个异常图 soft tokens，送入语言端 embedding。该迁移路线不使用 Qwen 原生视觉。

Jev 公开图片 API 不接收这套内部 embedding，因此不能直接提交 soft tokens、DINO latent 或缓存张量并称为等价接入。应提供原图或冻结裁片；转为文本摘要会改变可见信息，应单独报告。

## 11. 第一阶段冻结协议与数据条件

| 项目 | 约定 |
|---|---|
| 模型 | jev-1.14.0，记录响应实际版本，不用 latest |
| 主问题 | 固定一个只针对查询图的 Noul 异常存在性问题 |
| 分辨率 | EXIF 方向处理、RGB、等比缩放并填充至 448×448，无随机裁剪 |
| query | 已有 120 张同域开发验证图，0/1/4 组完全相同 |
| 支持 | 每类 4 张唯一训练正常图，1-shot 取首张；共 48 支持图 |
| 判断数 | 120×3=360 个目标判断 |
| 隔离 | 请求不含真实标签、答案、GT 框、异常面积、含标签路径或文件名 |
| 分数 | 真实 Noul 概率，不从生成文本猜概率 |
| 失败 | 明确保存原因，不能补零或静默丢弃 |

448×448 是冻结条件，不代表对细小缺陷最优；保存原始尺寸、缩放和填充。相同基础问题只改变正常支持上下文，参考图明确为已知正常，不能读取查询的具体缺陷标注。

### 数据审计结果

外层原始工程有 val_decision_120.jsonl：120 唯一来源，12 类各 5 正常和 5 异常。train_decision_decision.jsonl 有 1536 行、768 唯一来源，每来源重复 2 次；384 唯一正常图，每类 32 张。二者位于外层 industrial_vlm/prepared/multitask_v2/。发布副本未内置 prepared 原始清单或完整 VisA 数据集，本轮所需 168 张图已按需恢复至实验目录。

训练与验证 source_id、原始路径交集均为 0。外层 industrial_vlm/experiments/multitask_v2/source_byte_audit.json 有全部 120 验证和 768 训练来源历史 SHA；source_id 均为 visa_ 拼接字节 SHA，历史训练正常与验证字节 SHA 交集为 0。**历史审计不能代替当前原图复验。**

本轮固定支持选择：训练正常先按 source_id 去重，每类按固定种子和 source_id 的哈希排序取 4 张，1-shot 用首张。不根据结果挑选。冻结 ID、SHA、顺序、抽样算法。

原始清单指向服务器 /root/autodl-tmp/industrial_vlm/datasets/VisA/；现已从公开 Hugging Face 数据镜像 imaadd05/visa-anomaly-detection 的固定 commit **912ae1aa35ac0791e64524a42342850af1605363** 恢复本次所需原图。证据见 [recovery_audit.json](../../industrial_vlm/experiments/jev_baseline_20261110/recovery_audit.json)：**168/168 张校验成功，120 query + 48 正常支持，共 29,846,288 字节**；当前字节 SHA 均与历史记录匹配，query/support 的来源和字节交集均为 0。

恢复后的图片根目录为本发布副本 industrial_vlm/experiments/jev_baseline_20261110/data_images/VisA/。抽样规则固定为 sha256(“jev-support-v1|42|” + source_id) 升序，每类取 4 张，1-shot 使用首张。恢复过程未读取 confirmation manifest、未打开或评测 confirmation 图像；本轮转换后输入身份已保存在 frozen/materialized.json，并通过最终复验。

### 评价与校准

| 维度 | 指标 |
|---|---|
| 排序 | AUROC、非插值 AP、分类别结果 |
| 工作点 | Macro-F1、召回、FPR，区分固定 0.5 与开发选择阈值 |
| 概率 | Brier、明确分箱规则的 ECE、可靠性图 |
| 成本 | 延迟、usage、实际计费信息与调用量 |
| 完整性 | 计划数、成功数、失败数、逐图逐组记录、返回版本 |
| 比较 | 同 query 的配对差异与重采样区间 |

仅有 60 正常 query，5% FPR 约对应 3 张图，低误报指标分辨率有限。ECE 也受分箱和样本量影响，120 图校准只作开发诊断。保存原始概率，需要时用独立数据拟合校准，并在新留出集确认。

当前验证与既有外部结果已被查看，属于开发验证/回归，不得重新称盲测。最终方案需新留出确认。

## 12. ModelGo 渠道：0/1/4-shot 图片请求已通过预检

用户提供的 POST /v1/content_generation/tasks 只含 model 和 prompt，不能证明其支持 Jev 图像、typed 问题或 Noul 概率。

ModelGo 的 [TypeSafe System One 通用文档](https://modelgo.ai/docs/models/text/typesafe-systemone) 将入口定义为 **POST /v1/systemone**，但其关于 image assets、Box、Danger-Zone headers 和 multipart 的限制与用户当前模型卡不一致。[S14] 用户当前模型卡明确支持图像；实际接口检查进一步确认单图请求可用，因此该通用文档不能作为“当前渠道不支持图像”的充分依据。

2026-11-10 已完成真实网关预检，证据见 [gateway_probe.json](../../industrial_vlm/experiments/jev_baseline_20261110/gateway_probe.json) 和 [gateway_probe_references.json](../../industrial_vlm/experiments/jev_baseline_20261110/gateway_probe_references.json)。请求包含 assets 与图片 header，使用 448×448 合成图片，固定请求模型 jev-1.14.0：

| 预检条件 | 资产数 | HTTP/响应 | 当前结论 |
|---|---:|---|---|
| 0-shot 单图 | 1 | HTTP 200，合法 Noul，返回模型名匹配请求名 | 单图传输与 typed 概率返回已验证 |
| 1-shot 双图 | 2 | 首次 HTTP 502，后续重试 HTTP 200，合法 Noul | 双图请求已通过；保留首次 provider server error 记录 |
| 4-shot 五图 | 5 | HTTP 200，合法 Noul | 五图请求已通过 |

这些合成方块图片只用于接口检查，不来自工业 query，也不构成工业异常基线或模型准确性证据。另一次负例方块预检出现的 HTTP 502 仍保留，不据此计算准确率或推断视觉能力。少量成功请求也不证明长期服务稳定性，正式实验仍要保存失败与重试。响应中的 model 名称按 ModelGo 说明可能只是逻辑请求名回显，匹配请求名不等于独立证明上游实际权重版本；需在报告保留此溯源限制。

| 层次 | 当前能力 | 影响 |
|---|---|---|
| TypeSafe 官方 | 文本与实验性图像 | 模型层面可研究图像判断 |
| 当前 ModelGo 渠道 | 单图、双图、五图 assets 请求均返回合法 Noul | 0/1/4 图像请求与正式 360 个目标判断均已完成 |

ModelGo key 不能当作 TypeSafe 官方 key，也不应发送到其他域名。本次已按冻结协议完成 0/1/4-shot 共 360 个目标判断。接口能力、数据恢复和工业性能分别审计；之前的 502 保留为服务失败，不能直接归因为模型缺乏图像理解能力。

**本轮真实 AUROC、AP、工作点、校准、观察延迟和 token usage 已汇总到正式实验报告；没有核实货币账单，因此不估费用。** 凭证只由运行环境安全提供，不写入源码、配置、Markdown、日志、请求快照或 Git。

## 13. 研究顺序与完成标准

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 0/1/4-shot 图像判断 | 渠道与原图可用，冻结协议，完整 360 个目标判断，真实概率指标与失败报告 |
| 2 | 候选区域复核 | 固定候选，对照现有 scorer 门控，共同评价召回/误报/成本 |
| 3 | 动态工具路由 | 复用执行器，固定动作控制，报告预算与净收益 |
| 4 | 语义复核与拒答 | 分离规则校验，标注复核器错误，评价覆盖率与风险 |
| 5 | 本地算法研究 | 将评分、校准、拒答或路由落实为可训练目标，做匹配消融 |

前景在于判别、生成与代码执行分工。可研究“原图→视觉候选→窄判断→预算路由→Qwen解释/ROI/人工”的组合，但这些后续复核、路由和解释组合仍是待验证设计。更有价值的方向是可校准判别目标、域适应、拒答及成本约束路由，而不只是 API 集成。

## 14. 来源与代码索引

### 官方资料

- [S1] [Models](https://docs.typesafe.ai/models)：版本、发布日期和模态。
- [S2] [Images](https://docs.typesafe.ai/images)：实验状态、header、尺寸与文件限制。
- [S3] [Choice](https://docs.typesafe.ai/primitives/choice)：类别定义、单选与分布。
- [S4] [System One](https://docs.typesafe.ai/concepts/system-one)：定位与边界。
- [S5] [Machine Learning Primer](https://docs.typesafe.ai/introduction/machine-learning-primer)：RLCD 厂商说明。
- [S6] [Noul](https://docs.typesafe.ai/primitives/noul)：是非概率。
- [S7] [Score](https://docs.typesafe.ai/primitives/score)：有序等级评分。
- [S8] [Box](https://docs.typesafe.ai/primitives/box)：单框、坐标与存在性限制。
- [S9] [Companion Noul](https://docs.typesafe.ai/patterns/companion-noul)：类别适用性判断。
- [S10] [Confidence](https://docs.typesafe.ai/confidence)：分布与 confidence。
- [S11] [How to Build](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)：代码控制与独立判断。
- [S12] [Function Calling](https://docs.typesafe.ai/cookbooks/function_calling)：有限工具选择。
- [S13] [Coding Agents](https://docs.typesafe.ai/introduction/coding-agents)：Agent 辅助用法。
- [S14] [ModelGo System One](https://modelgo.ai/docs/models/text/typesafe-systemone)：渠道通用接口文档，其图片限制与当前模型卡和单图实测存在矛盾，应结合实际证据判断。

以上网页能力有版本性，后续记录实际请求日期、端点、协议与返回版本；厂商宣传不替代受控评测。

### 项目证据

| 文件 | 对应内容 |
|---|---|
| [score.py](../../industrial_vlm/src/industrial_vlm/score.py) | 两候选似然、连续分数和校准边界 |
| [多任务报告](../../industrial_vlm/reports/multitask_v2/实验报告.md) | 120 图与 82.39%/94.56% 开发 AUROC |
| [winclip_b2.py](../../seg_cross_part/winclip_b2.py) | DINO 正常记忆与参考排除 |
| [migration_language.py](../../migration_20261103/code/migration_language.py) | CLS 与 18 个 soft tokens |
| [tool_workflow.py](../../migration_20261103/advanced_20261104/tool_workflow.py) | planner、裁剪执行器 |
| [structured_geometry.py](../../migration_20261103/advanced_20261104/structured_geometry.py) | 结构、几何和完整分母 |
| [FINAL_REPORT.md](../../migration_20261103/advanced_20261104/FINAL_REPORT.md) | teacher、结构化定位与 crop0 |
| [FINAL_EVIDENCE_AUDIT.md](../../migration_20261103/advanced_20261104/FINAL_EVIDENCE_AUDIT.md) | 既有证据与复算边界 |

本文第 6 节的图像判断基线已完成，真实 Jev 结果见正式实验报告；候选复核、ROI 路由和语义复核仍待验证。已有 Qwen、DINO、teacher 与 ROI 数字用于定位研究问题，不构成这些后续 Jev 用法的收益证据。
