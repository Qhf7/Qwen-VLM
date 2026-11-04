# Native 阶段协议与结果解读审计

初审日期：2026-11-03；状态更新：2026-11-04。范围：当前 `migration_native_train.py`、第一阶段语言桥接代码、`native_smoke_s42_v3` 及已完成的 `native_baseline`、`native_map_s42` 产物。两条native主运行均已完成val380评分与自由生成、保存重载和当前分支官方exact检查；完整指标及输入干预见 [PHASE2_RESULTS.md](PHASE2_RESULTS.md)。本文保留协议与baseline invalid的解释边界，不增加实验或修改执行脚本。

## 1. 当前对照识别什么

native 两臂共享同一官方 Qwen3-VL checkpoint、完整冻结 BF16 权重、官方视觉/merger/三路 DeepStack、同一图片和类别问题模板、同一 val380；map 臂额外训练原始 PromptLearner，把18个连续 token 插在 `vision_end` 后。Qwen 保持冻结/eval，训练参数仅来自 PromptLearner。代码证据：`migration_native_train.py:222–247, 286–311, 326–352, 597–600, 613–627`。

因此 baseline→map 的差值适合表述为：**“在该监督自定义划分中，为冻结原生 Qwen 增加经标签训练的 teacher-map 辅助分支后的整体增量。”** baseline 没有可训练参数；map 臂有107,416,504参数。差值同时包含监督适配、额外 token/位置和 map 输入，不能单独归因于定位信息。原始 PromptLearner 是9个共享 base prompts + 9个 map-adaptive prompts，并非18个都随输入变化；其源码 SHA 与本次 `run.json.code` 一致：`798397f0…1209f30`。源码证据：`AnomalyGPT_models.py:32–72`。

训练后 `zero_map` 仍保留学到的9个 base prompts 和卷积参数/bias；它是同一模型的输入干预。donor 使用 val 标签选同类别相反标签图片，保持当前 RGB，允许 donor 复用；shuffle 保持像素值多重集合。三者分别描述内容/分布/位置敏感性，不能直接证明定位正确，也不能视为独立训练的匹配臂。代码：`migration_native_train.py:639–677`；`migration_metrics.py:158–204`。

## 2. 第一阶段与 native 阶段的信息差

| 项目 | Phase 1 | Native |
|---|---|---|
| 视觉信息 | 冻结 ImageBind CLS经可训练投影1 token + teacher-map 18 tokens；无原生视觉 | 原始 RGB经官方视觉/DeepStack + teacher-map 18 tokens；无CLS投影 |
| 可训练参数 | 111,614,904 | map 107,416,504；baseline 0 |
| 梯度聚合 | batch4 × accum1 | microbatch1 × accum4 |
| 位置/模板 | 文本位置、`Image features:`插槽 | 原生视觉三轴 MRoPE、图像chat模板 |

两阶段的训练臂都使用同一 common cache，1500更新/6000次平衡图像曝光、lr5e-4、AdamW/OneCycle、每250步按val AUROC选择best；seed42第一阶段两臂与native map完整主运行的采样序列SHA已核对一致：`69cd5e37e25e2555ae73f03b7d6ebe05302b0d89473167a8512735c75abe641f`。native baseline不训练，采样序列为空。微批次聚合下，若答案token数不同，batch token均值与逐图token均值的权重并不等价；也不保证BF16逐位一致。代码：`migration_language.py:201–224, 228–294`；`migration_train.py:172–224`；`migration_native_train.py:166–174, 715–739, 765–778`。实测配置：`lm_vicuna_s42/run.json`、`lm_qwen_s42/run.json`、`native_map_s42/run.json`。

实际几何必须分两层描述：先把RGB直接缩放到224×224，再交给官方processor；完整native两臂均记录 **实际视觉tower输入256×256，grid=[1,16,16]，64视觉tokens**，teacher map仍为224×224。native两臂共享此处理；跨阶段还包含预处理/分辨率差异。出处：`native_baseline/run.json.geometry_audit`、`native_map_s42/run.json.geometry_audit`；代码：`migration_native_train.py:301–324`。

因此 Phase1→Native 的差值不能称为“只打开原生视觉”的因果收益；更稳妥的是比较两条完整接口路线。Phase1 的Vicuna/Qwen三seed配对结论也不能直接替代native当前单seed证据。

## 3. Canonical回答、长度和评分

训练仅监督固定 `Yes, there is an anomaly in the image.` / `No, there is no anomaly in the image.` 加EOS，前缀/视觉/辅助token全部mask。当前 Qwen 实测两类答案**均11 tokens（含EOS）**；此处没有Yes/No答案长度不对称。loss为11项token CE的均值，首个Yes/No项系数为1/11；后缀在teacher forcing下已见正确Yes/No。loss下降可以来自固定措辞/EOS更容易预测，不能据此断言图像排序能力提升，也不表示首词梯度只占1/11。代码：`migration_native_train.py:254–259, 294–297, 196–212, 355–369`；实测：`native_map_s42/run.json.initialization_audit.canonical_answer_ids`。

推理评分在**未喂入答案**时读取第一个回答位置的 `logit(Yes)-logit(No)`；不是整句概率和，不含句长奖励，也没有把label作为提示输入。label只在训练选目标、评估算指标及donor诊断中使用。分数代表两候选相对偏好，不是校准后的异常概率。AUROC/AP衡量排序；阈值0另给出决策，零分判No。greedy则在完整词表取argmax，解析只认开头Yes/No（允许规定的think前缀）；生成有效也只证明符合决策格式，不保证后续解释正确。代码：`migration_native_train.py:301–311, 371–394`；`migration_metrics.py:39–110`。

分数通道与自由生成须分别报告。已有第一阶段 `lm_qwen_s42/val_generated_predictions.json` 出现2/380分数决策/生成不一致（其中一项score=0.25却生成No），s44有1/380；phase1评分是batch前向，生成是逐图cached前向，代码路径形状不同，不能仅凭同一句模板假定一致。native生成时共享同次首词前向，但两候选评分和完整词表生成的含义仍不同。完整native两臂均已保存380条自由生成，须与评分指标及invalid率分别解读，见 `native_baseline/val_metrics.json`、`native_map_s42/val_metrics.json` 和 [PHASE2_RESULTS.md](PHASE2_RESULTS.md)。第一阶段证据：上述预测文件；`migration_language.py:297–335`。

## 4. 选择边界与当前已验证事实

common teacher在同一val上按macro pixel AUROC选best，native adapter又在val380上每250步按image AUROC选best；最终报告是开发验证集上的选择后表现。划分使用真实异常训练（train9985/val380/test456），不是官方无监督协议；native代码只实现val预测。单seed、已反复用于选择的val结果不足以给出稳定泛化或显著性结论。代码：`dec_big.py:204–209, 271–279`；`migration_native_train.py:120–162, 644–647, 765–778`；边界亦见 `migration_phase1_protocol.md`。整体AUROC还可能包含类别间分数偏移，应同时看已有per-category及category-macro，代码：`migration_metrics.py:116–155`。

完整主运行已完成：baseline step0的val AUROC为0.626342、生成macro F1为0.218069、invalid267/380；map臂完成1500更新，按val AUROC选中step500，AUROC为0.866391、AP为0.907728、生成macro F1为0.812690、invalid0/380。两臂val380评分与自由生成齐全，`complete.json.passed=true`，best重载分数最大误差0；初始及selected分支12项官方/manual同形状exact检查全通过，三类logits最大差均0。指标和干预详见 [PHASE2_RESULTS.md](PHASE2_RESULTS.md)；原始证据为两目录各自的`val_metrics.json`、`train_complete.json`、`auxiliary_cache_check.json`、`selected_auxiliary_cache_check.json`及`complete.json`。gate验证实现一致性，效果结论仍受上述单seed、监督开发验证集和对照定义限制。

此前的真实GPU smoke作为链路补充证据保留：4更新，val24/generation8；base梯度0.00312125、adaptive梯度范数和1.81666566、冻结Qwen梯度0；best重载分数最大误差0。初始及selected的三类官方/manual同形状logits全部BF16逐位相等；官方自身cached/full最大误差分别0.25/0.5，仅为诊断。出处：`native_smoke_s42_v3/gradient_check.json`、`train_complete.json`、两份cache-check及`complete.json`；smoke不并入完整val380效果统计。

## 5. 完整 baseline 的高 invalid：格式与内容须分开

完整val380 baseline候选分数AUROC为0.626342、AP为0.703170；生成macro F1为0.218069，invalid为267/380（70.2632%，正常129/172、异常138/208）。113条格式有效输出全部以Yes开头，没有以No开头的有效输出。出处：`native_baseline/val_metrics.json`、`val_generated_predictions.json`。原始预测SHA256：`404b79b725ca954df91b9de31f03369973935640b226155a20f09625c0431896`。

只读原文归类显示，267条invalid全部非空：249条以`Based`开头、18条以`In`开头。按首段内容作互斥归类如下（只解释输出类型，不改原解析或重算官方指标）：

| invalid内容类型 | 数量 | 正常 / 异常 |
|---|---:|---:|
| 首段明确否定异常，如`there does not appear to be any anomaly`、`there are no obvious anomalies`、`free of anomalies` | 78 | 47 / 31 |
| `Based ... here is an analysis ...`等说明性分析导语，尚未给出可按既定规则读取的二元首词 | 172 | 74 / 98 |
| `In the context of anomaly detection ...`定义或分析导语，未明确首段否定 | 17 | 8 / 9 |

当前问题只问`Is there any anomaly in the image?`，**没有要求以Yes/No开头或只回答Yes/No**（`migration_native_train.py:286–297`）。baseline也未接受canonical回答监督；map臂则以固定Yes/No模板训练。因此baseline的高invalid主要表现为开放式回答与严格首词解析不匹配，不能直接解释成267次视觉检测完全失败；map生成指标的增量会同时包含回答格式适配。78条已给出否定结论的正文仍按预定规则判invalid，其中31条真实异常还体现内容判断错误，格式问题也不能用来豁免检测问题。

输出配置确为`max_new_tokens=32`（`native_baseline/run.json.config`；生成循环见`migration_native_train.py:381–393`）。267条invalid中258条末尾没有句终标点，可见`missh`等断词和未结束的列表/句子，符合token预算截断的表现；但末尾无标点不是可靠停止原因，JSON也未保存token IDs、实际生成token数或stop reason，**无法严谨报告恰好撞到32token上限的条数**。无论继续生成后是否会出现Yes/No，只要仍有说明性正文前缀，当前锚定解析仍会判invalid，所以延长输出并不自动消除格式失配。

示例：正常`candle/Normal/0402.JPG`的score=3.75（两候选中偏Yes），但生成以`Based on the image provided, there does not appear to be any significant anomaly.`开头。它说明第一步两候选相对偏好不能替代完整词表自由生成后的正文判断。报告保留原候选分数与原generated/invalid指标；这份归类只是机制解释，不把它重新包装成另一套检测准确率。

审计源码根：`/Users/qiaohongfei/anomalygpt_remote/files`。原始PromptLearner只读镜像：`AnomalyGPT/followup_20261026/server_snapshot/code/model/AnomalyGPT_models.py`。当前trainer SHA256：`63a95e25e87a2c4e1f0b67ad811151d90a687bf9747d6ec1f6bdd7765948d2dc`；证据JSON位于本目录。
