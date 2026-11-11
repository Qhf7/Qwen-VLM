# 外部连续分数与五臂工具实验可复用审计

本审计于服务器对已保存产物执行 CPU 统计；未加载模型、未发 API、未打开 RGB 或 mask、未重新训练或选择阈值。

## 结论

- 已找到真正的连续分数：migration 外部 `predictions.jsonl` 包含 float `score`；不能仅凭文件名判断它只是生成结果。
- 共 7 个条件 × 3 个训练 seed × 1,575 图 = 33,075 行；独立外部图仍只有 1,575 张（392 正常、1,183 异常、15 类）。21 次模型评价使用完全相同的 source/path/label 顺序与记录的 image SHA。
- 已重算全部 21 份外部指标、5 份工具指标并与冻结结果核对。所有 prediction SHA 与历史汇总一致。
- 足够复用外部 AUROC/AP 和固定零阈值二分类诊断；尚不足以声称已完成外部校准三态或原生 DD／Jev 的同条件对照。
- 现有五臂为原生 Qwen3-VL-8B、448、seed42、val380；model_tool 的 380 次 planner 全为 answer，0 次 crop，其最终分数与生成文本逐图等于 duplicate_original。不能把它写成自主裁剪或 Jev 工具头已提升。

## 三种评分口径

| 产物 | 连续分数 | 决策规则／阈值来源 | 可复用范围 |
|---|---|---|---|
| migration MVTec residual1575 | 首答 token logit(Yes) − logit(No) | 严格 >0；等于0判No | 本分支外部排序与固定二分类 |
| native 5-arm val380 | 最终独立 prompt 首答 token logit(Yes) − logit(No) | 严格 >0 | 原生开发工具对照 |
| native boolean validation120 | JSON true/false 完整候选序列 log likelihood 差值 | validation 的60张正常图，normal-only order statistic，strict >，校准经验FPR5% | 仅对应的base/adapter/prompt/分辨率contract |
| native MVTec pilot150 | 仅 raw_text＋parsed；无连续score/logit字段 | 生成 JSON 决策 | 不能由解析布尔值补成连续分数 |

生成 FPR 使用首个 Yes/No 的解析结果，invalid按既定规则计错；score FPR 使用连续 score 的严格阈值。两列即使数值相同，也属于不同决策链。AUROC/AP由连续score计算，不依赖生成解析。native normal-only阈值只控制校准样本经验FPR，不保证未来或跨域FPR；不能将其直接套在migration logit差值或Jev概率上。

## 外部结果（三 seed 均值±样本标准差）

| 条件 | AUROC | AP | score FPR(>0) | generated FPR | generated Recall | 类别macro AUROC |
|---|---:|---:|---:|---:|---:|---:|
| A | 70.88±4.60% | 85.79±3.63% | 61.22±6.02% | 61.22±6.02% | 92.34±2.08% | 81.86±3.76% |
| C | 71.14±4.58% | 86.00±3.53% | 65.05±9.23% | 65.05±9.23% | 93.55±2.71% | 82.40±3.02% |
| D | 71.41±5.37% | 86.07±4.19% | 66.84±5.30% | 66.84±5.30% | 93.91±1.47% | 82.43±3.31% |
| Cpair | 70.96±4.44% | 85.87±3.49% | 65.90±6.89% | 65.90±6.89% | 93.52±1.78% | 82.16±3.36% |
| R | 70.34±3.07% | 85.68±2.48% | 71.60±5.91% | 71.60±5.91% | 95.86±1.44% | 82.69±2.05% |
| Cstructured | 69.91±3.17% | 85.51±1.66% | 63.52±7.11% | 63.18±6.79% | 93.10±3.12% | 78.65±2.66% |
| S | 71.78±2.26% | 86.56±1.09% | 61.48±3.78% | 61.48±3.78% | 92.11±1.60% | 81.74±1.42% |

seed 的差异表示训练复现波动，不是三批独立测试图；不得把三个seed行拼接成4,725个独立样本来缩小置信区间。以上外部集已被历史研究查看，不能重新标注为盲测集。

条件定义：A：validation-selected language migration bridge baseline; no additional training for this condition；C：frozen bridge + language q_proj/v_proj LoRA; CE；D：joint bridge + language LoRA; CE；Cpair：frozen bridge + LoRA; paired normal/anomaly sampling, CE；R：frozen bridge + LoRA; paired CE + ranking；Cstructured：same selected C checkpoint; structured instruction/output control, no extra training；S：structured supervision continuation on language LoRA。

## 五臂开发结果（val380）

| 条件 | AUROC | Recall(>0) | score FPR | 实际crop数 | 图像数／视觉token | 平均部署秒 |
|---|---:|---:|---:|---:|---:|---:|
| no_tool | 75.17% | 31.73% | 2.91% | 0 | [1]／[196] | 0.675 |
| duplicate_original | 74.15% | 2.88% | 0.00% | 0 | [2]／[392] | 0.999 |
| center_crop | 71.96% | 38.94% | 3.49% | 380 | [2]／[392] | 1.089 |
| random_crop | 71.60% | 37.98% | 11.05% | 380 | [2]／[392] | 1.095 |
| model_tool | 74.15% | 2.88% | 0.00% | 0 | [2]／[392] | 1.240 |

center/random crop 的位置是固定控制，不使用GT；GT geometry仅在推理后审计。原图固定448；crop宽高各1/3，面积1/9，最多一次调用。除no_tool外均给两张图（无crop时第二张为原图重复），视觉token匹配，文本token与时延未完全匹配。planner跨控制复用；固定控制部署成本不含不必要planner开销。

## 下一步及预算门禁

1. 保留现有21份冻结外部分数，仅生成可追溯的分析表；本轮不重新挑阈值、checkpoint或seed。
2. 外部三态需先在合适的独立开发／校准来源固定两个阈值及拒答损失，记录calibration身份与score contract，再对预注册外部集一次评价；同一外部标签不能同时调阈值和报告泛化。
3. Jev工具头需冻结动作空间、统一图像预算、固定下游分类器，再与no_tool／duplicate／center／random／Qwen planner逐图配对；现有Jev异常判定分数不能冒充工具动作日志。
4. 如复用val380原始图做新的开发工具对照，完整五臂至少需要380×5=1,900次最终判定（若全重跑），外加planner调用；如只补Jev planner，调用数按新contract单独列出。此为计算次数示意，非批准预算，也不是新独立盲测。
5. 当前审计新增API=0、GPU forward=0；没有真实调用方案与费率前不捏造费用或时长。

## 准确路径与访问边界

- 外部汇总：`/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104/external_summary.json`；manifest：`/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104/external_manifest.jsonl`。
- A/C路径：`migration_20261103/lora_20261104/arm_C_s{42,43,44}/external_mvtec_residual_v1{_A,}/predictions.jsonl`。
- D/Cpair/R路径：`advanced_20261104/{arm_D,Cpair,R}_s{42,43,44}/external_mvtec_residual_v1/predictions.jsonl`；Cstructured为`arm_S_s*/external_mvtec_residual_v1_initial_c`；S为同根`external_mvtec_residual_v1`。
- 五臂：`/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104/tool_val/{run.json,predictions.jsonl,audit.json}`；汇总`/root/autodl-tmp/AnomalyGPT/runs/migration_20261103/advanced_20261104/tool_summary.json`。
- scorer：`/root/autodl-tmp/AnomalyGPT/migration_language.py:297-301`；阈值与指标：`migration_metrics.py:59-74`；native正常阈值：`industrial_vlm/src/industrial_vlm/score.py:165-187`。
- 恢复初期曾误将base_regression当external，读取其meta/metrics和阈值元数据后识别为native test并停止；未读native test scores/predictions/RGB。本报告不将这些误读元数据纳入新的评价。
- 本轮不读取stage13确认标签、不改01—13、不重开封存native test逐样本结果；已查看过的历史confirmation不能作为新round2调参集。
- 全部路径、schema、SHA、21 run与五臂细项见同名JSON。
