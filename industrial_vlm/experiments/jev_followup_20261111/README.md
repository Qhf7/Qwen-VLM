# Jev 服务器后续研究

运行服务器：`autodl-container-132444827f-9c594038`。
远端目录：`/root/autodl-tmp/industrial_vlm/experiments/jev_followup_20261111`。

本目录是独立后续实验，不覆盖 `jev_baseline_20261110`。所有模型、API、测试与统计执行均在服务器，本地仅编写、同步和查看产物。凭证不落盘；confirmation在最终系统冻结前保持隔离，阶段13已按一次性流程完成评价。

**当前阶段01—13已全部完成。** 总报告为 [CONTINUOUS_RESEARCH_COMPLETION.md](CONTINUOUS_RESEARCH_COMPLETION.md)，包含每阶段结果、分析、负结果、统计区间、图表及SHA证据；持续日志见 [PROGRESS.md](PROGRESS.md)。

## 阶段与证据

| 阶段 | 执行/评价 | 证据 |
|---|---|---|
| 01 | 120query+48支持原图SHA | stages/01_source_audit.json |
| 02 | Teacher缓存来源、训练暴露与完整SHA | stages/02_teacher_alignment.json |
| 03 | union95固定候选、448图像和几何 | stages/03_protocol.json、03_candidates.json |
| 04 | ROI与重复原图240配对判断 | runs/modelgo_roi；stages/04_prediction_freeze.json |
| 05 | 完整异常分母、正常误报、历史系统对照 | REPORT.md、stages/05_evaluation.json |
| 06 | 预算路由离线回放 | ROUTING_REPORT.md、stages/06_routing_results.json |
| 07 | 开发集交叉拟合校准/拒答诊断 | CALIBRATION_REPORT.md、stages/07_calibration_results.json |
| 08 | 新渲染共同0shot、种子43/44嵌套1/4shot | stage08_support（独立600目标） |
| 09 | 同448PNG的Qwen base/DD | stage09_qwen_matched（独立240推理） |

连续记录见 `PROGRESS.md`、`pipeline_events.jsonl` 和 `status.json`；每个正式运行有合同、started intent、结果和失败日志。服务器是运行状态的事实来源，本地同步副本可能滞后。

## 主要限制

- 120张图是已看开发集；Teacher训练暴露96，未见24仅含2张正常图。不能称独立盲测。
- union95有60张整图候选。ROI与duplicate在这些图上像素相同，差异不属于裁剪收益。
- 服务器Pillow12.3与第一阶段10.3生成的图像像素有差异。旧Jev分数仅历史对照；新实验共享相同新图像。
- 区域Noul不作整图概率校准；stage07只用既有全图分数，且不替代独立校准和confirmation。
- 预算回放不等于线上Agent执行，不会抹去已采集240响应的实际成本。

## 实际运行源码

Stage04—07使用的 `code/roi_api.py` SHA256为 `4f7b5ed1f703b3751f9fd4bb8c86131d684cec0bed3a9011f5108621e7aa9e01`，与服务器冻结合同一致。较晚增加hostname入口检查的版本保存为 `code/revisions/roi_api_postfreeze_d11dd9be.py`，没有用于该正式运行；它只增加socket导入、hostname常量和run入口校验，不改变请求或统计。不得用较晚版本覆盖旧合同后继续运行。

后续执行器自行核对目标hostname，并绑定真实传输器SHA。依赖分别沿用teacher/migration或原生Qwen环境，不合并升级。

## 持续研究阶段12—13

阶段08支持600与阶段10分辨率120已完整，阶段11联合报告保留为当时冻结快照，不随新研究覆盖。阶段08仅3项HTTP502耗尽使用独立运维修订，旧cap3账本/status不改，所有下游显式引用stage08_rescue/completion_certificate.json。

阶段12按早于结果的规则冻结4参考/seed43/448，使用互斥calibration120和heldout120，正式240完整、251started/11纯502/unknown0。预测冻结后只cal120拟合固定logistic，再评价heldout；独立1550项通过。结果见stage12/run/REPORT.md及独立补充。nativeval remainder不称blind。

阶段13主calibrated/次raw在阶段12首API之前声明，不按heldout表现切换。完整stage12后最终bundle冻结，42项审计通过，再一次性开启原固定240确认来源；1517项input-only审计通过。推理不读取protectedlabels/旧含标签文件，全部预测冻结后才最终评价，GT mask全程不读取。阶段13受global720/target3、累计失败30停止新提交、unknown永久硬停保护。

研究结束后新增CONTINUOUS_RESEARCH_COMPLETION.md汇总全阶段，保留早期报告和所有负结果。不将多臂API目标数当唯一图像数，不将Jev推理时few-shot参考当参数微调，不将Qwen相同PNG比较称同FLOPs。

阶段13最终已完成240/240、240started/0failed/0unknown，独立548项通过。主系统AUROC90.18%、Recall85%、FPR20.83%；校准比raw少5误报但多4漏检，结果不用于重新调参。正式全阶段1440目标/1588started/148明确失败/unknown0，对应600唯一query；Qwen240判断/480forward单列。确认集现已查看，后续新假设采用新协议与新留出数据。
