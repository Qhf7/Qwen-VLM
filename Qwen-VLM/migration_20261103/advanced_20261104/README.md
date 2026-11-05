# Qwen→AnomalyGPT 进阶实验（2026-11-04）

本轮已按顺序完成：外部协议冻结 → 桥接/语言LoRA联合适配D → 匹配CE/排序目标Cpair/R → mask监督结构化回答S → 原生Qwen ROI工具五臂 → 冻结外部评估。**21/21组外部运行完整，33075条原始预测复算通过。**

开发集提高没有转化为全面的外部收益。D的开发排序/F1/误报取舍较好；R的排序增益在外部未复现；S提高结构一致性但定位仍弱；原生工具模型没有选择裁剪。完整结论见[最终报告](FINAL_REPORT.md)。

| 阶段 | 完成情况与结果 | 证据 |
|---|---|---|
| D/Cpair/R | 9组×500步/2000曝光；18份best/final预测与18份独立回载报告通过。开发D−C AUROC +0.3275pp；R−Cpair +1.2015pp，同时FPR +3.6822pp | [结果](RESULTS.md)、[分析](ANALYSIS.md)、[回载](verification_status.json) |
| S | 三seed，9份预测复算、6份回载各104图。开发框IoU .030993→.068755，仍低于teacher .101963；相对原二分类C，生成F1下降2.0672pp | [结果](structured_summary/RESULTS.md)、[分析](STRUCTURED_ANALYSIS.md)、[审计](STRUCTURED_FULL_AUDIT.json) |
| 原生ROI工具 | 8图预检、380图×5臂完成。380次planner全选answer、crop0；模型臂等同重复原图控制 | [结果](TOOL_RESULTS.md)、[分析](TOOL_ANALYSIS.md) |
| 外部MVTec | 1575图、15类、7模型×3seed。D−C AUROC +0.2754pp/F1 −0.6024pp；R−Cpair AUROC −0.6143pp/FPR +5.6973pp | [结果](EXTERNAL_RESULTS.md)、[分析](EXTERNAL_ANALYSIS.md)、[冻结](external_selection_freeze.json) |
| 外部结构化与固定框 | S−Cstructured AUROC +1.8747pp，但框IoU .071108→.069815；teacher框 .224989，C门控teacher .210527 | [结构诊断](EXTERNAL_STRUCTURED_DIAGNOSTICS.md)、[固定框控制](external_teacher_control.md)、[控制审计](EXTERNAL_TEACHER_CONTROL_AUDIT.md) |

[最终证据审计](FINAL_EVIDENCE_AUDIT.md)核对原始预测、数据/权重身份声明、源码及生成策略绑定；服务器权重核验与本机result-only复算分别记录。[项目经历与简历](PROJECT_PROGRESS.md)提供可支持的技术表述。[开发图](figures/)与[外部图](external_figures/)均附来源SHA。

数据为含真实异常监督的自定义VisA train9985/val380；val用于选点和报告。外部为MVTec官方test扣除项目历史150图pilot后的1575图，公开数据的基座预训练暴露未知。外部结果没有用于反向选择权重、提示、阈值或超参。三seed是在同一批图上的训练随机性重复。

协议：[训练](PROTOCOL.md)、[结构化](STRUCTURED_PROTOCOL.md)、[工具](TOOL_PROTOCOL.md)、[外部](external_protocol.json)。原生工具使用完整Qwen视觉，是与压缩桥接LoRA分开的路线，本轮未做其外部评估。仓库保留代码、配置与轻量证据；权重、缓存和数据留在服务器。
