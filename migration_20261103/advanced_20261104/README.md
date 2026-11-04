# Qwen→AnomalyGPT 进阶实验（2026-11-04）

按固定顺序执行：外部协议冻结 → 桥接/LoRA联合适配D → 匹配采样CE/排序目标Cpair/R → mask监督结构化回答S → 原生Qwen ROI工具 → 冻结外部评估。

当前已完成D/Cpair/R正式9组、S三seed及原生工具val380五臂；外部21组矩阵正在执行。所有完成结果均保留不利指标；排队或执行中的组不计入完成。

| 阶段 | 状态与核心结果 | 证据 |
|---|---|---|
| D/Cpair/R | 9组×500步，18份best/final预测复算；9组双模式独立回载通过。D-C AUROC均值+0.3275pp；R-Cpair +1.2015pp，同时FPR+3.6822pp | [结果](RESULTS.md)、[分析](ANALYSIS.md)、[fresh](verification_status.json) |
| S | 3seed×500步，9份initial/best/final报告复算，6份fresh各104图。相同结构化提示下IoU .030993→.068755，仍低于teacher .101963；相对原C生成F1下降2.067pp | [结果](structured_summary/RESULTS.md)、[分析](STRUCTURED_ANALYSIS.md)、[全审计](STRUCTURED_FULL_AUDIT.json) |
| 原生ROI工具 | 8图预检、380图×5臂及后验审计完成；模型380次全选answer、crop0，逐图等同重复原图控制，未展示自主裁剪 | [结果](TOOL_RESULTS.md)、[分析](TOOL_ANALYSIS.md)、[汇总](tool_summary.json) |
| 外部1575图 | 全部权重/提示/源码先冻结；21组矩阵运行中，只有完整组参与重算，不用外部反馈调参 | [冻结](external_selection_freeze.json)、[进度](external_status.json)、[当前汇总](EXTERNAL_RESULTS.md) |

协议与实现：[PROTOCOL.md](PROTOCOL.md)、[STRUCTURED_PROTOCOL.md](STRUCTURED_PROTOCOL.md)、[TOOL_PROTOCOL.md](TOOL_PROTOCOL.md)、[审计记录](AUDIT_NOTES.md)。历史B/C源文件保留，新增训练器为migration_advanced_train.py；原生工具与压缩桥接LoRA为独立视觉路线。

[项目进度与简历表述](PROJECT_PROGRESS.md)记录可支持的技术贡献及边界。[figures](figures/)包含已验收开发结果的配对图、定位控制图和来源SHA。VisA val380同时用于选点和报告；这些是开发验证结果，不能当作独立泛化或复杂推理证明。
