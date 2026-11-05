# Qwen3-VL 工业异常检测：多模态适配与参数高效后训练

本仓库集中保存将 Qwen3-VL-8B 迁移到工业异常检测任务的代码、实验协议、结果和项目文档。完整项目位于 **[Qwen-VLM/](Qwen-VLM/)**，包括 AnomalyGPT 语言骨干迁移、LoRA/联合适配/排序与结构化训练、独立原生 Qwen QLoRA，以及早期跨零件分割实验。

固定 ImageBind/BigDec、重新训练桥接并冻结语言骨干时，三个随机种子的 VisA 开发验证 AUROC 从 Vicuna 的 **0.889684** 提高到 Qwen 的 **0.925779**；语言 q/v LoRA 达到 **0.936983**。这些是自定义有监督开发协议结果。冻结 MVTec 外部评估显示跨域误报仍高，不能据此声称已达到部署要求或公开基准 SOTA。

| 阅读入口 | 内容 |
|---|---|
| [项目目录与运行准备](Qwen-VLM/README.md) | 路线、代码结构、环境、复现入口 |
| [项目全程精读](Qwen-VLM/docs/Qwen3-VL多模态适配与后训练优化_项目全程精读.md) | 从问题、架构、训练到证据；附代码索引和面试问答 |
| [成果与创新总结](Qwen-VLM/docs/项目成果与创新总结.md) | 详细指标、贡献、限制及简历表述 |
| [最终实验报告](Qwen-VLM/migration_20261103/advanced_20261104/FINAL_REPORT.md) | D/Cpair/R/S、ROI 工具及 21 组外部评估 |
| [代码与文件附录](Qwen-VLM/docs/项目精读附录_代码与文件索引.md) | 按任务定位实现、快照与结果 |
| [面试问答](Qwen-VLM/docs/项目精读附录_面试问答.md) | 140 个问题与基于实际证据的回答 |
| [资产说明](Qwen-VLM/ASSETS.md) / [上游来源](Qwen-VLM/THIRD_PARTY.md) | 数据、模型获取与许可 |

仓库当前目录只保留本项目资料；独立论文精读库已从当前版本移除。本地原文件保留，历史提交不重写。模型权重、原始数据集、特征缓存及原始认证会话不上传，交付范围与 SHA256 见 [文件清单](Qwen-VLM/DELIVERY_MANIFEST.tsv) 和 [核验记录](Qwen-VLM/DELIVERY_AUDIT.json)。
