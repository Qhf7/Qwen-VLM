# Qwen3-VL → AnomalyGPT 迁移与后训练

本目录为 2026-11-03/04 修正视觉管线后的实验。模型、缓存和训练图片需自行准备，Git 中的小型 JSON/JSONL 是证据与配置，不能代替权重。

## 已完成结果

- [总结果](ALL_RESULTS.md)、[证据审计](FINAL_EVIDENCE_AUDIT.md)。
- [语言骨干三 seed 配对](PHASE1_RESULTS.md)：只改变语言骨干及必要分词器/模板，Image AUROC 0.889684→0.925779。
- [原生多模态接口](PHASE2_RESULTS.md)、[定位与蒸馏](PHASE3_5_RESULTS.md)、[ROI 对照](PHASE4_RESULTS.md)。
- [学生 ROI 结果](student_roi_20261104/STUDENT_ROI_RESULTS.md)、[单图推理](student_roi_20261104/INFERENCE_USAGE.md)、[失败诊断](student_roi_20261104/failure_analysis_224/REPORT.md)。
- [项目经历与下一阶段 LoRA 方案](PROJECT_EXPERIENCE_AND_LORA_PLAN.md)。
- [已完成语言LoRA三seed配对实验](lora_20261104/ANALYSIS.md)：B为桥接继续训练、C为冻结桥接的q/v LoRA；C best平均AUROC0.936983、生成Macro-F1 0.861732，同时报告召回/FPR、固定500步终点及独立重载核验。

数据为自定义有监督 train9985/val380，包含真实异常训练；这些属于开发验证结果。历史 test456 已参与开发，不能称为未见盲测。

## 源码和环境

`code/` 为迁移脚本快照；`student_roi_20261104/code/` 为后续学生 ROI 脚本；`../repair_20261103/code/` 为修正视觉管线及其依赖。源码 SHA 清单保存在各自目录。部分脚本的默认路径指向实验服务器，应通过环境变量或参数改为自己的路径。

服务器实际环境：Python 3.10、PyTorch 2.3.0+cu121、Transformers 4.57.1、PEFT 0.7.1，RTX4090D 24GB。上游 AnomalyGPT/ImageBind 代码、许可证及 PromptLearner 定义见 `../AnomalyGPT/followup_20261026/server_snapshot/`，来源说明见根目录 `THIRD_PARTY.md`。

迁移模块从 `ANOMALYGPT_REPO/code/code/model/AnomalyGPT_models.py` 加载 PromptLearner。复用时先准备 AnomalyGPT 上游布局，将 `code/*.py` 和 `repair_20261103/code/*.py` 置于运行根目录，设置 `ANOMALYGPT_REPO`、`QWEN3VL_MODEL`，准备官方 VisA、模型和 teacher 缓存。历史语言桥接训练入口：

```bash
python migration_train.py --backend qwen --seed 42 \
  --cache /absolute/path/teacher_cache.pt \
  --out /absolute/path/new_run
```

完整协议和缓存身份见 [PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md) 与每个 `lm_qwen_s*/run.json`。以上命令用于新训练；历史 checkpoint 与缓存不随仓库上传。
