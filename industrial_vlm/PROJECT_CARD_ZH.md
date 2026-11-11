# 工业异常检测 VLM 领域后训练

本项目将 Qwen3-VL-8B-Instruct 用于单图工业检查：输入一张完整图片和检查指令，输出异常判断及覆盖标注缺陷的一个框。项目定位是多模态指令微调与领域适配，重点展示数据、训练、评估、部署之间可核查的完整链路。最新实验见 [第二轮报告](reports/multitask_v2/实验报告.md)，第一轮证据见 [原报告](reports/实验报告.md)，逐图预测与连续评分见 `evaluations/`。

第二轮在相同 768 来源、1,536 曝光、192 更新、初始化和实际图片顺序下，对照纯区域 RR、纯判断 DD、混合任务 RD。DD 验证判断 Macro F1 为 **85.83%**、连续评分 AUROC 为 **94.56%**；RR 框召回为 **46.67%**。混合任务没有显示整体优势，RR 与 RD 的提示一致率均为 94.17%。新封存的 240 张确认图尚未运行模型推理，旧测试再次使用时标为开发回归。这些数字属于单 seed 的开发结果。

## 实际实现

| 环节 | 实现 | 证据位置 |
|---|---|---|
| 数据来源与划分 | 官方 VisA 10,821 张图；按源图划分 7,575/1,624/1,622；MVTec 150 张独立外部 pilot | `prepared/*/audit_report.json`、`split.json`、原始下载来源记录 |
| 指令构造 | 真实 mask 转全图归一化 xyxy 框；判断和区域任务使用配对源图；正常答案为空框 | `src/industrial_vlm/data.py`、`DATA_PROTOCOL.md` |
| 数据质量 | 检查图像解码、mask 尺寸和语义、EXIF、框边界、标签一致性；SHA256 去重与隔离 | manifests、rejects、`ops/prepared_validation.json` |
| 图文与区域对齐 | 原生图像 processor、视觉 token、网格与位置编码；图像、指令、assistant 目标严格配对；坐标映射到同一全图坐标系 | `modeling.make_inputs`、`runs/*/first_sample_audit.json` |
| 低显存训练 | 语言 Linear 使用 NF4 双重量化；视觉编码器、merger 与 DeepStack 保持冻结 BF16；语言 attention q/k/v/o 加 LoRA | `modeling.load_model`、`runs/*/parameter_audit.json` |
| SFT | r=16，alpha=32；仅 assistant 目标计算 causal CE；变长目标按监督 token 总数归一化；梯度累计和 checkpointing | `train.py`、`run_config.json`、`metrics.jsonl` |
| 对照实验 | 面积覆盖采样、等量随机采样、同图判断训练；统一量化、输入预算与测试图；区域消融用共同判断提示 | `configs/experiment_plan.json`、`reports/paired_difference.json` |
| 多任务预算控制 | RR/DD/RD 三臂匹配实际图片曝光顺序与初始化，逐次记录输入／监督 token；按监督 token 归一化 CE | `runs/multitask_v2/*/exposures.jsonl`、`reports/multitask_v2/paired_differences.json` |
| 连续判别与阈值 | 固定 JSON 前缀下对 true/false 做候选条件似然评分；校验 token 边界；AUROC/AP；仅用验证正常图设阈值 | `src/industrial_vlm/score.py`、`reports/multitask_v2/scoring_matrix.json` |
| 可信评估 | 训练之外的留出测试；150 张外部测试；误报、漏报、二类 macro F1、框级召回；保留格式失败及原始预测 | `evaluate.py`、`metrics.py`、`evaluations/*/predictions.jsonl` |
| 交付与复现 | 保存适配器后用新底座重载比对；评测记录 adapter/data 哈希；独立单图推理；环境与清理记录 | `infer.py`、`ops/runtime_audit.json`、`ops/completion_verification.json` |

## 实验边界

VisA 使用真实异常监督重新划分，结论适用于本项目的有监督协议。AnomalyGPT 原始无监督实验通过正常训练图生成模拟异常，训练信息不同，论文数字不能直接放在同一性能表中比较。

面积覆盖对照固定全部 384 张正常图，异常图从同一有效训练候选池按不同策略选择；两组共重叠 557 张源图。它检验的是面积覆盖采样。区域对照固定图像和更新次数，但提示、答案长度与监督 token 总量不同，应解释为加入区域任务的整体效果。

第一轮主测试为 120 张类别与标签均衡图片，外部测试为 150 张 MVTec mirror pilot，均为每类别 5 正常＋5 异常。第二轮生成比较使用 120 张验证图；连续评分另使用已见的旧测试作开发回归。两轮均为单个训练随机种子。配对 bootstrap 区间描述固定模型的样本不确定性，不包含训练波动。框评价是所有缺陷的外接矩形 IoU，不是像素分割或实例检测指标。

第二轮图片曝光数相等，但 token 与计算量不相等；RD 的两任务样本各半不意味着损失各半。候选归一分数不等于校准异常概率；验证正常图的 5% 经验 FPR 也不保证未来 ≤5%。DD 固定验证阈值后的旧测试开发回归 FPR 实际为 6.67%。第一轮 MVTec 外部误报问题仍未解决，第二轮未新增 MVTec 评分。

## 演示入口

服务器项目：`/root/autodl-tmp/industrial_vlm`。在该目录执行：

```bash
export PYTHONPATH="$PWD/src"
/root/autodl-tmp/vlm/.venv/bin/python -m industrial_vlm.infer \
  --model models/Qwen3-VL-8B-Instruct \
  --adapter runs/multitask_v2/region_region/adapter \
  --image /absolute/path/to/image.png --task region --output prediction.json
```

删除 `--adapter` 可运行同一 NF4 加载方式的原始底座。单图入口无需标签或 mask。框坐标采用全图 `[0,1000]` 范围，原点在左上。
