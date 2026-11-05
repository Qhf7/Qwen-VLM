# Qwen3-VL 多模态适配与参数高效后训练

本目录汇集本项目的代码、配置、实验结果和文档，复制自原工作目录，保留原文件。整理日期为 2026-11-05。本次整理只做归档、文档适配与核验，没有新增模型训练或推理。

## 从哪里开始

1. 阅读 [项目全程精读](docs/Qwen3-VL多模态适配与后训练优化_项目全程精读.md)，理解问题、架构、修复、训练及评测证据。文末含代码索引和面试问答。
2. 用 [成果与创新总结](docs/项目成果与创新总结.md) 查看数字与贡献边界，用 [迁移演进与训练清单](docs/Qwen迁移实验演进与训练清单.md) 区分历史实验。
3. 对照 [最终报告](migration_20261103/advanced_20261104/FINAL_REPORT.md) 与 [最终证据审计](migration_20261103/advanced_20261104/FINAL_EVIDENCE_AUDIT.md)，复核开发收益和外部负结果。
4. 通过 [代码与文件附录](docs/项目精读附录_代码与文件索引.md)、[源码配置表](docs/项目源码与配置清单.tsv) 和 [全包清单](DELIVERY_MANIFEST.tsv) 找到实际实现。
5. 简历准备可读 [项目经历](docs/大模型基座算法岗位_项目经历_整合版.md) 和 [140 题面试问答](docs/项目精读附录_面试问答.md)。

## 已得到的结果

主线采用含真实异常监督的自定义 VisA 划分：train 9,985、val 380（正常 172、异常 208）。val 用于选 checkpoint 和报告，以下数值均为 seed 42/43/44 的均值。

| 主线方案 | 开发 AUROC | 开发生成 Macro-F1 | 开发正常 FPR | MVTec 外部 AUROC |
|---|---:|---:|---:|---:|
| Vicuna，重新训练桥接、LLM 冻结 | 0.889684 | 0.816122 | 见 Phase1 报告 | 未运行 |
| A：Qwen，重新训练桥接、LLM 冻结 | 0.925779 | 0.845672 | 0.000000 | 0.708834 |
| C：语言 q/v LoRA | 0.936983 | 0.861732 | 0.019380 | 0.711363 |
| D：桥接低学习率＋语言 LoRA | 0.940258 | 0.866186 | 0.011628 | 0.714118 |
| Cpair：匹配 CE 控制 | 0.935334 | 0.860004 | 0.025194 | 0.709581 |
| R：匹配 CE＋ranking | 0.947348 | 0.871828 | 0.062016 | 0.703438 |

C 的可训练参数为 **3,833,856**（q/v，r=8、alpha=16）；D 额外更新桥接。C、D、Cpair/R 从 A 的同 seed 起点分别训练，不能读成逐级累计提升；R 的归因对照是 Cpair。只有 S 从 C 继续训练。首答案 Yes−No logit 差用于 AUROC/AP；自由生成答案用于 F1/Recall/FPR，两者分开报告。

冻结 MVTec 外部集含 1,575 图，7 模型×3 seed 共 **33,075** 条预测。外部生成 FPR 约 **61%–72%**，R 的开发排序收益未跨域复现。S 把结构有效率从 **0.641270 提高至 0.919577**，但异常框 IoU **0.071108→0.069815**；原生 ROI planner 在 380 图上全部选择 answer，crop=0。定位与自主工具收益尚未成立。

独立原生 Qwen QLoRA 分支保留原生视觉路径：768 张训练来源、1,536 次曝光、192 更新、seed42、val120；判断任务 DD 的 AUROC 从量化底座 **0.8239 提高至 0.9456**。该分支的模型、数据和预算与主线不同，详见 [独立实验报告](industrial_vlm/reports/multitask_v2/实验报告.md)，不能把两条路线的指标直接相减。

## 目录和主要入口

| 路径 | 用途 |
|---|---|
| [docs/](docs/) | 全程精读、成果、简历、代码附录、历史项目审计和图表 |
| [seg_cross_part/](seg_cross_part/) | Real-IAD 跨零件划分、DINO 分割、CLIP 对照、指令构建与评测 |
| [artifacts/](artifacts/) / [data/](data/) | 小型划分、manifest、统计、指令及历史结果 |
| [AnomalyGPT/](AnomalyGPT/) | 必要上游代码、复现指南、早期迁移与运行时快照 |
| [repair_20261103/](repair_20261103/) | mask、几何、特征轴修复，BigDec 训练与复评 |
| [migration_20261103/code/](migration_20261103/code/) | 视觉缓存、语言桥接、原生视觉对照、定位与推理 |
| [migration_20261103/lora_20261104/](migration_20261103/lora_20261104/) | B/C LoRA 训练、dtype/梯度/保存重载审计 |
| [migration_20261103/advanced_20261104/](migration_20261103/advanced_20261104/) | D/Cpair/R/S、工具、冻结外评与原始轻量预测 |
| [industrial_vlm/](industrial_vlm/) | 原生 Qwen NF4 QLoRA、三任务消融、评分和单图推理 |

日期目录是历史实验批次。重复的 runtime、配置和 tokenizer 属于当时的执行/保存快照；源码与 checkpoint 身份以对应 run 的 source manifest、SHA 和协议为准。

## 环境与运行准备

```bash
git clone https://github.com/Qhf7/VLM_Ali.git
cd VLM_Ali/Qwen-VLM

# 原生 Qwen 路线；在独立 CUDA 环境执行
python -m pip install -e ./industrial_vlm

# 已具备依赖时，检查评分逻辑，不加载模型权重
(cd industrial_vlm && PYTHONPATH=src python ops/test_score.py)
```

原生分支要求 Python≥3.10 及支持 BF16 的 CUDA；实际依赖见 [requirements-runtime.txt](industrial_vlm/requirements-runtime.txt)。历史 AnomalyGPT、当前语言迁移与原生 Qwen 的环境分别记录在各阶段协议中，不应混用后假定结果等价。

准备数据、底座及训练权重后，按路线运行：

- 原生 QLoRA：读 [工程 README](industrial_vlm/README.md)、[数据协议](industrial_vlm/DATA_PROTOCOL.md) 与 [多任务实验报告](industrial_vlm/reports/multitask_v2/实验报告.md)。
- 语言迁移：读 [Phase1 协议](migration_20261103/PHASE1_PROTOCOL.md)、[LoRA 协议](migration_20261103/lora_20261104/PROTOCOL.md) 和 [进阶协议](migration_20261103/advanced_20261104/PROTOCOL.md)。
- 单图推理：读 [语言迁移入口](migration_20261103/INFERENCE_USAGE.md) 或 [student ROI 入口](migration_20261103/student_roi_20261104/INFERENCE_USAGE.md)，选择结构匹配的 checkpoint。
- 早期分割：读 [项目说明](docs/项目说明.md) 与 [原始说明](docs/跨零件异常分割_原始说明.md)。

历史命令中的 `/root/autodl-tmp/...` 是当时的服务器路径，须根据本机资产映射准备运行目录。训练权重、原始图片和特征缓存没有包含在此包中，因此克隆仓库后需要另行准备资产才能训练或推理。不要改写旧 run、SHA 或 source manifest 来伪装为新运行。

## 交付核验

[ASSETS.md](ASSETS.md) 说明资产来源和省略范围；[THIRD_PARTY.md](THIRD_PARTY.md) 说明上游许可；[DELIVERY_AUDIT.json](DELIVERY_AUDIT.json) 记录复制完整性、源码身份、敏感信息和链接核验。[DELIVERY_MANIFEST.tsv](DELIVERY_MANIFEST.tsv) 包含除清单自身之外的每个交付文件及 SHA256，可在本目录用以下只读命令检查：

```bash
python - <<'PYVERIFY'
import csv, hashlib
from pathlib import Path
rows = list(csv.DictReader(Path('DELIVERY_MANIFEST.tsv').open(), delimiter='\t'))
bad = [r['path'] for r in rows if not Path(r['path']).is_file()
       or hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() != r['sha256']]
assert not bad, bad
print(f'PASS: {len(rows)} files')
PYVERIFY
```

原始对话与认证材料、原始数据集、权重/adapter tensor、特征缓存、重复压缩包及独立论文阅读库均未随包发布。相关论文的必要引用、AnomalyGPT/ImageBind 运行时及其许可仍保留；本项目的精读和实验审计文档属于交付内容。
