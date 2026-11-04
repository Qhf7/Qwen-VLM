# 工业异常检测与多模态领域适配

本仓库记录从跨零件异常分割、AnomalyGPT 语言迁移到原生 Qwen3-VL QLoRA 的代码、实验协议和结果。三个方向使用不同的数据与训练条件，结果分别解释。

| 路线 | 代码入口 | 已完成的工作 | 结果入口 |
|---|---|---|---|
| 原生 Qwen3-VL | [industrial_vlm](industrial_vlm/) | VisA 数据构建、语言层 NF4 QLoRA、三组任务消融、生成及连续评分、单图推理 | [第二轮实验报告](industrial_vlm/reports/multitask_v2/实验报告.md) |
| AnomalyGPT 语言迁移 | [followup_20261026](AnomalyGPT/followup_20261026/) | 保留 ImageBind/decoder/PromptLearner，比较冻结 Qwen、Qwen LoRA、Vicuna，三随机种子控制实验 | [最终实验结论](docs/strict_qwen_followup_20261026/最终实验结论.md) |
| 跨零件分割 | [seg_cross_part](seg_cross_part/) | Real-IAD 分组划分、DINOv2 分割、CLIP 语义对照、像素指标与 bootstrap | [项目说明](docs/项目说明.md)、[实验台账](docs/实验进度与版本记录.md) |

## 修正视觉管线后的迁移实验（2026-11-04）

[本轮迁移目录](migration_20261103/README.md)包含修正 mask、几何和特征布局后的源码、三 seed 配对实验、原生视觉适配、定位蒸馏、学生 ROI 推理及失败分析。固定 ImageBind/BigDec，训练桥接层并冻结语言骨干时，自定义监督开发验证 val380 的 Image AUROC 为 Vicuna **0.889684±0.013285**、Qwen **0.925779±0.032698**（均值±样本标准差）；生成 Macro-F1 为 **0.816122→0.845672**。

这与下文 2026-10-26 的历史小预算迁移实验采用不同管线和预算。当前语言 LoRA 延续实验将先比较 B（桥接继续训练）和 C（仅语言 LoRA）；协议与状态见[项目总结与方案](migration_20261103/PROJECT_EXPERIENCE_AND_LORA_PLAN.md)。新增 LoRA 结果在完成后单独报告。

## 原生 Qwen3-VL 的当前结果

底座为 Qwen3-VL-8B-Instruct，保留原生视觉输入路径，冻结视觉模块与底座参数，仅对语言注意力 q/k/v/o 投影注入 LoRA（r=16、alpha=32、dropout=0.05）。可训练参数为 **15,335,424**。

第二轮各组均使用 768 张训练来源图、1,536 次曝光、192 次优化更新，seed=42。同一份 120 张同域验证图包含 60 正常、60 异常，覆盖 12 类产品。

| 模型/训练任务 | AUROC | AP |
|---|---:|---:|
| NF4 量化底座 | 82.39% | 85.58% |
| RR：两次区域任务 | 91.49% | 93.01% |
| DD：两次判断任务 | **94.56%** | **95.26%** |
| RD：区域与判断各一次 | 92.01% | 93.54% |

连续分数为 `logP(true) - logP(false)`；DD 相对相同量化底座的 AUROC 提高 **12.17 个百分点**。DD 的生成式判断 Macro F1 为 **85.83%**，与按连续分数设阈值的 F1 属不同推理方式。

这些是单随机种子、定制有监督协议下的开发验证结果。训练使用真实异常标注；不能直接对照仅正常训练图的官方无监督基准。独立的 240 来源确认集尚未进行模型推理。

![原生 Qwen 验证结果](industrial_vlm/reports/multitask_v2/figures/validation_comparison.png)

AnomalyGPT 迁移实验使用另一套视觉路径和预算。2,005 张 VisA 诊断图上，三随机种子生成判断 Macro F1 为：冻结 Qwen 40.63±8.02%、Qwen LoRA 40.52±12.26%、Vicuna 35.14±2.29%；该小预算迁移实验尚未显示稳定的 LoRA 收益。

## 阅读入口

- [项目精读：从问题到代码、从样本到结论](docs/项目精读/00_导读与项目地图.md)：按 kelip 六站方法，分章解释三条路线、真实样本、张量形状、训练损失、指标及关键函数。
- [工作总结与简历详解](docs/工业异常检测多模态项目_工作总结与简历详解.md)：工作过程、结果口径、项目经历写法。
- [Qwen LoRA 实现步骤](docs/Qwen_LoRA_项目实现步骤详解.md)：数据、加载、注入、训练、梯度审计、保存与推理。
- [数据与监督协议](industrial_vlm/DATA_PROTOCOL.md)、[项目卡](industrial_vlm/PROJECT_CARD_ZH.md)。
- [原生 Qwen 评分比较](industrial_vlm/reports/multitask_v2/scoring_matrix.md)、[生成比较](industrial_vlm/reports/multitask_v2/validation_matrix.md)。
- [论文阅读索引](docs/论文与代码阅读索引_20261023.md)、[上游代码来源](THIRD_PARTY.md)。
- [原跨零件项目 README](docs/跨零件异常分割_原始说明.md)。

## 安装与验证

```bash
git clone https://github.com/Qhf7/VLM_Ali.git
cd VLM_Ali

# 在具备 CUDA 的 Linux 环境安装原生 Qwen 实验依赖
python -m pip install -e ./industrial_vlm

# 已有依赖环境中的 CPU 检查；不加载模型权重
(cd industrial_vlm && PYTHONPATH=src python ops/test_score.py)

# 分割指标检查另需 requirements-s0.txt 中的依赖
python -m pip install -r requirements-s0.txt
python -m seg_cross_part.check_metrics
```

原生训练要求支持 BF16 的 CUDA 环境。实际服务器依赖版本见 [requirements-runtime.txt](industrial_vlm/requirements-runtime.txt)；旧 AnomalyGPT 运行环境和原生 Qwen 环境分别记录，不能混装后假定等价。

## 准备数据与运行

模型权重、适配器、原始图片、特征缓存和重复压缩快照保存在本地或训练服务器。仓库包含源代码、历史配置、协议、汇总指标及选定的训练/评估元数据，详见 [同步说明](docs/仓库同步说明.md)。

原生路线先从官方来源准备 VisA，并将 Qwen3-VL-8B-Instruct 底座放在本地模型目录，再从 `industrial_vlm` 目录执行：

```bash
export PYTHONPATH="$PWD/src"
python -m industrial_vlm.data --data-root datasets/VisA --output prepared/visa \
  --train-size 768 --eval-size 120 --seed 42
python -m industrial_vlm.infer --model models/Qwen3-VL-8B-Instruct \
  --adapter runs/multitask_v2/decision_decision/adapter \
  --image /absolute/path/to/image.png --task decision --output prediction.json
```

上述推理命令需要另行取得对应适配器。历史配置中的 `/root/autodl-tmp/...` 是实验时的服务器路径，换环境时需调整输入与输出路径；复用完整协议请阅读 [原生路线 README](industrial_vlm/README.md) 和第二轮实验报告。训练未保存 optimizer/scheduler 状态，现有适配器检查点不支持精确恢复训练。

服务器辅助工具通过 SSH 自身认证，可使用 `--host`、`--port`，或环境变量 `INDUSTRIAL_VLM_SSH_HOST`、`INDUSTRIAL_VLM_SSH_PORT`；仓库不提供认证信息。

## 代码来源

项目自有代码与上游实现的边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。AnomalyGPT 和 ImageBind 的随附许可证保留在迁移运行时快照内。数据集、模型及上游代码分别遵循各自许可。
