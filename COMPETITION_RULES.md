# CSIG 复杂工业场景异常检测算法挑战赛：固定开发约束

本文件是本项目后续所有模型、推理和提交脚本的固定参考。

## 任务

- 输入：每个工业样本文件夹包含 5 个视角图像 `0.png` 至 `4.png`。
- Train：50 个已见类别的正常样本，只能用于正常建模、校准、自监督学习或合规伪异常构造。
- Test A：已见类别的正常/异常样本。
- Test B：包含已见类别和完全未见类别；测试标签不可访问、不可人工标注、不可用于调参或结果修正。
- 输出：`submission.csv` 与 `predicted_masks/`。

## 输出格式

`submission.csv` 必须包含：

```text
group_folder,anomaly_score
```

每个测试样本必须有 5 个 mask：

```text
predicted_masks/<category>/<sample>/0_mask.png
...
predicted_masks/<category>/<sample>/4_mask.png
```

mask 必须是单通道灰度图，尺寸为 448×448；像素值越大表示越可能异常。

## 评测重点

- 图像级：I-AUROC、I-AUPR。
- 像素级：P-AUROC、P-AUPR、P-F1max；其中本项目重点优化像素级 AP（P-AUPR）和 P-F1max。
- 零样本：未见类别上的图像级与像素级综合表现。
- 还需满足单帧处理延迟不超过 1 秒的要求。

## 允许使用

- 赛事官方 Train/Test 数据。
- 所有参赛者均可获得的通用公开预训练权重，例如 ImageNet、DINO/DINOv2、CLIP。
- 仅使用官方数据的正常样本建模、自监督训练、MixUp/CutMix、CutPaste、随机颜色/纹理扰动等合规方法。
- 公开论文、模型结构和算法思路，包括 Dinomaly、INP-Former、WinCLIP、AdaCLIP 等。

## 禁止使用

- 任何外部数据集、人工采集数据及其衍生数据。
- 使用 MVTec、VisA 或其他异常检测数据集专项训练/微调得到的权重、Prompt、Adapter、检测头。
- 外部在线 LLM/VLM API，外部模型生成的标签、伪标签、数据或结果修正。
- 访问、人工标注、探测或利用 Test A/Test B 标签进行训练、调参和结果修正。
- 硬编码类别答案或其他与测试标签相关的规则。

## 开发原则

- 所有新增实验写入新的目录/文件，不覆盖已验证模型和提交包。
- 优先保留跨模型的绝对异常强度校准；不要未经验证对每张 mask 做独立百分位拉伸。
- 后处理必须是可复现、与测试标签无关、可在统一硬件上运行的确定性流程。
- 每次提交前检查 750 条 CSV、3750 张 mask、mask 尺寸/模式及 ZIP 完整性。
