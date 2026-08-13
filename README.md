# VLM_Ali

Real-IAD Variety 复杂工业场景异常检测算法挑战赛项目。

本项目面向多类别、多视角工业异常检测，输入每个零件的 5 个相机视角图像，输出图像级异常分数和像素级异常 mask。项目重点探索冻结视觉基础模型、正常样本建模、伪异常训练和视觉语言模型（VLM）零样本异常检测。

## 任务概览

- Train：仅包含 50 个已见类别的正常样本。
- Test A：已见类别的正常与异常样本。
- Test B：包含已见类别和完全未见类别，用于零样本泛化评估。
- 每个样本包含 `0.png` 至 `4.png` 五个视角。
- 输出 mask 要求为单通道灰度图，尺寸为 `448×448`。

## 合规说明

本项目遵循赛事公告：

- 只使用赛事官方数据进行训练、校准和调参；
- 允许使用公开、通用的 ImageNet、DINO/DINOv2、CLIP 等预训练权重；
- 不使用 MVTec、VisA 等异常数据集专项训练得到的 Prompt、Adapter、检测头或模型权重；
- 不调用外部在线 LLM/VLM 服务生成标签、伪标签或结果修正；
- 不访问或利用测试集标签；
- 伪异常只由官方 Train 正常图片通过 CutPaste、纹理、颜色和几何扰动构造。

完整约束见 [COMPETITION_RULES.md](COMPETITION_RULES.md)。

## 主要代码

| 文件 | 说明 |
|---|---|
| `baseline_infer.py` | 冻结 ImageNet ViT 基线 |
| `multiscale_vit_infer.py` | 多尺度 ViT 异常检测 |
| `paper_adapt_infer.py` | INP-Former / Dinomaly 思路适配 |
| `pseudo_anomaly_head.py` | 基于官方正常样本构造伪异常并训练轻量 patch head |
| `clip_vlm_fusion.py` | 通用 CLIP 零样本语义分数融合 |
| `vlm_patch_prototype_infer.py` | CLIP 语义分数 + ViT patch 正常原型模型 |
| `dinov2_vlm_multiview_v2.py` | DINOv2 多层 patch memory bank + 多类型伪异常 head + 局部 CLIP + 五视角一致性 |
| `build_ensemble_submission.py` | 多模型融合和提交包生成 |
| `build_pseudo_fusion.py` | 伪异常结果融合 |
| `validate_submission.py` | 提交文件格式校验 |
| `COMPETITION_RULES.md` | 固定比赛规则和开发约束 |
| `VLM_PAPERS_AND_COMPLIANCE.md` | VLM 论文方向和合规边界 |

## 运行示例

以下命令在已配置好 PyTorch、TorchVision、CLIP 和相应通用权重的环境中运行。路径请根据服务器实际目录调整。

### VLM + Patch Prototype

```bash
python vlm_patch_prototype_infer.py \
  --data-root /path/to/Dataset \
  --clip-weight /path/to/ViT-B-32.pt \
  --output /path/to/submission_vlm_proto_v1 \
  --zip /path/to/submission_vlm_proto_v1.zip \
  --batch-size 20
```

### 校验提交包

```bash
python validate_submission.py \
  --root /path/to/submission_vlm_proto_v1 \
  --zip /path/to/submission_vlm_proto_v1.zip
```

校验内容包括：750 条 CSV 记录、3750 张 mask、目录结构、灰度模式、448×448 尺寸和 ZIP 完整性。

## 提交格式

压缩包根目录必须包含：

```text
submission.csv
predicted_masks/
└── <category>/<sample>/
    ├── 0_mask.png
    ├── 1_mask.png
    ├── 2_mask.png
    ├── 3_mask.png
    └── 4_mask.png
```

`submission.csv` 格式：

```csv
group_folder,anomaly_score
3_adapter/S0001,0.0321
```

`anomaly_score` 越大表示样本越可能异常；mask 像素值越大表示该像素越可能属于异常区域。

## 已验证结果

目前已知最佳 A 榜结果来自 ViT、INP-Former 和 Dinomaly 的融合版本：

```text
总分：59.0327
分类指标：0.8456
分割指标：0.9201
综合/零样本指标：0.8772
```

提交结果 ZIP、数据集和模型权重不纳入 Git 仓库，请在本地或服务器单独保存。

## 论文方向

项目参考了以下公开研究方向：

- WinCLIP：局部窗口图文匹配；
- AnomalyCLIP：异常语义提示；
- AdaCLIP：类别自适应视觉语言提示；
- Dinomaly：视觉基础模型异常检测；
- INP-Former：正常 patch 原型与多类别异常检测。

论文笔记和权重使用边界见 [VLM_PAPERS_AND_COMPLIANCE.md](VLM_PAPERS_AND_COMPLIANCE.md)。
