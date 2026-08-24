# 提交结果与模型方法总表

本文记录 CSIG Real-IAD Variety 赛题已完成测评的提交文件、官方 A 榜指标及对应方法。数据来自竞赛平台回传结果，按提交时间从新到旧排列。

## 指标说明

- `I-AUROC`、`I-AP`：图像级异常分类指标；
- `P-AUROC`、`P-AP`、`P-F1max`：像素级异常分割指标；
- 总分为竞赛平台公布结果；
- 所有表内结果均为实际提交结果，不是本地 Proxy；
- ZIP、数据集、权重和 checkpoint 不进入 Git 仓库。

## 全部已测评提交

| 提交文件 | 提交时间 | 状态 | 总分 | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | 模型与方法 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `1787492320428_submission_vlm_primary_v5.zip` | 2026-08-23 21:38:47 | 测评完成 | 51.8962 | 0.8499 | 0.8976 | 0.8539 | 0.0846 | 0.1622 | **VLM-Primary v5**：WinCLIP 多尺度局部窗口、AdaCLIP 式 Static/Dynamic Prompt、CLIP normal-reference；VLM 约占 dense mask 主导地位，DINOv2 被限制为最多 15% 残差。正常零件前景被大面积激活，导致 P-AP/F1max 明显下降。 |
| `1787394277884_submission_vlm_finetune_v4_resume.zip` | 2026-08-22 18:24:39 | 测评完成 | 59.6458 | 0.7642 | 0.8667 | 0.7506 | 0.3186 | 0.4386 | **VLM Finetune v4**：冻结 CLIP/DINOv2 主干，训练 Prompt、Visual Adapter、Anomaly Query、多状态 Pixel Head、动态跨模态 Gate 与五视角 MIL；只用官方 Train 伪异常训练。无约束 Gate 和图像排序退化影响最终成绩。 |
| `1787332900516_submission_qwen_dinov2_multiscale_v3_hybrid_stream.zip` | 2026-08-22 01:21:46 | 测评完成 | 64.5236 | 0.8252 | 0.8991 | 0.8688 | 0.3631 | 0.4249 | **DINOv2 Multi-scale v3 Hybrid**：浅/中/深多层特征、FPN 式多尺度 decoder、重叠局部窗口、困难正常负样本、五视角一致性；加入 Qwen3-VL 保守局部 proposal。像素 AUROC 提高，但图像排序和 P-AP/F1max 低于 v2。 |
| `1787332872330_submission_qwen_dinov2_multiscale_v3_dense_stream.zip` | 2026-08-22 01:21:18 | 测评完成 | 64.7091 | 0.8503 | 0.9074 | 0.8684 | 0.3546 | 0.4202 | **DINOv2 Multi-scale v3 Dense**：多层 DINOv2、FPN decoder、伪异常边界监督、hard-negative、流式 memory bank 和多视角约束；不加入 Qwen proposal。 |
| `1787247976052_submission_qwen_dinov2_hybrid_v2.zip` | 2026-08-21 01:46:17 | 测评完成 | **70.5793** | **0.9092** | **0.9564** | 0.8537 | **0.4579** | **0.5139** | **Qwen3-VL + DINOv2 Hybrid v2，当前最佳**：高分辨率多层 DINOv2、类别/视角 memory bank、重建分支、多类型伪异常 pixel head、Train-normal 校准；Qwen3-VL 只对紧凑候选区域做语义增强，不直接绘制 mask。 |
| `1787247967909_submission_qwen_dinov2_dense_v2.zip` | 2026-08-21 01:46:09 | 测评完成 | 70.4787 | 0.9084 | 0.9563 | 0.8537 | 0.4556 | 0.5124 | **DINOv2 Dense v2**：与 Hybrid v2 相同的 DINOv2 memory/reconstruction/pseudo-head 主定位链路，但不使用 Qwen3-VL proposal；证明主要像素收益来自视觉正常建模。 |
| `1787193288095_submission_qwen_dinov2_hybrid_v1_stream.zip` | 2026-08-20 10:34:49 | 测评完成 | 70.1399 | 0.8984 | 0.9521 | 0.8542 | 0.4524 | 0.5097 | **Qwen3-VL + DINOv2 Hybrid v1**：流式类别/视角 memory bank、重建与伪异常 head 产生 dense mask；Qwen3-VL 仅对 DINO 已有局部响应做有界增强。 |
| `1787193278429_submission_qwen_dinov2_dense_v1_stream.zip` | 2026-08-20 10:34:39 | 测评完成 | 70.0925 | 0.9008 | 0.9524 | 0.8542 | 0.4499 | 0.5085 | **DINOv2 Dense v1 Stream**：DINOv2 多层 patch 特征、流式 normal memory、特征重建、伪异常 head、Train-only 校准和五视角分数聚合。 |
| `1787078198607_submission_qwen3vl_crop_v3_calibrated.zip` | 2026-08-19 02:36:38 | 测评完成 | 37.7825 | 0.5704 | 0.7291 | 0.5954 | 0.0467 | 0.1418 | **Qwen3-VL Crop v3 Calibrated**：读取 Qwen crop 审计 JSONL，根据 reason 关键词、box 面积、多 crop/五视角一致性和原始分数进行离散校准；mask 仍主要来自粗 crop/box，连续像素排序能力弱。 |
| `1787078184958_submission_qwen3vl_crop_v2.zip` | 2026-08-19 02:36:25 | 测评完成 | 38.1680 | 0.5774 | 0.7195 | 0.6158 | 0.0553 | 0.1309 | **Qwen3-VL Crop v2**：直接用本地 Qwen3-VL 对局部 crop 进行异常问答，输出 `is_anomaly`、粗 box 和离散 anomaly score；缺少精细视觉特征与像素监督。 |
| `1787046802985_submission_object_aware_v4.zip` | 2026-08-18 17:53:29 | 测评完成 | 52.0344 | 0.8536 | 0.9227 | 0.7630 | 0.1180 | 0.2071 | **Object-Aware v4**：按类别/视角建立 Train-normal 灰度模板与方差，叠加前景估计、模板偏差、局部残差和边界抑制；重建过于激进，损失异常召回。 |
| `1787044360120_submission_object_aware_v5_soft.zip` | 2026-08-18 17:16:32 | 测评完成 | 58.7413 | 0.8536 | 0.9227 | 0.8782 | 0.2197 | 0.2776 | **Object-Aware v5 Soft**：保留原 mask 排序，仅使用 soft foreground gate、宽响应衰减和轻量边界惩罚；比 v4 更保守，恢复部分 P-AUROC/AP/F1。 |
| `1786641446355_submission_object_aware_v1.zip` | 2026-08-14 01:17:39 | 测评完成 | 57.7989 | 0.8536 | 0.9227 | 0.8555 | 0.2114 | 0.2683 | **Object-Aware v1**：从测试图估计零件前景，背景清黑；在前景内减去低频宽响应，并衰减轮廓和过大连通区域。图像分数沿用基础模型。 |
| `1786640788751_submission_fusion_v3_dryrun_from_v2.zip` | 2026-08-14 01:06:52 | 测评完成 | 59.2190 | 0.8536 | 0.9245 | 0.8772 | 0.2307 | 0.2869 | **Fusion v3 Dry-run**：以已验证基础预测为主，仅融合 DINOv2 v2 的高置信局部异常证据；用于验证保守候选融合，避免候选分支大面积改写 mask。 |
| `1786640684275_submission_object_aware_v3.zip` | 2026-08-14 01:04:57 | 测评完成 | 55.8984 | 0.8536 | 0.9227 | 0.8489 | 0.1665 | 0.2383 | **Object-Aware v3 Balanced**：结合 v1 前景门控与 v2 Train-normal residual，融合 gated base、局部高通和正常残差支持，并压制大连通区域。 |
| `1786554839515_submission_ensemble_v2.zip` | 2026-08-13 01:14:14 | 测评完成 | 55.0477 | 0.8300 | 0.9113 | 0.8438 | 0.1671 | 0.2289 | **Ensemble v2**：ViT/INP/Dinomaly 以 0.15/0.55/0.30 融合，逐 mask robust stretch，并把 mask 峰值/面积回灌图像分数；归一化扩大了部分假阳性。 |
| `1786552995965_submission_ensemble.zip` | 2026-08-13 00:43:39 | 测评完成 | 59.0327 | 0.8456 | 0.9201 | 0.8772 | 0.2307 | 0.2869 | **基础 Ensemble**：ViT、INP-like prototypes、Dinomaly-like statistics 分数 rank-normalize，以 0.25/0.35/0.40 融合图像分数和连续 mask。 |
| `1786551421124_submission_inpformer.zip` | 2026-08-13 00:17:09 | 测评完成 | 52.6225 | 0.7648 | 0.8824 | 0.8209 | 0.1565 | 0.2189 | **INP-Former 思路适配**：通用 ImageNet ViT-B/16 patch 特征，每类别/视角构建 6 个正常原型，以最近原型距离生成 mask；未使用外部异常专项权重。 |
| `1786546097297_submission_dinomaly.zip` | 2026-08-12 22:48:28 | 测评完成 | 43.7244 | 0.6572 | 0.8230 | 0.7528 | 0.0610 | 0.1086 | **Dinomaly 思路适配**：冻结 ImageNet ViT-B/16，以类别/视角正常 patch 均值和方差计算标准化重建偏差；分辨率和正常统计能力有限。 |
| `1786532462564_submission_vit_multiscale.zip` | 2026-08-12 19:01:02 | 测评完成 | 38.8017 | 0.6810 | 0.7962 | 0.5567 | 0.0450 | 0.1117 | **Multi-scale ViT**：冻结 ImageNet ViT-B/16；整图加四个 224 crop，拼接为 28x28 heatmap，并使用 Train-normal patch 统计校准。局部 tile 外观变化造成较多误报。 |
| `1786526576143_submission_vit_baseline.zip` | 2026-08-12 17:22:56 | 测评完成 | 41.9916 | 0.7562 | 0.8579 | 0.5958 | 0.0469 | 0.1193 | **Frozen ViT Baseline**：ImageNet ViT-B/16，按类别/视角/patch 位置建立 Train-normal 对角高斯统计，将 14x14 标准化距离插值为 448x448 mask。 |

## 当前排名结论

| 结论 | 对应提交 | 数值 |
|---|---|---:|
| 当前最高总分 | `submission_qwen_dinov2_hybrid_v2.zip` | **70.5793** |
| 当前最高图像级 AUROC | `submission_qwen_dinov2_hybrid_v2.zip` | **0.9092** |
| 当前最高图像级 AP | `submission_qwen_dinov2_hybrid_v2.zip` | **0.9564** |
| 当前最高像素级 AUROC | `submission_object_aware_v5_soft.zip` | **0.8782** |
| 当前最高像素级 AP | `submission_qwen_dinov2_hybrid_v2.zip` | **0.4579** |
| 当前最高像素级 F1max | `submission_qwen_dinov2_hybrid_v2.zip` | **0.5139** |

实际结果说明：DINOv2 的高分辨率正常残差、类别/视角 memory bank 和伪异常像素监督是当前最可靠的精细定位来源；Qwen/VLM 只有作为局部、受约束的语义分支时才带来增益。让 VLM 直接主导 dense mask 会把“零件前景语义”误当作“异常语义”，从而严重损害 P-AP 与 P-F1max。

## 正在训练但尚未提交：VLM-Guided Residual v6

| 版本 | 状态 | 榜单结果 | 方法 |
|---|---|---|---|
| `v8_vlm_guided_residual_v6` | 服务器训练与 Train-only Proxy 流水线运行中 | 尚未提交，不填写预测分数 | DINOv2 恢复为主定位分支；VLM 只提供受 DINO candidate gate 约束、幅值不超过上限的正/负语义 residual；加入正常表面、边缘、反光、纹理 hard-negative、正常面积和 score saturation 约束，并使用最终 uint8 PNG 进行 Proxy 选模。 |

v6 详细设计见 [`VLM_GUIDED_RESIDUAL_V6_METHOD.md`](VLM_GUIDED_RESIDUAL_V6_METHOD.md)。实时查看服务器进度：

```bash
bash watch_v6_progress.sh
```
