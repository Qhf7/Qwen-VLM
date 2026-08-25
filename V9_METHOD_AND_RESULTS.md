# V9 Best Hybrid VLM 方法与结果

## 1. 版本状态

本版本从真实榜单最高分的 `qwen_dinov2_hybrid_v2` 独立复制而来，代码位于
`v9_best_hybrid_vlm/`。旧版代码、旧 checkpoint、旧预测目录和旧提交 ZIP 均不修改。

当前服务器训练状态：

```text
RUN_ROOT=/root/autodl-tmp/v9_best_hybrid_vlm_run_20260825_122246
阶段=pseudo anomaly head
重建分支=8/8 epoch 已完成
pseudo head=训练中
榜单结果=尚未提交
```

因此，v9 当前没有可以填写的真实榜单分数。训练完成后会分别生成 dense control 和
VLM hybrid 两个 ZIP，再以竞赛平台返回的指标更新结果，不使用 proxy 分数冒充榜单结果。

## 2. 方法链路

```text
官方 Train 正常五视角图像
        |
        +--> 冻结 DINOv2 ViT-B/14
        |       |
        |       +--> 多层 patch 特征（1036 / 518）
        |       +--> Reconstruction branch
        |       +--> 类别/视角 memory bank + global bank
        |       +--> Pseudo anomaly pixel head
        |               |
        |               +--> dense anomaly map
        |
        +--> 程序化伪异常 + hard-normal 监督
                |
                +--> microcrack / low-contrast stain / scratch / dent / chip ...

Test_A 五视角图像
        |
        +--> DINO 三分支 dense control
        |       +--> Train-normal 分层校准
        |       +--> 多尺度融合
        |
        +--> 本地 Qwen3-VL crop audit JSONL
                +--> is_anomaly
                +--> 五档 anomaly score
                +--> reason 关键词
                +--> box 面积
                +--> 同视角 crop 一致性
                +--> 五视角证据数量
                        |
                        +--> 仅在 DINO 已响应像素上加入有界 residual
                                |
                                +--> submission_v9_dense.zip
                                +--> submission_v9_hybrid.zip
```

可渲染的详细架构图见 [V9_ARCHITECTURE.md](V9_ARCHITECTURE.md)。

## 3. 相比最佳 v2 的更新

| 环节 | Hybrid v2 | V9 更新 | 目的 |
|---|---|---|---|
| 主定位 | DINOv2 reconstruction、memory、pseudo head | 完整保留三分支 | 保持已验证的像素定位空间 |
| Train 校准 | 按排序前缀取样 | 类别分层固定随机抽样 | 避免后部类别没有独立校准 |
| 伪异常 | 多类型工业伪异常 | 增加微裂纹、低对比污渍 | 覆盖细线和弱对比缺陷 |
| 困难正常 | 背景和部分轮廓约束 | 边缘、高光、强纹理、背景联合监督 | 从训练阶段抑制整件泛亮 |
| Qwen3-VL | 局部候选语义增强 | score、reason、box、crop、五视角联合证据 | 减少单个错误 crop 的影响 |
| VLM mask | 局部候选增强 | 只能增强 DINO 已有响应，最大 residual 0.12 | 防止 VLM 把整个零件当异常 |
| 输出 | Hybrid ZIP | dense control + hybrid 两个 ZIP | 可直接做消融对照 |

## 4. 已测评方法和真实结果

下表列出目前最重要的已测评版本；全部提交的完整表格见
[SUBMISSION_RESULTS_AND_METHODS.md](SUBMISSION_RESULTS_AND_METHODS.md)。

| 方法 | 提交结果 | 总分 | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen ImageNet ViT baseline | `submission_vit_baseline.zip` | 41.9916 | 0.7562 | 0.8579 | 0.5958 | 0.0469 | 0.1193 |
| Dinomaly 思路适配 | `submission_dinomaly.zip` | 43.7244 | 0.6572 | 0.8230 | 0.7528 | 0.0610 | 0.1086 |
| INP-Former 思路适配 | `submission_inpformer.zip` | 52.6225 | 0.7648 | 0.8824 | 0.8209 | 0.1565 | 0.2189 |
| 基础 Ensemble | `submission_ensemble.zip` | 59.0327 | 0.8456 | 0.9201 | 0.8772 | 0.2307 | 0.2869 |
| Object-Aware v5 Soft | `submission_object_aware_v5_soft.zip` | 58.7413 | 0.8536 | 0.9227 | 0.8782 | 0.2197 | 0.2776 |
| DINOv2 Dense v1 Stream | `submission_qwen_dinov2_dense_v1_stream.zip` | 70.0925 | 0.9008 | 0.9524 | 0.8542 | 0.4499 | 0.5085 |
| Qwen3-VL + DINOv2 Hybrid v1 | `submission_qwen_dinov2_hybrid_v1_stream.zip` | 70.1399 | 0.8984 | 0.9521 | 0.8542 | 0.4524 | 0.5097 |
| DINOv2 Dense v2 | `submission_qwen_dinov2_dense_v2.zip` | 70.4787 | 0.9084 | 0.9563 | 0.8537 | 0.4556 | 0.5124 |
| **Qwen3-VL + DINOv2 Hybrid v2** | `submission_qwen_dinov2_hybrid_v2.zip` | **70.5793** | **0.9092** | **0.9564** | 0.8537 | **0.4579** | **0.5139** |
| DINOv2 Multi-scale v3 Dense | `submission_qwen_dinov2_multiscale_v3_dense_stream.zip` | 64.7091 | 0.8503 | 0.9074 | 0.8684 | 0.3546 | 0.4202 |
| VLM Finetune v4 | `submission_vlm_finetune_v4_resume.zip` | 59.6458 | 0.7642 | 0.8667 | 0.7506 | 0.3186 | 0.4386 |
| VLM-Primary v5 | `submission_vlm_primary_v5.zip` | 51.8962 | 0.8499 | 0.8976 | 0.8539 | 0.0846 | 0.1622 |

## 5. 结果解读

目前真实结果说明：DINOv2 的高分辨率 patch、正常样本 memory bank、重建分支和伪异常
像素监督是像素级 AP/F1max 的主要来源。Qwen3-VL 直接主导 dense mask 时，容易把零件
整体语义当作异常，VLM-Primary v5 的 P-AP 只有 `0.0846`；Qwen 作为受约束的局部
语义残差时，Hybrid v2 达到当前最高总分 `70.5793`。

V9 延续这个结论：DINO 决定异常的空间位置，Qwen 只回答已有候选是否具有缺陷语义。
VLM box 不会直接转换成矩形 mask，所有 VLM 增量都经过 DINO support gate、box 面积
约束和最大幅值限制。

## 6. 训练、合规与产物

- 训练、memory bank 和校准只使用官方 Train 正常图像；
- 伪异常从 Train 正常图像程序化生成；
- 使用通用公开 DINOv2、CLIP、Qwen3-VL 基础权重；
- 不使用外部异常数据、测试标签或在线 API；
- Qwen audit JSONL 由本地模型生成，仅作为结构化语义证据；
- 输出 mask 为 `448x448` 单通道灰度 PNG；
- v9 脚本发现已有 `RUN_ROOT` 或 ZIP 时会拒绝覆盖。

运行入口：

```bash
DATA_ROOT=/root/autodl-tmp/Dataset \
WEIGHT_DIR=/root/autodl-tmp/qwen_dinov2_hybrid_v2/weights \
VLM_AUDIT=/root/autodl-tmp/qwen3vl_local/submission_qwen3vl_crop_v2/qwen_crop_outputs.jsonl \
RUN_ROOT=/root/autodl-tmp/v9_best_hybrid_vlm_run_YYYYMMDD_HHMMSS \
PYTHON=/root/miniconda3/bin/python \
bash v9_best_hybrid_vlm/run_v9_pipeline.sh
```

本地训练状态和榜单成绩以服务器日志、提交平台结果为准。本文件不会将未完成训练
写成已取得的分数。
