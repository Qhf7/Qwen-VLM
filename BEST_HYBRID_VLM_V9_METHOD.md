# Best Hybrid VLM v9 方法说明

## 1. 版本定位

`v9_best_hybrid_vlm/` 从当前真实榜单最高分版本
`submission_qwen_dinov2_hybrid_v2.zip` 独立复制并继续开发。历史目录、checkpoint、
预测结果和 ZIP 均不修改、不覆盖。

已知真实榜单基线：

| 版本 | 总分 | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Pixel F1max |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL + DINOv2 Hybrid v2 | **70.5793** | **0.9092** | **0.9564** | 0.8537 | **0.4579** | **0.5139** |
| VLM-Guided Residual v6 | 49.2551 | 0.7452 | 0.8597 | 0.6732 | 0.1386 | 0.2674 |

v9 的目标不是让 VLM 重新生成一套 dense mask，而是在已经验证有效的 Hybrid v2
定位空间中增加受限语义证据。当前文件描述的是新实现和训练方案，不把尚未测评的 v9
写成已获得更高榜单分数。

## 2. 失败分析驱动的设计

v6 相比最佳 v2 丢失约 21.3 分，主要原因是：

1. 使用 DINO decoder + reference 替代了最佳 v2 的三分支定位，异常空间发生改变；
2. Pixel 99.5% 分位以下硬归零，低对比真实缺陷被过度抑制；
3. `background_keep=0`、`boundary_keep=0.32` 对边缘缺口和裂纹召回不利；
4. 图像 logit 经新的 arctan 校准后整体上移，Test_A 分数中位数从约 0.34 上升到 0.73；
5. 程序化异常 Proxy 与真实异常差异过大，Proxy 选出的 checkpoint 没有转化为榜单收益。

因此 v9 遵循：

```text
DINOv2 三分支决定异常在哪里
Qwen3-VL 判断已有候选是否具有异常语义
VLM 不允许在 DINO 无响应的位置产生亮像素
```

## 3. 完整链路

### 3.1 输入

每个样本包含五个固定视角。训练只读取官方 Train 正常图；测试读取 Test_A/B 图像，
不读取测试标签。

### 3.2 DINOv2 三分支主定位

冻结通用 DINOv2 ViT-B/14，在 1036 和 518 输入尺度提取多层 patch 特征。

| 分支 | 输入/训练 | 输出 | 作用 |
|---|---|---|---|
| Reconstruction | Train 正常多层 token，训练轻量 decoder | 多层重建余弦距离 | 检测偏离正常流形的纹理和结构 |
| Memory | `(category, view)` normal coreset，另有 global bank | patch 3-NN 距离 | 检测局部正常参照之外的区域，支持未见类别回退 |
| Pseudo head | 官方正常图上程序化伪异常 | patch-grid anomaly logit | 学习小缺陷形状、边界和低对比变化 |

三分支先在各自 Train-normal 分布上校准，再按 0.40/0.35/0.25 融合。多尺度默认
以 1036 高分辨率分支为主，保持 Hybrid v2 已验证的连续像素排序能力。

### 3.3 分层 Train-normal 校准

旧 v2 的 `calib-limit` 直接使用按类别、文件名排序后的前 N 组。若 N 小于全部 Train，
后部类别无法获得独立校准，只能回退 global bounds。

v9 使用确定性的类别分层抽样：

```text
limit 均匀分配到全部 Train 类别
每类内部使用固定随机种子抽样
pixel 校准保留 v2 的 90.0% / 99.9% 宽尾部映射
image 校准保留 50.0% / 99.0% 映射
```

这不是每张 mask 的百分位归一化。所有阈值均由官方 Train 正常样本预先拟合，同一类别
测试图共享同一映射。

### 3.4 更真实的伪异常

v9 在 v2 的 hairline、scratch、speck、dent、chip、contamination、discolour、
cut-paste、misalignment、hole 基础上新增：

- `microcrack`：细主裂纹和短分叉，使用中低对比度而不是纯黑线；
- `low_contrast_stain`：连续、软边界、小幅增益变化的局部污渍。

伪异常尺寸继续统一定义在 448 空间，避免原始图分辨率不同导致缺陷尺寸失真。

### 3.5 困难正常负样本

对每张官方正常图构造 `hard_normal_grid`：

- 零件强结构边缘；
- 镜面高光和低色差亮区；
- 高频正常纹理；
- 原有前景轮廓和背景区域。

这些区域在 pseudo head 的训练损失中被监督为正常。若伪异常与其重叠，异常 mask 优先，
不会错误地把合成缺陷同时标为困难正常区域。

### 3.6 Qwen3-VL 局部语义证据

Qwen3-VL 读取局部 crop 并输出结构化记录：

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.82,
  "defect_type": "scratch",
  "reason": "a thin visible scratch",
  "box": [120, 260, 520, 420]
}
```

v9 联合以下证据：

| 证据 | 处理 |
|---|---|
| `is_anomaly` | 必须为真，否则不产生正证据 |
| 原始 score | 映射为 0/0.45/0.60/0.75/0.90 五档 |
| reason | 缺陷词和肯定词增强；normal/reflection/uncertain 等否定词清零 |
| box 面积 | 紧凑区域可信度高；大于 7.5% 画幅拒绝 |
| 同视角 crop | 同缺陷类型且空间重合时小幅增强 |
| 五视角 | 两个及以上视角有证据时轻微增强；单视角不被硬删除 |

### 3.7 DINO 候选门控残差

对每个像素：

```text
support = clip((DINO - 0.012) / (0.18 - 0.012), 0, 1)
aligned = VLM_proposal * support
residual <= 0.12 * aligned
final = DINO + residual * (1 - DINO)
```

关键性质由单元测试保证：

- DINO 为零时，VLM box 内仍为零；
- 没有有效 VLM 证据时，mask 与 dense control 逐像素完全一致；
- VLM 最大像素残差不超过 0.12；
- box 只定义语义作用域，不会被直接保存成矩形 mask。

图像级分数只加入最多 `0.025 * aligned_support * (1 - base_score)` 的正残差。
这比 v2 的 4% 直接线性混合更保守，避免 v6 中 score 分布整体上移。

## 4. 与历史版本逐环节对比

| 环节 | Hybrid v2（70.5793） | v6（49.2551） | v9 |
|---|---|---|---|
| 主定位 | reconstruction + memory + pseudo head | decoder + reference | 完整保留 v2 三分支 |
| 分辨率 | 1036/518 多尺度 | 单一 DINO size | 1036/518 多尺度 |
| 像素校准 | Train 90/99.9，但类别截断 | 99.5 以下硬归零 | 类别分层 Train 90/99.9 |
| 边缘召回 | soft prior 0.55 | boundary 0.32 | 保留 v2 soft prior 0.55 |
| 伪异常 | 10 类 | 训练状态异常 | v2 + microcrack + low-contrast stain |
| 困难正常 | 背景、轮廓边缘 | 多状态损失 | 边缘 + 高光 + 强纹理 + 背景 |
| VLM 作用 | Qwen box 内乘法增强 | CLIP dense residual | DINO 支持门控的有界局部残差 |
| 多视角 | DINO score 聚合 | logit top-2 | DINO 聚合 + VLM 证据计数 |
| 图像 score | DINO 96% + Qwen 4% | 新 logit/arctan | v2 base + 最大 2.5% 对齐残差 |
| 输出 | hybrid ZIP | 单 ZIP | dense control + hybrid 两个 ZIP |

## 5. 训练与运行

本地检查：

```bash
python3 -m py_compile v9_best_hybrid_vlm/*.py
python3 v9_best_hybrid_vlm/test_v9.py
```

AutoDL 完整流水线：

```bash
DATA_ROOT=/root/autodl-tmp/Dataset \
WEIGHT_DIR=/root/autodl-tmp/weights \
CLIP_WEIGHT=/root/autodl-tmp/weights/ViT-B-16.pt \
VLM_AUDIT=/path/to/qwen_crop_audit.jsonl \
RUN_ROOT=/root/autodl-tmp/best_hybrid_vlm_v9_run_YYYYMMDD_HHMMSS \
PYTHON=/root/miniconda3/bin/python \
bash v9_best_hybrid_vlm/run_v9_pipeline.sh
```

输出：

```text
RUN_ROOT/submission_v9_dense.zip
RUN_ROOT/submission_v9_hybrid.zip
RUN_ROOT/dense_validate.log
RUN_ROOT/hybrid_validate.log
```

`RUN_ROOT` 必须是新目录；脚本发现同名目录或 ZIP 会直接停止，防止覆盖历史结果。

## 6. 验证和选模原则

不能根据 Test_A/B 标签调参。推荐在官方 Train 上按类别留出构造 proxy：

1. 留出完整类别，不进入 reconstruction、head、类别 memory 和类别校准；
2. 留出 `microcrack`、低对比污渍、edge chip 等生成器作为未见异常；
3. 分别评估 dense control 和 hybrid；
4. 仅当 hybrid 的 Pixel AP/F1max 提高，且正常边缘响应不增加时采用 hybrid；
5. 真实提交优先保留 dense control，防止 VLM 审计质量波动。

推荐选模指标：

```text
0.35 * held-out Pixel AP
+ 0.30 * held-out Pixel F1max
+ 0.15 * held-out Pixel AUROC
+ 0.12 * held-out Image AP
+ 0.08 * held-out Image AUROC
- 0.15 * normal boundary activation
- 0.10 * broad-mask penalty
```

## 7. 合规性

- reconstruction、pseudo head、memory bank、校准只使用官方 Train 正常图；
- 伪异常完全由官方 Train 图程序化生成或同图自裁剪；
- 只使用通用 DINOv2、CLIP、Qwen3-VL 基础权重；
- 不加载 MVTec、VisA 等外部异常专项权重；
- 不使用外部异常数据和测试标签；
- Qwen3-VL 必须本地推理，不调用在线 API；
- 不进行基于榜单标签或逐图结果的百分位归一化。

## 8. 当前状态和风险

代码已独立实现并通过静态编译与门控性质单元测试。尚未在 AutoDL 上完成 v9 全量训练，
也尚未获得真实榜单分数，因此不能承诺一定超过 70.5793。最主要风险仍是 Qwen 对工业
微小缺陷的语义敏感度有限；为此 v9 同时输出不依赖 VLM 的 dense control，且 VLM 残差
受到 DINO 支持、面积和幅值三重限制。
