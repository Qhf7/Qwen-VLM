# VLM-Guided Residual v6 方法说明

## 1. 版本目标

v6 是一个全新、独立的实验版本，目录为 `v8_vlm_guided_residual_v6/`。它不修改 v2、v3、v4、v5 的代码、checkpoint、预测目录或 ZIP。

该版本针对 v5 的真实榜单退化设计。v5 的 Pixel AUROC 为 0.8539，但 Pixel AP/F1max 只有 0.0846/0.1622。实测其正常零件内部平均响应约 0.6153，约 19.28% 像素高于 0.5，说明模型学成了“零件前景检测”，而不是“异常区域检测”。

v6 的原则是：

```text
DINOv2 决定异常可能出现在哪里
VLM 判断候选局部更像哪种异常或哪种正常困难状态
VLM 不允许脱离 DINO 候选区域独立点亮整件零件
```

## 2. 与历史版本逐环节对比

| 环节 | v2（70.5793） | v5（51.8962） | v6 新实现 |
|---|---|---|---|
| 空间主分支 | DINOv2 memory/reconstruction/pseudo head | VLM map 约占 85% 以上 | 多层 DINO decoder + 类别/视角 normal reference |
| VLM 角色 | Qwen 局部提议，小权重 | CLIP 直接主导 dense mask | Prompt/Adapter/patch-text 训练，但只输出有界 residual |
| 融合域 | 概率图融合 | 概率图线性加权 | `final_logit = dino_logit + bounded_vlm_residual` |
| 正 VLM 残差 | 无 | 可点亮任意前景 | 必须乘 DINO candidate gate |
| 负 VLM 残差 | 无 | 无明确约束 | 可压低 normal edge/reflection/texture |
| Normal reference | 类别/视角 DINO memory | CLIP memory，但 VLM 主导 | DINO 与 CLIP 双 reference，未见类回退 global bank |
| 正常困难区域 | 间接处理 | state 分类权重不足 | edge/reflection/hard texture 显式抑制与排序损失 |
| 正常组训练 | 有正常样本建模 | 35% normal group | 50% normal group + normal area/activation loss |
| 像素校准 | Train-normal 高分位 excess | 正常 90% 映射 0.1、99.95% 映射 0.9 | 99.5% 以下严格为 0，超出部分单调渐近映射 |
| Proxy 输入 | float map | float map | 最终 `uint8 PNG` 量化值 |
| 选模 | AP/F1/AUROC | 指标加权 | 指标减正常激活、边界、面积、饱和、tie 惩罚 |
| 图像分数 | DINO 主导 | 大量 sigmoid 饱和到 0.999 | 未饱和 logit 聚合，最后用 arctan 单调校准 |

## 3. 模型结构

### 3.1 DINOv2 主定位

输入 5 个视角的高分辨率图像。冻结 DINOv2 主干，提取浅层、中层、深层 token。多层 FPN decoder 输出 `dino_decoder_logit`。同时用官方 Train 正常图建立 `(category, view)` memory bank：

```text
current DINO patch -> kNN normal distance -> dino_reference_logit
```

每个类别内部再物理拆分建库样本和正常距离校准样本。已见类别使用类别/视角 bank 与对应正常尾部分布；只有超出正常距离 90% 分位的 patch 才产生 reference 证据。B 榜未见类别不存在对应 bank，自动使用只由官方 Train 构建并校准的 global normal bank。两项在 logit 空间融合，且 decoder 始终保留至少 55% 权重。

### 3.2 局部跨模态分支

冻结通用 CLIP 图像和文本主干，只训练：

- learnable prompt context；
- visual adapter；
- patch projection；
- local semantic decoder；
- 少量温度、候选阈值和残差比例参数。

每个 patch 对齐 12 种状态：background、normal surface、normal edge/texture、normal reflection，以及 scratch、crack、dent、stain、chip、foreign object、discoloration、deformation。

### 3.3 有界门控残差

```text
candidate_gate = sigmoid((dino_logit - threshold) / temperature)
semantic_sign = tanh(vlm_semantic_logit / 3)
positive_residual = relu(semantic_sign) * candidate_gate
negative_residual = min(semantic_sign, 0) * learned_fraction
vlm_residual = 1.25 * (positive_residual + negative_residual)
final_logit = dino_logit + vlm_residual
```

正残差最大不超过 1.25，并受 DINO 门控；因此 CLIP 即使把整个产品识别为“工业零件”，也不能独立产生整件高亮。负残差允许 VLM 使用 “normal edge/reflection/texture” 语义压低 DINO 假阳性。

## 4. 训练监督

仅使用赛事官方 Train 正常图。程序化伪异常覆盖 hairline、scratch、speck、dent、chip、contamination、cut-paste；crack、discoloration、misalignment、hole 留作代理验证的未见生成器。

总损失包括：

```text
1.00 final pixel segmentation
0.65 DINO-primary segmentation
0.30 multi-state classification
0.25 patch-text region alignment
0.40 normal-surface suppression
0.55 edge/reflection/hard-negative suppression
0.30 anomaly-vs-hard-normal ranking
0.45 normal predicted-area penalty
0.30 positive residual leakage penalty
0.08 VLM residual magnitude regularisation
0.18 view-level classification
0.12 five-view sample classification
0.12 CLIP global semantic classification
0.08 prompt anchor
```

这组损失直接约束模型训练阶段的错误，不依赖生成结果后再裁剪轮廓。

## 5. Train-only Proxy 与选模

类别按五折留出。fold 训练类别不允许进入 held-out 类别的 Prompt/Adapter 训练、类别 memory 或校准；held-out 类仅能使用 global bank/global calibration。

Proxy 在写入指标前执行与提交相同的 `round(mask * 255)`，再以量化后的值计算 Pixel AP、Pixel F1max 和 Pixel AUROC。除此之外记录：

- 正常图 mask mean；
- 正常图 `mask > 0.5` 面积；
- 正常 edge/reflection 响应；
- 全白像素面积；
- image score 大于等于 0.99 的比例；
- CSV 保留 8 位后分数 tie 比例。

checkpoint selection 以 Pixel AP/F1max 为主，并扣除上述病理响应。进度 JSON 文件通过严格文件名匹配排除，不可能被误读成指标报告。

## 6. 合规性

- 训练、memory、校准、Proxy 只使用官方 Train；
- 仅加载公开通用 DINOv2 与 CLIP 基础权重；
- 不使用外部异常数据或异常专项权重；
- 不调用在线 VLM/API 生成、修正或标注测试结果；
- Test 仅执行前向推理；
- 所有旧版本代码、结果和 ZIP 保持不变。

## 7. 风险与验证标准

v6 是针对已观测故障的结构性修复，不保证榜单必然超过 v2。进入最终推理前至少要求：

- held-out Proxy 的 Pixel AP/F1max 不低于 epoch0；
- normal mask mean 显著低于 v5；
- normal `area > 0.5` 接近 0；
- boundary response 不随训练上升；
- image score 无大面积 0.99 饱和；
- 生成的 ZIP 有 750 行 CSV 和 3750 张单通道 448x448 mask。
