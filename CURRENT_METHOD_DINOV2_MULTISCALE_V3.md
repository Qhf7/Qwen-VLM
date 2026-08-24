# 当前方法：DINOv2 Multi-scale Decoder v3

## 1. 方法定位

当前正在运行的方法是 `DINOv2 Multi-scale Decoder v3`。它是在已取得 A 榜
70.5793 分的 `Qwen3-VL + DINOv2 Hybrid v2` 基础上重新设计的像素级异常检测
方案，核心目标是提高当前瓶颈指标：

- Pixel AP；
- Pixel F1max；
- 微小划痕、污渍、凹陷和破损的边界定位能力；
- 对零件轮廓、反光、高光和正常重复纹理的抗误检能力。

v3 不是对已有提交 mask 做结果后处理，也不是继续微调 ensemble 权重。它新训练了
一个多层 DINOv2 特征驱动的多尺度像素 decoder，并使用官方 Train 正常样本构造
伪异常和困难正常负样本。

代码目录：

```text
v5_multiscale_v3/
```

服务器运行目录：

```text
/root/autodl-tmp/qwen_dinov2_multiscale_v3_run_20260821_190619
```

## 2. 当前运行状态

截至 2026-08-21，当前状态如下：

| 阶段 | 状态 | 结果 |
|---|---|---|
| 冒烟训练 | 已完成 | 2组样本、1轮，loss正常下降 |
| 多尺度 decoder训练 | 已完成 | 3/3轮 |
| 最终训练 loss | 已完成 | 0.41398 |
| 最终 checkpoint | 已生成 | `ckpt/decoder.pt` |
| 原始 Train-only proxy | 已完成 | Pixel AP 0.7512、F1max 0.7825 |
| Proxy AUROC修正 | 正在重算/已进入续跑链路 | 原公式存在大于1的计算错误，已修复 |
| 流式 memory bank | 等待 proxy 后执行 | 已改为有界 reservoir，避免60GB内存溢出 |
| Train正常分数校准 | 尚未完成 | 仅使用官方 Train |
| Test_A 推理 | 尚未完成 | 目标750组、3750视角 |
| Dense ZIP | 尚未生成 | 新文件名带 `dense_stream` |
| Qwen hybrid ZIP | 尚未生成 | 新文件名带 `hybrid_stream` |

训练得到的 checkpoint：

```text
ckpt/decoder_epoch1.pt
ckpt/decoder_epoch2.pt
ckpt/decoder_epoch3.pt
ckpt/decoder.pt
```

## 3. 完整方法链路

```text
官方 Train 正常五视角图像
        │
        ├── 工业风格伪异常构造
        │     ├── 细线划痕
        │     ├── 弯曲边缘划痕
        │     ├── 低对比度污渍
        │     ├── 凹陷、缺口、破损
        │     └── 同图 self-crop 异常
        │
        ├── 困难正常负样本
        │     ├── 零件轮廓
        │     ├── 正常结构边缘
        │     ├── 反光与高光
        │     └── 重复纹理和高频区域
        │
        ▼
冻结 DINOv2 ViT-B/14 @ 1036×1036
        │
        ├── 第3层：浅层纹理
        ├── 第6层：中浅层结构
        ├── 第9层：中深层语义
        └── 第11层：深层语义
        │
        ▼
每层独立投影 + 可学习层权重融合
        │
        ▼
多尺度像素 Decoder
        ├── patch-grid 上下文建模
        ├── 2倍空间上采样
        ├── 4倍空间上采样
        ├── 高分辨率主输出
        └── patch-grid 辅助输出
        │
        ▼
训练损失
        ├── BCE
        ├── Tversky
        ├── Boundary Loss
        ├── Auxiliary Loss
        └── Hard-negative Loss
        │
        ▼
推理阶段
        ├── 全图高分辨率 decoder 分支
        ├── 重叠局部窗口 decoder 分支
        ├── 类别/视角 memory bank
        ├── 全局 zero-shot memory bank
        ├── 前景与边界 soft prior
        └── 五视角置信度一致性
        │
        ▼
Dense anomaly mask + anomaly_score
        │
        ▼
可选 Qwen3-VL 局部语义 proposal 增强
        │
        ▼
submission.csv + 3750张 mask + ZIP
```

## 4. 输入与输出

### 4.1 训练输入

训练仅使用赛事官方 Train 正常样本。每个样本包含5个固定相机视角：

```text
类别名/Sxxxx/0.png
类别名/Sxxxx/1.png
类别名/Sxxxx/2.png
类别名/Sxxxx/3.png
类别名/Sxxxx/4.png
```

DINOv2 输入尺寸为：

```text
1036 × 1036
```

DINOv2 ViT-B/14 的 patch grid 为：

```text
74 × 74
```

### 4.2 训练监督信号

每张官方正常图被即时构造为：

```text
伪异常 RGB 图像
伪异常像素 mask
困难正常负样本权重图
```

困难负样本不是异常标签，而是用于告诉 decoder：这些区域虽然视觉响应很强，
但来自官方正常图，应输出较低异常概率。

### 4.3 推理输出

每个样本输出：

```text
anomaly_score：样本级异常概率
0_mask.png：视角0异常热力图
1_mask.png：视角1异常热力图
2_mask.png：视角2异常热力图
3_mask.png：视角3异常热力图
4_mask.png：视角4异常热力图
```

mask 要求：

```text
单通道灰度图
448 × 448
像素值范围 0～255
值越高表示越可能异常
```

## 5. DINOv2 多层特征融合

之前的 v2 将多个 DINOv2 层归一化后直接拼接。v3 为每一层设置独立投影：

```text
DINO layer feature
→ 1×1 convolution
→ GroupNorm
→ GELU
```

之后使用可学习权重进行融合：

\[
F = \sum_{l=1}^{L} \operatorname{softmax}(w_l) P_l(F_l)
\]

其中：

- \(F_l\)：第 l 层 DINOv2 特征；
- \(P_l\)：该层独立投影；
- \(w_l\)：训练得到的层权重；
- \(F\)：融合后的像素特征。

这种方式保留了不同深度的角色：

| DINOv2层级 | 主要信息 | 对异常检测的作用 |
|---|---|---|
| 浅层 | 纹理、颜色、边缘 | 细线划痕、污点、轻微纹理变化 |
| 中层 | 局部结构、部件形状 | 凹陷、破损、结构错位 |
| 深层 | 高层语义、整体结构 | 判断局部变化是否违背正常零件语义 |

## 6. 多尺度像素 Decoder

v2 的 pseudo head 主要在 DINOv2 patch grid 上用浅层卷积预测异常，然后直接插值
到448。对于5～15像素的缺陷，这容易形成块状或膨胀热力图。

v3 decoder 的空间路径是：

```text
74×74 融合特征
→ 上下文卷积
→ 148×148 特征
→ 局部卷积细化
→ 296×296 特征
→ 主异常 logits
→ 映射为448×448 mask
```

同时保留74×74辅助输出，使 patch 层级仍然学习整体异常区域。

多尺度 decoder 并不能凭空恢复 DINOv2 patch 中已经丢失的信息，但能够：

- 利用相邻 patch 上下文重建更连续的边界；
- 减少直接双线性插值产生的大方块；
- 使用高分辨率伪 mask 对窄划痕边缘进行监督；
- 分离局部尖峰异常与大面积正常结构响应。

## 7. 损失函数

当前训练目标为：

\[
L = 0.45L_{BCE}
  + 0.85L_{Tversky}
  + 0.25L_{Boundary}
  + 0.15L_{Aux}
  + 0.45L_{HardNegative}
\]

| 损失项 | 作用 |
|---|---|
| BCE | 基础像素分类与概率校准 |
| Tversky | 调节假阳性/假阴性，针对类别极不平衡 |
| Boundary | 强化划痕、缺口和污渍边缘 |
| Auxiliary | 保证低分辨率 patch grid 仍学习整体异常位置 |
| Hard-negative | 抑制轮廓、反光、高光和正常纹理误检 |

Tversky 中对假阳性的系数为0.72，对假阴性的系数为0.28。该设置更强调精度，
原因是当前 Pixel AP 和 F1max 的主要损失来自异常区域过宽及正常零件被高亮。

## 8. 工业风格伪异常

v3 的伪异常全部来自官方 Train 正常图或程序化绘制，不使用外部异常数据。

| 类型 | 构造方法 | 对应真实缺陷 |
|---|---|---|
| Hairline | 细、长、弯曲、低透明度线条 | 发丝划痕、裂纹 |
| Scratch | 深浅变化线条和局部毛刺 | 表面刮伤 |
| Contamination | 软边界、低对比度局部颜色变化 | 污渍、油污 |
| Dent | 中心阴影与边缘高光组合 | 凹陷、压痕 |
| Chip/Hole | 不规则缺口或局部缺失 | 破损、孔洞 |
| Discolour | 局部通道增益变化 | 色差、氧化 |
| Self crop | 同一官方图像内部小块变换 | 异物、错位纹理 |

伪异常尺寸以最终448空间定义，避免不同原图分辨率导致异常尺寸失控。

## 9. 困难正常负样本

困难负样本从官方正常图自动计算：

```text
高梯度正常边缘
高亮且低饱和度的反光区域
局部高频重复纹理
零件轮廓和结构边界
```

训练时这些区域被赋予额外的正常 BCE 惩罚，但如果与伪异常 mask 重叠则不作为
负样本。这一设计直接针对此前“整个零件轮廓被高亮”的失败模式。

## 10. Memory Bank

v3 继续使用两种正常特征库：

| Memory bank | 数据来源 | 作用 |
|---|---|---|
| 类别/视角 bank | 对应已见类别、对应相机视角的 Train正常 patch | 已见类别精确检测 |
| Global bank | 所有官方 Train类别正常 patch | 未见类别 zero-shot 保底 |

最终像素分数的基础融合为：

\[
M = 0.68M_{decoder} + 0.32M_{memory}
\]

第一次推理因无界保存3072维 patch 描述子，在 memory bank 构建到700/1000时超过
内存并退出。当前已修改为流式有界 reservoir：

- 每个类别/视角只保存固定容量描述子；
- 描述子暂存为 float16；
- 使用 reservoir sampling 保持样本覆盖；
- 完成收集后再执行 greedy coreset；
- 内存不再随训练样本数量线性增长。

## 11. 重叠局部高分辨率窗口

全图1036输入虽然比518高，但一个14×14 patch仍可能覆盖多个微小缺陷像素。
v3 在每个视角上额外选取两个重叠局部区域：

```text
左上区域
中心/右下区域
```

局部区域放大后再次输入相同 DINOv2 和 decoder，再映射回原448空间。最终使用：

```text
max(全图 decoder map, 局部窗口 decoder map)
```

该步骤没有改变 DINOv2 的14像素 patch embedding，而是通过放大局部区域提升缺陷
相对于 patch 的有效尺寸。

## 12. 五视角一致性

五台工业相机观察零件不同表面，因此不能直接要求五张 mask 在相同像素位置一致。
v3 只在异常证据强度层面建立一致性：

| 情况 | 处理 |
|---|---|
| 单视角弱响应 | 抑制，降低背景噪声 |
| 单视角强响应 | 保留，避免漏掉仅一个视角可见的缺陷 |
| 两个及以上视角有异常证据 | 小幅增强置信度 |
| 多视角均正常 | 保持低分 |

每个视角的异常证据取 anomaly map 顶部约0.2%像素均值，而不是使用单个最大像素。

## 13. 图像级异常分数

每个视角从校准后的 anomaly map 计算 top-k score。五个视角最终组合为：

\[
S_{image}=0.70\max(S_v)+0.30\operatorname{mean}(Top2(S_v))
\]

这样既保留单视角缺陷的强证据，又减少单像素噪声对样本级排序的影响。

## 14. Train-only Proxy

Proxy 仅从官方 Train 正常样本划分并在线构造伪异常，不读取 Test_A 或 Test_B 标签。

重点指标：

```text
Pixel AP
Pixel F1max
Pixel AUROC
```

原 proxy 结果：

```text
Pixel AP     0.7512
Pixel F1max  0.7825
```

原 Pixel AUROC 因实现错误得到大于1的数值，不具有效性。当前已修正为标准
Mann-Whitney/tie-aware 计算，并在续跑阶段重算为 `proxy_metrics_fixed.json`。

Proxy 只能用于比较方案，不代表真实 A/B 榜成绩。伪异常与真实工业缺陷存在域差异，
因此不能将0.75 proxy AP直接解释为榜单 Pixel AP。

## 15. Qwen3-VL 的作用

Qwen3-VL 不参与 decoder 训练，也不直接绘制最终 mask。它只在 dense DINOv2结果
完成后作为局部语义 proposal：

```text
Qwen crop 判断
→ 缺陷关键词与置信度检查
→ 小面积 box映射
→ 仅增强 box内已有 DINOv2 高响应
→ 不能把 DINOv2为0的区域直接点亮
```

因此：

- DINOv2 + decoder + memory bank负责像素排名；
- Qwen3-VL负责局部语义确认；
- Qwen proposal 面积受到严格限制；
- dense ZIP和hybrid ZIP分别生成，便于单独提交比较。

## 16. 与 v1/v2逐环节对比

| 环节 | Hybrid v1 | Hybrid v2 | 当前 v3 |
|---|---|---|---|
| 最佳总分 | 70.1399 | 70.5793 | 尚未提交 |
| Backbone | DINOv2 ViT-B/14 | DINOv2 ViT-B/14 | DINOv2 ViT-B/14 |
| 输入分辨率 | 1036为主 | 1036为主 | 1036全图 + 局部窗口 |
| 多层特征 | 多层特征参与三分支 | 多层特征拼接 | 每层独立投影和可学习融合 |
| 像素学习头 | 浅层 pseudo head | 加强的 pseudo head | FPN式多尺度 decoder |
| Decoder输出 | patch grid | patch grid | patch grid的4倍空间输出 |
| Reconstruction | 有 | 有 | 当前主链路不使用 |
| Memory bank | 类别/视角+global | 类别/视角+global | 保留，改为流式有界构建 |
| 伪异常 | 多类型 | 更偏小缺陷 | 工业细线、软污渍、低对比度缺陷 |
| 困难负样本 | 较少 | 背景、轮廓边缘 | 边缘、反光、高光、重复纹理 |
| 边界监督 | 无显式边界 loss | 无显式边界 loss | Boundary loss |
| 辅助监督 | 无 | 无 | patch-grid辅助输出 |
| 局部高分辨率 | 无 | 无 | 两个重叠局部窗口 |
| 五视角 | max/mean聚合 | max/mean聚合 | 弱响应抑制+多视角支持 |
| Proxy重点 | 综合参数调优 | 伪异常与融合调优 | Pixel AP/F1max和decoder验证 |
| Qwen3-VL | 保守 proposal | 面积限制更严格 | 延续保守 proposal，不主导mask |
| 内存管理 | 流式 memory | 流式 memory | reservoir + float16 + coreset |
| 输出 | dense/hybrid v1 | dense/hybrid v2 | dense/hybrid stream v3 |

## 17. 为什么 v3 可能改善 Pixel AP/F1max

之前 Pixel AUROC较高但 AP/F1max较低，说明模型大体能把异常像素排在正常像素之前，
但高分区域过宽、假阳性过多、最佳阈值下精确率不足。

v3 对应处理如下：

| 原问题 | v3处理 |
|---|---|
| 整个零件被轻度高亮 | 困难正常负样本与前景 soft prior |
| 零件边缘被当异常 | 轮廓 hard-negative loss |
| 反光、高光误报 | 显式高光正常负样本 |
| 微小划痕被 patch稀释 | 局部窗口放大推理 |
| mask呈大块状 | 多尺度 decoder与 boundary loss |
| 单一视角背景噪声 | 五视角置信度一致性 |
| 未见类别无类别 bank | global正常 memory bank |

## 18. 合规性说明

当前方法遵守比赛规则：

- 训练、校准和 proxy仅使用赛事官方 Train正常样本；
- 未使用 MVTec、VisA等外部异常检测数据集；
- 未使用外部异常检测专项训练权重；
- DINOv2为允许使用的公开通用预训练模型；
- 伪异常来自程序化构造或同一官方图像 self-crop；
- 未调用在线 API生成标签、mask或伪标签；
- Qwen3-VL为服务器本地推理，且只使用已有审计 proposal；
- Test_A仅用于规定的推理，不参与训练或参数拟合；
- 不读取 Test_A/Test_B标签。

## 19. 预计输出文件

续跑完成后，服务器运行目录预计包含：

```text
ckpt/proxy_metrics_fixed.json
dense_stream/submission.csv
dense_stream/predicted_masks/
submission_qwen_dinov2_multiscale_v3_dense_stream.zip
hybrid_stream/submission.csv
hybrid_stream/predicted_masks/
submission_qwen_dinov2_multiscale_v3_hybrid_stream.zip
```

每个 ZIP 应包含：

```text
submission.csv：750行样本结果
predicted_masks：3750张单通道448×448 PNG
```

## 20. 当前结论

v3 的实质变化是：从“DINOv2 patch 特征上的浅层异常分类头”升级为“分层 DINOv2
特征 + 多尺度像素 decoder + 工业伪异常 + 困难正常负样本 + 局部高分辨率窗口 +
五视角一致性”的完整像素检测链路。

当前三轮训练已经完成，正在使用修复后的流式 memory bank继续 proxy和推理。只有在
生成完整 ZIP 并提交 A榜后，才能判断这些结构改进是否真正超过 Hybrid v2 的70.5793分。
