# VLM 轻量异常适配 v4 方法说明

## 1. 方法目标

本方法针对 CSIG Real-IAD Variety 比赛中第一版 VLM 对微小缺陷不敏感的问题，在当前已验证的
`DINOv2 Multi-scale Decoder v3` 上增加一个真正参与训练的视觉语言分支。

目标不是让 Qwen3-VL 继续输出自然语言、离散分数或粗矩形框，而是让冻结的通用 CLIP 在官方
Train 正常图和程序化伪异常上进行轻量、局部、类别无关的工业异常适配，直接输出连续的像素级
异常热力图。

核心链路：

```text
官方 Train 正常五视角图
→ 程序化构造类型化伪异常与 hard-normal 状态
→ 冻结 DINOv2 多层特征
→ 冻结 CLIP 图像/文本 Transformer
→ 训练 Static/Dynamic Prompt、Visual Adapter、Anomaly Query
→ 多状态像素 Decoder
→ DINO/CLIP/Query 动态 Gate
→ 五视角 MIL
→ 448×448 mask + anomaly_score
```

## 2. 为什么第一版 VLM 对缺陷不敏感

| 原因 | 第一版表现 | 影响 |
|---|---|---|
| 全局图像 embedding | 一张图压缩为单个向量 | 微小划痕被整体零件语义淹没 |
| CLIP 原始目标偏图文语义 | 擅长物体类别和场景语义 | 不擅长毫米级表面变化 |
| 无工业领域适配 | 不理解接缝、轮廓、反光和加工纹理 | 大面积误检正常区域 |
| Prompt 仅 normal/defective | 缺少缺陷类型语义 | 无法区别划痕、凹陷、污渍 |
| 无 patch-mask 监督 | 文本与局部像素未对齐 | 高亮整个零件或粗框 |
| Qwen 自回归输出 | 分数集中在少数档位 | AUROC/AP 排序能力差 |
| 生成矩形框 | 框内包含大量正常像素 | Pixel AP、F1max 低 |

## 3. 冻结与训练范围

| 模块 | 状态 | 原因 |
|---|---|---|
| DINOv2 主干 | 冻结 | 保留通用局部视觉能力 |
| CLIP 图像 Transformer | 冻结 | 防止记住 50 个已见类别 |
| CLIP 文本 Transformer | 冻结 | 保留通用语言空间 |
| DINO 多尺度 Decoder | 从 v3 初始化，可训练 | 保留已验证的精细定位能力 |
| Static Prompt | 训练 | 学习工业异常上下文 |
| Dynamic Prompt | 训练 | 按材料、光照、表面生成图像条件偏移 |
| Visual Adapter | 训练 | 让 patch 对细线、低对比异常敏感 |
| Patch/Region Projection | 训练 | 对齐局部视觉和文本空间 |
| Anomaly Query | 训练 | 学习不同缺陷类型的局部响应 |
| 多状态 Pixel Head | 训练 | 明确区分背景、正常结构与缺陷 |
| Cross-modal Gate | 训练 | 动态融合 DINO、CLIP 和 Query |
| Five-view MIL | 训练 | 联合视角级和样本级异常判断 |

这不是全量微调 CLIP。主干参数保持不变，训练参数量小，能够降低显存、训练时间和 Zero-Shot
能力被破坏的风险。

## 4. Static Prompt 与 Dynamic Prompt

### 4.1 Static Prompt

使用 CoOp 式可学习上下文 token：

```text
[C1]...[Cm] background
[C1]...[Cm] normal component surface
[C1]...[Cm] normal structural edge
[C1]...[Cm] normal reflection and highlight
[C1]...[Cm] fine scratch
...
```

文本 Transformer 冻结，只更新 `C1...Cm`。初始上下文来自通用工业描述，而不是随机替换
类别词。每种状态保留不同后缀，使异常类型语义仍来自通用 CLIP 语言空间。

### 4.2 Dynamic Prompt

将冻结 CLIP 的全局图像特征输入小型 MLP，生成图像条件语义偏移：

```text
global image feature → MLP → dynamic context delta
```

该偏移作用在当前图像的状态文本原型上，使同一个 `scratch` 原型可以适应金属、塑料、透明材料、
深色表面和强反光等不同视觉条件。

## 5. CLIP Patch 特征和 Visual Adapter

使用 CLIP ViT-B/16 的 patch token，而不是只使用全局 CLS。图像在 448×448 下得到 28×28
token grid。冻结 CLIP Transformer，在输出 token 后加入残差 Adapter：

```text
LayerNorm → Linear(d,64) → GELU → Linear(64,d) → residual
```

Adapter 输出再投影到 CLIP 文本空间。它负责把通用物体语义特征调整为更适合工业微小缺陷的
局部特征。

## 6. 多尺度局部图文对齐

一次 CLIP forward 后，在 token grid 上进行窗口池化：

```text
1×1：细线、针孔、微小污点
3×3：局部划痕、缺口、裂纹
5×5：凹陷、变形、较大污染
```

每个位置与多状态文本原型计算余弦相似度：

\[
s_{p,c}=\frac{F_p T_c^T}{\tau}
\]

三种尺度使用可学习权重融合，得到连续的 patch-state logits。

## 7. 类型化伪异常和监督

训练只使用官方 Train 正常图。程序化生成器输出：

```text
pseudo_image
pseudo_mask
defect_type
severity
affected_view
```

类型映射：

| 生成类型 | 文本/Query 状态 |
|---|---|
| hairline、scratch | scratch |
| 分叉细线 | crack |
| dent | dent |
| contamination、speck | stain |
| chip、hole | chip |
| cutpaste | foreign object |
| discolour | discoloration |
| misalign | deformation |

同一五视角样本随机在 1～3 个视角注入缺陷，从而同时获得像素、视角和样本级监督。

## 8. Hard-normal 多状态监督

模型不再把所有非缺陷像素统一称为 normal，而是区分：

```text
background
normal surface
normal structural edge / repeated texture
normal reflection / highlight
```

这些区域从官方正常图本身提取。边缘、反光、孔洞和高梯度正常纹理被显式训练为正常困难负样本，
从训练阶段解决整件高亮和零件轮廓误检，而不是生成 mask 后再裁剪。

## 9. 多状态像素头

状态定义：

| ID | 状态 |
|---:|---|
| 0 | background |
| 1 | normal surface |
| 2 | normal edge / texture |
| 3 | normal reflection / highlight |
| 4 | scratch |
| 5 | crack |
| 6 | dent |
| 7 | stain |
| 8 | chip |
| 9 | foreign object |
| 10 | discoloration |
| 11 | deformation |

最终异常概率：

\[
M_{anomaly}=\sum_{c=4}^{11}P(c)
\]

背景、正常表面、边缘和反光均不计为异常，因此“零件概率”不会再被错误地当成“异常概率”。

## 10. Anomaly Query

使用八个由通用缺陷文本 embedding 初始化的 Query：

```text
scratch, crack, dent, stain,
chip, foreign object, discoloration, deformation
```

Query 通过 cross-attention 读取 CLIP patch 和 DINO 局部特征，分别产生缺陷类型响应。训练时
伪异常类型提供 Query 监督，推理时对所有异常 Query 进行 soft aggregation。

## 11. DINOv2、VLM 和 Query 动态融合

输入融合 Gate 的信息包括：

- v3 DINO dense anomaly probability；
- CLIP 多状态异常概率；
- Query 异常概率；
- CLIP 状态熵；
- DINO/CLIP 局部一致性；
- 当前类别是否有 Train memory bank。

Gate 输出逐像素权重：

\[
[w_d,w_v,w_q]=softmax(G(U))
\]

\[
M=w_dM_{dino}+w_vM_{vlm}+w_qM_{query}
\]

已见类别可更多依赖 DINO memory；未见类别在视觉 memory 不可靠时可以提高语言状态分支权重。

## 12. 五视角联合训练

五个视角不存在直接像素对应，因此不强制逐像素一致。每个视角提取：

```text
top-k pixel anomaly
strongest Query confidence
global normal/anomaly text margin
DINO/VLM agreement
```

五个视角经过轻量 Set/MIL 模块输出：

- 每个视角异常权重；
- 样本级 anomaly_score。

训练伪异常只注入 1～3 个视角，使模型学习“同一零件中哪些相机真正看到了缺陷”。

## 13. 损失函数

```text
L_total = 1.00 * L_pixel
        + 0.35 * L_patch_state
        + 0.25 * L_region_text
        + 0.30 * L_query_seg
        + 0.15 * L_global
        + 0.15 * L_multiview
        + 0.10 * L_prompt_anchor
        + 0.10 * L_cross_modal
```

| 损失 | 作用 |
|---|---|
| `L_pixel` | 保留 v3 的 BCE/Tversky/Boundary 精细分割能力 |
| `L_patch_state` | 训练 12 个像素状态 |
| `L_region_text` | 区域视觉特征和正确缺陷文本做 InfoNCE |
| `L_query_seg` | 训练对应缺陷 Query 的 mask |
| `L_global` | 样本/视角级正常异常判别 |
| `L_multiview` | 五视角 MIL |
| `L_prompt_anchor` | 约束 Prompt 不远离通用 CLIP 初始语义 |
| `L_cross_modal` | 约束高置信 DINO/VLM 区域一致，保留不确定区域互补 |

## 14. Zero-Shot 类别留出验证

将 50 个 Train 类别按名称稳定划分为五折。每折：

```text
40 类：训练 Prompt、Adapter、Query、Gate
10 类：完全不进入训练和类别 memory bank
```

留出类别只用于离线 proxy 评估，评估伪异常由确定性生成器产生，不回传梯度。模型选择目标：

```text
0.45 * heldout Pixel AP
+ 0.30 * heldout Pixel F1max
+ 0.15 * heldout Image AP
+ 0.10 * seen Pixel AP
```

最终确定结构和超参数后，再使用全部 50 类重新训练最终模型。

## 15. Qwen3-VL 的角色

Qwen3-VL 不参与主像素预测，也不生成训练伪标签。它只保留为：

- 高不确定样本低频语义复核；
- 缺陷类型和原因解释；
- DINO/CLIP 冲突区域审计；
- 答辩时的可解释性模块。

这种角色不会让粗框或离散自然语言分数破坏连续像素排序，也更容易满足单帧延迟限制。

## 16. 合规性

允许并采用：

- 官方 Train 正常图；
- 官方图像的程序化伪异常；
- 通用 DINOv2 和 CLIP 基础权重；
- 自行训练的 Prompt、Adapter、Query 和检测头。

不使用：

- MVTec、VisA 或其他外部异常数据；
- AdaCLIP/AnomalyCLIP 等异常专项权重；
- 在线 API 标注、生成 mask 或伪标签；
- Test_A/B 训练、调参或人工修正。

## 17. 独立版本与输出

本方法代码目录固定为：

```text
v6_vlm_zero_shot_v4/
```

服务器运行目录、checkpoint、日志和 submission ZIP 均使用独立名称，绝不覆盖 v1/v2/v3。

## 18. AMP 数值安全

跨模态 Gate 的最终输出是概率而不是单一分支的原始 logit。训练时先在 FP32 中执行：

```text
probability = clamp(final_map, 1e-5, 1-1e-5)
probability_logit = logit(probability)
```

然后使用 `binary_cross_entropy_with_logits`。这与概率 BCE 数学等价，并且在 PyTorch CUDA
autocast/FP16 下具有安全的反向传播；不得在 autocast 中直接调用
`binary_cross_entropy(probability, target)`。

## 19. 榜单反馈驱动的像素校准修正

v3 在 A榜上的实际变化为：

```text
P-AUROC: 0.8537 → 0.8684
P-AP:    0.4556 → 0.3546
P-F1max: 0.5124 → 0.4202
```

提交包抽样显示，v3约41.9%的像素为非零响应，而v2约5.0%；零件前景内v3约83.9%的像素
非零，形成大面积低强度响应。这会改善大范围粗排序的 AUROC，却严重损害高分区域精度和
F1max。

根因之一是旧推理链将图像级 top-k 正常分数的分位数用作像素级仿射校准边界，混淆了两个
量纲。v4必须分别拟合：

- Train正常像素分布：只用于448×448 mask校准；
- Train正常图像top-k分布：只用于anomaly_score校准。

像素校准使用固定 Train-only 全局/类别分位数，并设置固定零值底噪区间；不使用每张测试图
单独百分位归一化，也不读取测试标签。图像分数始终从未经过像素重标定的 raw dense evidence
计算，再使用独立的图像级正常分布校准。
