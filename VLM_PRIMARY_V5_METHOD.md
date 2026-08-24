# VLM-Primary v5：跨模态零样本工业异常检测

## 1. 目标

本版本从 v4 继续演进，但针对 v4 在 A 榜出现的三类退化进行结构性修复：

1. 50 类中只有前 10 类参与校准；
2. Train-only Proxy 与最终提交推理链路不一致；
3. 无约束 DINO/VLM/Query Gate 破坏了原有像素和图像排序。

v5 仍属于赛事范式二：VLM 是主要异常判断模型，DINOv2 只提供上限为 15% 的定位残差。

## 2. 论文思想与合规实现

| 论文方向 | 采用的思想 | 本项目合规实现 |
|---|---|---|
| WinCLIP | compositional prompts、多尺度局部窗口 | CLIP 1×1/3×3/5×5 patch-window 与正常/异常文本对齐 |
| WinCLIP+ | normal reference association | 官方 Train 正常图建立类别/视角及全局 CLIP patch reference bank |
| AdaCLIP | static + dynamic hybrid prompts | 冻结 CLIP 主干，只训练 context、动态 Prompt MLP和轻量 patch adapter |
| AnomalyCLIP 类方法 | object-agnostic normal/abnormal text space | 类别无关的工业正常状态和缺陷状态提示集合 |

不使用任何 MVTec、VisA 或其他异常数据训练权重，不调用在线模型服务，不使用 Test 标签。

## 3. v5 核心结构

### 3.1 VLM 主分支

通用 CLIP ViT-B/16 提取：

- global image embedding；
- 448 输入下的 dense patch embedding；
- 1×1、3×3、5×5局部窗口特征。

正常与异常状态不再使用“异常类别概率求和”。v4 有8个异常状态、3个正常状态，直接求和天然偏向异常。v5 使用数量无关的 log-mean-exp margin：

```text
semantic_logit = logmeanexp(anomaly similarities)
                 - logmeanexp(normal similarities)
```

重复增加同义 Prompt 不会改变先验概率。

### 3.2 WinCLIP+ 正常参考

使用官方 Train 正常图的 CLIP patch feature 建立：

- category-view reference bank；
- category-agnostic global reference bank。

已见类别使用 category-view bank；B榜未见类别自动使用 global bank。

### 3.3 DINOv2 有界残差

DINOv2 多尺度 decoder 从 v4/v3 继承，但不再参与无约束动态 Gate：

```text
final_map = (1 - alpha) * VLM_map + alpha * DINO_map
0 <= alpha <= 0.15
```

因此方法保持 VLM 主导，同时保留 DINO 对细小边界的定位能力。

### 3.4 五视角聚合

五视角不做像素配准。每个视角由 VLM dense、VLM global、CLIP normal reference 和少量 DINO evidence 产生分数，再以 strongest-view + top-2 mean 聚合。

## 4. 训练策略

冻结：

- CLIP image backbone；
- CLIP text backbone；
- DINOv2 backbone；
- DINO decoder。

训练：

- Static Prompt context；
- Dynamic Prompt MLP；
- CLIP patch adapter/projection；
- VLM local state decoder；
- semantic/reference/DINO 三个有界标量。

训练异常只来自官方正常图的程序化编辑。

训练生成器：

```text
hairline, scratch, speck, dent, chip, contamination, cutpaste
```

Proxy 留出生成器：

```text
crack, discolour, misalign, hole
```

留出类别与留出生成器同时验证，降低“识别自家伪异常”带来的虚高。

## 5. 统一 Proxy 和正式推理

`pipeline_v5.py` 是唯一权威推理入口。Proxy 和正式提交共同调用：

1. CLIP/DINO特征提取；
2. CLIP reference association；
3. VLM primary map；
4. bounded DINO residual；
5. object prior；
6. five-view confidence consistency；
7. image aggregation；
8. robust calibration。

不再允许 Proxy 绕过正式后处理。

## 6. 校准修复

每个类别均匀选择10组正常样本，50类共500组。代码强制断言50类全覆盖，缺少任何类别即停止。

校准不再使用硬截断 min-max，而采用严格单调 logistic 映射：

- 正常中位数映射到约0.1；
- 正常高分位映射到约0.9；
- 超过高分位的样本继续保持可排序，不会全部裁成1。

这直接修复 v4 中大量分数饱和为0或1的问题。

## 7. 模型选择

Fold-0 依次评估：

- epoch 0：不微调 VLM；
- epoch 1：轻量微调1轮；
- epoch 2：轻量微调2轮。

选择指标：

```text
0.35 * Pixel AP
+ 0.30 * Pixel F1max
+ 0.20 * Pixel AUROC
+ 0.10 * Image AP
+ 0.05 * Image AUROC
```

如果 epoch 0 最好，最终模型不会强行微调；避免“训练损失降低但真实泛化下降”。

## 8. 可靠性

- 每50步原子保存模型、AdamW、AMP、scheduler、样本顺序及所有 RNG 状态；
- 中断后精确恢复；
- 所有输出使用新目录；
- 推理中断会保留部分结果后重跑；
- ZIP 独立校验750行、3750张448×448单通道mask及路径完整性。
