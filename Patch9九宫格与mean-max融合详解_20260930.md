# Patch9 九宫格与 mean/max 融合详解

**更新时间：** 2026-09-30
**面向问题：** v1–v6 是否都使用九宫格；mean 融合和 max 融合究竟融合什么、发生在哪一层、公式是什么、为什么 max 容易过分割，以及 mean 为什么能抑制误报。

## 1. 先给出明确结论

### 1.1 不是 v1–v6 从一开始就都有九宫格

需要把“版本号”和“分类器版本号”分开看：

| 名称 | 是否使用 Patch9 | 说明 |
|---|---:|---|
| `decoder-only` | 否 | 早期分割基线，只把整张图 resize 到 448×448 后推理。没有局部 crop，也没有 stitching。 |
| **v1 分割模型** | 是，首次引入 | 训练阶段加入含缺陷局部 patch；推理阶段对整图和 9 个原始分辨率 patch 都前向，再回投融合。历史提交使用 center-weighted max。 |
| **v2 分割模型** | 是 | 继承 Patch9，增加空 patch 监督，并将融合改为 `mean_w0.5`。 |
| **v3 分割模型** | 是 | 继承 Patch9，主要改变异常样本曝光/采样和训练配方。 |
| **v4 分割模型** | 是 | 继承 Patch9，改变训练步数、采样比例、学习率退火等。 |
| **v5 分割模型** | 是 | 继承 Patch9，主要加入面积加权 Focal/Tversky 类 loss。 |
| **v5-hard** | 是 | v5 的独立 hard-mining 分支，仍使用同一 Patch9 几何和 mean 融合口径。 |
| **v6 分割模型** | 是 | 继承 Patch9，主要加入 hard-normal/数据采样方面的变化；没有撤销九宫格。 |

因此，准确说法是：**v1 首次把 Patch9 引入分割主线，v2–v6 都是在 Patch9 分割框架上继续迭代；只有更早的 decoder-only 是纯整图模型。**

还要特别区分分类器：`v1.0`、`v1.1`、`v1.2`、`v2.0` 是图像分类模型的版本号，不是 Patch9 分割版本。分类器通常只对整张图运行一次；Patch9 属于分割模型的输入和推理策略。

### 1.2 “使用九宫格”有两个层面

1. **局部输入：** 从原始高分辨率图片裁剪 3×3 个重叠窗口，每个窗口单独 resize 到 448×448，送入同一个 Qwen3 视觉塔和 mask decoder。
2. **结果融合：** 将 9 个 patch 的概率图从 448×448 映射回原图坐标，与整图概率图对齐，然后合并成最终 mask。

只做局部 crop、但不进行一致的坐标回投和融合，不能称为完整的 Patch9 推理链路。

## 2. Patch9 的几何过程

设原始图像大小为 `H×W`，模型输入边长为 `S=448`。Patch9 使用窗口边长约为图像边长的 0.4，并在横向、纵向各取左/中/右三个起点，形成 9 个重叠窗口：

\[
\mathcal{B}=\{B_i=(x_i^0,y_i^0,x_i^1,y_i^1)\}_{i=0}^{8}.
\]

每个 patch 从原始分辨率裁剪，而不是从已经 resize 的 448 图像再裁剪：

\[
I_i=\operatorname{Resize}(I[B_i],S,S).
\]

这样做的目的，是让原图中的 tiny 缺陷在 patch 内占据更大的相对面积。例如原图中只有几十个像素的缺陷，在整图 resize 后可能只剩几个像素；在 0.4 倍窗口中，它的相对尺寸大约提高到原来的 2.5 倍。

整图和 patch 都经过同一个分割模型：

\[
P_f=\sigma(D(V(\operatorname{Resize}(I)))),
\]

\[
P_i=\sigma(D(V(I_i))),\quad i=0,\ldots,8,
\]

其中：

* `V` 是冻结的 Qwen3-VL 视觉塔；
* `D` 是训练得到的像素 decoder；
* `σ` 是 sigmoid；
* `P_f` 和 `P_i` 是 448×448 的异常概率图，值域为 `[0,1]`。

Patch 输出不能直接与整图像素相加，因为它们坐标系不同。首先把 patch 概率图 resize 到对应窗口大小，再放回原图坐标：

\[
\tilde P_i(x,y)=\operatorname{BackProject}(P_i,B_i)(x,y).
\]

最后将原图坐标中的结果融合，再 resize 回提交所需的 448×448 mask。

## 3. 融合发生在模型的哪一部分

mean/max 融合都属于**推理阶段的后处理/几何 stitching**，不是神经网络内部的层：

```text
整图 / 9 个 patch
        ↓
同一个 Qwen3-VL 视觉塔 + 同一个 mask decoder
        ↓
10 张 sigmoid 概率图：P_full, P_0, ..., P_8
        ↓
各 patch 回投到原始 H×W 坐标
        ↓
mean 或 max 融合
        ↓
448×448 概率 mask → uint8 mask
        ↓
分类器 gate 通过时写入提交；否则写全零 mask
```

在本项目代码中，相关位置是：

* 旧的中心加权 max：`e1/patch9.py::stitch_probabilities`；
* 后续的加权 mean：`e1/patch9_v2_geometry.py::fuse_probabilities`；
* 提交脚本：`e1/generate_patch9_v2_competition_submission.py::patch9_probability`。

融合不改变模型参数，也不产生梯度；训练 loss 看不到这个操作。它只改变最终写出的概率图，因此必须和 checkpoint、输入 patch、阈值、gate 一起记录。

还要区分另一个名字相同的 `max`：提交文件中的组级异常分数是五个视角分类分数的最大值：

\[
s_{group}=\max_{v\in\{1,\ldots,5\}}p_{yes,v}.
\]

这不是 mask 的 max fusion。本文讨论的 max 主要指**同一视角内整图/九个 patch 概率图的空间融合**。

## 4. 旧版 center-weighted max 的定义

旧代码不是简单地把 10 张图逐像素取最大，而是对每个 patch 先乘一个中心权重。设 patch 在自身坐标中归一化为 `u,v∈[-1,1]`，权重为：

\[
w_i(u,v)=\operatorname{clip}\left(1-0.35(u^2+v^2),0.55,1.0\right).
\]

中心位置权重接近 1，patch 边缘最低为 0.55。它的目的，是降低 crop 边缘的插值和上下文截断伪影。

旧版 stitching 的逐像素公式可以写成：

\[
P_{max}(x,y)=
\max\left(
P_f^{\uparrow}(x,y),
\max_{i:(x,y)\in B_i}
w_i(x,y)\tilde P_i(x,y)
\right),
\]

其中 `P_f↑` 是整图概率图回投到原图坐标后的结果。最终输出为：

\[
M_{max}=\operatorname{Resize}(P_{max},448,448).
\]

### 4.1 max 的直观含义

max 相当于一个空间上的 OR：

> 只要整图或任意一个局部 patch 认为某个像素异常，最终就保留较高的异常概率。

因此它偏向“不要漏掉任何候选区域”，特别适合第一阶段的候选发现，但它并不要求多个观察结果相互同意。

## 5. v2 之后的 mean_w0.5 定义

设 `S_i(x,y)` 是 patch `i` 是否覆盖原图像素 `(x,y)` 的支持掩码：覆盖为 1，否则为 0。重叠 patch 的算术平均为：

\[
P_{patch\_mean}(x,y)=
\frac{\sum_{i=0}^{8}S_i(x,y)\tilde P_i(x,y)}
{\max(\sum_{i=0}^{8}S_i(x,y),1)}.
\]

再将整图证据和 patch 平均证据做凸组合：

\[
P_{mean\_w\alpha}(x,y)=
(1-\alpha)P_f^{\uparrow}(x,y)+
\alpha P_{patch\_mean}(x,y).
\]

本项目常用 `α=0.5`，因此：

\[
\boxed{P_{mean\_w0.5}=0.5P_f^{\uparrow}+0.5P_{patch\_mean}}.
\]

这不是把 10 张概率图简单除以 10，也不是只平均 9 个 patch 后丢弃整图。它是：

1. 先平均所有覆盖该像素的 patch 证据；
2. 再和整图证据按 1:1 融合。

如果某个位置没有 patch 覆盖，代码保留整图证据；正常的 Patch9 几何应覆盖全图。

## 6. 为什么 max 容易导致过分割和误报

### 6.1 概率统计角度：只需一个错误高响应

假设某个背景像素在每个 patch 中都可能产生小概率误报，单个 patch 超过阈值 `t` 的概率为 `q`。如果近似独立，9 个 patch 中至少一个超过阈值的概率为：

\[
P(\text{max}>t)=1-(1-q)^9.
\]

当 `q=0.05` 时：

\[
1-0.95^9\approx0.369.
\]

单个 patch 只有 5% 的误报机会，经过 9 个 patch 的 max 后，像素被误报的机会理论上可接近 37%。真实模型中的 patch 并不独立，但重叠窗口确实会把多个“有机会误报”的观察合并为一次保留。

### 6.2 空间角度：错误响应会被回投并扩散

patch 在 448×448 上预测后被 resize 回原图窗口。一个边界模糊、面积偏大的局部响应会被完整地投回窗口；max 不会因为其他 patch 认为这里正常而把它压回去。多个窗口的错误区域还可能在重叠区拼接成连续的大块 mask。

### 6.3 训练/推理分布不一致会放大问题

早期 v1 训练主要使用整图和含缺陷正 patch，异常图中的无缺陷 patch 没有充分作为空 mask 监督。推理时却对全部 9 个窗口运行：

* 正常图的 9 个窗口可能出现局部高响应；
* 异常图中不含缺陷的窗口也可能出现局部高响应；
* max 会把任意一个窗口的高响应直接保留。

这就是为什么“Patch9 提升 tiny Recall”与“Patch9 产生过分割”可以同时成立。问题不是局部 crop 本身错误，而是 max 的 OR 逻辑没有要求局部证据一致，也没有足够的空 patch 抑制。

### 6.4 中心权重不能从根本上解决语义误报

center weight 只对 patch 边缘降权，最高仍可保留中心的错误高响应。背景纹理如果位于 patch 中心，权重接近 1，max 仍会原样保留。因此 center-weighted max 可以减轻边界伪影，却无法解决“单个 patch 误判就覆盖整张 mask”的问题。

## 7. mean 为什么能抑制这个问题

### 7.1 需要多个观察共同支持

mean 把同一像素在多个 patch 中的证据平均。一个 patch 给出 0.9、其他 patch 给出 0.1 时：

\[
P_{patch\_mean}=\frac{0.9+0.1+0.1+\cdots}{n},
\]

它不会像 max 那样直接输出 0.9。局部偶然高响应的影响被限制在 `1/n` 左右，整图证据再通过 `0.5` 权重参与融合。

### 7.2 融合结果有界，面积更稳定

mean 是凸组合：

\[
0\le P_{mean\_w\alpha}\le1.
\]

每个 patch 对最终概率的影响受 `α` 和覆盖次数共同限制。它不能凭一个局部窗口把整片背景推到高概率，因此通常带来：

* 像素 precision 上升；
* 正常图非空预测率下降；
* 预测面积和 P95 面积下降；
* Dice 在误报占主导时上升。

### 7.3 mean 的代价：可能稀释 tiny 真阳性

mean 也有明确的代价。如果 tiny 缺陷只在一个局部 patch 中被看见，整图和其他 patch 都接近 0，那么平均后它的概率会被压低：

\[
0.9\text{ 的局部响应}
\xrightarrow{\text{与多个低响应平均}}
\text{较低的最终响应}.
\]

这会降低 tiny Recall。因此 mean 不是无条件优于 max，而是把系统从“高召回、高误报”移动到“更稳定的精确定位”。`mean_w0.5` 是在整图上下文和局部放大之间的折中；v5 的面积加权 loss 也是为了补偿 mean 对小缺陷信号的稀释。

## 8. 本项目本地实测：max 与 mean 的差异

以下数据来自 `artifacts/patch9_fusion_ablation/fusion_ablation.json`，使用同一批概率图进行融合消融，因此这里比较的是**融合算法**，不是不同 checkpoint。数值为开发集诊断结果，不能直接当作独立测试集或官网成绩。

| 指标 | center-weighted max | mean_w0.5 | mean - max |
|---|---:|---:|---:|
| Macro Dice | 0.5759 | **0.6780** | **+0.1021** |
| 像素 precision | 0.5558 | **0.8733** | **+0.3174** |
| 像素 recall | **0.8130** | 0.7123 | -0.1007 |
| 像素 Dice | 0.6603 | **0.7846** | **+0.1244** |
| 平均预测面积 | 0.03139 | **0.01751** | -44.2% |
| P95 预测面积 | 0.08641 | **0.05879** | -31.9% |
| tiny Dice | 0.2075 | **0.2192** | +0.0118 |
| tiny Recall | **0.6773** | 0.2936 | -0.3837 |
| small Dice | 0.4417 | **0.5726** | +0.1309 |
| small Recall | **0.7909** | 0.5982 | -0.1926 |
| 正常图非空数 | 148 | **54** | -94 |
| 正常图误报像素率 | 0.01485 | **0.00177** | -88.1% |

这张表非常适合面试时解释：

* max 的 Recall 更高，说明它更像“候选区域并集”，不愿意漏掉局部响应；
* mean 的 precision、Dice 和面积控制明显更好，说明它抑制了单 patch 噪声和重叠叠加；
* tiny Recall 的下降说明 mean 可能把只在一个窗口中出现的微弱真阳性稀释掉，所以需要空 patch 监督、面积加权 loss 或小目标专门采样一起配合；
* “mean 比 max 好”必须带限定：**对当前目标函数（降低过分割、提高 mask 质量）更好；对单纯追求 tiny Recall，max 可能更激进。**

## 9. v1 到 v6 中 mean/max 的版本关系

| 阶段 | 训练输入 | 历史推理融合 | 统一本地消融口径 |
|---|---|---|---|
| decoder-only | 只有整图 | 无 Patch9，因此无 stitching | `full_only` |
| v1 初版 | 整图 + 正 patch；推理全部 9 patch | center-weighted max | 为公平比较，后续重算为 `mean_w0.5` |
| v2 | Patch9 + 空 patch 监督 | mean_w0.5 | mean_w0.5 |
| v3/v4/v5 | 继承 Patch9，改变采样、步数、loss | mean_w0.5 | mean_w0.5 |
| v5-hard | 继承 v5 Patch9，hard-mining | mean_w0.5 | mean_w0.5 |
| v6 | 继承 Patch9，hard-normal/数据配方变化 | mean_w0.5 | mean_w0.5 |

所以历史材料里看到“v1 max”和“v1 mean_w0.5”并不代表两套 v1 权重，而是**同一个 v1 checkpoint 的两种推理口径**。版本比较时必须注明融合方法，否则会把训练改动和后处理改动混在一起。

## 10. 面试级回答模板

### 问：mean/max 融合发生在模型的哪一层？

答：它不在 Qwen3 的 transformer 层、视觉 token 层、mask decoder 或训练 loss 中，而是在推理阶段对整图和 9 个 patch 的 sigmoid 概率图做坐标回投后的 stitching。模型先产生 10 张概率图，融合模块再生成最终 mask，最后才做 uint8 量化和分类 gate 写出。

### 问：为什么 v1 要做 Patch9？

答：整图 resize 会让 tiny 缺陷在视觉 token 网格中只剩很少像素。Patch9 从原始高分辨率图像裁剪局部窗口，让缺陷在 patch 内占据更大比例，因此提高发现和召回能力。它解决的是“看不见”，不自动解决边界和误报。

### 问：为什么不用 max？它不是能保留最强异常信号吗？

答：max 确实保留最强信号，所以 tiny Recall 往往更高；但它等价于局部 OR，只要一个窗口产生错误高响应，最终就保留该响应。9 个重叠窗口会放大这种偶然误报，导致预测面积膨胀、正常图非空和过分割。它适合候选发现，不适合作为未经校准的最终 mask。

### 问：mean 为什么更稳定？

答：mean 要求多个观察对同一像素提供平均支持，并把单个 patch 的影响限制在有限权重内；它相当于对局部预测做空间集成和降噪，抑制单 patch 的偶然高响应。因此 precision、Dice、面积和正常图误报通常改善，但 tiny 只在一个 patch 中被看见时可能被稀释。

### 问：mean 是否总是比 max 好？

答：不是。它取决于目标。如果目标是“尽可能发现候选”，max 的高召回有价值；如果目标是提交精确 mask、控制正常图误报和面积膨胀，mean 更合适。本项目采用 `mean_w0.5`，是整图上下文和局部 patch 证据的折中，并通过空 patch 监督和面积加权 loss 补偿 tiny 信号损失。

### 问：下一步怎么继续改？

答：不要直接把 max 换回来。应固定 checkpoint 做 `max / mean_w0.3 / mean_w0.5 / mean_w0.7` 扫描，同时报告 tiny Recall、tiny Dice、像素 precision、预测面积、正常图非空率。若需要恢复 tiny Recall，可优先增加空 patch 质量、tiny 采样或轻量边界/面积加权 loss，而不是用 max 重新引入大面积误报。

## 11. 参考证据

* [九宫格 Patch 分割实验计划](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/patch9_segmentation_plan.md)
* [v1 分割模型结构与训练说明](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/v1分割模型结构与训练说明_20260930.md)
* [Patch9 空 mask 模型 stitching 推理评测报告](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/Patch9空mask模型stitching推理评测报告.md)
* [Patch9 融合策略消融报告](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/Patch9融合策略消融报告.md)
* [tiny 缺陷论文解决方案与本项目借鉴方案](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/tiny缺陷问题的论文解决方案与本项目借鉴方案_20260930.md)
* [完整版本评测对比报告](/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/artifacts/version_audit_20260923/完整评测对比报告.md)
