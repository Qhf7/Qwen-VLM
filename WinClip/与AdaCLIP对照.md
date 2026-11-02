# WinCLIP ↔ AdaCLIP 对照

> WinCLIP: CVPR 2023, AWS AI Labs + KAIST (arXiv 2303.14814)
> AdaCLIP: ECCV 2024, HUST + Polimi + ZJU + 腾讯优图 (arXiv 2407.15795)

## 一句话本质区别

**WinCLIP 认为 CLIP 已经会认缺陷，问题只是「怎么问」——所以所有改动都在文本侧（写 prompt）和取特征的方式上（滑窗），一个参数都不训。**
**AdaCLIP 认为 CLIP 不懂工业缺陷，问题在「怎么教」——所以拿辅助 AD 数据训一小撮可学习 prompt / 投影层，把 CLIP 改造成缺陷检测器。**

前者 0 可训练参数，后者有参数但主干冻结。

## 主表

| | WinCLIP / WinCLIP+ | AdaCLIP |
|---|---|---|
| **输入** | 240² 图 + 物体名（可选缺陷词）+ 0~4 张正常图 | 518² 图 + 物体名；**不要参考图** |
| **输出** | 15×15 → 上采样的热力图 + 图像级分数 | 37×37 热力图（6/12/18/24 层融合）+ 图像级分数 |
| **训练目标** | **无** | 像素：Dice + Focal；图像：Focal |
| **训练数据** | **无** | 辅助 AD 数据（MVTec AD / ClinicDB 为默认；评 MVTec 时用 VisA 训）；5 epoch，lr 0.01，单卡 A6000 |
| **推理方式** | 366 个窗口各过一次 ViT（3 个 batch），取 CLS | 一张图**一次**前向（ViT-L/14@336, 518²），取 patch token |
| **文本侧** | 242 条**手写** prompt → 平均 → 2 个原型（无参数） | 可学习 prompt（static + dynamic）+ 原文本塔 → 2 个文本嵌入 |
| **视觉侧** | 滑窗 CLS，15×15 网格，每格 16 像素 | 直接取 patch token，37×37 网格，每格 14 像素 |
| **「patch 和文本不对齐」怎么解** | **绕开**：patch token 不参与，只用量身定做的窗口 CLS（pAUROC 22.4 → 85.1） | **正面解**：加 projection 层把 patch 维度对齐到文本 + prompt 微调（85.1 → 88.7） |
| **图像级怎么算** | `½(ascore₀ + max M)` 或直接 `max M` | HSF：对 patch 嵌入做 KMeans，选最异常的簇取质心 → 语义富嵌入 → 算相似度 |
| **可训练参数** | **0**（208M 全冻结） | 主干冻结；只训 prompt token（J=4 层、M=5）+ projection + DPG 线性层 |
| **硬件 / 耗时** | T4，389 ms/张 | 单卡 A6000 训练 5 epoch；推理论文没报（我的推断：单次 518² ViT-L 前向，比 366 窗口便宜） |
| **少样本** | ✅ WinCLIP+ 用 1~4 张正常图 | ❌ 没有 few-shot 版本 |
| **换域** | 完全不适应 | 会在辅助数据上过拟合（作者自己承认，附录 7.1 有失败案例） |

## 数字对照

AdaCLIP 论文表 1 里两篇被放在同一张表（WinCLIP 的数字带 † 表示引自原文）：

| 数据集 | 指标 | WinCLIP | AdaCLIP |
|---|---|---|---|
| MVTec AD | 图像 AUROC / max-F1 | **91.8 / 92.9** | 89.2 / 90.6 |
| MVTec AD | 像素 AUROC / max-F1 | 85.1 / 31.6 | **88.7 / 43.4** |
| VisA | 图像 AUROC / max-F1 | 78.1 / 80.7 | **85.8 / 83.1** |
| VisA | 像素 AUROC / max-F1 | 79.6 / 14.8 | **95.5 / 37.7** |
| 7 个工业集平均 | 图像 AUROC | 82.8 | **90.2** |
| 7 个工业集平均 | 像素 AUROC | 80.7 | **94.2** |

（7 个集是 MVTec AD、VisA、MPDD、BTAD、KSDD、DAGM、DTD-Synthetic；平均值 WinCLIP 那列是我按表里逐项数字算的。）

## 三个反直觉的点

1. **MVTec 的图像级，零训练的 WinCLIP 反而更高（91.8 vs 89.2）。** AdaCLIP 在图像级上做了 HSF 这种复杂模块，结果在 MVTec 上还不如手写 prompt。它的优势主要在**像素级**（88.7 vs 85.1）和 **max-F1**（43.4 vs 31.6，差 12 个点）。
2. **AdaCLIP 的涨点里有相当一部分不是方法的功劳。** 它用 ViT-L/14@336 + 518²，WinCLIP 用 ViT-B/16+ + 240²。Backbone 大一号、分辨率高一倍，本身就能涨几个点。AdaCLIP 自己也补了 "不复现 AnomalyCLIP" 的说明，比较并不完全 apples-to-apples。
3. **WinCLIP 在 VisA 上的崩塌（像素 79.6）是 AdaCLIP 最大的机会。** VisA 里 4 个 PCB 子集，靠一句「正常/损坏的电路板」根本区分不开；而这恰恰是「用辅助数据学通用异常模式」最擅长的场景。

## 怎么选（按你的约束）

| 你的情况 | 选 | 原因 |
|---|---|---|
| 一个训练样本都没有、要立刻跑起来 | **WinCLIP** | 0 训练，拿 CLIP 权重就能出分 |
| 有别的类别的「图 + 像素标注」，且要像素级准 | **AdaCLIP** | 像素 max-F1 43.4 vs 31.6，这是实打实的差距 |
| 目标类有 1~4 张正常图、**没有异常标注** | **WinCLIP+** | AdaCLIP 用不上这几张图（它不需要参考图，也没有 few-shot 版） |
| 有正常图 + 有异常标注 + 有算力 | **AnomalyCLIP / AA-CLIP 一类** | AdaCLIP 已经不是 SOTA；2025 年的 AA-CLIP 像素 91.9、图像 90.5 |
| 要可解释「为什么判它异常」 | WinCLIP 略优 | 分数直接来自「像不像 damaged」这句话，AdaCLIP 的 prompt 是学出来的向量，说不出来 |
