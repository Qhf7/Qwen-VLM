# B0 Baseline 分析与下一步

> Baseline ID：`B0-smoke-v1`
> 完成日期：2026-09-01
> 状态：**冻结为工程冒烟基线和零样本负对照；尚不是全量 Dev-mini 统计基线**

---

## 1. 结论先行

本次 B0 在 MacBook Pro M3 Pro 18 GB 上完整验证了两个 Qwen-VL 模型的本地
MPS + FP16 推理链路：

1. 448×448 图片预处理正常；
2. 可从第一个生成位置读出 Yes/No next-token logits；
3. 可对同一零件的 5 视角做 max 聚合；
4. `Yes/No + JSON region` 契约解析率为 100%；
5. 单视角和 5 视角端到端延迟均已实测；
6. 2B 和 3B 都能在本机跑完，B0 推理不需要租服务器。

但从模型能力看，两者的零样本**硬判定都已退化**：

- `Qwen3-VL-2B-Instruct` 对 100 个视角全部生成 `Yes/center`；
- `Qwen2.5-VL-3B-Instruct` 对 100 个视角全部生成 `No/none`；
- 两者在 0.5 阈值下的样本级 accuracy 都是 50%。

3B 虽然全部生成 No，但 `P(Yes)` 仍有排序信息：5 视角 max 后 AUROC
为 0.7250，高于 2B 的 0.5400。因此冻结如下决策：

- **B0+ 准确性主线：`Qwen2.5-VL-3B-Instruct`**；
- **延迟对照：`Qwen3-VL-2B-Instruct`**；
- 后续不使用生成的 Yes/No 当分数，一律使用连续 token 概率；
- 当前位置词只是负对照，不能作为定位能力证据；
- 下一阶段仍然是 **S2/B0+ 零训练消融**，不是 E1，更不是 LoRA。

---

## 2. Baseline 的实验定义

### 2.1 数据范围

数据来自冻结的 `artifacts/splits/fold1/dev.parquet`，没有访问 `holdout-zs`。
按类别中位分辨率覆盖选出 10 类，每类选 1 个 OK 和 1 个 NG 完整样本：

```text
smd_receiver_module         battery_holder_connector
slide_switch                accurate_detection_switch
mobile_charging_connector   jam_jar_model
lilypad_led                 miniature_laser_module
PLCC_socket                 3_adapter
```

最终共：

- 10 个类别；
- 20 个完整样本（OK 10 / NG 10）；
- 每个样本固定 5 视角；
- 100 张不重复图片；
- 两个模型的 `image_path` 集合经程序断言完全一致。

冻结子集：

- `artifacts/b0/smoke20_manifest.parquet`
- `artifacts/b0/smoke20_manifest.csv`

### 2.2 输入与视觉 token

所有图片都先用 OpenCV `INTER_AREA` 直接缩放为 448×448，processor 的
`min_pixels` 与 `max_pixels` 也同时固定为 `448²`。

| 模型 | 实际视觉网格 | 说明 |
|---|---:|---|
| Qwen3-VL-2B | `1×28×28` | 448 输入已生效；该代视觉 patch 规格不同 |
| Qwen2.5-VL-3B | `1×32×32` | 448 / 14 = 32 |

两者网格不同不表示输入尺寸不一致，而是模型自身视觉 patch 定义不同。

### 2.3 输出契约

模型被要求生成：

```text
Yes {"region":"center"}
```

或：

```text
No {"region":"none"}
```

其中 `region` 只允许 3×3 九个位置词与 `none`。本 B0 遵循
`VLM训评路线-decoder-adapter-LoRA.md` 中收窄后的输出契约，**不让 VLM
生成多边形 mask**。因此 Pixel AUPR / Pixel F1 在 B0 中记为 `N/A`，
不用粗格子伪装像素分割结果。

### 2.4 Next-token 分数

tokenizer 中 Yes/No 均有“有前导空格”和“无前导空格”两个单 token 候选：

```text
Yes token ids = [7414, 9454]
No  token ids = [2308, 2753]
```

对第一个生成位置的 logits，先对同义 token 做 `logsumexp`，再在 Yes/No 之间归一化：

```text
L_yes = logsumexp(logits[7414], logits[9454])
L_no  = logsumexp(logits[2308], logits[2753])
P(Yes) = softmax([L_no, L_yes])[1]
```

这使 AUROC / AUPR 依赖连续分数，而不依赖最终生成的离散文字。

### 2.5 5 视角聚合

遵循 S0 冻结口径，一个完整样本的异常分数为：

```text
sample_score = max(P_yes_C1, P_yes_C2, ..., P_yes_C5)
```

这与“任一视角可见缺陷即样本为 NG”的样本标签定义一致。本轮没有调整聚合方式。

### 2.6 延迟口径

单视角 `total_s` 包含：

```text
读图 + direct 448 resize + processor + autoregressive generate + 解析
```

探针与断点续跑各会触发一次 MPS 图编译。因此：

- 冷启动单独报首视角；
- warm 单视角与 warm 5 视角的 p50/p95 排除首个完整样本；
- 模型加载时间不混入单样本延迟。

---

## 3. 完整结果

| 指标 | Qwen3-VL-2B | Qwen2.5-VL-3B |
|---|---:|---:|
| 实际参数 | 2.1275B | 3.7546B |
| 设备 / dtype | MPS / FP16 | MPS / FP16 |
| 样本 / 视角 | 20 / 100 | 20 / 100 |
| 首词 Yes/No 解析率 | 100% | 100% |
| JSON + 合法位置词率 | 100% | 100% |
| 生成 Yes 比例 | **100%** | **0%** |
| 视角级 accuracy@0.5 | 50% | 50% |
| 样本级 accuracy@0.5 | 50% | 50% |
| 5-view I-AUROC | 0.5400 | **0.7250** |
| 5-view I-AUPR | 0.5785 | **0.7879** |
| 5-view F1max | 0.6667 | **0.7500** |
| 10 类宏平均 I-AUROC | 0.5000 | **0.7000** |
| 异常视角位置词准确率 | 48% | 0% |
| 模型加载 | 11.1 s | 65.8 s |
| 冷启动首视角 | 15.45 s | 56.26 s |
| warm 单视角 p50 / p95 | 2.89 / 3.42 s | 4.06 / 5.07 s |
| warm 5 视角 p50 / p95 | 14.48 / 16.14 s | 21.30 / 23.62 s |
| `≤1 s / 完整样本` | 未达成 | 未达成 |

原始产物：

- `artifacts/b0/qwen3_vl_2b_views.csv`
- `artifacts/b0/qwen3_vl_2b_config.json`
- `artifacts/b0/qwen3_vl_2b_report.txt`
- `artifacts/b0/qwen25_vl_3b_views.csv`
- `artifacts/b0/qwen25_vl_3b_config.json`
- `artifacts/b0/qwen25_vl_3b_report.txt`
- `artifacts/b0/comparison.csv`
- `artifacts/b0/comparison.md`

---

## 4. 如何解读分类结果

### 4.1 “全部答错方向”与 AUROC 不矛盾

accuracy@0.5 测试的是**校准与固定阈值**，AUROC 测试的是**排序**。

3B 所有 `P(Yes)` 都小于 0.5，因此它生成的硬答案全是 No；但 NG 样本的分数
在多数类别中仍高于 OK，所以 AUROC 可以达到 0.725。这说明：

- 3B **存在弱异常排序信号**；
- 它的 Yes/No 对比没有在工业数据上校准；
- 后续不能用 0.5 作为默认工业判定阈值；
- 调阈值只能改 accuracy/F1，不会改 AUROC，不能把调阈值写成识别能力提升。

### 4.2 2B 的 AUROC 基本不可信

2B 的视角级 `P(Yes)` 范围只有：

```text
0.99998987 ~ 0.99999928
```

其动态范围不到 `1e-5`。OK/NG 的样本级中位数分别是：

```text
OK = 0.99999762
NG = 0.99999779
```

AUROC 0.54 完全依赖极小的浮点排序差，很容易被 prompt 、dtype 或硬件数值抖动
翻转。它不应被解读为“2B 有稳定的异常检测能力”。

### 4.3 3B 的排序信号更健康

3B 的视角级 `P(Yes)` 范围为：

```text
0.10819 ~ 0.36298
```

样本级中位数：

```text
OK = 0.20453
NG = 0.22406
```

10 个类别中，3B 在 7 类上满足 `score(NG) > score(OK)`；2B 只有 5 类。
这就是宏平均 AUROC 分别为 0.7 和 0.5 的直接含义。

3B 排序正确的 7 类：

```text
slide_switch, battery_holder_connector, mobile_charging_connector,
accurate_detection_switch, smd_receiver_module, PLCC_socket, lilypad_led
```

3B 排序反向的 3 类：

```text
jam_jar_model, miniature_laser_module, 3_adapter
```

`3_adapter` 的 `NG - OK` 分数差为 -0.0523，是最明显的反向类，同时也是大分辨率类。
这是 B0w 保留原尺度局部信号的一个明确动机，但尚不能只凭一个样本断定是 resize 导致。

### 4.4 当前样本量不足以支撑泛化结论

当前只有 10 个 OK 和 10 个 NG 样本。使用固定随机种子进行 10,000 次分层
bootstrap，得到粗略 95% 区间：

| 模型 | AUROC | AUROC 95% bootstrap | AUPR | AUPR 95% bootstrap |
|---|---:|---:|---:|---:|
| Qwen3-VL-2B | 0.5400 | 0.275 ~ 0.790 | 0.5785 | 0.425 ~ 0.808 |
| Qwen2.5-VL-3B | 0.7250 | 0.470 ~ 0.940 | 0.7879 | 0.609 ~ 0.945 |

3B 的点估计更好，但 AUROC 区间仍覆盖 0.5。因此本轮只允许得出：

> 3B 是进入 B0+ 的更好候选，不允许得出“3B 已经解决工业异常检测”。

---

## 5. 如何解读位置结果

2B 的 100 个预测位置全是 `center`，3B 的 100 个预测位置全是 `none`。
异常视角的 GT 位置分布为：

```text
center        24 / 50
top-center     7 / 50
middle-left    6 / 50
middle-right   4 / 50
其他五个格子    9 / 50
```

因此 2B 的 48% 位置准确率正好等于 `center` 在 GT 中的占比，它只是
**永远猜中心的类别先验**，不是定位能力。3B 在异常图上也全部回答 `none`，
位置准确率为 0。

这个结果与既有路线预测完全一致：

- 整图语义掩盖了小缺陷；
- 生成模块不能提供可信的精细定位；
- 后续局部信号应来自 decoder / crop，不能靠 LLM 猜格子或生成坐标；
- B0 的位置词只作输出契约和负对照保留。

---

## 6. 如何解读延迟

### 6.1 自回归生成是主要瓶颈

warm 阶段的耗时中位数：

| 模型 | resize | processor | generate | total | generate / total |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 0.022 s | 0.015 s | 2.853 s | 2.894 s | 约 98.6% |
| Qwen2.5-VL-3B | 0.026 s | 0.024 s | 4.007 s | 4.059 s | 约 98.7% |

图片解码、resize 和 processor 不是当前瓶颈；自回归生成占端到端耗时约 99%。
因此下一步正确的延迟实验不是优化 OpenCV，而是：

1. 只做一次 prefill / forward；
2. 直接读最后一个 prompt 位置的 Yes/No logits；
3. 不生成后续 JSON；
4. 尝试把同一样本的 5 视角做 batch prefill。

这是一个只改推理方式、不改模型与分数定义的独立消融，应在继续扩大
数据量之前先完成。

### 6.2 当前方案距离 1 s 目标很远

3B 的 warm 5 视角 p50 为 21.30 s，比 1 s 目标慢 21 倍；2B 也慢 14 倍。
这证明“5 视角逐张自回归生成”不能是最终工业推理形态。

这不否定 VLM 训练路线，但它对 S7 的结论已很明确：

- 推理期优先只跑 decoder + 轻量图级头；或
- VLM 只做一次 prefill 读 token 概率，不进入自回归生成；或
- 最后将 VLM 的软标签蒸馏到轻量模型。

---

## 7. Baseline 的成立范围与限制

### 7.1 已经成立的结论

- 本机能跑 2B 和 3B FP16 视觉模型推理；
- direct 448 预处理与 processor 链路正常；
- Yes/No logits 、5 视角 max 和 JSON 解析可重现；
- 两个通用 VLM 的零样本硬判定都不可直接使用；
- 3B 在当前子集上的连续排序信号优于 2B；
- VLM 粗位置生成没有证明定位能力；
- 自回归生成无法满足 1 s / 完整样本。

### 7.2 不能由当前结果推出的结论

- 不能说 3B 在 24 个类别上都能达到 0.725 AUROC；
- 不能说 3B 已经具有未见类别泛化能力；
- 不能将 2B 的 48% 位置准确率解读为定位能力；
- 不能用 20 个样本完成 prompt 选择后，仍把同一份数字宣称为未调参结果；
- 不能用 B0 粗格子代替 decoder 的 Pixel AUPR / Pixel F1。

### 7.3 为什么它仍然可以作为 baseline

Baseline 的用途不只是提供一个高分，还要冻结一个可归因的起点。
本 B0 已经冻结了：

```text
数据子集 + 448 输入 + prompt + token 分数 + 5-view max + 输出契约 + 延迟口径
```

它能作为之后 B0p / B0f / B0w 的同子集负对照。它不能替代全量 Dev-mini
上的最终 B0/B0+ 行。

---

## 8. 按既有路线，下一步应该怎么做

### 8.1 结论：先完成 S2/B0+，不开 E1

根据 `README.md` 、`研究路线.md` 和 `VLM训评路线-decoder-adapter-LoRA.md`，
阶段顺序必须是：

```text
B0 冒烟基线
  ↓
B0+ 零训练天花板（一次只改一项）
  ↓
准备 E1 训练数据（特别是 hard-normal）
  ↓
E1 Decoder + Prompt Learner
  ↓ 只有 E1 过线
E2 Visual Adapter
  ↓ 只有 E2 过线
E3 LLM LoRA
```

现在直接开 decoder 会产生一个无法回答的问题：E1 涨分究竟来自 decoder，
还是只因为之前的 prompt / 推理方式太差？所以 B0+ 是必须的归因屏障。

### 8.2 立即执行：B0f（forward-only）

这是下一个应该编码和运行的实验。

**只改一项**：用一次前向的最后 prompt token logits 替代
`generate(max_new_tokens=32)`；其他全部不变。

先在当前 100 视角上验证：

```text
max_abs(P_yes_forward - P_yes_generate) < 1e-5
样本级 AUROC / AUPR 与 B0 一致（只允许浮点误差）
```

然后测：

```text
单视角 forward-only p50/p95
5 视角串行 p50/p95
5 视角 batch prefill p50/p95
```

原因：当前耗时约 99% 来自 generate。如果 B0f 不先做，后面在全量 Dev-mini
上重复测 prompt 会浪费大量时间。

JSON 位置词在 B0f 中不再全量生成，只在固定的定位诊断子集上保留。
这不改变图级分数，也避免把无信息的 `center/none` 生成带入最终推理链。

### 8.3 第二步：B0p（prompt 独立消融）

在同一个 `smoke20` 上预先写死 3 个 prompt，使用 3B + forward-only 分别跑：

1. `P0`：当前 B0 prompt；
2. `P1`：对称定义 normal / anomalous，删去会诱导默认 Yes 的长缺陷列表；
3. `P2`：类别名无关的双假设对比，用 `<object>` 而不是产品名。

每个 prompt 独立一行，不同时改 resize、聚合或模型。选择时看：

- I-AUROC / I-AUPR / F1max；
- OK/NG 分数分布与动态范围；
- 10 类中 `NG > OK` 的类别数；
- 是否仍有“全 Yes”或“全 No”的硬答偏置。

不允许先看结果再无限改 prompt。三个模板全部预注册，三行都保留。

### 8.4 第三步：聚合方式离线消融

使用已保存的视角分数直接计算，不重跑模型：

```text
A0 = max(C1...C5)             # 当前基线
A1 = mean(top2(C1...C5))
A2 = mean(C1...C5)            # 诊断项，不预设更好
```

这三行只归因于多视角聚合。S0 的主口径仍然是 max；只有 top-2 mean 在更大
Dev-mini 上稳定更好时，才能建议更改主口径。

### 8.5 第四步：B0w（局部窗口）

只使用 B0p 中冻结的最佳 prompt，不再改文本。建议分两档：

1. **B0w-5crop**：一个全局 448 + 四个确定性局部 crop，用 max 聚合；
2. **B0w-native**：原尺度 448 滑窗，25% overlap，只在按分辨率和缺陷面积分层的
   小子集上跑。

S0 已测得 B0w-native 的每个 5 视角样本窗口数 p50=42、p95=174，因此不能
一开始就对 720 个 Dev-mini 样本全量扫描。只有 B0w-5crop 对大图类和小缺陷确实
有稳定收益，才打开 B0w-native。

B0w 必须单独报延迟。即使分数上升，若延迟成倍增长，也只能作为“局部信号
有用”的研究证据，不能直接作为部署方案。

### 8.6 第五步：扩展到冻结的全量 Dev-mini

`fold1/dev.parquet` 共有：

```text
18 类 × (20 OK + 20 NG) 完整样本
= 720 个样本
= 3600 个视角
```

在 `smoke20` 上完成 B0f / B0p / 聚合 / B0w 筛选后：

1. 冻结最佳零训练配置；
2. 只将冻结的 B0 和 B0+ 候选扩展到 720 样本；
3. 报 18 类宏平均指标和 bootstrap 区间；
4. 分大图类 / 小图类、大缺陷 / 小缺陷分层报告；
5. 这一行才是后续 E1 必须打败的**最终 B0+ 统计基线**。

按当前 3B 自回归速度，3600 视角约需 4.1 小时，所以 B0f 必须先做。

### 8.7 在进入 E1 之前，补齐训练数据

B0+ 配置冻结后，仍然不能立即开训。按既有路线，E1 前必须准备：

- 真实 NG 视角 + 真实 mask；
- OK 视角；
- 从正常边缘、反光、孔洞、文字、规律纹理中挖的 **hard-normal**；
- Cut-Paste / NSA 伪异常及其近似 mask；
- 类别均衡、OK/NG 均衡、缺陷面积分档的 sampler；
- 视角级标签严格以非空 mask 为准，491 张无有效 mask 的 NG 视角按正常处理。

**没有 hard-normal 不允许开 E1。** 否则 decoder 很容学成边缘、反光或物体性。

### 8.8 E1 的预注册过线条件

原文档使用“明显高于”和“同方向可复现”，在开训前应把它量化。建议冻结为：

```text
主决策：Dev-mini 宏平均 I-AUPR 相对最终 B0+ 绝对提升 ≥ 0.03
同时：宏平均 F1max 不下降
稳定性：至少 3/4 个冻结折为正增益
诊断：小缺陷档召回不下降，预测面积占比不爆炸
```

这个 0.03 是开训前的建议预注册阈值，不是已有文档规定的真理。如果要修改，
必须在看 E1 结果前修改并留痕。

E1 不过线时，只检查 hard-normal、decoder 取层和 `λ_seg`，不允许用“加 Visual Adapter”
或“加 LoRA”来遮盖失败。

---

## 9. Holdout 使用纪律的口径修正

既有文档同时存在两句彼此张力很大的要求：

1. `holdout-zs` 全程不调参、只在结论已定时看一次；
2. E1 / E2 / E3 是否开下一刀，又要看 holdout 是否提升或掉点。

如果真的每轮都用 holdout 做开关，holdout 就已变成 dev，不再是零样本裁判集。
为保持项目最核心的防自欺纪律，从本 baseline 起冻结为：

- B0p / B0w / 阈值 / 聚合 / ckpt 全部只在 train + dev-mini 上决定；
- E1 是否进 E2、E2 是否进 E3，也只看 dev-mini 和冻结折的一致性；
- B0+、E1、E2、E3 配置全部冻结后，再一次性跑每折 `holdout-zs`；
- holdout 只回答最终的零样本泛化问题，不再反向修改模型。

这是对既有路线中一个潜在口径冲突的显式修正，应在进入 E1 前同步回写到
`研究路线.md` 和 `VLM训评路线-decoder-adapter-LoRA.md`。

---

## 10. 下一阶段的执行清单

### 现在就做

- [x] 实现 `B0f` forward-only logits；
- [x] 验证 B0f 与 generate 首 token 分数一致；
- [x] 实测单视角与 5 视角 batch prefill 延迟；
- [x] 预注册 P0/P1/P2 三个 prompt；
- [x] 在 `smoke20` 上跑 prompt 独立消融；
- [x] 离线计算 max / top-2 mean / mean 聚合对照；
- [x] 只用最佳 prompt 跑 B0w-5crop；

### 筛选完成后做

- [x] 冻结 B0+ smoke20 候选配置；
- [x] 在 720 个 Dev-mini 样本上跑最终 B0+；
- [x] 生成 18 类宏平均、置信区间、分辨率/面积分层表；
- [x] 准备 hard-normal 清单与可复现生成脚本；
- [x] 构建 E1 平衡 sampler；
- [x] 在看到 E1 结果前冻结 E1 过线条件；

### 现在不做

- [ ] 不开 Visual Adapter；
- [ ] 不开 LLM LoRA；
- [ ] 不看 `holdout-zs`；
- [ ] 不同时改 prompt + crop + aggregation；
- [ ] 不用 Pixel AUROC 选模型；
- [ ] 不把 3×3 位置词写成像素分割结果。

---

## 11. B0+ smoke20 执行结果（2026-09-01）

本节是在第 10 节清单全部执行完后追加的结果，不回改预注册规则。模型、图片顺序、
`smoke20` 和 direct 448 均未改变，`holdout-zs` 未读取。

### 11.1 B0f 与延迟

正式 B0f 使用 `use_cache=False`、`logits_to_keep=1`，只读取 prefill 最后位置 logits。
固定 5 张图上的验证结果：

```text
max |P_yes(forward) - P_yes(generate first token)| = 4.87e-06
max |P_yes(batch5)  - P_yes(single)|               = 0
```

均通过预注册容差。模型已预热后的实测延迟为：

| 路径 | 重复数 | 总延迟 p50 | 总延迟 p95 | forward p50 | forward p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单视角 B0f | 100 | 1.875 s | 3.398 s | 1.816 s | 3.368 s |
| 5 视角 batch prefill | 5 | 28.757 s | 44.400 s | 27.742 s | 43.935 s |

按单图 p50 估算，5 次串行为 9.373 s；真实 batch5 反而慢到 3.07 倍。
这说明 M3 Pro 18GB 的 5 图 batch 已碰到统一内存压力，不能当作加速路径。
后续全量评分使用数值上已验证等价的单图串行 B0f；batch5 只保留为工程负结果。

### 11.2 Prompt 独立消融

三个 prompt 在推理前写入 `B0plus-preregistered-v1`，选择规则固定为：5-view max，
依次按 AUROC、AUPR、宏 AUROC 降序，完全相同时按 prompt ID。

| Prompt | AUROC | AUPR | F1max | 宏 AUROC | Yes率@0.5 | NG>OK 类数 | 分数跨度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | **0.7250** | 0.7879 | 0.7500 | **0.7000** | 0 | **7/10** | **0.2006** |
| P1 | 0.5550 | 0.5299 | 0.7407 | 0.6000 | 0 | 6/10 | 0.0506 |
| P2 | 0.6900 | **0.8101** | 0.7500 | 0.6000 | 0 | 6/10 | 0.1709 |

预注册规则选择 **P0**。P2 的 AUPR 更高，但第一排序键 AUROC 低于 P0，不能事后换规则。
P1 的对称措辞显著压缩分数动态范围。三者在固定 0.5 阈值下仍全部判 No，说明 prompt
没有解决校准问题；有效信息来自连续 logits 排序，不来自硬答案。

### 11.3 多视角聚合（离线，不重跑模型）

| Prompt | 聚合 | AUROC | AUPR | F1max | 宏 AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | max | 0.7250 | 0.7879 | 0.7500 | 0.7000 |
| P0 | top-2 mean | 0.7500 | 0.8001 | **0.7826** | 0.7000 |
| P0 | mean | **0.7700** | **0.8261** | 0.7619 | 0.7000 |
| P1 | max / top-2 / mean | 0.5550 / 0.5500 / 0.5700 | 0.5299 / 0.5344 / 0.5454 | 0.7407 / 0.7407 / 0.7407 | 0.6000 / 0.6000 / 0.7000 |
| P2 | max / top-2 / mean | 0.6900 / **0.7400** / 0.6900 | 0.8101 / **0.8318** / 0.7487 | 0.7500 / 0.7500 / 0.7143 | 0.6000 / **0.8000** / 0.8000 |

P0 mean 在 smoke20 上最好，但这只是 20 样本诊断，不能据此推翻 S0 的 max 主口径。
扩展到 720 样本时只需推理一次 P0，再离线复算三种聚合；只有结果稳定才改主口径。

### 11.4 B0w-5crop 与阶段门

只用预注册选出的 P0，输入改为全局图加原图四象限，每个 crop 独立缩放至 448，
先对 5 crop 取 max，再对 5 视角取 max：

| 配置 | AUROC | AUPR | F1max | 宏 AUROC | 5视角延迟 p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P0 direct | **0.7250** | **0.7879** | **0.7500** | 0.7000 | — |
| P0 5-crop | 0.5700 | 0.6695 | 0.7143 | 0.7000 | 39.324 / 50.294 s |

全局 crop 与 direct 的 100 个视角分数逐项完全一致，排除了实现差异。但四象限在
97/100 个视角上给出最高异常分；样本分数相对 direct 的中位增量，OK 为 0.0801，
NG 仅为 0.0527。局部放大主要强化了正常边缘和纹理的伪异常，而不是真缺陷。

因此 B0w-5crop 的 AUROC 下降 0.155、AUPR 下降 0.1184，同时延迟大幅增加。
按照第 8.5 节预先写明的门槛，**停止 B0w，不打开 B0w-native**。

### 11.5 本轮冻结决策与下一步

`smoke20` 的零训练候选冻结为：Qwen2.5-VL-3B、P0、direct 448、B0f；max 仍为
主聚合，top-2 mean/mean 作为同一次推理的离线候选。下一步严格按路线执行：

1. 在 `fold1/dev.parquet` 的 720 个样本、3600 个视角上只跑一次冻结的 P0 B0f；
2. 离线报告 max/top-2 mean/mean、18 类宏平均、bootstrap 区间及分辨率/缺陷面积分层；
3. 冻结最终 B0+ 统计基线，不运行 B0w-native；
4. 准备 hard-normal、Cut-Paste/NSA、真实 OK/NG+mask 和平衡 sampler；
5. 在看到 E1 结果前冻结 `I-AUPR +0.03、F1max 不降、至少 3/4 折正增益` 的阶段门；
6. 然后才进入 E1 Decoder + Prompt Adapter；此时仍不开 Visual Adapter、LoRA，
   也不读取 `holdout-zs`。

完整可复核报告见 `artifacts/b0plus/B0plus_results.md`，原始概率、逐项延迟和协议文件
均在 `artifacts/b0plus/`。

---

## 12. B0+ Dev-mini 720 最终统计基线（2026-09-01）

冻结配置为 `Qwen2.5-VL-3B-Instruct + P0 + direct448 + B0f`。Dev-mini
仍为 18 类，每类 20 OK + 20 NG，共 720 个完整样本和 3600 个唯一视角。
输入缓存逐像素哈希、样本数、视角数、类别数和 holdout 交集均通过断言。

| 聚合 | Overall AUROC | Overall AUPR | Overall F1max | 18 类宏 AUROC | 18 类宏 AUPR | 18 类宏 F1max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max（冻结主口径） | 0.6674 | 0.6697 | 0.6781 | **0.7208** | **0.7495** | **0.7380** |
| top-2 mean（离线） | **0.6763** | **0.6735** | 0.6805 | 0.7400 | 0.7648 | 0.7565 |
| mean（离线） | 0.6715 | 0.6525 | **0.6838** | **0.7483** | **0.7734** | **0.7623** |

`max` 的 2000 次分层 bootstrap 95% 区间为：宏 AUROC
`[0.6812, 0.7572]`，宏 AUPR `[0.7277, 0.7924]`，宏 F1max
`[0.7403, 0.7821]`。bootstrap 中 F1max 每次都重新选阈值，因而其中位数
不必等于原始样本点估计。

top-2 mean 相对 max 的宏 AUROC/AUPR/F1max 点差为
`+0.0192/+0.0152/+0.0185`；mean 为 `+0.0275/+0.0239/+0.0242`。
但 mean 的 pooled AUPR 反而下降 `0.0172`，说明类间标定仍不一致。因此三种结果
全部保留，但不根据同一份 Dev 数据事后把主口径从 max 改成 mean。

分层结果显示，max 在小/中/大分辨率档 AUROC 为
`0.6957/0.6409/0.7040`；小缺陷（最大 mask 面积≤1%）是最难档，对全部
360 个 OK 的 AUROC/AUPR 仅 `0.6085/0.2841`。这直接支持 E1 的 decoder
要优先改善局部小缺陷，而不是继续改 prompt。

### 12.1 CUDA 数值门禁与延迟

- forward-only 与 generate 首 token 最大分数差 `3.81e-06`，通过 `1e-5`容差；
- FP16 batch5 与单图最大分数差 `0.00522`，超过 `5e-4`容差；
- 5 张完全相同图的诊断 batch 仍出现同量级差异，重复 batch 则逐 bit 稳定，
  确认这是 FP16 不同 GEMM batch 形状的确定性舍入路径，不是 padding 或图片混线；
- 未放宽冻结容差，正式 3600 视角全部使用 batch=1 B0f。

4090D 上正式 5 视角整组端到端 p50/p95 为 `0.608/0.865 s`，5 次
forward 之和为 `0.524/0.752 s`，单视角 forward p50 约 `0.105 s`。
独立 batch5 prefill 为 `0.584 s`，但只作延迟测量，不参与正式打分。

完整表见 `artifacts/b0plus_dev720/DEV720_BASELINE.md`，原始视角分数、逐类指标、
bootstrap 原始抽样和配对差值均在同目录。

## 13. E1 训练前准备结果

- hard-normal：从 fold1/train 的 2993 个 `view_label=0` 候选中每类确定性
  选 40 张，合计 720；其中 53 张为样本 NG 但本视角无 mask，继续遵循视角级正常口径；
- sampler pool：`real_anomaly=15782`、`normal=2273`、`hard_normal=720`、
  `pseudo_anomaly=2695`；batch=8 时每路固定 2 个，100 batch 等额与重现自检通过；
- E1：冻结 3.7546B 基座，decoder 可训参数 2,693,643，prompt learner
  4,342,272，base 可训参数 0；
- 4090D 单图 mixed-precision 前后向通过，mask 形状 `[1,1,448,448]`、region
  形状 `[1,10]`，两个新模块均有有限梯度，峰值显存 `9.262 GiB`。

到此只完成数据、sampler 和模型前后向准备，**尚未启动 E1 正式训练**。
Visual Adapter、LoRA 与 holdout-zs 仍未开启。

## 14. 一句话总结

> 最终 B0+ 在 720 样本上的主口径宏 AUPR 为 0.7495；最难的是≤1%小缺陷。
> hard-normal、四路平衡 sampler 与 E1 Decoder + Prompt Adapter 前后向均已准备并自检，
> 下一步是编写完整 E1 训练/评测循环并先跑 fold1，仍不开 Visual Adapter、LoRA 或 holdout-zs。

---

## 15. E1 fold1 正式结果（2026-09-01）

E1 已按冻结日程完成 1000 步和四个完整 Dev-mini checkpoint 评测。
按 18 类宏 AUPR 选出的 best 为 step 500：宏 AUPR `0.7489`、
宏 F1max `0.7679`、小缺陷召回 `0.8214`。

它比 B0+ 的宏 AUPR `0.7495` 低 `0.0006`，没有达到 `0.7795` 的
E1 阶段门；小缺陷召回也下降。step 750/1000 另外出现了空掩码塌缩。
因此路线继续停在 E1 内部诊断，不开 Visual Adapter、LoRA，不读 holdout-zs。
完整协议、数值稳定性记录、四个 checkpoint 表和下一步见
`E1-fold1实验结果分析.md`。
