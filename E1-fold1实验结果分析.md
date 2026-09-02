# E1 Decoder + Prompt Adapter·fold1 正式实验结果

> 实验日期：2026-09-01
> 设备：RTX 4090D 24 GB
> 结论：**fold1 未达到 E1 阶段门，不打开 Visual Adapter 或 LoRA。**

## 1. 先说结果

E1 完成了 1000 个 optimizer step，并在 step 250/500/750/1000 对同一份
fold1 Dev-mini 做了四次完整评测。按预注册的 18 类宏 I-AUPR 选择 checkpoint，
best 是 **step 500**。

| 项目 | B0+ 主基线 | E1 fold1 best | 变化 | E1 目标 | 是否过门 |
| --- | ---: | ---: | ---: | ---: | --- |
| 18 类宏 I-AUPR | 0.7495 | **0.7489** | -0.0006 | ≥ 0.7795 | 否 |
| 18 类宏 F1max | 0.7380 | **0.7679** | +0.0299 | ≥ 0.7380 | 是 |
| 小缺陷召回 | 0.9881 | **0.8214** | -0.1667 | ≥ 0.9881 | 否 |
| OK 视角平均预测面积 | — | **0.00529** | — | ≤ 0.10 | 是 |
| 全视角预测面积 P95 | — | **0.03857** | — | ≤ 0.25 | 是 |

直白地说：E1 把 F1max 提高了，但没有提高排序质量；同时对小缺陷更保守，
漏检明显变多。这不是一个可以进入 E2/E3 的合格 E1。

## 2. 冻结的实验口径

- 基座：`Qwen2.5-VL-3B-Instruct`；
- 输入：`direct 448`；
- 提示：P0；
- 图级分数：Yes/No next-token logits；
- 样本分数：5 视角 `max`；
- 冻结模块：视觉塔、语言模型、`lm_head`；
- 可训模块：Decoder 2,693,643 参数，Prompt Learner 4,342,272 参数；
- 训练集：fold1 train，没有读取 `holdout-zs`；
- Dev-mini：18 类，每类 20 OK + 20 NG，共 720 个完整样本、3,600 个视角；
- 训练日程：1000 step，logical batch=8，micro-batch=1，累积 8 次；
- 每个逻辑 batch：2 real anomaly + 2 normal + 2 hard-normal + 2 pseudo anomaly；
- 总抽样：四路各 2000 次，8000 个位置及 pseudo donor 均在训练前冻结；
- 优化：AdamW，lr=`1e-4`，warmup 50，cosine decay，gradient clip=1；
- 损失：`BCE(cls) + 0.2*(focal+dice) + 0.1*CE(region)`；
- checkpoint 选择：只看 Dev-mini 18 类宏 I-AUPR，不看 Pixel AUROC。

## 3. 混合精度的数值修订

原训练实现用 FP16 autocast + GradScaler。定位到两次可重现溢出：

1. scale=1024 在固定日程 step 16 产生非有限梯度；
2. 像素损失改为 FP32 reduction 后，step 16 仍溢出；
3. scale=64 越过 step 16，但在 step 26 再次溢出。

两次都由非有限梯度门禁在 optimizer update 前中止，没有生成被接受的正式 checkpoint。
最终使用 4090D 原生支持的 **BF16 autocast**：冻结 Qwen 权重为 BF16，新增参数和
所有损失为 FP32，BF16 不使用 loss scaling。随后从相同 seed 和相同日程重新从
step 1 开始正式训练。

正式运行共 1000 行训练记录，非有限数值为 0，峰值显存 9.331 GiB。原始梯度
范数最大为 4626.98，因此 BF16 的更大指数范围是必要的；所有这些梯度都在
更新前按冻结协议被 clip 到 1。

## 4. 四个 checkpoint 的完整趋势

| Step | 宏 AUROC | 宏 AUPR | 宏 F1max | 小缺陷召回 | OK 面积均值 | 面积 P95 | 过门 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 250 | 0.7021 | 0.7240 | 0.7392 | **0.9167** | 0.00033 | 0.00174 | 否 |
| **500** | **0.7399** | **0.7489** | **0.7679** | 0.8214 | 0.00529 | 0.03857 | 否 |
| 750 | 0.7056 | 0.7108 | 0.7627 | 0.8333 | 0.00000091 | 0 | 否 |
| 1000 | 0.7271 | 0.7339 | 0.7659 | 0.8095 | 0 | 0 | 否 |

step 500 后继续训练没有进一步改善宏 AUPR。step 750/1000 的二值预测面积几乎
完全归零，说明在固定 0.5 mask 阈值下出现了**空掩码塌缩**。预测面积没有
“爆炸”，但归零同样不是好的分割结果。

四段训练的平均损失为：

| Step | 总损失 | 分类 | 分割 | 区域 |
| --- | ---: | ---: | ---: | ---: |
| 1–250 | 1.0351 | 0.6544 | 0.9987 | 1.8102 |
| 251–500 | 0.9178 | 0.5506 | 1.0042 | 1.6635 |
| 501–750 | 0.7951 | 0.4982 | 0.6956 | 1.5776 |
| 751–1000 | 0.7233 | 0.4585 | 0.5470 | 1.5542 |

分割损失下降但二值面积塌缩，表明当前 focal+dice 在大量背景像素和空 mask 上
存在退化解。不能只看 loss 下降就宣称定位能力提升。

## 5. best 的分类别表现

step 500 中表现较弱的类别是：

| 类别 | AUPR | F1max |
| --- | ---: | ---: |
| audio_jack_socket | 0.4908 | 0.6923 |
| side_press_switch | 0.5592 | 0.7170 |
| smd_receiver_module | 0.6126 | 0.6897 |
| battery_holder_connector | 0.6364 | 0.7500 |
| pneumatic_elbow | 0.6613 | 0.7308 |

表现较强的类别包括 `slide_switch` AUPR 0.9498、
`mobile_charging_connector` 0.9160 和 `3_adapter` 0.8326。类间差异很大，
不适合用 pooled 指标代替 18 类宏平均做决策。

## 6. 可复现性与产物

重新启动独立进程，从 `best.pt` 加载 step 500 权重后再跑 3,600 个视角：

- `p_yes` 最大绝对差：**0**；
- `predicted_area` 最大绝对差：**0**；
- 宏 AUPR/F1max：仍为 **0.7488847 / 0.7679272**。

主产物在 `artifacts/e1_fold1/`：

- `best.pt`：step 500 best checkpoint；
- `latest.pt`：step 1000 终点 checkpoint；
- `training_history.csv`：1000 步损失、梯度、学习率、耗时和显存；
- `step*_views.csv` / `step*_samples.csv` / `step*_per_class.csv`：逐视角、逐样本和逐类结果；
- `step*_metrics.json`：每个 checkpoint 的阶段门判定；
- `best_confirm_*`：独立加载 best 的复评结果；
- `frozen_training_protocol.json`：数据、日程和配置哈希。

## 7. 按研究路线的下一步

当前停在 **E1 内部诊断**，不进入 E2/E3，也不读 holdout-zs。建议顺序是：

1. 在训练日志中补充按 `real/pseudo/normal/hard-normal` 分开的分割损失和 mask 概率
   分位数，先定位空掩码塌缩由哪一路主导；
2. 作为一个独立 E1 消融，先修正空 mask 的 Dice 退化解：Dice 只对非空掩码计算，
   空掩码保留 focal/BCE 约束；其他配置不动；
3. 如果还不过门，再分别做 `lambda_seg` 或 decoder 取层的单变量消融，不同时改；
4. 每个候选仍只用 train + Dev-mini，仍需宏 AUPR≥0.7795、宏 F1max≥0.7380、
   小缺陷召回不降和面积不退化；
5. fold1 通过后仍不足以开 E2，还要在其余冻结折中至少达到 3/4 折正增益。

因此，当前的明确决策是：**保留 step 500 作为 E1 fold1 负结果 best，
先修 E1 损失/分割塌缩；不开 Visual Adapter，不开 LoRA。**
