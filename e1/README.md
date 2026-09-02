# E1 · Decoder + Prompt Adapter 准备区

本目录只实现研究路线的第一刀：冻结 Qwen2.5-VL-3B 的视觉塔、语言模型和 `lm_head`，
只训练多层 patch decoder 与由热图条件化的虚拟 prompt。这里不包含 Visual Adapter
或 LLM LoRA，也不读取 `holdout-zs` 做选择。

## 数据顺序

必须先冻结最终 B0+ Dev-mini 基线，再依次执行：

```bash
python3 e1/mine_hard_normals.py
python3 e1/build_sampler_pool.py
```

`mine_hard_normals.py` 只读取 `fold1/train.parquet` 中 `view_label == 0` 的视角，
按类别内的边缘密度、纹理、镜面高光、暗区和对比度确定性挖掘 40 张/类，合计 720 张。
491 张全量数据中的 maskless NG 视角仍按视角级正常口径处理，并在清单中单独标记。

sampler 每个 batch 对以下四路精确等额：

| source | 占比 | 约束 |
| --- | ---: | --- |
| real_anomaly | 25% | 必须有有效 mask；类别和面积档均衡 |
| normal | 25% | 视角级正常，排除已单列的 hard-normal |
| hard_normal | 25% | 每类 40 张确定性清单 |
| pseudo_anomaly | 25% | 只从样本级 OK 图生成 Cut-Paste / NSA |

`BalancedBatchSampler.set_epoch(epoch)` 控制可复现的逐 epoch 随机性。

正式训练时冻结基座以 BF16 运行，decoder 与 prompt learner 的参数及
所有损失保持 FP32，前向使用 CUDA BF16 autocast。BF16 不使用 `GradScaler`。

## 模型

`model.py` 从视觉塔第 7/15/23/31 层提取 32×32 patch 特征，恢复窗口注意力前的空间顺序，
融合后输出 448 热图与 3×3+none 区域分类。热图加权池化后的局部特征生成 4 个虚拟
prompt token，注入冻结 LLM，再读取 Yes/No next-token logits。

损失冻结为：

```text
L = BCE(Yes/No logits) + 0.2 × (focal + dice) + 0.1 × CE(region)
```

Checkpoint 只按 Dev-mini 宏平均 I-AUPR 选择，不使用 Pixel AUROC。

单图前后向冒烟必须从仓库根目录以模块形式执行：

```bash
python -m e1.smoke_model --image /absolute/path/to/image.png
```

## 阶段门

E1 只有同时满足以下条件，才能进入 E2：

- 相对最终 B0+，Dev-mini 宏平均 I-AUPR 绝对提升至少 0.03；
- 宏平均 F1max 不下降；
- 至少 3/4 个冻结折为正增益；
- 小缺陷召回不下降，预测面积占比不爆炸。

未过线时只检查 hard-normal、decoder 取层和 `lambda_seg`，不开 Visual Adapter 或 LoRA。

## 当前准备状态（2026-09-01）

- `hard_normals.csv`：720 张，18 类每类 40 张；
- `sampler_pool.parquet`：四路数量为 15782 / 2273 / 720 / 2695，
  每路总权重均为 0.25，100 个 batch 的等额与复现自检通过；
- `e1_smoke.json`：decoder 2,693,643 参数，prompt learner 4,342,272 参数，
  冻结基座可训参数为 0；
- RTX 4090D 单图前后向通过，新增两个模块均有有限梯度，
  峰值显存 9.262 GiB。

这些是训练前门禁。fold1 正式结果已在下文记录；其余冻结折尚未执行。

## Fold1 正式训练协议

在看 E1 分数前冻结为：

- 1000 个 optimizer step，逻辑 batch=8，micro-batch=1，累积 8 次；
- 四路每个逻辑 batch 各 2 个，全程共各 2000 次抽样；
- AdamW，decoder/prompt lr 都为 `1e-4`，warmup 50 step，余弦衰减；
- 冻结 Qwen BF16，新增参数和损失 FP32，BF16 autocast，不做 loss scaling；
- step 250/500/750/1000 评测完整 720 样本 Dev-mini；
- 只按 18 类宏 I-AUPR 保存 best checkpoint。

`fold1_schedule.parquet` 在训练前冻结全部 8000 个抽样位置和伪异常 donor，
不依赖运行时随机状态。每次更新都断言冻结基座无梯度且新增模块梯度有限。

FP16 在固定日程 step 16/26 的梯度溢出试验被门禁在更新前终止；
正式协议在没有产生可接受 checkpoint 前改为 4090D 原生 BF16，并从相同 seed
重新从 step 1 开始。

评测同时报告：宏 I-AUPR/F1max、≤1% 小缺陷召回、OK 视角平均预测面积和
全体视角预测面积 p95。Fold1 门禁为：

```text
macro AUPR >= 0.7795
macro F1max >= 0.7380
small-defect recall >= 0.988095
OK-view mean predicted area <= 0.10
all-view predicted area p95 <= 0.25
```

复现命令：

```bash
python -m e1.build_train448_cache \
  --cache-root /path/to/Train448 \
  --manifest artifacts/e1/train448_cache_manifest.csv

python -m e1.make_fold1_schedule

python -m e1.train_fold1 \
  --model-id models/Qwen2.5-VL-3B-Instruct \
  --cache-root /path/to/Train448 \
  --dev-cache-root Dataset448 \
  --out-dir artifacts/e1_fold1

python -m e1.evaluate_fold1 \
  --checkpoint artifacts/e1_fold1/best.pt \
  --tag best_confirm
```

## Fold1 正式结果（2026-09-01）

| Step | 宏 AUPR | 宏 F1max | 小缺陷召回 | OK 面积均值 | 面积 P95 | 过门 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 250 | 0.7240 | 0.7392 | 0.9167 | 0.00033 | 0.00174 | 否 |
| **500** | **0.7489** | **0.7679** | 0.8214 | 0.00529 | 0.03857 | 否 |
| 750 | 0.7108 | 0.7627 | 0.8333 | 0.00000091 | 0 | 否 |
| 1000 | 0.7339 | 0.7659 | 0.8095 | 0 | 0 | 否 |

best 为 step 500，独立重载复评的 3600 个 `p_yes` 和预测面积均与原评测逐项完全一致。
宏 AUPR 未达 0.7795，小缺陷召回也低于 0.988095；因此不开 Visual Adapter 或
LoRA。step 750/1000 的预测面积几乎归零，下一步只在 E1 内部诊断空 mask Dice
退化解、`lambda_seg`、decoder 取层和 hard-normal。完整分析见
`E1-fold1实验结果分析.md`。
