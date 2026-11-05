# 五路线迁移实验统一进度

> 2026-11-04续篇已完成Qwen学生图→原生Qwen ROI集成；GT/KD各val380与单图入口见[STUDENT_ROI_RESULTS.md](student_roi_20261104/STUDENT_ROI_RESULTS.md)。本文保留原五路线矩阵记录。

本地快照生成时间（UTC）：`2026-11-03T16:12:42.266034+00:00`。状态：**complete**。仅读现有 JSON；缺失结果为 ongoing，不计为0。

| 路线 | 状态 | 已有证据 / 最小完整矩阵 |
|---|---|---|
| 1. 语言骨干替换 | complete | Vicuna/Qwen × seeds42/43/44，完整val380 |
| 2. 原生视觉＋18 map token | complete | native baseline＋learned map；gate与训练结果分别报告 |
| 3. Qwen视觉特征及独立定位头 | memory complete；GT complete | memory12格；GT5格（IB224/pre224/post224/pre448/post448） |
| 4. ROI / 正常参考 | pilot complete；full val complete | 原图、中心、随机、teacher ROI、teacher ROI＋参考，共5条件 |
| 5. 同feature教师蒸馏 | complete | 与路线3匹配的KD2格（pre448/post448），KD权重0.500000 |

## 第一阶段三seed

配对完成：3 / 3；按预定seed42/43/44汇总，未挑选最好seed。数值均为验证集0–1，mean ± sample SD，n=3。

| 指标 | Vicuna | Qwen | 配对Δ（Qwen−Vicuna） |
|---|---:|---:|---:|
| image_auroc | 0.889684 ± 0.013285 | 0.925779 ± 0.032698 | 0.036095 ± 0.020330 |
| image_ap | 0.919143 ± 0.012122 | 0.941595 ± 0.026385 | 0.022451 ± 0.014278 |
| generated_macro_f1 | 0.816122 ± 0.005694 | 0.845672 ± 0.041649 | 0.029550 ± 0.044669 |
| generated_recall | 0.666667 ± 0.010008 | 0.719551 ± 0.074687 | 0.052885 ± 0.080016 |
| generated_fpr | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| generated_invalid_rate | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |

两臂冻结语言骨干、仅训练重初始化PromptLearner与CLS投影；不使用Qwen原生视觉。详细协议和各seed控制条件保留于 `PHASE1_RESULTS.md`。

## 第二阶段接口门禁

| Gate | 状态 | 样本数 | 模式 |
|---|---|---:|---|
| qwen3vl-aux-official-same-shape-parity-v1 | complete | 4 | run |
| qwen3vl-native-mrope-deepstack-parity-v1 | complete | 3 | run |

同一接口检查报告存在1份内容相同的镜像，仅记一次检查；所有路径保留在JSON的mirrored_paths，不构成独立重复实验。

官方与manual的相同形状对照已通过：prefill、单步cached、完整replay的logits及LM输入逐位一致；MRoPE、mask、DeepStack输入与cache长度检查通过。

| Cache index | 产品 / 标签 | Prefill最大差 | Cached最大差 | Full replay最大差 | 官方cached/full最大差（诊断） |
|---:|---|---:|---:|---:|---:|
| 9988 | candle / 0 | 0.000000 | 0.000000 | 0.000000 | 0.250000 |
| 10004 | candle / 1 | 0.000000 | 0.000000 | 0.000000 | 0.375000 |
| 10027 | capsules / 0 | 0.000000 | 0.000000 | 0.000000 | 0.312500 |
| 10039 | capsules / 1 | 0.000000 | 0.000000 | 0.000000 | 0.375000 |

官方自身BF16 cached/full replay的最大logit差为0.250000–0.375000。该比较改变attention query长度，记录为数值诊断；原先用此跨形状差异作桥接正确性gate导致误拒绝。相同形状官方/manual结果才直接检验桥接实现。

4个样本属于接口检查，不能替代训练收敛、完整验证或检测效果结论。下一token Yes/No诊断也不是图像首答案检测分数。

Native smoke局部检查：val24，生成8；complete.json=True。该smoke属于pilot，不进入fullval380主结果。

## 原生视觉完整验证

| 方案 | 状态 | Selected step | AUROC | AP | 生成macro F1 | 生成invalid |
|---|---|---:|---:|---:|---:|---:|
| native baseline | complete | 0 | 0.626342 | 0.703170 | 0.218069 | 0.702632 |
| native + learned map | complete | 500 | 0.866391 | 0.907728 | 0.812690 | 0.000000 |

Baseline未经过canonical回答监督，prompt未强制Yes/No首词；生成invalid与格式适配收益需结合NATIVE_PROTOCOL_AUDIT.md解读。两臂都是val开发结果。

## 正常样本记忆库

| 分辨率 | 特征 | Shots | 状态 | Pixel AUROC | Pixel AP | Map image AUROC |
|---:|---|---:|---|---:|---:|---:|
| 224 | pre | 16 | complete | 0.679571 | 0.028558 | 0.652784 |
| 224 | pre | 64 | complete | 0.689270 | 0.032635 | 0.653986 |
| 224 | post | 16 | complete | 0.722113 | 0.022756 | 0.701504 |
| 224 | post | 64 | complete | 0.751966 | 0.025178 | 0.703181 |
| 448 | pre | 16 | complete | 0.714047 | 0.018271 | 0.684537 |
| 448 | pre | 64 | complete | 0.726718 | 0.023452 | 0.699799 |
| 448 | post | 16 | complete | 0.790202 | 0.035138 | 0.737561 |
| 448 | post | 64 | complete | 0.809481 | 0.039597 | 0.740385 |
| 672 | pre | 16 | complete | 0.714658 | 0.014272 | 0.678472 |
| 672 | pre | 64 | complete | 0.728787 | 0.017092 | 0.690463 |
| 672 | post | 16 | complete | 0.799706 | 0.032227 | 0.727722 |
| 672 | post | 64 | complete | 0.814005 | 0.039527 | 0.762774 |

## 独立定位头

| 特征 | 分辨率 | 训练 | 状态 | Best step | Pixel AUROC | Pixel AP |
|---|---:|---|---|---:|---:|---:|
| imagebind | 224 | GT | complete | 1000 | 0.974284 | 0.390244 |
| post | 224 | GT | complete | 1200 | 0.935581 | 0.181908 |
| post | 448 | GT | complete | 1200 | 0.966730 | 0.295317 |
| post | 448 | GT_KD | complete | 1200 | 0.967149 | 0.302921 |
| pre | 224 | GT | complete | 1200 | 0.907415 | 0.217631 |
| pre | 448 | GT | complete | 1200 | 0.952411 | 0.336404 |
| pre | 448 | GT_KD | complete | 1200 | 0.953943 | 0.327643 |

同特征GT/KD按相同head、初始化seed、训练计划、优化器和预算匹配。差值为KD−GT，单位0–1；单seed描述性结果。

| 特征/分辨率 | 合同 | Δ Pixel AP | Δ Pixel AUROC | Δ Map image AUROC |
|---|---|---:|---:|---:|
| pre/448 | complete | -0.008761 | 0.001532 | 0.012215 |
| post/448 | complete | 0.007603 | 0.000419 | -0.008972 |

KD teacher仅在训练使用；当前无teacher验证/推理结论仅覆盖独立定位头，蒸馏map尚未接入语言模型或ROI系统。

## ROI及正常参考

| 范围 | N | 条件 | 状态 | AUROC | 生成macro F1 | Invalid |
|---|---:|---|---|---:|---:|---:|
| full_validation | 380 | original_only | complete | 0.753480 | 0.446888 | 0.000000 |
| full_validation | 380 | original_center_crop | complete | 0.713174 | 0.639874 | 0.000000 |
| full_validation | 380 | original_random_crop | complete | 0.726996 | 0.548301 | 0.000000 |
| full_validation | 380 | original_teacher_roi | complete | 0.837265 | 0.765826 | 0.000000 |
| full_validation | 380 | original_teacher_roi_reference | complete | 0.830417 | 0.731000 | 0.000000 |
| pilot | 48 | original_only | complete | 0.775174 | 0.495798 | 0.000000 |
| pilot | 48 | original_center_crop | complete | 0.695312 | 0.563636 | 0.000000 |
| pilot | 48 | original_random_crop | complete | 0.743924 | 0.563636 | 0.000000 |
| pilot | 48 | original_teacher_roi | complete | 0.835938 | 0.785714 | 0.000000 |
| pilot | 48 | original_teacher_roi_reference | complete | 0.838542 | 0.762483 | 0.000000 |

## 共享教师参考

冻结教师单独在val380上的pixel AUROC=0.975934、pixel AP=0.435522、map-p99 image AUROC=0.814317。该项是缓存参考，不是LM分数或新dense结果。
共享缓存：teacher seed=42，float32/square224，batch=8；decoder SHA `cddfa2df9ad63659f65aa3a798566e52d608ace43a1b37704484db5bbfc5617d`。完整cache/manifest SHA保留于JSON，训练与蒸馏需核对同一教师来源。

## 本轮矩阵与复现协议

- Normal memory：seed42，requested224/448/672 ×pre/post ×16/64；一次调用共享visual前向，16是64支持集的前缀；仅训练正常样本建库。当前extractor显式do_resize=False，实际输入等于请求分辨率；224对应pre14×14/post7×7。native LM路径仍使用官方smart resize，224请求可到256，二者分别记录。
- Dense GT：ImageBind224、pre224、post224、pre448、post448共5格；KD仅pre448与post448共2格。1200updates、B4、lr1e-3、cache8GiB、每200步val380、按macro pixel AP选best并重载。两对GT/KD须同head、seed、plan、优化器、调度和预算。
- Head参数：ImageBind697,345（FP16 tower/cache），pre664,577（BF16），post1,418,241（BF16）；head均FP32。结构相同不代表参数量或精度完全匹配。
- ROI pilot48与fullval380分别报告。teacher ROI依赖外部定位教师，参考图仅从同类别训练正常图选择；第三张参考图同时增加视觉token预算。
- 完整matrix的seed42属于探索性对照；后续43/44应做匹配复现，不能从多个seed/权重中挑最好结果。

## 代码推导的资源风险

以下是张量payload/曝光上界，不是实测耗时、总RSS或OOM保证。

| Dense特征 | 单图缓存bytes | 1GiB图数 | 当前runner8GiB图数 | Head参数 |
|---|---:|---:|---:|---:|
| imagebind224 | 655360 | 1638 | 13107 | 697345 |
| pre224 | 451584 | 2377 | 19021 | 664577 |
| post224 | 401408 | 2674 | 21399 | 1418241 |
| pre448 | 1806336 | 594 | 4755 | 664577 |
| post448 | 1605632 | 668 | 5349 | 1418241 |

Dense1200×B4有4800次训练曝光；6轮val380和1轮best重载val380共2660次，若全cache miss最多7460次逐图visual前向。Trainer默认1GiB可能因val插入驱逐训练项；当前runner显式8GiB，是否容纳全部实际唯一训练路径＋val由training_plan决定。速度须读实际elapsed/peak指标，不能用曝光数冒充测量。

ImageBind在CPU构造完整多模态模型并strict load约4.80GB权重后才裁掉非视觉模块，加载瞬时CPU tensor payload可接近两份权重并另有开销；Qwen只加载visual。Head的FP32激活/卷积workspace不包含在参数量估计内。

Normal memory全矩阵共享提取时最多3444次visual前向；每次仅一类别四个FP32 bank驻GPU，448/672约314MiB，similarity块2MiB。CPU四组scoremap与共享GT的float32 payload约364MiB，指标排序/拼接另需内存，不能视为总RAM。

## 证据边界

- Supervised custom60/20/20 uses real anomaly training images; this is not the official normal-only benchmark protocol.
- All score tables are validation development evidence. Pilot scores are not full-validation scores; historical test456 has been used in prior development.
- Image Yes-No candidate scores, free generation, pixel maps, and memory distances are separate outcome surfaces.
- No missing result becomes zero; no pilot, failed gate, unmatched pair, or incomplete seed aggregate becomes a completed result.
- Frozen Qwen failures/successes from one route do not establish presence or absence of anomaly information in its features.

源JSON路径与SHA256保存在 `all_summary.json` 的input_files；仅读结果、配置、完成状态和gate，不读取checkpoint、预测数组或test指标。
