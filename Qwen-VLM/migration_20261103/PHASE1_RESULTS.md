# 第一阶段语言骨干配对结果

状态：**complete**；已完成配对：3 / 3。

有监督 custom 60/20/20；train 9,985、val 380（正常172/异常208）。两臂冻结语言骨干，以同一 teacher cache 训练重新初始化的 PromptLearner 与 CLS 投影，均不使用 Qwen 原生视觉。

全部数值按 0–1 报告。候选分数是首答案 token 的 logit(Yes)−logit(No)；生成指标独立计算。缺失值以 — 标示，ongoing 不计为 0。

| Seed | 骨干 | 状态 | Best step | Real AUROC | Real AP | 生成 macro F1 | 生成 recall | 生成 FPR | 生成 invalid |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | vicuna | complete | 1500 | 0.904643 | 0.931196 | 0.811559 | 0.658654 | 0.000000 | 0.000000 |
| 42 | qwen | complete | 1500 | 0.958659 | 0.968372 | 0.892087 | 0.802885 | 0.000000 | 0.000000 |
| 43 | vicuna | complete | 1500 | 0.879263 | 0.906953 | 0.814303 | 0.663462 | 0.000000 | 0.000000 |
| 43 | qwen | complete | 1500 | 0.893266 | 0.915619 | 0.811559 | 0.658654 | 0.000000 | 0.000000 |
| 44 | vicuna | complete | 1500 | 0.885146 | 0.919282 | 0.822503 | 0.677885 | 0.000000 | 0.000000 |
| 44 | qwen | complete | 1500 | 0.925411 | 0.940794 | 0.833369 | 0.697115 | 0.000000 | 0.000000 |

正在写入的运行可显示已有数值；只有状态 complete 且配置匹配的配对进入 Δ。

| Seed | 配对状态 | Δ AUROC | Δ AP | Δ 生成 F1 | Δ recall | Δ FPR | Δ invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | complete | 0.054017 | 0.037175 | 0.080527 | 0.144231 | 0.000000 | 0.000000 |
| 43 | complete | 0.014004 | 0.008666 | -0.002744 | -0.004808 | 0.000000 | 0.000000 |
| 44 | complete | 0.040264 | 0.021512 | 0.010866 | 0.019231 | 0.000000 | 0.000000 |

Δ 均为 Qwen−Vicuna；FPR/invalid 越低越好。每个 seed 内核对初始化、decoder/cache/manifest、采样顺序哈希、可训练参数、步数、batch、accum、验证间隔和完整 val380，并核对其余共同训练配置。

三 seed mean ± sampleSD（样本标准差，ddof=1；n=3）：

| 指标 | Vicuna | Qwen | 配对 Δ |
|---|---:|---:|---:|
| image_auroc | 0.889684 ± 0.013285 | 0.925779 ± 0.032698 | 0.036095 ± 0.020330 |
| image_ap | 0.919143 ± 0.012122 | 0.941595 ± 0.026385 | 0.022451 ± 0.014278 |
| generated_macro_f1 | 0.816122 ± 0.005694 | 0.845672 ± 0.041649 | 0.029550 ± 0.044669 |
| generated_recall | 0.666667 ± 0.010008 | 0.719551 ± 0.074687 | 0.052885 ± 0.080016 |
| generated_fpr | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| generated_invalid_rate | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |

各控制条件的候选分数：

| Seed | 骨干 | 条件 | AUROC | AP |
|---|---|---|---:|---:|
| 42 | vicuna | real_map | 0.904643 | 0.931196 |
| 42 | vicuna | zero_map | 0.587084 | 0.624671 |
| 42 | vicuna | donor_opposite_label_map | 0.178486 | 0.398000 |
| 42 | vicuna | spatial_shuffle_map | 0.893728 | 0.922838 |
| 42 | qwen | real_map | 0.958659 | 0.968372 |
| 42 | qwen | zero_map | 0.844141 | 0.862169 |
| 42 | qwen | donor_opposite_label_map | 0.293339 | 0.427813 |
| 42 | qwen | spatial_shuffle_map | 0.953251 | 0.964693 |
| 43 | vicuna | real_map | 0.879263 | 0.906953 |
| 43 | vicuna | zero_map | 0.499050 | 0.547876 |
| 43 | vicuna | donor_opposite_label_map | 0.138571 | 0.419464 |
| 43 | vicuna | spatial_shuffle_map | 0.874916 | 0.902863 |
| 43 | qwen | real_map | 0.893266 | 0.915619 |
| 43 | qwen | zero_map | 0.591863 | 0.625428 |
| 43 | qwen | donor_opposite_label_map | 0.181001 | 0.409266 |
| 43 | qwen | spatial_shuffle_map | 0.884825 | 0.909787 |
| 44 | vicuna | real_map | 0.885146 | 0.919282 |
| 44 | vicuna | zero_map | 0.537204 | 0.585150 |
| 44 | vicuna | donor_opposite_label_map | 0.159129 | 0.400141 |
| 44 | vicuna | spatial_shuffle_map | 0.878843 | 0.914041 |
| 44 | qwen | real_map | 0.925411 | 0.940794 |
| 44 | qwen | zero_map | 0.650436 | 0.697168 |
| 44 | qwen | donor_opposite_label_map | 0.195243 | 0.417146 |
| 44 | qwen | spatial_shuffle_map | 0.914049 | 0.931955 |

控制条件的三 seed mean ± sampleSD：

| 条件 | 指标 | Vicuna | Qwen | 配对 Δ |
|---|---|---:|---:|---:|
| real_map | image_auroc | 0.889684 ± 0.013285 | 0.925779 ± 0.032698 | 0.036095 ± 0.020330 |
| real_map | image_ap | 0.919143 ± 0.012122 | 0.941595 ± 0.026385 | 0.022451 ± 0.014278 |
| zero_map | image_auroc | 0.541112 ± 0.044147 | 0.695480 ± 0.132033 | 0.154368 ± 0.089516 |
| zero_map | image_ap | 0.585899 ± 0.038403 | 0.728255 ± 0.121393 | 0.142356 ± 0.084178 |
| donor_opposite_label_map | image_auroc | 0.158728 ± 0.019961 | 0.223194 ± 0.061163 | 0.064466 ± 0.043751 |
| donor_opposite_label_map | image_ap | 0.405868 ± 0.011823 | 0.418075 ± 0.009309 | 0.012207 ± 0.020433 |
| spatial_shuffle_map | image_auroc | 0.882496 ± 0.009923 | 0.917375 ± 0.034334 | 0.034879 ± 0.024809 |
| spatial_shuffle_map | image_ap | 0.913247 ± 0.010011 | 0.935478 ± 0.027622 | 0.022231 ± 0.017862 |

一致性与完成情况：

- vicuna_s42：complete。输入完整、协议检查通过。
- qwen_s42：complete。输入完整、协议检查通过。
- vicuna_s43：complete。输入完整、协议检查通过。
- qwen_s43：complete。输入完整、协议检查通过。
- vicuna_s44：complete。输入完整、协议检查通过。
- qwen_s44：complete。输入完整、协议检查通过。

解释边界：

- 有监督自定义 60/20/20 划分；train9985、val380，使用真实异常训练。
- 本阶段仅比较冻结语言骨干；两臂均不使用 Qwen 原生视觉，桥接模块重新初始化。
- checkpoint 仅由 val AUROC 选择，均值覆盖预定 seeds42/43/44，不能挑选最好 seed。
- 本报告只汇总 val；历史456张 test 已用于旧开发，不能称为未查看确认集。
- 候选打分与自由生成分别报告；invalid 保留在生成指标分母，FPR 须结合 invalid 解读。
- donor 条件使用 val 标签构造同类别相反标签诊断，不是可部署推理算法。
- 相同图片曝光与更新预算不代表相同 token 数、计算量或两骨干能力的单独因果效应。

仅读取六组 run.json、train_complete.json 和 val_metrics.json；文件路径与 SHA256 见 phase1_summary.json。未读取 checkpoint、预测文件、训练日志或 test 指标。
