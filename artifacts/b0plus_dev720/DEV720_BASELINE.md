# B0+ Dev-mini 720 最终统计基线

冻结配置：Qwen2.5-VL-3B-Instruct + P0 + direct 448 + B0f；主聚合仍为 max。
本次只读取 fold1/dev.parquet，未读取或推理 holdout 样本。

## 总体与 18 类宏平均

| aggregation | overall_auroc | overall_aupr | overall_f1max | macro_auroc | macro_aupr | macro_f1max | hard_yes_rate_at_0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max | 0.6674 | 0.6697 | 0.6781 | 0.7208 | 0.7495 | 0.7380 | 0.0139 |
| top2_mean | 0.6763 | 0.6735 | 0.6805 | 0.7400 | 0.7648 | 0.7565 | 0.0056 |
| mean | 0.6715 | 0.6525 | 0.6838 | 0.7483 | 0.7734 | 0.7623 | 0.0000 |

## 2000 次分层 bootstrap 95% 区间

| aggregation | metric | ci_low | ci_median | ci_high |
| --- | --- | --- | --- | --- |
| max | macro_auroc | 0.6812 | 0.7211 | 0.7572 |
| max | macro_aupr | 0.7277 | 0.7606 | 0.7924 |
| max | macro_f1max | 0.7403 | 0.7613 | 0.7821 |
| top2_mean | macro_auroc | 0.7017 | 0.7399 | 0.7749 |
| top2_mean | macro_aupr | 0.7423 | 0.7743 | 0.8045 |
| top2_mean | macro_f1max | 0.7539 | 0.7765 | 0.7976 |
| mean | macro_auroc | 0.7118 | 0.7475 | 0.7826 |
| mean | macro_aupr | 0.7510 | 0.7830 | 0.8130 |
| mean | macro_f1max | 0.7620 | 0.7822 | 0.8035 |

## 相对 max 的配对 bootstrap 差值

| candidate | relative_to | metric | point_delta | ci_low | ci_median | ci_high | probability_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top2_mean | max | macro_auroc | 0.0192 | 0.0047 | 0.0191 | 0.0333 | 0.9955 |
| top2_mean | max | macro_aupr | 0.0152 | -0.0001 | 0.0141 | 0.0271 | 0.9740 |
| top2_mean | max | macro_f1max | 0.0185 | 0.0029 | 0.0151 | 0.0269 | 0.9905 |
| mean | max | macro_auroc | 0.0275 | 0.0008 | 0.0272 | 0.0535 | 0.9770 |
| mean | max | macro_aupr | 0.0239 | 0.0003 | 0.0229 | 0.0458 | 0.9765 |
| mean | max | macro_f1max | 0.0242 | 0.0049 | 0.0212 | 0.0388 | 0.9945 |

## 分辨率分层

| aggregation | resolution_stratum | auroc | aupr | f1max | n_samples | n_ok | n_ng |
| --- | --- | --- | --- | --- | --- | --- | --- |
| max | small<=640 | 0.6957 | 0.7467 | 0.7386 | 200 | 91 | 109 |
| max | medium641-1280 | 0.6409 | 0.6499 | 0.6818 | 362 | 179 | 183 |
| max | large>1280 | 0.7040 | 0.6581 | 0.6667 | 158 | 90 | 68 |
| top2_mean | small<=640 | 0.6889 | 0.7381 | 0.7273 | 200 | 91 | 109 |
| top2_mean | medium641-1280 | 0.6466 | 0.6490 | 0.6842 | 362 | 179 | 183 |
| top2_mean | large>1280 | 0.7278 | 0.6758 | 0.6857 | 158 | 90 | 68 |
| mean | small<=640 | 0.6849 | 0.7210 | 0.7237 | 200 | 91 | 109 |
| mean | medium641-1280 | 0.6480 | 0.6303 | 0.6844 | 362 | 179 | 183 |
| mean | large>1280 | 0.7142 | 0.6440 | 0.6795 | 158 | 90 | 68 |

## NG 最大缺陷面积分层

每个面积档的正样本与全部 360 个 OK 样本比较；recall 使用全 Dev 的 F1max 阈值。

| aggregation | area_stratum | auroc | aupr | f1max | n_ng | n_ok_reference | recall_at_full_f1_threshold | full_f1_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max | small<=1% | 0.6085 | 0.2841 | 0.3629 | 84 | 360 | 0.9881 | 0.1624 |
| max | medium1-3% | 0.6933 | 0.5133 | 0.5236 | 146 | 360 | 0.9863 | 0.1624 |
| max | large>3% | 0.6764 | 0.4318 | 0.4883 | 130 | 360 | 0.9923 | 0.1624 |
| top2_mean | small<=1% | 0.6107 | 0.2814 | 0.3518 | 84 | 360 | 0.7500 | 0.1919 |
| top2_mean | medium1-3% | 0.7000 | 0.5199 | 0.5491 | 146 | 360 | 0.8219 | 0.1919 |
| top2_mean | large>3% | 0.6921 | 0.4386 | 0.5094 | 130 | 360 | 0.8615 | 0.1919 |
| mean | small<=1% | 0.6121 | 0.2354 | 0.3683 | 84 | 360 | 0.9405 | 0.1481 |
| mean | medium1-3% | 0.6877 | 0.5030 | 0.5261 | 146 | 360 | 0.9452 | 0.1481 |
| mean | large>3% | 0.6917 | 0.4093 | 0.5115 | 130 | 360 | 0.9846 | 0.1481 |

## 4090D 延迟

正式执行模式为 serial。每组 5 视角总延迟 p50/p95 = 0.6077/0.8649s；5 次逐图 forward 之和 p50/p95 = 0.5240/0.7522s，折算单视角 forward p50 = 0.1048s。
独立的 5 视角 batch prefill forward 测量为 0.5836s；由于 batch/single 最大分数差 0.00522161 超过冻结容差 0.0005，它不参与正式打分。

## 跨硬件复核

与 MPS smoke20 重叠 100 个视角；P(Yes) 最大/中位绝对差为 0.0109068/3.12924e-07。

## 决策纪律

max 是预先冻结的主结果；top2_mean 和 mean 是同一批视角分数的离线诊断，完整保留但不因单次 Dev 结果事后替换主口径。holdout-zs 继续封存。
