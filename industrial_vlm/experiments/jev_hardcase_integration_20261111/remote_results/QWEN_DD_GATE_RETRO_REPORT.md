# Qwen DD gate 回顾结果

本报告重放已冻结的 24 张在线 pilot 开发图，不新增 API、不做 GPU forward、不在 24 张标签上调阈值。Qwen DD 的 `anomaly_log_odds>0` 是整图异常分数门控，不是 crop-needed 语义；所有路线指标均为 image-label proxy。

| 策略 | AUROC | AP | Recall | FPR | ROI调用数 |
|---|---:|---:|---:|---:|---:|
| qwen_dd_gate | 82.6389% | 85.0406% | 58.33% | 16.67% | 6 |
| learned_gate | 77.0833% | 81.6246% | 66.67% | 25.00% | 8 |
| matched_random | 77.7778% | 82.0214% | 66.67% | 25.00% | 8 |
| historical_uncertainty | 78.1250% | 81.8508% | 66.67% | 25.00% | 8 |
| geometry_small | 81.9444% | 84.5827% | 58.33% | 16.67% | 8 |
| new_duplicate | 82.9861% | 85.7263% | 58.33% | 16.67% | 0 |

Qwen DD 固定阈值在 13 个 eligible 候选中选择 6 个进入 ROI 回放。该结果只能用于低成本策略诊断；若 Qwen DD gate 不劣于 Jev gate，则没有为 gate 单独支付 Jev 调用费用的理由。
