# 连续异常分数：验证与开发回归

8 个评分运行均通过审计，每组 120 张图像（60 正常、60 异常）。使用相同 decision prompt，只计算 JSON 布尔值 token 的条件似然，再在 true/false 两候选之间归一化。

| 模型 | 验证 AUROC | 验证 AP | 旧测试开发回归 AUROC | 旧测试开发回归 AP |
|---|---:|---:|---:|---:|
| base | 82.39% | 85.58% | 83.39% | 86.95% |
| region_region | 91.49% | 93.01% | 92.83% | 94.17% |
| decision_decision | 94.56% | 95.26% | 95.26% | 95.65% |
| region_decision | 92.01% | 93.54% | 94.10% | 94.82% |

阈值只使用该模型验证集正常样本确定，随后固定应用于旧测试开发回归。

| 模型 | 回归阈值规则 | 异常召回 | 正常误报率 | Macro F1 | 异常 F1 | TP / FP |
|---|---|---:|---:|---:|---:|---:|
| base | 验证正常样本阈值 | 45.00% | 1.67% | 69.50% | 61.36% | 27 / 1 |
| base | 两候选等似然阈值 | 26.67% | 0.00% | 57.64% | 42.11% | 16 / 0 |
| region_region | 验证正常样本阈值 | 71.67% | 3.33% | 83.92% | 81.90% | 43 / 2 |
| region_region | 两候选等似然阈值 | 71.67% | 3.33% | 83.92% | 81.90% | 43 / 2 |
| decision_decision | 验证正常样本阈值 | 78.33% | 6.67% | 85.75% | 84.68% | 47 / 4 |
| decision_decision | 两候选等似然阈值 | 83.33% | 10.00% | 86.65% | 86.21% | 50 / 6 |
| region_decision | 验证正常样本阈值 | 66.67% | 1.67% | 82.05% | 79.21% | 40 / 1 |
| region_decision | 两候选等似然阈值 | 80.00% | 5.00% | 87.43% | 86.49% | 48 / 3 |

| 模型 | 固定 log-odds 阈值 | 校准正常样本 | 校准误报 / 正常 | 校准经验误报率 |
|---|---:|---:|---:|---:|
| base | -1.875 | 60 | 3 / 60 | 5.00% |
| region_region | 0 | 60 | 3 / 60 | 5.00% |
| decision_decision | 0.5 | 60 | 3 / 60 | 5.00% |
| region_decision | 0.75000003 | 60 | 3 / 60 | 5.00% |

阈值规则为 anomaly_log_odds > threshold，边界相等判正常；等似然阈值是 log-odds 0（归一分数 0.5）。排序使用 log-odds，避免 sigmoid 饱和制造额外相同分数。AUROC 按相同分数组处理；AP 为非插值 average precision，不采用 PR 梯形面积。

60 张校准正常图像的误报率步长为 1/60 ≈ 1.67%；5% 目标最多允许 3 次校准误报，相同分数可能使实际误报更少。这只约束校准集经验值，不保证未来或跨域误报率 ≤5%，也不使分数成为校准置信度。

验证集用于开发与阈值选择；旧 120 张测试图像已用于前轮评价，此处属于开发回归。这些结果不能替代最终独立确认。冻结 240 张确认集只读取 source ID 元数据核验隔离，未打开其图像或读取确认预测。

审计通过 256 项，包括文件 SHA256、原始似然与派生分数、指标重算、模型与输入约定、校准来源及阈值、split/source/hash 隔离。完整输入 hash 清单见同名 JSON。基础权重完整 hash 采用评分时记录，汇总时核验跨运行一致性；本脚本不重复读取基础权重或数据图像。

曲线分别标明验证开发集和旧测试开发回归，导出 PNG/PDF/SVG。

- [scoring_validation_curves.pdf](figures/scoring_validation_curves.pdf)
- [scoring_validation_curves.png](figures/scoring_validation_curves.png)
- [scoring_validation_curves.svg](figures/scoring_validation_curves.svg)
- [scoring_regression_curves.pdf](figures/scoring_regression_curves.pdf)
- [scoring_regression_curves.png](figures/scoring_regression_curves.png)
- [scoring_regression_curves.svg](figures/scoring_regression_curves.svg)
