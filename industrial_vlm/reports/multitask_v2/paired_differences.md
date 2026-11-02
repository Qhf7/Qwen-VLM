# 多任务SFT v2：验证集配对差异

对相同来源做2000次配对bootstrap，seed=17，报告百分位95%区间。差值为左臂减右臂，单位为百分点；FPR/FNR越低越好。

| 左 − 右（同提示） | 指标 | 左 % | 右 % | 差值pp | 95%区间pp | 有定义重采样数 |
|---|---|---:|---:|---:|---:|---:|
| region_decision_decision − region_region_decision | macro_f1 | 84.14 | 82.22 | +1.92 | [-3.29, +7.33] | 2000 |
| region_decision_decision − region_region_decision | balanced_accuracy | 84.17 | 82.50 | +1.67 | [-3.42, +6.67] | 2000 |
| region_decision_decision − region_region_decision | fpr | 11.67 | 5.00 | +6.67 | [+1.51, +13.85] | 2000 |
| region_decision_decision − region_region_decision | fnr | 20.00 | 30.00 | -10.00 | [-17.86, -3.08] | 2000 |
| region_decision_decision − region_region_decision | union_box_recall_iou_0.5 | — | — | — | — | 0 |
| region_decision_decision − decision_decision_decision | macro_f1 | 84.14 | 85.83 | -1.69 | [-6.46, +2.54] | 2000 |
| region_decision_decision − decision_decision_decision | balanced_accuracy | 84.17 | 85.83 | -1.67 | [-6.14, +2.64] | 2000 |
| region_decision_decision − decision_decision_decision | fpr | 11.67 | 13.33 | -1.67 | [-7.35, +3.85] | 2000 |
| region_decision_decision − decision_decision_decision | fnr | 20.00 | 15.00 | +5.00 | [-1.82, +12.50] | 2000 |
| region_decision_decision − decision_decision_decision | union_box_recall_iou_0.5 | — | — | — | — | 0 |
| region_region_decision − decision_decision_decision | macro_f1 | 82.22 | 85.83 | -3.61 | [-10.27, +2.81] | 2000 |
| region_region_decision − decision_decision_decision | balanced_accuracy | 82.50 | 85.83 | -3.33 | [-9.65, +2.74] | 2000 |
| region_region_decision − decision_decision_decision | fpr | 5.00 | 13.33 | -8.33 | [-16.95, +0.00] | 2000 |
| region_region_decision − decision_decision_decision | fnr | 30.00 | 15.00 | +15.00 | [+6.56, +24.56] | 2000 |
| region_region_decision − decision_decision_decision | union_box_recall_iou_0.5 | — | — | — | — | 0 |
| region_decision_region − region_region_region | macro_f1 | 80.00 | 83.26 | -3.26 | [-8.38, +2.00] | 2000 |
| region_decision_region − region_region_region | balanced_accuracy | 80.00 | 83.33 | -3.33 | [-8.37, +1.89] | 2000 |
| region_decision_region − region_region_region | fpr | 20.00 | 10.00 | +10.00 | [+1.64, +19.67] | 2000 |
| region_decision_region − region_region_region | fnr | 20.00 | 23.33 | -3.33 | [-8.70, +0.00] | 2000 |
| region_decision_region − region_region_region | union_box_recall_iou_0.5 | 46.67 | 46.67 | +0.00 | [-6.67, +6.45] | 2000 |

区间不包含零仅描述本次验证对比在指定重采样方法下的差异；任务预算取舍与开发选择仍需结合定位、判别及提示一致性共同解释。

## 解释边界

- 本轮结果来自同一批固定120来源的验证集，用于开发比较；冻结确认集未由本脚本读取或评估。
- 三臂来源及曝光序列、更新数和初始化匹配；输入token、监督token、任务难度及计算耗时不相等。
- 混合臂按监督token归一CE，两个任务的曝光数各半不意味着梯度权重各半。
- RD与RR/DD的比较反映固定曝光预算下的任务分配取舍，不能孤立解释为纯坐标信号的因果作用。
- 固定最终192更新；不同目标任务的验证CE不可用作三臂共同性能排名；旧96更新结果仅为历史参考。
- 配对bootstrap区间仅描述当前模型在验证来源重采样下的差异，不包含训练seed变化；四组多指标比较未做多重校正。
- Macro F1平均正常/异常两个标签；框指标为包围矩形的端到端召回，分母包含漏检与格式错误。
- 格式有效性沿用现有parser，允许JSON围栏、浮点坐标和多框，不等于严格整数单框合规。
