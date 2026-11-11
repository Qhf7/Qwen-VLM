# 固定经验 FPR 运行点诊断

本诊断仅使用 stage12 已有 calibration120（60 normal、60 anomaly）与既有 heldout120（60 normal、60 anomaly）。未发 API、未运行 GPU、未重拟合校准器，未读取 stage13。

## 事前固定规则

同时报告 α=5% 与 10%，不根据 held-out 结果反选。将 60 个校准 normal 分数降序排列，取第 floor(α×60)+1 高为阈值，且只有 score > threshold 判异常；因此分别使用第 4/7 高分数。阈值先持久化，再在本诊断执行中首次读取 held-out 标签。

冻结映射为 sigmoid(1.56720531322167×logit(clip(p,1e−6,1−1e−6))−0.15632219069119976)。正系数单调映射改变数值尺度，一般保持排名和对应阈值决策；本次逐对核验 ties、clip 及全部决策。

| 目标经验FPR | 分数 | 阈值 | 校准FP / normal | 校准实际FPR | Held-out FP / normal（FPR; Wilson95%） | Held-out TP / anomaly（Recall; Wilson95%） |
|---:|---|---:|---:|---:|---|---|
| 5.00% | raw | 0.63 | 3/60 | 5.00% | 2/60（3.33%; [0.92%, 11.36%]） | 45/60（75.00%; [62.77%, 84.22%]） |
| 5.00% | calibrated | 0.663240987762 | 3/60 | 5.00% | 2/60（3.33%; [0.92%, 11.36%]） | 45/60（75.00%; [62.77%, 84.22%]） |
| 10.00% | raw | 0.58 | 5/60 | 8.33% | 3/60（5.00%; [1.71%, 13.70%]） | 49/60（81.67%; [70.08%, 89.44%]） |
| 10.00% | calibrated | 0.586503596776 | 5/60 | 8.33% | 3/60（5.00%; [1.71%, 13.70%]） | 49/60（81.67%; [70.08%, 89.44%]） |

## 并列与单调性核验

```json
{
  "calibration": {
    "unordered_pairs": 7140,
    "pairwise_order_or_tie_mismatches": 0,
    "raw_tied_pairs": 93,
    "calibrated_tied_pairs": 93,
    "raw_unique_scores": 62,
    "calibrated_unique_scores": 62,
    "scores_clipped": 0,
    "decision_mismatches_by_alpha": {
      "0.05": 0,
      "0.1": 0
    }
  },
  "heldout_eval": {
    "unordered_pairs": 7140,
    "pairwise_order_or_tie_mismatches": 0,
    "raw_tied_pairs": 87,
    "calibrated_tied_pairs": 87,
    "raw_unique_scores": 62,
    "calibrated_unique_scores": 62,
    "scores_clipped": 0,
    "decision_mismatches_by_alpha": {
      "0.05": 0,
      "0.1": 0
    }
  }
}
```

所有等于阈值的样本均按 strict > 规则判正常。并列可能使校准实际 FPR 低于目标；本次 ties 数量及 raw/calibrated 决策差异见表和 JSON。

## 解释边界

- 这是已看开发验证集上的新运行点诊断，不是新的盲测，不读取或代表 stage13 确认集。
- 目标是校准集经验 FPR，不是保证真实总体 FPR 固定为 5%/10%。60 个 normal 使一张误报就相当于 1.67 个百分点，尾部阈值分别只由第 4/7 个极值决定；存在抽样误差、类别组成和分布迁移风险。
- Held-out FPR 和 Recall 必须同时报告计数与区间。Wilson95% 为边际描述性区间，未做多重比较校正；对用同一校准 normal 自适应选择的阈值，校准 FPR 的二项区间也不能担保阈值选择后总体风险。
- raw 与 calibrated 共享同一模型预测与正单调映射，排名不变时固定经验 FPR 下的 Recall 不会凭校准凭空提高。不得把尺度改善误写成检测排名提升。
- 两个 α 都是事前固定的报告点，不依据 held-out 表挑选后宣称最优方案。

协议 SHA256：`851c59fa71f79997ee033effe1164c89743124c8814e81cd796599b25b0f2fb1`；runner：`29d4a3f37a9464dbe35310bc75eeb4836ac3e4032124fe70ff34edacf2014ca9`；阈值：`ffbc25d10d69ef8ef3516b0cde7441aebf92301008ba4996f732d5383300b6ce`；结果：`db603bbeafbbd471cdb62c3b6076a1446b6c4edc3c4f6bad51e059ff3618cc78`。
