# Paired source-image bootstrap differences

All differences are left minus right in percentage points (pp). Lower FPR/FNR is better; higher F1, balanced accuracy, and union-box recall is better.

Bootstrap: 2000 paired source-cluster draws, seed 17; percentile 95% intervals.

| Left − right | Metric | Left % | Right % | Difference pp | 95% CI pp | Defined draws |
|---|---|---:|---:|---:|---:|---:|
| quality_region − base_region | macro_f1 | 83.26 | 59.43 | +23.83 | [+13.78, +34.27] | 2000 |
| quality_region − base_region | balanced_accuracy | 83.33 | 64.17 | +19.17 | [+11.01, +27.20] | 2000 |
| quality_region − base_region | fpr | 10.00 | 1.67 | +8.33 | [+0.00, +17.46] | 2000 |
| quality_region − base_region | fnr | 23.33 | 70.00 | -46.67 | [-60.94, -32.20] | 2000 |
| quality_region − base_region | union_box_recall_iou_0.5 | 25.00 | 1.67 | +23.33 | [+12.00, +35.09] | 2000 |
| quality_region − random_region | macro_f1 | 83.26 | 80.72 | +2.53 | [-1.64, +6.70] | 2000 |
| quality_region − random_region | balanced_accuracy | 83.33 | 80.83 | +2.50 | [-1.67, +6.70] | 2000 |
| quality_region − random_region | fpr | 10.00 | 11.67 | -1.67 | [-8.48, +5.88] | 2000 |
| quality_region − random_region | fnr | 23.33 | 26.67 | -3.33 | [-8.62, +0.00] | 2000 |
| quality_region − random_region | union_box_recall_iou_0.5 | 25.00 | 28.33 | -3.33 | [-10.21, +3.12] | 2000 |
| quality_region_decision_prompt − quality_decision | macro_f1 | 74.54 | 85.81 | -11.26 | [-19.29, -3.36] | 2000 |
| quality_region_decision_prompt − quality_decision | balanced_accuracy | 75.83 | 85.83 | -10.00 | [-17.17, -2.91] | 2000 |
| quality_region_decision_prompt − quality_decision | fpr | 1.67 | 10.00 | -8.33 | [-16.39, +0.00] | 2000 |
| quality_region_decision_prompt − quality_decision | fnr | 46.67 | 18.33 | +28.33 | [+17.46, +40.30] | 2000 |
| quality_region_decision_prompt − quality_decision | union_box_recall_iou_0.5 | — | — | — | — | 0 |
| external_quality_region − external_base_region | macro_f1 | 73.58 | 71.28 | +2.30 | [-6.62, +11.10] | 2000 |
| external_quality_region − external_base_region | balanced_accuracy | 74.00 | 72.67 | +1.33 | [-6.28, +8.45] | 2000 |
| external_quality_region − external_base_region | fpr | 38.67 | 5.33 | +33.33 | [+22.78, +44.19] | 2000 |
| external_quality_region − external_base_region | fnr | 13.33 | 49.33 | -36.00 | [-47.44, -24.71] | 2000 |
| external_quality_region − external_base_region | union_box_recall_iou_0.5 | 53.33 | 26.67 | +26.67 | [+12.79, +40.01] | 2000 |

## Interpretation limits

- One training seed and fixed small pilot test sets; intervals quantify source-sampling uncertainty conditional on these trained models.
- Intervals do not include training-seed variation, dataset shift, or benchmark pretraining contamination.
- Four planned comparisons and multiple metrics are descriptive; no multiplicity correction or SOTA claim is made.
- macro_f1 averages the normal/anomalous label F1 values, not product-category F1 values.
- Format failures remain in metric denominators under the shared evaluation policy. Decision-only localization is undefined, not zero.
- Bootstrap draws lacking a required label yield undefined rates; those draws are omitted for that metric and their defined counts are reported.
- Model path and base config hash are checked; this is not a hash audit of all base weight shards.

A CI excluding zero describes this pilot contrast under the stated resampling model. It does not establish broad superiority or state-of-the-art performance.
