# Industrial VLM experiment matrix

Values below are observed evaluation outputs. Rates are percentages; — means undefined or unavailable.

| Run | Branch | N / sources | Valid | Anomaly F1 | Macro F1 | BAcc | FPR | FNR | Normal failure | Empty boxes on normal | Union-box R@.3 | Union-box R@.5 | Latency s | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_decision | region_supervision_ablation | 120 / 120 | 100.00 | 55.42 | 65.93 | 69.17 | 0.00 | 61.67 | 0.00 | — | — | — | 0.674 | 6.981 |
| base_region | base_vs_sft | 120 / 120 | 100.00 | 45.57 | 59.43 | 64.17 | 1.67 | 70.00 | 1.67 | 98.33 | 6.67 | 1.67 | 1.232 | 6.993 |
| external_base_region | external_pilot | 150 / 150 | 100.00 | 64.96 | 71.28 | 72.67 | 5.33 | 49.33 | 5.33 | 94.67 | 28.00 | 26.67 | 1.224 | 6.987 |
| external_quality_region | external_pilot | 150 / 150 | 100.00 | 76.92 | 73.58 | 74.00 | 38.67 | 13.33 | 38.67 | 61.33 | 60.00 | 53.33 | 1.686 | 7.044 |
| overfit32_training_diagnostic | training_fit_diagnostic | 32 / 32 | 100.00 | 82.76 | 84.24 | 84.38 | 6.25 | 25.00 | 6.25 | 93.75 | 75.00 | 68.75 | 1.425 | 7.050 |
| quality_decision | region_supervision_ablation | 120 / 120 | 100.00 | 85.22 | 85.81 | 85.83 | 10.00 | 18.33 | 10.00 | — | — | — | 0.710 | 7.039 |
| quality_region | base_vs_sft | 120 / 120 | 100.00 | 82.14 | 83.26 | 83.33 | 10.00 | 23.33 | 10.00 | 90.00 | 46.67 | 25.00 | 1.446 | 7.050 |
| quality_region_decision_prompt | region_supervision_ablation | 120 / 120 | 100.00 | 68.82 | 74.54 | 75.83 | 1.67 | 46.67 | 1.67 | — | — | — | 0.723 | 7.039 |
| random_region | dataquality_sampling_ablation | 120 / 120 | 100.00 | 79.28 | 80.72 | 80.83 | 11.67 | 26.67 | 11.67 | 88.33 | 40.00 | 28.33 | 1.438 | 7.050 |

## Protocol and interpretation

Format failures remain in denominators. FPR counts explicit anomalous predictions on normal images; normal failure additionally counts invalid normal outputs. FNR includes invalid anomalous outputs.

Union-box recall uses the enclosing rectangle of all boxes and all anomalous region-task rows. It is neither instance-matched detection recall nor pixel/mask segmentation. Decision-only runs have no localization score.

The data-quality sampling branch compares category×label-matched random selection with defect-area coverage. It does not establish general label-noise removal.

## Matched evaluation conditions

- Condition 1: base_decision, quality_decision, quality_region_decision_prompt
- Condition 2: base_region, quality_region, random_region
- Condition 3: external_base_region, external_quality_region
- Condition 4: overfit32_training_diagnostic

## base_decision: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 40.00–67.44 |
| macro_f1 | 55.06–73.96 |
| balanced_accuracy | 62.50–75.44 |
| fpr | 0.00–0.00 |
| fnr | 48.44–75.00 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | —–— |
| union_box_recall_iou_0.5 | —–— |

## base_region: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 29.09–56.47 |
| macro_f1 | 49.58–66.98 |
| balanced_accuracy | 58.33–69.10 |
| fpr | 0.00–5.17 |
| fnr | 59.26–82.35 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 1.61–14.55 |
| union_box_recall_iou_0.5 | 0.00–6.35 |

## external_base_region: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 54.39–74.38 |
| macro_f1 | 62.96–78.53 |
| balanced_accuracy | 66.07–78.57 |
| fpr | 1.22–10.81 |
| fnr | 38.46–60.00 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 17.57–38.89 |
| union_box_recall_iou_0.5 | 17.11–38.89 |

## external_quality_region: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 68.71–83.33 |
| macro_f1 | 65.93–80.56 |
| balanced_accuracy | 67.24–80.47 |
| fpr | 26.67–50.00 |
| fnr | 5.56–20.31 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 48.68–71.23 |
| union_box_recall_iou_0.5 | 42.67–65.48 |

## overfit32_training_diagnostic: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 60.00–93.75 |
| macro_f1 | 69.46–93.75 |
| balanced_accuracy | 70.39–94.12 |
| fpr | 0.00–18.75 |
| fnr | 7.14–50.00 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 47.06–92.86 |
| union_box_recall_iou_0.5 | 46.15–87.50 |

## quality_decision: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 75.51–91.23 |
| macro_f1 | 77.83–91.65 |
| balanced_accuracy | 77.92–91.89 |
| fpr | 3.12–19.67 |
| fnr | 9.09–28.30 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | —–— |
| union_box_recall_iou_0.5 | —–— |

## quality_region: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 74.75–89.06 |
| macro_f1 | 76.64–88.94 |
| balanced_accuracy | 77.48–89.03 |
| fpr | 3.03–17.74 |
| fnr | 13.46–33.90 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 35.48–60.29 |
| union_box_recall_iou_0.5 | 14.93–36.92 |

## quality_region_decision_prompt: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 56.60–79.07 |
| macro_f1 | 64.48–81.95 |
| balanced_accuracy | 68.80–82.35 |
| fpr | 0.00–5.26 |
| fnr | 32.00–58.62 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | —–— |
| union_box_recall_iou_0.5 | —–— |

## random_region: source-cluster bootstrap intervals

| Metric | 95% interval |
|---|---:|
| anomaly_f1 | 70.83–86.40 |
| macro_f1 | 73.86–87.11 |
| balanced_accuracy | 74.09–86.71 |
| fpr | 4.92–19.64 |
| fnr | 15.62–38.10 |
| format_valid_rate | 100.00–100.00 |
| union_box_recall_iou_0.3 | 26.98–52.31 |
| union_box_recall_iou_0.5 | 16.95–39.68 |

Per-category and anomaly-area breakdowns, source paths, and evaluation metadata are retained in matrix.json.
