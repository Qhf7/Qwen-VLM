# Structured localization continuation: VisA development validation

Report state: partial. Expected seeds:42/43/44; included seeds:42,43.

All classification and localization results are recomputed from raw predictions. Each run uses the same380 validation images (172normal/208anomaly); seeds are training repeats, not independent test sets. Best checkpoints are selected and reported on that development validation set.

| Condition | AUROC mean±sample SD | AP | generated macro F1 | recall | FPR | format valid | anomaly mean box IoU | IoU≥.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.903979 ± 0.026119 | 0.927854 ± 0.020109 | 0.821291 ± 0.025448 | 0.689904 ± 0.064591 | 0.017442 ± 0.024667 | 0.892105 ± 0.126535 | 0.038943 ± 0.046235 | 0.014423 ± 0.020397 |
| best | 0.937696 ± 0.030813 | 0.949830 ± 0.028222 | 0.850264 ± 0.070334 | 0.733173 ± 0.132583 | 0.005814 ± 0.008222 | 0.956579 ± 0.013026 | 0.072709 ± 0.004542 | 0.026442 ± 0.023797 |
| final | 0.937297 ± 0.030250 | 0.949704 ± 0.028044 | 0.846309 ± 0.064740 | 0.721154 ± 0.115585 | 0.000000 ± 0.000000 | 0.938158 ± 0.039077 | 0.073321 ± 0.003676 | 0.031250 ± 0.016998 |

## Paired differences

Differences are computed within matching seed before aggregation; absolute fractional differences, not relative percentages.

| Difference | AUROC mean±sample SD | generated macro F1 | anomaly mean box IoU | format valid |
|---|---:|---:|---:|---:|
| best_minus_initial | 0.033717 ± 0.004694 | 0.028973 ± 0.044885 | 0.033766 ± 0.050777 | 0.064474 ± 0.139561 |
| final_minus_initial | 0.033318 ± 0.004131 | 0.025018 ± 0.039292 | 0.034378 ± 0.049912 | 0.046053 ± 0.165612 |

## Teacher-map bbox controls

| Fixed control | anomaly mean box IoU | IoU≥.5 | box center in GT box |
|---|---:|---:|---:|
| unconditional_box | 0.101963 ± 0.000000 | 0.086538 ± 0.000000 | 0.168269 ± 0.000000 |
| initial_score_gated | 0.096127 ± 0.003129 | 0.084135 ± 0.003400 | 0.156250 ± 0.003400 |

## Resource and audit boundaries

Teacher-map bbox controls and complete per-seed/per-category results are in summary.json. Cache availability determines whether the teacher baseline was independently recomputed; no stored-only control is mislabeled as recomputed.

S adds mask-derived target supervision, a schema prompt, longer answers, and 2000 further image exposures while updating language LoRA. Its difference from original binary C cannot isolate a LoRA mechanism or demonstrate equal-token/equal-supervision gains. Spatial outputs are union-box predictions, not segmentation or causal explanations. Invalid/missing/No boxes score zero on all-anomaly localization metrics; conditional valid-positive metrics retain their coverage denominator.

This summary never opens external evaluation directories and does not choose a checkpoint from external data.

| Seed | best step | training seconds | exposures/s | peak GiB | best reload | final reload |
|---|---:|---:|---:|---:|---|---|
| 42 | 400 | 226.26 | 8.839 | 17.631 | complete | complete |
| 43 | 500 | 225.50 | 8.869 | 17.631 | complete | missing |
