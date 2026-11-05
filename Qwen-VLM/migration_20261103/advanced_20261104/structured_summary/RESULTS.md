# Structured localization continuation: VisA development validation

Report state: complete. Expected seeds:42/43/44; included seeds:42,43,44.

All classification and localization results are recomputed from raw predictions. Each run uses the same380 validation images (172normal/208anomaly); seeds are training repeats, not independent test sets. Best checkpoints are selected and reported on that development validation set.

| Condition | AUROC mean±sample SD | AP | generated macro F1 | recall | FPR | format valid | anomaly mean box IoU | IoU≥.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.899975 ± 0.019728 | 0.923891 ± 0.015790 | 0.821152 ± 0.017996 | 0.719551 ± 0.068724 | 0.054264 ± 0.066119 | 0.849123 ± 0.116396 | 0.030993 ± 0.035475 | 0.009615 ± 0.016654 |
| best | 0.939489 ± 0.022009 | 0.952932 ± 0.020666 | 0.841060 ± 0.052226 | 0.716346 ± 0.098176 | 0.005814 ± 0.005814 | 0.959649 ± 0.010635 | 0.068755 ± 0.007564 | 0.022436 ± 0.018202 |
| final | 0.939009 ± 0.021595 | 0.952962 ± 0.020617 | 0.849197 ± 0.046051 | 0.727564 ± 0.082481 | 0.001938 ± 0.003357 | 0.950000 ± 0.034412 | 0.074495 ± 0.003300 | 0.024038 ± 0.017334 |

## Paired differences

Differences are computed within matching seed before aggregation; absolute fractional differences, not relative percentages.

| Difference | AUROC mean±sample SD | generated macro F1 | anomaly mean box IoU | format valid |
|---|---:|---:|---:|---:|
| best_minus_initial | 0.039514 ± 0.010576 | 0.019908 ± 0.035410 | 0.037762 ± 0.036566 | 0.110526 ± 0.126890 |
| final_minus_initial | 0.039035 ± 0.010323 | 0.028045 ± 0.028274 | 0.043502 ± 0.038669 | 0.100877 ± 0.150768 |

## Teacher-map bbox controls

| Fixed control | anomaly mean box IoU | IoU≥.5 | box center in GT box |
|---|---:|---:|---:|
| unconditional_box | 0.101963 ± 0.000000 | 0.086538 ± 0.000000 | 0.168269 ± 0.000000 |
| initial_score_gated | 0.097394 ± 0.003115 | 0.084936 ± 0.002776 | 0.158654 ± 0.004808 |

## Resource and audit boundaries

Teacher-map bbox controls and complete per-seed/per-category results are in summary.json. Cache availability determines whether the teacher baseline was independently recomputed; no stored-only control is mislabeled as recomputed.

S adds mask-derived target supervision, a schema prompt, longer answers, and 2000 further image exposures while updating language LoRA. Its difference from original binary C cannot isolate a LoRA mechanism or demonstrate equal-token/equal-supervision gains. Spatial outputs are union-box predictions, not segmentation or causal explanations. Invalid/missing/No boxes score zero on all-anomaly localization metrics; conditional valid-positive metrics retain their coverage denominator.

This summary never opens external evaluation directories and does not choose a checkpoint from external data.

| Seed | best step | training seconds | exposures/s | peak GiB | best reload | final reload |
|---|---:|---:|---:|---:|---|---|
| 42 | 400 | 226.26 | 8.839 | 17.631 | complete | complete |
| 43 | 500 | 225.50 | 8.869 | 17.631 | complete | complete |
| 44 | 300 | 222.41 | 8.992 | 17.631 | complete | complete |
