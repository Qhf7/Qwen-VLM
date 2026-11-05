# Frozen external transfer audit

State: **complete**. Each included run is recomputed from all1575 saved raw predictions in the frozen source order.

| Model | Seeds | AUROC mean±SD | AP | Generated F1 | Recall | FPR | Category AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3/3 | 0.708834±0.045954 | 0.857853 | 0.671986 | 0.923359 | 0.612245 | 0.818616 |
| C | 3/3 | 0.711363±0.045839 | 0.859958 | 0.657848 | 0.935475 | 0.650510 | 0.824035 |
| D | 3/3 | 0.714118±0.053733 | 0.860675 | 0.651824 | 0.939138 | 0.668367 | 0.824338 |
| Cpair | 3/3 | 0.709581±0.044448 | 0.858676 | 0.653811 | 0.935193 | 0.659014 | 0.821618 |
| R | 3/3 | 0.703438±0.030671 | 0.856832 | 0.636020 | 0.958580 | 0.715986 | 0.826927 |
| Cstructured | 3/3 | 0.699067±0.031734 | 0.855060 | 0.667084 | 0.930966 | 0.631803 | 0.786499 |
| S | 3/3 | 0.717814±0.022595 | 0.865601 | 0.670401 | 0.921105 | 0.614796 | 0.817356 |

Differences are computed within the same seed before aggregation. Partial seed groups remain explicitly partial.

| Contrast | Pairs | ΔAUROC | ΔAP | ΔGenerated F1 | ΔRecall | ΔFPR |
|---|---:|---:|---:|---:|---:|---:|
| C_minus_A | 3/3 | 0.002529 | 0.002105 | -0.014138 | 0.012116 | 0.038265 |
| D_minus_C | 3/3 | 0.002754 | 0.000717 | -0.006024 | 0.003663 | 0.017857 |
| D_minus_A | 3/3 | 0.005284 | 0.002822 | -0.020162 | 0.015779 | 0.056122 |
| R_minus_Cpair | 3/3 | -0.006143 | -0.001844 | -0.017791 | 0.023387 | 0.056973 |
| Cpair_minus_C | 3/3 | -0.001782 | -0.001282 | -0.004037 | -0.000282 | 0.008503 |
| Cstructured_minus_C | 3/3 | -0.012297 | -0.004897 | 0.009236 | -0.004508 | -0.018707 |
| S_minus_Cstructured | 3/3 | 0.018747 | 0.010540 | 0.003317 | -0.009862 | -0.017007 |

Checkpoint steps, source hashes, training parameters, inference costs, per-category seed means/differences and localization denominators are in the JSON artifact.

A is the original selected source bridge; Cstructured reuses C weights with structured instructions. S adds mask supervision and answer tokens. Basic max generation32 and structured80 differ; cross-schema comparisons do not isolate a LoRA mechanism.

This is a public cross-dataset audit. VisA validation selects checkpoints; MVTec results cannot select models or prompts. Foundation pretraining exposure is unknown. Sample SD across3 seeds is not a significance test.
