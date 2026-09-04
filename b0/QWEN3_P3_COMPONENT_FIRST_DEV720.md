# Qwen3-VL-2B P3 Component-First Dev720 Report

## Scope

This is a post-registration prompt diagnostic. It does not replace the frozen
Qwen2.5-VL-3B B0+ P0 baseline. The model, Dev720 manifest, direct 448x448
preprocessing, five views per group, and forward-only scoring are fixed. Only
the prompt and its anomaly-token convention changed.

| Item | Value |
|---|---|
| Model | `Qwen3-VL-2B-Instruct` |
| Prompt | `P3_COMPONENT_FIRST` |
| Data | fold1 Dev720: 18 classes, 720 groups, 3,600 views |
| Input check | full manifest and cached-pixel hash passed; holdout overlap = 0 |
| Normal / defect token | `Yes` / `No` |
| Continuous anomaly score | `P(No)` from the first-token logits |
| Primary aggregation | 5-view `max` |
| Execution | B0f single-view forward; batch5 differed by 0.00214, above the 0.0005 gate |
| Bootstrap | 2,000 group-level draws, stratified within class x label, seed 20260901 |

## Prompt

```text
Inspect one view of an industrial component.

First identify internally what kind of component it is and its expected repeated structure, edges, holes, markings, color, and surface. Then decide whether any visible local region breaks that expected structure.

The first generated token must be exactly Yes or No. Yes means the component is normal. No means the component has a visible defect.
A defect includes a visible scratch, crack, deformation, discoloration, stain, missing or broken material, pit, burn mark, coating problem, contaminant, or other localized inconsistency. Normal edges, holes, reflections, shadows, text, and regular texture are not defects.

Then output exactly one JSON object. Its component value must be a short component-type description, and its region must be one of: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right, none.
For normal use: Yes {"component":"short component type","region":"none"}
For defective use: No {"component":"short component type","region":"center"}
Do not output Markdown or an explanation.
```

Chinese meaning: first infer the part type and what its normal repeated
structure should look like; then judge whether a local inconsistency is a
defect. `Yes` means normal and `No` means defective, so the reported anomaly
score is `P(No)`.

## Aggregate Results

`max` is the frozen five-view aggregation. `top2_mean` and `mean` are
diagnostic offline aggregations and are not substituted for the primary score.

| Aggregation | Overall AUROC | Overall I-AUPR | Overall F1max | Macro AUROC | Macro I-AUPR | Macro F1max | Normal median P(No) | Defect median P(No) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max (primary) | 0.6620 | 0.6616 | 0.6732 | 0.7867 | 0.8019 | 0.7736 | 0.0097 | 0.0150 |
| top2_mean | 0.6571 | 0.6653 | 0.6757 | 0.7988 | 0.8162 | 0.7920 | 0.0086 | 0.0127 |
| mean | 0.6500 | 0.6637 | 0.6794 | 0.8064 | 0.8238 | 0.8063 | 0.0063 | 0.0090 |

The primary max score passes the requested macro I-AUPR threshold of 0.779:
0.8019. Its 95% stratified-bootstrap interval is [0.7821, 0.8366]. This is a
P3 diagnostic result, not a claim that the frozen P0 baseline changed.

### P(No) Distribution By Label

| Aggregation | Label | Mean | Median | P05 | P95 |
|---|---|---:|---:|---:|---:|
| max | normal | 0.0144 | 0.0097 | 0.0030 | 0.0536 |
| max | defect | 0.0266 | 0.0150 | 0.0041 | 0.0805 |
| top2_mean | normal | 0.0117 | 0.0086 | 0.0027 | 0.0411 |
| top2_mean | defect | 0.0212 | 0.0127 | 0.0037 | 0.0632 |
| mean | normal | 0.0081 | 0.0063 | 0.0020 | 0.0258 |
| mean | defect | 0.0136 | 0.0090 | 0.0029 | 0.0388 |

## Primary Per-Class Results

| Class | AUROC | I-AUPR | F1max |
|---|---:|---:|---:|
| 3_adapter | 0.7588 | 0.7140 | 0.7619 |
| PLCC_socket | 0.8625 | 0.8939 | 0.7907 |
| accurate_detection_switch | 0.6888 | 0.7276 | 0.7037 |
| audio_jack_socket | 0.6675 | 0.6588 | 0.7234 |
| battery_holder_connector | 0.7600 | 0.8000 | 0.7347 |
| battery_socket_connector | 0.7925 | 0.8114 | 0.7755 |
| ceramic_fuse | 0.8013 | 0.8064 | 0.7895 |
| ceramic_wave_filter | 0.7888 | 0.7897 | 0.7547 |
| jam_jar_model | 0.6388 | 0.6403 | 0.7018 |
| knob_cap | 0.7913 | 0.8290 | 0.7568 |
| lilypad_led | 0.7188 | 0.7021 | 0.7273 |
| miniature_laser_module | 0.6925 | 0.7234 | 0.7273 |
| mobile_charging_connector | 0.8625 | 0.8889 | 0.8571 |
| nylon_ball_head | 0.8363 | 0.8459 | 0.7727 |
| pneumatic_elbow | 0.9325 | 0.9382 | 0.8947 |
| side_press_switch | 0.9213 | 0.9308 | 0.8421 |
| slide_switch | 0.9338 | 0.9542 | 0.9189 |
| smd_receiver_module | 0.7125 | 0.7802 | 0.6923 |

## Bootstrap Intervals

| Aggregation | Metric | 95% CI |
|---|---|---:|
| max | macro I-AUPR | [0.7821, 0.8366] |
| max | macro F1max | [0.7744, 0.8161] |
| max | overall I-AUPR | [0.6386, 0.6850] |
| top2_mean | macro I-AUPR | [0.7969, 0.8508] |
| top2_mean | macro F1max | [0.7914, 0.8333] |
| mean | macro I-AUPR | [0.8048, 0.8593] |
| mean | macro F1max | [0.8036, 0.8432] |

## Complete-Response Audit

Thirty-six balanced views, one normal and one defective view for each class,
were generated with 64 tokens. This audit validates response behavior only; it
does not replace first-token-logit scoring.

| Check | Result |
|---|---:|
| Responses containing `component` and `region` keys | 36 / 36 |
| Responses beginning with `Yes` | 35 / 36 |
| Responses beginning with `No` | 0 / 36 |
| Normal examples beginning with `Yes` | 17 / 18 |
| Defect examples beginning with `Yes` | 18 / 18 |
| One malformed first-token response | 1 / 36 (JSON started directly) |

The generated text still has a strong normal-answer bias, so greedy text
decisions are not usable as a defect classifier. The first-token probability is
different: normal and defective groups have separated `P(No)` distributions,
which explains why the ranking metrics above are useful despite nearly all
greedy answers being `Yes`.

## Reproducible Artifacts

* `artifacts/b0_cross/qwen3_vl_2b_component_first_p3_dev720/frozen_protocol.json`
* `artifacts/b0_cross/qwen3_vl_2b_component_first_p3_dev720/P3_COMPONENT_FIRST_views.csv`
* `artifacts/b0_cross/qwen3_vl_2b_component_first_p3_dev720/generation_audit.csv`
* `artifacts/b0_cross/qwen3_vl_2b_component_first_p3_dev720/analysis/CROSS_MODEL_REPORT.md`
* The CSV files beside that report contain every aggregation, all 18 per-class
  rows, score distributions, calibration diagnostic, and bootstrap draws.
