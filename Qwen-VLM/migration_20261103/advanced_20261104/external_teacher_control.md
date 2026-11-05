# External post-hoc fixed teacher-box control

State: **complete**. All21 external runs and all6 structured completion/inference/SHA proofs passed before raw prediction, saved target or CPU cache use.

The existing VisA validation rule is reused without fitting: map >= imagewise95th percentile, union bbox, normalized0..1000 and outward rounding. Gated controls emit this box only when the saved same-seed Cstructured score >0. No new RGB, masks or model queries.

| Condition | Seed | All anomalies | IoU | IoU≥0.5 | Center in GT box | Grid3 | Anomaly box coverage | Normal box rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unconditional teacher box (shared) | — | 1183 | 0.224989 | 0.182587 | 0.381234 | 0.475063 | 1.000000 | 1.000000 |
| Cstructured_score_gt0_teacher_box | 42 | 1183 | 0.207184 | 0.172443 | 0.339814 | 0.431107 | 0.923077 | 0.614796 |
| Cstructured_generated | 42 | 1183 | 0.146712 | 0.077768 | 0.279797 | 0.391378 | 0.852071 | 0.563776 |
| S_generated | 42 | 1183 | 0.066040 | 0.010144 | 0.345731 | 0.421809 | 0.834320 | 0.571429 |
| Cstructured_score_gt0_teacher_box | 43 | 1183 | 0.204527 | 0.169907 | 0.335587 | 0.425190 | 0.906171 | 0.576531 |
| Cstructured_generated | 43 | 1183 | 0.034934 | 0.012680 | 0.087912 | 0.100592 | 0.437025 | 0.280612 |
| S_generated | 43 | 1183 | 0.092012 | 0.027895 | 0.329670 | 0.387997 | 0.868977 | 0.563776 |
| Cstructured_score_gt0_teacher_box | 44 | 1183 | 0.219871 | 0.179205 | 0.366864 | 0.459848 | 0.962806 | 0.714286 |
| Cstructured_generated | 44 | 1183 | 0.031679 | 0.017751 | 0.078614 | 0.101437 | 0.332206 | 0.262755 |
| S_generated | 44 | 1183 | 0.051395 | 0.009298 | 0.347422 | 0.411665 | 0.797971 | 0.530612 |

Complete per-category denominators/results, three-seed descriptive means/SD, and same-seed differences are in the JSON report. The unconditional control is one shared deterministic result, not three independent seeds.

This is an external post-hoc fixed control. It does not identify a causal LoRA, language reasoning or mask supervision effect and cannot select a model. Teacher box/grid validity is assigned by code. Unconditional boxes on normal images do not demonstrate useful detection; center-in-GT-box is not pixel-mask localization.
