# Jev follow-up figures

## Files

- figures/followup_roi_routing_calibration.png: 300-DPI raster, 5400 x 2460 pixels.
- figures/followup_roi_routing_calibration.svg: scalable figure with editable English text.
- figures/followup_roi_routing_calibration.provenance.json: source, script, font and output hashes.

## What the three panels show

(A) ROI and duplicate gates use full denominators: all60 anomalies for recall and mean IoU; all60 normal images for FPR. Rates and IoU are shown in original0..1 units on separate axes. Rejected anomaly boxes contribute zero IoU.
(B) Fixed routing policies compare all-anomaly mean IoU with successful logical request count for120 queries. The Y axis is explicitly truncated. Points are exact, without jitter or interpolating curves. Matched-random seed44 and the uncertainty route coincide. Logical replay budgets do not undo the240 calls and38 failures actually collected.
(C) Whole-image raw and5fold OOF logistic Brier scores share the same120 source-level folds. Calibration is fixedC1/L2 and not a foundation-model training result. QwenDD Brier worsens slightly after OOF; the plot preserves this negative result.

## Caption

Scope: the Teacher trained on 96/120 query sources; the remaining 24 contain only 2 normal images. These are viewed development diagnostics.
Candidates: 60/120 union-95% boxes cover the full image and have identical query/crop pixels. ROI mean-IoU delta = +0.00118; paired 95% CI crosses zero.
Budget: panel B shows replay budgets, not actual bills; all 240 paired calls were collected (+38 failed attempts). Random seed 44 overlaps the uncertainty point.
Calibration: whole-image scores only; OOF protocol formed after ROI evaluation but before calibration fitting. No independent calibration or confirmation was completed.

## Exact values

| Gate | Anomaly recall | Normal FPR | All-anomaly mean IoU |
|---|---:|---:|---:|
| duplicate | 0.650000000 | 0.183333333 | 0.068542326 |
| roi | 0.716666667 | 0.216666667 | 0.069718680 |

| Routing policy | Logical successful requests | All-anomaly mean IoU | Anomaly recall | Normal FPR |
|---|---:|---:|---:|---:|
| duplicate | 120 | 0.068542326 | 0.650000000 | 0.183333333 |
| roi | 120 | 0.069718680 | 0.716666667 | 0.216666667 |
| sequential_uncertain | 162 | 0.069689213 | 0.700000000 | 0.233333333 |
| geometry_small | 153 | 0.069410464 | 0.683333333 | 0.233333333 |
| matched_random_s42 | 162 | 0.069550343 | 0.683333333 | 0.233333333 |
| matched_random_s43 | 162 | 0.068959945 | 0.683333333 | 0.233333333 |
| matched_random_s44 | 162 | 0.069689213 | 0.700000000 | 0.233333333 |

| Score | Raw Brier | OOF logistic Brier |
|---|---:|---:|
| jev_0shot | 0.174259167 | 0.167705501 |
| jev_1shot | 0.157224167 | 0.147661360 |
| jev_4shot | 0.141269167 | 0.134715695 |
| qwen_base | 0.298766034 | 0.165891020 |
| qwen_decision_decision | 0.093156673 | 0.094009218 |

## Interpretation boundaries

- The figure shows diagnostic trade-offs, not an independent test or a production guarantee.
- ROI/duplicate is the matched new-arm comparison. Historical Jev rendering differs, and Qwen max_pixels524288 differs from Jev448; cross-model values are not an architectural causal comparison.
- The ROI paired mean-IoU difference and interval remain in stage05; the caption says the interval crosses zero. No error bars are drawn; omission is not a claim of significance.
- Stage07 raw/OOF uses p>=0.5; native Qwen default uses strict logodds>0. Equality cases differ and are documented in the calibration report.
- Font is verified DejaVu Sans. All figure text is English to avoid missing CJK glyphs.

## Provenance

Script SHA256: 065826ba5849796257d3459a2daebf62119a1b39a61dd2ce26ac362bd6f25527
stage05 SHA256: fe2702586f60d575b319790c7ec37b737bc189d864d6f6238b98f34588992a92
stage06 SHA256: 51e0f1791d01c163859377197f56f27b1e7819657e600c5138da122ff323f0fd
stage07 SHA256: a5bb87f0f6863143ca44b8b60a3c030d37d9bb710a8af9f6308481f0719f9e8b
Generated on the designated AutoDL server. No API/model call, RGB or GT-mask read, or frozen scientific artifact change.
