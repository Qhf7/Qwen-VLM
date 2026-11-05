# External structured output diagnostics

Read-only posthoc diagnostics; frozen prompts, parser, checkpoints and thresholds are unchanged.

| Model | Seed | n (normal/anomaly) | Strict valid | Yes / No / invalid | Main strict failure |
|---|---:|---:|---:|---:|---|
| Cstructured | 42 | 1575 (392/1183) | 1471/1575 | 1333 / 242 / 0 | location_bbox_center_inconsistency (104) |
| Cstructured | 43 | 1575 (392/1183) | 907/1575 | 1295 / 280 / 0 | location_bbox_center_inconsistency (667) |
| Cstructured | 44 | 1575 (392/1183) | 652/1575 | 1419 / 156 / 0 | location_bbox_center_inconsistency (923) |
| S | 42 | 1575 (392/1183) | 1417/1575 | 1369 / 206 / 0 | location_bbox_center_inconsistency (158) |
| S | 43 | 1575 (392/1183) | 1515/1575 | 1309 / 266 / 0 | location_bbox_center_inconsistency (60) |
| S | 44 | 1575 (392/1183) | 1413/1575 | 1314 / 261 / 0 | location_bbox_center_inconsistency (162) |

Strict failure counts are mutually exclusive first failures; excluding the `valid` entry, they sum to all invalid outputs. The JSON includes each category, true-label denominator, bbox geometry, location and decoded-text length distributions.

- Cstructured seed 42: external format 0.9340 (1471/1575); VisA format 0.9816 (373/380). Different class/category composition prevents treating this as a paired image comparison.
- Cstructured seed 43: external format 0.5759 (907/1575); VisA format 0.8026 (305/380). Different class/category composition prevents treating this as a paired image comparison.
- Cstructured seed 44: external format 0.4140 (652/1575); VisA format 0.7632 (290/380). Different class/category composition prevents treating this as a paired image comparison.
- S seed 42: external format 0.8997 (1417/1575); VisA format 0.9474 (360/380). Different class/category composition prevents treating this as a paired image comparison.
- S seed 43: external format 0.9619 (1515/1575); VisA format 0.9658 (367/380). Different class/category composition prevents treating this as a paired image comparison.
- S seed 44: external format 0.8971 (1413/1575); VisA format 0.9658 (367/380). Different class/category composition prevents treating this as a paired image comparison.

Coverage: 6/6 requested outputs passed; 6/6 of the full structured suite were included in this invocation.
Prompt source: `structured_model.py::_template_ids`; external uses the generic noun `an industrial component`, while VisA uses category descriptions. Both retain the same schema instructions and 80-token budget.
Token counts are available only when a local tokenizer is explicitly supplied; saved decoded text excludes EOS, so reencoded length is a diagnostic and does not establish the original generation token trace.
