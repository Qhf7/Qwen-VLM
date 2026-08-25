# v9 architecture figure — visual specification

## Purpose

Explain the complete v9 path in a paper-readable figure: normal-only training
on the left, test-time zero-shot inference on the right, and the principle that
Qwen3-VL adds semantic evidence but cannot replace dense DINOv2 localization.

## Composition

- Landscape canvas: 1800 × 1000.
- Two panels: `(a) TRAIN-ONLY NORMAL MODELING` and `(b) TEST-TIME ZERO-SHOT INFERENCE`.
- Dashed divider separates offline learning from online inference.
- Data flows left-to-right or top-to-bottom; the only cross-panel connector is
  the calibrated dense-control artifact entering the test-time support gate.
- Every module is an editable draw.io card with explicit geometry.

## Semantic palette

| Color | Meaning |
|---|---|
| Pale blue | input, pseudo labels, dense output/control |
| Pale teal | DINOv2 features and learned dense modules |
| Pale gray | hard-normal supervision and calibration |
| Pale orange | Qwen3-VL semantic evidence |
| Pale lavender | fusion and submission output |

The SVG and draw.io files use the same palette and arrow grammar. VLM boxes are
never converted directly into masks; the bounded residual is gated by dense
DINO support.
