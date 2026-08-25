# Best Hybrid VLM v9

`v9_best_hybrid_vlm` is an independent continuation of the best evaluated
baseline, `qwen_dinov2_hybrid_v2`. It does not modify `v4_hybrid_v2`, prior
checkpoints, predictions, or submitted ZIP files.

The detailed Chinese design and analysis are in
`../BEST_HYBRID_VLM_V9_METHOD.md`.

## Design

The dense anomaly map remains DINOv2-first:

```text
multi-layer DINOv2 features at 1036 and 518
  -> reconstruction distance + category/view memory bank + pseudo-anomaly head
  -> train-normal calibration and multi-scale fusion
  -> dense control mask
  -> bounded Qwen3-VL local semantic residual
  -> hybrid mask
```

Qwen3-VL crop boxes are never rasterized as masks. A Qwen proposal can only
increase an existing DINO anomaly response, by at most 0.12 per pixel. The
pipeline always produces a DINO dense-control ZIP and a VLM-hybrid ZIP so the
semantic branch can be evaluated independently.

## v9 Changes

- Category-stratified Train-normal calibration rather than a sorted prefix.
- Additional `microcrack` and `low_contrast_stain` pseudo defects.
- Hard normal supervision for structural edges, highlights, strong textures,
  foreground boundaries, and background.
- Local Qwen3-VL evidence combines structured flag, score tier, reason,
  compact-box area, overlapping crops, and five-view agreement.
- Residual and image-score contributions are bounded to protect the validated
  Hybrid v2 ranking behaviour.

## Requirements

- Official `Dataset` directory, including `Train` and `Test_A`.
- Generic public DINOv2 ViT-B/14 weights in `WEIGHT_DIR`.
- Local Qwen3-VL crop-audit JSONL for all Test_A groups. It must be generated
  without test labels or external anomaly-specific data.
- Python packages used by the existing baseline: `torch`, `timm`, `numpy`,
  `Pillow`, `scikit-learn`, and `tqdm`.

All trainable components use only official Train normal images and procedural
pseudo anomalies. No Test label is read at any point.

## Local Checks

Run from the repository root:

```bash
python3 -m py_compile v9_best_hybrid_vlm/*.py
python3 v9_best_hybrid_vlm/test_v9.py
bash -n v9_best_hybrid_vlm/run_v9_pipeline.sh
python3 v9_best_hybrid_vlm/validate_submission_v9.py \
  submission_qwen_dinov2_hybrid_v2.zip
```

The final command checks historical ZIP compatibility only; it does not change
the historical submission.

## AutoDL Pipeline

Choose a fresh `RUN_ROOT`. `run_v9_pipeline.sh` refuses to run when that
directory already exists, so it cannot overwrite prior outputs.

```bash
DATA_ROOT=/root/autodl-tmp/Dataset \
WEIGHT_DIR=/root/autodl-tmp/weights \
CLIP_WEIGHT=/root/autodl-tmp/weights/ViT-B-16.pt \
VLM_AUDIT=/root/autodl-tmp/qwen_crop_audit.jsonl \
RUN_ROOT=/root/autodl-tmp/best_hybrid_vlm_v9_run_YYYYMMDD_HHMMSS \
PYTHON=/root/miniconda3/bin/python \
bash v9_best_hybrid_vlm/run_v9_pipeline.sh
```

Stages:

1. Train the reconstruction branch and pseudo-anomaly head with Train normals.
2. Build memory banks and calibrate branches using Train normals.
3. Produce `submission_v9_dense.zip`, the DINO control submission.
4. Apply bounded local Qwen3-VL evidence and produce
   `submission_v9_hybrid.zip`.
5. Validate both archives: 750 CSV rows, 3,750 masks, grayscale 448x448 PNGs.

Outputs are stored only beneath `RUN_ROOT`:

```text
checkpoints/
train.log
dense_infer.log
dense_validate.log
hybrid_fuse.log
hybrid_validate.log
submission_v9_dense.zip
submission_v9_hybrid.zip
```

## Submission Choice

Submit both new archives separately. Use the dense-control result as the
ablation check. Adopt the hybrid result only if it improves pixel AP/F1max
without reducing image-level ranking; it is intentionally an augmentation of
the established Hybrid v2 location model, not a replacement for it.
