# v4 Model-Side Segmentation Pipeline

This pipeline improves pixel segmentation at the model/source level, not by
editing an existing submission mask.

## What It Does

- Extracts high-resolution DINOv2 patch features.
- Builds per-category/per-view memory banks from official Train normal images.
- Generates pseudo anomalies from official Train normal images only.
- Trains a pixel-level pseudo anomaly head.
- Uses Train-only proxy validation to tune fusion and area parameters.
- Generates a fresh submission zip from model inference.

## Key Difference From Object-Aware Post-Processing

`build_object_aware_segmentation_*.py` reads an existing submission and changes
its masks.  The v4 pipeline does not depend on old masks.  It uses the original
Train/Test images, DINOv2 features, memory banks, trained heads, and calibrated
branch fusion to produce new masks.

## Server Commands

Run from `/root/autodl-tmp` after the dataset, repo, and weights are present.

```bash
cd /root/csig_competition
git pull origin main

python3 v4/train_branches.py \
  --data-root /root/autodl-tmp/Dataset \
  --arch dinov2_vitb14 \
  --size 1036 \
  --weight-dir /root/csig_competition/weights \
  --out /root/autodl-tmp/v4_ckpt \
  --epochs-recon 8 \
  --epochs-head 6 \
  --batch-images 8

python3 v4/tune.py \
  --data-root /root/autodl-tmp/Dataset \
  --arch dinov2_vitb14 \
  --weight-dir /root/csig_competition/weights \
  --fine-size 1036 \
  --batch 12 \
  --budget 4096 \
  --background-keep 0.08 \
  --boundary-keep 0.55 \
  --out /root/autodl-tmp/v4_tune_report.json

python3 v4/infer.py \
  --data-root /root/autodl-tmp/Dataset \
  --test-split Test_A \
  --arch dinov2_vitb14 \
  --weight-dir /root/csig_competition/weights \
  --ckpt /root/autodl-tmp/v4_ckpt \
  --fine-size 1036 \
  --coarse-size 518 \
  --batch 12 \
  --budget 4096 \
  --global-budget 16384 \
  --output /root/autodl-tmp/submission_v4_model \
  --zip /root/autodl-tmp/submission_v4_model.zip \
  --fuse-mode weighted \
  --highpass 0.0 \
  --gamma 1.0 \
  --gate-strength 0.0 \
  --area-hardness 0.5 \
  --background-keep 0.08 \
  --boundary-keep 0.55

python3 validate_submission.py \
  --root /root/autodl-tmp/submission_v4_model \
  --zip /root/autodl-tmp/submission_v4_model.zip
```

Submit:

```text
/root/autodl-tmp/submission_v4_model.zip
```

## Notes

- If proxy tuning finds better parameters, use `v4_tune_report.json["BEST"]["config"]`
  to override the inference defaults.
- If the model still highlights full part silhouettes, reduce
  `--boundary-keep` to `0.45` and increase `--area-hardness` to `0.75`.
- If masks become too sparse, keep `--highpass 0.0`, `--gamma 1.0`, and raise
  `--background-keep` to `0.12`.
