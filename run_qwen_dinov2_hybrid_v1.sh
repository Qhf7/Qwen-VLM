#!/usr/bin/env bash
# Runs only after the independently started Train-only pixel-head job writes
# ckpt/head.pt.  Every output uses a new versioned path; old submissions are
# deliberately never removed or overwritten.
set -euo pipefail

ROOT=/root/autodl-tmp/qwen_dinov2_hybrid_v1
PY=/root/autodl-tmp/qwen3vl_local/venv/bin/python
DATA=/root/autodl-tmp/Dataset
AUDIT=/root/autodl-tmp/qwen3vl_local/submission_qwen3vl_crop_v2/qwen_crop_outputs.jsonl

while [ ! -f "$ROOT/ckpt/head.pt" ]; do
  sleep 60
done

cd "$ROOT"

# Hyperparameters are selected solely on official Train normal images and
# procedurally derived pseudo-defects. The proxy does not open Test_A.
$PY tune.py \
  --data-root "$DATA" \
  --arch dinov2_vitb14 \
  --weight-dir "$ROOT/weights" \
  --fine-size 1036 \
  --batch 8 \
  --budget 2048 \
  --background-keep 0.08 \
  --boundary-keep 0.50 \
  --out "$ROOT/tune_report_train_only.json"

# The proxy-selected configuration is read explicitly before fresh model
# inference. No existing submission PNG is an input to this command.
eval "$($PY - <<'PY'
import json
r = json.load(open('/root/autodl-tmp/qwen_dinov2_hybrid_v1/tune_report_train_only.json'))
c = r['BEST']['config']
for k in ('highpass','gamma','post_blur','gate','area_target','area_hardness'):
    print(f'{k.upper()}={c[k]!r}')
PY
)"

DINO_OUT=/root/autodl-tmp/qwen_dinov2_hybrid_v1_dino_dense
$PY infer.py \
  --data-root "$DATA" \
  --test-split Test_A \
  --arch dinov2_vitb14 \
  --weight-dir "$ROOT/weights" \
  --ckpt "$ROOT/ckpt" \
  --fine-size 1036 \
  --coarse-size 518 \
  --batch 8 \
  --budget 4096 \
  --global-budget 16384 \
  --fuse-mode gated \
  --highpass "$HIGHPASS" \
  --gamma "$GAMMA" \
  --post-blur "$POST_BLUR" \
  --gate-strength "$GATE" \
  --area-target "$AREA_TARGET" \
  --area-hardness "$AREA_HARDNESS" \
  --background-keep 0.08 \
  --boundary-keep 0.50 \
  --output "$DINO_OUT" \
  --zip /root/autodl-tmp/submission_qwen_dinov2_dense_v1.zip

HYBRID_OUT=/root/autodl-tmp/qwen_dinov2_hybrid_v1_submission
$PY qwen_dinov2_hybrid_v1_fuse.py \
  --dino-root "$DINO_OUT" \
  --audit "$AUDIT" \
  --data-root "$DATA" \
  --split Test_A \
  --output "$HYBRID_OUT" \
  --zip /root/autodl-tmp/submission_qwen_dinov2_hybrid_v1.zip \
  --boost 0.35

$PY - <<'PY'
import csv
from pathlib import Path
from PIL import Image

root = Path('/root/autodl-tmp/qwen_dinov2_hybrid_v1_submission')
rows = list(csv.DictReader((root / 'submission.csv').open(encoding='utf-8')))
masks = sorted((root / 'predicted_masks').rglob('*_mask.png'))
bad = [str(p) for p in masks if Image.open(p).mode != 'L' or Image.open(p).size != (448, 448)]
assert len(rows) == 750, len(rows)
assert len(masks) == 3750, len(masks)
assert not bad, bad[:3]
print({'csv_rows': len(rows), 'masks': len(masks), 'bad_masks': len(bad)})
PY
