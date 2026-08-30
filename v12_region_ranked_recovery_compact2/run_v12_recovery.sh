#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
WEIGHT_DIR=${WEIGHT_DIR:?set WEIGHT_DIR}
CKPT=${CKPT:?set CKPT}
REGION_RANKER=${REGION_RANKER:?set REGION_RANKER}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
REFERENCE_CACHE="${REFERENCE_CACHE:-$RUN_ROOT/reference_cache_v12_recovery.pt}"

mkdir -p "$RUN_ROOT"

if [[ ! -s "$REFERENCE_CACHE" ]]; then
  echo "V12_RECOVERY_STAGE lowmem_reference"
  "$PYTHON" "$HERE/lowmem_reference_v12.py" \
    --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --ckpt "$CKPT" \
    --output "$REFERENCE_CACHE" --fine-size 1036 --coarse-size 518 \
    --batch 2 --budget 2048 --global-budget 8192 --calib-limit 300
else
  echo "V12_RECOVERY reuse reference $REFERENCE_CACHE"
fi

echo "V12_RECOVERY_STAGE region_ranked_inference"
"$PYTHON" "$HERE/infer_stream_memory.py" \
  --data-root "$DATA_ROOT" --test-split Test_A --arch dinov2_vitb14 \
  --weight-dir "$WEIGHT_DIR" --ckpt "$CKPT" \
  --region-ranker "$REGION_RANKER" --reference-cache "$REFERENCE_CACHE" \
  --fine-size 1036 --coarse-size 518 --batch 4 --calib-limit 500 \
  --highpass 0.0 --gamma 1.0 --gate-strength 0.0 \
  --area-target 0.02 --area-hardness 0.0 --fine-weight 0.65 \
  --background-keep 0.08 --boundary-keep 0.55 \
  --region-reject-scale 0.84 --region-relative-margin 0.10 \
  --region-max-area 0.018 \
  --output "$RUN_ROOT/submission_v12_region_ranked_recovery" \
  --zip "$RUN_ROOT/submission_v12_region_ranked_recovery.zip" \
  2>&1 | tee "$RUN_ROOT/inference.log"

echo "V12_RECOVERY_STAGE validate"
"$PYTHON" "$HERE/validate_submission_v9.py" \
  "$RUN_ROOT/submission_v12_region_ranked_recovery.zip" \
  2>&1 | tee "$RUN_ROOT/validate.log"
"$PYTHON" "$HERE/mask_stats_v10.py" \
  "$RUN_ROOT/submission_v12_region_ranked_recovery.zip" \
  2>&1 | tee "$RUN_ROOT/mask_stats.json"
echo "V12_RECOVERY_COMPLETE RUN_ROOT=$RUN_ROOT"
