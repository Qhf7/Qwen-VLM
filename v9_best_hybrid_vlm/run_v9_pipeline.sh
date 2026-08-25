#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the competition Dataset directory}
WEIGHT_DIR=${WEIGHT_DIR:?set WEIGHT_DIR to the generic DINOv2 weights directory}
VLM_AUDIT=${VLM_AUDIT:?set VLM_AUDIT to the Qwen3-VL crop audit JSONL}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new, non-existing v9 run directory}
CLIP_WEIGHT=${CLIP_WEIGHT:-}

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/checkpoints"

echo "V9_STAGE train_branches"
"$PYTHON" "$HERE/train_branches.py" \
  --data-root "$DATA_ROOT" --arch dinov2_vitb14 --size 1036 \
  --weight-dir "$WEIGHT_DIR" --out "$RUN_ROOT/checkpoints" \
  --epochs-recon 8 --epochs-head 6 --batch-images 8 \
  2>&1 | tee "$RUN_ROOT/train.log"

echo "V9_STAGE dense_inference"
DENSE_ARGS=(
  --data-root "$DATA_ROOT" --test-split Test_A --arch dinov2_vitb14
  --weight-dir "$WEIGHT_DIR" --ckpt "$RUN_ROOT/checkpoints"
  --fine-size 1036 --coarse-size 518 --batch 10
  --calib-limit 500 --budget 4096 --global-budget 16384
  --highpass 0.0 --gamma 1.0 --gate-strength 0.0
  --area-target 0.02 --area-hardness 0.5 --fine-weight 0.65
  --background-keep 0.08 --boundary-keep 0.55
  --output "$RUN_ROOT/submission_v9_dense"
  --zip "$RUN_ROOT/submission_v9_dense.zip"
)
if [[ -n "$CLIP_WEIGHT" ]]; then
  DENSE_ARGS+=(--clip-weight "$CLIP_WEIGHT")
fi
"$PYTHON" "$HERE/infer_stream_memory.py" "${DENSE_ARGS[@]}" \
  2>&1 | tee "$RUN_ROOT/dense_infer.log"
"$PYTHON" "$HERE/validate_submission_v9.py" "$RUN_ROOT/submission_v9_dense.zip" \
  2>&1 | tee "$RUN_ROOT/dense_validate.log"

echo "V9_STAGE vlm_guided_fusion"
"$PYTHON" "$HERE/vlm_guided_fuse_v9.py" \
  --dino-root "$RUN_ROOT/submission_v9_dense" \
  --audit "$VLM_AUDIT" --data-root "$DATA_ROOT" --split Test_A \
  --output "$RUN_ROOT/submission_v9_hybrid" \
  --zip "$RUN_ROOT/submission_v9_hybrid.zip" --require-full-audit \
  2>&1 | tee "$RUN_ROOT/hybrid_fuse.log"
"$PYTHON" "$HERE/validate_submission_v9.py" "$RUN_ROOT/submission_v9_hybrid.zip" \
  2>&1 | tee "$RUN_ROOT/hybrid_validate.log"

echo "V9_PIPELINE_COMPLETE RUN_ROOT=$RUN_ROOT"
