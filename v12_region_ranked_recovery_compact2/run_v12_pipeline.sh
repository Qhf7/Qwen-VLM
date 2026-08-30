#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the official competition Dataset directory}
WEIGHT_DIR=${WEIGHT_DIR:?set WEIGHT_DIR to the generic DINOv2 weights directory}
CKPT=${CKPT:?set CKPT to the completed v9/v10 branch checkpoint directory}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new, non-existing v12 run directory}
CLIP_WEIGHT=${CLIP_WEIGHT:-}
VLM_AUDIT=${VLM_AUDIT:-}

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing v12 RUN_ROOT: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

echo "V12_STAGE train_region_ranker"
"$PYTHON" "$HERE/train_region_ranker_v12.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --ckpt "$CKPT" \
  --output "$RUN_ROOT/region_ranker_v12.pt" \
  --reference-cache "$RUN_ROOT/reference_cache_v12.pt" \
  --fine-size 1036 --coarse-size 518 --batch 10 --batch-groups 2 \
  --calib-limit 300 --rank-limit 300 --normal-limit 150 --epochs 80 \
  2>&1 | tee "$RUN_ROOT/region_ranker_train.log"

echo "V12_STAGE region_ranked_inference"
INFER_ARGS=(
  --data-root "$DATA_ROOT" --test-split Test_A --arch dinov2_vitb14
  --weight-dir "$WEIGHT_DIR" --ckpt "$CKPT"
  --region-ranker "$RUN_ROOT/region_ranker_v12.pt"
  --reference-cache "$RUN_ROOT/reference_cache_v12.pt"
  --fine-size 1036 --coarse-size 518 --batch 10
  --calib-limit 500 --budget 4096 --global-budget 16384
  --highpass 0.0 --gamma 1.0 --gate-strength 0.0
  --area-target 0.02 --area-hardness 0.0 --fine-weight 0.65
  --background-keep 0.08 --boundary-keep 0.55
  --output "$RUN_ROOT/submission_v12_region_ranked"
  --zip "$RUN_ROOT/submission_v12_region_ranked.zip"
)
if [[ -n "$CLIP_WEIGHT" ]]; then
  INFER_ARGS+=(--clip-weight "$CLIP_WEIGHT")
fi
"$PYTHON" "$HERE/infer_stream_memory.py" "${INFER_ARGS[@]}" \
  2>&1 | tee "$RUN_ROOT/region_ranked_infer.log"

echo "V12_STAGE validate_and_stats"
"$PYTHON" "$HERE/validate_submission_v9.py" \
  "$RUN_ROOT/submission_v12_region_ranked.zip" \
  2>&1 | tee "$RUN_ROOT/validate.log"
"$PYTHON" "$HERE/mask_stats_v10.py" \
  "$RUN_ROOT/submission_v12_region_ranked.zip" \
  2>&1 | tee "$RUN_ROOT/mask_stats.json"

if [[ -n "$VLM_AUDIT" ]]; then
  echo "V12_STAGE optional_bounded_vlm_confirmation"
  "$PYTHON" "$HERE/vlm_guided_fuse_v10.py" \
    --dino-root "$RUN_ROOT/submission_v12_region_ranked" \
    --audit "$VLM_AUDIT" --data-root "$DATA_ROOT" --split Test_A \
    --output "$RUN_ROOT/submission_v12_region_ranked_hybrid" \
    --zip "$RUN_ROOT/submission_v12_region_ranked_hybrid.zip" \
    --support-floor 0.04 --support-high 0.25 --max-residual 0.06 \
    --image-residual 0.012 --require-full-audit \
    2>&1 | tee "$RUN_ROOT/vlm_confirmation.log"
  "$PYTHON" "$HERE/validate_submission_v9.py" \
    "$RUN_ROOT/submission_v12_region_ranked_hybrid.zip" \
    2>&1 | tee "$RUN_ROOT/hybrid_validate.log"
fi

echo "V12_PIPELINE_COMPLETE RUN_ROOT=$RUN_ROOT"
