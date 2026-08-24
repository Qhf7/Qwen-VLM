#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WEIGHT_DIR="${WEIGHT_DIR:?set WEIGHT_DIR}"
CLIP_WEIGHT="${CLIP_WEIGHT:?set CLIP_WEIGHT}"
V3_CHECKPOINT="${V3_CHECKPOINT:?set V3_CHECKPOINT}"
ROOT="${ROOT:-/root/autodl-tmp/qwen_dinov2_vlm_finetune_v4_final_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT"
"$PYTHON" - <<PY
import clip
model, _ = clip.load(r"$CLIP_WEIGHT", device="cpu", jit=False)
assert hasattr(model.visual, "conv1") and model.visual.proj is not None
print("CLIP smoke OK", type(model.visual).__name__, model.visual.output_dim)
PY

# Validate the exact final-training source (including any independent loss
# fix) under CUDA autocast before committing several hours to all categories.
"$PYTHON" "$CODE/train_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --v3-checkpoint "$V3_CHECKPOINT" \
  --out "$ROOT/smoke_ckpt" --heldout-fold -1 --epochs 1 --limit 1 --log-every 1 \
  2>&1 | tee "$ROOT/smoke_final.log"

# heldout-fold=-1 means no category is excluded because category_fold is 0..4.
"$PYTHON" "$CODE/train_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --v3-checkpoint "$V3_CHECKPOINT" \
  --out "$ROOT/final_ckpt" --heldout-fold -1 --epochs 4 \
  2>&1 | tee "$ROOT/train_final.log"

"$PYTHON" "$CODE/infer_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --checkpoint "$ROOT/final_ckpt/vlm_v4.pt" \
  --output "$ROOT/submission_vlm_finetune_v4" \
  --zip-path "$ROOT/submission_vlm_finetune_v4.zip" \
  2>&1 | tee "$ROOT/infer_final.log"

"$PYTHON" "$CODE/validate_v4_submission.py" \
  --data-root "$DATA_ROOT" --zip-path "$ROOT/submission_vlm_finetune_v4.zip" \
  2>&1 | tee "$ROOT/validate_final.log"

echo "ROOT=$ROOT"
