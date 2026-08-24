#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WEIGHT_DIR="${WEIGHT_DIR:?set WEIGHT_DIR}"
CLIP_WEIGHT="${CLIP_WEIGHT:?set CLIP_WEIGHT to the generic OpenAI ViT-B-16 checkpoint}"
V3_CHECKPOINT="${V3_CHECKPOINT:?set V3_CHECKPOINT to decoder.pt}"
ROOT="${ROOT:-/root/autodl-tmp/qwen_dinov2_vlm_finetune_v4_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT"

"$PYTHON" - <<PY
import clip, torch
model, _ = clip.load(r"$CLIP_WEIGHT", device="cpu", jit=False)
assert hasattr(model.visual, "conv1") and model.visual.proj is not None
print("CLIP smoke OK", type(model.visual).__name__, model.visual.output_dim)
PY

echo "Running one-group end-to-end gradient smoke test..."
"$PYTHON" "$CODE/train_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --v3-checkpoint "$V3_CHECKPOINT" \
  --out "$ROOT/smoke_ckpt" --heldout-fold 0 --epochs 1 --limit 1 --log-every 1 \
  2>&1 | tee "$ROOT/smoke.log"

"$PYTHON" "$CODE/train_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --v3-checkpoint "$V3_CHECKPOINT" \
  --out "$ROOT/fold0_ckpt" --heldout-fold 0 --epochs 4 \
  2>&1 | tee "$ROOT/train_fold0.log"

"$PYTHON" "$CODE/proxy_v4.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" \
  --clip-weight "$CLIP_WEIGHT" --checkpoint "$ROOT/fold0_ckpt/vlm_v4.pt" \
  --out "$ROOT/fold0_ckpt/proxy.json" \
  2>&1 | tee "$ROOT/proxy_fold0.log"

echo "Fold-0 proxy complete. Review held-out Pixel AP/F1max before final all-category training."
echo "ROOT=$ROOT"
