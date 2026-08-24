#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WEIGHT_DIR="${WEIGHT_DIR:?set WEIGHT_DIR}"
ROOT="${ROOT:-/root/autodl-tmp/qwen_dinov2_multiscale_v3}"
CKPT="$ROOT/ckpt"
DENSE="$ROOT/dense"
mkdir -p "$CKPT"

python3 train_v3.py --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --out "$CKPT" --size 1036 --epochs 3 --batch-images 2
python3 proxy_v3.py --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --checkpoint "$CKPT/decoder.pt" --out "$CKPT/proxy_metrics.json" --size 1036
python3 infer_v3.py --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --checkpoint "$CKPT/decoder.pt" --output "$DENSE" --zip-path "$ROOT/submission_qwen_dinov2_multiscale_v3_dense.zip" --size 1036

AUDIT="${AUDIT:-/root/autodl-tmp/qwen3vl_local/submission_qwen3vl_crop_v2/qwen_crop_outputs.jsonl}"
HYBRID="$ROOT/hybrid"
if [[ -f "$AUDIT" ]]; then
  python3 qwen_dinov2_hybrid_v2_fuse.py \
    --dino-root "$DENSE" \
    --audit "$AUDIT" \
    --data-root "$DATA_ROOT" \
    --split Test_A \
    --output "$HYBRID" \
    --zip "$ROOT/submission_qwen_dinov2_multiscale_v3_hybrid.zip" \
    --require-full-audit \
    --proposal-area-target 0.014 \
    --proposal-area-hardness 0.80
else
  echo "Qwen audit not found; dense output remains available" >&2
fi
