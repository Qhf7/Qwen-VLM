#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WEIGHT_DIR="${WEIGHT_DIR:?set WEIGHT_DIR}"
CLIP_WEIGHT="${CLIP_WEIGHT:?set CLIP_WEIGHT}"
V4_CHECKPOINT="${V4_CHECKPOINT:?set V4_CHECKPOINT}"
ROOT="${ROOT:?set a fresh ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$CODE/.." && pwd)"

mkdir -p "$ROOT"

if [[ -s "$ROOT/submission_vlm_primary_v5.zip" ]] \
  && grep -q '^VALID:' "$ROOT/validate.log" 2>/dev/null; then
  echo "PIPELINE_ALREADY_COMPLETE ROOT=$ROOT"
  exit 0
fi

FOLD="$ROOT/fold0"
mkdir -p "$FOLD"
"$PYTHON" "$CODE/train_v5_resume.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
  --v4-checkpoint "$V4_CHECKPOINT" --out "$FOLD/checkpoints" \
  --heldout-fold 0 --epochs 2 --checkpoint-every 50 \
  2>&1 | tee -a "$FOLD/train.log"

for epoch in 0 1 2; do
  checkpoint="$FOLD/checkpoints/vlm_primary_v5_epoch${epoch}.pt"
  [[ "$epoch" != 0 ]] || checkpoint="$FOLD/checkpoints/vlm_primary_v5_initial.pt"
  proxy="$FOLD/proxy_epoch${epoch}.json"
  if [[ ! -s "$proxy" ]]; then
    "$PYTHON" "$CODE/proxy_v5.py" \
      --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
      --checkpoint "$checkpoint" --out "$proxy" --groups-per-category 5 \
      2>&1 | tee "$FOLD/proxy_epoch${epoch}.log"
  fi
done

"$PYTHON" "$CODE/select_v5_checkpoint.py" \
  --proxy-dir "$FOLD" --out "$FOLD/selection.json" \
  2>&1 | tee "$FOLD/selection.log"
BEST_EPOCH="$($PYTHON - "$FOLD/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_epoch"])
PY
)"
echo "BEST_EPOCH=$BEST_EPOCH"

FINAL="$ROOT/final"
if [[ "$BEST_EPOCH" == 0 ]]; then
  mkdir -p "$FINAL/checkpoints"
  "$PYTHON" - "$FOLD/checkpoints/vlm_primary_v5_initial.pt" "$FINAL/checkpoints/vlm_primary_v5.pt" <<'PY'
import sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
source["heldout_fold"] = -1
torch.save(source, sys.argv[2])
PY
else
  mkdir -p "$FINAL"
  "$PYTHON" "$CODE/train_v5_resume.py" \
    --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
    --v4-checkpoint "$V4_CHECKPOINT" --out "$FINAL/checkpoints" \
    --heldout-fold -1 --epochs "$BEST_EPOCH" --checkpoint-every 50 \
    2>&1 | tee -a "$FINAL/train.log"
fi

if [[ -e "$ROOT/submission_vlm_primary_v5" || -e "$ROOT/submission_vlm_primary_v5.zip" ]]; then
  interrupted="$ROOT/interrupted_inference_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$interrupted"
  [[ ! -e "$ROOT/submission_vlm_primary_v5" ]] || mv "$ROOT/submission_vlm_primary_v5" "$interrupted/"
  [[ ! -e "$ROOT/submission_vlm_primary_v5.zip" ]] || mv "$ROOT/submission_vlm_primary_v5.zip" "$interrupted/"
fi

"$PYTHON" "$CODE/infer_v5.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
  --checkpoint "$FINAL/checkpoints/vlm_primary_v5.pt" \
  --output "$ROOT/submission_vlm_primary_v5" \
  --zip-path "$ROOT/submission_vlm_primary_v5.zip" \
  2>&1 | tee "$ROOT/infer.log"

"$PYTHON" "$CODE/validate_v5_submission.py" \
  --data-root "$DATA_ROOT" --zip-path "$ROOT/submission_vlm_primary_v5.zip" \
  2>&1 | tee "$ROOT/validate.log"

echo "PIPELINE_COMPLETE ROOT=$ROOT BEST_EPOCH=$BEST_EPOCH"
