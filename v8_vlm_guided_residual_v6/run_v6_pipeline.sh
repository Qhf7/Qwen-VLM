#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
WEIGHT_DIR="${WEIGHT_DIR:?set WEIGHT_DIR}"
CLIP_WEIGHT="${CLIP_WEIGHT:?set CLIP_WEIGHT}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:?set INITIAL_CHECKPOINT}"
ROOT="${ROOT:?set a fresh ROOT}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$CODE/.." && pwd)"

mkdir -p "$ROOT/fold0/checkpoints"
if [[ -s "$ROOT/submission_vlm_guided_residual_v6.zip" ]] \
  && grep -q '^VALID:' "$ROOT/validate.log" 2>/dev/null; then
  echo "PIPELINE_ALREADY_COMPLETE ROOT=$ROOT"
  exit 0
fi

"$PYTHON" "$CODE/train_v6.py" \
  --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" --out "$ROOT/fold0/checkpoints" \
  --heldout-fold 0 --epochs 2 --checkpoint-every 25 --log-every 5 \
  2>&1 | tee -a "$ROOT/fold0/train.log"

for epoch in 0 1 2; do
  checkpoint="$ROOT/fold0/checkpoints/epoch${epoch}.pt"
  proxy="$ROOT/fold0/proxy_epoch${epoch}.json"
  if [[ ! -s "$proxy" ]]; then
    "$PYTHON" "$CODE/proxy_v6.py" \
      --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
      --checkpoint "$checkpoint" --out "$proxy" \
      --bank-cache "$ROOT/fold0/reference_epoch${epoch}.pt" --groups-per-category 5 \
      2>&1 | tee "$ROOT/fold0/proxy_epoch${epoch}.log"
  fi
done

"$PYTHON" "$CODE/select_v6_checkpoint.py" \
  --proxy-dir "$ROOT/fold0" --out "$ROOT/fold0/selection.json" \
  2>&1 | tee "$ROOT/fold0/selection.log"
BEST_EPOCH="$($PYTHON - "$ROOT/fold0/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["best_epoch"])
PY
)"
echo "BEST_EPOCH=$BEST_EPOCH"

mkdir -p "$ROOT/final/checkpoints"
if [[ "$BEST_EPOCH" == 0 ]]; then
  if [[ ! -s "$ROOT/final/checkpoints/vlm_guided_residual_v6.pt" ]]; then
    "$PYTHON" - "$ROOT/fold0/checkpoints/epoch0.pt" "$ROOT/final/checkpoints/vlm_guided_residual_v6.pt" <<'PY'
import sys, torch
x = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
x["heldout_fold"] = -1
torch.save(x, sys.argv[2])
PY
  fi
else
  "$PYTHON" "$CODE/train_v6.py" \
    --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
    --initial-checkpoint "$INITIAL_CHECKPOINT" --out "$ROOT/final/checkpoints" \
    --heldout-fold -1 --epochs "$BEST_EPOCH" --checkpoint-every 25 --log-every 5 \
    2>&1 | tee -a "$ROOT/final/train.log"
fi

if [[ ! -s "$ROOT/submission_vlm_guided_residual_v6.zip" ]]; then
  if [[ -e "$ROOT/submission_vlm_guided_residual_v6" || -e "$ROOT/submission_vlm_guided_residual_v6.zip" ]]; then
    interrupted="$ROOT/interrupted_inference_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$interrupted"
    [[ ! -e "$ROOT/submission_vlm_guided_residual_v6" ]] || mv "$ROOT/submission_vlm_guided_residual_v6" "$interrupted/"
    [[ ! -e "$ROOT/submission_vlm_guided_residual_v6.zip" ]] || mv "$ROOT/submission_vlm_guided_residual_v6.zip" "$interrupted/"
  fi
  "$PYTHON" "$CODE/infer_v6.py" \
    --data-root "$DATA_ROOT" --weight-dir "$WEIGHT_DIR" --clip-weight "$CLIP_WEIGHT" \
    --checkpoint "$ROOT/final/checkpoints/vlm_guided_residual_v6.pt" \
    --bank-cache "$ROOT/final/reference_banks.pt" \
    --output "$ROOT/submission_vlm_guided_residual_v6" \
    --zip-path "$ROOT/submission_vlm_guided_residual_v6.zip" \
    2>&1 | tee "$ROOT/infer.log"
fi

"$PYTHON" "$BASE/v7_vlm_primary_v5/validate_v5_submission.py" \
  --data-root "$DATA_ROOT" --zip-path "$ROOT/submission_vlm_guided_residual_v6.zip" \
  2>&1 | tee "$ROOT/validate.log"
echo "PIPELINE_COMPLETE ROOT=$ROOT BEST_EPOCH=$BEST_EPOCH"
