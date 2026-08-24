#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="root@connect.westc.seetacloud.com"
PORT="21091"
KEY="${HOME}/.ssh/id_ed25519"
STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_CODE="/root/autodl-tmp/vlm_guided_residual_v6_code_${STAMP}"
REMOTE_RUN="/root/autodl-tmp/vlm_guided_residual_v6_run_${STAMP}"
SCREEN_NAME="vlm_guided_residual_v6_${STAMP}"
STATE="/root/autodl-tmp/vlm_guided_residual_v6_state.txt"
ARCHIVE="/tmp/vlm_guided_residual_v6_${STAMP}.tar.gz"
trap 'rm -f "$ARCHIVE"' EXIT

tar -C "$LOCAL_ROOT" \
  --exclude='._*' --exclude='.DS_Store' --exclude='__pycache__' \
  -czf "$ARCHIVE" \
  v8_vlm_guided_residual_v6 v7_vlm_primary_v5 v6_vlm_zero_shot_v4 v5_multiscale_v3
scp -i "$KEY" -o IdentitiesOnly=yes -P "$PORT" "$ARCHIVE" "$HOST:/root/autodl-tmp/"
ssh -i "$KEY" -o IdentitiesOnly=yes -p "$PORT" "$HOST" bash -s -- \
  "$STAMP" "$REMOTE_CODE" "$REMOTE_RUN" "$SCREEN_NAME" "$STATE" <<'REMOTE'
set -euo pipefail
STAMP="$1"; REMOTE_CODE="$2"; REMOTE_RUN="$3"; SCREEN_NAME="$4"; STATE="$5"
ARCHIVE="/root/autodl-tmp/vlm_guided_residual_v6_${STAMP}.tar.gz"
DATA_ROOT=/root/autodl-tmp/Dataset
CLIP_WEIGHT=/root/autodl-tmp/weights/ViT-B-16.pt
DINO_WEIGHT="$(find /root/autodl-tmp -type f -name dinov2_vitb14_pretrain.pth | head -1)"
WEIGHT_DIR="${DINO_WEIGHT%/*}"
INITIAL_CHECKPOINT=/root/autodl-tmp/qwen_dinov2_vlm_finetune_v4_resume_run_20260822_125041/final_ckpt/vlm_v4.pt
for path in "$ARCHIVE" "$CLIP_WEIGHT" "$DINO_WEIGHT" "$INITIAL_CHECKPOINT"; do
  [[ -s "$path" ]] || { echo "missing: $path" >&2; exit 2; }
done
[[ -d "$DATA_ROOT/Train" && -d "$DATA_ROOT/Test_A" ]] || exit 3
[[ ! -e "$REMOTE_CODE" && ! -e "$REMOTE_RUN" ]] || exit 4
mkdir -p "$REMOTE_CODE" "$REMOTE_RUN"; tar -xzf "$ARCHIVE" -C "$REMOTE_CODE"
chmod +x "$REMOTE_CODE/v8_vlm_guided_residual_v6/"*.py "$REMOTE_CODE/v8_vlm_guided_residual_v6/"*.sh
/root/miniconda3/bin/python -m py_compile "$REMOTE_CODE/v8_vlm_guided_residual_v6/"*.py
/root/miniconda3/bin/python "$REMOTE_CODE/v8_vlm_guided_residual_v6/test_v6.py"
cat > "$STATE" <<STATE
STAMP=$STAMP
REMOTE_CODE=$REMOTE_CODE
REMOTE_RUN=$REMOTE_RUN
SCREEN_NAME=$SCREEN_NAME
DATA_ROOT=$DATA_ROOT
WEIGHT_DIR=$WEIGHT_DIR
CLIP_WEIGHT=$CLIP_WEIGHT
INITIAL_CHECKPOINT=$INITIAL_CHECKPOINT
STATE
screen -dmS "$SCREEN_NAME" bash -lc "
  set -euo pipefail
  cd '$REMOTE_CODE/v8_vlm_guided_residual_v6'
  DATA_ROOT='$DATA_ROOT' WEIGHT_DIR='$WEIGHT_DIR' CLIP_WEIGHT='$CLIP_WEIGHT' \
    INITIAL_CHECKPOINT='$INITIAL_CHECKPOINT' ROOT='$REMOTE_RUN' PYTHON=/root/miniconda3/bin/python \
    bash run_v6_pipeline.sh >> '$REMOTE_RUN/screen.log' 2>&1
"
echo "SCREEN=$SCREEN_NAME"; echo "REMOTE_CODE=$REMOTE_CODE"; echo "REMOTE_RUN=$REMOTE_RUN"
REMOTE
