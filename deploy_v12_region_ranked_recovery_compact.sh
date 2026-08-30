#!/usr/bin/env bash
set -euo pipefail
LOCAL_ROOT=$(cd "$(dirname "$0")" && pwd)
HOST=${HOST:-root@connect.westc.seetacloud.com}; PORT=${PORT:-21091}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
REMOTE_CODE=/root/autodl-tmp/v12_region_ranked_recovery_compact_code_${STAMP}
REMOTE_RUN=/root/autodl-tmp/v12_region_ranked_recovery_compact_run_${STAMP}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/Dataset}
WEIGHT_DIR=${WEIGHT_DIR:-/root/autodl-tmp/v4_old_20260814_024248/weights}
CKPT=${CKPT:-/root/autodl-tmp/v9_best_hybrid_vlm_run_20260825_122246/checkpoints}
REGION_RANKER=${REGION_RANKER:-/root/autodl-tmp/v12_region_ranked_run_20260828_123200/output/region_ranker_v12.pt}
ARCHIVE=/tmp/v12_region_ranked_recovery_compact_${STAMP}.tar.gz
tar --exclude='__pycache__' --exclude='.DS_Store' -czf "$ARCHIVE" -C "$LOCAL_ROOT" v12_region_ranked_recovery_compact
scp -P "$PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "$ARCHIVE" "$HOST:/root/autodl-tmp/"
REMOTE_ARCHIVE=/root/autodl-tmp/$(basename "$ARCHIVE")
ssh -p "$PORT" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "$HOST" bash -s -- \
  "$REMOTE_CODE" "$REMOTE_RUN" "$REMOTE_ARCHIVE" "$DATA_ROOT" "$WEIGHT_DIR" "$CKPT" "$REGION_RANKER" <<'REMOTE'
set -euo pipefail
CODE=$1; RUN=$2; ARCHIVE=$3; DATA=$4; WEIGHTS=$5; CKPT=$6; RANKER=$7
[[ ! -e "$CODE" && ! -e "$RUN" ]] || { echo "compact path exists" >&2; exit 2; }
for p in "$DATA/Train" "$DATA/Test_A" "$WEIGHTS/dinov2_vitb14_pretrain.pth" "$CKPT/recon.pt" "$CKPT/head.pt" "$RANKER"; do [[ -e "$p" ]] || { echo "missing $p" >&2; exit 3; }; done
mkdir -p "$CODE" "$RUN"; tar -xzf "$ARCHIVE" -C "$CODE" --strip-components=1; cd "$CODE"
/root/miniconda3/bin/python -m py_compile ./*.py
PYTHONPATH="$CODE" /root/miniconda3/bin/python test_v12.py
PYTHONPATH="$CODE" /root/miniconda3/bin/python test_v10.py
bash -n run_v12_recovery.sh
cat > "$RUN/start_compact.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export DATA_ROOT=$(printf '%q' "$DATA")
export WEIGHT_DIR=$(printf '%q' "$WEIGHTS")
export CKPT=$(printf '%q' "$CKPT")
export REGION_RANKER=$(printf '%q' "$RANKER")
export RUN_ROOT=$(printf '%q' "$RUN/output")
export PYTHON=/root/miniconda3/bin/python
cd $(printf '%q' "$CODE")
bash run_v12_recovery.sh 2>&1 | tee $(printf '%q' "$RUN/pipeline.log")
EOF
chmod +x "$RUN/start_compact.sh"
SCREEN=v12_region_ranked_recovery_compact_$(date +%Y%m%d_%H%M%S)
screen -dmS "$SCREEN" bash "$RUN/start_compact.sh"
sleep 3; echo "SCREEN=$SCREEN"; echo "REMOTE_CODE=$CODE"; echo "REMOTE_RUN=$RUN"; tail -20 "$RUN/pipeline.log" 2>/dev/null || true
REMOTE
echo "STAMP=$STAMP"; echo "REMOTE_CODE=$REMOTE_CODE"; echo "REMOTE_RUN=$REMOTE_RUN"
