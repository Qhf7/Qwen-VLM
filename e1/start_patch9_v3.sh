#!/usr/bin/env bash
# Patch9 v3 training launcher. Mirrors start_segmentation_patch9.sh but uses the
# v3 module (fixes root-cause-3: anomaly/non-empty exposure raised back to 8/step).
# Launch:
#   bash e1/start_patch9_v3.sh validate        # forward/data/gradient check only
#   bash e1/start_patch9_v3.sh train           # 12000-step training (foreground)
#   nohup bash e1/start_patch9_v3.sh train &   # background (recommended)
set -Eeuo pipefail
cd /root/autodl-tmp/vlm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
out=${OUT_DIR:-artifacts/segmentation_qwen3_patch9_mixed_v3_12000}
mode=${1:-train}
if [ "$mode" = "validate" ]; then
  mkdir -p "$out"
  test ! -e "$out/forward_validation.json" || exit 74
  exec /root/autodl-tmp/vlm/.venv/bin/python -u -m e1.train_segmentation_patch9_v3 \
    --out "$out" --validate-only
fi
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || exit 73
test ! -e "$out/training_history.csv" || exit 74
exec >"$out/run.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$out/exit_code.txt"' EXIT
exec /root/autodl-tmp/vlm/.venv/bin/python -u -m e1.train_segmentation_patch9_v3 \
  --out "$out" --steps "${STEPS:-12000}"
