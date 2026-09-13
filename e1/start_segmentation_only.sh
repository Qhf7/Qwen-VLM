#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/vlm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
out=${OUT_DIR:-artifacts/segmentation_qwen3_decoder_only}
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || exit 73
test ! -e "$out/training_history.csv" || exit 74
exec >"$out/run.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$out/exit_code.txt"' EXIT
/root/autodl-tmp/vlm/.venv/bin/python -u -m e1.train_segmentation_only --out "$out" --steps "${STEPS:-1000}"
