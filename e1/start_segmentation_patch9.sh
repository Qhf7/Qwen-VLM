#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/vlm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
out=${OUT_DIR:-artifacts/segmentation_qwen3_patch9_12000}
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || exit 73
if [[ -n "${RESUME:-}" ]]; then
  test -f "$RESUME" || { echo "resume checkpoint not found: $RESUME" >&2; exit 75; }
else
  test ! -e "$out/training_history.csv" || exit 74
fi
exec >"$out/run.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$out/exit_code.txt"' EXIT
args=(--out "$out" --steps "${STEPS:-12000}")
[[ -n "${RESUME:-}" ]] && args+=(--resume "$RESUME")
exec /root/autodl-tmp/vlm/.venv/bin/python -u -m e1.train_segmentation_patch9 "${args[@]}"
