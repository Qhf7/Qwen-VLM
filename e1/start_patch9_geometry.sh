#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vlm
mode=${1:?Usage: start_patch9_geometry.sh validate|train output_dir}
out=${2:?Specify a fresh output directory}
case "$mode" in
  validate) extra=(--validate-only) ;;
  train) extra=() ;;
  *) exit 2 ;;
esac
test ! -e "$out/training_history.csv"
mkdir -p "$out"
exec 8>artifacts/patch9_geometry_gpu.lock
flock -n 8 || exit 73
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u -m e1.train_segmentation_patch9_v2_geometry \
  --train artifacts/splits/fold1/train.parquet --dev artifacts/splits/fold1/dev.parquet \
  --root Dataset --gt-root Dataset --steps 12000 --out "$out" "${extra[@]}" \
  >"$out/run.log" 2>&1
