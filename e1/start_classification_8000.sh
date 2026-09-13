#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vlm
source .venv/bin/activate
exec 9>artifacts/classification_verified.lock
flock -n 9 || exit 1
run=artifacts/e1_real_classification_qwen3_full_8000
pool=artifacts/e1_real_classification_8000
cache=Train448_verified_20260905
manifest=artifacts/e1_real_classification_verified/train448_cache_manifest.csv
test ! -e "$run/training_history.csv"
mkdir -p "$run" "$pool"
python -m unittest e1.test_single_image_protocol -v
python -m e1.make_real_classification_schedule --cache-root "$cache" --cache-manifest "$manifest" --out-dir "$pool" --steps 8000
python -m e1.validate_real_data --cache-root "$cache" --manifest "$manifest" --pool-dir "$pool"
python -m e1.validate_classification_forward --pool "$pool/real_classification_pool.parquet" --cache-root "$cache" --out "$pool/forward_validation.json"
python -m e1.train_classification_fold1 --model-id models/Qwen3-VL-2B-Instruct --pool "$pool/real_classification_pool.parquet" --schedule "$pool/real_classification_schedule.parquet" --cache-root "$cache" --cache-manifest "$manifest" --out-dir "$run" --real-only --feature-layers 5,11,17,23 --max-steps 8000 --eval-steps 250,500,750,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000
