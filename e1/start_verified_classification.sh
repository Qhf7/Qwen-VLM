#!/usr/bin/env bash
# Audit a versioned cache, then train the single-image classifier from step 1.
set -euo pipefail
cd /root/autodl-tmp/vlm
source .venv/bin/activate
expected_sha=${1:?expected archive SHA256 required}
exec 9>artifacts/classification_verified.lock
flock -n 9 || { echo 'Another verified launcher is active'; exit 1; }
if pgrep -f '[p]ython.*-m e1.train' >/dev/null; then
    echo 'Training already active; leaving it untouched'; exit 1
fi
python - "$expected_sha" <<'PY'
import hashlib, sys
from pathlib import Path
p=Path('train448_verified.tar')
with p.open('rb') as f:
    digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==sys.argv[1], 'Cache archive checksum mismatch'
PY
cache_root=Train448_verified_20260905
pool_dir=artifacts/e1_real_classification_verified
run_dir=artifacts/e1_real_classification_qwen3_verified
if test -e "$run_dir/training_history.csv"; then
    echo 'Existing run found; refusing to overwrite'; exit 1
fi
mkdir -p "$cache_root" "$pool_dir" "$run_dir"
tar -xf train448_verified.tar -C "$cache_root"
python -m unittest e1.test_single_image_protocol -v
python -m e1.make_real_classification_schedule \
  --cache-root "$cache_root" --cache-manifest "$pool_dir/train448_cache_manifest.csv" --out-dir "$pool_dir"
python -m e1.validate_real_data \
  --cache-root "$cache_root" --manifest "$pool_dir/train448_cache_manifest.csv" --pool-dir "$pool_dir"
python -m e1.validate_classification_forward \
  --pool "$pool_dir/real_classification_pool.parquet" --cache-root "$cache_root" \
  --out "$pool_dir/forward_validation.json"
if pgrep -f '[p]ython.*-m e1.train' >/dev/null; then
    echo 'Training became active; leaving it untouched'; exit 1
fi
python -m e1.train_classification_fold1 \
  --model-id models/Qwen3-VL-2B-Instruct \
  --pool "$pool_dir/real_classification_pool.parquet" \
  --schedule "$pool_dir/real_classification_schedule.parquet" \
  --cache-root "$cache_root" --cache-manifest "$pool_dir/train448_cache_manifest.csv" \
  --out-dir "$run_dir" --real-only --feature-layers 5,11,17,23 \
  2>&1 | tee "$run_dir/run.log"
