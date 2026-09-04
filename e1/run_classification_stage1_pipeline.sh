#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/vlm
source .venv/bin/activate

mkdir -p artifacts/b0_classification_only/qwen25
mkdir -p artifacts/b0_classification_only/qwen3
mkdir -p artifacts/b0_classification_only/analysis
mkdir -p artifacts/e1_classification_fold1

python -m b0.run_dev720 \
  --model-id models/Qwen2.5-VL-3B-Instruct \
  --subset artifacts/splits/fold1/dev.parquet \
  --root Dataset448 \
  --preprocessed-cache-manifest artifacts/b0plus/dev720_cache_manifest.csv \
  --prompt-id P4_CLASSIFICATION_ONLY \
  --anomaly-token yes \
  --generation-audit-count 36 \
  --forward-generate-atol 2e-5 \
  --out-dir artifacts/b0_classification_only/qwen25 \
  2>&1 | tee artifacts/b0_classification_only/qwen25/run.log

python -m b0.run_dev720 \
  --model-id models/Qwen3-VL-2B-Instruct \
  --subset artifacts/splits/fold1/dev.parquet \
  --root Dataset448 \
  --preprocessed-cache-manifest artifacts/b0plus/dev720_cache_manifest.csv \
  --prompt-id P4_CLASSIFICATION_ONLY \
  --anomaly-token yes \
  --generation-audit-count 36 \
  --forward-generate-atol 2e-5 \
  --out-dir artifacts/b0_classification_only/qwen3 \
  2>&1 | tee artifacts/b0_classification_only/qwen3/run.log

python -m b0.analyze_cross_model \
  --input \
    artifacts/b0_classification_only/qwen25/P4_CLASSIFICATION_ONLY_views.csv \
    artifacts/b0_classification_only/qwen3/P4_CLASSIFICATION_ONLY_views.csv \
  --names qwen25_vl_3b qwen3_vl_2b \
  --score-column p_yes \
  --out-dir artifacts/b0_classification_only/analysis

python -m e1.train_classification_fold1 \
  --model-id models/Qwen2.5-VL-3B-Instruct \
  --out-dir artifacts/e1_classification_fold1 \
  2>&1 | tee artifacts/e1_classification_fold1/run.log

python -m e1.analyze_classification_convergence \
  --run-dir artifacts/e1_classification_fold1 \
  2>&1 | tee artifacts/e1_classification_fold1/convergence.log

