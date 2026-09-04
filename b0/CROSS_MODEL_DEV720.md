# Cross-model Dev720 protocol

This comparison keeps `Qwen2.5-VL-3B-Instruct` as the frozen B0+ accuracy
reference. `Qwen3-VL-2B-Instruct` is evaluated on exactly the same
`fold1/dev.parquet` views, prompt (`P0`), direct 448 preprocessing, and
single-view B0f scoring. No holdout data, Visual Adapter, or LoRA is used.

## Run the missing Qwen3 file

Run on the CUDA host that has the dataset and model cache. The output is
resumable and can be interrupted safely:

```bash
python -m b0.run_dev720 \
  --model-id models/Qwen3-VL-2B-Instruct \
  --subset artifacts/splits/fold1/dev.parquet \
  --root Dataset448 \
  --preprocessed-cache-manifest artifacts/b0plus/dev720_cache_manifest.csv \
  --out-dir artifacts/b0_cross/qwen3_vl_2b_dev720
```

The runner must finish with 3,600 unique `image_path` rows and 720 groups.
The existing `artifacts/b0_cross/qwen3_vl_2b_dev720_views.csv` is an
incomplete prior run and is intentionally rejected by the analysis gate.

## Analyze both models

```bash
python -m b0.analyze_cross_model \
  --input artifacts/b0plus_dev720/P0_views.csv \
          artifacts/b0_cross/qwen3_vl_2b_dev720/P0_views.csv \
  --names qwen25_vl_3b qwen3_vl_2b \
  --out-dir artifacts/b0_cross_dev720 \
  --bootstrap-repeats 2000
```

The script validates the 18-class/720-group/3,600-view shape, five views per
group, constant group labels, unique images, and exact manifest equality before
writing any comparison report. It writes max/top-2-mean/mean metrics,
stratified group bootstrap intervals, paired calibration diagnostics, and a
human-readable `CROSS_MODEL_REPORT.md`. Calibration is a diagnostic split only;
it cannot replace the frozen max primary result.
