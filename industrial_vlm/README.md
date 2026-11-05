# Industrial VLM: single-image Qwen3-VL QLoRA

This project builds a supervised industrial inspection pilot on a single RTX 4090D 24GB. The model receives one unmodified image and an instruction, and generates an anomaly decision and one box enclosing the annotated defects. Training uses the native Qwen3-VL visual path, frozen visual parameters, language-only NF4 weights and attention LoRA. There is no independent segmentation head in this version.

The initial experiment compares the original quantized base, area-coverage SFT, an equal-size random-sampling SFT, and decision-only SFT. The two sampling arms use identical normal sources and matched category/label order. Classification is also compared under one common decision-only evaluation prompt. The [initial experiment plan](configs/experiment_plan.json) defines those training and evaluation budgets.

Both experiment rounds are complete. Start with the **[v2 Chinese experiment report](reports/multitask_v2/实验报告.md)**, [frozen protocol](reports/multitask_v2/实验协议.md), [generation comparisons](reports/multitask_v2/validation_matrix.md), and [continuous scoring report](reports/multitask_v2/scoring_matrix.md). Three fresh adapters compare region-only, decision-only and mixed-task SFT at the same 768 sources, 1,536 exposures and 192 updates. Mixed tasks show no overall advantage; decision-only validation Macro F1 is 85.83% and AUROC is 94.56%, while region-only box recall is 46.67%. These are single-seed development results. The new 240-image confirmation set remains unused by model inference.

V2 adds conditional true/false likelihood scoring, tie-aware AUROC/AP and validation-normal threshold calibration. A 5% empirical calibration FPR is not a future FPR guarantee: decision-only reaches 6.67% on the previously evaluated 120-image test, now explicitly treated as development regression. V2 passes [101 training/generation checks](experiments/multitask_v2/verification.json), 256 scoring checks and a [1,008-image byte audit](experiments/multitask_v2/source_byte_audit.json). The [delivery manifest](experiments/multitask_v2/delivery_manifest.json) indexes source, adapter and result hashes; its `reports/multitask_v2/code_snapshot.tar.gz` excludes weights and separately stored result rows and figures.

The preserved [first-round report](reports/实验报告.md), [paired comparisons](reports/paired_difference.md), [project card](PROJECT_CARD_ZH.md), and [64-check completion audit](ops/completion_verification.json) retain the original findings. Domain improvement came with higher false positives; area-coverage sampling had no resolved advantage over random sampling. The original snapshot remains separate from v2.

## Server layout

Project: `/root/autodl-tmp/industrial_vlm`

| Directory | Contents |
|---|---|
| `models/Qwen3-VL-8B-Instruct` | Existing local base weights, structurally checked and SHA256 recorded |
| `datasets/VisA` | Complete official VisA archive extraction |
| `datasets/MVTec_AD_pilot` | Fixed 150-image external pilot from a revision-pinned public mirror, plus masks |
| `prepared/visa` | Source manifest, supervised splits, training arms, heldout evaluation questions |
| `prepared/mvtec` | External questions; no SFT samples |
| `runs` | Training configurations, parameter audits, losses, checkpoints and final adapters |
| `evaluations` | Individual predictions, metrics and input/weight identities |
| `reports` | Experiment matrix |
| `ops` | Deployment, cleanup, runtime and data audit records |
| `logs` | One log per pipeline stage |

The retained runtime interpreter is `/root/autodl-tmp/vlm/.venv/bin/python`. Its installed packages are recorded in `ops/runtime_audit.json`. Old datasets, models unrelated to this experiment, code and training artifacts were removed under the user's authorization; `ops/cleanup_manifest.json` lists the exact paths. Local paper-reading materials were outside that server cleanup.

## Data and supervision

Read [DATA_PROTOCOL.md](DATA_PROTOCOL.md) for the complete protocol. All 10,821 validated VisA images are re-split into 7,575 train / 1,624 validation / 1,622 test sources before instruction construction. Only training annotations construct SFT targets and update model weights; validation and test annotations support validation or scoring. This uses real annotated training anomalies, whereas the original AnomalyGPT unsupervised experiments construct simulated anomalies from normal training images.

VisA original masks contain class indices: **0 is background and every nonzero index is anomaly foreground**, following the [official preparation script](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py). MVTec masks are validated as binary. One mask becomes a union bounding rectangle, not a set of inferred defect instances. Coordinates are integer xyxy in `[0,1000]` relative to the full image.

The pilot uses 768 training images in each arm and 120 balanced heldout images for the primary test. Complete holdout manifests are also exported. The MVTec evaluation contains 150 selected official test images; it is an external pilot, not the complete official benchmark. Exact-byte duplicate isolation does not establish physical-part identity isolation, and public benchmark exposure during foundation pretraining is unknown.

## Reproduce

V2 actual training and scoring imports are preserved separately under `experiments/multitask_v2/runtime_src` and `scoring_runtime_src`, with frozen source manifests. Read the [v2 report](reports/multitask_v2/实验报告.md) for CPU verification commands and inference examples. `ops/run_multitask_v2.py` and `ops/run_scores_v2.py` record the full execution matrix. Keep completed result directories intact when preparing another run. The retained runtime uses PyTorch 2.5.1/CUDA 12.4, rather than the original image's preinstalled versions.

The following commands describe the **first-round** protocol:

From the server project directory, use the recorded interpreter and native inputs:

```bash
cd /root/autodl-tmp/industrial_vlm
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
PY=/root/autodl-tmp/vlm/.venv/bin/python

$PY -m industrial_vlm.data --data-root datasets/VisA --output prepared/visa \
  --train-size 768 --eval-size 120 --seed 42
$PY ops/validate_prepared.py
$PY -m industrial_vlm.train --config configs/quality_region.json
$PY -m industrial_vlm.train --config configs/random_region.json
$PY -m industrial_vlm.train --config configs/quality_decision.json

$PY -m industrial_vlm.evaluate \
  --model models/Qwen3-VL-8B-Instruct --adapter runs/quality_region/adapter \
  --data prepared/visa/test_balanced_120.jsonl --image-root datasets \
  --output evaluations/manual_quality_region --max-pixels 524288 --max-new-tokens 128
$PY -m industrial_vlm.report --runs evaluations --output reports
$PY -m industrial_vlm.compare --evaluations evaluations --output reports
$PY ops/plot_results.py --root "$PWD" --output reports/figures
$PY ops/plot_method.py --output reports/figures
$PY ops/plot_examples.py --root "$PWD" --output reports/examples
$PY ops/verify_completion.py --root "$PWD"
$PY ops/write_report_zh.py --root "$PWD"
```

`ops/run_pipeline.py` performs the interface checks, a separate small training-fit diagnostic, all three training arms, the matched evaluation matrix and external evaluation sequentially. Logs and `pipeline_state.json` record progress. Evaluation supports `--resume` with data/config and adapter-file identity checks. Training checkpoints contain adapter weights and configuration; optimizer-state training resume is not implemented. Preserve an incomplete run and restart a fresh run with the same budget rather than silently joining histories.

The manual evaluation uses a separate output directory. To resume a pipeline evaluation, reproduce all recorded arguments, including absolute model/adapter paths and branch metadata, and pass `--resume`. Validation CE in this pilot uses the first 24 fixed validation examples at updates 32, 64 and 96, not all 120 validation examples. Final update 96 is selected in advance.

For an unlabeled image, the standalone command uses the same model loader, prompt and structured-output parser:

```bash
$PY -m industrial_vlm.infer \
  --model models/Qwen3-VL-8B-Instruct --adapter runs/quality_region/adapter \
  --image /absolute/path/to/image.png --task region --output prediction.json
```

Do not overwrite adapters or change datasets/configurations under an existing result directory. Use a new experiment directory for a changed protocol. The local `ops/remote.py` helper uses an already authenticated SSH control socket and does not contain authentication secrets.

## Interpret results

`macro_f1` is the mean F1 of the two labels, normal and anomalous. It is not an average across the 12 product categories. Per-category results are separate. Region-task classification includes malformed-output failures; common decision-only prompts provide the cleaner classification comparison.

The reported schema-valid rate accepts an optional JSON code fence, finite numeric coordinates and multiple boxes; predicted boxes are reduced to their enclosing rectangle for union-box evaluation. It is not exact integer/one-box output compliance. A failed or missing localization remains in the denominator of end-to-end anomaly localization recall. Anomaly scores and AUROC are not inferred from discrete generated answers.

Area coverage versus random sampling tests one data-design choice. It does not establish that annotations are cleaner. A single seed and small pilot do not establish a statistically significant method improvement. Source-level confidence intervals summarize each run; paired improvement claims require a paired difference analysis.

The region-supervision comparison matches source images, order and optimizer updates. Its training prompt, response length and supervised-token budget differ, so the contrast measures the overall effect of adding the region task. It does not isolate a box-only signal at equal compute.
