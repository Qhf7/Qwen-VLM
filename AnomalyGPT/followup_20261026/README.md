# Strict AnomalyGPT language migration follow-up

This directory audits a small migration pilot: ImageBind features, the released
VisA decoder/PromptLearner, and either Qwen3-VL-8B's language model or
Vicuna-7B-v1.1. Qwen's native vision tower is not used. Normal-reference
replacement and PandaGPT mixed training are omitted consistently.

The final research interpretation is in
`docs/strict_qwen_followup_20261026/` at the workspace root. Historical snapshots
remain historical evidence; inspect completion fields before using their values.

## Protocol

- Real diagnostic set: 2,005 official VisA test sources, 925 normal and 1,080
  anomaly, 12 categories. The source list was frozen before inference. This set
  has now been inspected and is a development diagnostic, not a future untouched
  confirmation set.
- Excluded 24 balanced synthetic test sources and 133 official-test sources
  overlapping the native project's sealed confirmation set. Eight normal candle
  sources from the earlier test8 pilot remain; a separate 1,997-source sensitivity
  analysis excludes them without changing the primary set.
- Square 224 RGB bicubic; palette mask indices >0, then nearest resize; no normal
  references. Four per-layer softmax maps are averaged before scoring.
- Pixel AUROC/AP are computed within each product category over all its images,
  then averaged equally across 12 categories. Image heatmap score is max(map).
- Generated answers use greedy decoding, own EOS and decoded conversation stop.
  Invalid answers remain in the denominator. Yes/No continuation likelihood is
  a separate diagnostic, not generated-answer accuracy.
- Three matched arms × seeds 42/43/44: frozen Qwen, Qwen q/k/v/o LoRA, frozen
  Vicuna. Same saved bridge tensors and explicit image schedule within seed;
  16 AdamW updates × batch2, target cap128. Actual exposure is 32 variants per
  seed, not all 96 cached variants. Different tokenizers imply different compute.

## Continue and verify

On the server, the main experiment is
`/root/autodl-tmp/AnomalyGPT/experiments/strict_qwen_bridge_v0` and runtime is
`/root/autodl-tmp/conda-envs/qwen3vl2/bin/python`.

`run_followup_safe.py --experiment EXPERIMENT --archive-partial` verifies
completed artifacts and preserves partial directories before restarting them.
It holds a process lock and records source snapshots, hashes, and per-job
validation. Inspect the GPU/process state before resuming; do not start concurrent
GPU jobs. `safe_state.json` supersedes the old runner's `state.json`.

Vicuna's audited safetensors resource is recorded in `vicuna_resource.json`.
If its temporary directory is lost after restart, regenerate it with
`convert_vicuna_weights.py` under the existing modern PyTorch environment and
verify all 323 tensors equal before resuming. Do not change the main runtime.

After all 32 main tasks complete:

```bash
python summarize_followup.py --root FOLLOWUP --out FOLLOWUP/final_report
python diagnose_generation.py --root FOLLOWUP --out FOLLOWUP/generation_diagnostics.json
python sensitivity_previous_test8.py --root FOLLOWUP --out FOLLOWUP/final_report/sensitivity_previous_test8.json
python prompt_map_ablation.py --experiment EXPERIMENT --root FOLLOWUP --out FOLLOWUP/map_prompt_diagnostic
python plot_followup.py --summary FOLLOWUP/final_report/summary.json --out FIGURES --historical
python archive_followup.py --root FOLLOWUP --out NEW_SNAPSHOT.tar.gz
```

The map intervention uses a fixed 48-image subset of the already viewed real
set. It swaps only predicted maps entering PromptLearner with opposite-label
images within category. Each image's global feature and question stay fixed.
It records map/adaptive prompt separation, answer changes, and same-condition
candidate-score repeat variation. Sensitivity alone does not establish correct
localization or detection gains. Its output directory must be new.

## Audit boundaries

The initialization checkpoint is released and already trained, not random.
Its full prior training manifest is unavailable. Shared PromptLearner weights
were originally paired with Vicuna and do not guarantee equal semantic alignment
to Qwen. This pilot cannot rank overall backbone capability or claim full paper
reproduction, unseen-category generalization, or statistical superiority.

The old synthetic layer-concatenation metric and mean batch teacher-forcing token
accuracy are preserved for audit. They must not be described as final-map quality
or free-answer detection accuracy. Corrected synthetic image labels use explicit
labels/conversation semantics; pixel masks keep the >0.3 threshold.

Lightweight archives include code, metadata, logs, predictions, and per-file
SHA256. Weights, feature tensors, and pixel arrays remain at the source paths;
the archive is not a full model/data backup. The summary checks pixel metric
aggregation but does not independently reread all pixel arrays.
