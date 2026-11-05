# Native ROI protocol audit — 2026-11-03

## Scope and disposition

Read-only review of `/Users/qiaohongfei/anomalygpt_remote/files/migration_roi_eval.py` at SHA256 `324aaeb97eea674d63b322129a01cc6694547f8f884726450f327f390e9cc711`. No ROI code, native trainer, parity script, or runner was modified. No SSH or GPU operation was performed in this audit.

**Disposition: no execution-blocking defect found in the reviewed paths.** The CPU contract tests described below pass. The 448-pixel, three-image configuration has a bounded inference design and a modest KV cache; actual 24 GiB GPU peak memory remains a runtime measurement, not an audit result.

## Native inference and scoring

- `main` loads the official `Qwen3VLForConditionalGeneration` in BF16 with SDPA, uses `.eval()` and `requires_grad_(False)`, and surrounds all condition/query evaluation with `torch.inference_mode()` (`migration_roi_eval.py:377–429`). No training graph or adapter fitting is present.
- `score_and_generate` invokes the official top-level forward with `use_cache=False, logits_to_keep=1`, reads the unprocessed first-answer-token logits for the independently checked single tokens `Yes` and `No`, and scores `logit(Yes)-logit(No)` at threshold zero (`285–307`). The score is computed before any generated text and does not use generation processors.
- The score output/logits are explicitly deleted before a separate official `model.generate` call. A clean `GenerationConfig` sets greedy, one-beam generation, `use_cache=True`, and a default 32-token answer budget; `use_model_defaults=False` avoids importing model-shipped generation settings (`303–310`, `404–407`). Both paths reset `model.model.rope_deltas` for a fresh prompt.
- The official Transformers 4.57.1 source clears `pixel_values` after the cached generation prefill and computes the remaining text steps using the native position/cache path. The ROI script does not substitute a manual MRoPE, merger, or DeepStack implementation.
- Score metrics and generated-text metrics are separate channels in `migration_metrics.evaluate_predictions`; generation is parsed by the existing initial-Yes/No rule, including invalid-answer penalties.
- The runtime source gate checks Transformers 4.57.1 and modeling source SHA256 `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`, then audits the full model dtype. An explicitly supplied native parity report must have both `passed` and `all_cases_exercised_cached_decoding`, and its modeling source must match the runtime source (`354–358`, `385–398`).

The existing `native_parity.json` passed on RTX 4090 D and exercised square, landscape, and portrait **single-image** cached decoding. It is evidence for the official native path; it is not a measured three-image ROI smoke. `--parity-report` is optional in the CLI, so the formal ROI invocation should supply the existing passing report explicitly.

## 448-pixel geometry and memory

- Every full image, target crop, and optional normal reference is a 448×448 RGB image. The official processor receives `images_kwargs={"do_resize": False}`. Actual `image_grid_thw`, patch size, merge size, placeholder count, and per-view token equality are validated and recorded for every query (`218–281`).
- With the audited patch size 16 and merge size 2, each view must have grid `[1,28,28]`: 784 vision patches and 196 merged language-model image tokens. One, two, and three images therefore use 196, 392, and 588 visual language-model tokens. The script rejects a changed actual resolution instead of silently accepting processor resizing.
- Official 4.57.1 vision SDPA splits attention at `cu_seqlens` for each image. Three images are processed as three 784-patch attention chunks; they do not form one 2,352-patch cross-image vision attention matrix. The text decoder does attend across all merged image tokens as intended.
- The official model index reports `total_size=17534247392` bytes, about 16.330 GiB of BF16 weights. For the audited 36 text layers, 8 KV heads, and head dimension 128, the BF16 KV cache is `36 × 2 × 8 × 128 × 2 = 147456` bytes per token. Three images plus an illustrative 160 text tokens and 32 generated tokens give about 109.7 MiB of KV cache. The actual text count is recorded; 160 is a sizing assumption, not a measured prompt length.
- The script evaluates one unpadded query at a time. It frees the score tensors before generation, keeps JSON/CPU metadata rather than GPU tensors in `records`, and deletes `batch` and PIL views after every query. No cross-query GPU cache is retained beyond the small `rope_deltas` state, which is reset at each forward (`292–310`, `414–438`).

**Memory interpretation:** the structure is consistent with controlled three-image inference on an otherwise free 24 GiB RTX 4090 D. The weight and KV figures omit allocator reservations, CUDA context, SDPA workspaces, and transient activation peaks, so they do not prove the run fits. Before a longer pilot, run one query with the reference condition and inspect `peak_gpu_allocated_bytes` and `peak_gpu_reserved_bytes`. These fields are already recorded per condition. Allocated peak is reset per condition; the reserved allocator pool can carry over from earlier conditions, so reserved peaks are not independent incremental-memory comparisons across conditions.

## Split isolation and crop controls

- `select_validation_indices` admits only `split == "val"`. The 48-image pilot balances category/label strata deterministically; labels influence subset composition and final metrics, not the ROI location or prompt. `--limit 0` returns the entire validation split (`66–89`). All conditions reuse the same selected indices and question.
- The ROI center is determined by the cached teacher probability map's first row-major maximum. No ground-truth mask is read, and the target label does not enter `roi_spec` or prompt construction. The cached direct-square-224 coordinates map linearly to the original image x/y coordinates (`110–149`).
- Center, seeded-random, and teacher crops have the same normalized area (default 1/9), are sampled from the original RGB with fractional bounds and bicubic resampling, and are resized to the same 448×448 output. Border crops are translated to remain inside the image without shrinking. For a non-square raw image, this is a square in normalized coordinates and a proportional rectangular region in raw pixels; it is not a fixed raw-pixel square.
- The random control is deterministic by seed plus target path. It samples a uniformly valid crop location; center and teacher controls apply the same fixed-size border translation rule.
- Only validation RGB targets and the selected known-normal training RGB references are opened for inference. The shared-cache integrity sweep can validate cached entries from all splits, including test entries if present; no test index is selected, scored, used as a reference, or included in ROI metrics.

## Reference eligibility and interpretation

`build_reference_bank` selects only `split == "train"`, `label == 0`, same-category entries, and normalizes finite nonzero cached ImageBind CLS vectors. `select_reference` requires a validation query, chooses the maximum same-category cosine with a deterministic first-index tie rule, and rechecks train/normal/category/different-path eligibility (`152–192`). Neither validation-normal nor test-normal entries enter the reference bank.

The controlled location comparison is **center crop vs. random crop vs. teacher ROI**, all of which supply two equal-resolution images with equal crop area and visual-token budgets. `original_only` supplies one image and therefore has fewer tokens. The reference arm supplies three images and adds known-normal training information selected by ImageBind similarity. An improvement over either arm alone must not be attributed solely to better ROI localization or solely to the native Qwen representation. These information/token differences are already disclosed in `plan.json` under `token_budget_comparison`.

## Local CPU verification

A temporary-fixture test was executed with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python3`; it did not instantiate a Qwen model or call CUDA. It passed all of the following:

1. AST parse and compile of the reviewed ROI file.
2. Non-square source RGB, teacher peak at a border, exact equal normalized crop areas, contained crop bounds, and deterministic random crop.
3. All five conditions produce one/two/three 448×448 views; the reference arm reuses the same teacher crop.
4. Mock official-processor contract checks `do_resize=False`, actual `[1,28,28]` grids, image placeholder counts, BF16 pixel values, and 588 merged visual tokens for three views.
5. A cache fixture containing validation, training-normal, training-anomaly, and test-normal entries selects only the validation target and training-normal reference. A full CPU CLI plan with `--limit 0 --with-reference` writes five conditions and never queries CUDA availability.
6. Mock scoring/generation runs inside inference mode, resets native rope state, uses `use_cache=False` for scoring and `use_cache=True` for generation, and proves the score tensor has been released before generation starts.

These are contract/CPU checks. The actual installed processor, full three-image native forward, GPU peak, and resulting validation metrics must be established by the planned runtime pilot. No effect-size claim is made by this audit.

## Source anchors

- Local ROI source: `/Users/qiaohongfei/anomalygpt_remote/files/migration_roi_eval.py`.
- Existing native execution evidence: `migration_20261103/native_parity.json` and `migration_20261103/native_aux_official/report.json`.
- Audited official implementation: `https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`.
- Official processor keyword routing: `https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py`.
- Official config and weight index: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json` and `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/model.safetensors.index.json`.
