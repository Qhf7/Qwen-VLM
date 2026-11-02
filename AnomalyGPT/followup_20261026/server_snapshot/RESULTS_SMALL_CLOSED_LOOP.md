# Strict AnomalyGPT → Qwen3-VL small closed loop

## Scope

This is the strict replacement branch. It keeps the original AnomalyGPT ImageBind visual encoder, ImageBind normal/abnormal text anchors, Image Decoder, Prompt Learner, anomaly-map construction, and synthetic anomaly protocol. Qwen3-VL-8B is used only through its `language_model`; the native Qwen vision tower is not used in this branch.

The Qwen language model is frozen in the main run. Trainable modules are `image_decoder`, `prompt_learner`, and `llama_proj`. Qwen language LoRA is not included in the main result and must be a separate future ablation.

## Completed steps

1. Fixed ImageBind cache construction. The original dataset eagerly loaded 8,659 images and could be killed before caching. `datasets/visa.py` now loads images lazily and supports `VISA_SPLIT=train/test`.
2. Cached 32 train sources × 2 variants = 64 files in `cache_visa32/`.
3. Trained the main bridge for 16 steps, batch size 2, with focal + dice region loss and assistant-only language loss.
4. Cached 8 independent test sources × 2 variants = 16 files in `cache_visa_test8/` using `VISA_SPLIT=test`.
5. Evaluated the main checkpoint, an untrained initialization baseline, and two controlled ablations.

## Results

Metrics are computed on the same 16 held-out synthetic variants. Pixel masks are thresholded at 0.3 for AUROC/AP. This is a smoke-scale diagnostic, not a final VisA benchmark.

| Run | Training data | Region loss | Token acc | Combined loss | Pixel AUROC | Pixel AP |
|---|---|---:|---:|---:|---:|---:|
| Initial baseline | decoder/prompt initialization only | yes | 0.6473 | 3.0888 | 0.8749 | 0.6718 |
| Main strict bridge | 32 sources, normal + synthetic | yes | 0.7067 | 2.4905 | 0.8822 | 0.6812 |
| No-region ablation | 32 sources, normal + synthetic | no | 0.7317 | 2.9383 | 0.8532 | 0.6100 |
| Normal-only composition ablation | 32 normal variants only | yes | 0.7061 | 2.6972 | 0.9443 | 0.6724 |

The main run improves language token accuracy and slightly improves localization over initialization. Removing region supervision reduces pixel AUROC/AP despite a higher token accuracy. The normal-only result is exploratory and should not be claimed as a general data-quality improvement: it uses the same tiny 16-step budget and can reflect optimization variance and the synthetic test protocol.

## Artifacts

- Main checkpoint: `runs/cached_strict_32/pytorch_model.pt`
- Main run metadata: `runs/cached_strict_32/run.json`
- Main test metrics: `runs/cached_strict_32/test_metrics.json`
- No-region checkpoint/metrics: `runs/ablation_no_region/`
- Normal-only checkpoint/metrics: `runs/ablation_normal_only/`
- Initial baseline metrics: `runs/initial_baseline_test_metrics.json`
- Train cache manifest: `cache_visa32/manifest.json`
- Test cache manifest: `cache_visa_test8/manifest.json`
- Checkpoint hashes: `checkpoint_sha256.txt`

## Limits and next experiment

This is not a full AnomalyGPT reproduction and not a final comparison with the AnomalyGPT paper. The test set is a small held-out source cache and uses the existing synthetic anomaly generator; no real VisA pixel benchmark is claimed here. The next credible stage is a larger source-disjoint train/validation/test cache, at least three seeds, and a separately labeled Qwen language LoRA ablation. Do not merge that LoRA result into the strict frozen-language main line.
