# Balanced source-disjoint extension and Qwen language LoRA ablation

## Protocol

The original 32-source smoke loop was extended with a balanced source-disjoint split. VisA `1cls.csv` contains 12 industrial classes. We selected 4 normal train sources per class and 2 normal test sources per class with seed 42. Train and test image paths have zero intersection. Each normal source is paired with one deterministic synthetic anomaly produced by the retained AnomalyGPT `patch_ex` protocol.

- Train: 48 sources, 96 cached variants.
- Test: 24 sources, 48 cached variants.
- Main trainable bridge: Image Decoder, Prompt Learner, and `llama_proj`.
- ImageBind and Qwen language base are frozen in the strict baseline.
- Qwen LoRA ablation: q/k/v/o in `language_model`, rank 8, alpha 16, dropout 0.05. The LoRA adapter is saved separately.

## Matched 16-step comparison

Both runs use the same balanced cache, seed 42, batch size 2, learning rate 1e-3, and 16 optimization steps.

| Run | Trainable language side | Token accuracy | Combined loss | Pixel AUROC | Pixel AP |
|---|---|---:|---:|---:|---:|
| Frozen strict bridge | none | 0.7414 | 2.6334 | 0.8088 | 0.5705 |
| Strict bridge + Qwen LoRA | q/k/v/o LoRA | **0.9128** | **1.6805** | 0.8041 | **0.5915** |

The LoRA branch substantially improves assistant-token fit under this small budget. Its pixel AUROC is slightly lower than the frozen baseline, while pixel AP is slightly higher. This does not establish a localization improvement; longer training and multiple seeds are required.

A separate 32-step frozen run reached token accuracy 0.7641 and pixel AUROC 0.8126, but it is not used for the matched LoRA comparison because the step budget differs.

## LoRA implementation gate

The extracted Qwen3-VL `language_model` lacks the PEFT generation hook expected by the installed PEFT version. The LoRA branch adds a local `prepare_inputs_for_generation` hook before wrapping the language model. The gate passed manual `inputs_embeds` forward/backward with explicit 3×batch×sequence position IDs.

- Trainable LoRA parameters: 7,667,712.
- Fraction of Qwen language parameters: 0.1012%.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Peak VRAM during bridge LoRA training: approximately 11.85 GB.

## Artifacts

- Balanced train split: `splits/balanced_train.csv`
- Balanced test split: `splits/balanced_test.csv`
- Split manifest: `splits/balanced_manifest.json`
- Balanced train cache: `cache_balanced_train48/`
- Balanced test cache: `cache_balanced_test24/`
- Frozen 16-step checkpoint: `runs/balanced_strict_16/pytorch_model.pt`
- Frozen 16-step metrics: `runs/balanced_strict_16/test_metrics.json`
- Frozen 32-step checkpoint: `runs/balanced_strict_48/pytorch_model.pt`
- Frozen 32-step metrics: `runs/balanced_strict_48/test_metrics.json`
- LoRA bridge checkpoint: `runs/lora_strict_16/pytorch_model.pt`
- LoRA adapter: `runs/lora_strict_16/lora_adapter/`
- LoRA metrics: `runs/lora_strict_16/test_metrics.json`
- LoRA training script: `code/cached_train_strict_lora.py`
- LoRA evaluation script: `code/eval_cached_strict_lora.py`

## Interpretation and next stage

The current evidence supports a clear engineering conclusion: Qwen language LoRA is feasible on one RTX 4090D and improves generated-answer fitting, but it should be reported separately from the strict frozen-language baseline because it changes the optimization target and does not yet improve pixel localization.

The next paper-quality experiment should use at least three seeds, a larger class-balanced source-disjoint split, real anomalous test images in addition to synthetic pairs, and a fixed evaluation script that reports image-level AUROC, pixel AUROC/AP, and answer-level macro-F1. The present 16-step results are a feasibility study, not a final benchmark.

## Three-seed matched summary

Seeds 42, 43, and 44 use the same 16-step protocol and source-disjoint balanced caches.

| Run | Token accuracy mean ± std | Combined loss mean ± std | Pixel AUROC mean ± std | Pixel AP mean ± std |
|---|---:|---:|---:|---:|
| Frozen strict bridge | 0.7006 ± 0.0443 | 2.6760 ± 0.1785 | 0.8087 ± 0.0030 | 0.5755 ± 0.0075 |
| Strict bridge + Qwen LoRA | **0.8910 ± 0.0231** | **1.7187 ± 0.0347** | 0.8059 ± 0.0024 | **0.5937 ± 0.0081** |

The three-seed result rules out the interpretation that the language gain came from one lucky seed. Under this feasibility budget, LoRA consistently improves answer fitting while leaving localization approximately unchanged or slightly lower in AUROC.

Seed-level metrics are stored in `runs/seed_summary.json`. Additional checkpoints and metrics are under `runs/balanced_strict_s43/`, `runs/balanced_strict_s44/`, `runs/lora_strict_s43/`, and `runs/lora_strict_s44/`.
