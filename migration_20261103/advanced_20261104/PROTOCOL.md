# Advanced migration experiments — frozen 2026-11-04

## Evidence and order

User-authorized order: lock external evaluation; D joint bridge/LoRA; paired CE versus CE+ranking; mask-supervised structured answers; bounded model-selected ROI tools. Existing VisA val380 is development data and historical test456 is not blind. Do not select configurations or prompts by MVTec results. Freeze all stages before opening external results.

## D: joint adaptation

Initialize from each seed's original Qwen bridge (phase1 best step1500), rather than the trained C adapter. Train bridge at 1e-5 and language q_proj/v_proj LoRA at 1e-4, rank8/alpha16/dropout0. 500 updates, microbatch1 × accumulation4, 2000 exposures; same sampling schedule as historical C. 42/43/44 paired comparisons to original C. Both bridge and adapter saved; step0 eligible for best AUROC. Report selected best and fixed final separately, AUROC/AP, generated F1/recall/FPR, cost. Preflight repeated 8 training examples, 40 updates, 8 validation examples, exact zero-adapter/reload checks and nonzero gradient checks; no efficacy conclusion from preflight.

## Paired objective comparison

Cpair = frozen bridge + language LoRA, CE only. R = identical initialization, data sequence, optimizer, microbatch2 × accumulation2, 500 updates, 2000 exposures, with CE + lambda softplus(-(score_anomaly-score_normal)). Pair normal/anomaly within category; category-uniform replacement sampling. First-answer score is differentiable logit(Yes)-logit(No). CE is global supervised-token mean per two-image microbatch, then mean over accumulation. Compare R to Cpair, rather than attributing changes relative to historical C solely to ranking. Both use q/v rank8 alpha16 and LR1e-4.

Lambda rule frozen before formal training: 16 train-only normal/anomaly pairs, fixed seed100042, median initial gradient-norm ratio ||grad CE||/||grad rank||, clamp [0.01,1.0]. Actual median14.063660528604052 yields lambda1.0. This calibrates scale; it is not validation tuning or an efficacy result. Paired preflights both required before paired formal runs. Per-stage gate fix on 2026-11-04 recovered completed D preflight without rerunning/overwriting it.

## Source provenance and artifacts

Historical B/C trainer remains the Git version under lora_20261104; new experiments use migration_advanced_train.py here. The original D preflight was executed using the transient expanded trainer named migration_lora_train.py; formal runs record new trainer SHA. Existing historical source manifests are preserved. Every run saves source SHA, base/bridge/cache identity, schedule, target modules, trainables, gradients, predictions, best/final checkpoints, and exact in-process reload. Fresh-process audits cover complete original evaluation batches spread across first/middle/last validation areas to retain identical batch companions.

No optimizer state is resumable. Existing completed outputs are read and skipped; incomplete outputs must be preserved under an explicit failed-attempt name before rerunning. No overwrite of failed evidence.

## External protocol

See external_mvtec/protocol.json and manifest.jsonl (SHA256 4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a). Retained official MVTec test1575 (normal392/anomaly1183,15 categories), after excluding150 previously scored pilot images. Generic industrial-component prompt; manifest semantic name generic_industrial_component maps to internal description key industrial_component. No model-dependent image selection, no external reference images, masks, labels or images used for training. Encoder cache reads only images. Labels are used only after inference for metrics; masks only post-inference structured evaluation. Public cross-dataset audit, foundation pretraining exposure unknown. Original A and selected C/D/R weights and prompts are frozen on VisA before external queries. Keep unfavorable results and category metrics.

## Structured and ROI stage boundaries

Structured stage starts from original C best42/43/44, uses only VisA training masks as output supervision, fixed bridge and continued language LoRA. Define exact output schema and invalid-output penalties before formal training. Teacher-map box baseline and all-anomaly denominators required. Bounded tool stage must record actual LM tool requests, returned crop observations, final answer, invalid schema rate, call rate, latency, memory; no GT at inference. Fixed crop with matched visual budget is a required control. Native Qwen vision, if needed by ROI implementation, is a distinct complete-system comparison, not an attribution to the compressed bridge route.

## Execution refinements before S/external queries

All2758 external image/mask files are SHA-verified on the server; the existing archive was truncated, so intact entries were recovered and895 missing files were downloaded through a mirror then checked against the same frozen hashes. Neither source identity nor the manifest changed. No external model query was made by staging.

Optional batched greedy decoding is a new isolated module, leaving historical generation intact. For every structured checkpoint condition (initial/final/best), compare full tokens and stopping IDs with scalar decoding on fixed VisA val8; only exact agreement enables the batch candidate. Any difference or candidate exception selects scalar and records it. Verifiers force the stored policy. This is an engineering implementation gate, not external validation tuning or proof of exact equality on all images.
