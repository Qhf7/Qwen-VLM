# VisA mask-supervised structured localization continuation v1

## Frozen question and design

Can language-only LoRA, continued from the original C best checkpoint, convert the existing compressed CLS + teacher-map representation into a parseable anomaly decision and spatial answer? This extension adds **training-mask-derived spatial supervision and longer answers**. It is not an equal-supervision replacement of binary CE and does not demonstrate causal explanations, chain-of-thought gains, or native Qwen vision capability.

Original C best for each matching seed 42/43/44 is fixed as initialization, irrespective of D/ranking/external scores. The bridge, base language tensors, teacher and encoder remain frozen. Only the same 72 language attention q/v LoRA matrices (r=8, alpha=16, dropout=0) continue training; no extra adapter is stacked. Formal protocol: LR 1e-4, AdamW weight decay .01, OneCycleLR pct_start .05, 500 updates, microbatch1 × accum4, 2000 image exposures, deterministic balanced label sampling with seed+100000. Target token mean CE including EOS is averaged over accumulation, with global gradient clipping 1.0. No ranking term is used in this stage.

Use all full VisA validation 380 images for checkpoint selection by maximum first-answer Yes−No image AUROC, evaluated every100 updates. Initial C under the new structured prompt is step0 and is eligible; earliest ties win. Report initial/best/final, and the original C binary prompt results separately. A gain relative to initial C under a new prompt is not sufficient to establish a gain over original C.

## Inputs and target contract

Inputs remain image-only cached CLS1024 + teacher map224, passed through the fixed 19-token bridge. No mask, GT box, target text, or label is passed at inference. Normal/anomaly ground-truth labels and masks are used only to form train labels; validation targets are evaluation references. Historical test masks and predictions are not opened.

The first line is exactly Yes or No, retaining a one-token Yes/No candidate and the same score definition. The second line is one JSON object with exactly bbox and location:

    Yes
    {"bbox":[150,200,400,400],"location":"top_left"}

Normal target: No followed by {"bbox":null,"location":"none"}. For an anomalous image, binarize original mask values >0, enclose **all** foreground components in one union box, and normalize edge coordinates to0..1000. Min edges round outward down and max exclusive edges outward up. The 3×3 location is determined from the normalized box center. This is an explicit coarse spatial target; disconnected anomalies may produce large union boxes, and tiny defects may be information-limited by compressed224 visual features. No diagnosis, defect type or physical root cause is invented.

Fixed prompt lists the schema, coordinates and nine allowed location labels. It does not contain a sample-specific answer. Greedy decoding has a fixed80-token budget across initial/best/final; malformed/truncated results remain failures. No grammar-based decoder silently repairs generation. Optional batch4 decoding is an engineering optimization: each initial/final/best weight state runs a fixed VisA val8 scalar-versus-left-padded-batch token+EOS exact parity gate. A failure or OOM fixes that condition to scalar before any metric result is evaluated. Each decision, token difference, elapsed time and generation implementation SHA is serialized in its generation_policy.json. Fresh-process verification and external evaluation force that saved decision; neither chooses implementation from external data. The historical parent generator is unchanged. If selected best is step0 or the final step, re-score every validation image after checkpoint reload and require exact agreement, then reproduce at least eight samples in their original evaluation batches and require exact generated text. Only after these checks may best reuse the already completed initial/final raw generations. Best copies that endpoint generation policy byte-for-byte and records reference/checkpoint/policy hashes and checked sample identities in best_generation_reuse_audit.json. An intermediate selected checkpoint receives a new VisA parity gate and full generation pass. Training, selection and model predictions are unchanged by this reuse.

## Metrics and controls

- Standard overall/per-category AUROC/AP from first-token logits, generated macro F1/recall/FPR/invalid decisions from initial Yes/No, with label counts. Format validity is separately stricter than classification validity.
- Strict schema validity requires exactly two lines, exact two JSON keys, integer valid ordered bounds, Yes/box or No/null consistency, and location consistent with predicted box center.
- On **all anomalous images**, report mean box IoU, IoU≥.5, predicted box center inside GT union box, and 3×3 location accuracy. Invalid/No/missing box is zero. These are box metrics, not pixel segmentation or a pixel-mask pointing game.
- Also report mean IoU/IoU≥.5 conditional on valid positive boxes alongside the number of such predictions. Never present the conditional subset as overall localization quality.
- Per-category localization results expose imbalance. Normal images affect classification/format validity but do not inflate anomaly-only localization scores.
- Fixed teacher-map bbox control: per-image95th percentile threshold, union of pixels ≥threshold, no mask fitting. Show unconditional boxes and boxes gated by initial structured-prompt C score>0 separately. This establishes a simple localization-teacher alternative; the compressed LLM already receives this teacher map.
- The initial original C with structured prompt is the frozen-weights instruction-only baseline. The binary original C development report remains an additional classification reference.
- Same-process save/load scores must be exact. A fresh process reconstructs original C + structured adapter, verifies hashes/tensors, checks balanced labels/all categories using preserved original batch composition, and requires exact scores and greedy generations.

Engineering preflight:40updates on eight repeated balanced **training** examples and eight balanced validation examples, same LR; requires loss reduction, finite nonzero gradients, frozen parameter identity and exact reload. Its small validation results are diagnostics, not reported model performance. Validation mask references must never determine gradients.

## Commands on the authorized training server

Upload advanced_generation.py, structured_geometry.py, structured_model.py, structured_train.py, structured_verify.py alongside the new migration_advanced_train.py; shared migration_language.py, migration_train.py, migration_metrics.py, dec_big.py, visa_pipeline.py are already at repository root. Set ANOMALYGPT_REPO=/root/autodl-tmp/AnomalyGPT. The caller must hold the shared GPU lock while running GPU commands; these scripts do not acquire it internally so a serial orchestration process can own it.

    cd /root/autodl-tmp/AnomalyGPT
    PY=/root/autodl-tmp/conda-envs/qwen3vl2/bin/python
    OUT=runs/migration_20261103/advanced_20261104
    $PY structured_geometry.py --self-test
    $PY structured_geometry.py --cache runs/migration_20261103/teacher_cache.pt --out "$OUT/structured_targets.jsonl"
    $PY structured_train.py --cache runs/migration_20261103/teacher_cache.pt --targets "$OUT/structured_targets.jsonl" --source-c runs/migration_20261103/lora_20261104/arm_C_s42 --base-manifest runs/migration_20261103/lora_20261104/base_manifest.json --out "$OUT/preflight_S_s42" --seed 42 --preflight --steps 40 --val-every 20

Formal command, repeated for matching seeds42/43/44 after successful preflight:

    $PY structured_train.py --cache runs/migration_20261103/teacher_cache.pt --targets "$OUT/structured_targets.jsonl" --source-c runs/migration_20261103/lora_20261104/arm_C_s42 --base-manifest runs/migration_20261103/lora_20261104/base_manifest.json --out "$OUT/arm_S_s42" --seed 42
    $PY structured_verify.py --run "$OUT/arm_S_s42" --mode final
    $PY structured_verify.py --run "$OUT/arm_S_s42" --mode best

Target builder opens only train/val masks, saves mask SHA256 and manifest/cache identity, and will fail on missing or empty anomaly masks. Mask files are not bundled locally in this workspace; the authorized server's VisA root is required for the actual dataset audit. Local CPU tests use known geometry and malformed outputs; those tests are not training or measured model results.
