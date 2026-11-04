# Advanced source audit — 2026-11-04

Source/inventory read only; no SSH, GPU execution, or new external predictions queried by this auditor.

## Training mechanics

- configure_arm allows bridge only in B/D and language LoRA_A/LoRA_B only in C/D/R. D optimizer groups use separate LRs; optimizer parameter-ID coverage is asserted. Frozen head/base remain frozen.
- C/R bridge tensor digests remain fixed. Frozen parameter versions/gradients are checked at validation and completion.
- paired_schedule emits same-category normal then anomaly pairs with category-uniform replacement sampling. Identical seeds/exposures produce identical Cpair/R sample indices. Formal microbatch2/accum2 preserves exactly one pair per ranking loss.
- Ranking uses differentiable forward_batch at prefix_length−1, score=logit(Yes)−logit(No), loss=softplus(−(score_anomaly−score_normal)). Direction and gradient path are correct. Ranking CE and Cpair model.loss share shifted supervised-token mean CE including EOS.
- D checkpoint stores/restores both bridge and adapter. C/R restore adapters against immutable source bridge. Adapter keys and tensors are checked exactly. In-process best reload first perturbs a trainable tensor.
- R versus Cpair isolates ranking. R versus historical C additionally changes sampling/microbatch.

## Important findings sent to parent

The parent may already have fixed these; this records findings rather than asserting they remain outstanding.

1. Runner originally required all3 preflights before first D formal job. D needs only D gate; paired arms need Cpair/R gates.
2. dec_big.atomic_json requires string paths; external evaluator originally passed Path.
3. External evaluator should verify external cache sidecar SHA, decoder SHA and pipeline match training identity, plus source bridge/base-manifest file hashes.
4. Fresh exact reload should preserve complete original eval batches and cover categories/labels. Historical first32 already contains candle normal16/anomaly15 and capsules normal1; it is not all-normal.
5. Best selection admits step0, so best verifier must allow zero LoRA_B iff selected step0. Final verifier can require nonzero LoRA_B.
6. Preserve historical B/C trainer and manifests; advanced runs need separate trainer with actual SHA. Preserve transient D preflight provenance.

## Existing external inventory audit

- Pilot manifest, external_base_region predictions, external_quality_region predictions each have150 unique source IDs; all three sets are identical.
- MVTec retained1575 source IDs have zero intersection with either prior pilot prediction inventory.
- Retained MVTec image SHA256s have zero intersection with all10821 unique image SHA256s in industrial_vlm/prepared/visa/manifest.jsonl.
- VisA inventory SHA256: ef4efe06080560639f1c24854ebb19500cc82ba73c6c9b27be94d406a9e8ae4a.
- That inventory covers full VisA content but uses a different split (7575/1624/1622) from teacher cache (9985/380/456). This is full-content hash disjointness, not direct remote-cache image-hash inspection.

Evidence supports a branch-specific public cross-dataset audit. It does not establish private unseen data, unknown foundation pretraining exposure, or absence of every historical project use.

## Offline CPU contract regression

Executed test_advanced_contracts.py successfully against migration_advanced_train.py. The test extracts the actual eight trainer functions with AST and runs them with a randomly initialized tiny two-layer LlamaModel (no pretrained weights/data/downloads and no remote runtime stubs).

Verified all B/C/D/R trainable whitelists, FP32 trainables, exact optimizer tensor coverage without duplicate tensors, target module count, and per-group LR. Two optimizer updates verify bridge and adapter gradient groups, LoRA_A nonzero after LoRA_B updates, and unchanged frozen tensors. D adapter+bridge disk checkpoint restores every trained tensor and identical outputs in both the mutated existing model and a fresh model construction; wrong identity is rejected. A 2000-exposure schedule is seed reproducible and consists solely of same-category training normal/anomaly pairs.

Result artifact: cpu_contract_audit.json (records exact source SHA and local torch2.10.0/transformers4.43.1/PEFT0.11.1). Remote runtime versions differ, so this supplements rather than replaces the real Qwen engineering preflight and fresh-process audits.
