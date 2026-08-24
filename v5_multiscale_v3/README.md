# DINOv2 Multi-scale Decoder v3

This is an independent successor to `v4_hybrid_v2`; it does not reuse or
overwrite v2 checkpoints or submissions.

Core changes:

1. Layer-aware DINOv2 fusion: shallow texture, middle structure and deep
   semantic tokens are projected independently instead of concatenated.
2. A top-down multi-scale pixel decoder predicts at 4x the patch-grid
   resolution and is supervised against high-resolution pseudo masks.
3. Industrial pseudo defects emphasize hairline scratches, edge damage,
   low-contrast stains and subtle local photometric changes.
4. Hard normal negatives include object silhouettes, specular highlights,
   high-gradient normal edges and repeated texture responses.
5. Overlapping local windows add a fine-detail anomaly branch without
   replacing the full-view global context.
6. Five-view consistency is confidence-aware: repeated evidence is boosted,
   while isolated low-confidence responses are suppressed. It does not assume
   pixel correspondence between cameras.
7. Train-only proxy model selection ranks configurations primarily by pixel
   AP and pixel F1max, with pixel AUROC as a secondary tie-breaker.

Compliance:

- training uses official Train normal images only;
- pseudo anomalies are generated procedurally or from self-crops of the same
  official image;
- DINOv2 is a permitted generic public backbone;
- no online API, external anomaly labels or specialised anomaly checkpoint is
  used.

Server output names are deliberately separate:

```text
/root/autodl-tmp/qwen_dinov2_multiscale_v3_ckpt
/root/autodl-tmp/qwen_dinov2_multiscale_v3_dense
/root/autodl-tmp/qwen_dinov2_multiscale_v3_hybrid
/root/autodl-tmp/submission_qwen_dinov2_multiscale_v3.zip
```

Run:

```bash
bash v5_multiscale_v3/run_v3.sh
```
