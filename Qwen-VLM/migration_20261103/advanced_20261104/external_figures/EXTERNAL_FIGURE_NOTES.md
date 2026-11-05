# External figure notes



**Complete21/21 saved-result report.**



All runs: same public MVTec official-test remainder1575 images (392normal,1183anomaly),15categories, seeds42/43/44.

external_basic: C→D and Cpair→R pair matching seeds. AUROC is from Yes-minus-No logits; generated FPR/recall are operating points, not ROC curves.

external_auroc_scales: global AUROC versus equal-category macro AUROC. Global includes cross-category comparisons; macro ranks within category and gives15 categories equal weight.

external_structured: Cstructured→S AUROC, generated binary macro F1 and union-box IoU on all1183 anomalies. Invalid/No/missing boxes count zero. Coverage and strict format validity are in provenance.

Cstructured and S use the same schema prompt and80-token budget. S adds training-mask-derived targets, answer tokens and continuation exposures; teacher-map visual inputs remain.

Native Qwen external was not evaluated. Foundation pretraining exposure is unknown. VisA development validation selected checkpoints; three training repeats are not independent test sets or significance evidence.

The plotting script reads the summary only. The upstream summarizer performs raw-prediction recomputation; plotting does not repeat model or raw-prediction audits.

Source summary SHA256: 6ffabc0fd57beaf6c666a9ec984cc69e34ab378de8c3d6cd82b59ba92d4f74f9

Frozen manifest SHA256: 4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a
