# External saved-result audit — partial snapshot

Audit scope: read-only CPU recomputation of the downloaded saved external
artifacts. No SSH, GPU/model invocation, data staging, RGB/mask access, checkpoint
selection, prompt edits, threshold changes or hyperparameter changes.

The local snapshot reports the external queue running R seed42. The following
completed conditions passed full raw-prediction recomputation:

| Condition | Seed | Saved predictions | Audit |
|---|---:|---:|---|
| A | 42 | 1575 | pass |
| C | 42 | 1575 | pass |
| D | 42 | 1575 | pass |
| Cpair | 42 | 1575 | pass |

Overall audit state is **partial:4/21**. There are zero invalid completed runs in
this snapshot. The remaining17 conditions are missing/incomplete locally and
are excluded from all completed-run denominators. R42 has a saved generation
policy and progress only; it is not counted as completed.

For each completed condition the summarizer checked the frozen manifest's exact
source order, unique source IDs, all1575 scores/generations, prediction SHA,
recomputed overall/per-category metrics, saved metrics equality, selected
checkpoint step/SHA, shared external cache SHA, and generation-policy evidence.
The four saved policies select batched decoding after their existing VisA-only
engineering gate. This states implementation provenance, not model quality.

All nine code source SHA256 values in the immutable external selection freeze
match the current local snapshot: basic evaluator, structured evaluator,
migration language, structured model/geometry, migration metrics, advanced
trainer/generation and external cache builder. No source was changed by this
audit.

The CPU command wrote audit outputs outside the result tree:

    python3 migration_20261103/advanced_20261104/summarize_external.py --include-structured --out /tmp/vlm_external_partial_audit.json --markdown-out /tmp/vlm_external_partial_audit.md

No completed-result contract mismatch or summarizer runtime blocker was found.
This note draws no model comparison, generalization or significance conclusion
from the partial snapshot. Check all21 conditions and three matched seeds before
the final external report.

## Snapshot identities

Frozen manifest SHA256: 4cf2709614c8fd6aabe1268ef3c9f8687a07a0849d3d0f0341d1756570f9497a.

| Condition/seed | summary SHA256 | predictions SHA256 |
|---|---|---|
| A/42 | 0f11e89d70cad26266092014154ec6f1ef7a7e946b0d53613485857d595bb4e3 | 716c8207365d2c33d1fc9bd42ae839b157c5b8e4e44693cf396cb4933f2e0018 |
| C/42 | cb3a6af273de7537bbc053d512496e9b47fdeade0d2cdf217cc91db79e33a1f5 | 5212f9b987958893d0cc16044793aad3ac1c70d59fd70142a872460f7759a9db |
| D/42 | 1fd7cb338ade9c877e8c2b3baa58c3ea3cf4dd6ec47afedd3692b6b8a751a5f0 | 6b4fd9dc0ff0ea3599ca36362e977d54ee0c21925ef3783c56eca3907bc2c770 |
| Cpair/42 | 68bbe55919fc6e9e83806f0617d28648f65906f719ba2e4f1e6f0064f5355639 | 3ae564ee1921e220b39fcdc45f8cbcaee1e65ec141907f33928c9c7a2a58c814 |

The audit JSON SHA256 is f1cb5d01b5cc00d27e3753b0e35a53bf5339f25281e3a9c49308fb83a53285fe.
