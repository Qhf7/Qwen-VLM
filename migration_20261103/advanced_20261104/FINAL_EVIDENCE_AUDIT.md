# Final saved evidence audit

State: **passed**. CPU saved-artifact audit; no model, GPU, SSH, RGB, masks, tensor weights or caches are read.

- advanced: passed
- structured: passed
- tool: passed
- external: passed

Coverage: D/Cpair/R 9 runs, 18 recomputed validation files and 18 fresh reports; S 3 runs, 9 recomputed validation files and 6 fresh104 reports; tool preflight8 and validation380 with five conditions and completed posterior SHA audits; external21 ×1575 with six identical target artifacts.
The complete tool result remains valid with zero model-selected crops; audit success does not imply a positive performance result.
Existing completed fresh reports provide model reconstruction evidence. This script verifies saved report/source/policy bindings and never reruns a model.

| Model | Seeds | Local checkpoint bytes verified | Saved server attestation |
|---|---|---|---|
| A | 42,43,44 | False,False,False | True,True,True |
| C | 42,43,44 | False,False,False | True,True,True |
| D | 42,43,44 | False,False,False | True,True,True |
| Cpair | 42,43,44 | False,False,False | True,True,True |
| R | 42,43,44 | False,False,False | True,True,True |
| Cstructured | 42,43,44 | False,False,False | True,True,True |
| S | 42,43,44 | False,False,False | True,True,True |

These booleans describe different execution environments. Saved True values are server attestations; this audit verifies local checkpoint absence and the exact frozen checkpoint SHA binding. It does not verify tensor bytes.

| Source recovery | Seeds | Source and verification |
|---|---|---|
| A parameter count | 42,43,44 | Local original lm_qwen run.json; prompt plus CLS parameter decomposition, seed/backend/cache/decoder, and exact server-summary equality |
| Cstructured training cost | 42,43,44 | Exact pre-external frozen original C run/complete JSON; distinct from S continuation cost |

The original A bridge run.json SHA was not pre-bound in the external selection freeze. Its local SHA is recorded; this metadata has weaker provenance than the frozen original C source. All recovered numbers are compared exactly within the existing numeric tolerance.
Only run/output path strings under the two known local/server roots are normalized. Every relative path must match the expected model and seed; unknown roots and illegal paths are rejected.

Audit script SHA: `bd96be8a84d15a97bb76a766c850ece2397f863ee9b2e20eecfa6e5793cb8ff4`. All input summary/source hashes are in the JSON.
