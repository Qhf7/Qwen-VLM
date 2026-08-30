# v12 Region-Ranked Anomaly

v12 is an isolated continuation of v10/v11. It does not overwrite their code,
checkpoints or submission ZIP files.

The core change is a Train-only, rejectable candidate-region ranker. DINOv2
reconstruction, normal memory and pseudo-head maps first generate dense
proposals. Multi-threshold connected components then become candidate regions.
A lightweight MLP ranks candidates using anomaly strength, threshold
persistence, local contrast, cross-branch agreement, area, compactness,
foreground overlap, outline overlap and image-gradient evidence.

The ranker is trained only on official Train normals plus programmatic pseudo
anomalies. About 20% of Train categories are held out completely for a
zero-shot pixel proxy. Another held-out normal subset determines a view-level
rejection threshold. A view is allowed to output an all-black mask; Top-1 is
never forced. At most three independent candidates can survive.

Run:

```bash
DATA_ROOT=/root/autodl-tmp/Dataset \
WEIGHT_DIR=/root/autodl-tmp/weights \
CKPT=/root/autodl-tmp/<completed-branch-checkpoint> \
VLM_AUDIT=/root/autodl-tmp/<optional-local-qwen-audit.jsonl> \
RUN_ROOT=/root/autodl-tmp/v12_region_ranked_run_YYYYMMDD_HHMMSS \
PYTHON=/root/miniconda3/bin/python \
bash run_v12_pipeline.sh
```

Outputs:

```text
submission_v12_region_ranked.zip
submission_v12_region_ranked_hybrid.zip  # only when VLM_AUDIT is set
```

The optional Qwen branch can only confirm already selected DINO regions. It
cannot create a mask on an all-zero region-ranked view, and its pixel and image
residuals are smaller than in v11.

Local checks:

```bash
python3 -m py_compile v12_region_ranked_anomaly/*.py
PYTHONPATH=v12_region_ranked_anomaly python3 v12_region_ranked_anomaly/test_v12.py
PYTHONPATH=v12_region_ranked_anomaly python3 v12_region_ranked_anomaly/test_v10.py
bash -n v12_region_ranked_anomaly/run_v12_pipeline.sh
```
