"""Run one identical, reproducible dev audit for v1--v5.

The script deliberately evaluates every view before applying the classifier
gate.  It therefore produces a valid gate=0 ablation instead of confusing a
missing prediction cache with an all-zero mask.  Each version gets the same
3600-view split, original-resolution full+patch9 fusion, uint8 quantisation,
pooled AUROC/AP/F1max, per-class/area breakdown, per-image records, and a
ranked bad-case list.

Example (after downloading checkpoints)::

  PYTHONPATH=. python e1/evaluate_versions_local.py \
    --model-id models/huggingface/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<id> \
    --checkpoint-dir checkpoints --out artifacts/version_audit_20260923
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


VERSIONS = {
    "v1": "v1.pt",  # patch9_emptymask_12000/step12000
    "v2": "v2.pt",  # patch9_mixed_12000/best
    "v3": "v3.pt",  # patch9_mixed_v3_12000/step12000
    "v4": "v4.pt",  # patch9_mixed_v4_24000/step24000
    "v5": "v5.pt",  # patch9_mixed_v5_24000/step24000
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, default=Path("artifacts/version_audit_20260923"))
    p.add_argument("--model-id", default="models/Qwen3-VL-2B-Instruct")
    p.add_argument("--test", type=Path, default=Path("artifacts/splits/fold1/dev_unbiased.parquet"))
    p.add_argument("--image-root", type=Path, default=Path("Dataset"))
    p.add_argument("--gt-root", type=Path, default=Path("Dataset"))
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    p.add_argument("--only", nargs="*", choices=tuple(VERSIONS), default=list(VERSIONS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not args.test.exists():
        raise SystemExit(f"missing test split: {args.test}; run e1/build_dev_unbiased.py first")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for version in args.only:
        checkpoint = args.checkpoint_dir / VERSIONS[version]
        if not checkpoint.exists():
            raise SystemExit(f"missing {version} checkpoint: {checkpoint}")
        destination = args.out / version
        report = destination / "submission_fusion_metrics.json"
        if report.exists() and not args.force:
            print(f"[{version}] exists; use --force to rerun", flush=True)
        else:
            cmd = [sys.executable, "-m", "e1.evaluate_submission_full_audit",
                   "--model-id", str(args.model_id), "--segmentation-checkpoint", str(checkpoint),
                   "--test", str(args.test), "--image-root", str(args.image_root),
                   "--gt-root", str(args.gt_root), "--out", str(destination),
                   "--fusion-weights", "0.5", "--gate-thresholds", "0.5", "0.0"]
            env = None
            if args.device:
                import os
                env = os.environ.copy(); env["VLM_DEVICE"] = args.device
            print(f"[{version}] running 3600-view audit", flush=True)
            subprocess.run(cmd, check=True, env=env)
        payload = json.loads(report.read_text(encoding="utf-8"))
        for key, metrics in payload["results"].items():
            gate, weight = key.split("|w")
            rows.append({"version": version, "gate": float(gate[4:]), "fusion_weight": float(weight), **metrics})
    table = pd.DataFrame(rows).sort_values(["gate", "pixel_f1max"], ascending=[False, False])
    table.to_csv(args.out / "summary.csv", index=False)
    (args.out / "summary.json").write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
