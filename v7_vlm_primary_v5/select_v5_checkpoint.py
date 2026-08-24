#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(args):
    candidates = []
    # Progress state files intentionally use names such as
    # ``proxy_epoch0.calibration_progress.json``.  They are not proxy metric
    # reports and must never be considered as checkpoints.
    paths = [
        path for path in sorted(args.proxy_dir.glob("proxy_epoch*.json"))
        if re.fullmatch(r"proxy_epoch[0-9]+", path.stem)
    ]
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        epoch = int(path.stem.replace("proxy_epoch", ""))
        # Pixel metrics dominate, but image AP remains a guard against the
        # score-ranking collapse observed in v4.
        selection = (
            0.35 * data["p_ap"]
            + 0.30 * data["p_f1max"]
            + 0.20 * data["p_auroc"]
            + 0.10 * data["i_ap"]
            + 0.05 * data["i_auroc"]
        )
        candidates.append({"epoch": epoch, "selection": selection, "metrics": data, "path": str(path)})
    if not candidates:
        raise RuntimeError(f"no proxy_epoch*.json under {args.proxy_dir}")
    best = max(candidates, key=lambda row: row["selection"])
    result = {"best_epoch": best["epoch"], "best_selection": best["selection"], "candidates": candidates}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
