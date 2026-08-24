#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def selection_score(data):
    quality = (
        0.40 * data["p_ap"] + 0.32 * data["p_f1max"] + 0.12 * data["p_auroc"]
        + 0.11 * data["i_ap"] + 0.05 * data["i_auroc"]
    )
    return quality - float(data.get("selection_penalty", 0.0))


def main(args):
    candidates = []
    for path in sorted(args.proxy_dir.glob("proxy_epoch*.json")):
        if not re.fullmatch(r"proxy_epoch[0-9]+", path.stem):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates.append({
            "epoch": int(path.stem.replace("proxy_epoch", "")),
            "selection": selection_score(data), "metrics": data, "path": str(path),
        })
    if not candidates:
        raise RuntimeError(f"no proxy metric files under {args.proxy_dir}")
    best = max(candidates, key=lambda x: x["selection"])
    result = {"best_epoch": best["epoch"], "best_selection": best["selection"], "candidates": candidates}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--proxy-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
