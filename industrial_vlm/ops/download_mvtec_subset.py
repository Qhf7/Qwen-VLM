"""Fetch a predeclared external pilot from a revision-pinned public MVTec mirror.

Selection uses paths and labels only, never model predictions. This is not the
complete MVTec AD test set and must be reported as an external pilot subset.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(url):
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60, context=context) as response:
                    return response.read()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))

    def tree(category):
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{category}?recursive=true&limit=1000"
        entries = json.loads(read(url))
        return category, {r["path"]: r for r in entries if r["type"] == "file"}

    selected = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for category, entries in executor.map(tree, CATEGORIES):
            normal = sorted(p for p in entries if "/test/good/" in p and p.endswith(".png"))
            anomaly = sorted(p for p in entries if "/test/" in p and "/good/" not in p and p.endswith(".png"))
            rng = random.Random(f"{args.seed}:{category}")
            for is_anomalous, pool in [(False, normal), (True, anomaly)]:
                for path in sorted(rng.sample(pool, min(args.per_label, len(pool)))):
                    selected.append({**entries[path], "category": category, "is_anomalous": is_anomalous})
                    if is_anomalous:
                        mask = path.replace("/test/", "/ground_truth/").removesuffix(".png") + "_mask.png"
                        selected.append({**entries[mask], "category": category, "is_mask": True})
    for path in ["README.md", "license.txt", "readme.txt"]:
        selected.append({"path": path})

    def download(entry):
        path = args.output / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("oid")
        if not path.exists() or (expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/" + urllib.parse.quote(entry["path"])
            path.write_bytes(read(url))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise RuntimeError(f"Mirror LFS hash mismatch: {path}")
        return {**entry, "downloaded_bytes": path.stat().st_size, "sha256": digest}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, item in enumerate(executor.map(download, selected), 1):
            records.append(item)
            if i % 20 == 0:
                print(json.dumps({"downloaded": i, "total": len(selected)}), flush=True)
    provenance = {"repo": REPO, "revision": REVISION, "seed": args.seed,
                  "per_category_per_label": args.per_label,
                  "protocol": "Fixed label-stratified pilot; no image inspection or model-dependent selection.",
                  "scope": "Partial MVTec AD official test images via public mirror, not the full benchmark.",
                  "n_images": sum("is_anomalous" in r for r in records), "files": records}
    (args.output / "subset_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
