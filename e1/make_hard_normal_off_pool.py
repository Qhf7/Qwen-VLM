"""Build the E1 hard-normal content ablation pool.

The four sampler slots stay unchanged. Only the image content in the
hard_normal slots is replaced by a deterministic same-class normal image.
This isolates whether the mined hard-normal images are causing the E1
failure without changing batch size, seed, or source quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/e1/sampler_pool.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_hard_normal_off"))
    parser.add_argument("--allowed-schedule", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("Train448"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(args.pool).reset_index(drop=True)
    required = {"source_type", "class_name", "image_path"}
    missing = required - set(pool.columns)
    if missing:
        raise RuntimeError(f"sampler pool missing columns: {sorted(missing)}")

    normal = pool[pool.source_type.eq("normal")].sort_values(
        ["class_name", "image_path"], kind="mergesort"
    )
    allowed = None
    if args.allowed_schedule is not None:
        schedule = pd.read_parquet(args.allowed_schedule)
        allowed = set(schedule.image_path.astype(str)) | set(schedule.donor_path.astype(str))
        allowed.discard("")
        normal = normal[normal.image_path.astype(str).isin(allowed)]
    normal = normal[normal.image_path.map(lambda value: (args.cache_root / str(value)).is_file())]
    hard = pool[pool.source_type.eq("hard_normal")].sort_values(
        ["class_name", "image_path"], kind="mergesort"
    )
    if hard.empty or normal.empty:
        raise RuntimeError("both normal and hard_normal rows are required")

    replacements: dict[str, list[str]] = {
        str(class_name): part.image_path.astype(str).tolist()
        for class_name, part in normal.groupby("class_name", sort=True)
    }
    output = pool.copy()
    audit_rows = []
    for class_name, part in hard.groupby("class_name", sort=True):
        candidates = replacements.get(str(class_name), [])
        if not candidates:
            raise RuntimeError(f"no normal replacement for class {class_name}")
        for offset, (index, row) in enumerate(part.iterrows()):
            replacement = candidates[offset % len(candidates)]
            audit_rows.append({
                "pool_index": int(row.pool_index),
                "class_name": str(class_name),
                "original_image_path": str(row.image_path),
                "replacement_image_path": replacement,
            })
            output.at[index, "image_path"] = replacement
            output.at[index, "hard_normal_content_ablation"] = True

    if "hard_normal_content_ablation" not in output:
        output["hard_normal_content_ablation"] = False
    else:
        output["hard_normal_content_ablation"] = (
            output["hard_normal_content_ablation"].fillna(False).astype(bool)
        )
    output.to_parquet(args.out_dir / "sampler_pool.parquet", index=False)
    output.to_csv(args.out_dir / "sampler_pool.csv", index=False)
    audit = pd.DataFrame(audit_rows).sort_values("pool_index", kind="mergesort")
    audit.to_csv(args.out_dir / "replacement_audit.csv", index=False)
    protocol = {
        "protocol": "E1-hard-normal-content-off-v1",
        "base_pool": str(args.pool),
        "source_quota_unchanged": True,
        "source_type_unchanged": True,
        "replacement_rule": "same-class normal image, sorted deterministic cyclic mapping",
        "allowed_schedule": str(args.allowed_schedule) if args.allowed_schedule else None,
        "hard_normal_rows": int(len(hard)),
        "replacement_rows": int(len(audit)),
        "class_counts": audit.class_name.value_counts().sort_index().to_dict(),
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
