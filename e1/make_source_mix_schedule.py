"""Freeze a single-variable E1 source-mixture schedule.

The default experiment changes only the per-batch source quota from 2/2/2/2
to 3/1/1/3 (real/pseudo anomaly increased; normal/hard-normal reduced).
All rows, seed, donor rule, and 1000-step schedule remain deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1.balanced_sampler import SOURCES
from e1.data import E1TrainingDataset


def parse_quotas(value: str) -> dict[str, int]:
    parsed = {}
    for item in value.split(","):
        key, raw = item.split("=", 1)
        parsed[key.strip()] = int(raw)
    if set(parsed) != set(SOURCES):
        raise ValueError(f"quotas must contain exactly {SOURCES}")
    if any(v <= 0 for v in parsed.values()):
        raise ValueError("all quotas must be positive")
    return parsed


def parse_area_weights(value: str) -> dict[str, float]:
    if not value:
        return {}
    parsed = {}
    for item in value.split("|"):
        key, raw = item.rsplit(":", 1)
        parsed[key.strip()] = float(raw)
    expected = {"tiny<=0.5%", "small0.5-2%", "large>2%"}
    if set(parsed) != expected or any(v <= 0 for v in parsed.values()):
        raise ValueError(f"real-area-weights must contain positive weights for {sorted(expected)}")
    total = sum(parsed.values())
    return {key: value / total for key, value in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/e1/sampler_pool.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1_source_mix_real3_pseudo3"))
    parser.add_argument("--cache-root", type=Path, default=Path("Train448"))
    parser.add_argument("--allowed-schedule", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--quotas", default="real_anomaly=3,normal=1,hard_normal=1,pseudo_anomaly=3")
    parser.add_argument(
        "--real-area-weights", default="",
        help="optional bin weights, e.g. tiny<=0.5%:0.5|small0.5-2%:0.3|large>2%:0.2",
    )
    args = parser.parse_args()
    quotas = parse_quotas(args.quotas)
    real_area_weights = parse_area_weights(args.real_area_weights)
    if sum(quotas.values()) != args.batch_size:
        raise ValueError("source quotas must sum to batch-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(args.pool).reset_index(drop=True)
    allowed = None
    if args.allowed_schedule is not None:
        base = pd.read_parquet(args.allowed_schedule)
        allowed = set(base.image_path.astype(str)) | set(base.donor_path.astype(str))
        allowed.discard("")
        pool = pool[pool.image_path.astype(str).isin(allowed)].reset_index(drop=True)
    # The schedule stores dataset_index values into this exact filtered pool;
    # save it so training cannot accidentally pair the schedule with the
    # unfiltered source pool.
    cells = {}
    real_cells_by_bin = {}
    for source in SOURCES:
        frame = pool[pool.source_type.eq(source)].copy()
        frame = frame[frame.image_path.map(lambda value: (args.cache_root / str(value)).is_file())]
        if source == "real_anomaly":
            frame = frame[frame.mask_path.map(lambda value: (args.cache_root / str(value)).is_file())]
        if frame.empty:
            raise ValueError(f"source pool is empty: {source}")
        cells[source] = [part.index.to_numpy(dtype=np.int64) for _, part in frame.groupby("sampler_cell", sort=True)]
        if source == "real_anomaly" and real_area_weights:
            for area_bin in real_area_weights:
                area_frame = frame[frame.area_train_bin.eq(area_bin)]
                if area_frame.empty:
                    raise ValueError(f"real anomaly area bin is empty: {area_bin}")
                real_cells_by_bin[area_bin] = [
                    part.index.to_numpy(dtype=np.int64)
                    for _, part in area_frame.groupby("sampler_cell", sort=True)
                ]

    dataset = E1TrainingDataset(pool, cache_root=Path("."), global_seed=args.seed)
    rng = np.random.default_rng(args.seed)
    records = []
    for step in range(1, args.steps + 1):
        batch = []
        for source in SOURCES:
            for _ in range(quotas[source]):
                if source == "real_anomaly" and real_area_weights:
                    bins = list(real_area_weights)
                    area_bin = str(rng.choice(bins, p=[real_area_weights[b] for b in bins]))
                    source_cells = real_cells_by_bin[area_bin]
                else:
                    source_cells = cells[source]
                cell = source_cells[int(rng.integers(0, len(source_cells)))]
                batch.append((source, int(cell[rng.integers(0, len(cell))])))
        rng.shuffle(batch)
        for position, (source, index) in enumerate(batch):
            row = pool.iloc[index]
            donor = dataset._donor_path(str(row.image_path)) if source == "pseudo_anomaly" else ""
            records.append({
                "step": step,
                "batch_position": position,
                "dataset_index": index,
                "pool_index": int(row.pool_index),
                "source_type": source,
                "class_name": str(row.class_name),
                "image_path": str(row.image_path),
                "mask_path": str(row.mask_path) if source == "real_anomaly" else "",
                "donor_path": donor,
            })
    schedule = pd.DataFrame(records)
    if len(schedule) != args.steps * args.batch_size:
        raise RuntimeError("wrong schedule length")
    for _, part in schedule.groupby("step", sort=True):
        counts = part.source_type.value_counts().to_dict()
        if any(int(counts.get(source, 0)) != quotas[source] for source in SOURCES):
            raise RuntimeError(f"quota violation: {counts}")
    schedule.to_parquet(args.out_dir / "fold1_schedule.parquet", index=False)
    schedule.to_csv(args.out_dir / "fold1_schedule.csv", index=False)
    pool.to_parquet(args.out_dir / "sampler_pool.parquet", index=False)
    pool.to_csv(args.out_dir / "sampler_pool.csv", index=False)

    required_images = sorted((set(schedule.image_path) | set(schedule.donor_path)) - {""})
    required_masks = sorted(set(schedule.mask_path) - {""})
    missing_images = [p for p in required_images if not (args.cache_root / p).is_file()]
    missing_masks = [p for p in required_masks if not (args.cache_root / p).is_file()]
    if missing_images or missing_masks:
        raise FileNotFoundError(f"missing images={len(missing_images)} masks={len(missing_masks)}")
    pd.DataFrame([{"path": p, "kind": "image"} for p in required_images] + [{"path": p, "kind": "mask"} for p in required_masks]).to_csv(args.out_dir / "required_files.csv", index=False)
    protocol = {
        "protocol": "E1-source-mixture-v1",
        "base_pool": str(args.pool),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "quotas": quotas,
        "draws": len(schedule),
        "source_draws": schedule.source_type.value_counts().sort_index().to_dict(),
        "required_images": len(required_images),
        "required_masks": len(required_masks),
        "single_variable": "real anomaly area-bin weights only" if real_area_weights else "source quota only",
        "real_area_weights": real_area_weights or None,
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
