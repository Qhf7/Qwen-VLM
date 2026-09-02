"""Freeze the exact 1000-step fold1 sampling and pseudo-donor schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from e1.balanced_sampler import BalancedBatchSampler, SOURCES
from e1.data import E1TrainingDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/e1/sampler_pool.parquet"))
    parser.add_argument("--config", type=Path, default=Path("e1/config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(args.pool).reset_index(drop=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(training["seed"])
    batch_size = int(training["logical_batch_size"])
    max_steps = int(training["max_steps"])
    sampler = BalancedBatchSampler(
        pool, batch_size=batch_size, steps_per_epoch=max_steps, seed=seed
    )
    dataset = E1TrainingDataset(pool, cache_root=Path("."), global_seed=seed)
    dataset.set_epoch(0)
    records = []
    for step, indices in enumerate(sampler, 1):
        batch = pool.iloc[indices]
        counts = batch.source_type.value_counts()
        if any(int(counts.get(source, 0)) != batch_size // len(SOURCES) for source in SOURCES):
            raise RuntimeError(f"step {step} violates the four-source quota")
        for position, index in enumerate(indices):
            row = pool.iloc[index]
            donor_path = (
                dataset._donor_path(str(row.image_path))
                if row.source_type == "pseudo_anomaly" else ""
            )
            records.append({
                "step": step,
                "batch_position": position,
                "dataset_index": int(index),
                "pool_index": int(row.pool_index),
                "source_type": str(row.source_type),
                "class_name": str(row.class_name),
                "image_path": str(row.image_path),
                "mask_path": str(row.mask_path) if row.source_type == "real_anomaly" else "",
                "donor_path": donor_path,
            })
    schedule = pd.DataFrame(records)
    if len(schedule) != max_steps * batch_size:
        raise RuntimeError("frozen schedule has the wrong number of draws")
    schedule.to_parquet(args.out_dir / "fold1_schedule.parquet", index=False)
    schedule.to_csv(args.out_dir / "fold1_schedule.csv", index=False)

    required = []
    for path in sorted(set(schedule.image_path) | set(schedule.donor_path) - {""}):
        required.append({"path": path, "kind": "image"})
    for path in sorted(set(schedule.mask_path) - {""}):
        required.append({"path": path, "kind": "mask"})
    required_frame = pd.DataFrame(required)
    required_frame.to_csv(args.out_dir / "fold1_required_files.csv", index=False)
    summary = {
        "protocol": "E1-fold1-fixed-schedule-v1",
        "seed": seed,
        "steps": max_steps,
        "batch_size": batch_size,
        "draws": len(schedule),
        "source_draws": schedule.source_type.value_counts().sort_index().to_dict(),
        "unique_primary_images": int(schedule.image_path.nunique()),
        "unique_donor_images": int(schedule.loc[schedule.donor_path.ne(""), "donor_path"].nunique()),
        "required_images": int((required_frame.kind == "image").sum()),
        "required_masks": int((required_frame.kind == "mask").sum()),
    }
    (args.out_dir / "fold1_schedule_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
