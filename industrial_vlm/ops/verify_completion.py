"""Verify the fixed pilot's artifacts after training and evaluation complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from industrial_vlm.metrics import compute_metrics
    from industrial_vlm.compare import _load_run

    checks = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    check(read(root / "pipeline_state.json")["status"] == "complete", "pipeline complete")
    splits = read(root / "prepared/visa/split.json")
    sets = {key: set(splits[key]) for key in ("train", "val", "test")}
    check(not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or
               sets["val"] & sets["test"]), "VisA source splits disjoint")
    visa_hashes = {row["image_sha256"] for row in rows(root / "prepared/visa/manifest.jsonl")}
    external_hashes = {row["image_sha256"] for row in rows(root / "prepared/mvtec/external_manifest.jsonl")}
    check(not visa_hashes & external_hashes, "VisA / external original image SHA256 disjoint")
    training_sources = set()
    training = {}
    for name in ("quality_region", "random_region", "quality_decision"):
        directory = root / "runs" / name
        summary, config = read(directory / "summary.json"), read(directory / "run_config.json")
        check(summary["status"] == "complete" and summary["optimizer_steps"] == 96 and
              summary["microsteps"] == 768, f"{name}: fixed 96 updates / 768 exposures")
        check(summary["reload_check"]["passed"], f"{name}: saved adapter fresh-base reload passed")
        check(sha(Path(config["train_file"])) == config["train_sha256"], f"{name}: training-data hash matches")
        source_ids = {row["source_id"] for row in rows(Path(config["train_file"]))}
        check(len(source_ids) == 768 and source_ids <= sets["train"], f"{name}: 768 distinct train-only sources")
        training_sources |= source_ids
        events = rows(directory / "metrics.jsonl")
        updates = [event for event in events if event.get("event") == "train"]
        check([event["optimizer_step"] for event in updates] == list(range(1, 97)) and
              all(math.isfinite(event["loss"]) for event in updates), f"{name}: complete finite loss history")
        training[name] = summary

    expected = {name: 120 for name in ("base_region", "quality_region", "random_region",
                "base_decision", "quality_region_decision_prompt", "quality_decision")}
    expected.update(external_base_region=150, external_quality_region=150)
    evaluations = {}
    for name, n in expected.items():
        run, error = _load_run(root / "evaluations", name)
        check(run is not None, f"{name}: completed evaluation ({error or 'valid'})")
        prediction_rows = run["rows"]
        check(len(prediction_rows) == n and len({r["source_id"] for r in prediction_rows}) == n,
              f"{name}: expected {n} independent source rows")
        check(not ({r["source_id"] for r in prediction_rows} & training_sources), f"{name}: no SFT source overlap")
        if not name.startswith("external_"):
            check({r["source_id"] for r in prediction_rows} <= sets["test"], f"{name}: VisA test sources only")
        metadata = run["config"]
        if metadata.get("adapter"):
            directory = Path(metadata["adapter"])
            actual = {p.name: sha(p) for p in sorted(directory.iterdir())
                      if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"}}
            check(actual == metadata["adapter_files_sha256"], f"{name}: adapter file hashes match evaluated bytes")
        metrics = read(root / "evaluations" / name / "metrics.json")
        check(compute_metrics(prediction_rows)["overall"] == metrics["overall"], f"{name}: metrics recomputed from archived predictions")
        evaluations[name] = {"n": n, "predictions_sha256": sha(root / "evaluations" / name / "predictions.jsonl")}

    diagnostic = root / "evaluations/overfit32_training_diagnostic"
    check(read(diagnostic / "meta.json")["status"] == "complete" and
          len(rows(diagnostic / "predictions.jsonl")) == 32, "separate 32-source training-fit diagnostic complete")
    paired = read(root / "reports/paired_difference.json")
    check(paired["status"] == "complete" and len(paired["comparisons"]) == 4, "four planned paired comparisons complete")
    check(not read(root / "reports/figures/figure_data.json")["warnings"], "all result figures have complete matched inputs")
    output = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "training": training, "evaluations": evaluations,
              "scope": "Artifact/data/identity/metric consistency; not an independent retraining or physical-part isolation proof."}
    path = root / "ops/completion_verification.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "n_checks": len(checks), "output": str(path)}))


if __name__ == "__main__":
    main()
