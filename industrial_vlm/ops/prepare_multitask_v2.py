"""Freeze matched-source multitask arms and an unopened confirmation set.

This script only reads existing JSON/JSONL metadata. It never opens image files,
loads a model, or generates confirmation predictions. Existing outputs are never
overwritten. Use --dry-run to validate the full preparation without writing it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


SEED = 42
CONFIRMATION_SEED = 20261023
ARMS = ("region_region", "decision_decision", "region_decision")
REGION_PROMPT = (
    'Inspect this industrial image. Return JSON only: {"is_anomalous": true or false, '
    '"boxes": [[x1,y1,x2,y2]]}. Use integer coordinates from 0 to 1000 relative to '
    'the full image. For an anomaly, give one box enclosing all defective pixels; '
    'for a normal image, use an empty boxes list.'
)
DECISION_PROMPT = (
    'Inspect this industrial image for defects. Return JSON only: '
    '{"is_anomalous": true or false}.'
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_ids(rows: list[dict]) -> set[str]:
    return {row["source_id"] for row in rows}


def _counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(
        f"{row['category']}|{'anomaly' if row['is_anomalous'] else 'normal'}"
        for row in rows
    ).items()))


def _sequence_hash(rows: list[dict]) -> str:
    return _sha256_bytes("".join(row["source_id"] + "\n" for row in rows).encode())


def _instruction(record: dict, task: str) -> dict:
    answer = {"is_anomalous": record["is_anomalous"]}
    if task == "region":
        answer["boxes"] = record["boxes"]
    return {
        "id": record["source_id"] + "__" + task,
        "source_id": record["source_id"], "image": record["image"],
        "category": record["category"], "is_anomalous": record["is_anomalous"],
        "anomaly_area_fraction": record.get("anomaly_area_fraction", record.get("mask_area_fraction")),
        "boxes": record["boxes"],
        "prompt": REGION_PROMPT if task == "region" else DECISION_PROMPT,
        "answer": json.dumps(answer, separators=(",", ":")),
    }


def _validate_instruction(row: dict, inventory: dict[str, dict], task: str) -> None:
    source_id = row["source_id"]
    _require(source_id in inventory, f"Instruction source absent from inventory: {source_id}")
    record = inventory[source_id]
    for field in ("image", "category", "is_anomalous", "boxes"):
        _require(row[field] == record[field], f"Instruction/inventory mismatch: {source_id}:{field}")
    expected = _instruction(record, task)
    for field in ("prompt", "answer"):
        _require(row[field] == expected[field], f"Unexpected {task} instruction: {source_id}:{field}")


def build_artifacts(project_root: Path, remote_root: str) -> tuple[dict[str, bytes], dict[str, bytes], dict]:
    """Compute all artifacts and checks in memory, without creating directories."""
    project_root = project_root.resolve()
    visa = project_root / "prepared" / "visa"
    input_paths: set[Path] = set()

    def read(name: str) -> list[dict]:
        path = visa / name
        input_paths.add(path)
        return _read_rows(path)

    manifest = read("manifest.jsonl")
    inventory = {row["source_id"]: row for row in manifest}
    _require(len(inventory) == len(manifest), "Inventory source IDs are not unique")
    _require(len({row["image_sha256"] for row in manifest}) == len(manifest), "Inventory image hashes are not unique")
    split_path = visa / "split.json"
    input_paths.add(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    _require(set(split) == {"train", "val", "test"}, "Unexpected split keys")
    split_sets = {name: set(ids) for name, ids in split.items()}
    _require(all(len(split[name]) == len(ids) for name, ids in split_sets.items()), "Duplicate source in split")
    _require(not (split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"]
                  or split_sets["val"] & split_sets["test"]), "Source split overlap")
    _require(set.union(*split_sets.values()) == set(inventory), "Split/inventory coverage mismatch")
    for name, ids in split_sets.items():
        _require(ids == {row["source_id"] for row in manifest if row["split"] == name}, f"Inventory split mismatch: {name}")
    full_test, full_val = read("test.jsonl"), read("val.jsonl")
    for name, rows in (("test", full_test), ("val", full_val)):
        _require(len(rows) == len(_source_ids(rows)), f"Duplicate instruction source in {name}")
        _require(_source_ids(rows) == split_sets[name], f"Full {name} instruction/split mismatch")
        for row in rows:
            _validate_instruction(row, inventory, "region")

    region, decision = read("train_quality_region.jsonl"), read("train_quality_decision.jsonl")
    _require(len(region) == len(decision) == 768, "Expected exactly 768 source images per original training arm")
    _require(len(_source_ids(region)) == 768, "Original training sources are not unique")
    _require([row["source_id"] for row in region] == [row["source_id"] for row in decision], "Original task source orders differ")
    _require(_source_ids(region) <= split_sets["train"], "Training source outside train split")
    for rows, task in ((region, "region"), (decision, "decision")):
        for row in rows:
            _validate_instruction(row, inventory, task)
    categories = sorted({row["category"] for row in manifest})
    _require(len(categories) == 12, "Expected 12 VisA categories")
    expected_keys = {f"{cat}|{label}" for cat in categories for label in ("normal", "anomaly")}
    _require(set(_counts(region)) == expected_keys and set(_counts(region).values()) == {32}, "Expected 12 categories x 2 labels x 32 training sources")

    val_region = read("val_balanced_120.jsonl")
    _require(len(val_region) == len(_source_ids(val_region)) == 120, "Expected 120 unique validation sources")
    _require(_source_ids(val_region) <= split_sets["val"], "Validation source outside validation split")
    _require(set(_counts(val_region)) == expected_keys and set(_counts(val_region).values()) == {5}, "Validation is not category/label balanced")
    for row in val_region:
        _validate_instruction(row, inventory, "region")
    decision_val_path = visa / "val_balanced_120_decision.jsonl"
    if decision_val_path.exists():
        val_decision = read(decision_val_path.name)
        _require([row["source_id"] for row in val_region] == [row["source_id"] for row in val_decision], "Validation prompt orders differ")
    else:
        val_decision = [_instruction(row, "decision") for row in val_region]
    for row in val_decision:
        _validate_instruction(row, inventory, "decision")

    by_image = {row["image"]: row["image_sha256"] for row in manifest}
    evaluated_ids: set[str] = set()
    evaluated_hashes: set[str] = set()
    prediction_files = sorted((project_root / "evaluations").rglob("predictions.jsonl"))
    _require(bool(prediction_files), "No existing prediction files found; refusing to assume an empty evaluation history")
    evaluation_audit = []
    for path in prediction_files:
        input_paths.add(path)
        rows = _read_rows(path)
        for row in rows:
            # Only source identity metadata participates in selection. Model
            # outputs and outcome fields are deliberately never inspected.
            source_id = row["source_id"]
            evaluated_ids.add(source_id)
            image_hash = row.get("image_sha256")
            if source_id in inventory:
                image_hash = inventory[source_id]["image_sha256"]
            if not image_hash:
                image_hash = by_image.get(row.get("image"))
            if not image_hash:
                match = re.search(r"_([0-9a-f]{64})$", source_id)
                image_hash = match.group(1) if match else None
            _require(isinstance(image_hash, str) and re.fullmatch(r"[0-9a-f]{64}", image_hash) is not None,
                     f"Cannot resolve evaluated image identity from metadata: {path}:{source_id}")
            evaluated_hashes.add(image_hash)
        evaluation_audit.append({"path": str(path.relative_to(project_root)), "rows": len(rows),
                                 "unique_sources": len(_source_ids(rows))})

    source_map = {task: {row["source_id"]: row for row in rows}
                  for task, rows in (("region", region), ("decision", decision))}
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for row in region:
        strata[(row["category"], row["is_anomalous"])].append(row["source_id"])
    first_epoch_task = {}
    for key, ids in sorted(strata.items()):
        ids = sorted(ids)
        random.Random(f"{SEED}:multitask_task_assignment:{key}").shuffle(ids)
        for index, source_id in enumerate(ids):
            first_epoch_task[source_id] = "region" if index < len(ids) // 2 else "decision"
    orders = []
    rng = random.Random(SEED)
    for _ in range(2):
        order = [row["source_id"] for row in region]
        rng.shuffle(order)
        orders.append(order)
    train_arms: dict[str, list[dict]] = {name: [] for name in ARMS}
    for epoch, order in enumerate(orders, 1):
        for source_id in order:
            for arm in ARMS:
                if arm == "region_region":
                    task = "region"
                elif arm == "decision_decision":
                    task = "decision"
                else:
                    task = first_epoch_task[source_id]
                    if epoch == 2:
                        task = "decision" if task == "region" else "region"
                row = deepcopy(source_map[task][source_id])
                row.update(id=f"{source_id}__multitask_v2__{arm}__epoch{epoch}__{task}",
                           task=task, source_epoch=epoch)
                train_arms[arm].append(row)
    reference_sequence = [row["source_id"] for row in train_arms[ARMS[0]]]
    for arm, rows in train_arms.items():
        _require(len(rows) == len({row["id"] for row in rows}) == 1536, f"Invalid exposure rows/IDs: {arm}")
        _require([row["source_id"] for row in rows] == reference_sequence, f"Source exposure order mismatch: {arm}")
        _require(set(Counter(row["source_id"] for row in rows).values()) == {2}, f"Incorrect source repetitions: {arm}")
    mixed = train_arms["region_decision"]
    _require(all(Counter(row["task"] for row in mixed if row["source_id"] == sid) == {"region": 1, "decision": 1}
                 for sid in _source_ids(region)), "Mixed arm must expose both tasks for every source")
    mixed_counts = Counter((row["source_epoch"], row["category"], row["is_anomalous"], row["task"]) for row in mixed)
    _require(len(mixed_counts) == 96 and set(mixed_counts.values()) == {16}, "Mixed task strata are not balanced within each epoch")

    forbidden_ids = split_sets["train"] | split_sets["val"] | evaluated_ids
    forbidden_hashes = {inventory[sid]["image_sha256"] for sid in split_sets["train"] | split_sets["val"]} | evaluated_hashes
    candidates = [row for row in manifest if row["split"] == "test" and row["source_id"] not in forbidden_ids
                  and row["image_sha256"] not in forbidden_hashes]
    confirmation_strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in candidates:
        confirmation_strata[(row["category"], row["is_anomalous"])].append(row)
    confirmation = []
    candidate_counts = {}
    for category in categories:
        for label in (False, True):
            members = sorted(confirmation_strata[(category, label)], key=lambda row: row["source_id"])
            candidate_counts[f"{category}|{'anomaly' if label else 'normal'}"] = len(members)
            _require(len(members) >= 10, f"Fewer than 10 unopened confirmation candidates: {category},{label}")
            random.Random(f"{CONFIRMATION_SEED}:confirmation:{category}:{label}").shuffle(members)
            confirmation.extend(members[:10])
    confirmation = sorted(confirmation, key=lambda row: row["source_id"])
    confirmation_ids = _source_ids(confirmation)
    confirmation_hashes = {row["image_sha256"] for row in confirmation}
    _require(len(confirmation) == len(confirmation_ids) == len(confirmation_hashes) == 240, "Confirmation sources/hashes are not unique")
    _require(not confirmation_ids & forbidden_ids, "Confirmation source leakage")
    _require(not confirmation_hashes & forbidden_hashes, "Confirmation hash leakage")

    prepared = {f"train_{arm}.jsonl": _jsonl_bytes(rows) for arm, rows in train_arms.items()}
    prepared.update({
        "val_region_120.jsonl": _jsonl_bytes(val_region),
        "val_decision_120.jsonl": _jsonl_bytes(val_decision),
        "confirmation_region_240.jsonl": _jsonl_bytes([_instruction(row, "region") for row in confirmation]),
        "confirmation_decision_240.jsonl": _jsonl_bytes([_instruction(row, "decision") for row in confirmation]),
        "confirmation_sources.jsonl": _jsonl_bytes(confirmation),
    })
    remote_root = remote_root.rstrip("/")
    configs = {}
    for arm in ARMS:
        config = {
            "model": remote_root + "/models/Qwen3-VL-8B-Instruct",
            "train_file": remote_root + f"/prepared/multitask_v2/train_{arm}.jsonl",
            "val_file": remote_root + "/prepared/multitask_v2/val_" + ("decision" if arm == "decision_decision" else "region") + "_120.jsonl",
            "image_root": remote_root + "/datasets/VisA",
            "output": remote_root + "/runs/multitask_v2/" + arm,
            "max_steps": 192, "seed": SEED, "sample_order": "sequential",
            "max_pixels": 524288, "max_length": 2048,
            "gradient_accumulation_steps": 8, "learning_rate": 1e-4,
            "warmup_steps": 10, "save_steps": 64, "eval_steps": 64,
            "val_max_samples": 24, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        }
        configs[arm + ".json"] = _json_bytes(config)
    plan = {
        "schema_version": "industrial-vlm-multitask-v2-plan-v1",
        "protocol_date": "2026-10-23",
        "status": "frozen_before_v2_training_and_before_confirmation_predictions",
        "purpose": "Compare repeated region, repeated decision, and paired multitask SFT with matched source exposures",
        "arms": list(ARMS), "unique_training_sources_per_arm": 768,
        "source_exposures_per_arm": 1536, "optimizer_updates_per_arm": 192,
        "gradient_accumulation_steps": 8, "seed": SEED,
        "source_order": "One shared random.Random(42), sequential shuffle of a fresh original 768-source list for each of two epochs; identical exposure sequence in all arms",
        "mixed_task_assignment": "Within each category/label group of 32, fixed seeded shuffle assigns 16 region and 16 decision in epoch1; epoch2 reverses every source task",
        "training_initialization": "Every arm starts from the same fresh quantized base plus freshly initialized LoRA; no adapter field and no previous checkpoint initialization",
        "model_selection": "Fixed final update192; validation CE is diagnostic only, never checkpoint selection by test/confirmation results",
        "validation": {
            "sources": "Existing fixed 120 validation sources, paired region and decision prompts, original order preserved",
            "ce_audit": "steps64/128/192, fixed first24; region_region and region_decision use region prompts, decision_decision uses decision prompts",
            "ce_comparability": "Different target tasks have different lengths and difficulty; CE across task prompts is not a common decision metric",
            "generation_protocol": "Five main evaluations on the same fixed120 validation sources using final192 adapters: region_region under region and decision prompts, region_decision under region and decision prompts, decision_decision under decision prompt",
            "main_evaluations": [
                {"arm": "region_region", "prompt": "region"},
                {"arm": "region_region", "prompt": "decision"},
                {"arm": "region_decision", "prompt": "region"},
                {"arm": "region_decision", "prompt": "decision"},
                {"arm": "decision_decision", "prompt": "decision"},
            ],
            "primary_comparison": "Test region_decision versus region_region for common decision-prompt performance, within-adapter prompt decision consistency, and region-prompt localization tradeoff; no improvement is assumed",
            "decision_decision_region_prompt": "Not a main comparison; the decision-only target arm is evaluated primarily with its decision prompt",
        },
        "confirmation": {
            "sources": 240, "per_category_per_label": 10, "seed": CONFIRMATION_SEED,
            "selection": "Metadata-only category/label sampling from original test split after excluding every previous evaluated source/hash and all train/val sources/hashes",
            "run_this_round": False,
            "policy": "Do not generate/read confirmation outputs this round. Use only after protocol/model selection is frozen in a later final evaluation",
        },
        "fairness": {
            "matched": ["source identity/order/exposure count", "updates", "gradient accumulation", "base model and fresh initialization", "seed", "LoRA configuration", "image pixel and text length limits", "optimizer and schedule"],
            "not_matched": ["supervised target token count", "input prompt token count", "task difficulty"],
            "loss": "Existing assistant-only CE is normalized by actual supervised tokens per accumulated update; region answers can therefore contribute more tokens than decision answers in the mixed arm",
            "report": "Record actual supervised/input token counts, GPU memory, wall time; double-budget v2 versus96-update v1 is descriptive and not a causal multitask control",
        },
        "limits": ["single training seed", "supervised source split", "source SHA256 isolation does not prove physical-part or near-duplicate independence", "public benchmark pretraining contamination is unknown", "union rectangle localization, not instance/pixel segmentation"],
    }
    prepared["plan.json"] = _json_bytes(plan)
    script_path = Path(__file__).resolve()
    input_paths.add(script_path)
    input_audit = []
    for path in sorted(input_paths):
        content = path.read_bytes()
        try:
            name = str(path.relative_to(project_root))
        except ValueError:
            name = str(path)
        input_audit.append({"path": name, "sha256": _sha256_bytes(content), "bytes": len(content)})
    audit = {
        "schema_version": "industrial-vlm-multitask-v2-provenance-v1",
        "inputs": input_audit, "evaluation_history": evaluation_audit,
        "evaluated_unique_sources": len(evaluated_ids), "evaluated_unique_hashes": len(evaluated_hashes),
        "inventory_sources": len(inventory), "source_split_counts": {key: len(ids) for key, ids in split_sets.items()},
        "training_unique_sources": 768, "training_exposure_rows": 1536,
        "source_exposure_sequence_sha256": {arm: _sequence_hash(rows) for arm, rows in train_arms.items()},
        "source_epoch_sequence_sha256": [_sha256_bytes("".join(sid + "\n" for sid in order).encode()) for order in orders],
        "task_exposures": {arm: dict(sorted(Counter(row["task"] for row in rows).items())) for arm, rows in train_arms.items()},
        "mixed_epoch_task_counts": {str(epoch): dict(sorted(Counter(row["task"] for row in mixed if row["source_epoch"] == epoch).items())) for epoch in (1, 2)},
        "confirmation_candidate_counts": dict(sorted(candidate_counts.items())),
        "confirmation_counts": _counts(confirmation),
        "confirmation_source_order_sha256": _sequence_hash(confirmation),
        "checks": {
            "source_exposure_sequence_identical_all_arms": True,
            "row_ids_unique_within_each_arm": True,
            "each_source_exposed_twice_in_each_arm": True,
            "mixed_each_source_has_one_region_and_one_decision": True,
            "mixed_each_category_label_task_has16_sources_per_epoch": True,
            "confirmation_unique_sources_and_hashes": True,
            "confirmation_vs_train_source_overlap": len(confirmation_ids & split_sets["train"]),
            "confirmation_vs_val_source_overlap": len(confirmation_ids & split_sets["val"]),
            "confirmation_vs_evaluated_source_overlap": len(confirmation_ids & evaluated_ids),
            "confirmation_vs_train_val_hash_overlap": len(confirmation_hashes & {inventory[sid]["image_sha256"] for sid in split_sets["train"] | split_sets["val"]}),
            "confirmation_vs_evaluated_hash_overlap": len(confirmation_hashes & evaluated_hashes),
            "raw_image_io_performed": False,
            "confirmation_predictions_generated_or_read": False,
            "model_outcomes_used_for_confirmation_selection": False,
        },
        "prepared_artifact_sha256": {name: _sha256_bytes(content) for name, content in sorted(prepared.items())},
        "config_artifact_sha256": {name: _sha256_bytes(content) for name, content in sorted(configs.items())},
        "hash_scope": "Provenance audit excludes its own hash; all generated data/plan/config artifacts are hashed above",
    }
    prepared["provenance_audit.json"] = _json_bytes(audit)
    return prepared, configs, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-root", default="/root/autodl-tmp/industrial_vlm")
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT_ROOT/prepared/multitask_v2")
    parser.add_argument("--config-dir", type=Path, help="Default: PROJECT_ROOT/configs/multitask_v2")
    parser.add_argument("--dry-run", action="store_true", help="Validate and compute artifacts in memory without any filesystem writes")
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root / "prepared/multitask_v2").expanduser().resolve()
    config_dir = (args.config_dir or project_root / "configs/multitask_v2").expanduser().resolve()
    _require(output_dir != config_dir, "Prepared and configuration directories must differ")
    _require(not output_dir.exists(), f"Refusing to overwrite existing prepared directory: {output_dir}")
    _require(not config_dir.exists(), f"Refusing to overwrite existing configuration directory: {config_dir}")
    prepared, configs, audit = build_artifacts(project_root, args.remote_root)
    if not args.dry_run:
        # All parsing, invariants and sampling complete before any output is
        # created. mkdir(exist_ok=False) also rejects a concurrent writer.
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        config_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
        config_dir.mkdir(exist_ok=False)
        for name, content in prepared.items():
            with (output_dir / name).open("xb") as handle:
                handle.write(content)
        for name, content in configs.items():
            with (config_dir / name).open("xb") as handle:
                handle.write(content)
    print(json.dumps({
        "dry_run": args.dry_run, "prepared_directory": str(output_dir), "config_directory": str(config_dir),
        "prepared_files": len(prepared), "config_files": len(configs),
        "training_exposure_rows_per_arm": audit["training_exposure_rows"],
        "confirmation_rows": sum(audit["confirmation_counts"].values()),
        "confirmation_evaluated": False, "checks": audit["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
