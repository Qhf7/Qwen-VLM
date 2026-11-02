"""Freeze train-only normal curation and a matched high-resolution control.

Both phases are standard-library metadata operations: no image is opened and no
model is loaded. Repeated calls verify existing bytes and never overwrite them.
Run init before mining, then select after the base scorer has completed all1152
candidates. Training/evaluation are deliberately separate executables.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

SELECTION_SEED = 20261023
TRAIN_SEED = 42
REMOTE_ROOT = "/root/autodl-tmp/industrial_vlm"
SCHEMA = "industrial-vlm-curation-v3-v1"
SCORE_SCHEMA = "industrial-vlm-boolean-score-v1"
ARMS = ("normal_random", "normal_hard", "region_highres")
DECISION_PROMPT = (
    'Inspect this industrial image for defects. Return JSON only: '
    '{"is_anomalous": true or false}.'
)
REGION_PROMPT = (
    'Inspect this industrial image. Return JSON only: {"is_anomalous": true or false, '
    '"boxes": [[x1,y1,x2,y2]]}. Use integer coordinates from 0 to 1000 relative to '
    'the full image. For an anomaly, give one box enclosing all defective pixels; '
    'for a normal image, use an empty boxes list.'
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def lbytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                   for row in rows).encode()


def rows(path: Path) -> list[dict]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in result), f"Expected JSON objects: {path}")
    return result


def dump_new(root: Path, artifacts: dict[str, bytes], dry_run: bool) -> dict:
    # Detect every conflict before making any write. Existing exact artifacts
    # are verified, not rewritten; missing files are created with exclusive open.
    existing = []
    for name, content in artifacts.items():
        path = root / name
        if path.exists():
            require(path.is_file() and path.read_bytes() == content,
                    f"Refusing to overwrite changed existing artifact: {path}")
            existing.append(name)
    if not dry_run:
        for name, content in artifacts.items():
            if name in existing:
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
    return {"verified_existing": len(existing), "created": 0 if dry_run else len(artifacts) - len(existing),
            "dry_run": dry_run, "artifact_sha256": {k: sha(v) for k, v in artifacts.items()}}


def instruction(record: dict, identifier: str) -> dict:
    return {"id": identifier, "source_id": record["source_id"],
            "image": record["image"], "image_sha256": record["image_sha256"],
            "category": record["category"], "is_anomalous": record["is_anomalous"],
            "anomaly_area_fraction": record.get("anomaly_area_fraction", record.get("mask_area_fraction")),
            "boxes": record["boxes"], "prompt": DECISION_PROMPT,
            "answer": json.dumps({"is_anomalous": record["is_anomalous"]}, separators=(",", ":"))}


def source_sequence(data: list[dict]) -> str:
    return sha("".join(r["source_id"] + "\n" for r in data).encode())


def validate_reference(data: list[dict], inventory: dict[str, dict], task: str) -> None:
    for row in data:
        sid = row["source_id"]
        require(sid in inventory, f"Unknown source: {sid}")
        record = inventory[sid]
        for field in ("image", "category", "is_anomalous", "boxes"):
            require(row[field] == record[field], f"Reference inventory mismatch: {sid}:{field}")
        require(type(row["is_anomalous"]) is bool, f"Nonboolean label: {sid}")
        expected = {"is_anomalous": row["is_anomalous"]}
        if task == "region":
            expected["boxes"] = row["boxes"]
        require(json.loads(row["answer"]) == expected, f"Reference target mismatch: {sid}")
        require(row["prompt"] == (REGION_PROMPT if task == "region" else DECISION_PROMPT),
                f"Reference prompt mismatch: {sid}")


def init_artifacts(root: Path, remote_root: str) -> tuple[dict[str, bytes], dict]:
    paths = ["prepared/visa/manifest.jsonl", "prepared/visa/split.json",
             "prepared/multitask_v2/train_decision_decision.jsonl",
             "prepared/multitask_v2/train_region_region.jsonl",
             "prepared/multitask_v2/val_decision_120.jsonl",
             "prepared/multitask_v2/val_region_120.jsonl",
             "prepared/multitask_v2/confirmation_sources.jsonl",
             "configs/multitask_v2/decision_decision.json",
             "configs/multitask_v2/region_region.json",
             "evaluations/multitask_v2_scores/base_validation/meta.json",
             "src/industrial_vlm/modeling.py", "src/industrial_vlm/data.py",
             "src/industrial_vlm/score.py", "ops/prepare_curation_v3.py"]
    raw = {name: (root / name).read_bytes() for name in paths}
    manifest = rows(root / paths[0])
    inventory = {r["source_id"]: r for r in manifest}
    require(len(inventory) == len(manifest), "Duplicate source IDs in inventory")
    require(len({r["image_sha256"] for r in manifest}) == len(manifest), "Duplicate image hashes in inventory")
    for record in manifest:
        require(type(record["is_anomalous"]) is bool, "Inventory labels must be booleans")
        require(re.fullmatch(r"[a-f0-9]{64}", record["image_sha256"]) is not None, "Invalid source SHA256")
        require(record["source_id"] == "visa_" + record["image_sha256"], "Source ID does not bind image hash")
    split = json.loads(raw[paths[1]])
    require(set(split) == {"train", "val", "test"}, "Unexpected split keys")
    for part, ids in split.items():
        require(len(ids) == len(set(ids)), f"Duplicate split source: {part}")
        require(set(ids) == {r["source_id"] for r in manifest if r["split"] == part},
                f"Inventory/split mismatch: {part}")
    require(sum(len(v) for v in split.values()) == len(inventory), "Split coverage mismatch")
    dd, rr = rows(root / paths[2]), rows(root / paths[3])
    validate_reference(dd, inventory, "decision")
    validate_reference(rr, inventory, "region")
    require(len(dd) == len(rr) == 1536, "Expected1536 v2 exposure rows")
    require([r["source_id"] for r in dd] == [r["source_id"] for r in rr], "V2 schedules differ")
    require(len({r["source_id"] for r in dd}) == 768, "Expected768 reference sources")
    require(set(Counter(r["source_id"] for r in dd).values()) == {2}, "Reference source repetition mismatch")
    require(all(inventory[r["source_id"]]["split"] == "train" for r in dd), "Reference training leakage")
    categories = sorted({r["category"] for r in manifest})
    require(len(categories) == 12, "Expected12 categories")
    require(Counter((r["category"], r["is_anomalous"]) for r in dd) ==
            Counter({(c, label): 64 for c in categories for label in (False, True)}), "Reference strata mismatch")
    for epoch in (1, 2):
        part = [r for r in dd if r["source_epoch"] == epoch]
        require(len(part) == len({r["source_id"] for r in part}) == 768, "Reference epoch incomplete")
    val_d, val_r = rows(root / paths[4]), rows(root / paths[5])
    validate_reference(val_d, inventory, "decision")
    validate_reference(val_r, inventory, "region")
    require(len(val_d) == len(val_r) == 120, "Expected120 validation sources")
    require([r["source_id"] for r in val_d] == [r["source_id"] for r in val_r], "Validation order mismatch")
    require(all(inventory[r["source_id"]]["split"] == "val" for r in val_d), "Validation split mismatch")
    # Only source IDs are used from the sealed confirmation manifest. No image,
    # mask, label or prediction is accessed for preparation or selection.
    confirmation_ids = {r["source_id"] for r in rows(root / paths[6])}
    require(len(confirmation_ids) == 240, "Expected240 sealed confirmation IDs")
    require(confirmation_ids <= set(split["test"]), "Confirmation source outside test")
    candidates = []
    for category in categories:
        pool = sorted([r for r in manifest if r["split"] == "train" and
                       r["category"] == category and r["is_anomalous"] is False],
                      key=lambda r: r["source_id"])
        require(len(pool) >= 96, f"Too few training normals: {category}")
        selected = random.Random(f"{SELECTION_SEED}:candidate_pool:{category}").sample(pool, 96)
        for record in sorted(selected, key=lambda r: r["source_id"]):
            candidate = instruction(record, record["source_id"] + "__curation_v3__mining_decision")
            candidate["split"] = "train"
            candidates.append(candidate)
    require(len(candidates) == len({r["source_id"] for r in candidates}) == 1152, "Invalid mining pool")
    require(not ({r["source_id"] for r in candidates} & confirmation_ids), "Confirmation mining overlap")
    base_meta = json.loads(raw[paths[9]])
    require(base_meta["status"] == "complete" and base_meta["config"]["adapter"] is None, "Invalid base reference metadata")
    base_files = base_meta["config"]["score_model_contract"]["base_files_sha256"]
    anomaly_ids = sorted({r["source_id"] for r in dd if r["is_anomalous"]})
    require(len(anomaly_ids) == 384, "Expected384 fixed training anomaly sources")
    anomaly_areas = sorted(float(inventory[sid]["mask_area_fraction"]) for sid in anomaly_ids)
    require(all(math.isfinite(v) and 0 < v <= 1 for v in anomaly_areas), "Invalid training anomaly areas")
    q_position = (len(anomaly_areas) - 1) * 0.25
    q_left, q_right = math.floor(q_position), math.ceil(q_position)
    small_threshold = anomaly_areas[q_left] + (anomaly_areas[q_right] - anomaly_areas[q_left]) * (q_position - q_left)
    plan = {"schema_version": SCHEMA, "remote_root": remote_root,
            "selection_seed": SELECTION_SEED, "training_seed": TRAIN_SEED,
            "mining": {"source_split": "train", "label": False, "candidates_per_category": 96,
                       "n_candidates": 1152, "model": f"{remote_root}/models/Qwen3-VL-8B-Instruct",
                       "adapter": None, "prompt": DECISION_PROMPT, "max_pixels": 524288,
                       "max_length": 2048, "data_role": "train_mining",
                       "score_output": "evaluations/curation_v3/mining_base_train_normal",
                       "base_files_sha256": base_files,
                       "rule": "Base model boolean log likelihood odds; no validation/test outcomes or adapters participate"},
            "arms": {"normal_random": "Fresh DD: random32/category from same96 normal pool",
                     "normal_hard": "Fresh DD:16 highest base log odds + fixed-seed random16 from remaining80/category",
                     "region_highres": "Exact v2 RR exposure manifest bytes/order; max_pixels1048576 only input-budget change"},
            "training": {"unique_sources": 768, "normal_sources": 384, "anomaly_sources": 384,
                         "anomalies": "Fixed v2 DD384 anomaly identities and original schedule",
                         "exposure_rows": 1536, "epochs": 2, "updates": 192,
                         "gradient_accumulation_steps": 8, "sample_order": "sequential",
                         "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                         "max_length": 2048, "fresh_base_and_lora": True},
            "fairness": {"matched": ["category-label schedule", "anomaly identity/order",
                         "normal count", "exposures", "updates", "optimizer", "initialization seed"],
                         "not_matched": ["normal source identity", "normal visual content",
                         "actual token counts/runtime"],
                         "training_audit_required": "Verify actual initial trainable parameter hashes and exposure logs across arms",
                         "highres": "Paired to existing v2 RR at524288; same source/order/config except budget and output paths"},
            "evaluation": {"validation_sources": 120, "reuse": "Exact frozen v2 validation bytes, development comparison",
                           "small_area_threshold_train_anomaly_q1": small_threshold,
                           "small_area_threshold_training_sources": 384,
                           "small_area_threshold_source_ids_sha256": canonical(anomaly_ids),
                           "small_area_q1_definition": "data._quantile linear interpolation: position=(n-1)*0.25; ordered[floor(position)] + (ordered[ceil(position)]-ordered[floor(position)])*(position-floor(position)); fixed384 unique v2 DD train anomalies only; small iff area_fraction <= threshold",
                           "confirmation_sources": 240, "confirmation_run_this_round": False,
                           "confirmation_policy": "Read IDs for isolation only; no image/model/prediction access"},
            "limits": ["single training seed", "normal labels assumed correct; no manual defect review",
                       "public benchmark contamination unknown", "source hashes do not establish physical-part independence",
                       "hard means model-confusing, not necessarily ambiguous or mislabeled",
                       "v2 validation has already informed this development direction"]}
    artifacts = {"prepared/curation_v3/mining_normal_candidates.jsonl": lbytes(candidates),
                 "prepared/curation_v3/plan.json": jbytes(plan),
                 "prepared/curation_v3/val_decision_120.jsonl": raw[paths[4]],
                 "prepared/curation_v3/val_region_120.jsonl": raw[paths[5]]}
    audit = {"schema_version": SCHEMA, "phase": "init",
             "inputs": {name: sha(content) for name, content in sorted(raw.items())},
             "artifacts": {name: sha(content) for name, content in sorted(artifacts.items())},
             "candidate_counts": dict(Counter(r["category"] for r in candidates)),
             "candidate_source_sequence_sha256": source_sequence(candidates),
             "candidate_source_hashes": {r["source_id"]: r["image_sha256"] for r in candidates},
             "confirmation_ids_sha256": canonical(sorted(confirmation_ids)),
             "checks": {"train_only_normals": True, "unique_sources_and_hashes": True,
                        "confirmation_id_overlap": 0, "raw_image_io_performed": False,
                        "confirmation_predictions_read_or_generated": False}}
    artifacts["prepared/curation_v3/init_audit.json"] = jbytes(audit)
    return artifacts, {"inventory": inventory, "categories": categories, "candidates": candidates,
                       "dd": dd, "rr": rr, "plan": plan, "audit": audit}


def normalized(log_true: float, log_false: float) -> dict[str, float]:
    require(all(type(v) in (float, int) and math.isfinite(v) and v <= 1e-7
                for v in (log_true, log_false)), "Invalid raw log likelihoods")
    margin = log_true - log_false
    require(math.isfinite(margin), "Nonfinite score margin")
    correction = math.log1p(math.exp(-abs(margin)))
    a, b = min(margin, 0.0) - correction, min(-margin, 0.0) - correction
    return {"anomaly_score": math.exp(a), "log_normalized_true": a,
            "log_normalized_false": b, "anomaly_log_odds": margin}


def read_mining(root: Path, state: dict) -> tuple[dict[str, dict], dict]:
    directory = root / state["plan"]["mining"]["score_output"]
    score_path, meta_path = directory / "scores.jsonl", directory / "meta.json"
    meta = json.loads(meta_path.read_text())
    require(meta.get("schema_version") == SCORE_SCHEMA and meta.get("status") == "complete",
            "Mining scores must have complete native scorer metadata")
    require(meta.get("scores_sha256") == sha(score_path.read_bytes()), "Completed mining scores hash mismatch")
    data = state["candidates"]
    config = meta["config"]
    plan = state["plan"]["mining"]
    require(config.get("n_expected") == meta.get("n_completed") == len(data) == 1152, "Mining count mismatch")
    require(config.get("adapter") is None and config.get("model") == plan["model"], "Mining must use the unchanged base, no adapter")
    require(config.get("data_role") == "train_mining", "Mining role must be explicitly train_mining")
    require(config.get("limit") is None and config.get("calibration_sha256") is None, "Mining cannot use a limit/calibration")
    require(config.get("data") == state["plan"]["remote_root"] + "/prepared/curation_v3/mining_normal_candidates.jsonl", "Unexpected mining input path")
    require(config.get("data_sha256") == sha((root / "prepared/curation_v3/mining_normal_candidates.jsonl").read_bytes()), "Mining input bytes changed")
    require(config.get("subset_sha256") == canonical([(r["id"], r["source_id"], r["prompt"]) for r in data]), "Mining subset hash mismatch")
    contract = dict(config["score_model_contract"])
    contract_hash = contract.pop("contract_sha256", None)
    require(canonical(contract) == contract_hash, "Mining model contract hash mismatch")
    for field, expected in {"schema_version": SCORE_SCHEMA, "prompt": DECISION_PROMPT,
                            "json_prefix": '{"is_anomalous":', "candidates": ["true", "false"],
                            "adapter_files_sha256": None, "max_pixels": 524288, "max_length": 2048,
                            "base_files_sha256": plan["base_files_sha256"],
                            "likelihood": "sum log p(boolean token | native image, prompt, common JSON prefix, preceding boolean tokens)",
                            "normalization": "two-option log-softmax; no length normalization; not calibrated confidence"}.items():
        require(contract.get(field) == expected, f"Unexpected mining scorer contract: {field}")
    for name in ("modeling.py", "data.py", "score.py"):
        require(contract["code_sha256"].get(name) == state["audit"]["inputs"]["src/industrial_vlm/" + name],
                f"Mining code differs from frozen input: {name}")
    expected_hashes = {r["id"]: r["image_sha256"] for r in data}
    require(config.get("image_sha256_by_id") == expected_hashes, "Mining image hash inventory mismatch")
    expected = {r["id"]: r for r in data}
    result = {}
    scored = rows(score_path)
    require(len(scored) == len(data), "Incomplete/extra mining score rows")
    for score in scored:
        rid = score.get("id")
        require(rid in expected and rid not in result, "Unknown/duplicate mining score ID")
        source = expected[rid]
        for field, value in {"schema_version": SCORE_SCHEMA, "source_id": source["source_id"],
                             "image": source["image"], "image_sha256": source["image_sha256"],
                             "category": source["category"], "label": False, "prompt": DECISION_PROMPT}.items():
            require(score.get(field) == value, f"Mining score provenance mismatch: {rid}:{field}")
        require(type(score["label"]) is bool, "Mining labels must be boolean")
        recomputed = normalized(score["log_likelihood_true"], score["log_likelihood_false"])
        for field, value in recomputed.items():
            require(type(score.get(field)) in (int, float) and math.isfinite(score[field]) and
                    math.isclose(score[field], value, rel_tol=1e-10, abs_tol=1e-12),
                    f"Mining derived score mismatch: {rid}:{field}")
        result[rid] = score
    require(set(result) == set(expected), "Mining source coverage mismatch")
    return {r["source_id"]: r for r in result.values()}, {"score_sha256": sha(score_path.read_bytes()),
            "meta_sha256": sha(meta_path.read_bytes()), "contract_sha256": contract_hash}


def distribution(values: list[float]) -> dict:
    values = sorted(values)
    require(bool(values), "Cannot summarize empty scores")
    return {"n": len(values), "min": values[0], "q25": values[(len(values)-1)//4],
            "median": values[(len(values)-1)//2], "q75": values[3*(len(values)-1)//4],
            "max": values[-1], "mean": sum(values)/len(values),
            "quantile_definition": "lower empirical order statistic, zero-based floor(p*(n-1))"}


def select_artifacts(root: Path, state: dict) -> tuple[dict[str, bytes], dict]:
    scores, score_audit = read_mining(root, state)
    inventory, dd = state["inventory"], state["dd"]
    selected = {"normal_random": {}, "normal_hard": {}}
    category_audit = {}
    for category in state["categories"]:
        pool = sorted(r["source_id"] for r in state["candidates"] if r["category"] == category)
        random_ids = random.Random(f"{SELECTION_SEED}:normal_random:{category}").sample(pool, 32)
        ranked = sorted(pool, key=lambda sid: (-scores[sid]["anomaly_log_odds"], sid))
        highest = ranked[:16]
        rest = sorted(set(pool) - set(highest))
        random_rest = random.Random(f"{SELECTION_SEED}:normal_hard_remainder:{category}").sample(rest, 16)
        hard_ids = highest + random_rest
        for arm, ids in (("normal_random", random_ids), ("normal_hard", hard_ids)):
            require(len(ids) == len(set(ids)) == 32, f"Invalid normal selection: {arm}:{category}")
            ids = sorted(ids)
            random.Random(f"{SELECTION_SEED}:normal_schedule:{category}").shuffle(ids)
            selected[arm][category] = ids
        category_audit[category] = {"candidate_sources": pool, "normal_random": random_ids,
                                   "hard_top16": highest, "hard_random16": random_rest,
                                   "normal_hard": hard_ids,
                                   "random_hard_overlap": len(set(random_ids) & set(hard_ids)),
                                   "pool_log_odds": distribution([scores[s]["anomaly_log_odds"] for s in pool]),
                                   "random_log_odds": distribution([scores[s]["anomaly_log_odds"] for s in random_ids]),
                                   "hard_log_odds": distribution([scores[s]["anomaly_log_odds"] for s in hard_ids]),
                                   "hard_ranking_tie_break": "source_id ascending"}
    artifacts = {}
    trains = {}
    # A source-to-source bijection preserves each reference normal's two
    # schedule positions, while anomaly identities and every category/label
    # slot remain exactly as the frozen v2 DD exposure order.
    reference_normals = defaultdict(list)
    for row in dd:
        if not row["is_anomalous"] and row["source_id"] not in reference_normals[row["category"]]:
            reference_normals[row["category"]].append(row["source_id"])
    for arm in ("normal_random", "normal_hard"):
        mapping = {old: new for category in state["categories"]
                   for old, new in zip(reference_normals[category], selected[arm][category], strict=True)}
        train = []
        for original in dd:
            sid = original["source_id"] if original["is_anomalous"] else mapping[original["source_id"]]
            record = inventory[sid]
            row = instruction(record, f"{sid}__curation_v3__{arm}__epoch{original['source_epoch']}__decision")
            row.update(task="decision", source_epoch=original["source_epoch"])
            train.append(row)
        require(len(train) == len({r["id"] for r in train}) == 1536, "Training ID/count mismatch")
        require(len({r["source_id"] for r in train}) == 768 and
                set(Counter(r["source_id"] for r in train).values()) == {2}, "Training exposure mismatch")
        require([(r["category"], r["is_anomalous"], r["source_epoch"]) for r in train] ==
                [(r["category"], r["is_anomalous"], r["source_epoch"]) for r in dd], "Category/label schedule mismatch")
        require([r["source_id"] for r in train if r["is_anomalous"]] ==
                [r["source_id"] for r in dd if r["is_anomalous"]], "Anomaly schedule mismatch")
        validate_reference(train, inventory, "decision")
        require(all(inventory[r["source_id"]]["split"] == "train" for r in train), "Selected training leakage")
        trains[arm] = train
        artifacts[f"prepared/curation_v3/train_{arm}.jsonl"] = lbytes(train)
    artifacts["prepared/curation_v3/train_region_highres.jsonl"] = (root / "prepared/multitask_v2/train_region_region.jsonl").read_bytes()
    trains["region_highres"] = state["rr"]
    for arm in ARMS:
        config = json.loads((root / ("configs/multitask_v2/region_region.json" if arm == "region_highres"
                                    else "configs/multitask_v2/decision_decision.json")).read_text())
        remote = state["plan"]["remote_root"]
        config.update(train_file=f"{remote}/prepared/curation_v3/train_{arm}.jsonl",
                      val_file=f"{remote}/prepared/curation_v3/val_{'region' if arm == 'region_highres' else 'decision'}_120.jsonl",
                      output=f"{remote}/runs/curation_v3/{arm}", max_pixels=1048576 if arm == "region_highres" else 524288)
        expected = {"max_steps": 192, "seed": 42, "sample_order": "sequential", "max_length": 2048,
                    "gradient_accumulation_steps": 8, "learning_rate": 0.0001, "warmup_steps": 10,
                    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        require(all(config.get(k) == v for k, v in expected.items()), "Frozen reference training settings differ")
        artifacts[f"configs/curation_v3/{arm}.json"] = jbytes(config)
    all_source_ids = set().union(*({r["source_id"] for r in t} for t in trains.values()))
    audit = {"schema_version": SCHEMA, "phase": "select", "mining": score_audit,
             "init_audit_sha256": sha((root / "prepared/curation_v3/init_audit.json").read_bytes()),
             "selection_seed": SELECTION_SEED, "category_selection": category_audit,
             "training_source_hashes": {sid: inventory[sid]["image_sha256"] for sid in sorted(all_source_ids)},
             "source_exposure_sequence_sha256": {a: source_sequence(t) for a, t in trains.items()},
             "task_exposures": {a: dict(Counter(r["task"] for r in t)) for a, t in trains.items()},
             "artifacts": {name: sha(content) for name, content in sorted(artifacts.items())},
             "checks": {"train_only_mining": True, "complete1152_native_base_scores": True,
                        "normal32_anomaly32_per_category_per_epoch": True,
                        "each_source_exposed_twice": True, "category_label_schedule_matched": True,
                        "anomaly_identity_and_order_matched": True, "region_highres_exact_v2_manifest_bytes": True,
                        "normal_source_identity_matched": False, "actual_initial_trainable_hashes_verified": False,
                        "raw_image_io_performed": False, "confirmation_predictions_read_or_generated": False},
             "hash_scope": "This audit excludes its own hash. Initial trainable hashes require later runtime verification."}
    artifacts["prepared/curation_v3/selection_audit.json"] = jbytes(audit)
    return artifacts, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("init", "select"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    initial, state = init_artifacts(root, args.remote_root.rstrip("/"))
    if args.phase == "init":
        outcome = dump_new(root, initial, args.dry_run)
    else:
        require(all((root / name).exists() for name in initial), "Run init and freeze its outputs before mining/select")
        dump_new(root, initial, True)  # frozen inputs and original candidate selection must remain byte-identical
        artifacts, _ = select_artifacts(root, state)
        outcome = dump_new(root, artifacts, args.dry_run)
    print(json.dumps({"phase": args.phase, "status": "passed", **outcome}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
