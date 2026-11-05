"""Verify actual completed training/evaluation artifacts, without opening confirmation images."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("region_region", "decision_decision", "region_decision")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    exp = ROOT / "experiments/multitask_v2"
    checks, details = {}, {}

    def check(name, passed):
        checks[name] = bool(passed)

    ledgers = json.loads((exp / "input_hashes.json").read_text())
    for path, digest in ledgers.items():
        check("input_bytes:" + path, hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest)
    runtime = exp / "runtime_src"
    for path, digest in json.loads((exp / "training_source_manifest.json").read_text())["files"].items():
        check("runtime_bytes:" + path, hashlib.sha256((runtime / path).read_bytes()).hexdigest() == digest)
    exposure_orders, initial = [], []
    for arm in ARMS:
        directory = ROOT / "runs/multitask_v2" / arm
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "run_config.json").read_text())
        actual = rows(directory / "exposures.jsonl")
        planned = rows(ROOT / "prepared/multitask_v2" / f"train_{arm}.jsonl")
        source_order = [r["source_id"] for r in actual]
        check(arm + ":complete", summary["status"] == "complete")
        check(arm + ":budget", summary["optimizer_steps"] == 192 and summary["microsteps"] == len(actual) == 1536)
        check(arm + ":source_sequence", source_order == [r["source_id"] for r in planned])
        check(arm + ":sample_ids", [r["id"] for r in actual] == [r["id"] for r in planned])
        check(arm + ":task_sequence", [r["task"] for r in actual] == [
            "region" if "boxes" in json.loads(r["answer"]) else "decision" for r in planned])
        check(arm + ":step_indices", [(r["optimizer_step"], r["microstep"]) for r in actual] == [
            (i // 8 + 1, i % 8) for i in range(1536)])
        check(arm + ":each_source_twice", len(Counter(source_order)) == 768 and set(Counter(source_order).values()) == {2})
        check(arm + ":token_sum", sum(r["target_tokens"] for r in actual) == summary["total_target_tokens"])
        check(arm + ":trace_hash", hashlib.sha256((directory / "exposures.jsonl").read_bytes()).hexdigest() == summary["exposure_trace_sha256"])
        check(arm + ":source_hash", hashlib.sha256("".join(s + "\n" for s in source_order).encode()).hexdigest() == summary["source_exposure_sha256"])
        check(arm + ":fresh_base", config["adapter"] is None and config["sample_order"] == "sequential")
        check(arm + ":reload", summary["reload_check"]["passed"])
        initial.append(summary["initial_trainable_sha256"])
        exposure_orders.append(source_order)
        tasks = {}
        for task in ("region", "decision"):
            subset = [r for r in actual if r["task"] == task]
            tasks[task] = {"n": len(subset), "target_tokens": sum(r["target_tokens"] for r in subset)}
        details[arm] = {"tasks": tasks, "summary": summary}
    check("equal_actual_source_exposure_sequences", all(order == exposure_orders[0] for order in exposure_orders))
    check("equal_lora_initial_parameters", len(set(initial)) == 1)
    confirmation = {r["source_id"] for r in rows(ROOT / "prepared/multitask_v2/confirmation_region_240.jsonl")}
    eval_counts = {}
    for name in ("region_region_region", "region_region_decision", "decision_decision_decision",
                 "region_decision_region", "region_decision_decision"):
        directory = ROOT / "evaluations/multitask_v2" / name
        meta = json.loads((directory / "meta.json").read_text())
        predictions = rows(directory / "predictions.jsonl")
        check(name + ":evaluation_complete", meta["status"] == "complete" and meta["n_completed"] == len(predictions) == 120)
        check(name + ":no_confirmation", not confirmation & {r["source_id"] for r in predictions})
        config = meta["config"]
        for path, digest in config["adapter_files_sha256"].items():
            local = ROOT / "runs/multitask_v2" / name.rsplit("_", 1)[0] / "adapter" / path
            check(name + ":adapter:" + path, hashlib.sha256(local.read_bytes()).hexdigest() == digest)
        eval_counts[name] = len(predictions)
    result = {"status": "passed" if all(checks.values()) else "failed", "n_checks": len(checks),
              "checks": checks, "training": details, "validation_counts": eval_counts,
              "confirmation_evaluated": False}
    (exp / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failures": [name for name, passed in checks.items() if not passed]}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
