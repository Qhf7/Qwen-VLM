"""Run the second, exposure-matched experiment on one GPU; never touch confirmation data."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/autodl-tmp/vlm/.venv/bin/python"
EXP = ROOT / "experiments/multitask_v2"
STATE = EXP / "state.json"
ARMS = ("region_region", "decision_decision", "region_decision")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_state(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_unix": time.time(), **value}, indent=2) + "\n")
    temporary.replace(STATE)


def freeze_runtime():
    EXP.mkdir(parents=True, exist_ok=True)
    source = EXP / "runtime_src"
    if not source.exists():
        shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
        files = sorted(p for p in source.rglob("*.py"))
        (EXP / "training_source_manifest.json").write_text(json.dumps({
            "created_unix": time.time(),
            "files": {str(p.relative_to(source)): sha(p) for p in files},
            "note": "Training/evaluation import this frozen runtime; later score module is archived separately.",
        }, indent=2) + "\n")
    manifest = json.loads((EXP / "training_source_manifest.json").read_text())
    if {str(p.relative_to(source)) for p in source.rglob("*.py")} != set(manifest["files"]):
        raise RuntimeError("Frozen runtime module inventory changed")
    for path, expected in manifest["files"].items():
        if sha(source / path) != expected:
            raise RuntimeError(f"Frozen runtime changed: {path}")
    return source


def run(stage, args, marker):
    marker = ROOT / marker
    if marker.name == "summary.json" and marker.is_file() and json.loads(marker.read_text()).get("status") == "complete":
        summary = json.loads(marker.read_text())
        stored = json.loads((marker.parent / "run_config.json").read_text())
        supplied = json.loads(Path(args[args.index("--config") + 1]).read_text())
        if any(stored.get(key) != value for key, value in supplied.items()):
            raise RuntimeError(f"Completed training config changed: {stage}")
        for key, digest_key in (("train_file", "train_sha256"), ("val_file", "val_sha256")):
            if sha(Path(stored[key])) != stored[digest_key]:
                raise RuntimeError(f"Completed training input changed: {stage}")
        if (not summary.get("reload_check", {}).get("passed")
                or sha(marker.parent / "exposures.jsonl") != summary["exposure_trace_sha256"]
                or not (marker.parent / "adapter/adapter_model.safetensors").is_file()):
            raise RuntimeError(f"Completed training artifacts failed audit: {stage}")
        print(json.dumps({"event": "already_complete", "stage": stage}), flush=True)
        return
    if marker.name == "summary.json" and marker.parent.exists() and any(marker.parent.iterdir()):
        raise RuntimeError(f"Incomplete training directory must be preserved: {marker.parent}")
    command = [PYTHON, "-u", *args]
    env = {**os.environ, "PYTHONPATH": str(EXP / "runtime_src"), "OMP_NUM_THREADS": "8",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    write_state({"status": "running", "stage": stage, "command": command})
    print(json.dumps({"event": "start", "stage": stage, "unix": time.time()}), flush=True)
    with (EXP / "logs" / (stage + ".log")).open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_state({"status": "failed", "stage": stage, "exit_code": result.returncode})
        raise RuntimeError(f"{stage} failed; inspect experiments/multitask_v2/logs/{stage}.log")
    if not marker.is_file() or json.loads(marker.read_text()).get("status") != "complete":
        raise RuntimeError(f"Missing completion marker: {marker}")
    print(json.dumps({"event": "finished", "stage": stage, "unix": time.time()}), flush=True)


def main():
    prepared = ROOT / "prepared/multitask_v2"
    plan_path = prepared / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError("Prepare and freeze the protocol first")
    split = json.loads((ROOT / "prepared/visa/split.json").read_text())
    load_rows = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    confirmation = {r["source_id"] for r in load_rows(prepared / "confirmation_region_240.jsonl")}
    source_orders = []
    for arm in ARMS:
        rows = load_rows(prepared / f"train_{arm}.jsonl")
        counts = __import__("collections").Counter(r["source_id"] for r in rows)
        if len(rows) != 1536 or len(counts) != 768 or set(counts.values()) != {2}:
            raise RuntimeError(f"Invalid exposure budget: {arm}")
        if not set(counts) <= set(split["train"]) or set(counts) & confirmation:
            raise RuntimeError(f"Training isolation failed: {arm}")
        source_orders.append([r["source_id"] for r in rows])
    if not all(order == source_orders[0] for order in source_orders):
        raise RuntimeError("Training sources are not aligned")
    for task in ("region", "decision"):
        val_ids = {r["source_id"] for r in load_rows(prepared / f"val_{task}_120.jsonl")}
        if len(val_ids) != 120 or not val_ids <= set(split["val"]) or val_ids & confirmation:
            raise RuntimeError("Validation isolation failed")
    EXP.mkdir(parents=True, exist_ok=True)
    # Freeze data/config bytes before the first GPU stage.
    inputs = (sorted(prepared.glob("*.json*")) + sorted((ROOT / "configs/multitask_v2").glob("*.json"))
              + [Path(__file__).resolve()])
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    ledger = EXP / "input_hashes.json"
    if ledger.exists() and json.loads(ledger.read_text()) != provenance:
        raise RuntimeError("Prepared inputs changed after experiment start")
    if not ledger.exists():
        ledger.write_text(json.dumps(provenance, indent=2) + "\n")
    freeze_runtime()
    (EXP / "logs").mkdir(exist_ok=True)
    for arm in ARMS:
        run("train_" + arm, ["-m", "industrial_vlm.train", "--config",
            str(ROOT / "configs/multitask_v2" / (arm + ".json"))],
            f"runs/multitask_v2/{arm}/summary.json")
    for arm, tasks in (("region_region", ("region", "decision")),
                       ("decision_decision", ("decision",)),
                       ("region_decision", ("region", "decision"))):
        for task in tasks:
            name = arm + "_" + task
            output = ROOT / "evaluations/multitask_v2" / name
            args = ["-m", "industrial_vlm.evaluate", "--model", str(ROOT / "models/Qwen3-VL-8B-Instruct"),
                    "--adapter", str(ROOT / "runs/multitask_v2" / arm / "adapter"),
                    "--data", str(prepared / f"val_{task}_120.jsonl"),
                    "--image-root", str(ROOT / "datasets"), "--output", str(output),
                    "--run-name", name, "--branch", "multitask_v2_validation",
                    "--max-pixels", "524288", "--max-new-tokens", "128", "--bootstrap-samples", "200"]
            if (output / "meta.json").exists():
                args.append("--resume")
            run("val_" + name, args, f"evaluations/multitask_v2/{name}/meta.json")
    write_state({"status": "complete", "stage": "training_and_validation_generation",
                 "confirmation_evaluated": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if EXP.exists():
            old = json.loads(STATE.read_text()) if STATE.exists() else {}
            write_state({**old, "status": "failed", "error": str(exc)})
        raise
