"""Run the fixed single-GPU pilot sequentially with durable status and logs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/autodl-tmp/vlm/.venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "8",
       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
       "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
STATE = ROOT / "pipeline_state.json"


def state(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_unix": time.time(), **value}, indent=2))
    temporary.replace(STATE)


def run(name, arguments, marker):
    marker = ROOT / marker
    if marker.exists():
        item = json.loads(marker.read_text())
        if marker.name != "meta.json" and item.get("status") == "complete":
            print(json.dumps({"event": "already_complete", "stage": name}), flush=True)
            return
    if marker.name == "summary.json" and marker.parent.exists() and any(marker.parent.iterdir()):
        raise RuntimeError(
            f"Incomplete training directory {marker.parent}; preserve it and start a fresh run. "
            "Optimizer-state resume is not implemented; adapters are not optimizer checkpoints."
        )
    command = [PYTHON, "-u", *arguments]
    state({"status": "running", "stage": name, "command": command})
    print(json.dumps({"event": "start", "stage": name, "unix": time.time()}), flush=True)
    with (ROOT / "logs" / (name + ".log")).open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state({"status": "failed", "stage": name, "exit_code": result.returncode,
               "log": str(ROOT / "logs" / (name + ".log"))})
        raise SystemExit(result.returncode)
    print(json.dumps({"event": "finished", "stage": name, "unix": time.time()}), flush=True)


def evaluate(name, adapter, dataset, branch):
    arguments = ["-m", "industrial_vlm.evaluate", "--model", str(ROOT / "models/Qwen3-VL-8B-Instruct"),
                 "--data", str(ROOT / dataset), "--image-root", str(ROOT / "datasets"),
                 "--output", str(ROOT / "evaluations" / name), "--run-name", name,
                 "--branch", branch, "--max-pixels", "524288", "--max-new-tokens", "128",
                 "--bootstrap-samples", "200"]
    if adapter:
        arguments += ["--adapter", str(ROOT / adapter)]
    if (ROOT / "evaluations" / name / "meta.json").exists():
        arguments.append("--resume")
    run(name, arguments, "evaluations/" + name + "/meta.json")


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    plan = ROOT / "configs/experiment_plan.json"
    assert plan.is_file(), "Freeze the experiment plan first."
    prepared = ROOT / "prepared/visa"
    rows = [json.loads(line) for line in (prepared / "train_quality_region.jsonl").read_text().splitlines()]
    # Select a small source-balanced TRAINING subset solely for fit diagnostics.
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["is_anomalous"]), []).append(row)
    tiny = []
    for i in range(2):
        for key in sorted(groups):
            if len(tiny) < 32 and i < len(groups[key]):
                tiny.append(groups[key][i])
    tiny_file = prepared / "overfit32.jsonl"
    if not tiny_file.exists():
        tiny_file.write_text("".join(json.dumps(r) + "\n" for r in tiny))
    # Keep the original diagnostic inventory when returning to a saved run.
    # Main-arm sampling may have been rebuilt before the main experiments;
    # rewriting this file would invalidate the completed diagnostic identities.
    saved_tiny = [json.loads(line) for line in tiny_file.read_text().splitlines()]
    train_ids = set(json.loads((prepared / "split.json").read_text())["train"])
    assert len(saved_tiny) == 32 and len({r["source_id"] for r in saved_tiny}) == 32
    assert {r["source_id"] for r in saved_tiny} <= train_ids
    run("real_data_smoke", ["-m", "industrial_vlm.train", "--model", str(ROOT / "models/Qwen3-VL-8B-Instruct"),
        "--train-file", str(prepared / "overfit32.jsonl"), "--output", str(ROOT / "runs/real_data_smoke"),
        "--smoke", "--gradient-accumulation-steps", "1", "--max-pixels", "524288", "--eval-steps", "0"],
        "runs/real_data_smoke/summary.json")
    run("overfit32", ["-m", "industrial_vlm.train", "--model", str(ROOT / "models/Qwen3-VL-8B-Instruct"),
        "--train-file", str(prepared / "overfit32.jsonl"), "--output", str(ROOT / "runs/overfit32"),
        "--max-steps", "48", "--gradient-accumulation-steps", "4", "--max-pixels", "524288",
        "--learning-rate", "0.0002", "--warmup-steps", "3", "--save-steps", "0", "--eval-steps", "0"],
        "runs/overfit32/summary.json")
    evaluate("overfit32_training_diagnostic", "runs/overfit32/adapter", "prepared/visa/overfit32.jsonl", "training_fit_diagnostic")
    for arm in ["quality_region", "random_region", "quality_decision"]:
        run("train_" + arm, ["-m", "industrial_vlm.train", "--config", str(ROOT / "configs" / (arm + ".json"))],
            "runs/" + arm + "/summary.json")
    region_test = "prepared/visa/test_balanced_120.jsonl"
    decision_test = "prepared/visa/test_balanced_120_decision.jsonl"
    evaluate("base_region", None, region_test, "base_vs_sft")
    evaluate("quality_region", "runs/quality_region/adapter", region_test, "base_vs_sft")
    evaluate("random_region", "runs/random_region/adapter", region_test, "dataquality_sampling_ablation")
    evaluate("base_decision", None, decision_test, "region_supervision_ablation")
    evaluate("quality_region_decision_prompt", "runs/quality_region/adapter", decision_test, "region_supervision_ablation")
    evaluate("quality_decision", "runs/quality_decision/adapter", decision_test, "region_supervision_ablation")
    external = ROOT / "prepared/mvtec/external_test.jsonl"
    assert external.is_file() and len(external.read_text().splitlines()) == 150, "Expected the fixed 150-image external pilot"
    evaluate("external_base_region", None, "prepared/mvtec/external_test.jsonl", "external_pilot")
    evaluate("external_quality_region", "runs/quality_region/adapter", "prepared/mvtec/external_test.jsonl", "external_pilot")
    run("report", ["-m", "industrial_vlm.report", "--runs", str(ROOT / "evaluations"),
                   "--output", str(ROOT / "reports")], "reports/nonexistent_marker.json")
    state({"status": "complete", "stage": "all", "report": str(ROOT / "reports/matrix.md")})
    print(json.dumps({"event": "complete", "unix": time.time()}), flush=True)


if __name__ == "__main__":
    main()
