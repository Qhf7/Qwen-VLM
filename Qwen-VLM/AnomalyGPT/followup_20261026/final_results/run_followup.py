"""Run the frozen follow-up plan sequentially on one GPU; persist every job state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, obj):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def plan(root, out):
    py = sys.executable
    code = root / "code"
    qwen = "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"
    init = root / "ckpt/train_visa/pytorch_model.pt"
    vicuna = root / "pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1"
    resource_path = out / "vicuna_resource.json"
    if resource_path.exists():
        resource = json.loads(resource_path.read_text())
        if not resource.get("all_tensors_equal"):
            raise ValueError("Converted Vicuna resource lacks tensor-equivalence confirmation")
        vicuna = Path(resource["safetensors_path"])
    tasks = []

    def add(name, command, marker):
        tasks.append({"name": name, "command": [str(v) for v in command], "marker": str(marker), "log": str(out / "logs" / (name + ".log"))})

    historical = [("initial", init)]
    for seed, suffix in [(42, "16"), (43, "s43"), (44, "s44")]:
        historical += [(f"frozen{seed}", root / f"runs/balanced_strict_{suffix}/pytorch_model.pt"),
                       (f"lora{seed}", root / f"runs/lora_strict_{suffix}/pytorch_model.pt")]
    for name, checkpoint in historical:
        result = out / "synthetic_corrected" / (name + ".json")
        add("synthetic_" + name, [py, out / "corrected_synthetic_eval.py", "--cache", root / "cache_balanced_test24",
                                 "--checkpoint", checkpoint, "--model-source", code / "model/AnomalyGPT_models.py", "--out", result], result)

    def evaluation(name, backend, checkpoint, seed, adapter=None):
        destination = out / "real_evaluations" / name
        command = [py, out / "evaluate_models.py", "--backend", backend, "--cache", out / "real_cache",
                   "--checkpoint", checkpoint, "--init-ckpt", init, "--code-dir", code,
                   "--out", destination, "--seed", seed, "--batch-size", 64, "--max-new-tokens", 32]
        if backend == "qwen":
            command += ["--model-factory", "train_qwen_matched:factory", "--qwen-ckpt", qwen]
        else:
            command += ["--vicuna-ckpt", vicuna]
        if adapter:
            command += ["--lora-adapter", adapter]
        add(name, command, destination / "metrics.json")

    # The historical model evaluation precedes any new matched training.
    for seed, suffix in [(42, "16"), (43, "s43"), (44, "s44")]:
        evaluation(f"historical_frozen{seed}", "qwen", root / f"runs/balanced_strict_{suffix}/pytorch_model.pt", seed)
        evaluation(f"historical_lora{seed}", "qwen", root / f"runs/lora_strict_{suffix}/pytorch_model.pt", seed,
                   root / f"runs/lora_strict_{suffix}/lora_adapter")
    evaluation("shared_initial_qwen42", "qwen", out / "matched_inputs/seed42/bridge_init.pt", 42)

    for seed in (42, 43, 44):
        shared = out / f"matched_inputs/seed{seed}"
        for arm in ("qwen_frozen", "qwen_lora", "vicuna"):
            destination = out / "matched_runs" / f"{arm}_{seed}"
            command = [py, out / ("train_vicuna_matched.py" if arm == "vicuna" else "train_qwen_matched.py"),
                       "--cache", root / "cache_balanced_train48", "--code-dir", code,
                       "--sample-order", shared / "sample_order.json", "--save-path", destination,
                       "--seed", seed, "--max-steps", 16, "--batch-size", 2, "--max-tgt-len", 128, "--lr", 0.001]
            if arm == "vicuna":
                command += ["--vicuna-ckpt", vicuna, "--init-ckpt", shared / "bridge_init.pt",
                            "--projection-init-checkpoint", shared / "bridge_init.pt"]
            else:
                command += ["--qwen-ckpt", qwen, "--bridge-init", shared / "bridge_init.pt",
                            "--lora", "qkvo" if arm == "qwen_lora" else "none"]
            add(f"train_{arm}{seed}", command, destination / "pytorch_model.pt")
        for arm in ("qwen_frozen", "qwen_lora", "vicuna"):
            destination = out / "matched_runs" / f"{arm}_{seed}"
            evaluation(f"matched_{arm}{seed}", "vicuna" if arm == "vicuna" else "qwen",
                       destination / "pytorch_model.pt", seed,
                       destination / "lora_adapter" if arm == "qwen_lora" else None)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--only", default="", help="Optional comma-separated task names")
    args = parser.parse_args()
    root = args.experiment.resolve()
    out = root / "followup_20261026"
    (out / "logs").mkdir(exist_ok=True)
    tasks = plan(root, out)
    if args.only:
        names = set(args.only.split(","))
        tasks = [t for t in tasks if t["name"] in names]
        if {t["name"] for t in tasks} != names:
            raise ValueError("Unknown task name")
    plan_path = out / ("selected_plan.json" if args.only else "plan.json")
    write(plan_path, tasks)
    state_path = out / ("selected_state.json" if args.only else "state.json")
    state = {"started_at": time.time(), "completed": [], "running": None, "failed": None}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    for task in tasks:
        if Path(task["marker"]).exists():
            state["completed"].append({"name": task["name"], "existing": True})
            continue
        state["running"] = task["name"]
        write(state_path, state)
        print("[start]", task["name"], flush=True)
        with open(task["log"], "w") as log:
            result = subprocess.run(task["command"], cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not Path(task["marker"]).exists():
            state["failed"] = {"name": task["name"], "exit_code": result.returncode, "log": task["log"]}
            state["running"] = None
            write(state_path, state)
            raise RuntimeError(state["failed"])
        state["completed"].append({"name": task["name"], "completed_at": time.time()})
        print("[done]", task["name"], flush=True)
    state["running"] = None
    state["finished_at"] = time.time()
    write(state_path, state)
    print("FOLLOWUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
