"""Queue fixed boolean scoring after v2 training; validation-calibrated regression only."""
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
EXP = ROOT / "experiments/multitask_v2"
PYTHON = "/root/autodl-tmp/vlm/.venv/bin/python"
STATE = EXP / "scoring_state.json"
OUTPUT = ROOT / "evaluations/multitask_v2_scores"
ARMS = ("base", "region_region", "decision_decision", "region_decision")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_unix": time.time(), **value}, indent=2) + "\n")
    temporary.replace(STATE)


def main():
    # Run from a separate frozen import tree so training source inventory stays intact.
    runtime = EXP / "scoring_runtime_src"
    if not runtime.exists():
        shutil.copytree(EXP / "runtime_src", runtime, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "src/industrial_vlm/score.py", runtime / "industrial_vlm/score.py")
        paths = sorted(runtime.rglob("*.py"))
        (EXP / "scoring_source_manifest.json").write_text(json.dumps({
            "files": {str(p.relative_to(runtime)): sha(p) for p in paths},
            "runner_sha256": sha(Path(__file__)),
            "scope": "Frozen scoring import tree plus runner; boolean scores never use confirmation manifests.",
        }, indent=2) + "\n")
    provenance = json.loads((EXP / "scoring_source_manifest.json").read_text())
    def validate_frozen_inputs():
        if {str(p.relative_to(runtime)) for p in runtime.rglob("*.py")} != set(provenance["files"]):
            raise RuntimeError("Frozen scoring module inventory changed")
        for path, digest in provenance["files"].items():
            if sha(runtime / path) != digest:
                raise RuntimeError(f"Frozen scoring source changed: {path}")
        if sha(Path(__file__)) != provenance["runner_sha256"]:
            raise RuntimeError("Scoring runner changed")
        for path, digest in json.loads((EXP / "input_hashes.json").read_text()).items():
            if sha(ROOT / path) != digest:
                raise RuntimeError(f"Frozen experiment input changed: {path}")
    validate_frozen_inputs()
    state({"status": "waiting", "stage": "waiting_for_training_and_validation"})
    wait_deadline = time.monotonic() + 2 * 60 * 60
    while True:
        training = json.loads((EXP / "state.json").read_text())
        if training["status"] == "complete":
            break
        if training["status"] == "failed":
            raise RuntimeError("Training pipeline failed; scoring not started")
        if time.monotonic() > wait_deadline:
            raise TimeoutError("Training completion not observed within two hours; inspect training state/process")
        time.sleep(15)
    validate_frozen_inputs()
    sys.path.insert(0, str(runtime))
    from industrial_vlm.score import _read_scores, calibrate_normal_threshold
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(runtime), "OMP_NUM_THREADS": "8",
           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def run(name, args):
        state({"status": "running", "stage": name})
        print(json.dumps({"event": "start", "stage": name, "unix": time.time()}), flush=True)
        with (EXP / "logs" / (name + ".log")).open("a") as handle:
            result = subprocess.run([PYTHON, "-u", "-m", "industrial_vlm.score", *args],
                                    env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: exit={result.returncode}")
        print(json.dumps({"event": "finished", "stage": name, "unix": time.time()}), flush=True)

    for arm in ARMS:
        threshold = OUTPUT / (arm + "_threshold.json")
        for role, data in (("validation", ROOT / "prepared/multitask_v2/val_decision_120.jsonl"),
                           ("regression", ROOT / "prepared/visa/test_balanced_120_decision.jsonl")):
            name = arm + "_" + role
            directory = OUTPUT / name
            args = ["score", "--model", str(ROOT / "models/Qwen3-VL-8B-Instruct"),
                    "--data", str(data), "--image-root", str(ROOT / "datasets"),
                    "--output", str(directory), "--data-role", role, "--run-name", name,
                    "--max-pixels", "524288", "--max-length", "2048"]
            if arm != "base":
                args += ["--adapter", str(ROOT / "runs/multitask_v2" / arm / "adapter")]
            if role == "regression":
                args += ["--calibration", str(threshold)]
            if (directory / "meta.json").exists():
                args.append("--resume")
            run("score_" + name, args)
            if role == "validation":
                if threshold.exists():
                    calibration = json.loads(threshold.read_text())
                    score_rows, metadata = _read_scores(directory)
                    expected_threshold = calibrate_normal_threshold(score_rows, 0.05)
                    if (calibration["source_scores_sha256"] != sha(directory / "scores.jsonl")
                            or calibration["score_model_contract"] != metadata["config"]["score_model_contract"]
                            or calibration["source_data_sha256"] != metadata["config"]["data_sha256"]
                            or calibration["kind"] != "validation_normal_threshold"
                            or calibration["calibration_source_ids"] != sorted(r["source_id"] for r in score_rows)
                            or any(calibration.get(key) != value for key, value in expected_threshold.items())):
                        raise RuntimeError("Existing calibration differs")
                else:
                    run("calibrate_" + arm, ["calibrate", "--scores", str(directory),
                                             "--output", str(threshold), "--target-fpr", "0.05"])
    state({"status": "complete", "stage": "all_boolean_scores_and_validation_thresholds",
           "confirmation_evaluated": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        old = json.loads(STATE.read_text()) if STATE.exists() else {}
        state({**old, "status": "failed", "error": str(exc)})
        raise
