"""Validate and resume the frozen one-GPU follow-up plan sequentially.

This is a separate entry point; it never edits run_followup.py. Completed jobs
are verified and skipped, including legacy jobs adopted after validation.
Partial outputs fail by default. --archive-partial renames only that task's
output to a timestamped sibling before restarting; it never deletes files.
Any completion claim with broken integrity is an error, not an automatic rerun.
Do not start while the original runner is still executing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from run_followup import plan


_HASH_CACHE = {}
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SYNTHETIC_VERSION = "four_layer_average_image_label_v2"


def sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def option(task, *flags):
    command = task["command"]
    for flag in flags:
        if flag in command:
            index = command.index(flag)
            if index + 1 >= len(command):
                raise ValueError(f"Missing value for {flag}")
            return command[index + 1]
    return None


def task_kind(task):
    script = Path(task["command"][1]).name
    if script == "corrected_synthetic_eval.py":
        return "synthetic"
    if script in {"train_qwen_matched.py", "train_vicuna_matched.py"}:
        return "train"
    if script == "evaluate_models.py":
        return "evaluation"
    raise ValueError(f"Unsupported task script: {script}")


def task_output(task):
    value = option(task, "--save-path", "--out")
    if not value:
        raise ValueError("Task has no explicit output path")
    return Path(value).resolve()


def script_dependencies(task, root):
    script = Path(task["command"][1]).resolve()
    followup = script.parent
    paths = {script}
    kind = task_kind(task)
    if kind in {"train", "evaluation"}:
        paths.add(followup / "train_vicuna_matched.py")
        if script.name == "train_qwen_matched.py" or option(task, "--backend") == "qwen":
            paths.add(followup / "train_qwen_matched.py")
        paths.update({root / "code/model/AnomalyGPT_models.py", root / "code/utils/loss.py"})
    if kind == "evaluation":
        paths.add(followup / "exact_bf16_ranking.py")
    if kind == "synthetic":
        paths.add(Path(option(task, "--model-source")))
    return sorted(path.resolve() for path in paths if path.is_file())


def current_sources(task, root):
    return {str(path): sha256(path) for path in script_dependencies(task, root)}


def source_trace(digest, filename, root, archives):
    if not isinstance(digest, str) or not SHA_PATTERN.fullmatch(digest):
        raise ValueError(f"Missing or invalid recorded source hash for {filename}")
    followup = root / "followup_20261026"
    candidates = [followup / filename, root / "code" / filename]
    candidates += [directory / filename for directory in archives]
    for directory in [followup / "safe_runtime", followup / "runtime_src", followup / "runtime_sources",
                      followup / "source_snapshots", followup / "server_snapshot", *archives]:
        if directory.is_dir():
            candidates.extend(directory.rglob(Path(filename).name))
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and sha256(path) == digest:
            return {"sha256": digest, "traceable_file": str(path)}
    raise ValueError(f"Recorded {filename} SHA256 is not traceable to current or archived source: {digest}; supply --runtime-archive")


def preserve_recorded_sources(evidence, followup, task_name):
    """Adoption freezes the actual matching legacy bytes, not today's code."""
    traces = evidence.get("recorded_source_trace", {})
    if not traces:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = followup / "safe_runtime" / task_name / ("adopted-" + stamp)
    destination.mkdir(parents=True, exist_ok=False)
    for index, (name, trace) in enumerate(sorted(traces.items())):
        source = Path(trace["traceable_file"])
        target = destination / f"{index:02d}" / source.name
        target.parent.mkdir()
        shutil.copy2(source, target)
        verify_hash(target, trace["sha256"], "adopted recorded source")
        trace["frozen_archive"] = str(target)


def recorded_sources(metadata, task, root, archives):
    recorded = metadata.get("source_sha256")
    script = Path(task["command"][1]).name
    if isinstance(recorded, str):
        # Legacy evaluators do not record fast-metric helper hashes. Their
        # evaluator itself still has to match an actual current/archived file.
        return {"runner": source_trace(recorded, script, root, archives)}
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Run has no recorded evaluator/runner source SHA256")
    names = {"runner": script, "shared_training": "train_vicuna_matched.py",
             "qwen_model": "model/openllama_qwen3vl_strict.py",
             "bridge_architecture": "model/AnomalyGPT_models.py", "pixel_losses": "utils/loss.py"}
    return {key: source_trace(value, names.get(key, Path(key).name), root, archives)
            for key, value in recorded.items()}


def verify_hash(path, expected, label):
    if not isinstance(expected, str) or not SHA_PATTERN.fullmatch(expected):
        raise ValueError(f"Missing SHA256 for {label}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {label}: {path}")
    return actual


def validate_task(task, root, archives=(), expected_sources=2005):
    """Return integrity evidence; raise on empty markers or partial products."""
    marker = Path(task["marker"])
    if not marker.is_file() or marker.stat().st_size == 0:
        raise ValueError("Completion marker is missing or empty")
    kind = task_kind(task)
    evidence = {"kind": kind, "artifacts": {str(marker.resolve()): sha256(marker)}}
    if kind == "synthetic":
        data = read_json(marker)
        if data.get("evaluation_version") != SYNTHETIC_VERSION:
            raise ValueError("Synthetic result is not image-label revision v2")
        checkpoint = Path(option(task, "--checkpoint", "--initial-checkpoint")).resolve()
        if Path(data.get("checkpoint", "")).resolve() != checkpoint:
            raise ValueError("Synthetic checkpoint path differs from task")
        verify_hash(checkpoint, data.get("checkpoint_sha256"), "synthetic checkpoint")
        source = Path(option(task, "--model-source"))
        verify_hash(source, data.get("decoder_source_sha256"), "decoder source")
        records = data.get("records", [])
        if not records or len(records) != data.get("pooled", {}).get("num_variants"):
            raise ValueError("Synthetic records/counts are incomplete")
        evidence.update({"checkpoint_sha256": data["checkpoint_sha256"], "num_records": len(records)})
        return evidence
    directory = marker.parent
    metadata_path = directory / "run.json"
    metadata = read_json(metadata_path)
    evidence["artifacts"][str(metadata_path.resolve())] = sha256(metadata_path)
    evidence["recorded_source_trace"] = recorded_sources(metadata, task, root, archives)
    if kind == "train":
        steps = int(option(task, "--max-steps") or 16)
        if steps != 16 or metadata.get("steps") != steps:
            raise ValueError("Training run has not completed the frozen 16 steps")
        history_path = directory / "history.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        if len(history) != steps or [row.get("step") for row in history] != list(range(1, steps + 1)):
            raise ValueError("Training history is not 16 ordered completed updates")
        fingerprint = metadata.get("actual_initial_bridge_fingerprint")
        if not isinstance(fingerprint, str) or not SHA_PATTERN.fullmatch(fingerprint):
            raise ValueError("Actual initial bridge tensor fingerprint is absent")
        verify_hash(marker, metadata.get("checkpoint_sha256"), "training checkpoint")
        schedule_path = directory / "sample_order.json"
        schedule = read_json(schedule_path).get("batches")
        if schedule != [row.get("files") for row in history]:
            raise ValueError("History source exposure differs from saved schedule")
        verify_hash(schedule_path, metadata.get("sample_order_sha256"), "training schedule")
        input_schedule = Path(option(task, "--sample-order"))
        if read_json(input_schedule).get("batches")[:steps] != schedule:
            raise ValueError("Saved schedule differs from the task's frozen input schedule")
        initialization = Path(option(task, "--bridge-init", "--init-ckpt"))
        recorded_initialization = metadata.get("initial_bridge_sha256", metadata.get("initial_checkpoint_sha256"))
        verify_hash(initialization, recorded_initialization, "shared bridge initialization")
        for path in (history_path, schedule_path, directory / "token_budget.json"):
            if not path.is_file():
                raise ValueError(f"Missing training audit: {path}")
            evidence["artifacts"][str(path.resolve())] = sha256(path)
        if option(task, "--lora") == "qkvo":
            adapter = directory / "lora_adapter"
            adapter_files = [p for p in adapter.glob("*") if p.is_file()]
            if not any(p.suffix == ".safetensors" for p in adapter_files) or not (adapter / "adapter_config.json").is_file():
                raise ValueError("LoRA adapter files are incomplete")
            evidence["artifacts"].update({str(path.resolve()): sha256(path) for path in adapter_files})
        evidence["actual_initial_bridge_fingerprint"] = fingerprint
        return evidence
    metrics = read_json(marker)
    if metadata.get("completed") is not True or metadata.get("completed_num_sources") != expected_sources or metrics.get("num_sources") != expected_sources:
        raise ValueError(f"Evaluation run/metrics do not claim {expected_sources} completed sources")
    prediction_path = directory / "predictions.jsonl"
    checkpoint = option(task, "--checkpoint")
    if checkpoint:
        bridge = metadata.get("bridge", {})
        if Path(bridge.get("path", "")).resolve() != Path(checkpoint).resolve():
            raise ValueError("Evaluation bridge checkpoint path differs from task")
        verify_hash(checkpoint, bridge.get("sha256"), "evaluated bridge checkpoint")
    adapter = option(task, "--lora-adapter")
    if adapter:
        recorded_adapter = metadata.get("adapter_hashes", {})
        if not recorded_adapter:
            raise ValueError("Evaluated LoRA adapter hashes are absent")
        for filename, expected in recorded_adapter.items():
            if Path(filename).name != filename:
                raise ValueError("Adapter filename is not a simple filename")
            verify_hash(Path(adapter) / filename, expected, "evaluated LoRA adapter")
    verify_hash(prediction_path, metadata.get("prediction_sha256"), "predictions")
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    paths = [row.get("source_path") for row in predictions]
    if len(predictions) != expected_sources or len(set(paths)) != expected_sources or None in paths:
        raise ValueError("Evaluation predictions contain an incomplete or duplicate source set")
    for row in predictions:
        expected_id = hashlib.sha256(str(row["source_path"]).encode()).hexdigest()[:24]
        if row.get("source_id") != expected_id:
            raise ValueError("Prediction source ID does not hash its source path")
        map_path = directory / row["map_file"]
        if not map_path.is_file() or map_path.stat().st_size == 0:
            raise ValueError(f"Missing output map: {map_path}")
    cache = Path(option(task, "--cache"))
    verify_hash(cache / "manifest.json", metadata.get("cache_manifest_sha256"), "real cache manifest")
    cache_hashes = metadata.get("cache_hashes", {})
    if len(cache_hashes) != expected_sources:
        raise ValueError("Evaluation cache source hashes are incomplete")
    for filename, expected in cache_hashes.items():
        if Path(filename).name != filename:
            raise ValueError("Cache hash filename is not a simple filename")
        verify_hash(cache / filename, expected, "cached source " + filename)
    if set(row.get("cache_file") for row in predictions) != set(cache_hashes):
        raise ValueError("Predictions and hashed cache source sets differ")
    evidence["artifacts"][str(prediction_path.resolve())] = sha256(prediction_path)
    evidence.update({"num_sources": expected_sources, "prediction_sha256": metadata["prediction_sha256"]})
    return evidence


def completion_claimed(task):
    """Protect claimed finished products from automatic archive/retraining."""
    marker = Path(task["marker"])
    try:
        if task_kind(task) == "synthetic":
            return marker.is_file() and marker.stat().st_size > 0 and "evaluation_version" in read_json(marker)
        data = read_json(marker.parent / "run.json")
        return data.get("completed") is True or data.get("steps") == 16
    except (OSError, ValueError, TypeError):
        return False


def archive_partial(task, allowed_root):
    output = task_output(task)
    allowed_root = Path(allowed_root).resolve()
    if not output.is_relative_to(allowed_root) or output == allowed_root:
        raise ValueError("Refusing to archive an output outside the dedicated follow-up directory")
    if not output.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = output.with_name(output.name + ".partial-" + stamp)
    output.rename(destination)
    return {"from": str(output), "to": str(destination)}


def snapshot_sources(sources, destination):
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for index, (source, expected) in enumerate(sorted(sources.items())):
        target = destination / f"{index:02d}" / Path(source).name
        target.parent.mkdir()
        shutil.copy2(source, target)
        verify_hash(target, expected, "runtime source snapshot")
        result[source] = {"sha256": expected, "archive": str(target)}
    return result


def assert_original_runner_idle(root):
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if not re.search(r"(?:^|[/\s])run_followup\.py(?:\s|$)", line):
            continue
        if str(root) in line:
            raise RuntimeError("The original run_followup.py is still running for this experiment; wait for it to exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--only", default="")
    parser.add_argument("--archive-partial", action="store_true")
    parser.add_argument("--runtime-archive", action="append", default=[], type=Path)
    args = parser.parse_args()
    root = args.experiment.resolve()
    followup = root / "followup_20261026"
    if not (root / "code").is_dir() or not followup.is_dir():
        parser.error("Experiment must contain code/ and followup_20261026/")
    assert_original_runner_idle(root)
    jobs = followup / "safe_jobs"
    jobs.mkdir(exist_ok=True)
    lock_handle = (followup / "safe_runner.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another safe runner holds this follow-up's GPU sequence lock")
    tasks = plan(root, followup)
    if args.only:
        names = {name.strip() for name in args.only.split(",") if name.strip()}
        tasks = [task for task in tasks if task["name"] in names]
        if {task["name"] for task in tasks} != names:
            parser.error("Unknown task name in --only")
    state_path = followup / "safe_state.json"
    state = {"started_at": time.time(), "completed": [], "running": None, "failed": None, "archives": []}
    write_json(followup / "safe_selected_plan.json", tasks)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    try:
        for task in tasks:
            state["running"] = task["name"]
            write_json(state_path, state)
            manifest_path = jobs / (task["name"] + ".json")
            previous = read_json(manifest_path) if manifest_path.is_file() else None
            validation_archives = list(args.runtime_archive)
            if previous:
                for trace in previous.get("validation", {}).get("recorded_source_trace", {}).values():
                    for field in ("frozen_archive", "traceable_file"):
                        if trace.get(field):
                            validation_archives.append(Path(trace[field]).parent)
            sources = current_sources(task, root)
            output = task_output(task)
            output_exists = output.exists() and (output.is_file() or any(output.iterdir()))
            if previous and previous.get("status") == "completed" and not output_exists:
                raise RuntimeError("Previously completed task output is missing; preserved without automatic retraining: " + task["name"])
            if output_exists:
                try:
                    verified = validate_task(task, root, validation_archives)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    if (previous and previous.get("status") == "completed") or completion_claimed(task):
                        raise RuntimeError(f"Completed output integrity failed for {task['name']}; preserved, not retrained: {error}") from error
                    if not args.archive_partial:
                        raise RuntimeError(f"Partial output for {task['name']}: {error}. Inspect it, then use --archive-partial to rename this task's output without deletion") from error
                    archive = archive_partial(task, followup)
                    if archive:
                        state["archives"].append(archive)
                else:
                    if previous and previous.get("status") == "completed":
                        if previous.get("command") != task["command"]:
                            raise RuntimeError("Completed task command differs; preserved without rerunning: " + task["name"])
                        if previous.get("validation", {}).get("artifacts") != verified.get("artifacts"):
                            raise RuntimeError("Completed task artifacts changed: " + task["name"])
                        status = "verified_existing"
                    else:
                        preserve_recorded_sources(verified, followup, task["name"])
                        adopted = {"task": task["name"], "status": "completed", "adopted": True,
                                   "command": task["command"], "adopted_at": time.time(),
                                   "current_sources_at_adoption": sources, "validation": verified}
                        write_json(manifest_path, adopted)
                        status = "adopted_legacy_complete"
                    prior_sources = previous.get("source_sha256", previous.get("current_sources_at_adoption")) if previous else sources
                    state["completed"].append({"name": task["name"], "status": status,
                                                "current_sources_match_original_manifest": prior_sources == sources})
                    print("[skip-verified]", task["name"], status, flush=True)
                    continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archive = followup / "safe_runtime" / task["name"] / stamp
            snapshot = snapshot_sources(sources, archive)
            job = {"task": task["name"], "status": "started", "command": task["command"],
                   "source_sha256": sources, "runtime_sources": snapshot, "started_at": time.time()}
            if previous:
                previous_path = manifest_path.with_name(manifest_path.stem + ".previous-" + stamp + ".json")
                manifest_path.rename(previous_path)
            write_json(manifest_path, job)
            log = Path(task["log"])
            log.parent.mkdir(exist_ok=True, parents=True)
            if log.exists():
                log.rename(log.with_name(log.name + ".previous-" + stamp))
            print("[start]", task["name"], flush=True)
            with log.open("w") as stream:
                result = subprocess.run(task["command"], cwd=root / "code", env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                job.update({"status": "failed", "exit_code": result.returncode, "failed_at": time.time(), "log": str(log)})
                write_json(manifest_path, job)
                raise RuntimeError(f"Task {task['name']} exited {result.returncode}; log: {log}")
            if current_sources(task, root) != sources:
                raise RuntimeError("Source files changed while task ran: " + task["name"])
            try:
                verified = validate_task(task, root, [archive, *args.runtime_archive])
            except (ValueError, OSError, KeyError, TypeError) as error:
                job.update({"status": "failed_validation", "failed_at": time.time(), "error": str(error)})
                write_json(manifest_path, job)
                raise
            job.update({"status": "completed", "completed_at": time.time(), "validation": verified})
            write_json(manifest_path, job)
            state["completed"].append({"name": task["name"], "status": "executed_and_verified"})
            print("[done-verified]", task["name"], flush=True)
        state["running"] = None
        state["finished_at"] = time.time()
        write_json(state_path, state)
        print("SAFE_FOLLOWUP_COMPLETE", flush=True)
    except BaseException as error:
        state["failed"] = {"name": state["running"], "type": type(error).__name__, "message": str(error), "at": time.time()}
        state["running"] = None
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
