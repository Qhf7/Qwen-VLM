"""Archive completed v2 execution sources and index the actual result hashes.

Example: python ops/archive_multitask_v2.py --root /root/autodl-tmp/industrial_vlm
Requires completed training/scoring, passed training verification, and a complete
scoring audit. Existing v2 delivery files require explicit --force to rebuild.
The original reports/code_snapshot.tar.gz and ops/source_manifest.json are never
written. Model and adapter weights are recorded by hash, never put in this tar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


ARMS = ("region_region", "decision_decision", "region_decision")
GENERATIONS = ("region_region_region", "region_region_decision", "decision_decision_decision",
               "region_decision_region", "region_decision_decision")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict:
    _require(path.is_file(), f"Required completed artifact is missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(result, dict), f"Expected a JSON object: {path}")
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(root: Path, path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"Expected a regular file: {path}")
    _require(path.resolve().is_relative_to(root), f"Path escapes project: {path}")
    return {"path": str(path.relative_to(root)), "sha256": _sha(path), "bytes": path.stat().st_size}


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _verify_runtime(root: Path, directory: Path, manifest_path: Path) -> None:
    manifest = _read(manifest_path)
    observed = {str(path.relative_to(directory)) for path in directory.rglob("*.py")}
    _require(observed == set(manifest["files"]), f"Frozen runtime inventory differs: {directory}")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        _require(path.resolve().is_relative_to(directory.resolve()), f"Unsafe runtime-manifest path: {relative}")
        _require(_sha(path) == expected, f"Frozen runtime bytes differ: {path}")
    if "runner_sha256" in manifest:
        _require(_sha(root / "ops/run_scores_v2.py") == manifest["runner_sha256"], "Scoring runner differs from frozen manifest")


def completed_result_index(root: Path) -> tuple[dict, list[Path]]:
    """Check terminal states and build metadata locators, without model loading."""
    experiment = root / "experiments/multitask_v2"
    state_path, scoring_state_path = experiment / "state.json", experiment / "scoring_state.json"
    verification_path = experiment / "verification.json"
    scoring_audit_path = root / "reports/multitask_v2/scoring_matrix.json"
    byte_audit_path = experiment / "source_byte_audit.json"
    state, scoring_state = _read(state_path), _read(scoring_state_path)
    verification, scoring_audit = _read(verification_path), _read(scoring_audit_path)
    _require(state.get("status") == "complete", "Training/generation pipeline is not complete; no delivery archive created")
    _require(scoring_state.get("status") == "complete", "Scoring pipeline is not complete; no delivery archive created")
    _require(verification.get("status") == "passed" and verification.get("checks")
             and all(value is True for value in verification["checks"].values()), "Training verification has not passed")
    _require(scoring_audit.get("status") == "complete" and scoring_audit.get("audit_status") == "passed"
             and scoring_audit.get("checks")
             and all(value is True for value in scoring_audit["checks"].values()),
             "Scoring matrix audit has not passed")
    byte_audit = _read(byte_audit_path)
    _require(byte_audit.get("status") == "passed" and byte_audit.get("unique_images") == 1008
             and byte_audit.get("confirmation_images_opened") is False, "Delivery image-byte audit is not complete")
    _require(state.get("confirmation_evaluated") is False and scoring_state.get("confirmation_evaluated") is False
             and verification.get("confirmation_evaluated") is False, "Completion metadata does not preserve unopened confirmation protocol")
    input_path = experiment / "input_hashes.json"
    for relative, expected in _read(input_path).items():
        path = root / relative
        _require(path.resolve().is_relative_to(root), f"Unsafe frozen-input path: {relative}")
        _require(_sha(path) == expected, f"Frozen experiment input changed: {relative}")
    _verify_runtime(root, experiment / "runtime_src", experiment / "training_source_manifest.json")
    _verify_runtime(root, experiment / "scoring_runtime_src", experiment / "scoring_source_manifest.json")
    index = {
        "completion_gates": [_record(root, path) for path in (state_path, scoring_state_path, verification_path, scoring_audit_path)],
        "training": {}, "adapters": {}, "generation_evaluations": {}, "boolean_scores": {}, "validation_thresholds": {},
        "figures": [],
    }
    for arm in ARMS:
        directory = root / "runs/multitask_v2" / arm
        summary = _read(directory / "summary.json")
        _require(summary.get("status") == "complete" and summary.get("optimizer_steps") == 192
                 and summary.get("microsteps") == 1536 and summary.get("reload_check", {}).get("passed") is True,
                 f"Incomplete/unverified training: {arm}")
        _require(_sha(directory / "exposures.jsonl") == summary["exposure_trace_sha256"], f"Exposure trace changed: {arm}")
        names = ("summary.json", "run_config.json", "parameter_audit.json", "exposures.jsonl", "metrics.jsonl")
        index["training"][arm] = {"status": "complete", "optimizer_steps": 192,
                                  "files": [_record(root, directory / name) for name in names]}
        adapter = directory / "adapter"
        _require((adapter / "adapter_model.safetensors").is_file(), f"Missing trained adapter: {arm}")
        index["adapters"][arm] = {"included_in_code_archive": False,
                                  "files": [_record(root, path) for path in sorted(adapter.rglob("*")) if path.is_file()]}
    for name in GENERATIONS:
        directory = root / "evaluations/multitask_v2" / name
        meta, metrics = _read(directory / "meta.json"), _read(directory / "metrics.json")
        n = _line_count(directory / "predictions.jsonl")
        _require(meta.get("status") == "complete" and meta.get("n_completed") == n == 120
                 and metrics.get("metadata", {}).get("complete") is True and metrics["overall"]["n"] == n,
                 f"Incomplete generation evaluation: {name}")
        arm = name.rsplit("_", 1)[0]
        for relative, expected in meta["config"]["adapter_files_sha256"].items():
            path = root / "runs/multitask_v2" / arm / "adapter" / relative
            _require(_sha(path) == expected, f"Adapter differs from generation evaluation: {name}:{relative}")
        index["generation_evaluations"][name] = {"status": "complete", "n": n,
                                                "files": [_record(root, directory / file) for file in ("meta.json", "metrics.json", "predictions.jsonl")]}
    for arm in ("base", *ARMS):
        for role in ("validation", "regression"):
            name = arm + "_" + role
            directory = root / "evaluations/multitask_v2_scores" / name
            meta, metrics = _read(directory / "meta.json"), _read(directory / "metrics.json")
            score_path = directory / "scores.jsonl"
            n = _line_count(score_path)
            _require(meta.get("status") == "complete" and meta.get("n_completed") == n == meta["config"]["n_expected"] == 120,
                     f"Incomplete boolean scoring: {name}")
            _require(metrics.get("metadata", {}).get("complete") is True
                     and meta["config"]["data_role"] == role, f"Scoring metrics/role mismatch: {name}")
            _require(_sha(score_path) == meta["scores_sha256"], f"Completed score bytes changed: {name}")
            index["boolean_scores"][name] = {"status": "complete", "n": n, "role": role,
                                           "files": [_record(root, directory / file) for file in ("meta.json", "metrics.json", "scores.jsonl")]}
        threshold_path = root / "evaluations/multitask_v2_scores" / (arm + "_threshold.json")
        threshold = _read(threshold_path)
        validation_scores = root / "evaluations/multitask_v2_scores" / (arm + "_validation") / "scores.jsonl"
        _require(threshold.get("kind") == "validation_normal_threshold"
                 and threshold["source_scores_sha256"] == _sha(validation_scores), f"Threshold is not tied to validation scores: {arm}")
        index["validation_thresholds"][arm] = _record(root, threshold_path)
    for stem in ("validation_comparison", "training_dynamics", "scoring_validation_curves", "scoring_regression_curves"):
        for extension in ("png", "pdf", "svg"):
            index["figures"].append(_record(root, root / "reports/multitask_v2/figures" / (stem + "." + extension)))
    required_evidence = [state_path, scoring_state_path, verification_path, input_path, byte_audit_path,
                         experiment / "training_source_manifest.json", experiment / "scoring_source_manifest.json"]
    return index, required_evidence


def archive_sources(root: Path, required_evidence: list[Path]) -> list[Path]:
    """Select execution sources, metadata, documentation; never select weights."""
    selected = set(required_evidence)
    selected.update((root / "src").rglob("*.py"))
    for pattern in ("*multitask_v2*.py", "*scores_v2*.py", "*score*.py"):
        selected.update((root / "ops").glob(pattern))
    selected.add(root / "ops/audit_v2_source_bytes.py")
    for relative in ("configs/multitask_v2", "prepared/multitask_v2"):
        directory = root / relative
        selected.update(path for path in directory.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"})
    for relative in ("experiments/multitask_v2/runtime_src", "experiments/multitask_v2/scoring_runtime_src"):
        selected.update((root / relative).rglob("*.py"))
    selected.update(path for path in (root / "reports/multitask_v2").rglob("*")
                    if path.is_file() and path.suffix in {".md", ".json"})
    # The original inventory/split locates image identities, and the old120
    # manifest is the explicitly seen regression population used by scoring.
    for relative in ("prepared/visa/manifest.jsonl", "prepared/visa/split.json",
                     "prepared/visa/test_balanced_120_decision.jsonl", "README.md", "DATA_PROTOCOL.md",
                     "PROJECT_CARD_ZH.md", "pyproject.toml", "requirements-runtime.txt"):
        path = root / relative
        _require(path.is_file(), f"Required source/documentation is missing: {relative}")
        selected.add(path)
    for relative in (".gitignore", "requirements.txt", "uv.lock", "ops/score_cpu_audit.log",
                     "ops/environment_full_freeze.txt", "ops/runtime_audit.json"):
        path = root / relative
        if path.is_file():
            selected.add(path)
    files = sorted(path for path in selected if "__pycache__" not in path.parts)
    _require(bool(files), "No source files selected")
    for path in files:
        _record(root, path)
        _require(path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}
                 and "models" not in path.relative_to(root).parts and "adapter" not in path.relative_to(root).parts,
                 f"Weight/model file selected for code archive: {path}")
    return files


def _publish(temporary: Path, destination: Path, force: bool) -> None:
    if force:
        os.replace(temporary, destination)
    else:
        # link is atomic and fails if another writer created the destination.
        os.link(temporary, destination)
        temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Explicitly rebuild existing v2 delivery files; never touches original snapshot/manifest")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    archive = root / "reports/multitask_v2/code_snapshot.tar.gz"
    manifest_path = root / "experiments/multitask_v2/delivery_manifest.json"
    if not args.force:
        _require(not archive.exists() and not manifest_path.exists(),
                 "Existing v2 archive/manifest preserved. Pass --force explicitly to rebuild v2 delivery")
    protected = [root / "reports/code_snapshot.tar.gz", root / "ops/source_manifest.json"]
    protected_before = {str(path.relative_to(root)): _sha(path) for path in protected if path.is_file()}
    result_index, required_evidence = completed_result_index(root)
    paths = archive_sources(root, required_evidence)
    records = [_record(root, path) for path in paths]
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".code_snapshot.", suffix=".tmp", dir=archive.parent, delete=False) as raw:
            temporary_archive = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as handle:
                    for path, record in zip(paths, records):
                        content = path.read_bytes()
                        _require(hashlib.sha256(content).hexdigest() == record["sha256"], f"Source changed during archive: {path}")
                        info = tarfile.TarInfo(record["path"])
                        info.size, info.mode, info.mtime = len(content), 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        handle.addfile(info, io.BytesIO(content))
        # Verify the bytes inside the tar against the per-file delivery ledger.
        by_path = {record["path"]: record for record in records}
        with tarfile.open(temporary_archive, "r:gz") as handle:
            _require(set(handle.getnames()) == set(by_path), "Archive inventory differs from source records")
            for member in handle:
                stored = handle.extractfile(member)
                _require(stored is not None, f"Archive member is not a regular file: {member.name}")
                with stored:
                    _require(hashlib.sha256(stored.read()).hexdigest() == by_path[member.name]["sha256"],
                             f"Archive member hash mismatch: {member.name}")
        protected_after = {str(path.relative_to(root)): _sha(path) for path in protected if path.is_file()}
        _require(protected_before == protected_after, "Original snapshot/source manifest changed during v2 archive")
        manifest = {
            "schema_version": "industrial-vlm-multitask-v2-delivery-v1", "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Current source plus frozen actual training/scoring runtimes, v2 data/config/protocol/report metadata; model/adapter weights and result rows are indexed separately",
            "archive": {"path": str(archive.relative_to(root)), "sha256": _sha(temporary_archive),
                        "bytes": temporary_archive.stat().st_size, "n_files": len(records)},
            "files": records, "actual_results": result_index,
            "original_artifacts_preserved": protected_before,
            "confirmation_predictions_created_or_read": False,
            "archive_self_reference": "Delivery manifest and code_snapshot.tar.gz are excluded from the tar; all archived source bytes are verified before publication",
        }
        with tempfile.NamedTemporaryFile(prefix=".delivery_manifest.", suffix=".tmp", dir=manifest_path.parent,
                                         mode="w", encoding="utf-8", delete=False) as handle:
            temporary_manifest = Path(handle.name)
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        _publish(temporary_archive, archive, args.force)
        temporary_archive = None
        _publish(temporary_manifest, manifest_path, args.force)
        temporary_manifest = None
        print(json.dumps({"status": "complete", "archive": str(archive), "archive_sha256": manifest["archive"]["sha256"],
                          "files": len(records), "delivery_manifest": str(manifest_path), "original_artifacts_preserved": True}, indent=2))
    finally:
        for path in (temporary_archive, temporary_manifest):
            if path is not None:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
