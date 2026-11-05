"""Read-only verification of a completed v2 delivery, without model/data loading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = read(root / "experiments/multitask_v2/delivery_manifest.json")
    require(manifest["status"] == "complete", "Delivery is not complete")
    require(manifest["confirmation_predictions_created_or_read"] is False,
            "Unexpected confirmation use")
    result = manifest["actual_results"]
    for key, expected in (("training", 3), ("adapters", 3), ("generation_evaluations", 5),
                          ("boolean_scores", 8), ("validation_thresholds", 4), ("figures", 12)):
        require(len(result[key]) == expected, f"Unexpected result inventory: {key}")

    indexed = {}
    for record in records(manifest):
        relative = record["path"]
        path = root / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root),
                f"Unsafe delivery path: {relative}")
        require(path.is_file() and not path.is_symlink(), f"Missing regular file: {relative}")
        expected = (record["sha256"], record["bytes"])
        if relative in indexed:
            require(indexed[relative] == expected, f"Conflicting records: {relative}")
            continue
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                f"File differs from delivery: {relative}")
        indexed[relative] = expected

    by_path = {record["path"]: record for record in manifest["files"]}
    require(len(by_path) == len(manifest["files"]) == manifest["archive"]["n_files"],
            "Duplicate/missing archive source record")
    archive = root / manifest["archive"]["path"]
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        require(len(members) == len(by_path) and {item.name for item in members} == set(by_path),
                "Archive inventory differs from delivery")
        for member in members:
            expected = by_path[member.name]
            require(member.isfile() and member.size == expected["bytes"],
                    f"Invalid archive member: {member.name}")
            stored = handle.extractfile(member)
            require(stored is not None, f"Unreadable archive member: {member.name}")
            with stored:
                require(hashlib.sha256(stored.read()).hexdigest() == expected["sha256"],
                        f"Archive member differs: {member.name}")
    for relative, expected in manifest["original_artifacts_preserved"].items():
        path = root / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root)
                and path.is_file() and sha(path) == expected,
                f"First-round artifact differs: {relative}")

    verification = read(root / "experiments/multitask_v2/verification.json")
    scoring = read(root / "reports/multitask_v2/scoring_matrix.json")
    for name, audit, status in (("training", verification, "passed"), ("scoring", scoring, "complete")):
        require(audit["status"] == status and audit["checks"]
                and all(value is True for value in audit["checks"].values()), f"Failed audit: {name}")
    require(scoring["audit_status"] == "passed", "Scoring audit did not pass")
    byte_audit = read(root / "experiments/multitask_v2/source_byte_audit.json")
    require(byte_audit["status"] == "passed" and byte_audit["unique_images"] == 1008
            and byte_audit["confirmation_images_opened"] is False, "Image-byte audit did not pass")
    # Only archived metadata and result files are opened, never raw images,
    # foundation weights, CUDA, or confirmation predictions.
    print(json.dumps({"status": "passed", "root": str(root), "indexed_files_verified": len(indexed),
                      "archive_members_verified": len(by_path), "archive_sha256": manifest["archive"]["sha256"],
                      "training_checks": len(verification["checks"]), "scoring_checks": len(scoring["checks"]),
                      "original_artifacts_preserved": True, "confirmation_images_or_predictions_opened": False},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
