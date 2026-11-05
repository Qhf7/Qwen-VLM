"""Archive experiment metadata and predictions without copying weights/caches.

The manifest records the pipeline state at snapshot time. A snapshot is never
itself evidence that an experiment finished. Source files are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile


METADATA_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".csv", ".log", ".txt", ".yaml", ".yml", ".png", ".pdf", ".svg"}
LARGE_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".npz", ".npy"}
OMIT_DIRECTORIES = {"__pycache__", ".git", ".ssh", "progress_results", "server_snapshot", "final_results", "restart_results", "main_complete_results", "local_archives", "archives"}
SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519", "known_hosts", "credentials", "secrets"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def archive(root, output):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_path = output.with_name(output.name.removesuffix(".tar.gz") + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new archive path; existing snapshots are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_state, legacy_state = read_json(root / "safe_state.json"), read_json(root / "state.json")
    inventory, retained_large, skipped = [], [], Counter()
    manifest = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "source_sha256": sha256(Path(__file__).read_bytes()),
        "effective_pipeline_state": safe_state if safe_state is not None else legacy_state,
        "scope": "Metadata, source snapshots, logs, predictions and reports; no tensor weights, feature caches or pixel arrays",
        "completion_rule": "Snapshot state and experiment audits determine completeness; an archive alone does not",
        "large_artifacts": "Retained at source paths; authoritative recorded hashes remain in included run/audit JSON files",
        "files": inventory, "large_files_retained_at_source": retained_large,
    }
    with tempfile.TemporaryDirectory(prefix="followup-archive-", dir=output.parent) as temporary:
        temporary_archive = Path(temporary) / "snapshot.tar.gz"
        with tarfile.open(temporary_archive, "w:gz") as tar:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part in OMIT_DIRECTORIES for part in relative.parts):
                    skipped["omitted_directory"] += 1
                    continue
                if path == output or path == manifest_path:
                    continue
                if path.is_symlink():
                    skipped["symlink"] += 1
                    continue
                if not path.is_file():
                    continue
                if relative.as_posix() == "_archive_manifest.json":
                    skipped["previous_archive_manifest"] += 1
                    continue
                lower = path.name.lower()
                if lower in SENSITIVE_NAMES or any(word in lower for word in ("password", "credential", "secret", "private_key")):
                    skipped["sensitive_filename"] += 1
                    continue
                if path.suffix.lower() in LARGE_SUFFIXES:
                    retained_large.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
                    continue
                if path.suffix.lower() not in METADATA_SUFFIXES:
                    skipped["non_metadata_extension"] += 1
                    continue
                data = path.read_bytes()
                item = {"path": relative.as_posix(), "bytes": len(data), "sha256": sha256(data)}
                inventory.append(item)
                info = tarfile.TarInfo(item["path"])
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
            manifest["skipped_counts"] = dict(skipped)
            manifest["included_file_count"] = len(inventory)
            encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
            info = tarfile.TarInfo("_archive_manifest.json")
            info.size, info.mode = len(encoded), 0o644
            tar.addfile(info, io.BytesIO(encoded))
        # Verify every archived byte against the manifest before publishing.
        with tarfile.open(temporary_archive) as tar:
            for item in inventory:
                actual = tar.extractfile(item["path"]).read()
                if len(actual) != item["bytes"] or sha256(actual) != item["sha256"]:
                    raise ValueError(f"Archive integrity failure: {item['path']}")
        temporary_archive.rename(output)
    manifest_path.write_bytes(encoded)
    return {"archive": str(output), "manifest": str(manifest_path),
            "archive_sha256": sha256(output.read_bytes()), "included_file_count": len(inventory),
            "large_files_retained": len(retained_large),
            "pipeline_running": (manifest["effective_pipeline_state"] or {}).get("running")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.name.endswith(".tar.gz"):
        parser.error("--out must end in .tar.gz")
    print(json.dumps(archive(args.root, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
