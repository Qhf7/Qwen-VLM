"""Archive the delivered source/configuration and record their exact SHA256."""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = []
    for directory in ("src", "ops", "configs"):
        paths += [p for p in (root / directory).rglob("*") if p.is_file() and
                  "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".toml", ".txt"} and
                  p.name not in {"source_manifest.json"}]
    paths += [root / name for name in ("README.md", "DATA_PROTOCOL.md", "PROJECT_CARD_ZH.md",
                                       "pyproject.toml", "requirements-runtime.txt", ".gitignore")]
    paths = sorted(set(paths))
    records = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    archive = root / "reports/code_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in paths:
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Delivered source, configuration, documentation and operations audits; data manifests and trained weights archived separately.",
              "files": records, "archive": str(archive),
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (root / "ops/source_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(records), "archive": str(archive), "sha256": result["archive_sha256"]}))


if __name__ == "__main__":
    main()
