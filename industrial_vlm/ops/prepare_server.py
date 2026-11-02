"""One-time, explicitly scoped cleanup for this user's new industrial VLM project."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path("/root/autodl-tmp")
OLD = BASE / "vlm"
NEW = BASE / "industrial_vlm"
KEEP_MODEL = "Qwen3-VL-8B-Instruct"


def size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(root) / name
            if not item.is_symlink():
                total += item.stat().st_size
    return total


def main() -> None:
    assert BASE.resolve() == Path("/root/autodl-tmp")
    assert OLD.is_dir()
    model = OLD / "models" / KEEP_MODEL
    destination = NEW / "models" / KEEP_MODEL
    assert model.is_dir() or destination.is_dir()
    (NEW / "ops").mkdir(parents=True, exist_ok=True)
    (NEW / "models").mkdir(exist_ok=True)
    (NEW / "datasets").mkdir(exist_ok=True)
    names = ["Dataset", "artifacts", "seg_cross_part", "data", "docs"]
    targets = [OLD / name for name in names]
    targets += [BASE / "vlm-five-view-retired-20260905", BASE / "archive_c2"]
    targets += [p for p in (OLD / "models").iterdir() if p.name != KEEP_MODEL]
    # Old loose code/config/log files belong to the retired project. Preserve its
    # environment until a fully recorded runtime is established for this project.
    targets += [p for p in OLD.iterdir() if p.is_file() and p.name != ".DS_Store"]
    before = {
        "created_unix": time.time(),
        "authorization": "User requested deletion of unrelated previous server datasets and code.",
        "preserve": [str(destination), str(OLD / ".venv")],
        "delete": [{"path": str(p), "bytes": size_bytes(p) if p.is_dir() else p.stat().st_size}
                   for p in targets if p.exists()],
        "disk_before": subprocess.check_output(["df", "-h", str(BASE)], text=True),
    }
    (NEW / "ops" / "cleanup_manifest.json").write_text(json.dumps(before, indent=2))
    if model.exists():
        assert not destination.exists(), "Never overwrite an existing model."
        model.rename(destination)
    for p in targets:
        assert p.is_relative_to(BASE) and p != BASE and p != OLD
        assert not p.is_symlink(), f"Review unexpected symlink: {p}"
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    before["disk_after"] = subprocess.check_output(["df", "-h", str(BASE)], text=True)
    before["completed_unix"] = time.time()
    (NEW / "ops" / "cleanup_manifest.json").write_text(json.dumps(before, indent=2))
    print(json.dumps(before, indent=2))


if __name__ == "__main__":
    main()
