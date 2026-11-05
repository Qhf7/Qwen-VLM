"""Verify delivered training/validation/regression image bytes; do not open confirmation images."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    inventory = {r["source_id"]: r for r in rows(ROOT / "prepared/visa/manifest.jsonl")}
    groups = {"training": ROOT / "prepared/multitask_v2/train_region_region.jsonl",
              "validation": ROOT / "prepared/multitask_v2/val_decision_120.jsonl",
              "regression": ROOT / "prepared/visa/test_balanced_120_decision.jsonl"}
    selected = {name: {r["source_id"] for r in rows(path)} for name, path in groups.items()}
    records = []
    for source in sorted(set.union(*selected.values())):
        row = inventory[source]
        path = Path(row["image"])
        digest = hashlib.sha256()
        before = path.stat()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Image changed during audit: {source}")
        if digest.hexdigest() != row["image_sha256"]:
            raise RuntimeError(f"Image bytes disagree with original inventory: {source}")
        records.append({"source_id": source, "image": str(path), "sha256": digest.hexdigest(),
                        "bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
    result = {"status": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Image byte identity at delivery; actual training order is recorded separately",
              "group_counts": {name: len(ids) for name, ids in selected.items()},
              "unique_images": len(records), "confirmation_images_opened": False, "images": records}
    (ROOT / "experiments/multitask_v2/source_byte_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "images"}))


if __name__ == "__main__":
    main()
