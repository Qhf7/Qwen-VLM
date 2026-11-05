#!/usr/bin/env python3
"""Read-only experiment inventory and source-split audit; standard library only.

Metadata files are hashed by default. Tensor/checkpoint files are inventoried by
size and are never deserialized; --hash-weights also computes their SHA256.
Reports describe observable metadata, not the provenance of a training dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import subprocess


METADATA_SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml", ".md", ".txt", ".log"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "site-packages", "node_modules", ".ssh"}
SECRET_WORDS = {"password", "passwd", "secret", "credential", "private_key", "access_token", "api_key"}
SOURCE_COLUMNS = ("source_path", "image_path", "path", "image", "filename", "file_name")
SAFE_RUN_KEYS = {
    "seed", "step", "steps", "max_steps", "epochs", "batch_size", "accum", "lr",
    "init_ckpt", "init_ckpt_path", "checkpoint", "checkpoint_path", "init_checkpoint",
    "imagebind_ckpt", "imagebind_ckpt_path", "qwen_ckpt", "vicuna_ckpt_path",
    "data_root", "train_split", "test_split", "split_csv", "cache_dir", "train_cache",
    "test_cache", "save_path", "lora", "lora_rank", "lora_alpha", "rank", "r",
    "region_loss", "pixel_loss", "normal_only", "max_sources", "max_train", "max_tgt_len",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def excluded_name(path: Path) -> bool:
    low = path.name.lower()
    return low == ".env" or low.startswith(".env.") or low.endswith((".pem", ".key")) or any(word in low for word in SECRET_WORDS)


def iter_files(root: Path):
    # Do not follow symlinks to external model stores or credential directories.
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS and not excluded_name(Path(d)))
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file() and not excluded_name(path):
                yield path


def safe_run_fields(value, prefix="") -> dict:
    fields = {}
    if not isinstance(value, dict):
        return fields
    for key, item in value.items():
        low = str(key).lower()
        if any(word in low for word in SECRET_WORDS):
            continue
        label = f"{prefix}.{key}" if prefix else str(key)
        if low in SAFE_RUN_KEYS and (item is None or isinstance(item, (str, int, float, bool))):
            fields[label] = item
        elif isinstance(item, dict) and low in {"args", "config", "training", "model", "metadata", "run"}:
            fields.update(safe_run_fields(item, label))
    return fields


def source_identity(value: str) -> str:
    # Preserve full source paths rather than conflating repeated basenames.
    return str(PurePosixPath(value.strip().replace("\\", "/")))


def read_split(path: Path) -> tuple[dict, set[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        source_column = next((key for key in SOURCE_COLUMNS if key in columns), None)
        if source_column is None:
            return {"path": str(path), "error": "No recognized source path column", "columns": columns}, set()
        paths = []
        classes = Counter()
        labels = Counter()
        blank_paths = 0
        rows = 0
        for row in reader:
            rows += 1
            value = row.get(source_column, "") or ""
            if value.strip():
                paths.append(source_identity(value))
            else:
                blank_paths += 1
            category = row.get("class_name", row.get("class", row.get("object", "")))
            if category:
                classes[str(category)] += 1
            label = row.get("label", row.get("is_anomalous", row.get("variant", "")))
            if label != "":
                labels[str(label)] += 1
    counts = Counter(paths)
    return {
        "path": str(path), "sha256": digest(path), "source_column": source_column,
        "rows": rows, "blank_paths": blank_paths, "unique_sources": len(counts),
        "repeated_sources": {path: count for path, count in counts.items() if count > 1},
        "class_rows": dict(sorted(classes.items())), "label_rows": dict(sorted(labels.items())),
    }, set(paths)


def git_state(root: Path) -> dict:
    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", choices=("metadata", "all"), default="metadata")
    parser.add_argument("--hash-weights", action="store_true", help="Stream tensor files for SHA256; never torch.load")
    parser.add_argument("--train-split", type=Path)
    parser.add_argument("--test-split", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Experiment directory does not exist: {root}")
    output = args.output.expanduser().resolve()
    files = []
    runs = []
    split_candidates = []
    for path in iter_files(root):
        if path.resolve() == output:
            continue
        suffix = path.suffix.lower()
        weight = suffix in WEIGHT_SUFFIXES
        if args.scope == "metadata" and suffix not in METADATA_SUFFIXES and not weight:
            continue
        row = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
               "kind": "tensor_or_checkpoint" if weight else "metadata"}
        if not weight or args.hash_weights:
            row["sha256"] = digest(path)
        else:
            row["sha256"] = None
            row["hash_status"] = "not_requested"
        files.append(row)
        if path.name in {"run.json", "args.json", "summary.json", "config.json", "training_config.json"} and path.stat().st_size < 2_000_000:
            try:
                fields = safe_run_fields(json.loads(path.read_text()))
                if fields:
                    runs.append({"file": row["path"], "fields": fields})
            except (ValueError, OSError) as exc:
                runs.append({"file": row["path"], "error": type(exc).__name__})
        if suffix == ".csv" and any(part in path.name.lower() for part in ("train", "test", "split", "val")):
            split_candidates.append(path)
    train = args.train_split.expanduser().resolve() if args.train_split else None
    test = args.test_split.expanduser().resolve() if args.test_split else None
    if train is None:
        candidates = [p for p in split_candidates if "train" in p.name.lower()]
        train = candidates[0] if len(candidates) == 1 else None
    if test is None:
        candidates = [p for p in split_candidates if "test" in p.name.lower()]
        test = candidates[0] if len(candidates) == 1 else None
    splits = {"candidates": [str(p.relative_to(root)) for p in split_candidates]}
    identities = {}
    for role, path in (("train", train), ("test", test)):
        if path:
            if path.is_file():
                splits[role], identities[role] = read_split(path)
            else:
                splits[role] = {"path": str(path), "error": "missing_file"}
    if "train" in identities and "test" in identities:
        overlap = sorted(identities["train"] & identities["test"])
        splits.update({"path_overlap_count": len(overlap), "path_overlap": overlap})
    else:
        splits["overlap_status"] = "not_checked: select exactly one train and test CSV"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "scope": args.scope, "hash_weights": args.hash_weights, "git": git_state(root),
        "files": files, "run_metadata": runs, "split_audit": splits,
        "checkpoint_inventory": [row for row in files if row["kind"] == "tensor_or_checkpoint"],
        "limitations": [
            "No tensor file is deserialized and no training or inference is run.",
            "Path overlap concerns these CSVs only; initial checkpoints need their own data provenance.",
            "Path isolation does not prove byte, near-duplicate, or physical-object isolation.",
            "Repeated sources may be intentional normal/synthetic variants; inspect the variant protocol.",
            "Initialization fields are reported when recorded; missing metadata is not proof of random initialization.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "files": len(files), "tensor_files": len(report["checkpoint_inventory"]),
                      "split_overlap": splits.get("path_overlap_count"), "run_metadata_files": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
