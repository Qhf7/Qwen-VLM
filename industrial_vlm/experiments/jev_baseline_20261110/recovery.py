"""Restore only the frozen Jev development queries and normal supports.

Images come from an immutable public VisA mirror and must match the historical
source-byte audit before they can be installed. No model API or credentials are
used. The confirmation split is neither read nor downloaded.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from datetime import datetime, timezone
import urllib.parse
import urllib.request

COMMIT = "912ae1aa35ac0791e64524a42342850af1605363"
MIRROR = "imaadd05/visa-anomaly-detection"
BASE_URL = f"https://huggingface.co/datasets/{MIRROR}/resolve/{COMMIT}/"
QUERY_MANIFEST = "industrial_vlm/prepared/multitask_v2/val_decision_120.jsonl"
TRAIN_MANIFEST = "industrial_vlm/prepared/multitask_v2/train_decision_decision.jsonl"
BYTE_AUDIT = "industrial_vlm/experiments/multitask_v2/source_byte_audit.json"
SELECTION = "sha256('jev-support-v1|42|' + source_id), ascending, first 4 per category"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def prepare(source_root: Path) -> tuple[list[dict], dict]:
    queries = jsonl(source_root / QUERY_MANIFEST)
    train = jsonl(source_root / TRAIN_MANIFEST)
    audit = json.loads((source_root / BYTE_AUDIT).read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("Historical source-byte audit is not passed")
    historical = {row["source_id"]: row for row in audit["images"]}
    if len(queries) != 120 or len({row["source_id"] for row in queries}) != 120:
        raise ValueError("Expected exactly 120 unique frozen query sources")
    query_ids = {row["source_id"] for row in queries}
    query_categories = {row["category"] for row in queries}
    if len(query_categories) != 12:
        raise ValueError("Expected 12 query categories")
    unique_train = {}
    for row in train:
        sid = row["source_id"]
        if sid in unique_train:
            old = unique_train[sid]
            if any(old[key] != row[key] for key in ("image", "category", "is_anomalous")):
                raise ValueError("Conflicting repeated training source")
        unique_train[sid] = row
    if query_ids & set(unique_train):
        raise ValueError("Training/query source overlap")
    supports = []
    for category in sorted(query_categories):
        candidates = [row for row in unique_train.values()
                      if row["category"] == category and row["is_anomalous"] is False]
        candidates.sort(key=lambda row: (sha256_bytes(("jev-support-v1|42|" + row["source_id"]).encode()),
                                         row["source_id"]))
        if len(candidates) < 4:
            raise ValueError(f"Insufficient normal training supports: {category}")
        supports.extend(candidates[:4])
    selected = []
    for role, rows in (("query", queries), ("normal_support", supports)):
        for row in rows:
            sid = row["source_id"]
            origin = historical.get(sid)
            if origin is None or origin["image"] != row["image"]:
                raise ValueError(f"Missing or mismatched historical byte provenance: {sid}")
            if sid != "visa_" + origin["sha256"]:
                raise ValueError("Source ID does not match historical SHA256")
            relative = row["image"].split("/VisA/", 1)[1]
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or not relative.endswith(".JPG"):
                raise ValueError("Unsafe or unexpected source path")
            selected.append({"role": role, "source_id": sid, "category": row["category"],
                             "relative_path": relative, "expected_sha256": origin["sha256"],
                             "expected_bytes": origin["bytes"],
                             "url": BASE_URL + urllib.parse.quote(relative, safe="/")})
    if len(selected) != 168 or len({row["relative_path"] for row in selected}) != 168:
        raise ValueError("Expected 168 distinct selected images")
    if {row["expected_sha256"] for row in selected[:120]} & {row["expected_sha256"] for row in selected[120:]}:
        raise ValueError("Query/support byte identity overlap")
    meta = {"query_count": 120, "support_count": 48, "support_selection": SELECTION,
            "support_source_ids": {category: [row["source_id"] for row in supports
                                                   if row["category"] == category]
                                   for category in sorted(query_categories)},
            "input_sha256": {name: sha256_bytes((source_root / name).read_bytes())
                               for name in (QUERY_MANIFEST, TRAIN_MANIFEST, BYTE_AUDIT)},
            "confirmation_images_opened": False, "confirmation_manifest_read": False,
            "query_support_source_overlap": 0, "query_support_byte_overlap": 0}
    return selected, meta


def recover_one(row: dict, image_root: Path, attempts: int, timeout: float) -> dict:
    result = dict(row)
    destination = image_root / row["relative_path"]
    result["local_path"] = str(destination)
    if destination.exists():
        data = destination.read_bytes()
        if len(data) == row["expected_bytes"] and sha256_bytes(data) == row["expected_sha256"]:
            return {**result, "status": "cached_verified", "attempts": 0,
                    "bytes": len(data), "sha256": sha256_bytes(data)}
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "Qwen-VLM-VisA-Recovery/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected HTTP status {response.status}")
                data = response.read(row["expected_bytes"] + 1)
            actual_sha = sha256_bytes(data)
            if len(data) != row["expected_bytes"] or actual_sha != row["expected_sha256"]:
                raise ValueError(f"Rejected byte mismatch: size={len(data)}, sha256={actual_sha}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(data)
                os.replace(temp, destination)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            return {**result, "status": "downloaded_verified", "attempts": attempt,
                    "bytes": len(data), "sha256": actual_sha}
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"[:500]
            if attempt < attempts:
                time.sleep(attempt)
    return {**result, "status": "failed", "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--attempts", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 60:
        parser.error("timeout must be between 1 and 60 seconds")
    directory = Path(__file__).resolve().parent
    image_root = directory / "data_images" / "VisA"
    selected, meta = prepare(args.source_root.resolve())
    started = datetime.now(timezone.utc).isoformat()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(recover_one, row, image_root, args.attempts, args.timeout) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            if len(records) % 12 == 0 or record["status"] == "failed":
                print(json.dumps({"completed": len(records), "expected": len(selected),
                                  "failed": sum(row["status"] == "failed" for row in records)}, ensure_ascii=False), flush=True)
    order = {row["source_id"]: i for i, row in enumerate(selected)}
    records.sort(key=lambda row: order[row["source_id"]])
    failed = sum(row["status"] == "failed" for row in records)
    output = {"schema": "jev-visa-recovery-v1", "status": "complete" if not failed else "incomplete",
              "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
              "source_repository": MIRROR, "source_commit": COMMIT,
              "official_source": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
              "workers": args.workers, "attempt_limit": args.attempts, "timeout_seconds": args.timeout,
              **meta, "verified_count": len(records) - failed, "failed_count": failed,
              "verified_bytes": sum(row.get("bytes", 0) for row in records),
              "images": records}
    atomic_json(directory / "recovery_audit.json", output)
    print(json.dumps({key: output[key] for key in ("status", "verified_count", "failed_count", "verified_bytes")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
