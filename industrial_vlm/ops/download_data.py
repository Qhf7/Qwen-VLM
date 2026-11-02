"""Download official archives with resume, integrity records and safe extraction."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    head = urllib.request.urlopen(urllib.request.Request(args.url, method="HEAD"), timeout=40)
    size = int(head.headers["Content-Length"])
    if not args.archive.exists() or args.archive.stat().st_size != size:
        parts = args.archive.with_suffix(args.archive.suffix + ".parts")
        parts.mkdir(exist_ok=True)
        chunk_size = 16 * 1024 * 1024
        spans = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]

        def fetch(span):
            start, end = span
            path = parts / str(start)
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(8):
                try:
                    request = urllib.request.Request(args.url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as output:
                        if response.status != 206:
                            raise RuntimeError("Server did not honor HTTP range request")
                        while data := response.read(1024 * 1024):
                            output.write(data)
                    if path.stat().st_size != end - start + 1:
                        raise RuntimeError("Incomplete download range")
                    return path
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(min(2 ** attempt, 30))

        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, span) for span in spans]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                print(json.dumps({"completed_parts": done, "total_parts": len(spans),
                                  "seconds": round(time.time() - started, 1)}), flush=True)
        with args.archive.open("wb") as output:
            for start, _ in spans:
                with (parts / str(start)).open("rb") as source:
                    while data := source.read(8 * 1024 * 1024):
                        output.write(data)
        import shutil
        shutil.rmtree(parts)
    digest = hashlib.sha256()
    with args.archive.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with tarfile.open(args.archive) as tar:
        members = tar.getmembers()
        tar.extractall(args.destination, members=members, filter="data")
    record = {"url": args.url, "bytes": args.archive.stat().st_size,
              "sha256": digest.hexdigest(), "members": len(members),
              "archive": str(args.archive), "destination": str(args.destination),
              "completed_unix": time.time(), "checksum_origin": "Computed after download; not a publisher signature."}
    target = args.destination / "download_provenance.json"
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
