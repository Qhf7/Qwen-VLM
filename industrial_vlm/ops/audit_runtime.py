"""Record the actual GPU runtime and validate local model shard structure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import struct
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = []
    keys = set()
    for name in sorted(set(index["weight_map"].values())):
        path = args.model / name
        with path.open("rb") as file:
            header_len = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_len))
            end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
            assert path.stat().st_size == 8 + header_len + end, f"Truncated shard: {path}"
            keys.update(k for k in header if k != "__metadata__")
            digest = hashlib.sha256()
            file.seek(0)
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        shards.append({"file": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    assert set(index["weight_map"]) == keys, "Index and shard tensors disagree."
    packages = {}
    for name in ["torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes",
                 "numpy", "Pillow", "safetensors", "huggingface-hub"]:
        packages[name] = importlib.metadata.version(name)
    record = {"python": platform.python_version(), "packages": packages,
              "gpu": torch.cuda.get_device_name(0),
              "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
              "torch_cuda": torch.version.cuda, "bf16_supported": torch.cuda.is_bf16_supported(),
              "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],text=True).strip(),
              "model": str(args.model), "model_config": json.loads((args.model / "config.json").read_text()),
              "shards": shards, "tensor_count": len(keys),
              "verification": "Shard headers, tensor index and file sizes checked; hashes record local bytes, not publisher signatures."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "model_config"}, indent=2))


if __name__ == "__main__":
    main()
