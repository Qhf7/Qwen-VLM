"""Convert the existing tensor-only Vicuna shards using a modern weights-only loader."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8388608), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        raise RuntimeError("Conversion requires PyTorch >=2.6")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    index = json.loads((args.source / "pytorch_model.bin.index.json").read_text())
    mapping, records = {}, []
    for old_name in sorted(set(index["weight_map"].values())):
        source = args.source / old_name
        new_name = old_name.replace("pytorch_model-", "model-").replace(".bin", ".safetensors")
        tensors = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        expected = {k for k, v in index["weight_map"].items() if v == old_name}
        if set(tensors) != expected:
            raise ValueError("Weight shard/index tensor set mismatch")
        tensors = {k: v.detach().contiguous() for k, v in tensors.items()}
        destination = args.out / new_name
        save_file(tensors, destination, metadata={"format": "pt"})
        with safe_open(destination, framework="pt", device="cpu") as loaded:
            for key, value in tensors.items():
                if not torch.equal(value, loaded.get_tensor(key)):
                    raise ValueError("Converted tensor differs: " + key)
                mapping[key] = new_name
        records.append({"source": str(source), "source_sha256": sha256(source), "destination": new_name,
                        "destination_sha256": sha256(destination), "tensor_count": len(tensors), "all_tensors_equal": True})
        print("[converted]", new_name, len(tensors), flush=True)
        del tensors
    for name in ["config.json", "generation_config.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy2(args.source / name, args.out / name)
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": index["metadata"], "weight_map": mapping}, indent=2))
    (args.out / "conversion_audit.json").write_text(json.dumps({"torch": torch.__version__, "weights_only": True,
        "source": str(args.source.resolve()), "records": records, "total_tensors": len(mapping), "all_tensors_equal": True}, indent=2))
    print("CONVERSION_COMPLETE", len(mapping), flush=True)


if __name__ == "__main__":
    main()
