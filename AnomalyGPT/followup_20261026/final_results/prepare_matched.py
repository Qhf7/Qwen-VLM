"""Create explicit bridge initializations and source schedules for paired LM arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8388608), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.experiment
    args.out.mkdir(parents=True, exist_ok=True)
    cache = root / "cache_balanced_train48"
    names = [p.name for p in sorted(cache.glob("*.pt"))]
    if len(names) != 96:
        raise ValueError("Expected frozen 96-variant training cache")
    source = root / "ckpt/train_visa/pytorch_model.pt"
    state = torch.load(source, map_location="cpu", weights_only=True)
    retained = {k: v for k, v in state.items() if k.startswith(("image_decoder.", "prompt_learner."))}
    if len(retained) != 21:
        raise ValueError("Unexpected retained AnomalyGPT modules")
    protocol = {"seeds": [42, 43, 44], "steps": 16, "batch_size": 2, "max_tgt_len": 128,
                "lr": 0.001, "adamw_betas": [0.9, 0.95], "weight_decay": 0.001,
                "reference_mode": "none", "geometry": "square224",
                "init_checkpoint": str(source.resolve()), "init_sha256": digest(source),
                "cache_manifest_sha256": digest(cache / "manifest.json"),
                "models": ["Qwen3-VL-8B language frozen", "Qwen3-VL-8B language qkvo LoRA r8", "Vicuna-7B-v1.1 frozen"],
                "purpose": "same cache/order/bridge initialization and 16-step budget; language tokenizer and model version differ",
                "pretrained_scope": "released train_visa decoder and PromptLearner; complete pretraining source provenance unavailable",
                "runs": []}
    for seed in protocol["seeds"]:
        dest = args.out / f"seed{seed}"
        if dest.exists():
            raise FileExistsError(dest)
        dest.mkdir()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            projection = torch.nn.Linear(1024, 4096)
        bridge = {**retained, **{f"llama_proj.{k}": v.detach().to(torch.bfloat16) for k, v in projection.state_dict().items()}}
        torch.save(bridge, dest / "bridge_init.pt")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(names), generator=generator).tolist()
        batches = [[names[i] for i in indices[j:j + 2]] for j in range(0, 32, 2)]
        (dest / "sample_order.json").write_text(json.dumps({"batches": batches}, indent=2))
        protocol["runs"].append({"seed": seed, "bridge_init": str(dest / "bridge_init.pt"),
                                 "bridge_init_sha256": digest(dest / "bridge_init.pt"),
                                 "sample_order": str(dest / "sample_order.json"),
                                 "sample_order_sha256": digest(dest / "sample_order.json")})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
