"""Freeze official VisA test sources and cache ImageBind features, without labels in model inputs."""
from __future__ import annotations

import argparse
import collections
import csv
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--confirmation-sources", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    root = args.experiment.resolve()
    data_root = root / "visa_data"
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "code"))
    from model.ImageBind import imagebind_model
    from model.ImageBind.models.imagebind_model import ModalityType
    from cache_imagebind_features_v2 import text_features
    from datasets.visa import describles

    official_csv = data_root / "split_csv/1cls.csv"
    official = list(csv.DictReader(official_csv.open()))
    confirmation = [json.loads(s) for s in args.confirmation_sources.read_text().splitlines() if s.strip()]
    excluded_paths = {r["image_relative"] for r in confirmation}
    excluded_hashes = {r["image_sha256"] for r in confirmation}
    previous_test = json.loads((root / "cache_balanced_test24/manifest.json").read_text())["records"]
    previous_paths = set()
    for rec in previous_test:
        parts = Path(rec["source_path"]).parts
        idx = parts.index(rec["class_name"])
        previous_paths.add(str(Path(*parts[idx:])))
    train_rows = list(csv.DictReader((root / "splits/balanced_train.csv").open()))
    train_paths = {r["image"] for r in train_rows}
    train_hashes = {sha256(data_root / x) for x in train_paths}
    selected = []
    exclusions = collections.Counter()
    for row in official:
        if row["split"] != "test":
            continue
        path = row["image"]
        if path in excluded_paths:
            exclusions["native_confirmation_path"] += 1
            continue
        if path in previous_paths:
            exclusions["previous_synthetic_test_source"] += 1
            continue
        if path in train_paths:
            raise ValueError("official test overlaps current training source")
        digest = sha256(data_root / path)
        if digest in excluded_hashes:
            exclusions["native_confirmation_hash"] += 1
            continue
        if digest in train_hashes:
            raise ValueError("real test overlaps training image bytes")
        if row["label"] not in {"normal", "anomaly"}:
            raise ValueError("unexpected official label")
        if row["label"] == "anomaly" and not (data_root / row["mask"]).is_file():
            raise FileNotFoundError(row["mask"])
        selected.append({**row, "image_sha256": digest, "label_int": int(row["label"] == "anomaly")})
    selected.sort(key=lambda x: (x["object"], x["label_int"], x["image"]))
    if args.limit:
        # Smoke covers both labels in every class; the full protocol is deterministic.
        by_group = collections.defaultdict(list)
        for row in selected:
            by_group[(row["object"], row["label_int"])].append(row)
        selected = [r for key in sorted(by_group) for r in by_group[key][:args.limit]]
    frozen_path = args.out / "sources.json"
    frozen = {"official_csv_sha256": sha256(official_csv), "confirmation_metadata_sha256": sha256(args.confirmation_sources),
              "geometry": "square224_RGB_bicubic_mask_index_positive_nearest", "reference_mode": "none",
              "exclude_previous_synthetic_test_sources": True, "exclusions": dict(exclusions),
              "source_count": len(selected), "per_class_label": dict(collections.Counter(f'{r["object"]}:{r["label"]}' for r in selected)), "records": selected}
    if frozen_path.exists():
        old = json.loads(frozen_path.read_text())
        if old != frozen:
            raise ValueError("existing frozen real-evaluation sources differ")
    else:
        write_json(frozen_path, frozen)
    print("[split]", frozen["source_count"], frozen["exclusions"], flush=True)
    torch.manual_seed(20261026)
    np.random.seed(20261026)
    random.seed(20261026)
    model, _ = imagebind_model.imagebind_huge({"device": 0})
    state = torch.load(root / "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth", map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model = model.to("cuda:0", dtype=torch.float16).eval()
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    prototypes = {}
    records = []
    for i, row in enumerate(selected):
        path = args.out / f"{i:05d}.pt"
        record = {"file": path.name, "source_path": row["image"], "class_name": row["object"], "label": row["label_int"], "image_sha256": row["image_sha256"]}
        if not path.exists():
            rgb = Image.open(data_root / row["image"]).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            if row["label_int"]:
                raw = np.asarray(Image.open(data_root / row["mask"]))
                if raw.ndim != 2:
                    raise ValueError("VisA mask must contain palette/index labels")
                mask = np.asarray(Image.fromarray((raw > 0).astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST)).copy()
            else:
                mask = np.zeros((224, 224), dtype=np.uint8)
            with torch.inference_mode():
                emb = model({ModalityType.VISION: norm(rgb).unsqueeze(0).cuda().half()})[ModalityType.VISION]
                if row["object"] not in prototypes:
                    prototypes[row["object"]] = text_features(model, row["object"], "cuda:0")[0]
            item = {**record, "global_feature": emb[0][0].cpu().half(),
                    "patch_features": [x.cpu().half() for x in emb[1]], "text_features": prototypes[row["object"]].cpu().half(),
                    "mask": torch.from_numpy(mask).unsqueeze(0), "question": describles[row["object"]] + " Is there any anomaly in the image?"}
            temp = Path(str(path) + ".tmp")
            torch.save(item, temp)
            os.replace(temp, path)
        records.append(record)
        if (i + 1) % 50 == 0 or i + 1 == len(selected):
            write_json(args.out / "manifest.json", {"records": records, "completed": i + 1 == len(selected), "geometry": frozen["geometry"], "reference_mode": "none"})
            print(f"[cache] {i + 1}/{len(selected)}", flush=True)
    print("[done]", len(records), flush=True)


if __name__ == "__main__":
    main()
