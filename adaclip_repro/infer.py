"""Run official AdaCLIP weights on one or more local images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .model import DEFAULT_CLIP_WEIGHT, ROOT, build_model


def parse_item(value: str) -> tuple[Path, str]:
    try:
        path, class_name = value.rsplit("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("item format must be IMAGE_PATH::CLASS_NAME") from exc
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise argparse.ArgumentTypeError(f"image not found: {image_path}")
    if not class_name.strip():
        raise argparse.ArgumentTypeError("CLASS_NAME cannot be empty")
    return image_path, class_name.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True, type=parse_item,
                        help="repeatable IMAGE_PATH::CLASS_NAME")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "third_party/AdaCLIP/weights/pretrained_all.pth")
    parser.add_argument("--clip-weight", type=Path, default=DEFAULT_CLIP_WEIGHT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adaclip/inference")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    model, device = build_model(args.checkpoint, device=args.device,
                                clip_weight=args.clip_weight, seed=args.seed,
                                precision=args.precision)
    load_seconds = time.perf_counter() - load_start
    rows = []

    for index, (image_path, class_name) in enumerate(args.item):
        pil_image = Image.open(image_path).convert("RGB")
        original_bgr = cv2.imread(str(image_path))
        original_bgr = cv2.resize(original_bgr, (model.image_size, model.image_size))
        dtype = torch.float16 if args.precision == "fp16" else torch.float32
        image_tensor = model.preprocess(pil_image).unsqueeze(0).to(device=device, dtype=dtype)

        start = time.perf_counter()
        with torch.inference_mode():
            anomaly_map, anomaly_score = model.clip_model(image_tensor, [class_name], aggregation=True)
        seconds = time.perf_counter() - start
        amap = gaussian_filter(
            anomaly_map[0].detach().float().cpu().numpy(), sigma=4
        ).astype(np.float32)
        score = float(anomaly_score[0].detach().cpu())

        stem = f"{index:03d}_{image_path.stem}"
        np.save(args.output / f"{stem}_map.npy", amap)
        gray = np.clip(amap * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.output / f"{stem}_map.png"), gray)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heat, 0.5, original_bgr, 0.5, 0.0)
        cv2.imwrite(str(args.output / f"{stem}_overlay.png"), overlay)
        rows.append({
            "image_path": str(image_path),
            "class_name": class_name,
            "anomaly_score": score,
            "map_min": float(amap.min()),
            "map_max": float(amap.max()),
            "map_mean": float(amap.mean()),
            "inference_seconds": seconds,
            "map_npy": str((args.output / f"{stem}_map.npy").resolve()),
        })

    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "clip_weight": str(args.clip_weight.expanduser().resolve()),
        "device": device,
        "image_size": model.image_size,
        "prompting_depth": 4,
        "prompting_length": 5,
        "prompting_type": "SD",
        "prompting_branch": "VL",
        "use_hsf": True,
        "k_clusters": 20,
        "seed": args.seed,
        "precision": args.precision,
        "hsf_order_sensitive": True,
        "model_load_seconds": load_seconds,
        "predictions": rows,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['image_path']}\tscore={row['anomaly_score']:.6f}\t{row['inference_seconds']:.3f}s")


if __name__ == "__main__":
    main()
