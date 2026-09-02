"""Evaluate AdaCLIP on one of the 14 datasets used by the paper.

This runner never selects a checkpoint on the target dataset.  It loads one
fixed checkpoint, performs one evaluation pass, and reports both paper metrics
and stricter sparse-defect diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .metrics import binary_metrics, macro_average, pixel_metrics
from .model import DEFAULT_CLIP_WEIGHT, OFFICIAL_ROOT, ROOT, build_model


SUPPORTED = (
    "mvtec", "visa", "mpdd", "btad", "sdd", "dagm", "dtd",
    "headct", "brain_mri", "br35h", "isic", "colondb", "clinicdb", "tn3k",
)


def _percent_dict(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        result[key] = float(value * 100.0) if np.isfinite(value) else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="parent of mvtec_anomaly_detection, VisA_20220922, ...")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, default=DEFAULT_CLIP_WEIGHT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--limit", type=int, default=0,
                        help="debug only; 0 evaluates the complete paper dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/adaclip/paper_eval")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    args.output.mkdir(parents=True, exist_ok=True)

    model, device = build_model(args.checkpoint, device=args.device,
                                clip_weight=args.clip_weight, seed=args.seed,
                                precision=args.precision)
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    # Dataset roots are computed at import time, so override config first.
    import config
    config.DATA_ROOT = str(data_root)
    from dataset import get_data

    class_names, dataset, dataset_roots = get_data(
        dataset_type_list=args.dataset,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False,
    )
    if args.limit:
        # Deterministic balanced smoke subset. Taking the first N is unsafe
        # because official meta files are grouped by defect species.
        positive = [i for i, row in enumerate(dataset.data_all) if row["anomaly"]]
        negative = [i for i, row in enumerate(dataset.data_all) if not row["anomaly"]]
        n_positive = min(len(positive), args.limit // 2)
        n_negative = min(len(negative), args.limit - n_positive)
        indices = positive[:n_positive] + negative[:n_negative]
        if len(indices) < args.limit:
            remaining = [i for i in range(len(dataset)) if i not in set(indices)]
            indices.extend(remaining[:args.limit - len(indices)])
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    by_class = defaultdict(lambda: {"labels": [], "scores": [], "masks": [], "maps": []})
    latencies = []
    with torch.inference_mode():
        for item in tqdm(loader, desc=f"AdaCLIP {args.dataset}"):
            dtype = torch.float16 if args.precision == "fp16" else torch.float32
            image = item["img"].to(device=device, dtype=dtype)
            names = item["cls_name"]
            start = time.perf_counter()
            amap, score = model.clip_model(image, names, aggregation=True)
            latencies.append(time.perf_counter() - start)
            amap = gaussian_filter(
                amap[0].detach().float().cpu().numpy(), sigma=4
            ).astype(np.float32)
            mask = (item["img_mask"][0, 0].numpy() > 0.5)
            name = names[0]
            by_class[name]["labels"].append(int(item["anomaly"][0]))
            by_class[name]["scores"].append(float(score[0].detach().cpu()))
            by_class[name]["masks"].append(mask)
            by_class[name]["maps"].append(amap)
            del image, score

    rows = []
    for class_name in class_names:
        values = by_class.get(class_name)
        if not values:
            continue
        image_result = asdict(binary_metrics(values["labels"], values["scores"]))
        pixel_result = asdict(pixel_metrics(values["masks"], values["maps"]))
        row = {"class_name": class_name, "n_images": len(values["labels"])}
        row.update({f"i_{key}": value for key, value in image_result.items()})
        row.update({f"p_{key}": value for key, value in pixel_result.items()})
        rows.append(row)

    metric_columns = [key for key in rows[0] if key not in {"class_name", "n_images"}]
    average = macro_average([{key: row[key] for key in metric_columns} for row in rows])
    average_row = {"class_name": "Average", "n_images": sum(row["n_images"] for row in rows), **average}
    all_rows = rows + [average_row]
    csv_path = args.output / f"{args.dataset}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    paper_comparable = _percent_dict({
        "I-AUROC": average.get("i_auroc", np.nan),
        "I-maxF1": average.get("i_f1max", np.nan),
        "I-AP": average.get("i_aupr", np.nan),
        "P-AUROC": average.get("p_auroc", np.nan),
        "P-maxF1": average.get("p_f1max", np.nan),
        "P-AP": average.get("p_aupr", np.nan),
        "AUPRO@0.3": average.get("p_aupro_30", np.nan),
    })
    report = {
        "dataset": args.dataset,
        "dataset_root": str(data_root),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "device": device,
        "limit": args.limit,
        "seed": args.seed,
        "precision": args.precision,
        "is_full_evaluation": args.limit == 0,
        "limited_subset": "balanced anomaly/normal, deterministic" if args.limit else None,
        "protocol": "fixed checkpoint; target dataset is never used for checkpoint selection",
        "paper_comparable_percent": paper_comparable,
        "latency_seconds": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "per_class_csv": str(csv_path.resolve()),
    }
    json_path = args.output / f"{args.dataset}_run.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
