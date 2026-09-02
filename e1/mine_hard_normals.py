"""只从 fold1/train 的视角级正常图中确定性挖掘 hard-normal。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SIDE = 448
CUES = ("edge_density", "texture_logvar", "specular_fraction", "dark_fraction", "contrast")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"))
    rgb = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 50, 150)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return {
        "edge_density": float((edges > 0).mean()),
        "texture_logvar": float(np.log1p(laplacian.var())),
        "specular_fraction": float(((hsv[..., 2] >= 245) & (hsv[..., 1] <= 40)).mean()),
        "dark_fraction": float((hsv[..., 2] <= 25).mean()),
        "contrast": float(gray.std() / 255.0),
    }


def select_hard_normals(
    features: pd.DataFrame, per_class_per_cue: int, per_class_total: int,
) -> pd.DataFrame:
    selected_rows = []
    for class_name, part in features.groupby("class_name", sort=True):
        part = part.copy()
        for cue in CUES:
            part[f"{cue}_percentile"] = part[cue].rank(method="average", pct=True)
        part["composite"] = part[[f"{cue}_percentile" for cue in CUES]].max(axis=1)
        reasons: dict[str, set[str]] = {}
        for cue in CUES:
            ranked = part.sort_values([cue, "image_path"], ascending=[False, True]).head(per_class_per_cue)
            for image_path in ranked.image_path:
                reasons.setdefault(image_path, set()).add(cue)
        chosen = set(reasons)
        if len(chosen) < per_class_total:
            filler = part[~part.image_path.isin(chosen)].sort_values(
                ["composite", "image_path"], ascending=[False, True]
            )
            for image_path in filler.image_path.head(per_class_total - len(chosen)):
                reasons.setdefault(image_path, set()).add("composite_fill")
                chosen.add(image_path)
        # cue 并集可能超过总额；优先多 cue 同时命中，再按 composite 和路径稳定截断。
        chosen_frame = part[part.image_path.isin(chosen)].copy()
        chosen_frame["n_cues"] = chosen_frame.image_path.map(lambda value: len(reasons[value]))
        chosen_frame = chosen_frame.sort_values(
            ["n_cues", "composite", "image_path"], ascending=[False, False, True]
        ).head(per_class_total)
        chosen_frame["selected_by"] = chosen_frame.image_path.map(
            lambda value: "+".join(sorted(reasons[value]))
        )
        chosen_frame["hard_normal_rank_in_class"] = np.arange(1, len(chosen_frame) + 1)
        selected_rows.append(chosen_frame)
        if len(chosen_frame) != per_class_total:
            raise RuntimeError(f"{class_name} 只能选出 {len(chosen_frame)} 个 hard-normal")
    return pd.concat(selected_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/fold1/train.parquet"))
    parser.add_argument("--holdout-classes", type=Path, default=Path("artifacts/splits/fold1/holdout_classes.txt"))
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/e1"))
    parser.add_argument("--per-class-per-cue", type=int, default=8)
    parser.add_argument("--per-class-total", type=int, default=40)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.train)
    holdout = set(args.holdout_classes.read_text(encoding="utf-8").splitlines())
    overlap = sorted(set(manifest.class_name) & holdout)
    if overlap:
        raise RuntimeError(f"train 混入 holdout 类: {overlap}")
    candidates = manifest[manifest.view_label.eq(0)].copy().sort_values(
        ["class_name", "image_path"], kind="mergesort"
    )
    if candidates.class_name.nunique() != 18:
        raise RuntimeError("hard-normal 候选不是 18 个 train 类")
    missing = [path for path in candidates.image_path if not (args.root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张候选图")

    feature_path = args.out_dir / "hard_normal_features.csv"
    existing = pd.read_csv(feature_path) if feature_path.is_file() else pd.DataFrame()
    done = set(existing.image_path) if len(existing) else set()
    records = existing.to_dict("records") if len(existing) else []
    rows = candidates.to_dict("records")
    for index, row in enumerate(rows, 1):
        if row["image_path"] in done:
            continue
        records.append({
            "class_name": row["class_name"],
            "group_path": row["group_path"],
            "image_path": row["image_path"],
            "view_idx": row["view_idx"],
            "sample_label": row["sample_label"],
            "view_label": row["view_label"],
            "is_maskless_ng_view": int(row["sample_label"] == 1 and row["view_label"] == 0),
            **image_features(args.root / row["image_path"]),
        })
        if index % 50 == 0:
            pd.DataFrame(records).to_csv(feature_path, index=False)
            print(f"[hard-normal features] {index}/{len(rows)}", flush=True)
    features = pd.DataFrame(records).sort_values(["class_name", "image_path"])
    features.to_csv(feature_path, index=False)
    if len(features) != len(candidates) or features.image_path.nunique() != len(candidates):
        raise RuntimeError("hard-normal feature 产物不完整")

    selected = select_hard_normals(
        features, args.per_class_per_cue, args.per_class_total
    )
    selected.to_csv(args.out_dir / "hard_normals.csv", index=False)
    summary = (
        selected.groupby("class_name", as_index=False)
        .agg(
            n_hard_normals=("image_path", "size"),
            n_maskless_ng_views=("is_maskless_ng_view", "sum"),
            edge_density_median=("edge_density", "median"),
            texture_logvar_median=("texture_logvar", "median"),
            specular_fraction_median=("specular_fraction", "median"),
        )
    )
    summary.to_csv(args.out_dir / "hard_normal_summary.csv", index=False)
    protocol = {
        "protocol": "E1-hard-normal-v1",
        "train_sha256": sha256_file(args.train),
        "holdout_classes_sha256": sha256_file(args.holdout_classes),
        "holdout_intersection": overlap,
        "candidate_rule": "fold1/train view_label == 0",
        "cues": list(CUES),
        "resize": "direct 448 cv2.INTER_AREA",
        "per_class_per_cue": args.per_class_per_cue,
        "per_class_total": args.per_class_total,
        "selected_total": len(selected),
        "classes": sorted(selected.class_name.unique()),
    }
    (args.out_dir / "hard_normal_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] {len(selected)} hard-normal views", flush=True)


if __name__ == "__main__":
    main()
