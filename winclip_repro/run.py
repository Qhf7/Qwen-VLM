from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


STATE_LEVEL = {
    "normal": ["{}", "flawless {}", "perfect {}", "unblemished {}",
               "{} without flaw", "{} without defect", "{} without damage"],
    "anomaly": ["damaged {}", "{} with flaw", "{} with defect", "{} with damage"],
}
TEMPLATES = [
    "a cropped photo of the {}.", "a cropped photo of a {}.",
    "a close-up photo of a {}.", "a close-up photo of the {}.",
    "a bright photo of a {}.", "a bright photo of the {}.",
    "a dark photo of a {}.", "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.", "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.", "a blurry photo of the {}.",
    "a photo of a {}.", "a photo of the {}.",
    "a photo of a small {}.", "a photo of the small {}.",
    "a photo of a large {}.", "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
]


def _texts(obj: str):
    normal = [s.format(obj) for s in STATE_LEVEL["normal"]]
    anomaly = [s.format(obj) for s in STATE_LEVEL["anomaly"]]
    return ([t.format(s) for s in normal for t in TEMPLATES],
            [t.format(s) for s in anomaly for t in TEMPLATES])


def _window_indices(grid: int, kernel: int):
    return [
        [r * grid + c for r in range(i, i + kernel) for c in range(j, j + kernel)]
        for i in range(grid - kernel + 1)
        for j in range(grid - kernel + 1)
    ]


class LocalWinCLIP:
    """WinCLIP windows using a locally available OpenAI CLIP checkpoint.

    This follows the paper's token-window computation.  The paper's default
    ViT-B/16+ checkpoint is replaceable; the workspace uses ViT-L/14@336.
    """

    def __init__(self, checkpoint: str, device: str = "cpu", batch_size: int = 32):
        import clip  # OpenAI CLIP package, already used by AdaCLIP in this workspace
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, self.preprocess = clip.load(checkpoint, device=self.device, jit=False)
        self.model.eval()
        visual = self.model.visual
        self.grid = int(visual.input_resolution // visual.conv1.kernel_size[0])
        self.dim = visual.conv1.out_channels
        self._window_cache = {}

    @torch.inference_mode()
    def text_features(self, obj_name: str):
        import clip
        normal, anomaly = _texts(obj_name)
        nf = self.model.encode_text(clip.tokenize(normal).to(self.device)).float()
        af = self.model.encode_text(clip.tokenize(anomaly).to(self.device)).float()
        nf = nf / nf.norm(dim=-1, keepdim=True)
        af = af / af.norm(dim=-1, keepdim=True)
        nf = nf.mean(0, keepdim=True); af = af.mean(0, keepdim=True)
        nf = nf / nf.norm(dim=-1, keepdim=True)
        af = af / af.norm(dim=-1, keepdim=True)
        return torch.cat([nf, af], dim=0)

    @torch.inference_mode()
    def _tokens(self, image_tensor: torch.Tensor):
        v = self.model.visual
        x = v.conv1(image_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        return x

    @torch.inference_mode()
    def _encode_sequences(self, seq: torch.Tensor):
        v = self.model.visual
        x = v.ln_pre(seq).permute(1, 0, 2)
        x = v.transformer(x).permute(1, 0, 2)
        x = v.ln_post(x[:, 0, :])
        if v.proj is not None:
            x = x @ v.proj
        return x.float() / x.float().norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def predict(self, image: Image.Image, obj_name: str):
        text = self.text_features(obj_name)
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        tokens = self._tokens(tensor)
        global_f = self._encode_sequences(tokens)
        logits = 100.0 * global_f @ text.T
        image_score = logits.softmax(-1)[0, 1].item()
        maps = [np.full((self.grid, self.grid), image_score, dtype=np.float32)]
        for kernel in (2, 3):
            windows = _window_indices(self.grid, kernel)
            scores = []
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start:start + self.batch_size]
                idx = torch.tensor([[0] + [p + 1 for p in w] for w in batch],
                                   dtype=torch.long, device=self.device)
                seq = tokens.index_select(1, idx.reshape(-1)).reshape(len(batch), idx.shape[1], -1)
                f = self._encode_sequences(seq)
                s = (100.0 * f @ text.T).softmax(-1)[:, 1]
                scores.extend(s.float().cpu().tolist())
            acc = np.zeros((self.grid, self.grid), dtype=np.float64)
            cnt = np.zeros_like(acc)
            for score, w in zip(scores, windows):
                for p in w:
                    r, c = divmod(p, self.grid)
                    acc[r, c] += 1.0 / max(float(score), 1e-8)
                    cnt[r, c] += 1.0
            maps.append((cnt / np.maximum(acc, 1e-8)).astype(np.float32))
        amap = np.mean(np.stack(maps), axis=0)
        return float(image_score), amap


def _resolve_item(raw: str, root: Path, manifest):
    path_s, obj = raw.split("::", 1)
    path = Path(path_s)
    if not path.is_absolute():
        # Accept either a project-relative path (Dataset/...) or a path
        # relative to --root (3_adapter/...).
        path = path if path.exists() else root / path
    path = path.resolve()
    rel = str(path.relative_to(root.resolve()))
    row = manifest[manifest.image_path == rel].iloc[0] if manifest is not None and (manifest.image_path == rel).any() else None
    return path, obj, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--root", type=Path, default=Path("Dataset"))
    ap.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.parquet"))
    ap.add_argument("--item", action="append", required=True,
                    help="image path::English object name; repeat for multiple images")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pandas as pd
    manifest = pd.read_parquet(args.manifest) if args.manifest.exists() else None
    model = LocalWinCLIP(args.clip_checkpoint, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({
        "checkpoint": str(args.clip_checkpoint), "device": args.device,
        "batch_size": args.batch_size, "root": str(args.root),
        "items": args.item, "method": "WinCLIP-0shot-local-ViT-L-14@336"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    pixel_y, pixel_s = [], []
    for raw in args.item:
        path, obj, row = _resolve_item(raw, args.root, manifest)
        with Image.open(path) as im:
            image = im.convert("RGB")
        score, amap = model.predict(image, obj)
        stem = path.stem
        np.save(args.output / f"{stem}_anomaly_map.npy", amap)
        Image.fromarray(np.uint8(np.clip((amap - amap.min()) / (np.ptp(amap) + 1e-8), 0, 1) * 255)).resize(image.size).save(args.output / f"{stem}_anomaly_map.png")
        rec = {"image_path": str(path), "object": obj, "score": score,
               "map_min": float(amap.min()), "map_max": float(amap.max())}
        if row is not None:
            rec.update({"view_label": int(row.view_label), "sample_label": int(row.sample_label),
                        "mask_path": str(args.root / row.mask_path) if row.view_label else None})
            gt_img = (np.asarray(Image.open(args.root / row.mask_path).convert("L")) > 0
                      if row.view_label else np.zeros((image.height, image.width), dtype=bool))
            gt_small = np.asarray(Image.fromarray(gt_img.astype(np.uint8)).resize(
                (amap.shape[1], amap.shape[0]), Image.Resampling.NEAREST), dtype=bool)
            pixel_y.extend(gt_small.ravel().tolist())
            pixel_s.extend(amap.ravel().tolist())
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    (args.output / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if records:
        with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if len(records) > 1 and all("view_label" in x for x in records):
        y = np.array([x["view_label"] for x in records])
        s = np.array([x["score"] for x in records])
        if len(np.unique(y)) == 2:
            summary = {"I_AUROC": float(roc_auc_score(y, s)),
                       "I_AUPR": float(average_precision_score(y, s))}
            py, ps = np.asarray(pixel_y), np.asarray(pixel_s)
            if len(np.unique(py)) == 2:
                summary.update({"P_AUROC_at_map_grid": float(roc_auc_score(py, ps)),
                                "P_AUPR_at_map_grid": float(average_precision_score(py, ps))})
            (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary))


if __name__ == "__main__":
    main()
