#!/usr/bin/env python3
"""DINOv2 VLM v3 for Real-IAD Variety.

Compared with v2, this version removes per-test-image percentile stretching.
All pixel calibration comes from official Train normal images.  The DINOv2
backbone and CLIP model are public generic weights only; the learned pseudo
head is trained from synthetic defects generated on official normal images.
"""
import argparse
import csv
import json
import math
import random
import time
import zipfile
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


SEED = 20260813
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def groups(root):
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "__MACOSX"):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            paths = [sample_dir / f"{i}.png" for i in range(5)]
            if all(p.exists() for p in paths):
                yield cat_dir.name, sample_dir.name, paths


def load_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def dino_model(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def normalize_images(images, size, device):
    arr = []
    for im in images:
        im = im.resize((size, size), Image.Resampling.BICUBIC)
        x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        arr.append(torch.from_numpy(x))
    x = torch.stack(arr).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    return (x - mean) / std


@torch.inference_mode()
def dino_batch(model, images, device, size=518):
    x = normalize_images(images, size, device)
    outs = model.get_intermediate_layers(x, n=[3, 6, 9, 11], reshape=False, return_class_token=True)
    patches = []
    cls_tokens = []
    for patch, cls in outs:
        patches.append(F.normalize(patch, dim=-1))
        cls_tokens.append(F.normalize(cls, dim=-1))
    patch_tokens = F.normalize(torch.cat(patches, dim=-1), dim=-1)
    cls_tokens = F.normalize(torch.cat(cls_tokens, dim=-1), dim=-1)
    return cls_tokens.cpu().numpy(), patch_tokens.cpu().numpy()


def grid_sample_indices(grid, per_image):
    side = int(math.sqrt(grid))
    if per_image >= grid:
        return np.arange(grid)
    rows = max(1, int(math.sqrt(per_image)))
    cols = max(1, int(math.ceil(per_image / rows)))
    ys = np.linspace(0, side - 1, rows).round().astype(int)
    xs = np.linspace(0, side - 1, cols).round().astype(int)
    ids = []
    for y in ys:
        for x in xs:
            ids.append(y * side + x)
    return np.asarray(ids[:per_image], dtype=np.int64)


def farthest_sample(x, k):
    if len(x) <= k:
        return x
    rng = np.random.default_rng(SEED)
    first = int(rng.integers(0, len(x)))
    chosen = [first]
    sims = x @ x[first]
    min_dist = 1.0 - sims
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        chosen.append(idx)
        min_dist = np.minimum(min_dist, 1.0 - (x @ x[idx]))
    return x[np.asarray(chosen)]


def prompts(category):
    name = category.replace("_", " ")
    normal = [
        f"a photo of a normal {name} industrial product",
        f"a clean intact {name} manufactured part",
        f"a flawless {name} surface",
    ]
    abnormal = [
        f"a photo of a defective {name} industrial product",
        f"a damaged {name} manufactured part",
        f"an anomalous {name} surface",
    ]
    local = [
        f"{name} with a fine scratch",
        f"{name} with a crack",
        f"{name} with a dent",
        f"{name} with a broken edge",
        f"{name} with stain or contamination",
        f"{name} with missing material",
    ]
    return normal, abnormal, local


def make_pseudo(image, kind):
    image = image.copy()
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    rng = np.random.default_rng(random.randint(0, 2**31 - 1))
    x = random.randint(int(w * 0.06), int(w * 0.82))
    y = random.randint(int(h * 0.06), int(h * 0.82))
    rw = random.randint(max(8, int(w * 0.025)), max(12, int(w * 0.22)))
    rh = random.randint(max(4, int(h * 0.010)), max(8, int(h * 0.12)))
    x1 = min(w - 1, x + rw)
    y1 = min(h - 1, y + rh)

    if kind == 0:
        pts = []
        for i in range(random.randint(4, 9)):
            px = x + i * max(1, rw // 7)
            py = y + int(np.sin(i * 0.9) * rh * 0.4) + random.randint(-rh, rh)
            pts.append((min(w - 1, px), int(np.clip(py, 0, h - 1))))
        width = random.randint(1, 3)
        color = random.choice([(15, 15, 15), (235, 235, 235), (80, 80, 80)])
        draw_img.line(pts, fill=color, width=width)
        draw_mask.line(pts, fill=255, width=max(2, width + 2))
    elif kind == 1:
        overlay = Image.new("RGB", (x1 - x, y1 - y), random.choice([(30, 30, 30), (220, 220, 220), (120, 90, 60)]))
        alpha = Image.new("L", overlay.size, 0)
        ad = ImageDraw.Draw(alpha)
        ad.ellipse((0, 0, overlay.size[0] - 1, overlay.size[1] - 1), fill=random.randint(85, 170))
        patch = image.crop((x, y, x1, y1))
        patch = Image.composite(overlay, patch, alpha.filter(ImageFilter.GaussianBlur(3)))
        image.paste(patch, (x, y))
        draw_mask.ellipse((x, y, x1, y1), fill=255)
    elif kind == 2:
        crop = image.crop((x, y, x1, y1))
        crop = crop.filter(ImageFilter.GaussianBlur(random.uniform(2, 5)))
        arr = np.asarray(crop, np.float32)
        arr = np.clip(arr * random.uniform(0.45, 0.80), 0, 255).astype(np.uint8)
        image.paste(Image.fromarray(arr), (x, y))
        draw_mask.ellipse((x, y, x1, y1), fill=255)
    elif kind == 3:
        crop = np.asarray(image.crop((x, y, x1, y1)), np.float32)
        noise = rng.normal(0, random.uniform(25, 55), crop.shape)
        crop = np.clip(crop + noise, 0, 255).astype(np.uint8)
        image.paste(Image.fromarray(crop), (x, y))
        draw_mask.rectangle((x, y, x1, y1), fill=255)
    elif kind == 4:
        color = tuple(int(v) for v in rng.integers(70, 190, size=3))
        draw_img.rectangle((x, y, x1, y1), fill=color)
        draw_mask.rectangle((x, y, x1, y1), fill=255)
    else:
        pts = [(x, y), (x1, y + random.randint(-rh, rh)), (x + random.randint(0, rw), y1)]
        draw_img.polygon(pts, fill=random.choice([(10, 10, 10), (240, 240, 240), (150, 150, 150)]))
        draw_mask.polygon(pts, fill=255)
    return image, mask


def robust_quantiles(values, qs=(50, 90, 97.5, 99.5)):
    values = np.asarray(values, np.float32)
    if len(values) == 0:
        return np.asarray([0.0, 0.1, 0.2, 0.3], np.float32)
    return np.percentile(values, qs).astype(np.float32)


def calibrate_map(raw, q):
    # q comes from Train normal pixels for the same category/view when possible.
    low = float(q[1])
    high = float(q[3])
    scaled = (raw - low) / max(high - low, 1e-6)
    return np.clip(scaled, 0.0, 1.0)


def train_pseudo_head(model, train_items, device, limit_groups, batch_size):
    feats = []
    labels = []
    chosen = train_items[: min(limit_groups, len(train_items))]
    for idx, (_, _, paths) in enumerate(chosen):
        view_ids = range(5)
        images = []
        masks = []
        for v in view_ids:
            normal = load_image(paths[v])
            aug, mask = make_pseudo(normal, (idx + v) % 6)
            images.extend([normal, aug])
            masks.extend([Image.new("L", normal.size, 0), mask])
        for st in range(0, len(images), batch_size):
            _, patches = dino_batch(model, images[st : st + batch_size], device)
            for j, patch in enumerate(patches):
                mask = masks[st + j].resize((37, 37), Image.Resampling.BILINEAR)
                y = (np.asarray(mask, np.float32).reshape(-1) / 255.0 > 0.12).astype(np.float32)
                feats.append(patch.astype(np.float16))
                labels.append(y)
        if idx and idx % 50 == 0:
            print("pseudo train groups", idx, flush=True)
    x = torch.from_numpy(np.concatenate(feats, axis=0).astype(np.float32)).to(device)
    y = torch.from_numpy(np.concatenate(labels, axis=0).astype(np.float32)).to(device)
    x = F.normalize(x, dim=1)
    head = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Dropout(0.10), nn.Linear(256, 1)).to(device)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    pos_weight = torch.tensor([min(30.0, neg / pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=1.5e-3, weight_decay=1e-4)
    head.train()
    for epoch in range(6):
        order = torch.randperm(x.shape[0], device=device)
        losses = []
        for st in range(0, x.shape[0], 16384):
            ix = order[st : st + 16384]
            logits = head(x[ix]).squeeze(1)
            loss = loss_fn(logits, y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print("pseudo epoch", epoch + 1, "loss", float(np.mean(losses)), flush=True)
    head.eval()
    return head


def clip_texts(model, categories, device):
    text = {}
    with torch.inference_mode():
        for category in categories:
            normal, abnormal, local = prompts(category)
            tokens = clip.tokenize(normal + abnormal + local).to(device)
            z = model.encode_text(tokens)
            z = F.normalize(z, dim=-1)
            text[category] = (
                z[: len(normal)].mean(0),
                z[len(normal) : len(normal) + len(abnormal)].mean(0),
                z[len(normal) + len(abnormal) :].mean(0),
            )
    return text


@torch.inference_mode()
def encode_clip_crops(model, preprocess, image, device):
    image = image.resize((448, 448), Image.Resampling.BICUBIC)
    crops = [image]
    for y in (0, 112, 224):
        for x in (0, 112, 224):
            crops.append(image.crop((x, y, x + 224, y + 224)))
    x = torch.stack([preprocess(c) for c in crops]).to(device)
    z = model.encode_image(x)
    return F.normalize(z, dim=-1)


def build_memory_and_calibration(model, train_items, device, batch_size, reservoir_per_key, proto_per_key):
    sample_ids = grid_sample_indices(1369, 64)
    reservoirs = {}
    cls_bank = {}
    all_patches = []

    for st in range(0, len(train_items), batch_size):
        chunk = train_items[st : st + batch_size]
        images = [load_image(path) for _, _, paths in chunk for path in paths]
        cls, patches = dino_batch(model, images, device)
        cls = cls.reshape(len(chunk), 5, -1)
        patches = patches.reshape(len(chunk), 5, 1369, -1)
        for i, (category, _, _) in enumerate(chunk):
            for view in range(5):
                key = (category, view)
                selected = patches[i, view, sample_ids]
                bucket = reservoirs.setdefault(key, [])
                if len(bucket) < reservoir_per_key:
                    room = reservoir_per_key - len(bucket)
                    bucket.extend(selected[:room].astype(np.float16))
                cls_bank.setdefault(key, []).append(cls[i, view].astype(np.float32))
                all_patches.append(selected[:8].astype(np.float16))
        if st % max(batch_size * 8, 1) == 0:
            print("memory features", min(st + batch_size, len(train_items)), flush=True)

    global_proto = np.concatenate(all_patches, axis=0).astype(np.float32)
    global_proto = farthest_sample(F.normalize(torch.from_numpy(global_proto), dim=1).numpy(), min(2048, len(global_proto)))

    prototypes = {}
    cls_mu = {}
    for key, values in reservoirs.items():
        x = np.asarray(values, np.float32)
        x = F.normalize(torch.from_numpy(x), dim=1).numpy()
        prototypes[key] = farthest_sample(x, min(proto_per_key, len(x))).astype(np.float32)
        c = np.mean(np.stack(cls_bank[key]), axis=0)
        cls_mu[key] = c / max(np.linalg.norm(c), 1e-6)

    pixel_scores = {}
    image_scores = {}
    for key in prototypes:
        pixel_scores[key] = []
        image_scores[key] = []
    pixel_scores[("__global__", -1)] = []
    image_scores[("__global__", -1)] = []

    for st in range(0, len(train_items), batch_size):
        chunk = train_items[st : st + batch_size]
        images = [load_image(path) for _, _, paths in chunk for path in paths]
        _, patches = dino_batch(model, images, device)
        patches = patches.reshape(len(chunk), 5, 1369, -1)
        for i, (category, _, _) in enumerate(chunk):
            for view in range(5):
                z = patches[i, view]
                key = (category, view)
                dist = 1.0 - (z @ prototypes[key].T).max(1)
                pixel_scores[key].extend(dist.tolist())
                image_scores[key].append(float(np.percentile(dist, 99.0)))
                gdist = 1.0 - (z @ global_proto.T).max(1)
                pixel_scores[("__global__", -1)].extend(gdist[::16].tolist())
                image_scores[("__global__", -1)].append(float(np.percentile(gdist, 99.0)))
        if st % max(batch_size * 8, 1) == 0:
            print("normal calibration", min(st + batch_size, len(train_items)), flush=True)

    pixel_q = {key: robust_quantiles(vals) for key, vals in pixel_scores.items()}
    image_q = {key: robust_quantiles(vals) for key, vals in image_scores.items()}
    return prototypes, global_proto.astype(np.float32), cls_mu, pixel_q, image_q


def save_mask(path, mask37):
    x = torch.from_numpy(mask37.astype(np.float32))[None, None]
    x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)[0, 0].numpy()
    # Mild power sharpening is deterministic and does not use per-image ranks.
    x = np.clip(x, 0.0, 1.0) ** 1.25
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(x * 255)), "L").save(path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--reservoir-per-key", type=int, default=3072)
    parser.add_argument("--proto-per-key", type=int, default=384)
    parser.add_argument("--pseudo-groups", type=int, default=320)
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    dino = dino_model(device)
    clip_model, clip_preprocess = clip.load(str(args.clip_weight), device=str(device), jit=False)
    clip_model.eval()

    train_items = list(groups(args.data_root / "Train"))
    test_items = list(groups(args.data_root / "Test_A"))
    categories = sorted({x[0] for x in train_items + test_items})

    prototypes, global_proto, cls_mu, pixel_q, image_q = build_memory_and_calibration(
        dino, train_items, device, args.batch_size, args.reservoir_per_key, args.proto_per_key
    )
    pseudo_head = train_pseudo_head(dino, train_items, device, args.pseudo_groups, max(2, args.batch_size))
    texts = clip_texts(clip_model, categories, device)

    pseudo_normal = []
    clip_normal = []
    for category, _, paths in train_items[: min(240, len(train_items))]:
        images = [load_image(p) for p in paths]
        _, patches = dino_batch(dino, images, device)
        with torch.inference_mode():
            logits = pseudo_head(F.normalize(torch.from_numpy(patches.reshape(-1, patches.shape[-1])).float().to(device), dim=1))
            pseudo_normal.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
        n, b, local = texts[category]
        for image in images:
            z = encode_clip_crops(clip_model, clip_preprocess, image, device)
            clip_normal.append(float((z[1:] @ (local - n)).mean().cpu()))
    pseudo_q = robust_quantiles(pseudo_normal)
    clip_q = robust_quantiles(clip_normal)

    out_root = args.output / "predicted_masks"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for st in range(0, len(test_items), args.batch_size):
        chunk = test_items[st : st + args.batch_size]
        images = [load_image(path) for _, _, paths in chunk for path in paths]
        cls, patches = dino_batch(dino, images, device)
        cls = cls.reshape(len(chunk), 5, -1)
        patches = patches.reshape(len(chunk), 5, 1369, -1)
        for i, (category, sample, paths) in enumerate(chunk):
            view_scores = []
            group_ref = np.median(cls[i], axis=0)
            group_ref = group_ref / max(np.linalg.norm(group_ref), 1e-6)
            n, b, local = texts[category]
            for view in range(5):
                z = patches[i, view]
                key = (category, view)
                proto = prototypes.get(key, global_proto)
                pq = pixel_q.get(key, pixel_q[("__global__", -1)])
                iq = image_q.get(key, image_q[("__global__", -1)])
                dist = 1.0 - (z @ proto.T).max(1)
                dino_map = calibrate_map(dist, pq).reshape(37, 37)
                dino_img = float(np.clip((np.percentile(dist, 99.0) - iq[1]) / max(iq[3] - iq[1], 1e-6), 0, 1))

                with torch.inference_mode():
                    logits = pseudo_head(F.normalize(torch.from_numpy(z).float().to(device), dim=1))
                    pseudo = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                pseudo_map = calibrate_map(pseudo, pseudo_q).reshape(37, 37)
                pseudo_img = float(np.clip((np.percentile(pseudo, 99.0) - pseudo_q[1]) / max(pseudo_q[3] - pseudo_q[1], 1e-6), 0, 1))

                clip_z = encode_clip_crops(clip_model, clip_preprocess, load_image(paths[view]), device)
                clip_local = float((clip_z[1:] @ (local - n)).mean().cpu())
                clip_gate = float(np.clip((clip_local - clip_q[1]) / max(clip_q[3] - clip_q[1], 1e-6), 0, 1))
                view_consistency = float(1.0 - (cls[i, view] @ group_ref))
                view_consistency = float(np.clip(view_consistency / 0.25, 0, 1))

                mask = np.clip(0.68 * dino_map + 0.32 * pseudo_map, 0, 1)
                mask = mask * (0.82 + 0.18 * max(clip_gate, view_consistency))
                mask = np.clip(mask, 0, 1)
                save_mask(out_root / category / sample / f"{view}_mask.png", mask)

                view_score = 0.52 * dino_img + 0.25 * pseudo_img + 0.13 * clip_gate + 0.10 * view_consistency
                view_scores.append(float(np.clip(view_score, 0, 1)))
            score = 0.70 * max(view_scores) + 0.30 * float(np.mean(view_scores))
            rows.append({"group_folder": f"{category}/{sample}", "anomaly_score": f"{score:.8f}"})
        print("test", min(st + args.batch_size, len(test_items)), flush=True)

    with (args.output / "submission.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "method": "DINOv2 v3 train-calibrated patch memory + multi-view pseudo anomaly head + local CLIP gate",
        "changes_from_v2": [
            "removed per-test-image percentile mask stretching",
            "category/view memory bank uses farthest-point prototypes",
            "pixel calibration uses Train normal quantiles",
            "pseudo anomaly head uses all five views",
            "CLIP is used as local gate instead of global-only score",
        ],
        "dino_source": "facebookresearch/dinov2 dinov2_vits14 public generic weight",
        "external_data": False,
        "external_specialized_weights": False,
        "online_model": False,
        "train_groups": len(train_items),
        "test_groups": len(test_items),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output / "submission.csv", "submission.csv")
        for path in sorted(out_root.glob("*/*/*_mask.png")):
            zf.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
