"""Industrial-style pseudo anomalies and normal hard-negative targets."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pseudo_base import synthesize
from hard_negatives import hard_negative_map


def synthesize_v3(image, rng):
    # The base generator already produces self-crop, scratch, dent and chip
    # variants. Bias the new training distribution toward narrow defects.
    kinds = ["hairline", "scratch", "contamination", "dent", "discolour", "chip"]
    kind = kinds[int(rng.integers(len(kinds)))]
    defective, mask = synthesize(image, rng, kind=kind, n_defects=(1, 3))
    w, h = defective.size
    draw = ImageDraw.Draw(defective, "RGBA")
    md = ImageDraw.Draw(mask)
    scale = min(w, h) / 448.0
    if rng.random() < 0.55:
        # Low-contrast curved hairline / edge scratch.
        cx = float(rng.uniform(0.20, 0.80) * w)
        cy = float(rng.uniform(0.20, 0.80) * h)
        length = float(rng.uniform(10, 34) * scale)
        ang = float(rng.uniform(0, math.pi))
        pts = []
        for j in range(13):
            t = j / 12 - 0.5
            bend = rng.uniform(-0.22, 0.22) * math.sin((t + 0.5) * math.pi)
            x = cx + math.cos(ang + bend) * length * t
            y = cy + math.sin(ang + bend) * length * t
            pts.append((x, y))
        width = max(1, int(round(rng.uniform(0.7, 1.8) * scale)))
        delta = int(rng.uniform(12, 42))
        base = np.asarray(image.crop((max(0, int(cx-3*scale)), max(0, int(cy-3*scale)), min(w, int(cx+3*scale+1)), min(h, int(cy+3*scale+1)))), np.float32)
        value = int(np.clip(base.mean() - delta if base.size and base.mean() > 110 else base.mean() + delta, 0, 255))
        draw.line(pts, fill=(value, value, value, int(rng.uniform(110, 190))), width=width)
        md.line(pts, fill=255, width=width)
    if rng.random() < 0.35:
        # Subtle stain with a soft boundary, not a large binary blob.
        cx = int(rng.uniform(0.22, 0.78) * w)
        cy = int(rng.uniform(0.22, 0.78) * h)
        r = max(2, int(rng.uniform(4, 13) * scale))
        layer = Image.new("L", (w, h), 0)
        ld = ImageDraw.Draw(layer)
        ld.ellipse((cx-r, cy-r, cx+r, cy+r), fill=int(rng.uniform(80, 170)))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(scale))))
        arr = np.asarray(defective, np.float32)
        alpha = np.asarray(layer, np.float32)[..., None] / 255.0
        arr = arr * (1 - alpha * 0.20) + 20 * alpha
        defective = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), "RGB")
        md = ImageDraw.Draw(mask)
        md.ellipse((cx-r, cy-r, cx+r, cy+r), fill=180)
    return defective, mask


def make_training_targets(image, rng, out_size=448):
    defective, mask = synthesize_v3(image, rng)
    target = np.asarray(mask.resize((out_size, out_size), Image.Resampling.BILINEAR), np.float32) / 255.0
    hard = hard_negative_map(image, out_size)
    return defective, target.astype(np.float32), hard.astype(np.float32)
