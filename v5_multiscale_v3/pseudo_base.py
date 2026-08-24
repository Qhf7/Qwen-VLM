#!/usr/bin/env python3
"""Synthetic defect generation from official normal images only.

Why this file exists in its current form
----------------------------------------
The v3 generator drew defects sized up to 0.22*width of the *native* image.
On a 3599 px source that is a 790 px blob, i.e. ~98 px once mapped to the
448 px mask grid.  Real Real-IAD defects (scratches, dents, chips, small
contamination) occupy roughly 0.02%-1% of the frame, i.e. ~5-45 px at 448.
Training a head on oversized blobs teaches it to paint broad regions, which
is exactly what destroys P-AUPR and P-F1max.

Everything here is therefore specified in *448-space* and then mapped back
to native resolution, so the physical defect size is scale-invariant across
the 2256-3599 px range of source images.

Compliance: defects are synthesised purely from the official Train normal
images (self-crops, procedural noise, geometric and photometric edits).  No
external texture source, no anomaly dataset, no test image is used.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MASK_SIZE = 448

# Defect scale in 448-space.  v2 is more aggressively biased toward tiny
# defects because A榜低分主要来自 broad false positives, not missed huge blobs.
SIZE_BANDS = (
    (1.5, 5.0, 0.38),    # tiny: pinholes, hairline scratch width
    (5.0, 13.0, 0.42),   # small: typical scratch / dent / chip
    (13.0, 28.0, 0.16),  # medium
    (28.0, 48.0, 0.04),  # large, kept rare
)

KINDS = (
    "hairline",
    "scratch",
    "speck",
    "dent",
    "chip",
    "contamination",
    "discolour",
    "cutpaste",
    "misalign",
    "hole",
)

KIND_WEIGHTS = np.asarray(
    [0.22, 0.20, 0.16, 0.12, 0.08, 0.10, 0.04, 0.04, 0.02, 0.02],
    np.float64,
)
KIND_WEIGHTS = KIND_WEIGHTS / KIND_WEIGHTS.sum()


def _sample_size(rng: np.random.Generator) -> float:
    bands = SIZE_BANDS
    probs = np.asarray([b[2] for b in bands], np.float64)
    probs = probs / probs.sum()
    i = int(rng.choice(len(bands), p=probs))
    lo, hi, _ = bands[i]
    # log-uniform inside the band keeps small sizes well represented
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def foreground_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Rough bounding box of the part, to avoid painting on empty background.

    Real defects are on the object.  Placing synthetic defects on flat
    background teaches a trivially separable cue and wastes capacity.  The
    estimate uses only pixel intensity of the (normal) training image.
    """
    small = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR).convert("L"), np.float32)
    med = float(np.median(small))
    # object is whichever side of the median has more local contrast
    grad = np.abs(np.gradient(small)[0]) + np.abs(np.gradient(small)[1])
    mask = grad > np.percentile(grad, 70)
    if mask.sum() < 16:
        mask = np.abs(small - med) > (small.std() * 0.5 + 1e-3)
    ys, xs = np.nonzero(mask)
    if len(ys) < 8:
        return 0, 0, image.size[0] - 1, image.size[1] - 1
    w, h = image.size
    sx, sy = w / 64.0, h / 64.0
    pad = 2
    x0 = max(0, int((xs.min() - pad) * sx))
    x1 = min(w - 1, int((xs.max() + pad) * sx))
    y0 = max(0, int((ys.min() - pad) * sy))
    y1 = min(h - 1, int((ys.max() + pad) * sy))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return 0, 0, w - 1, h - 1
    return x0, y0, x1, y1


def foreground_probability(image: Image.Image, size: int = 128) -> np.ndarray:
    """Soft foreground probability from the normal image itself.

    The earlier generator sampled inside a rough object bounding box.  That
    still places many pseudo defects on holes, background pockets, or broad
    silhouette edges, teaching the head to light the whole part outline.  This
    v2 sampler uses local contrast and distance-to-empty-background so defects
    land on the part interior more often.
    """
    gray = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR).convert("L"), np.float32)
    grad = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
    dev = np.abs(gray - float(np.median(gray)))
    g = grad / (np.percentile(grad, 98) + 1e-6)
    d = dev / (np.percentile(dev, 98) + 1e-6)
    prob = np.clip(0.55 * d + 0.45 * g, 0.0, 1.0)
    prob = np.asarray(
        Image.fromarray(np.uint8(prob * 255), "L").filter(ImageFilter.GaussianBlur(2.0)),
        np.float32,
    ) / 255.0
    # Suppress the outer image rim; many industrial cameras have stable border
    # artefacts that are not defects.
    yy, xx = np.mgrid[0:size, 0:size]
    border = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy]).astype(np.float32)
    border = np.clip(border / max(size * 0.08, 1.0), 0.0, 1.0)
    prob *= border
    if float(prob.sum()) <= 1e-6:
        prob += 1.0
    return prob


def _sample_foreground_point(image: Image.Image, rng: np.random.Generator) -> tuple[float, float]:
    prob = foreground_probability(image, 128)
    flat = prob.reshape(-1).astype(np.float64)
    flat = flat / flat.sum()
    idx = int(rng.choice(len(flat), p=flat))
    gy, gx = divmod(idx, prob.shape[1])
    w, h = image.size
    x = (gx + rng.random()) / prob.shape[1] * w
    y = (gy + rng.random()) / prob.shape[0] * h
    return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))


def _poly_line(rng, cx, cy, length, curve, n=10):
    ang = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n):
        t = i / (n - 1) - 0.5
        a = ang + curve * t
        pts.append((cx + math.cos(a) * length * t, cy + math.sin(a) * length * t))
    return pts


def synthesize(
    image: Image.Image,
    rng: np.random.Generator,
    kind: str | None = None,
    n_defects: tuple[int, int] = (1, 2),
) -> tuple[Image.Image, Image.Image]:
    """Return (defective_image, mask) at the input image's native resolution.

    The mask is native resolution; callers downsample it to whatever grid the
    backbone produces.
    """
    out = image.copy()
    w, h = out.size
    mask = Image.new("L", (w, h), 0)
    scale = min(w, h) / MASK_SIZE  # native px per 448-px unit

    fx0, fy0, fx1, fy1 = foreground_box(image)
    count = int(rng.integers(n_defects[0], n_defects[1] + 1))

    for _ in range(count):
        k = kind if kind is not None else str(rng.choice(KINDS, p=KIND_WEIGHTS))
        size448 = _sample_size(rng)
        size = max(2.0, size448 * scale)
        if rng.random() < 0.82:
            cx, cy = _sample_foreground_point(image, rng)
        else:
            cx = float(rng.uniform(fx0, fx1))
            cy = float(rng.uniform(fy0, fy1))
        _apply_defect(out, mask, k, cx, cy, size, scale, rng)

    return out, mask


def _local_stats(image: Image.Image, cx, cy, size):
    w, h = image.size
    r = int(max(4, size * 2))
    box = (
        max(0, int(cx - r)),
        max(0, int(cy - r)),
        min(w, int(cx + r) + 1),
        min(h, int(cy + r) + 1),
    )
    crop = np.asarray(image.crop(box), np.float32)
    return crop.reshape(-1, 3).mean(0), crop.reshape(-1, 3).std(0) + 1.0


def _apply_defect(image, mask, kind, cx, cy, size, scale, rng):
    w, h = image.size
    di = ImageDraw.Draw(image, "RGBA")
    dm = ImageDraw.Draw(mask)
    mu, sd = _local_stats(image, cx, cy, size)

    def dark(mult=0.45):
        return tuple(int(np.clip(v * mult, 0, 255)) for v in mu)

    def bright(add=90):
        return tuple(int(np.clip(v + add, 0, 255)) for v in mu)

    if kind == "hairline":
        # 1-2 px wide at 448: the hardest and most common real defect class
        length = size * rng.uniform(3.0, 9.0)
        pts = _poly_line(rng, cx, cy, length, rng.uniform(-0.7, 0.7))
        wid = max(1, int(round(rng.uniform(1.0, 2.0) * scale)))
        col = dark(rng.uniform(0.3, 0.6)) if rng.random() < 0.6 else bright(rng.uniform(60, 130))
        di.line(pts, fill=col + (int(rng.uniform(170, 255)),), width=wid)
        dm.line(pts, fill=255, width=wid)

    elif kind == "scratch":
        length = size * rng.uniform(2.0, 6.0)
        pts = _poly_line(rng, cx, cy, length, rng.uniform(-1.2, 1.2))
        wid = max(1, int(round(rng.uniform(1.5, 4.0) * scale)))
        col = dark(rng.uniform(0.35, 0.7)) if rng.random() < 0.5 else bright(rng.uniform(50, 120))
        di.line(pts, fill=col + (255,), width=wid)
        dm.line(pts, fill=255, width=wid)
        # scratches often have a brighter burr along one side
        if rng.random() < 0.4:
            off = wid * 0.8
            pts2 = [(x + off, y + off) for x, y in pts]
            di.line(pts2, fill=bright(rng.uniform(30, 80)) + (140,), width=max(1, wid // 2))

    elif kind == "speck":
        for _ in range(int(rng.integers(1, 5))):
            r = max(1.0, size * rng.uniform(0.15, 0.5))
            jx = cx + rng.uniform(-size, size)
            jy = cy + rng.uniform(-size, size)
            box = (jx - r, jy - r, jx + r, jy + r)
            col = dark(rng.uniform(0.2, 0.5)) if rng.random() < 0.7 else bright(100)
            di.ellipse(box, fill=col + (255,))
            dm.ellipse(box, fill=255)

    elif kind == "dent":
        # smooth local intensity depression with a soft edge; low contrast
        r = max(2.0, size)
        box = (int(cx - r), int(cy - r), int(cx + r) + 1, int(cy + r) + 1)
        box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return
        crop = np.asarray(image.crop(box), np.float32)
        yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
        ny = (yy - crop.shape[0] / 2) / max(crop.shape[0] / 2, 1)
        nx = (xx - crop.shape[1] / 2) / max(crop.shape[1] / 2, 1)
        rad = np.sqrt(nx**2 + ny**2)
        prof = np.clip(1.0 - rad, 0, 1) ** rng.uniform(0.6, 1.6)
        gain = rng.uniform(0.55, 0.85)
        shaded = crop * (1.0 - prof[..., None] * (1.0 - gain))
        # slight highlight on one rim, like a real dent under directional light
        rim = np.clip(1.0 - np.abs(rad - 0.85) * 6, 0, 1)
        shaded = shaded + rim[..., None] * rng.uniform(10, 40)
        image.paste(Image.fromarray(np.clip(shaded, 0, 255).astype(np.uint8)), box[:2])
        dm.ellipse((cx - r * 0.9, cy - r * 0.9, cx + r * 0.9, cy + r * 0.9), fill=255)

    elif kind == "chip":
        # irregular polygon with exposed (brighter or darker) material
        n = int(rng.integers(3, 7))
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n + rng.uniform(-0.3, 0.3)
            rr = size * rng.uniform(0.5, 1.3)
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
        col = bright(rng.uniform(50, 140)) if rng.random() < 0.5 else dark(rng.uniform(0.2, 0.5))
        di.polygon(pts, fill=col + (255,))
        dm.polygon(pts, fill=255)

    elif kind == "contamination":
        # blurred irregular stain, partially transparent
        r = max(2.0, size)
        box = (
            max(0, int(cx - r)),
            max(0, int(cy - r)),
            min(w, int(cx + r) + 1),
            min(h, int(cy + r) + 1),
        )
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 3 or bh < 3:
            return
        blob = rng.random((max(3, bh // 3), max(3, bw // 3))).astype(np.float32)
        blob = np.asarray(
            Image.fromarray(np.uint8(blob * 255), "L")
            .resize((bw, bh), Image.Resampling.BICUBIC)
            .filter(ImageFilter.GaussianBlur(max(1.0, r * 0.15))),
            np.float32,
        ) / 255.0
        alpha = np.clip((blob - 0.45) * 3.2, 0, 1) * rng.uniform(0.5, 0.95)
        crop = np.asarray(image.crop(box), np.float32)
        tint = np.asarray(
            dark(rng.uniform(0.25, 0.6)) if rng.random() < 0.6 else bright(rng.uniform(40, 110)),
            np.float32,
        )
        blended = crop * (1 - alpha[..., None]) + tint * alpha[..., None]
        image.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), box[:2])
        m = np.asarray(mask.crop(box), np.float32)
        m = np.maximum(m, (alpha > 0.25) * 255.0)
        mask.paste(Image.fromarray(m.astype(np.uint8), "L"), box[:2])

    elif kind == "discolour":
        # local hue/brightness shift with no structural change: tests whether
        # the model relies on texture alone
        r = max(2.0, size)
        box = (
            max(0, int(cx - r)),
            max(0, int(cy - r)),
            min(w, int(cx + r) + 1),
            min(h, int(cy + r) + 1),
        )
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return
        crop = np.asarray(image.crop(box), np.float32)
        gains = 1.0 + rng.uniform(-0.35, 0.35, size=3)
        shifted = np.clip(crop * gains, 0, 255)
        yy, xx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
        ny = (yy - crop.shape[0] / 2) / max(crop.shape[0] / 2, 1)
        nx = (xx - crop.shape[1] / 2) / max(crop.shape[1] / 2, 1)
        soft = np.clip(1.2 - np.sqrt(nx**2 + ny**2), 0, 1)[..., None]
        image.paste(
            Image.fromarray(np.clip(crop * (1 - soft) + shifted * soft, 0, 255).astype(np.uint8)),
            box[:2],
        )
        dm.ellipse((cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85), fill=255)

    elif kind == "cutpaste":
        # paste a patch from elsewhere in the SAME image (self-supervised,
        # no external texture) with a small geometric change
        rw = int(max(2, size * rng.uniform(0.8, 1.8)))
        rh = int(max(2, size * rng.uniform(0.5, 1.5)))
        if rw >= w - 2 or rh >= h - 2:
            return
        sx = int(rng.integers(0, w - rw - 1))
        sy = int(rng.integers(0, h - rh - 1))
        patch = image.crop((sx, sy, sx + rw, sy + rh))
        if rng.random() < 0.5:
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        patch = patch.rotate(float(rng.uniform(-25, 25)), expand=False)
        x0 = int(np.clip(cx - rw / 2, 0, w - rw - 1))
        y0 = int(np.clip(cy - rh / 2, 0, h - rh - 1))
        image.paste(patch, (x0, y0))
        dm.rectangle((x0, y0, x0 + rw, y0 + rh), fill=255)

    elif kind == "misalign":
        # local shift of image content: mimics assembly/position defects
        rw = int(max(3, size * rng.uniform(1.2, 2.5)))
        rh = int(max(3, size * rng.uniform(1.2, 2.5)))
        x0 = int(np.clip(cx - rw / 2, 0, w - rw - 1))
        y0 = int(np.clip(cy - rh / 2, 0, h - rh - 1))
        shift = int(max(1, size * rng.uniform(0.25, 0.7)))
        src = (
            int(np.clip(x0 + shift, 0, w - rw - 1)),
            int(np.clip(y0 + shift, 0, h - rh - 1)),
        )
        patch = image.crop((src[0], src[1], src[0] + rw, src[1] + rh))
        image.paste(patch, (x0, y0))
        dm.rectangle((x0, y0, x0 + rw, y0 + rh), fill=255)

    else:  # hole / missing material
        r = max(1.5, size * rng.uniform(0.5, 1.0))
        box = (cx - r, cy - r, cx + r, cy + r)
        col = dark(rng.uniform(0.08, 0.3))
        di.ellipse(box, fill=col + (255,))
        # bright rim
        di.ellipse(box, outline=bright(rng.uniform(40, 100)) + (200,), width=max(1, int(scale)))
        dm.ellipse(box, fill=255)


def defect_area_fraction(mask: Image.Image) -> float:
    a = np.asarray(mask.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR), np.float32)
    return float((a > 96).mean())
