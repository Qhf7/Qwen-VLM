#!/usr/bin/env python3
"""Rule-compliant pixel post-processing candidate.

Keeps the best ensemble's image-level scores unchanged.  For each mask, blend
the original calibrated heatmap with a small fixed Gaussian-smoothed version.
This suppresses isolated one-pixel noise while retaining thin defects.  No
labels, external data, per-image percentile stretching, or test-time tuning
are used.
"""
from pathlib import Path
from PIL import Image, ImageFilter
import csv, zipfile, json
import numpy as np

BASE = Path('submission_ensemble')
OUT = Path('submission_ensemble_pixel_v1')
ZIP = Path('submission_ensemble_pixel_v1.zip')

def main():
    OUT.mkdir(exist_ok=True)
    with (BASE/'submission.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with (OUT/'submission.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['group_folder','anomaly_score'])
        w.writeheader(); w.writerows(rows)
    n = 0
    for src in sorted((BASE/'predicted_masks').glob('*/*/*_mask.png')):
        with Image.open(src) as im:
            raw = im.convert('L')
            a = np.asarray(raw, dtype=np.float32) / 255.0
            smooth = np.asarray(raw.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32) / 255.0
        # Fixed global blend; absolute score calibration is retained.
        out = np.uint8(np.round(np.clip(0.70*a + 0.30*smooth, 0, 1)*255))
        dst = OUT/'predicted_masks'/src.relative_to(BASE/'predicted_masks')
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, 'L').save(dst, 'PNG', optimize=True)
        n += 1
    meta = {
        'method': 'best ensemble masks + fixed 0.8px Gaussian blend',
        'image_scores': 'copied unchanged from submission_ensemble',
        'external_specialized_weights': False,
        'external_data': False,
        'test_labels_used': False,
        'mask_blend': '0.70 raw + 0.30 GaussianBlur(radius=0.8)'
    }
    (OUT/'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'submission.csv', 'submission.csv')
        for p in sorted((OUT/'predicted_masks').glob('*/*/*_mask.png')):
            z.write(p, p.relative_to(OUT).as_posix())
    print('rows', len(rows), 'masks', n, 'zip_bytes', ZIP.stat().st_size)

if __name__ == '__main__':
    main()
