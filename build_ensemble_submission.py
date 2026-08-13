#!/usr/bin/env python3
import argparse, csv, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def read_scores(root):
    with (root / 'submission.csv').open(encoding='utf-8') as f:
        return {r['group_folder']: float(r['anomaly_score']) for r in csv.DictReader(f)}

def rank_norm(d):
    keys = list(d)
    vals = np.asarray([d[k] for k in keys], np.float32)
    order = np.argsort(np.argsort(vals, kind='mergesort'), kind='mergesort')
    return {k: float(order[i] / max(len(vals)-1, 1)) for i, k in enumerate(keys)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vit', type=Path, required=True)
    ap.add_argument('--inp', type=Path, required=True)
    ap.add_argument('--dino', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    roots = [a.vit, a.inp, a.dino]
    weights = np.asarray([0.25, 0.35, 0.40], np.float32)
    scores = [rank_norm(read_scores(r)) for r in roots]
    keys = sorted(scores[0])
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        val = float(sum(w * s[key] for w, s in zip(weights, scores)))
        rows.append({'group_folder': key, 'anomaly_score': f'{val:.8f}'})
        cat, sample = key.split('/', 1)
        for v in range(5):
            ims = []
            for root in roots:
                p = root / 'predicted_masks' / cat / sample / f'{v}_mask.png'
                with Image.open(p) as im:
                    ims.append(np.asarray(im.convert('L'), np.float32) / 255.0)
            hm = np.clip(sum(w * im for w, im in zip(weights, ims)), 0, 1)
            out = a.output / 'predicted_masks' / cat / sample / f'{v}_mask.png'
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.round(hm * 255)), 'L').save(out, 'PNG', optimize=True)
    with (a.output / 'submission.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['group_folder', 'anomaly_score'])
        w.writeheader(); w.writerows(rows)
    with (a.output / 'run_metadata.json').open('w') as f:
        f.write('{"method":"rank-normalized ensemble: calibrated ViT 0.25 + INP-Former 0.35 + Dinomaly 0.40","external_specialized_weights":false}')
    out_zip = a.output.parent / (a.output.name + '.zip')
    masks = sorted((a.output / 'predicted_masks').glob('*/*/*_mask.png'))
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(a.output / 'submission.csv', 'submission.csv')
        for p in masks: z.write(p, p.relative_to(a.output).as_posix())
    print('rows', len(rows), 'masks', len(masks), 'zip_bytes', out_zip.stat().st_size)
    print('score_range', rows[0], rows[-1])

if __name__ == '__main__': main()
