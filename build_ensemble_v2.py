#!/usr/bin/env python3
import csv, zipfile, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def read_scores(root):
    with (root/'submission.csv').open(encoding='utf-8') as f:
        return {r['group_folder']: float(r['anomaly_score']) for r in csv.DictReader(f)}

def rank_norm(d):
    ks=list(d); x=np.asarray([d[k] for k in ks],np.float32)
    order=np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort')
    return {k:float(order[i]/max(1,len(x)-1)) for i,k in enumerate(ks)}

def normalize_mask(a):
    x=a.astype(np.float32)/255.0
    # Per-image robust stretch preserves spatial contrast while removing
    # model-specific global gray-level bias.
    lo,hi=np.percentile(x,[50,99.5])
    if hi<=lo+1e-6: return np.clip(x,0,1)
    return np.clip((x-lo)/(hi-lo),0,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vit',type=Path,required=True)
    ap.add_argument('--inp',type=Path,required=True)
    ap.add_argument('--dino',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--zip',type=Path,required=True)
    a=ap.parse_args()
    roots=[a.vit,a.inp,a.dino]
    # A榜结果 indicate INP is strongest; Dinomaly remains useful for mask diversity.
    w=np.asarray([0.15,0.55,0.30],np.float32)
    score_maps=[rank_norm(read_scores(r)) for r in roots]
    keys=sorted(score_maps[0]); out=a.output; out.mkdir(exist_ok=True)
    rows=[]
    for key in keys:
        cat,sample=key.split('/',1); source_score=float(sum(w[i]*score_maps[i][key] for i in range(3)))
        view_scores=[]
        for v in range(5):
            ims=[]
            for r in roots:
                p=r/'predicted_masks'/cat/sample/f'{v}_mask.png'
                with Image.open(p) as im: ims.append(normalize_mask(np.asarray(im.convert('L'))))
            hm=np.clip(sum(w[i]*ims[i] for i in range(3)),0,1)
            # Aggregate with peak + small-area evidence; avoids all-black and
            # avoids making a single noisy pixel dominate the image score.
            flat=hm.ravel(); peak=float(np.percentile(flat,99.5)); area=float((flat>0.45).mean())
            view_scores.append(0.75*peak+0.25*min(area*8,1))
            p=out/'predicted_masks'/cat/sample/f'{v}_mask.png'; p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.uint8(np.round(hm*255)),'L').filter(ImageFilter.GaussianBlur(0.6)).save(p,'PNG',optimize=True)
        mask_score=0.7*max(view_scores)+0.3*float(np.mean(view_scores))
        final=0.65*source_score+0.35*mask_score
        rows.append({'group_folder':key,'anomaly_score':f'{final:.8f}'})
    with (out/'submission.csv').open('w',newline='',encoding='utf-8') as f:
        z=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);z.writeheader();z.writerows(rows)
    zip_path=a.zip
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'submission.csv','submission.csv')
        for p in sorted((out/'predicted_masks').glob('*/*/*_mask.png')): z.write(p,p.relative_to(out).as_posix())
    print('rows',len(rows),'masks',len(list((out/'predicted_masks').glob('*/*/*_mask.png'))),'zip_bytes',zip_path.stat().st_size)

if __name__=='__main__':main()
