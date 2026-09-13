"""Audit the inputs required before frozen-Qwen3 segmentation training.

This is deliberately a read-only check. It verifies the fold manifest, 448px
image cache, GT masks, labels and class balance, then writes a JSON report.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def sha256(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dev',type=Path,default=Path('artifacts/splits/fold1/dev.parquet'))
    p.add_argument('--train',type=Path,default=Path('artifacts/splits/fold1/train.parquet'))
    p.add_argument('--image-root',type=Path,default=None); p.add_argument('--train-image-root',type=Path,default=Path('Train448_verified_20260905')); p.add_argument('--dev-image-root',type=Path,default=Path('Dataset448'))
    p.add_argument('--gt-root',type=Path,default=Path('Dataset'))
    p.add_argument('--out',type=Path,default=Path('artifacts/segmentation_readiness.json')); a=p.parse_args()
    rows=[]; errors=[]
    for split,path in [('train',a.train),('dev',a.dev)]:
        d=pd.read_parquet(path)
        for i,r in d.iterrows():
            image=str(r.image_path); ip=(a.train_image_root if split=='train' else a.dev_image_root)/image
            if not ip.is_file(): errors.append({'split':split,'image':image,'error':'missing_cached_image'}); continue
            try:
                with Image.open(ip) as im:
                    if im.size!=(448,448): errors.append({'split':split,'image':image,'error':f'bad_image_size:{im.size}'})
            except Exception as e: errors.append({'split':split,'image':image,'error':f'image_read:{e}'})
            mp=r.get('mask_path',None); mask_ok=False; area=0
            if isinstance(mp,str) and mp and (a.gt_root/mp).is_file():
                try:
                    with Image.open(a.gt_root/mp) as m:
                        x=np.asarray(m.convert('L'))>0; area=int(x.sum()); mask_ok=True
                except Exception as e: errors.append({'split':split,'image':image,'error':f'mask_read:{e}'})
            elif float(r.get('view_label', r.get('sample_label',0)))>0:
                errors.append({'split':split,'image':image,'error':'missing_positive_mask'})
            rows.append({'split':split,'class_name':str(r.get('class_name','')),'label':int(r.get('view_label',r.get('sample_label',0))), 'mask_ok':mask_ok,'mask_area':area})
    out={'protocol':'frozen-qwen3-segmentation-readiness-v1','dev_sha256':sha256(a.dev),'train_sha256':sha256(a.train),'n_rows':len(rows),'errors':errors,'error_count':len(errors),'splits':{}}
    df=pd.DataFrame(rows)
    for split,g in df.groupby('split'):
        out['splits'][split]={'rows':len(g),'classes':int(g.class_name.nunique()),'positive':int(g.label.sum()),'negative':int((g.label==0).sum()),'positive_masks':int((g.mask_ok & g.label.eq(1)).sum()),'nonempty_masks':int((g.mask_area>0).sum())}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(1 if errors else 0)
if __name__=='__main__': main()
