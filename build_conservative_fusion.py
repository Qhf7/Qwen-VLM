#!/usr/bin/env python3
import csv, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def read(root):
    with (root/'submission.csv').open() as f:
        return {r['group_folder']:float(r['anomaly_score']) for r in csv.DictReader(f)}
def rank(d):
    k=list(d); x=np.asarray([d[z] for z in k]); o=np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort')
    return {z:float(o[i]/max(1,len(k)-1)) for i,z in enumerate(k)}
def main():
    base=Path('submission_ensemble'); v2=Path('submission_ensemble_v2'); out=Path('submission_ensemble_conservative'); out.mkdir(exist_ok=True)
    a=rank(read(base)); b=rank(read(v2)); rows=[]
    for key in sorted(a): rows.append({'group_folder':key,'anomaly_score':f'{(0.9*a[key]+0.1*b[key]):.8f}'})
    with (out/'submission.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
    for p in (base/'predicted_masks').glob('*/*/*_mask.png'):
        q=out/'predicted_masks'/p.relative_to(base/'predicted_masks');q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    zpath=Path('submission_ensemble_conservative.zip')
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'submission.csv','submission.csv')
        for p in sorted((out/'predicted_masks').glob('*/*/*_mask.png')):z.write(p,p.relative_to(out).as_posix())
    print('rows',len(rows),'masks',len(list((out/'predicted_masks').glob('*/*/*_mask.png'))),'zip',zpath.stat().st_size)
if __name__=='__main__':main()
