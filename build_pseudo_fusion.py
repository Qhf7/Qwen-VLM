#!/usr/bin/env python3
import argparse,csv,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
def scores(r):
 with (r/'submission.csv').open() as f:return {x['group_folder']:float(x['anomaly_score']) for x in csv.DictReader(f)}
def rank(d):
 k=list(d);x=np.asarray([d[z] for z in k]);o=np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort');return {z:float(o[i]/max(1,len(k)-1)) for i,z in enumerate(k)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--pseudo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--zip',type=Path,required=True);a=ap.parse_args(); base=a.base;pseudo=a.pseudo;out=a.output;out.mkdir(exist_ok=True);b=rank(scores(base));q=rank(scores(pseudo));rows=[]
 for key in sorted(b):
  cat,s=key.split('/',1);val=.8*b[key]+.2*q[key];rows.append({'group_folder':key,'anomaly_score':f'{val:.8f}'})
  for v in range(5):
   with Image.open(base/'predicted_masks'/cat/s/f'{v}_mask.png') as im:x=np.asarray(im.convert('L'),np.float32)/255
   with Image.open(pseudo/'predicted_masks'/cat/s/f'{v}_mask.png') as im:y=np.asarray(im.convert('L'),np.float32)/255
   # Pseudo head is intentionally a modest residual signal; preserve the
   # validated ensemble spatial map as the primary mask.
   z=np.clip(.8*x+.2*y,0,1);p=out/'predicted_masks'/cat/s/f'{v}_mask.png';p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.uint8(z*255),'L').filter(ImageFilter.GaussianBlur(.5)).save(p,'PNG',optimize=True)
 with (out/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
 with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(out/'submission.csv','submission.csv')
  for p in sorted((out/'predicted_masks').glob('*/*/*_mask.png')):z.write(p,p.relative_to(out).as_posix())
 print('rows',len(rows),'masks',len(list((out/'predicted_masks').glob('*/*/*_mask.png'))),'zip_bytes',a.zip.stat().st_size)
if __name__=='__main__':main()
