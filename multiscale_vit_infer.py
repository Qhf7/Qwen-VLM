#!/usr/bin/env python3
"""Rule-compliant multiscale frozen ViT baseline.

Uses only official normal Train images to estimate category/view patch statistics.
The generic ImageNet ViT-B/16 weights are reused at native 224px input.  Each
448px image is evaluated as a full-image view and four overlapping 224px tiles;
tile patch scores are stitched back to a 28x28 heatmap, improving sensitivity
to small defects without any external anomaly data or specialized weights.
"""
from __future__ import annotations
import argparse, csv, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.models import ViT_B_16_Weights, vit_b_16

def groups(root):
    for cat in sorted(p for p in root.iterdir() if p.is_dir() and p.name != '__MACOSX'):
        for sample in sorted(p for p in cat.iterdir() if p.is_dir()):
            ps=[sample/f'{i}.png' for i in range(5)]
            if all(p.exists() for p in ps): yield cat.name, sample.name, ps

def load_model(device):
    w=ViT_B_16_Weights.DEFAULT; m=vit_b_16(weights=w).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m,w.transforms()

@torch.inference_mode()
def feat(model,x):
    x=model._process_input(x); n=x.shape[0]
    x=torch.cat([model.class_token.expand(n,-1,-1),x],1)
    x=model.encoder(x)
    return torch.nn.functional.normalize(x[:,0],dim=-1).cpu(), torch.nn.functional.normalize(x[:,1:],dim=-1).cpu()

def variants(im, transform):
    # Full image plus four 224px crops from the 448px coordinate system.
    im=im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
    crops=[im, im.crop((0,0,224,224)), im.crop((224,0,448,224)),
           im.crop((0,224,224,448)), im.crop((224,224,448,448))]
    return torch.stack([transform(z) for z in crops])

def extract(root,model,transform,device,bs):
    items=list(groups(root)); out=[]
    for st in range(0,len(items),bs):
        chunk=items[st:st+bs]; tensors=[]
        for _,_,ps in chunk:
            for p in ps:
                with Image.open(p) as im: tensors.append(variants(im,transform))
        b=torch.cat(tensors).to(device); c,p=feat(model,b)
        c=c.view(len(chunk),5,5,-1); p=p.view(len(chunk),5,5,196,-1)
        for j,(cat,sample,_) in enumerate(chunk): out.append((cat,sample,c[j].numpy(),p[j].numpy()))
        if st==0 or (st//bs)%10==0: print(f'extracted {min(st+bs,len(items))}/{len(items)}',flush=True)
    return out

def stats(records):
    by={}; glob={}
    for cat,_,_,p in records:
        for v in range(5):
            # Full-image stream and tile streams have different spatial meaning.
            for scale in range(5):
                by.setdefault((cat,v,scale),[]).append(p[v,scale])
                glob.setdefault((v,scale),[]).append(p[v,scale])
    def finish(d, concat=False):
        z={}
        for k,a in d.items():
            x=np.stack(a).astype('float32'); z[k]=(x.mean(0),x.std(0)+0.05)
        return z
    return finish(by),finish(glob)

def robust(x,lo=1.,hi=8.): return np.clip((x-lo)/(hi-lo),0,1)

def score(p,mean,std):
    z=np.abs((p-mean)/std); return np.percentile(z,90,axis=1)

def stitch(tile_scores):
    # tile maps are 14x14 and placed in a 28x28 grid; full map is upsampled.
    full=tile_scores[0].reshape(14,14)
    grid=np.zeros((28,28),np.float32); cnt=np.zeros_like(grid)
    for i,(r,c) in enumerate([(0,0),(0,14),(14,0),(14,14)],1):
        grid[r:r+14,c:c+14]+=tile_scores[i].reshape(14,14); cnt[r:r+14,c:c+14]+=1
    grid/=np.maximum(cnt,1)
    full=np.asarray(Image.fromarray(full).resize((28,28),Image.Resampling.BICUBIC))
    return 0.45*full+0.55*grid

def save(path,s):
    # `s` is already calibrated to [0, 1] against official-normal scores.
    x=np.clip(s,0,1).reshape(28,28)
    im=Image.fromarray(np.uint8(x*255), 'L').resize((448,448),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1))
    path.parent.mkdir(parents=True,exist_ok=True); im.save(path,'PNG',optimize=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=Path('Dataset')); ap.add_argument('--output',type=Path,default=Path('submission_vit_multiscale')); ap.add_argument('--batch-size',type=int,default=4)
    a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',dev)
    m,t=load_model(dev); t0=time.time(); tr=extract(a.data_root/'Train',m,t,dev,a.batch_size); by,gl=stats(tr)
    # Calibrate each stream from official normal data. Fixed absolute thresholds
    # are inappropriate because ViT feature scales differ by category/view.
    cal={}
    for v in range(5):
        for sc in range(5):
            vals=[]
            for cat,_,_,p in tr:
                mean,std=by.get((cat,v,sc),gl[(v,sc)])
                vals.extend(score(p[v,sc],mean,std).tolist())
            lo,hi=np.percentile(np.asarray(vals,np.float32),[50,99.5])
            cal[(v,sc)]=(float(lo),float(max(hi,lo+1e-3)))
    te=extract(a.data_root/'Test_A',m,t,dev,a.batch_size)
    rows=[]; root=a.output/'predicted_masks'; a.output.mkdir(parents=True,exist_ok=True)
    for cat,sample,_,p in te:
        vs=[]
        for v in range(5):
            ss=[]
            for sc in range(5):
                mean,std=by.get((cat,v,sc),gl[(v,sc)])
                raw=score(p[v,sc],mean,std); lo,hi=cal[(v,sc)]
                ss.append(np.clip((raw-lo)/(hi-lo),0,1))
            hm=stitch(ss); save(root/cat/sample/f'{v}_mask.png',hm); vs.append(float(np.percentile(hm,99)))
        ns=[float(np.clip(x,0,1)) for x in vs]; rows.append({'group_folder':f'{cat}/{sample}','anomaly_score':f'{(0.7*max(ns)+0.3*np.mean(ns)):.8f}'})
    with (a.output/'submission.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']); w.writeheader(); w.writerows(rows)
    meta={'method':'frozen ImageNet ViT-B/16 multiscale; official-normal quantile calibration','train_groups':len(tr),'test_groups':len(te),'mask_size':[448,448],'external_specialized_weights':False,'elapsed_seconds':round(time.time()-t0,2)}
    (a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
