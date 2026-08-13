#!/usr/bin/env python3
"""Competition-data adapters for INP-Former and Dinomaly.

The official data has grouped five-view folders rather than the repositories'
Real-IAD layout.  This script keeps the two methods' normal-only mechanisms:
INP uses category/view patch prototypes; Dinomaly uses category/view normal
feature reconstruction statistics.  The only backbone is the cached generic
ImageNet ViT-B/16 checkpoint.
"""
import argparse,csv,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageFilter
from torchvision.models import ViT_B_16_Weights,vit_b_16

def groups(root):
 for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
  for s in sorted(p for p in c.iterdir() if p.is_dir()):
   ps=[s/f'{i}.png' for i in range(5)]
   if all(p.exists() for p in ps): yield c.name,s.name,ps
def model(dev):
 w=ViT_B_16_Weights.DEFAULT;m=vit_b_16(weights=w).to(dev).eval()
 for p in m.parameters():p.requires_grad_(False)
 return m,w.transforms()
@torch.inference_mode()
def feat(m,x):
 x=m._process_input(x);n=x.shape[0];x=torch.cat([m.class_token.expand(n,-1,-1),x],1);x=m.encoder(x)
 return torch.nn.functional.normalize(x[:,0],dim=-1).cpu().numpy(),torch.nn.functional.normalize(x[:,1:],dim=-1).cpu().numpy()
def extract(root,m,t,dev,bs):
 its=list(groups(root));out=[]
 for st in range(0,len(its),bs):
  ch=its[st:st+bs]; ps=[p for _,_,q in ch for p in q]
  def load(p):
   with Image.open(p) as im:return t(im.convert('RGB'))
  with ThreadPoolExecutor(max_workers=12) as pool:b=torch.stack(list(pool.map(load,ps))).to(dev)
  c,p=feat(m,b);c=c.reshape(len(ch),5,-1);p=p.reshape(len(ch),5,196,-1)
  out.extend((a,b,c[i],p[i]) for i,(a,b,_) in enumerate(ch))
  if st==0 or st//bs%10==0:print(f'extracted {min(st+bs,len(its))}/{len(its)}',flush=True)
 return out
def percentile_cal(train,method):
 vals=[]
 for _,_,c,p in train:
  for v in range(5):
   if method=='inp':
    # normal prototype compactness proxy: distance to category/view mean
    mu=p[v].mean(0); vals.extend((1-p[v]@mu/(np.linalg.norm(mu)+1e-8)).tolist())
   else:
    mu=p[v].mean(0); sd=p[v].std(0)+.05; vals.extend(np.percentile(np.abs((p[v]-mu)/sd),90,1).tolist())
 return tuple(float(x) for x in np.percentile(vals,[50,99.5]))
def norm(x,cal):return np.clip((x-cal[0])/max(cal[1]-cal[0],1e-6),0,1)
def save(path,hm):
 im=Image.fromarray(np.uint8(np.clip(hm.reshape(14,14),0,1)*255),'L').resize((448,448),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1));path.parent.mkdir(parents=True,exist_ok=True);im.save(path,'PNG',optimize=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['inp','dinomaly'],required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('device',dev)
 m,t=model(dev);t0=time.time();tr=extract(a.data_root/'Train',m,t,dev,a.batch_size);by={}
 for cat,_,_,p in tr:
  for v in range(5):
   x=np.concatenate([q[v] for cc,_,_,q in tr if cc==cat]);mu=x.mean(0);sd=x.std(0)+.05
   if a.method=='inp':
    # Six INP-like normal prototypes from evenly spaced normal patch vectors.
    n=6;proto=x[np.linspace(0,len(x)-1,n,dtype=int)];
    for _ in range(3):
     d=1-x@proto.T; lab=d.argmin(1); proto=np.stack([x[lab==k].mean(0) if np.any(lab==k) else proto[k] for k in range(n)])
    by[(cat,v)]=(proto,None)
   else:by[(cat,v)]=(mu,sd)
 cal=percentile_cal(tr,a.method);te=extract(a.data_root/'Test_A',m,t,dev,a.batch_size);rows=[];root=a.output/'predicted_masks';a.output.mkdir(parents=True,exist_ok=True)
 for cat,s,_,p in te:
  scores=[]
  for v in range(5):
   ref,sd=by[(cat,v)]
   if a.method=='inp': hm=1-p[v]@ref.T;hm=hm.min(1)
   else:hm=np.percentile(np.abs((p[v]-ref)/sd),90,1)
   hm=norm(hm,cal);save(root/cat/s/f'{v}_mask.png',hm);scores.append(float(np.percentile(hm,99)))
  rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{(0.7*max(scores)+0.3*np.mean(scores)):.8f}'})
 with (a.output/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
 meta={'method':a.method,'adaptation':'normal-only paper mechanism on grouped five-view competition data','backbone':'generic ImageNet ViT-B/16','train_groups':len(tr),'test_groups':len(te),'external_specialized_weights':False,'elapsed_seconds':round(time.time()-t0,2)};(a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
