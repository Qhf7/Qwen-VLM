#!/usr/bin/env python3
"""Normal-only pseudo-anomaly training on the competition data.

No external anomaly data is used. Frozen generic ImageNet ViT features are
extracted from official normal Train images and synthetic CutPaste/occlusion/
texture defects. Only a small patch classifier is trained.
"""
from __future__ import annotations
import argparse,csv,json,random,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageFilter,ImageEnhance
from torchvision.models import ViT_B_16_Weights,vit_b_16

def groups(root):
 for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
  for s in sorted(p for p in c.iterdir() if p.is_dir()):
   ps=[s/f'{i}.png' for i in range(5)]
   if all(p.exists() for p in ps): yield c.name,s.name,ps

def load_model(dev):
 w=ViT_B_16_Weights.DEFAULT;m=vit_b_16(weights=w).to(dev).eval()
 for p in m.parameters():p.requires_grad_(False)
 return m,w.transforms()

def synth(im,src,kind):
 im=im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC); src=src.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
 x=im.copy(); mask=Image.new('L',(448,448),0); md=mask.load()
 w=random.randint(32,150); h=random.randint(24,130); x0=random.randint(0,448-w); y0=random.randint(0,448-h)
 if kind==0:
  sx=random.randint(0,448-w); sy=random.randint(0,448-h)
  patch=src.crop((sx,sy,sx+w,sy+h)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
  patch=ImageEnhance.Contrast(patch).enhance(random.uniform(0.5,1.8)); x.paste(patch,(x0,y0))
 elif kind==1:
  patch=Image.new('RGB',(w,h),tuple(random.randint(0,255) for _ in range(3))); x.paste(patch,(x0,y0))
 else:
  a=np.random.rand(h,w).astype('float32'); a=np.asarray(Image.fromarray(np.uint8(a*255),'L').filter(ImageFilter.GaussianBlur(5)),np.float32)/255
  base=np.asarray(x.crop((x0,y0,x0+w,y0+h)),np.float32); tex=np.clip(base*(0.45+1.2*a[...,None]),0,255).astype('uint8'); x.paste(Image.fromarray(tex), (x0,y0))
 for yy in range(y0,y0+h):
  for xx in range(x0,x0+w): md[xx,yy]=255
 return x,mask

@torch.inference_mode()
def feat(m,x):
 x=m._process_input(x);n=x.shape[0];x=torch.cat([m.class_token.expand(n,-1,-1),x],1);x=m.encoder(x)
 return torch.nn.functional.normalize(x[:,0],dim=-1).cpu(),torch.nn.functional.normalize(x[:,1:],dim=-1).cpu()

def tensor(im,t): return t(im.convert('RGB'))

def extract_train(items,m,t,dev,bs):
 clean=[]; pseudo=[]; labels=[]
 all_paths=[p for _,_,ps in items for p in ps]
 for st in range(0,len(all_paths),bs):
  ps=all_paths[st:st+bs]; ims=[]; pims=[]; labs=[]
  for i,p in enumerate(ps):
   with Image.open(p) as im:
    ims.append(tensor(im,t));
    with Image.open(all_paths[(st+i+13)%len(all_paths)]) as src:
     z,mask=synth(im,src,(st+i)%3);pims.append(tensor(z,t))
    ma=np.asarray(mask.resize((14,14),Image.Resampling.BOX),np.float32)/255; labs.append((ma>0.18).astype('float32').reshape(-1))
  c,p=feat(m,torch.stack(ims).to(dev)); _,q=feat(m,torch.stack(pims).to(dev))
  clean.append(p);pseudo.append(q);labels.extend(labs)
  if st==0 or st//bs%20==0:print(f'extracted {min(st+bs,len(all_paths))}/{len(all_paths)}',flush=True)
 return torch.cat(clean),torch.cat(pseudo),torch.tensor(np.asarray(labels),dtype=torch.float32)

def extract_test(items,m,t,dev,bs):
 out=[];flat=[p for _,_,ps in items for p in ps]
 for st in range(0,len(flat),bs):
  ps=flat[st:st+bs]; ims=[]
  for p in ps:
   with Image.open(p) as im:ims.append(tensor(im,t))
  _,q=feat(m,torch.stack(ims).to(dev));out.append(q)
  if st==0 or st//bs%20==0:print(f'test {min(st+bs,len(flat))}/{len(flat)}',flush=True)
 return torch.cat(out).view(len(items),5,196,-1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--epochs',type=int,default=8);a=ap.parse_args();random.seed(7);np.random.seed(7);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('device',dev)
 m,t=load_model(dev);t0=time.time();tr=list(groups(a.data_root/'Train'));te=list(groups(a.data_root/'Test_A')); clean,pseudo,y=extract_train(tr,m,t,dev,a.batch_size)
 # lightweight patch head: normal=0, synthetic defect=1
 d=clean.shape[-1];head=torch.nn.Sequential(torch.nn.Linear(d,256),torch.nn.GELU(),torch.nn.LayerNorm(256),torch.nn.Linear(256,1)).to(dev)
 opt=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-4);pos=(y.sum()+1)/(y.numel()-y.sum()+1);loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos.to(dev))
 x=torch.cat([clean,pseudo],0).to(dev); yy=torch.cat([torch.zeros(clean.shape[0],clean.shape[1]),y],0).to(dev).reshape(-1,1)
 ds=torch.utils.data.TensorDataset(x.reshape(-1,d),yy);dl=torch.utils.data.DataLoader(ds,batch_size=4096,shuffle=True)
 head.train()
 for ep in range(a.epochs):
  ls=[]
  for xb,yb in dl:opt.zero_grad();pred=head(xb);loss=loss_fn(pred,yb);loss.backward();opt.step();ls.append(loss.item())
  print('epoch',ep+1,'loss',float(np.mean(ls)),flush=True)
 head.eval();test=extract_test(te,m,t,dev,a.batch_size);rows=[];root=a.output/'predicted_masks';a.output.mkdir(parents=True,exist_ok=True);all_scores=[]
 with torch.inference_mode():
  for cat,s,_, in te:
   pass
 idx=0
 for cat,s,_ in te:
  vs=[]
  for v in range(5):
   z=torch.sigmoid(head(test[idx,v].to(dev))).reshape(14,14).detach().cpu().numpy(); z=np.clip((z-0.05)/0.85,0,1);vs.append(float(np.percentile(z,99)));im=Image.fromarray(np.uint8(z*255),'L').resize((448,448),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1));p=root/cat/s/f'{v}_mask.png';p.parent.mkdir(parents=True,exist_ok=True);im.save(p,'PNG',optimize=True)
  idx+=1;ns=[np.clip(x,0,1) for x in vs];rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{(0.7*max(ns)+0.3*np.mean(ns)):.8f}'})
 with (a.output/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
 meta={'method':'frozen generic ViT + normal-only CutPaste/occlusion/texture pseudo-anomaly patch head','train_groups':len(tr),'test_groups':len(te),'external_specialized_weights':False,'elapsed_seconds':round(time.time()-t0,2)};(a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
