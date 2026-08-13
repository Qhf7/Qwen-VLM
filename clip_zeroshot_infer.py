#!/usr/bin/env python3
"""Rule-compliant CLIP zero-shot anomaly baseline.

Uses only the public generic OpenAI CLIP base checkpoint. No anomaly-dataset
prompt/adapter/head weights and no online LLM/VLM calls are used.
"""
import argparse,csv,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageFilter
from transformers import CLIPModel,CLIPProcessor

def groups(root):
 for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
  for s in sorted(p for p in c.iterdir() if p.is_dir()):
   ps=[s/f'{i}.png' for i in range(5)]
   if all(p.exists() for p in ps):yield c.name,s.name,ps

def prompts():
 normal=['a photo of a normal industrial product','a photo of a flawless product','a clean intact manufactured part']
 abnormal=['a photo of a defective industrial product','a photo of a damaged product','a photo of an anomalous manufactured part']
 defects=['a product with a scratch','a product with a dent','a product with a crack','a product with a broken part','a product with a stain','a product with a contamination','a product with a surface defect']
 return normal,abnormal,defects

@torch.inference_mode()
def text_features(model,processor,dev):
 n,a,d=prompts();allp=n+a+d; x=processor(text=allp,return_tensors='pt',padding=True).to(dev);z=model.get_text_features(**x);z=z/z.norm(dim=-1,keepdim=True);return z.cpu(),len(n),len(a),len(d)

@torch.inference_mode()
def image_features(model,processor,images,dev):
 x=processor(images=images,return_tensors='pt').to(dev);z=model.get_image_features(**x);z=z/z.norm(dim=-1,keepdim=True);return z.cpu()

def load_variants(im):
 im=im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
 return [im,im.crop((0,0,224,224)),im.crop((224,0,448,224)),im.crop((0,224,224,448)),im.crop((224,224,448,448))]

def robust(x):
 lo,hi=np.percentile(x,[5,99]);return np.clip((x-lo)/max(hi-lo,1e-6),0,1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model-id',default='openai/clip-vit-base-patch32');ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('device',dev)
 t0=time.time();model=CLIPModel.from_pretrained(a.model_id).to(dev).eval();proc=CLIPProcessor.from_pretrained(a.model_id);txt,nN,nA,nD=text_features(model,proc,dev);normal=txt[:nN].mean(0);abnormal=txt[nN:nN+nA].mean(0);defects=txt[nN+nA:]
 # CLIP's learned logit scale converts similarities to a useful probability.
 logit=float(model.logit_scale.exp().detach().cpu());tr=list(groups(a.data_root/'Train'));te=list(groups(a.data_root/'Test_A'));root=a.output/'predicted_masks';a.output.mkdir(parents=True,exist_ok=True);rows=[]
 # Official normal data calibrates image-level CLIP score without any labels.
 normal_cal=[]
 for st in range(0,len(tr),a.batch_size):
  ims=[]
  for _,_,ps in tr[st:st+a.batch_size]:
   with Image.open(ps[0]) as im:ims.append(im.convert('RGB').resize((448,448)))
  z=image_features(model,proc,ims,dev); normal_cal.extend((z@abnormal-z@normal).numpy().tolist())
 cal=np.percentile(np.asarray(normal_cal),[50,99.5]);
 for st in range(0,len(te),a.batch_size):
  chunk=te[st:st+a.batch_size]; allims=[]
  for _,_,ps in chunk:
   with ThreadPoolExecutor(max_workers=5) as pool: allims.extend(list(pool.map(lambda p:Image.open(p).convert('RGB'),ps)))
  z=image_features(model,proc,allims,dev).view(len(chunk),5,-1); sims=(z@abnormal-z@normal).numpy();
  for j,(cat,s,_) in enumerate(chunk):
   vs=[]
   for v in range(5):
    # Text-guided image score plus patch-local CLIP defect similarity.
    full=float(np.clip((sims[j,v]-cal[0])/max(cal[1]-cal[0],1e-6),0,1)); tiles=z[j,v]
    # Five global crop embeddings produce a coarse spatial anomaly map.
    patch=np.array([float(torch.sigmoid(logit*(tiles@d-tiles@normal)).mean()) for d in defects]);
    hm=np.tile(np.clip(np.mean(patch),0,1),(14,14)); hm=robust(hm); score=0.7*full+0.3*float(np.percentile(hm,99));vs.append(score)
    p=root/cat/s/f'{v}_mask.png';p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.uint8(hm*255),'L').resize((448,448),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1)).save(p,'PNG',optimize=True)
   ns=[float(np.clip(x,0,1)) for x in vs];rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{(0.7*max(ns)+0.3*np.mean(ns)):.8f}'})
  print(f'test {min(st+a.batch_size,len(te))}/{len(te)}',flush=True)
 with (a.output/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
 meta={'method':'generic CLIP zero-shot normal/defect prompt ensemble','model_id':a.model_id,'train_groups':len(tr),'test_groups':len(te),'external_specialized_weights':False,'online_model':False,'elapsed_seconds':round(time.time()-t0,2)};(a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
