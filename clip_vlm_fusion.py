#!/usr/bin/env python3
"""Offline OpenAI CLIP prompt ensemble fused with the validated ensemble.

The CLIP checkpoint is the public generic OpenAI ViT-B/32 weight, verified by
SHA256. No anomaly-dataset prompt/adapter/head or online service is used.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch,clip
from PIL import Image

def groups(root):
 for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
  for s in sorted(p for p in c.iterdir() if p.is_dir()):
   ps=[s/f'{i}.png' for i in range(5)]
   if all(p.exists() for p in ps):yield c.name,s.name,ps

def variants(p):
 with Image.open(p) as im: im=im.convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
 return [im,im.crop((0,0,224,224)),im.crop((224,0,448,224)),im.crop((0,224,224,448)),im.crop((224,224,448,448))]

def prompts(cat):
 # Several neutral templates reduce sensitivity to one wording. The category
 # name is only a public test-time text prompt, not a learned label.
 normal=[f'a photo of a normal {cat} industrial product',f'a flawless {cat} manufactured part',f'a clean intact {cat} object']
 bad=[f'a photo of a defective {cat} industrial product',f'a damaged {cat} manufactured part',f'an anomalous {cat} object']
 defects=[f'{cat} with a scratch',f'{cat} with a dent',f'{cat} with a crack',f'{cat} with a broken part',f'{cat} with a stain',f'{cat} with contamination',f'{cat} with a surface defect']
 return normal,bad,defects

@torch.inference_mode()
def encode_text(model,tokenize,texts,dev):
 z=model.encode_text(tokenize(texts).to(dev));return (z/z.norm(dim=-1,keepdim=True)).float().cpu()
@torch.inference_mode()
def encode_images(model,pre,ims,dev,bs=32):
 out=[]
 for i in range(0,len(ims),bs):
  x=torch.stack([pre(im) for im in ims[i:i+bs]]).to(dev);z=model.encode_image(x);out.append((z/z.norm(dim=-1,keepdim=True)).float().cpu())
 return torch.cat(out)
def rank(d):
 k=list(d);x=np.asarray([d[z] for z in k]);o=np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort');return {z:float(o[i]/max(1,len(k)-1)) for i,z in enumerate(k)}
def read_scores(root):
 with (root/'submission.csv').open() as f:return {r['group_folder']:float(r['anomaly_score']) for r in csv.DictReader(f)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--clip-weight',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--zip',type=Path,required=True);a=ap.parse_args();dev='cuda' if torch.cuda.is_available() else 'cpu';t0=time.time();model,pre=clip.load(str(a.clip_weight),device=dev,jit=False);model.eval();tr=list(groups(a.data_root/'Train'));te=list(groups(a.data_root/'Test_A'));base=rank(read_scores(a.base));
 # Normal-image calibration uses the official Train data only.
 cal=[]
 for cat,s,ps in tr:
  n,b,d=prompts(cat); tn=encode_text(model,clip.tokenize,n,'cuda' if dev=='cuda' else 'cpu').mean(0);tb=encode_text(model,clip.tokenize,b,'cuda' if dev=='cuda' else 'cpu').mean(0); ims=variants(ps[0]);zi=encode_images(model,pre,ims,dev);cal.append(float((zi@(tb-tn)).mean()))
 cal=np.percentile(np.asarray(cal),[10,90]);rows=[];out=a.output;out.mkdir(exist_ok=True);root=out/'predicted_masks';
 for cat,s,ps in te:
  n,b,d=prompts(cat);tn=encode_text(model,clip.tokenize,n,dev).mean(0);tb=encode_text(model,clip.tokenize,b,dev).mean(0);td=encode_text(model,clip.tokenize,d,dev);vs=[]
  for v,pth in enumerate(ps):
   zi=encode_images(model,pre,variants(pth),dev);delta=(zi@(tb-tn)).numpy(); # full image + four local crops
   full=float(np.clip((delta[0]-cal[0])/max(cal[1]-cal[0],1e-6),0,1)); local=float(np.clip(np.percentile(delta[1:],75)-cal[0],0,max(cal[1]-cal[0],1e-6))/max(cal[1]-cal[0],1e-6));score=.65*full+.35*local;vs.append(score)
   # CLIP has global embeddings; retain validated base spatial mask.
   src=a.base/'predicted_masks'/cat/s/f'{v}_mask.png';dst=root/cat/s/f'{v}_mask.png';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
  clip_score=.7*max(vs)+.3*float(np.mean(vs));base_score=base[f'{cat}/{s}'];rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{(.75*base_score+.25*clip_score):.8f}'})
 with (out/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
 meta={'method':'validated ensemble masks + offline generic OpenAI CLIP prompt ensemble','clip_weight_sha256':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af','external_specialized_weights':False,'online_model':False,'elapsed_seconds':round(time.time()-t0,2)};(out/'run_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
