#!/usr/bin/env python3
"""Independent compliant VLM + patch-prototype anomaly detector.

The model is built from the official data and generic public weights only:
OpenAI CLIP ViT-B/32 for zero-shot semantic scoring and ImageNet ViT-B/16
patch features for pixel localization. No previous submission scores/masks,
external anomaly data, or test labels are used.
"""
import argparse,csv,json,time,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16
import clip

def groups(root):
    for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
        for s in sorted(p for p in c.iterdir() if p.is_dir()):
            ps=[s/f'{i}.png' for i in range(5)]
            if all(p.exists() for p in ps): yield c.name,s.name,ps

def load_vit(dev):
    w=ViT_B_16_Weights.DEFAULT; m=vit_b_16(weights=w).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m,w.transforms()

@torch.inference_mode()
def vit_feat(m,x):
    x=m._process_input(x); n=x.shape[0]
    x=torch.cat([m.class_token.expand(n,-1,-1),x],1)
    x=m.encoder(x)
    return torch.nn.functional.normalize(x[:,0],dim=-1),torch.nn.functional.normalize(x[:,1:],dim=-1)

def read_batch(items,trans,dev):
    ps=[p for _,_,q in items for p in q]
    with ThreadPoolExecutor(max_workers=12) as pool:
        xs=list(pool.map(lambda p: trans(Image.open(p).convert('RGB')),ps))
    return torch.stack(xs).to(dev)

def prompts(cat):
    n=[f'a normal {cat} industrial product',f'a flawless {cat} manufactured part',f'a clean intact {cat} object']
    a=[f'a defective {cat} industrial product',f'a damaged {cat} manufactured part',f'an anomalous {cat} object']
    d=[f'{cat} with a scratch',f'{cat} with a dent',f'{cat} with a crack',f'{cat} with a stain',f'{cat} with a surface defect']
    return n,a,d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--clip-weight',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--zip',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=20); a=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t0=time.time()
    vm,vt=load_vit(dev); cm,cp=clip.load(str(a.clip_weight),device=str(dev),jit=False); cm.eval()
    tr=list(groups(a.data_root/'Train')); te=list(groups(a.data_root/'Test_A'))
    # Category/view normal patch prototype and global scale, official Train only.
    sums={}; sqs={}; cnt={}; cat_global={}
    for st in range(0,len(tr),a.batch_size):
        ch=tr[st:st+a.batch_size]; x=read_batch(ch,vt,dev)
        _,p=vit_feat(vm,x); p=p.view(len(ch),5,196,-1).cpu().numpy()
        for i,(cat,_,_) in enumerate(ch):
            for v in range(5):
                z=p[i,v]; k=(cat,v); sums[k]=sums.get(k,0)+z; sqs[k]=sqs.get(k,0)+(z*z); cnt[k]=cnt.get(k,0)+1
                cat_global.setdefault(cat,[]).append(z)
        if st==0 or st%200==0: print('train',min(st+a.batch_size,len(tr)),flush=True)
    proto={}; scale={}
    for k in sums:
        mu=sums[k]/cnt[k]; var=np.maximum(sqs[k]/cnt[k]-mu*mu,1e-4); proto[k]=mu/np.linalg.norm(mu,axis=1,keepdims=True); scale[k]=float(np.sqrt(var).mean())
    # Zero-shot text embeddings and normal-image calibration using Train.
    text={};
    with torch.inference_mode():
        for cat in sorted({x[0] for x in tr+te}):
            n,b,d=prompts(cat); z=cm.encode_text(clip.tokenize(n+b+d).to(dev)); z=z/z.norm(dim=-1,keepdim=True); text[cat]=(z[:len(n)].mean(0),z[len(n):len(n)+len(b)].mean(0),z[len(n)+len(b):])
    normal_clip=[]
    for st in range(0,len(tr),a.batch_size):
        ch=tr[st:st+a.batch_size]; ims=[]
        for cat,_,ps in ch:
            with Image.open(ps[0]) as im: ims.append(cp(im.convert('RGB')))
        with torch.inference_mode(): zi=cm.encode_image(torch.stack(ims).to(dev)); zi=zi/zi.norm(dim=-1,keepdim=True)
        for i,(cat,_,_) in enumerate(ch):
            n,b,_=text[cat]; normal_clip.append(float(zi[i]@(b-n)))
    c10,c90=np.percentile(normal_clip,[5,99]); out=a.output; out.mkdir(parents=True,exist_ok=True); root=out/'predicted_masks'; rows=[]
    for st in range(0,len(te),a.batch_size):
        ch=te[st:st+a.batch_size]; x=read_batch(ch,vt,dev)
        with torch.inference_mode():
            gi,pp=vit_feat(vm,x); pp=pp.view(len(ch),5,196,-1).cpu().numpy(); zi=cm.encode_image(torch.stack([cp(Image.open(p).convert('RGB')) for _,_,q in ch for p in [q[0]]]).to(dev)); zi=zi/zi.norm(dim=-1,keepdim=True)
        for i,(cat,s,_) in enumerate(ch):
            n,b,d=text[cat]; clip_s=float(zi[i]@(b-n)); clip_s=float(np.clip((clip_s-c10)/max(c90-c10,1e-6),0,1)); vs=[]
            for v in range(5):
                z=pp[i,v]; mu=proto[(cat,v)]; dist=1-(z@mu.T).max(1); dist=(dist-np.percentile(dist,5))/max(np.percentile(dist,99.5)-np.percentile(dist,5),1e-6); dist=np.clip(dist,0,1); hm=dist.reshape(14,14)
                # Area-aware pixel map: bilinear upsampling preserves local evidence.
                h=torch.from_numpy(hm).float()[None,None]; h=torch.nn.functional.interpolate(h,size=(448,448),mode='bilinear',align_corners=False)[0,0].numpy(); h=np.clip(h,0,1)
                ps=Path(root/cat/s/f'{v}_mask.png'); ps.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.uint8(h*255),'L').save(ps,'PNG',optimize=True)
                vs.append(float(np.percentile(dist,99.5)))
            patch_s=.7*max(vs)+.3*float(np.mean(vs)); rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{(.65*patch_s+.35*clip_s):.8f}'})
        print('test',min(st+a.batch_size,len(te)),flush=True)
    with (out/'submission.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']);w.writeheader();w.writerows(rows)
    meta={'method':'generic CLIP zero-shot + ImageNet ViT patch normal prototypes + five-view fusion','train_groups':len(tr),'test_groups':len(te),'external_data':False,'external_specialized_weights':False,'online_model':False,'elapsed_seconds':round(time.time()-t0,2)};(out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'submission.csv','submission.csv')
        for p in sorted(root.glob('*/*/*_mask.png')): z.write(p,p.relative_to(out).as_posix())
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__': main()
