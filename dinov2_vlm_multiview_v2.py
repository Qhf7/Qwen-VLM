#!/usr/bin/env python3
"""DINOv2 multi-layer memory bank + local CLIP + multi-view inference.

This is an independent model for the competition.  It does not read any
previous submission score or mask.  Only official Train normal images are
used for prototypes and calibration; pseudo anomalies are generated on the
fly from those images.  Public generic DINOv2 and OpenAI CLIP weights only.
"""
import argparse,csv,json,time,random,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFilter
import clip

SEED=20260813
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def groups(root):
    for c in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='__MACOSX'):
        for s in sorted(p for p in c.iterdir() if p.is_dir()):
            ps=[s/f'{i}.png' for i in range(5)]
            if all(p.exists() for p in ps): yield c.name,s.name,ps

def load_image(p):
    with Image.open(p) as im: return im.convert('RGB')

def dino_model(dev):
    m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=True).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def dino_batch(m,ims,dev):
    # DINOv2 accepts tensors in [0,1], and its internal preprocessing is not
    # applied by the hub model, so use the official ImageNet normalization.
    x=torch.stack([torch.from_numpy(np.asarray(im.resize((518,518)),np.float32).transpose(2,0,1))/255. for im in ims]).to(dev)
    x=F.normalize(x,dim=1) if False else x
    mean=torch.tensor([0.485,0.456,0.406],device=dev)[None,:,None,None]
    std=torch.tensor([0.229,0.224,0.225],device=dev)[None,:,None,None]
    x=(x-mean)/std
    with torch.inference_mode():
        outs=m.get_intermediate_layers(x,n=[3,6,9,11],reshape=False,return_class_token=True)
    # concatenate normalized patch tokens from four depths: B,N,1536.
    patches=[]; cls=[]
    for item in outs:
        if isinstance(item,tuple): p,c=item
        else: p=item; c=item[:,0]; p=item[:,1:]
        patches.append(F.normalize(p,dim=-1)); cls.append(F.normalize(c,dim=-1))
    return torch.cat(cls,dim=-1).cpu().numpy(), torch.cat(patches,dim=-1).cpu().numpy()

def prompts(cat):
    normal=[f'a normal {cat} industrial product',f'a clean intact {cat} manufactured part']
    bad=[f'a defective {cat} industrial product',f'a damaged {cat} manufactured part']
    defects=[f'{cat} with a scratch',f'{cat} with a dent',f'{cat} with a crack',f'{cat} with a stain',f'{cat} with a surface defect']
    return normal,bad,defects

def pseudo(im,kind):
    im=im.copy(); w,h=im.size; mask=Image.new('L',(w,h),0); d=ImageDraw.Draw(mask)
    x=random.randint(int(w*.08),int(w*.75)); y=random.randint(int(h*.08),int(h*.75));
    rw=random.randint(max(4,int(w*.015)),max(8,int(w*.18))); rh=random.randint(max(2,int(h*.006)),max(6,int(h*.07)))
    if kind==0: # thin curved scratch / multi-segment line
        pts=[]
        for k in range(random.randint(3,7)): pts.append((x+k*rw//5,y+random.randint(-rh,rh)))
        ImageDraw.Draw(im).line(pts,fill=(20,20,20),width=random.randint(1,4)); d.line(pts,fill=255,width=random.randint(2,6))
    elif kind==1: # stain/texture block
        col=tuple(random.randint(0,255) for _ in range(3)); ImageDraw.Draw(im).rectangle((x,y,x+rw,y+rh),fill=col); d.rectangle((x,y,x+rw,y+rh),fill=255)
    elif kind==2: # dent-like local contrast
        crop=im.crop((x,y,min(w,x+rw),min(h,y+rh))).filter(ImageFilter.GaussianBlur(5)); im.paste(crop,(x,y)); d.ellipse((x,y,min(w,x+rw),min(h,y+rh)),fill=255)
    elif kind==3: # broken/erased patch
        ImageDraw.Draw(im).rectangle((x,y,x+rw,y+rh),fill=(128,128,128)); d.rectangle((x,y,x+rw,y+rh),fill=255)
    else: # noise-like colored small patch
        pix=np.asarray(im).copy(); yy=slice(y,min(h,y+rh)); xx=slice(x,min(w,x+rw)); noise=np.random.default_rng().normal(0,45,pix[yy,xx].shape); pix[yy,xx]=np.clip(pix[yy,xx].astype(np.float32)+noise,0,255).astype(np.uint8); im=Image.fromarray(pix); d.rectangle((x,y,min(w,x+rw),min(h,y+rh)),fill=255)
    return im,mask

def robust_cal(vals):
    a=np.asarray(vals,np.float32); q=np.percentile(a,[5,50,95]); return q

def train_pseudo_head(dm,vt,train_items,dev,limit=240):
    """Train a small patch classifier from official-normal images only.

    Every positive example is generated on the fly by one of five deterministic
    pseudo-defect families.  The DINOv2 backbone stays frozen.
    """
    xs=[]; ys=[]
    chosen=train_items[:limit]
    for j,(_,_,ps) in enumerate(chosen):
        im=load_image(ps[0]); aug,mask=pseudo(im,j%5)
        _,pn=dino_batch(dm,[im,aug],dev); pn=pn.reshape(2,1369,-1)
        mm=np.asarray(mask.resize((37,37),Image.Resampling.BILINEAR),np.float32)/255.
        xs.append(pn[0]); ys.append(np.zeros(1369,np.float32)); xs.append(pn[1]); ys.append((mm>0.12).astype(np.float32).reshape(-1))
        if j and j%80==0: print('pseudo features',j,flush=True)
    x=torch.from_numpy(np.concatenate(xs)).float().to(dev); y=torch.from_numpy(np.concatenate(ys)).float().to(dev)
    x=F.normalize(x,dim=1); head=nn.Linear(x.shape[1],1).to(dev)
    pos=max(float(y.sum()),1.0); neg=max(float((1-y).sum()),1.0); opt=torch.optim.AdamW(head.parameters(),lr=3e-3,weight_decay=1e-4)
    pw=torch.tensor([min(20.0,neg/pos)],device=dev); loss_fn=nn.BCEWithLogitsLoss(pos_weight=pw)
    head.train()
    for ep in range(8):
        order=torch.randperm(x.shape[0],device=dev)
        for st in range(0,x.shape[0],8192):
            ix=order[st:st+8192]; loss=loss_fn(head(x[ix]).squeeze(1),y[ix]); opt.zero_grad(); loss.backward(); opt.step()
        if ep in (0,7): print('pseudo head epoch',ep+1,'loss',float(loss),flush=True)
    head.eval()
    return head

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--clip-weight',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--zip',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=16); a=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t0=time.time(); print('device',dev,flush=True)
    dm=dino_model(dev); cm,cp=clip.load(str(a.clip_weight),device=str(dev),jit=False); cm.eval()
    tr=list(groups(a.data_root/'Train')); te=list(groups(a.data_root/'Test_A')); cats=sorted({x[0] for x in tr+te})
    # category/view memory bank: keep several normal prototypes using fixed
    # deterministic bins, instead of collapsing every patch to one centroid.
    banks={}; global_cls={}; normal_patch_scores=[]; normal_clip_scores=[]
    for st in range(0,len(tr),a.batch_size):
        ch=tr[st:st+a.batch_size]; ims=[load_image(p) for _,_,q in ch for p in q]; cls,pat=dino_batch(dm,ims,dev); cls=cls.reshape(len(ch),5,-1); pat=pat.reshape(len(ch),5,1369,-1)
        for i,(cat,s,_) in enumerate(ch):
            for v in range(5): banks.setdefault((cat,v),[]).append(pat[i,v]); global_cls.setdefault((cat,v),[]).append(cls[i,v])
        if st%160==0: print('memory train',min(st+a.batch_size,len(tr)),flush=True)
    prototypes={}; cls_mu={}
    for k,arrs in banks.items():
        z=np.concatenate(arrs,axis=0); # deterministic stratified prototypes
        ids=np.linspace(0,len(z)-1,min(24,len(z))).astype(int); prototypes[k]=z[ids]
        cls_mu[k]=np.mean(np.stack(global_cls[k]),axis=0); cls_mu[k]/=max(np.linalg.norm(cls_mu[k]),1e-6)
    pseudo_head=train_pseudo_head(dm,None,tr,dev,limit=min(240,len(tr)))
    # local CLIP texts; category-specific prompts are evaluated on 2x2 local crops.
    texts={}
    with torch.inference_mode():
        for cat in cats:
            n,b,d=prompts(cat); z=cm.encode_text(clip.tokenize(n+b+d).to(dev)); z=z/z.norm(dim=-1,keepdim=True); texts[cat]=(z[:len(n)].mean(0),z[len(n):len(n)+len(b)].mean(0),z[len(n)+len(b):])
    # Train-only calibration distributions: normal DINO distances and local CLIP scores.
    for cat,v in list(cls_mu)[:]: pass
    for cat,v in list(cls_mu):
        pass
    for st in range(0,min(len(tr),240),a.batch_size):
        ch=tr[st:st+a.batch_size]; ims=[load_image(q[0]) for _,_,q in ch]; ci,pa=dino_batch(dm,ims,dev)
        for i,(cat,_,_) in enumerate(ch):
            vv=0; z=pa[i].reshape(1369,-1); mu=prototypes[(cat,vv)]; dist=1-(z@mu.T).max(1); normal_patch_scores.append(float(np.percentile(dist,99.5)))
        crops=[]
        for im in ims:
            im=im.resize((448,448)); crops.extend([im.crop((x,y,x+224,y+224)) for y in (0,224) for x in (0,224)])
        with torch.inference_mode(): zz=cm.encode_image(torch.stack([cp(x) for x in crops]).to(dev)); zz=zz/zz.norm(dim=-1,keepdim=True)
        for i,(cat,_,_) in enumerate(ch):
            n,b,_=texts[cat]; normal_clip_scores.append(float((zz[i*4:(i+1)*4]@(b-n)).mean().cpu()))
    pq=robust_cal(normal_patch_scores); cq=robust_cal(normal_clip_scores)
    # Calibrate the pseudo head on held-in normal images; no test data/labels.
    pseudo_normal=[]
    for _,_,ps in tr[:80]:
        _,pn=dino_batch(dm,[load_image(ps[0])],dev)
        with torch.inference_mode(): pseudo_normal.extend(torch.sigmoid(pseudo_head(F.normalize(torch.from_numpy(pn[0]).float().to(dev),dim=1))).squeeze(1).cpu().numpy().tolist())
    phq=robust_cal(pseudo_normal)
    root=a.output/'predicted_masks'; a.output.mkdir(parents=True,exist_ok=True); rows=[]
    for st in range(0,len(te),a.batch_size):
        ch=te[st:st+a.batch_size]; ims=[load_image(p) for _,_,q in ch for p in q]; ci,pa=dino_batch(dm,ims,dev); ci=ci.reshape(len(ch),5,-1); pa=pa.reshape(len(ch),5,1369,-1)
        # local CLIP grid for all views
        crops=[]
        for _,_,q in ch:
            for p in q:
                im=load_image(p).resize((448,448)); crops.extend([im.crop((x,y,x+224,y+224)) for y in (0,224) for x in (0,224)])
        with torch.inference_mode(): zz=cm.encode_image(torch.stack([cp(x) for x in crops]).to(dev)); zz=zz/zz.norm(dim=-1,keepdim=True)
        for i,(cat,s,_) in enumerate(ch):
            n,b,_=texts[cat]; view_scores=[]; maps=[]
            group_ci=ci[i]; group_ref=np.median(group_ci,axis=0); group_ref/=max(np.linalg.norm(group_ref),1e-6)
            for v in range(5):
                z=pa[i,v]; mu=prototypes[(cat,v)]; dist=1-(z@mu.T).max(1); dino_raw=float(np.percentile(dist,99.5));
                dino_s=float(np.clip((dino_raw-pq[1])/max(pq[2]-pq[1],1e-6),0,1)); hm=np.clip((dist-np.percentile(dist,5))/max(np.percentile(dist,99.5)-np.percentile(dist,5),1e-6),0,1).reshape(37,37)
                with torch.inference_mode():
                    ph=torch.sigmoid(pseudo_head(F.normalize(torch.from_numpy(z).float().to(dev),dim=1))).squeeze(1).cpu().numpy().reshape(37,37)
                ph=(ph-phq[1])/max(phq[2]-phq[1],1e-6); ph=np.clip(ph,0,1)
                local=(zz[(i*5+v)*4:(i*5+v+1)*4]@(b-n)).mean().item(); local_s=float(np.clip((local-cq[1])/max(cq[2]-cq[1],1e-6),0,1))
                view_cons=float(1-(ci[i,v]@group_ref)); view_cons=float(np.clip(view_cons/0.25,0,1))
                # Local VLM gates only pixels in cells with semantic evidence;
                # no post-hoc normalization of the final mask.
                grid=np.array([float(torch.sigmoid(8*(x-(n@b).item())).cpu()) for x in (zz[(i*5+v)*4:(i*5+v+1)*4]@texts[cat][2].mean(0))]).mean(); gate=0.65+0.35*grid
                hm=np.clip((0.62*hm+0.23*ph)*(0.78+0.22*gate),0,1); maps.append(hm); view_scores.append(0.50*dino_s+0.22*float(np.percentile(ph,99.5))+0.18*local_s+0.10*view_cons)
                h=torch.from_numpy(hm).float()[None,None]; h=F.interpolate(h,size=(448,448),mode='bilinear',align_corners=False)[0,0].numpy(); p=root/cat/s/f'{v}_mask.png'; p.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.uint8(np.round(h*255)),'L').save(p,'PNG',optimize=True)
            # dynamic confidence fusion: max catches one-view defects, mean
            # stabilizes normal groups; relative model reliability uses scores.
            final=float(np.clip(0.72*max(view_scores)+0.28*np.mean(view_scores),0,1)); rows.append({'group_folder':f'{cat}/{s}','anomaly_score':f'{final:.8f}'})
        print('test',min(st+a.batch_size,len(te)),flush=True)
    with (a.output/'submission.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['group_folder','anomaly_score']); w.writeheader(); w.writerows(rows)
    meta={'method':'DINOv2 multi-layer patch memory bank + trained multi-type pseudo-anomaly patch head + local CLIP grid alignment + five-view consistency + Train-only calibration','dino_source':'facebookresearch/dinov2 dinov2_vits14 official generic weight','pseudo_types':['scratch','stain','dent','erase','noise'],'external_data':False,'external_specialized_weights':False,'online_model':False,'train_groups':len(tr),'test_groups':len(te),'elapsed_seconds':round(time.time()-t0,2)}; (a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(a.output/'submission.csv','submission.csv')
        for p in sorted(root.glob('*/*/*_mask.png')): z.write(p,p.relative_to(a.output).as_posix())
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__': main()
