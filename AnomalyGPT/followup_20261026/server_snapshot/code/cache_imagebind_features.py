import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from model.ImageBind import imagebind_model, data
from model.ImageBind.models.imagebind_model import ModalityType
from datasets.visa import VisaDataset

PROMPT_NORMAL=['{}','flawless {}','perfect {}','unblemished {}','{} without flaw','{} without defect','{} without damage']
PROMPT_ABNORMAL=['damaged {}','broken {}','{} with flaw','{} with defect','{} with damage']
PROMPT_TEMPLATES=['a photo of a {}.','a photo of the {}.']

def text_features(model, class_name, device):
    groups=[]
    for states in (PROMPT_NORMAL, PROMPT_ABNORMAL):
        texts=[]
        for s in states:
            for template in PROMPT_TEMPLATES:
                texts.append(template.format(s.format(class_name)))
        ids=data.load_and_transform_text(texts, device)
        with torch.no_grad():
            z=model({ModalityType.TEXT: ids})[ModalityType.TEXT][0]
        z=z.reshape(len(states)*len(PROMPT_TEMPLATES),1024).mean(0,keepdim=True)
        z=z/z.norm(dim=-1,keepdim=True)
        groups.append(z.cpu().half())
    return torch.cat(groups,dim=0).unsqueeze(0)  # 1,2,1,1024

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_root',required=True); p.add_argument('--out',required=True); p.add_argument('--imagebind_ckpt',required=True); p.add_argument('--max_sources',type=int,default=32); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    model,_=imagebind_model.imagebind_huge({'device':0}); ck=torch.load(args.imagebind_ckpt,map_location='cpu'); model.load_state_dict(ck,strict=True); del ck
    model=model.to(args.device,dtype=torch.float16).eval()
    ds=VisaDataset(args.data_root)
    n=min(args.max_sources,len(ds)); records=[]; text_cache={}; t0=time.time()
    with torch.no_grad():
        for i in range(n):
            origin, conv_n, synth, conv_a, cls, mask, path=ds[i]
            imgs=torch.stack([origin,synth],0).to(args.device,dtype=torch.float16)
            emb=model({ModalityType.VISION:imgs})[ModalityType.VISION]
            global_f=emb[0].detach().cpu().half(); patch_f=[x.detach().cpu().half() for x in emb[1]]
            if cls not in text_cache: text_cache[cls]=text_features(model,cls,args.device)
            tf=text_cache[cls]
            variants=[(conv_n,torch.zeros_like(mask)),(conv_a,mask)]
            for j,(conv,gt) in enumerate(variants):
                rec={'global_feature':global_f[j], 'patch_features':[x[:,j:j+1,:] for x in patch_f], 'text_features':tf[0], 'mask':gt.float(), 'text':conv, 'class_name':cls, 'source_index':i, 'variant':j, 'source_path':path}
                fn=out/f'{i:05d}_{j}.pt'; torch.save(rec,fn); records.append({'file':fn.name,'source_index':i,'variant':j,'class_name':cls,'source_path':path})
            if (i+1)%4==0: print(f'[cache] {i+1}/{n} elapsed={(time.time()-t0)/60:.1f}m',flush=True)
    json.dump({'max_sources':n,'records':records,'classes':sorted(text_cache),'seed':args.seed},open(out/'manifest.json','w'),indent=2)
    print(f'[done] records={len(records)} seconds={time.time()-t0:.1f}',flush=True)
if __name__=='__main__': main()
