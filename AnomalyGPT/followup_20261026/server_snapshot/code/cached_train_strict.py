import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel, process_batch_instance
from utils.loss import FocalLoss, BinaryDiceLoss
import torch.nn.functional as F

class CachedDataset(Dataset):
    def __init__(self, root, variant_filter='all'):
        self.root=Path(root); self.files=sorted(self.root.glob('*.pt'))
        if variant_filter in ('normal','abnormal'):
            v='0' if variant_filter=='normal' else '1'
            self.files=[f for f in self.files if f.stem.endswith('_'+v)]
    def __len__(self): return len(self.files)
    def __getitem__(self,i): return torch.load(self.files[i],map_location='cpu')
    @staticmethod
    def collate(batch):
        return {'global_features':torch.stack([x['global_feature'] for x in batch]),'patch_features':[torch.cat([x['patch_features'][l] for x in batch],dim=1) for l in range(4)],'text_features':torch.stack([x['text_features'] for x in batch],dim=0),'masks':torch.stack([x['mask'] for x in batch]),'texts':[x['text'] for x in batch],'class_names':[x['class_name'] for x in batch]}

def cached_forward(self, inputs):
    dev=self.device; dtype=torch.bfloat16
    globals_=inputs['global_features'].to(dev,dtype=dtype)
    img_embeds=self.llama_proj(globals_).unsqueeze(1)
    patch_inputs=[x.to(dev,dtype=dtype) for x in inputs['patch_features']]
    patch_tokens=self.image_decoder(patch_inputs)
    feats=inputs['text_features'].to(dev,dtype=dtype)
    maps=[]
    loss_pixel=0.0
    gt=inputs['masks'].to(dev).float(); gt=(gt>0.3).float()
    for p in patch_tokens:
        p=p/p.norm(dim=-1,keepdim=True)
        amap=100.0*(p @ feats.transpose(-2,-1))
        B,L,C=amap.shape; H=int(np.sqrt(L))
        amap=F.interpolate(amap.permute(0,2,1).view(B,2,H,H),size=224,mode='bilinear',align_corners=True)
        amap=F.softmax(amap,dim=1)
        loss_pixel=loss_pixel+self.loss_focal(amap,gt)+self.loss_dice(amap[:,1],gt)
        maps.append(amap[:,1])
    anomaly=torch.stack(maps).mean(0).unsqueeze(1)
    soft=self.prompt_learner(anomaly)
    ids,tgt,attn=process_batch_instance(self.llama_tokenizer,inputs['texts'],self.max_tgt_len)
    emb,tgt,attn=self.prompt_wrap(img_embeds,ids,tgt,attn,soft)
    out=self._forward_lm(emb,attn,tgt); labels=tgt[:,2:]; pred=out.logits.argmax(-1)[:,1:-1]
    valid=labels.ne(-100); acc=((pred==labels)&valid).sum().item()/max(1,valid.sum().item())
    return out.loss+self.pixel_weight*loss_pixel,acc

OpenLLAMAPEFTQwenStrictModel.forward_cached=cached_forward

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--variant_filter',choices=['all','normal','abnormal'],default='all'); p.add_argument('--pixel_weight',type=float,default=1.0); p.add_argument('--imagebind_ckpt',required=True); p.add_argument('--qwen_ckpt',required=True); p.add_argument('--init_ckpt',required=True); p.add_argument('--save_path',required=True); p.add_argument('--max_steps',type=int,default=16); p.add_argument('--batch_size',type=int,default=2); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--max_tgt_len',type=int,default=64); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model=OpenLLAMAPEFTQwenStrictModel(imagebind_ckpt_path=args.imagebind_ckpt,vicuna_ckpt_path=args.qwen_ckpt,init_ckpt_path=args.init_ckpt,max_tgt_len=args.max_tgt_len,stage=1)
    model.train(); model.llama_model.eval(); model.visual_encoder.eval()
    for p0 in model.parameters(): p0.requires_grad=False
    for mod in (model.image_decoder,model.prompt_learner,model.llama_proj):
        for p0 in mod.parameters(): p0.requires_grad=True
    trainable=[p0 for p0 in model.parameters() if p0.requires_grad]
    ds=CachedDataset(args.cache,args.variant_filter); model.pixel_weight=args.pixel_weight; loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=ds.collate)
    opt=torch.optim.AdamW(trainable,lr=args.lr,betas=(0.9,0.95),weight_decay=1e-3); os.makedirs(args.save_path,exist_ok=True); t0=time.time(); step=0
    while step<args.max_steps:
        for batch in loader:
            loss,acc=model.forward_cached(batch); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); step+=1
            print(f'[step] {step}/{args.max_steps} loss={float(loss):.5f} acc={acc:.4f} sec={(time.time()-t0):.1f} vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB',flush=True)
            if step>=args.max_steps: break
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(('image_decoder.','prompt_learner.','llama_proj.'))}; torch.save(state,Path(args.save_path)/'pytorch_model.pt'); json.dump({'args':vars(args),'steps':step,'num_files':len(ds)},open(Path(args.save_path)/'run.json','w'),indent=2)
if __name__=='__main__': main()
