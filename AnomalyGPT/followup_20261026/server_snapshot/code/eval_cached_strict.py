import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from cached_train_strict import CachedDataset
from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--imagebind_ckpt',required=True); p.add_argument('--qwen_ckpt',required=True)
    p.add_argument('--init_ckpt',required=True); p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--max_tgt_len',type=int,default=64); p.add_argument('--save_json',required=True)
    args=p.parse_args()
    torch.manual_seed(42)
    model=OpenLLAMAPEFTQwenStrictModel(imagebind_ckpt_path=args.imagebind_ckpt,vicuna_ckpt_path=args.qwen_ckpt,init_ckpt_path=args.init_ckpt,max_tgt_len=args.max_tgt_len,stage=1)
    state=torch.load(args.checkpoint,map_location='cpu')
    miss,unexpected=model.load_state_dict(state,strict=False)
    print('[load] missing',len(miss),'unexpected',len(unexpected),flush=True)
    model.pixel_weight=1.0
    model.eval(); model.llama_model.eval(); model.visual_encoder.eval()
    ds=CachedDataset(args.cache)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=ds.collate)
    all_scores=[]; all_gt=[]; all_au=[]; total_loss=0.; total_acc_num=0.; total_acc_den=0.; n=0; t0=time.time()
    for batch in loader:
        # language/combined loss and token accuracy
        loss,acc=model.forward_cached(batch)
        b=len(batch['texts']); total_loss += float(loss)*b; n += b
        # recompute valid token counts for weighted accuracy from the same batch
        ids,tgt,attn = __import__('model.openllama_qwen3vl_strict',fromlist=['process_batch_instance']).process_batch_instance(model.llama_tokenizer,batch['texts'],model.max_tgt_len)
        # use the already computed forward logits to avoid a second LM pass is not exposed;
        # report mean per-batch accuracy as a transparent diagnostic.
        total_acc_num += float(acc)*b; total_acc_den += b
        globals_=batch['global_features'].to(model.device,dtype=torch.bfloat16)
        patch_inputs=[x.to(model.device,dtype=torch.bfloat16) for x in batch['patch_features']]
        patch_tokens=model.image_decoder(patch_inputs)
        feats=batch['text_features'].to(model.device,dtype=torch.bfloat16)
        gt=batch['masks'].float().numpy()
        for layer in patch_tokens:
            z=F.normalize(layer,dim=-1)
            amap=100.0*(z @ feats.transpose(-2,-1))
            B,L,C=amap.shape; H=int(np.sqrt(L))
            amap=F.interpolate(amap.permute(0,2,1).reshape(B,2,H,H),size=224,mode='bilinear',align_corners=True)
            amap=F.softmax(amap,dim=1)[:,1]
            all_scores.append(amap.detach().float().cpu().numpy())
            all_gt.append(gt[:,0])
    scores=np.concatenate(all_scores,axis=0) # layers are concatenated; gt repeats below
    gt=np.concatenate(all_gt,axis=0)
    # each layer contributes one score block, so repeat GT to match score blocks
    if scores.shape[0] != gt.shape[0]:
        rep=scores.shape[0]//gt.shape[0]; gt=np.repeat(gt,rep,axis=0)
    y=(gt>0.3).astype(np.uint8).reshape(-1); s=scores.reshape(-1)
    out={'num_variants':len(ds),'mean_combined_loss':total_loss/max(1,n),'mean_batch_token_acc':total_acc_num/max(1,total_acc_den),'pixel_auroc':float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None,'pixel_ap':float(average_precision_score(y,s)) if y.sum()>0 else None,'seconds':time.time()-t0,'cache':args.cache,'checkpoint':args.checkpoint}
    os.makedirs(os.path.dirname(args.save_json),exist_ok=True); json.dump(out,open(args.save_json,'w'),indent=2)
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
