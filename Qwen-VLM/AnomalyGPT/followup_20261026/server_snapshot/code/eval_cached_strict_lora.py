import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from peft import PeftModel
from cached_train_strict import CachedDataset
from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--bridge_checkpoint',required=True); p.add_argument('--lora_adapter',required=True); p.add_argument('--imagebind_ckpt',required=True); p.add_argument('--qwen_ckpt',required=True); p.add_argument('--init_ckpt',required=True); p.add_argument('--batch_size',type=int,default=2); p.add_argument('--max_tgt_len',type=int,default=64); p.add_argument('--save_json',required=True); args=p.parse_args()
    torch.manual_seed(42)
    model=OpenLLAMAPEFTQwenStrictModel(imagebind_ckpt_path=args.imagebind_ckpt,vicuna_ckpt_path=args.qwen_ckpt,init_ckpt_path=args.init_ckpt,max_tgt_len=args.max_tgt_len,stage=1)
    state=torch.load(args.bridge_checkpoint,map_location='cpu'); miss,unexpected=model.load_state_dict(state,strict=False); print('[bridge_load] missing',len(miss),'unexpected',len(unexpected),flush=True)
    if not hasattr(model.lm,'prepare_inputs_for_generation'):
        model.lm.prepare_inputs_for_generation=lambda input_ids=None,past_key_values=None,**kwargs: kwargs
    model.lm=PeftModel.from_pretrained(model.lm,args.lora_adapter,is_trainable=False)
    model.pixel_weight=1.0; model.eval(); model.llama_model.eval(); model.visual_encoder.eval()
    ds=CachedDataset(args.cache); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=ds.collate)
    all_scores=[]; all_gt=[]; total_loss=0.; total_acc=0.; n=0; t0=time.time()
    with torch.no_grad():
      for batch in loader:
        loss,acc=model.forward_cached(batch); b=len(batch['texts']); total_loss += float(loss)*b; total_acc += float(acc)*b; n += b
        patch_inputs=[x.to(model.device,dtype=torch.bfloat16) for x in batch['patch_features']]; patch_tokens=model.image_decoder(patch_inputs); feats=batch['text_features'].to(model.device,dtype=torch.bfloat16); gt=batch['masks'].float().numpy()
        for layer in patch_tokens:
            z=F.normalize(layer,dim=-1); amap=100.0*(z @ feats.transpose(-2,-1)); B,L,C=amap.shape; H=int(np.sqrt(L)); amap=F.interpolate(amap.permute(0,2,1).reshape(B,2,H,H),size=224,mode='bilinear',align_corners=True); amap=F.softmax(amap,dim=1)[:,1]; all_scores.append(amap.float().cpu().numpy()); all_gt.append(gt[:,0])
    scores=np.concatenate(all_scores,axis=0); gt=np.concatenate(all_gt,axis=0)
    if scores.shape[0]!=gt.shape[0]: gt=np.repeat(gt,scores.shape[0]//gt.shape[0],axis=0)
    y=(gt>0.3).astype(np.uint8).reshape(-1); s=scores.reshape(-1)
    out={'num_variants':len(ds),'mean_combined_loss':total_loss/max(1,n),'mean_batch_token_acc':total_acc/max(1,n),'pixel_auroc':float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None,'pixel_ap':float(average_precision_score(y,s)) if y.sum()>0 else None,'seconds':time.time()-t0,'cache':args.cache,'bridge_checkpoint':args.bridge_checkpoint,'lora_adapter':args.lora_adapter}
    os.makedirs(os.path.dirname(args.save_json),exist_ok=True); json.dump(out,open(args.save_json,'w'),indent=2); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
