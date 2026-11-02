import argparse, json, os, random, time
import torch
from torch.utils.data import DataLoader
from datasets.visa import VisaDataset
from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--qwen_ckpt', default='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct'); p.add_argument('--imagebind_ckpt', default='../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth'); p.add_argument('--init_ckpt', default='../ckpt/train_visa/pytorch_model.pt'); p.add_argument('--data_root', default='../data/VisA'); p.add_argument('--max_train', type=int, default=1); p.add_argument('--max_tgt_len', type=int, default=128); p.add_argument('--backward', action='store_true'); args=p.parse_args()
    torch.manual_seed(42)
    model=OpenLLAMAPEFTQwenStrictModel(imagebind_ckpt_path=args.imagebind_ckpt,vicuna_ckpt_path=args.qwen_ckpt,init_ckpt_path=args.init_ckpt,max_tgt_len=args.max_tgt_len,stage=1)
    if args.backward:
        model.train()
    else:
        model.eval()
    model.llama_model.eval(); model.visual_encoder.eval()
    ds=VisaDataset(args.data_root); ds.paths=ds.paths[:args.max_train]; ds.x=ds.x[:args.max_train]
    batch=ds.collate([ds[0]])
    torch.cuda.reset_peak_memory_stats(); t=time.time()
    if args.backward:
        loss, acc = model(batch)
    else:
        with torch.no_grad():
            loss, acc = model(batch)
    print(json.dumps({'loss':float(loss),'acc':float(acc),'seconds':time.time()-t,'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9}), flush=True)
    if args.backward:
        loss.backward(); print('BACKWARD_OK', flush=True)
    print('BRIDGE_OK', flush=True)
if __name__=='__main__': main()
