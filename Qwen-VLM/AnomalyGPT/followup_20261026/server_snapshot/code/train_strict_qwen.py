import argparse, json, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.visa import VisaDataset
from model.openllama_qwen3vl_strict import OpenLLAMAPEFTQwenStrictModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qwen_ckpt', default='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct')
    p.add_argument('--imagebind_ckpt', default='../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth')
    p.add_argument('--init_ckpt', default='../ckpt/train_visa/pytorch_model.pt')
    p.add_argument('--data_root', default='../visa_data')
    p.add_argument('--save_path', required=True)
    p.add_argument('--max_train', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max_tgt_len', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def save_trainable(model, path, step, args):
    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.startswith(('image_decoder.', 'prompt_learner.', 'llama_proj.'))}
    torch.save(state, os.path.join(path, 'pytorch_model.pt'))
    with open(os.path.join(path, 'run.json'), 'w') as f:
        json.dump({'step': step, 'args': vars(args), 'trainable_keys': sorted(state)}, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    model = OpenLLAMAPEFTQwenStrictModel(
        imagebind_ckpt_path=args.imagebind_ckpt,
        vicuna_ckpt_path=args.qwen_ckpt,
        init_ckpt_path=args.init_ckpt,
        max_tgt_len=args.max_tgt_len,
        stage=1,
    )
    model.train(); model.llama_model.eval(); model.visual_encoder.eval()
    for p in model.parameters(): p.requires_grad = False
    for module in (model.image_decoder, model.prompt_learner, model.llama_proj):
        for p in module.parameters(): p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('[trainable_num]', sum(p.numel() for p in trainable), flush=True)
    ds = VisaDataset(args.data_root)
    if args.max_train > 0:
        ds.paths = ds.paths[:args.max_train]; ds.x = ds.x[:args.max_train]
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=ds.collate)
    print('[dataset]', len(ds), flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-3)
    step = 0; micro = 0; t0 = time.time(); opt.zero_grad(set_to_none=True)
    total = args.epochs * len(loader)
    for epoch in range(args.epochs):
        for batch in loader:
            loss, acc = model(batch)
            (loss / args.accum).backward(); micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                print(f'[step] {step} loss={float(loss):.5f} acc={float(acc):.4f} vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB elapsed={(time.time()-t0)/60:.1f}m', flush=True)
                if args.max_steps and step >= args.max_steps:
                    save_trainable(model, args.save_path, step, args); print('[done]', flush=True); return
    save_trainable(model, args.save_path, step, args); print('[done]', flush=True)

if __name__ == '__main__': main()
