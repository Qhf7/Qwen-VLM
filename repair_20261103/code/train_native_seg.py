"""B 方案：decoder + prompt_learner 联合训练，文字 CE + 像素 Focal/Dice 监督。"""
import argparse, glob, hashlib, json, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import sys
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT")
from qwen3vl_native_bridge import NativeBridge, text_anchors, data, eq1_map_with_cls
from eval_native import score_one, score_one_img
from ImageBind.models.imagebind_model import ModalityType
from visa_pipeline import (PIPELINE_VERSION, load_image_tensor, load_mask_tensor,
                           mask_path_for_image, auroc)

VISA = "/root/autodl-tmp/industrial_vlm/datasets/VisA"
DESC = {"candle":"a candle","cashew":"a cashew","chewinggum":"a chewinggum","fryum":"a fryum",
        "macaroni1":"a macaroni","macaroni2":"a macaroni","pcb1":"a pcb","pcb2":"a pcb",
        "pcb3":"a pcb","pcb4":"a pcb","pipe_fryum":"a pipe fryum","capsules":"a capsule"}
YES = "Yes, there is an anomaly in the image."
NO  = "No, there is no anomaly in the image."

def collect(val_ratio=15):
    tr, va = [], []
    for cls in sorted(os.listdir(VISA)):
        base = os.path.join(VISA, cls, "Data/Images")
        if not os.path.isdir(base): continue
        for lab, sub in ((0,"Normal"),(1,"Anomaly")):
            for p in sorted(glob.glob(os.path.join(base, sub, "*"))):
                h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 100
                (va if h < val_ratio else tr).append((p, cls, lab))
    return tr, va

def load_mask(p, cls, size=224):
    return load_mask_tensor(p, device="cuda", size=size)

def seg_loss(M, gt, gamma=2.0):
    p = M.clamp(1e-5, 1-1e-5)
    fl = -((gt * (1-p).pow(gamma) * p.log()) + ((1-gt) * p.pow(gamma) * (1-p).log()))
    focal = fl.mean()
    inter = (p*gt).sum(); dice = 1 - (2*inter + 1) / (p.sum() + gt.sum() + 1)
    return focal + dice

def eq1_map_train_orig(m, pixel_224, anchors):
    emb = m.encoder({ModalityType.VISION: pixel_224})
    patches = m.decoder([x.float() for x in emb['vision'][1]])
    maps = []
    for p in patches:
        p = p / p.norm(dim=-1, keepdim=True)
        lg = 100.0 * p @ anchors.transpose(-2,-1).float()
        B, L, C = lg.shape; H = int(L**0.5)
        up = F.interpolate(lg.permute(0,2,1).view(B,C,H,H), size=224, mode='bilinear', align_corners=True)
        maps.append(torch.softmax(up, dim=1)[:, 1])
    return torch.stack(maps).mean(0)

def eq1_map_train(m, pixel_224, anchors, want_cls=False):
    emb = m.encoder({ModalityType.VISION: pixel_224})
    cls = emb['vision'][0].float() if want_cls else None
    patches = m.decoder([x.float() for x in emb['vision'][1]])
    maps = []
    for p in patches:
        p = p / p.norm(dim=-1, keepdim=True)
        lg = 100.0 * p @ anchors.transpose(-2,-1).float()
        B, L, C = lg.shape; Hh = int(L**0.5)
        up = F.interpolate(lg.permute(0,2,1).view(B,C,Hh,Hh), size=224, mode='bilinear', align_corners=True)
        maps.append(torch.softmax(up, dim=1)[:, 1])
    M = torch.stack(maps).mean(0)
    return (M, cls) if want_cls else M

def mask_path(p, cls, lab):
    return mask_path_for_image(p, lab)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--out", default="runs/native_seg")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", action="store_true")
    ap.add_argument("--imgtok", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.clean:
        from split_visa import splits
        tr, va, _te = splits()
        print("[split] 官方划分：train 含 official-train 正常图 + test 60%", flush=True)
    else:
        tr, va = collect()
    tr_n=[x for x in tr if x[2]==0]; tr_a=[x for x in tr if x[2]==1]
    va_n=[x for x in va if x[2]==0]; va_a=[x for x in va if x[2]==1]
    print("[data] train %d (%d/%d)  val %d (%d/%d)" % (len(tr),len(tr_n),len(tr_a),len(va),len(va_n),len(va_a)), flush=True)

    m = NativeBridge()
    if args.adapter:
        class _VA(nn.Module):
            def __init__(self, d=4096, r=1024):
                super().__init__()
                self.fc1 = nn.Linear(d, r); self.act = nn.GELU(); self.fc2 = nn.Linear(r, d)
                nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
            def forward(self, x): return self.fc2(self.act(self.fc1(x)))
        m.vis_adapter = _VA().cuda()
        print("[adapter] 视觉适配器已启用（零初始化，起点等价恒等）", flush=True)
    m.decoder = m.decoder.float()
    for p in m.decoder.parameters(): p.requires_grad = True
    anchors = {c: text_anchors(m.encoder, c.replace("_"," "), "cuda").float() for c in set(i[1] for i in tr)}
    params = list(m.prompt_learner.parameters()) + list(m.decoder.parameters())
    if args.adapter: params += list(m.vis_adapter.parameters())
    if args.imgtok:  params += list(m.llama_proj.parameters())
    print("[train] %.2fM params (decoder %d)" % (sum(p.numel() for p in params)/1e6,
          sum(p.numel() for p in m.decoder.parameters())), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    yes_id = m.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = m.processor.tokenizer.encode("No",  add_special_tokens=False)[0]
    log = open(os.path.join(args.out,"train.log"),"w")
    def say(s):
        print(s, flush=True); log.write(s+"\n"); log.flush()

    def checkpoint(step, val_auroc=None):
        ck = {"prompt_learner": m.prompt_learner.state_dict(),
              "image_decoder": m.decoder.state_dict(), "step": step,
              "pipeline_version": PIPELINE_VERSION,
              "training_protocol": ("supervised_custom_60_20_20" if args.clean
                                    else "supervised_legacy_hash_85_15"),
              "seed": args.seed}
        if val_auroc is not None: ck["val_auroc"] = val_auroc
        if args.adapter: ck["vis_adapter"] = m.vis_adapter.state_dict()
        if args.imgtok: ck["llama_proj"] = m.llama_proj.state_dict()
        return ck

    def val_auc(n=150):
        m.prompt_learner.eval(); m.decoder.eval(); sc, lb = [], []
        with torch.no_grad():
            for p,cls,lab in random.sample(va_n,n)+random.sample(va_a,n):
                pil = Image.open(p).convert("RGB").resize((224,224), Image.Resampling.BICUBIC)
                x = load_image_tensor(p, device="cuda")
                if args.imgtok:
                    M, ce_ = eq1_map_train(m, x.half(), anchors[cls], want_cls=True)
                    ie = m.llama_proj(ce_)
                else:
                    M = eq1_map_train(m, x.half(), anchors[cls]); ie = None
                t = m.prompt_learner(M.unsqueeze(1))
                q = "This is a photo of %s for anomaly detection. Is there any anomaly in the image?" % DESC.get(cls,"an object")
                sc.append(score_one_img(m, pil, q, t, yes_id, no_id, ie)); lb.append(lab)
        m.prompt_learner.train(); m.decoder.train()
        return auroc(lb, sc)

    best=-1.0
    for step in range(args.steps):
        opt.zero_grad(); tot=0.0; tot_seg=0.0
        for _ in range(args.accum):
            pool = tr_a if random.random()<0.5 else tr_n
            p,cls,lab = random.choice(pool)
            pil = Image.open(p).convert("RGB").resize((224,224), Image.Resampling.BICUBIC)
            x = load_image_tensor(p, device="cuda")
            if args.imgtok:
                M, cls_emb = eq1_map_train(m, x.half(), anchors[cls], want_cls=True)
                img_emb = m.llama_proj(cls_emb)
            else:
                M = eq1_map_train(m, x.half(), anchors[cls]); img_emb = None
            gt = load_mask(mask_path(p,cls,lab), cls)
            t = m.prompt_learner(M.unsqueeze(1))
            q = "This is a photo of %s for anomaly detection. Is there any anomaly in the image?" % DESC.get(cls,"an object")
            b = m.build_batch(pil, q, YES if lab else NO, t, img_emb)
            ce,_ = m.forward(b)
            sl = seg_loss(M, gt)
            loss = ce + args.seg_weight*sl
            (loss/args.accum).backward(); tot += float(ce); tot_seg += float(sl)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if (step+1)%10==0:
            say("[step %d/%d] ce %.4f seg %.4f lr %.2e vram %.2f peak %.2f"
                % (step+1,args.steps,tot/args.accum,tot_seg/args.accum,sched.get_last_lr()[0],
                   torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9))
        if (step+1)%100==0:
            v = val_auc(); say("  [val] AUROC %.4f (300 imgs) at %d" % (v, step+1))
            if v>best:
                best=v
                torch.save(checkpoint(step+1, v), os.path.join(args.out,"best.pt"))
                say("  [save] best AUROC %.4f" % v)
        if (step+1)%500==0:
            torch.save(checkpoint(step+1),
                       os.path.join(args.out,"step_%d.pt"%(step+1)))
    torch.save(checkpoint(args.steps), os.path.join(args.out,"final.pt"))
    say("[done]")

if __name__ == "__main__":
    main()
