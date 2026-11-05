"""Vicuna-7B 对照臂：与 Qwen 版共享划分/decoder/prompt_learner/损失，只换视觉入口。"""
import argparse, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT/code/code/model")
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT")
from ImageBind import imagebind_model, data
from ImageBind.models.imagebind_model import ModalityType
from AnomalyGPT_models import LinearLayer, PromptLearner
from transformers import LlamaForCausalLM, LlamaTokenizer
from visa_pipeline import (PIPELINE_VERSION, load_image_tensor, load_mask_tensor,
                           mask_path_for_image, auroc)

REPO = "/root/autodl-tmp/AnomalyGPT"
IB   = os.path.join(REPO, "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth")
VIC  = os.path.join(REPO, "pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1")   # v0 只剩配置，无权重
CKPT = os.path.join(REPO, "ckpt/train_visa/pytorch_model.pt")
data.BPE_PATH = os.path.join(REPO, "code/code/model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
H = 4096
PROMPT_NORMAL   = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
PROMPT_ABNORMAL = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
TEMPLATES       = ['a photo of a {}.', 'a photo of the {}.']

def _module_state(checkpoint, key):
    """Read nested experiment checkpoints or flat released module prefixes."""
    if key in checkpoint and isinstance(checkpoint[key], dict):
        return checkpoint[key]
    prefix = key + "."
    state = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    if not state:
        raise KeyError("Checkpoint has no %s parameters" % key)
    return state

def _load_decoder_init(module, path, decoder_kind="linear"):
    """Load a decoder only when its architecture and pipeline match."""
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Decoder checkpoint must be a state dictionary: %s" % path)
    if decoder_kind == "upsample":
        if checkpoint.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("Upsample decoder requires a checkpoint from pipeline %s" % PIPELINE_VERSION)
        if not isinstance(checkpoint.get("dec"), dict):
            raise ValueError("Upsample decoder checkpoint must contain a 'dec' state dictionary")
        state = checkpoint["dec"]
    elif "image_decoder" in checkpoint or any(k.startswith("image_decoder.") for k in checkpoint):
        state = _module_state(checkpoint, "image_decoder")
    elif "decoder" in checkpoint or any(k.startswith("decoder.") for k in checkpoint):
        state = _module_state(checkpoint, "decoder")
    elif isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = [k for k in set(expected) & set(state)
              if not torch.is_tensor(state[k]) or state[k].shape != expected[k].shape]
    if missing or unexpected or shapes:
        raise ValueError("Decoder checkpoint is incompatible with %s decoder: "
                         "missing=%s unexpected=%s shape_mismatch=%s"
                         % (decoder_kind, missing, unexpected, sorted(shapes)))
    module.load_state_dict(state, strict=True)

def fixed_validation_items(items, limit=300):
    """Choose one reproducible stratified subset without changing training RNG."""
    if limit < 2:
        raise ValueError("Validation limit must be at least 2")
    normal = sorted((x for x in items if not x[2]), key=lambda x: x[0])
    anomaly = sorted((x for x in items if x[2]), key=lambda x: x[0])
    if not normal or not anomaly:
        raise ValueError("Validation requires both normal and anomalous images")
    rng = random.Random(42)
    n_normal = min(len(normal), limit // 2)
    n_anomaly = min(len(anomaly), limit - n_normal)
    n_normal = min(len(normal), limit - n_anomaly)
    return rng.sample(normal, n_normal) + rng.sample(anomaly, n_anomaly)

def training_scheduler(optimizer, steps, max_lr):
    # Preserve the full-run schedule. Tiny smoke runs have no OneCycle warmup
    # segment and use cosine instead, which is defined for every positive step.
    if steps < 40:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=max_lr * 1e-3)
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=steps, pct_start=0.05)

def text_anchors(enc, cls, device):
    def build(states):
        sents = [t.format(s.format(cls)) for s in states for t in TEMPLATES]
        toks = data.load_and_transform_text(sents, device)
        with torch.no_grad():
            e = enc({ModalityType.TEXT: toks})[ModalityType.TEXT][0]
        e = e.reshape(1, len(sents), 1024).mean(1, keepdim=True)
        return e / e.norm(dim=-1, keepdim=True)
    return torch.cat([build(PROMPT_NORMAL), build(PROMPT_ABNORMAL)], dim=1)

class VicunaBridge(nn.Module):
    def __init__(self, decoder_init=None, decoder_kind="linear", freeze_decoder=False):
        super().__init__()
        if decoder_kind not in {"linear", "upsample"}:
            raise ValueError("Unknown decoder kind: %s" % decoder_kind)
        if decoder_kind == "upsample" and not decoder_init:
            raise ValueError("Upsample decoder requires explicit --decoder-init")
        self.decoder_kind = decoder_kind
        self.freeze_decoder = freeze_decoder
        self.tok = LlamaTokenizer(vocab_file=os.path.join(VIC, "tokenizer.model"))
        self.lm = LlamaForCausalLM.from_pretrained(VIC, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda().eval()
        for p in self.lm.parameters(): p.requires_grad = False
        self.encoder, _ = imagebind_model.imagebind_huge({"layers": [7,15,23,31]})
        self.encoder.load_state_dict(torch.load(IB, map_location="cpu"), strict=True)
        self.encoder.half().cuda().eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        if decoder_kind == "upsample":
            from dec_big import BigDec, patch_features
            self.decoder = BigDec().float().cuda()
            self._patch_features = patch_features
        else:
            self.decoder = LinearLayer(1280, 1024, 4).float().cuda()
        self.prompt_learner = PromptLearner(1, H).cuda()
        self.llama_proj = nn.Linear(1024, H).cuda()
        sd = torch.load(CKPT, map_location="cpu")
        if decoder_kind == "linear":
            self.decoder.load_state_dict(_module_state(sd, "image_decoder"), strict=True)
        self.prompt_learner.load_state_dict(_module_state(sd, "prompt_learner"), strict=True)
        if "llama_proj" in sd or any(k.startswith("llama_proj.") for k in sd):
            self.llama_proj.load_state_dict(_module_state(sd, "llama_proj"), strict=True)
        if decoder_init:
            _load_decoder_init(self.decoder, decoder_init, decoder_kind)
        for p in self.decoder.parameters(): p.requires_grad = not freeze_decoder
        if freeze_decoder: self.decoder.eval()
        self.bos = self.tok.bos_token_id

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.lm.eval()
        if self.freeze_decoder: self.decoder.eval()
        return self

    def ids(self, s):
        return self.tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

    def anomaly_map(self, pixel, anchors):
        with torch.no_grad():
            emb = self.encoder({ModalityType.VISION: pixel})
        if self.decoder_kind == "upsample":
            features = self._patch_features(emb['vision'][1], pixel.shape[0])
            M = self.decoder(features).sigmoid().squeeze(1)
            if M.shape != (pixel.shape[0], 224, 224):
                raise ValueError("Unexpected upsample map shape: %s" % (tuple(M.shape),))
            return M, emb['vision'][0].float()
        patches = self.decoder([x.float() for x in emb['vision'][1]])
        maps = []
        for p in patches:
            p = p / p.norm(dim=-1, keepdim=True)
            lg = 100.0 * p @ anchors.transpose(-2,-1).float()
            B, L, C = lg.shape; Hs = int(L**0.5)
            up = F.interpolate(lg.permute(0,2,1).view(B,C,Hs,Hs), size=224, mode='bilinear', align_corners=True)
            maps.append(torch.softmax(up, dim=1)[:, 1])
        return torch.stack(maps).mean(0), emb['vision'][0].float()

    def build(self, pixel, cls_emb, tok18, desc, question, answer):
        e_before = self.lm.get_input_embeddings()(self.ids("### Human: <Img>"))
        e_mid    = self.lm.get_input_embeddings()(self.ids("</Img> "))
        e_text   = self.lm.get_input_embeddings()(self.ids(desc + " " + question + "\n### Assistant: "))
        ans_ids  = self.ids(answer + self.tok.eos_token)
        e_ans    = self.lm.get_input_embeddings()(ans_ids)
        bos      = self.lm.get_input_embeddings()(torch.tensor([[self.bos]], device="cuda"))
        img      = self.llama_proj(cls_emb).unsqueeze(1).half()
        emb = torch.cat([bos, e_before, img, e_mid, tok18.half(), e_text, e_ans], dim=1)
        n_prefix = emb.shape[1] - ans_ids.shape[1]
        labels = torch.cat([torch.full((1, n_prefix), -100, dtype=torch.long, device="cuda"), ans_ids], dim=1)
        return emb, labels

    def forward(self, emb, labels):
        return self.lm(inputs_embeds=emb, labels=labels).loss

def seg_loss(M, gt, gamma=2.0):
    p = M.clamp(1e-5, 1-1e-5)
    fl = -((gt*(1-p).pow(gamma)*p.log()) + ((1-gt)*p.pow(gamma)*(1-p).log()))
    return fl.mean() + (1 - (2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1))

def score_one(m, pixel, cls_emb, tok18, desc, question, yes_id, no_id):
    e_before = m.lm.get_input_embeddings()(m.ids("### Human: <Img>"))
    e_mid    = m.lm.get_input_embeddings()(m.ids("</Img> "))
    e_text   = m.lm.get_input_embeddings()(m.ids(desc + " " + question + "\n### Assistant: "))
    bos      = m.lm.get_input_embeddings()(torch.tensor([[m.bos]], device="cuda"))
    img      = m.llama_proj(cls_emb).unsqueeze(1).half()
    emb = torch.cat([bos, e_before, img, e_mid, tok18.half(), e_text], dim=1)
    with torch.no_grad():
        lg = m.lm(inputs_embeds=emb).logits[0, -1].float()
    return float(lg[yes_id] - lg[no_id])

DESC = {"candle":"a candle","cashew":"a cashew","chewinggum":"a chewinggum","fryum":"a fryum",
        "macaroni1":"a macaroni","macaroni2":"a macaroni","pcb1":"a pcb","pcb2":"a pcb",
        "pcb3":"a pcb","pcb4":"a pcb","pipe_fryum":"a pipe fryum","capsules":"a capsule"}
YES = "Yes, there is an anomaly in the image."
NO  = "No, there is no anomaly in the image."

def load_mask(p, size=224):
    return load_mask_tensor(p, device="cuda", size=size)

def mask_path(p, label=1):
    return mask_path_for_image(p, label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/vicuna_clean")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--decoder-kind", choices=["linear", "upsample"], default="linear")
    ap.add_argument("--freeze-decoder", action="store_true")
    ap.add_argument("--decoder-init", default="",
                    help="Decoder checkpoint; required for upsample and checked against its architecture/pipeline")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-limit", type=int, default=300)
    args = ap.parse_args()
    if args.steps < 1 or args.accum < 1 or args.val_every < 1 or args.val_limit < 2:
        raise ValueError("Positive steps, accumulation, validation interval and validation limit >=2 required")
    evaluation_state = None
    if args.eval:
        if not args.ckpt:
            raise ValueError("--eval requires --ckpt")
        evaluation_state = torch.load(args.ckpt, map_location="cpu")
        args.decoder_kind = evaluation_state.get("decoder_kind", "linear")
        args.freeze_decoder = evaluation_state.get("freeze_decoder", args.freeze_decoder)
        if args.decoder_kind == "upsample":
            args.decoder_init = args.ckpt
    elif args.decoder_kind == "upsample" and not args.decoder_init:
        raise ValueError("--decoder-kind upsample requires explicit --decoder-init")
    if not args.eval and any(os.path.exists(os.path.join(args.out, name)) for name in ("best.pt", "final.pt")):
        raise FileExistsError("Choose a fresh output directory; existing training checkpoints are preserved")
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); m = VicunaBridge(decoder_init=args.decoder_init or None,
                                     decoder_kind=args.decoder_kind,
                                     freeze_decoder=args.freeze_decoder)
    print("[load] %.0fs vram %.2fGB" % (time.time()-t0, torch.cuda.memory_allocated()/1e9), flush=True)
    from split_visa import splits
    tr, va, te = splits()
    anchors = {c: text_anchors(m.encoder, c.replace("_"," "), "cuda") for c in set(x[1] for x in tr)}
    yes_id = m.tok.encode("Yes", add_special_tokens=False)[0]
    no_id  = m.tok.encode("No",  add_special_tokens=False)[0]

    if args.eval:
        sd = evaluation_state
        m.prompt_learner.load_state_dict(sd["prompt_learner"], strict=True)
        if m.decoder_kind == "linear" and "image_decoder" in sd:
            m.decoder.load_state_dict(sd["image_decoder"], strict=True)
        if "llama_proj" in sd: m.llama_proj.load_state_dict(sd["llama_proj"], strict=True)
        m.eval()
        res = {}
        g = torch.Generator().manual_seed(0)
        for cond in ["real","zero","random"]:
            sc, lb = [], []
            for p,cls,lab in te:
                x = load_image_tensor(p, device="cuda")
                with torch.no_grad():
                    M, ce_ = m.anomaly_map(x.half(), anchors[cls])
                    t18 = m.prompt_learner(M.unsqueeze(1))
                if cond=="zero": t18 = torch.zeros_like(t18)
                elif cond=="random":
                    r = torch.randn(t18.shape, generator=g).to(t18.device, t18.dtype)
                    t18 = r / r.norm(dim=-1,keepdim=True) * t18.norm(dim=-1,keepdim=True)
                sc.append(score_one(m, x, ce_, t18, DESC.get(cls,"an object"),
                                    "Is there any anomaly in the image?", yes_id, no_id))
                lb.append(lab)
            auc=auroc(lb, sc)
            pred=[1 if v>0 else 0 for v in sc]
            tp=sum(1 for a,b in zip(lb,pred) if a==1 and b==1); fp=sum(1 for a,b in zip(lb,pred) if a==0 and b==1)
            tn=sum(1 for a,b in zip(lb,pred) if a==0 and b==0); fn=sum(1 for a,b in zip(lb,pred) if a==1 and b==0)
            f1=(2*tp/max(2*tp+fp+fn,1)+2*tn/max(2*tn+fn+fp,1))/2
            res[cond]={"auroc":auc,"acc":(tp+tn)/len(lb),"f1_macro":f1,"recall":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1)}
            print("  -> %s: AUROC %.4f acc %.4f F1 %.4f" % (cond,auc,(tp+tn)/len(lb),f1), flush=True)
        import json; json.dump(res, open(os.path.join(args.out,"metrics.json"),"w"), indent=2)
        return

    tr_n=[x for x in tr if x[2]==0]; tr_a=[x for x in tr if x[2]==1]
    validation_items = fixed_validation_items(va, args.val_limit)
    params = [p for module in (m.prompt_learner, m.decoder, m.llama_proj)
              for p in module.parameters() if p.requires_grad]
    print("[train] %.2fM  train %d (%d/%d)" % (sum(p.numel() for p in params)/1e6, len(tr), len(tr_n), len(tr_a)), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = training_scheduler(opt, args.steps, args.lr)
    log=open(os.path.join(args.out,"train.log"),"w")
    def say(s): print(s,flush=True); log.write(s+"\n"); log.flush()
    def checkpoint(step, val_auroc=None):
        ck = {"prompt_learner": m.prompt_learner.state_dict(),
              "llama_proj": m.llama_proj.state_dict(), "step": step,
              "pipeline_version": PIPELINE_VERSION,
              "training_protocol": "supervised_custom_60_20_20",
              "seed": args.seed, "decoder_init": args.decoder_init or CKPT,
              "decoder_kind": args.decoder_kind, "freeze_decoder": args.freeze_decoder,
              "val_every": args.val_every, "val_limit": args.val_limit,
              "val_seed": 42, "config": vars(args).copy()}
        ck["dec" if args.decoder_kind == "upsample" else "image_decoder"] = m.decoder.state_dict()
        if val_auroc is not None: ck["val_auroc"] = val_auroc
        return ck
    def val_auc():
        m.eval(); sc,lb=[],[]
        with torch.no_grad():
            for p,cls,lab in validation_items:
                x=load_image_tensor(p, device="cuda")
                M,ce_=m.anomaly_map(x.half(),anchors[cls]); t18=m.prompt_learner(M.unsqueeze(1))
                sc.append(score_one(m,x,ce_,t18,DESC.get(cls,"an object"),"Is there any anomaly in the image?",yes_id,no_id)); lb.append(lab)
        m.train()
        return auroc(lb, sc)
    import json
    run_config = {"pipeline_version": PIPELINE_VERSION,
                  "training_protocol": "supervised_custom_60_20_20",
                  "decoder_kind": args.decoder_kind, "freeze_decoder": args.freeze_decoder,
                  "train_n": len(tr), "val_pool_n": len(va), "val_n": len(validation_items),
                  "test_n": len(te), "parameters": sum(p.numel() for p in params),
                  "selection": "fixed validation image AUROC", "config": vars(args)}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print("[config]", json.dumps(run_config), flush=True)
    m.train()
    best=-1.0
    for step in range(args.steps):
        opt.zero_grad(); tot=0.0; tseg=0.0
        for _ in range(args.accum):
            pool = tr_a if random.random()<0.5 else tr_n
            p,cls,lab = random.choice(pool)
            x = load_image_tensor(p, device="cuda")
            M, ce_ = m.anomaly_map(x.half(), anchors[cls])
            gt = load_mask(mask_path(p, lab))
            t18 = m.prompt_learner(M.unsqueeze(1))
            emb, labels = m.build(x, ce_, t18, DESC.get(cls,"an object"), "Is there any anomaly in the image?", YES if lab else NO)
            loss = m.forward(emb, labels) + 1.0*seg_loss(M, gt)
            if not torch.isfinite(loss): raise ValueError("Nonfinite Vicuna training loss")
            (loss/args.accum).backward(); tot += float(loss); tseg += float(seg_loss(M, gt))
        grad = torch.nn.utils.clip_grad_norm_(params,1.0)
        if not torch.isfinite(grad): raise ValueError("Nonfinite Vicuna training gradient")
        if step == 0: say("[gradient] norm %.6f decoder_frozen=%s map_shape=%s" % (float(grad), args.freeze_decoder, tuple(M.shape)))
        opt.step(); sched.step()
        if (step+1)%10==0:
            say("[step %d/%d] loss %.4f seg %.4f lr %.2e peak %.2f" % (step+1,args.steps,tot/args.accum,tseg/args.accum,
                sched.get_last_lr()[0], torch.cuda.max_memory_allocated()/1e9))
        if (step+1)%args.val_every==0 or step+1==args.steps:
            v=val_auc(); say("  [val] AUROC %.4f (%d fixed imgs) at %d" % (v,len(validation_items),step+1))
            if v>best:
                best=v
                torch.save(checkpoint(step+1, v), os.path.join(args.out,"best.pt"))
                say("  [save] best %.4f" % v)
        if (step+1)%500==0:
            torch.save(checkpoint(step+1), os.path.join(args.out,"step_%d.pt" % (step+1)))
    torch.save(checkpoint(args.steps), os.path.join(args.out,"final.pt"))
    say("[done]")

if __name__ == "__main__":
    main()
