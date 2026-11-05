"""原生视觉 bridge 的评测：同一批图上比较 真实18token / 零token / 随机token。
指标：图像级 AUROC、准确率、二类宏F1（候选打分，不依赖生成解析）。
用法:
  python eval_native.py --ckpt runs/native_formal/best.pt --limit 600
  python eval_native.py --ckpt runs/native_formal/best.pt --conditions real zero random
"""
import argparse, glob, hashlib, json, os, random, sys, time
import torch
from PIL import Image
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT")
from qwen3vl_native_bridge import (NativeBridge, eq1_map_with_cls, text_anchors,
                                  load_module_checkpoint)
from visa_pipeline import (PIPELINE_VERSION, load_image_tensor, load_image_pil,
                           auroc, average_precision)

VISA = "/root/autodl-tmp/industrial_vlm/datasets/VisA"
DESC = {"candle":"a candle","cashew":"a cashew","chewinggum":"a chewinggum","fryum":"a fryum",
        "macaroni1":"a macaroni","macaroni2":"a macaroni","pcb1":"a pcb","pcb2":"a pcb",
        "pcb3":"a pcb","pcb4":"a pcb","pipe_fryum":"a pipe fryum","capsules":"a capsule"}

def collect_val(val_ratio=0.15):
    va = []
    for cls in sorted(os.listdir(VISA)):
        base = os.path.join(VISA, cls, "Data/Images")
        if not os.path.isdir(base): continue
        for lab, sub in ((0,"Normal"),(1,"Anomaly")):
            for p in sorted(glob.glob(os.path.join(base, sub, "*"))):
                h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 100
                if h < val_ratio*100: va.append((p, cls, lab))
    return va

@torch.no_grad()
def prompt_embeds(m, pil, question, tokens, img_emb=None):
    proc = m.processor
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    q = proc(text=[text], images=[pil], return_tensors="pt")
    ids = q["input_ids"].cuda()
    emb = m.qwen.get_input_embeddings()(ids)
    vis, deep = m.qwen.model.get_image_features(pixel_values=q["pixel_values"].cuda(),
                                                image_grid_thw=q["image_grid_thw"].cuda())
    vis = vis[0] if isinstance(vis,(list,tuple)) else vis
    if getattr(m, "vis_adapter", None) is not None:
        vis = vis + m.vis_adapter(vis.float()).to(vis.dtype)
    pos = (ids[0] == m.image_token_id).nonzero(as_tuple=True)[0]
    emb[0, pos] = vis.to(emb.dtype)
    ins = int(pos[-1]) + 1
    n = tokens.shape[1]
    tok_block = tokens.to(emb.dtype)
    if img_emb is not None:
        tok_block = torch.cat([img_emb.to(emb.dtype).view(1,1,-1), tok_block], dim=1)
    emb = torch.cat([emb[:, :ins], tok_block, emb[:, ins:]], dim=1)
    vpm = torch.zeros(1, emb.shape[1], dtype=torch.bool, device=emb.device); vpm[0, pos] = True
    return emb, torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device), vpm, deep

@torch.no_grad()
def score_one(m, pil, question, tokens, yes_id, no_id, img_emb=None):
    emb, am, vpm, deep = prompt_embeds(m, pil, question, tokens, img_emb)
    out = m.lm(inputs_embeds=emb, attention_mask=am, visual_pos_masks=vpm,
               deepstack_visual_embeds=deep)
    lg = m.lm_head(out.last_hidden_state[:, -1]).float()[0]
    return float(lg[yes_id] - lg[no_id])

def score_one_img(m, pil, question, tokens, yes_id, no_id, img_emb=None):
    return score_one(m, pil, question, tokens, yes_id, no_id, img_emb)

def f1s(y, p):
    tp=sum(1 for a,b in zip(y,p) if a==1 and b==1); fn=sum(1 for a,b in zip(y,p) if a==1 and b==0)
    fp=sum(1 for a,b in zip(y,p) if a==0 and b==1); tn=sum(1 for a,b in zip(y,p) if a==0 and b==0)
    return (2*tp/max(2*tp+fp+fn,1) + 2*tn/max(2*tn+fn+fp,1))/2, tp/max(tp+fn,1), fp/max(fp+tn,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/native_formal/best.pt")
    ap.add_argument("--conditions", nargs="+", choices=["real","zero","random"],
                    default=["real","zero","random"])
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--out", default="runs/native_eval_corrected")
    ap.add_argument("--clean-test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.clean_test:
        from split_visa import splits
        items = splits()[2]
        print("[split] supervised custom test: hash >=80 of official-test sources", flush=True)
    else:
        items = collect_val()
        random.seed(0); random.shuffle(items); items = items[:args.limit]
    print("[data] %d val images (%d anomaly)" % (len(items), sum(i[2] for i in items)), flush=True)

    if not items:
        raise ValueError("No evaluation images selected")
    m = NativeBridge()
    imgtok_enabled = False
    if os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        load_module_checkpoint(m.prompt_learner, sd, "prompt_learner")
        if ("image_decoder" in sd or "decoder" in sd or
                any(k.startswith(("image_decoder.", "decoder.")) for k in sd)):
            load_module_checkpoint(m.decoder, sd, "image_decoder", ("decoder",))
            print("[ckpt] image_decoder 也已加载", flush=True)
        if "vis_adapter" in sd:
            class _VA(torch.nn.Module):
                def __init__(self, d=4096, r=1024):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(d, r); self.act = torch.nn.GELU(); self.fc2 = torch.nn.Linear(r, d)
                def forward(self, x): return self.fc2(self.act(self.fc1(x)))
            m.vis_adapter = _VA().cuda()
            m.vis_adapter.load_state_dict(sd["vis_adapter"], strict=True)
            print("[ckpt] vis_adapter 已加载", flush=True)
        if "llama_proj" in sd or any(k.startswith("llama_proj.") for k in sd):
            load_module_checkpoint(m.llama_proj, sd, "llama_proj")
            imgtok_enabled = True
            print("[ckpt] ImageBind CLS token 已启用", flush=True)
        print("[ckpt] %s step=%s val=%s" % (args.ckpt, sd.get("step"), sd.get("val")), flush=True)
    else:
        print("[ckpt] MISSING -> 用发布初始化权重（基线）", flush=True)
    m.decoder.float().eval()
    m.prompt_learner.eval()
    m.llama_proj.eval()
    if m.vis_adapter is not None:
        m.vis_adapter.eval()
    anchors = {c: text_anchors(m.encoder, c.replace("_"," "), "cuda") for c in set(i[1] for i in items)}
    tok_ = m.processor.tokenizer
    yes_id = tok_.encode("Yes", add_special_tokens=False)[0]
    no_id  = tok_.encode("No",  add_special_tokens=False)[0]
    g = torch.Generator().manual_seed(0)

    summary = {}
    for cond in args.conditions:
        y, s, t0 = [], [], time.time()
        for i,(p,cls,lab) in enumerate(items):
            pil = load_image_pil(p)
            x = load_image_tensor(p, device="cuda")
            M, cls_emb = eq1_map_with_cls(m.encoder, m.decoder, x.half(), anchors[cls])
            with torch.no_grad():
                real = m.prompt_learner(M.unsqueeze(1).float())
                img_emb = m.llama_proj(cls_emb) if imgtok_enabled else None
            if cond == "real":
                tok = real
            elif cond == "zero":
                tok = torch.zeros_like(real)
            else:
                r = torch.randn(real.shape, generator=g).to(real.device, real.dtype)
                tok = r / r.norm(dim=-1, keepdim=True) * real.norm(dim=-1, keepdim=True)
            q = "This is a photo of %s for anomaly detection. Is there any anomaly in the image?" % DESC.get(cls,"an object")
            s.append(score_one(m, pil, q, tok, yes_id, no_id, img_emb)); y.append(lab)
            if (i+1) % 100 == 0:
                print("  [%s] %d/%d %.0fs" % (cond, i+1, len(items), time.time()-t0), flush=True)
        pred = [1 if v > 0 else 0 for v in s]
        f1, rec, fpr = f1s(y, pred)
        summary[cond] = {"n": len(y), "auroc": auroc(y, s),
                         "average_precision": average_precision(y, s),
                         "acc": sum(int(a==b) for a,b in zip(y,pred))/len(y),
                         "f1_macro": f1, "recall": rec, "fpr": fpr,
                         "pipeline_version": PIPELINE_VERSION,
                         "imgtok_enabled": imgtok_enabled,
                         "checkpoint": os.path.abspath(args.ckpt),
                         "split": "custom_test" if args.clean_test else "legacy_hash_val"}
        with open(os.path.join(args.out, "pred_%s.jsonl" % cond), "w") as f:
            for (p,cls,lab),v in zip(items, s):
                f.write(json.dumps({"image":p,"class":cls,"label":lab,"score":v}, ensure_ascii=False)+"\n")
        print("  -> %s: AUROC %.4f  acc %.4f  F1 %.4f" % (cond, summary[cond]["auroc"],
              summary[cond]["acc"], f1), flush=True)
    json.dump(summary, open(os.path.join(args.out,"metrics.json"),"w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
