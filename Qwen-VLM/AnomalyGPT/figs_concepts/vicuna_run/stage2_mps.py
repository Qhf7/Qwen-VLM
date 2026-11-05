"""把 18 个 token 拼进 Vicuna，生成回答。
   加载策略：mmap 逐个张量直接拷到 MPS，避免 12.6GB 的 CPU 中间副本。"""
import os, json, time, gc
import numpy as np, torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
OUT  = os.path.dirname(os.path.abspath(__file__))
VIC  = os.path.join(ROOT, "code/pretrained_ckpt/vicuna_ckpt/7b_v0")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)

txt = json.load(open(os.path.join(OUT, "prompt_text.json")))
P_BEFORE = "### Human: <Img>"
P_MIDDLE = "</Img> "
P_AFTER  = txt["desc"] + " " + txt["question"] + "\n### Assistant:"

print("[1/4] tokenizer ...", flush=True)
tokenizer = LlamaTokenizer.from_pretrained(VIC, use_fast=False)

print("[2/4] building skeleton on meta ...", flush=True)
config = LlamaConfig.from_pretrained(VIC)
with init_empty_weights():
    model = LlamaForCausalLM(config)

# meta → 真实设备（一次性分配空张量）
print("      allocating %.1f GB on %s ..." % (12.6, DEV), flush=True)
model = model.to_empty(device=DEV)
# rotary 的 inv_freq 是 buffer，meta 上算不出来，手动补
for name, m in model.named_modules():
    if hasattr(m, "inv_freq") and m.inv_freq is not None:
        dim = m.inv_freq.shape[0] * 2
        m.inv_freq = (1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).to(DEV)

idx = json.load(open(os.path.join(VIC, "pytorch_model.bin.index.json")))
wmap = idx["weight_map"]
shards = sorted(set(wmap.values()))
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())

print("[3/4] streaming %d shards → %s ..." % (len(shards), DEV), flush=True)
t0 = time.time(); done = 0
for shard in shards:
    sd = torch.load(os.path.join(VIC, shard), map_location="cpu", mmap=True, weights_only=True)
    for k, v in sd.items():
        if k in params:
            params[k].data.copy_(v.to(DEV))
        elif k in buffers:
            buffers[k].data.copy_(v.to(DEV))
        done += 1
        if done % 60 == 0:
            print("      %d tensors, %.0fs, mps alloc %.1f GB" %
                  (done, time.time()-t0, torch.mps.current_allocated_memory()/1e9), flush=True)
    del sd; gc.collect()
print("      loaded %d tensors in %.0fs" % (done, time.time()-t0), flush=True)
model.eval()

emb = model.get_input_embeddings()
def toks(s):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
bos_e = emb(torch.tensor([[tokenizer.bos_token_id]], device=DEV))
pre_e, mid_e, post_e = emb(toks(P_BEFORE)), emb(toks(P_MIDDLE)), emb(toks(P_AFTER))

results = {}
for k in ["defect", "normal"]:
    d = np.load(os.path.join(OUT, "embeds_%s.npz" % k))
    E = torch.tensor(d["E_img"], dtype=torch.float16, device=DEV)[:, None, :]
    P = torch.tensor(d["prompts"], dtype=torch.float16, device=DEV)
    inp = torch.cat([bos_e, pre_e, E, mid_e, P, post_e], dim=1)
    print("[4/4] %s: len=%d, generating ..." % (k, inp.shape[1]), flush=True)
    t0 = time.time()
    out = model.generate(inputs_embeds=inp, max_new_tokens=32, do_sample=True, top_p=0.1,
                         temperature=1.0, use_cache=True,
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True).strip()
    pr = torch.softmax(out.scores[0][0].float(), -1)
    top = torch.topk(pr, 6)
    yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no  = tokenizer.encode("No",  add_special_tokens=False)[0]
    results[k] = dict(text=text, seconds=round(time.time()-t0,1), maxM=float(d["M"].max()),
                      p_yes=float(pr[yes]), p_no=float(pr[no]),
                      top6=[(tokenizer.decode([i]), round(float(p),4)) for p,i in zip(top.values, top.indices)])
    print("      %.1fs  P(Yes)=%.4f P(No)=%.4f" % (time.time()-t0, pr[yes], pr[no]), flush=True)
    print("      top6:", results[k]["top6"], flush=True)
    print("      →", text, flush=True)

json.dump(results, open(os.path.join(OUT,"generation.json"),"w"), ensure_ascii=False, indent=2)
print("done")
