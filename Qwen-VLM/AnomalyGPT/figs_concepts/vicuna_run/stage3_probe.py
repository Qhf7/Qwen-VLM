"""加载 Vicuna（逐张量直接拷到 MPS，不做 to_empty 双份分配），
   跑 5 个条件：真实 token / 随机 token / 零 token / 换图 token / 正常图。"""
import os, json, time, gc
import numpy as np, torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from accelerate import init_empty_weights

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
OUT  = os.path.dirname(os.path.abspath(__file__))
VIC  = os.path.join(ROOT, "code/pretrained_ckpt/vicuna_ckpt/7b_v0")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)

txt = json.load(open(os.path.join(OUT, "prompt_text.json")))
P_BEFORE, P_MIDDLE = "### Human: <Img>", "</Img> "
P_AFTER = txt["desc"] + " " + txt["question"] + "\n### Assistant:"

print("[1/4] tokenizer", flush=True)
tokenizer = LlamaTokenizer.from_pretrained(VIC, use_fast=False)

print("[2/4] meta 骨架", flush=True)
config = LlamaConfig.from_pretrained(VIC)
with init_empty_weights():
    model = LlamaForCausalLM(config)
modules = dict(model.named_modules())

print("[3/4] 逐张量 → %s" % DEV, flush=True)
idx = json.load(open(os.path.join(VIC, "pytorch_model.bin.index.json")))["weight_map"]
t0 = time.time(); n = 0
for shard in sorted(set(idx.values())):
    sd = torch.load(os.path.join(VIC, shard), map_location="cpu", mmap=True, weights_only=True)
    for k, v in sd.items():
        mod_name, attr = k.rsplit(".", 1)
        setattr(modules[mod_name], attr, torch.nn.Parameter(v.to(DEV), requires_grad=False))
        n += 1
        if n % 80 == 0:
            print("      %d 张量 %.0fs  mps=%.1fGB" % (n, time.time()-t0,
                  torch.mps.current_allocated_memory()/1e9), flush=True)
    del sd; gc.collect()
print("      完成 %d 张量，%.0fs" % (n, time.time()-t0), flush=True)

def fix_rotary(mod):
    for name, child in mod.named_children():
        if isinstance(child, LlamaRotaryEmbedding):
            setattr(mod, name, LlamaRotaryEmbedding(config=config).to(DEV))
        else:
            fix_rotary(child)
fix_rotary(model)
model.eval()
print("      mps=%.1fGB" % (torch.mps.current_allocated_memory()/1e9), flush=True)

emb = model.get_input_embeddings()
def toks(s):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
bos_e = emb(torch.tensor([[tokenizer.bos_token_id]], device=DEV))
pre_e, mid_e, post_e = emb(toks(P_BEFORE)), emb(toks(P_MIDDLE)), emb(toks(P_AFTER))
pid_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
pid_no  = tokenizer.encode("No",  add_special_tokens=False)[0]

d_d = np.load(os.path.join(OUT, "embeds_defect.npz"))
d_n = np.load(os.path.join(OUT, "embeds_normal.npz"))
T_def = torch.tensor(d_d["prompts"], dtype=torch.float16)     # [1,18,4096]
T_nor = torch.tensor(d_n["prompts"], dtype=torch.float16)
g = torch.Generator().manual_seed(0)
T_rnd = torch.randn(T_def.shape, generator=g).to(torch.float16)
T_rnd = T_rnd / T_rnd.norm(dim=-1, keepdim=True) * T_def.norm(dim=-1, keepdim=True)
T_zero = torch.zeros_like(T_def)

def run(tag, E_img, prompts, gen=24):
    inp = torch.cat([bos_e, pre_e,
                     torch.tensor(E_img, dtype=torch.float16, device=DEV)[:, None, :],
                     mid_e, prompts.to(DEV), post_e], dim=1)
    t0 = time.time()
    out = model.generate(inputs_embeds=inp, attention_mask=torch.ones(inp.shape[:2], device=DEV),
                         max_new_tokens=gen, do_sample=True, top_p=0.1, temperature=1.0,
                         use_cache=True, eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True).strip()
    pr = torch.softmax(out.scores[0][0].float(), -1)
    top = torch.topk(pr, 5)
    r = dict(tag=tag, text=text, secs=round(time.time()-t0, 1),
             p_yes=round(float(pr[pid_yes]), 4), p_no=round(float(pr[pid_no]), 4),
             top5=[(tokenizer.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices)])
    print("  %-22s P(Yes)=%.4f P(No)=%.4f  %.1fs  → %s" % (tag, r["p_yes"], r["p_no"], r["secs"], text[:70]), flush=True)
    return r

res = []
res.append(run("缺陷图 / 真实token", d_d["E_img"], T_def))
res.append(run("缺陷图 / 随机token", d_d["E_img"], T_rnd))
res.append(run("缺陷图 / 零token",   d_d["E_img"], T_zero))
res.append(run("缺陷图 / 换成正常图的token", d_d["E_img"], T_nor))
res.append(run("正常图 / 真实token", d_n["E_img"], T_nor))
json.dump(res, open(os.path.join(OUT, "probe_results.json"), "w"), ensure_ascii=False, indent=2)
print("done")
