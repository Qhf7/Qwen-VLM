"""Vicuna-7B int8 权重 + 逐层反量化（每通道 scale），内存约 6.7GB。
   跑 5 个条件，看 LLM 到底依赖不依赖那 18 个 token。"""
import os, json, time, gc
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from accelerate import init_empty_weights

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
OUT  = os.path.dirname(os.path.abspath(__file__))
VIC  = os.path.join(ROOT, "code/pretrained_ckpt/vicuna_ckpt/7b_v0")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_grad_enabled(False)

class QLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = None; self.s = None; self.bias = None
    def forward(self, x):
        w = self.q.to(x.dtype) * self.s.to(x.dtype)
        return F.linear(x, w, self.bias)

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

print("[3/4] 流式加载并量化 → %s" % DEV, flush=True)
idx = json.load(open(os.path.join(VIC, "pytorch_model.bin.index.json")))["weight_map"]
ql = {}; n = 0; t0 = time.time()
for shard in sorted(set(idx.values())):
    sd = torch.load(os.path.join(VIC, shard), map_location="cpu", mmap=True, weights_only=True)
    for k, v in sd.items():
        mod_name, attr = k.rsplit(".", 1)
        base = modules[mod_name]
        if isinstance(base, nn.Linear):
            if mod_name not in ql:
                ql[mod_name] = QLinear()
            if attr == "weight":
                w = v.float()
                s = w.abs().amax(dim=1, keepdim=True) / 127.0
                s = torch.clamp(s, min=1e-8)
                q = (w / s).round().clamp_(-127, 127).to(torch.int8)
                ql[mod_name].q = q.to(DEV)
                ql[mod_name].s = s.to(torch.float16).to(DEV)
                del w, q, s
            else:
                ql[mod_name].bias = nn.Parameter(v.to(DEV), requires_grad=False)
                parent, child = mod_name.rsplit(".", 1) if "." in mod_name else (None, mod_name)
                if parent is None: setattr(model, child, ql[mod_name])
                else: setattr(modules[parent], child, ql[mod_name])
        else:
            setattr(base, attr, nn.Parameter(v.to(DEV), requires_grad=False))
        n += 1
        if n % 80 == 0:
            print("      %d 张量 %.0fs alloc=%.1fGB" % (n, time.time()-t0,
                  torch.mps.current_allocated_memory()/1e9 if DEV=="mps" else 0), flush=True)
    del sd; gc.collect()
for name, m in ql.items():
    if m.q is not None and m.bias is None:
        parent, child = name.rsplit(".", 1) if "." in name else (None, name)
        if parent is None: setattr(model, child, m)
        else: setattr(modules[parent], child, m)
print("      完成 %d 张量 %.0fs" % (n, time.time()-t0), flush=True)

def fix_rotary(mod):
    for name, child in mod.named_children():
        if isinstance(child, LlamaRotaryEmbedding):
            setattr(mod, name, LlamaRotaryEmbedding(config=config).to(DEV))
        else:
            fix_rotary(child)
fix_rotary(model)
model.eval()
if DEV == "mps":
    print("      mps allocated = %.1f GB" % (torch.mps.current_allocated_memory()/1e9), flush=True)

emb = model.get_input_embeddings()
def toks(s):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
bos_e = emb(torch.tensor([[tokenizer.bos_token_id]], device=DEV))
pre_e, mid_e, post_e = emb(toks(P_BEFORE)), emb(toks(P_MIDDLE)), emb(toks(P_AFTER))
pid_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
pid_no  = tokenizer.encode("No",  add_special_tokens=False)[0]

d_d = np.load(os.path.join(OUT, "embeds_defect.npz"))
d_n = np.load(os.path.join(OUT, "embeds_normal.npz"))
T_def = torch.tensor(d_d["prompts"], dtype=torch.float16)
T_nor = torch.tensor(d_n["prompts"], dtype=torch.float16)
g = torch.Generator().manual_seed(0)
T_rnd = torch.randn(T_def.shape, generator=g).to(torch.float16)
T_rnd = T_rnd / T_rnd.norm(dim=-1, keepdim=True) * T_def.norm(dim=-1, keepdim=True)
T_zero = torch.zeros_like(T_def)

def run(tag, E_img, prompts, gen=20):
    inp = torch.cat([bos_e, pre_e,
                     torch.tensor(E_img, dtype=torch.float16, device=DEV)[:, None, :],
                     mid_e, prompts.to(DEV), post_e], dim=1)
    t0 = time.time()
    out = model.generate(inputs_embeds=inp,
                         attention_mask=torch.ones(inp.shape[:2], device=DEV),
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
    print("  %-24s P(Yes)=%.4f P(No)=%.4f %.1fs → %s" % (tag, r["p_yes"], r["p_no"], r["secs"], text[:80]), flush=True)
    return r

res = []
res.append(run("缺陷图 / 真实token", d_d["E_img"], T_def))
res.append(run("缺陷图 / 随机token", d_d["E_img"], T_rnd))
res.append(run("缺陷图 / 零token",   d_d["E_img"], T_zero))
res.append(run("缺陷图 / 用正常图的token", d_d["E_img"], T_nor))
res.append(run("正常图 / 真实token", d_n["E_img"], T_nor))
json.dump(res, open(os.path.join(OUT, "probe_results.json"), "w"), ensure_ascii=False, indent=2)
print("done")
