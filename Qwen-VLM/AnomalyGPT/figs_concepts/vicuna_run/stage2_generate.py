"""阶段 2：把 E_img + 18 个 token 拼进 Vicuna 输入序列，生成回答（CPU fp16）。"""
import os, json, time
import numpy as np, torch
from transformers import LlamaForCausalLM, LlamaTokenizer

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
OUT = os.path.dirname(os.path.abspath(__file__))
VICUNA = os.path.join(ROOT, "code/pretrained_ckpt/vicuna_ckpt/7b_v0")
torch.set_num_threads(8)

txt = json.load(open(os.path.join(OUT, "prompt_text.json")))
p_before_text = "### Human: <Img>"
p_middle_text = "</Img> "
p_after_text = txt["desc"] + " " + txt["question"] + "\n### Assistant:"

print("[1/3] tokenizer ...", flush=True)
tokenizer = LlamaTokenizer.from_pretrained(VICUNA, use_fast=False)

print("[2/3] loading Vicuna-7B fp16 on CPU ...", flush=True)
t0 = time.time()
model = LlamaForCausalLM.from_pretrained(VICUNA, torch_dtype=torch.float16,
                                         low_cpu_mem_usage=True, device_map="cpu")
model.eval()
print("      loaded in %.1fs" % (time.time() - t0), flush=True)

emb = model.get_input_embeddings()
def toks(s):
    return tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"]

bos_e = emb(torch.tensor([[tokenizer.bos_token_id]]))
pre_e, mid_e, post_e = emb(toks(p_before_text)), emb(toks(p_middle_text)), emb(toks(p_after_text))

results = {}
for k in ["defect", "normal"]:
    d = np.load(os.path.join(OUT, "embeds_%s.npz" % k))
    E_img = torch.tensor(d["E_img"], dtype=torch.float16)[:, None, :]
    prompts = torch.tensor(d["prompts"], dtype=torch.float16)
    inputs = torch.cat([bos_e, pre_e, E_img, mid_e, prompts, post_e], dim=1)
    print("[3/3] %s: seq len=%d, prefill ..." % (k, inputs.shape[1]), flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs_embeds=inputs, max_new_tokens=36, do_sample=True,
                             top_p=0.1, temperature=1.0, use_cache=True,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
    dt = time.time() - t0
    seq = out.sequences[0]
    text = tokenizer.decode(seq, skip_special_tokens=True).strip()
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    top = torch.topk(probs, 6)
    first = [(tokenizer.decode([i]), round(float(p), 4)) for p, i in zip(top.values, top.indices)]
    pid_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    pid_no = tokenizer.encode("No", add_special_tokens=False)[0]
    results[k] = dict(text=text, seconds=round(dt, 1), maxM=float(d["M"].max()),
                      first_token_top6=first,
                      p_yes=float(probs[pid_yes]), p_no=float(probs[pid_no]))
    print("      %.1fs | P(Yes)=%.4f P(No)=%.4f" % (dt, probs[pid_yes], probs[pid_no]), flush=True)
    print("      top6:", first, flush=True)
    print("      →", text, flush=True)

json.dump(results, open(os.path.join(OUT, "generation.json"), "w"), ensure_ascii=False, indent=2)
print("done")
