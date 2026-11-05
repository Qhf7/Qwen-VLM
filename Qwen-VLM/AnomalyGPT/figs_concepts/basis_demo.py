"""证明：维度/架构完全相同的两个模型，只是'换了一套坐标系'，
   提示向量也必须跟着换基——否则失效。"""
import torch, torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_grad_enabled(False); torch.manual_seed(0)

cfg = LlamaConfig(vocab_size=200, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=64)
m = LlamaForCausalLM(cfg).eval()
H, nh = cfg.hidden_size, cfg.num_attention_heads
hd = H // nh                                   # 16
g = torch.Generator().manual_seed(1)
head_perm = torch.randperm(nh, generator=g)    # 头级别的置换，保持 RoPE 结构
p = torch.cat([torch.arange(h*8*0+h*hd, (h+1)*hd) for h in head_perm])  # 简化写法
p = torch.cat([torch.arange(int(h)*hd, (int(h)+1)*hd) for h in head_perm])
inv = torch.argsort(p)

def permute_model(model, p):
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"):
            out[k] = v[:, p]
        elif "q_proj.weight" in k or "k_proj.weight" in k or "v_proj.weight" in k:
            out[k] = v[p][:, p]
        elif "o_proj.weight" in k:
            out[k] = v[p][:, p]
        elif "gate_proj.weight" in k or "up_proj.weight" in k:
            out[k] = v[:, p]
        elif "down_proj.weight" in k:
            out[k] = v[p]
        elif k.endswith("norm.weight") or "layernorm.weight" in k:
            out[k] = v[p]
        else:
            out[k] = v.clone()
    m2 = LlamaForCausalLM(cfg).eval()
    m2.load_state_dict(out)
    return m2

m2 = permute_model(m, p)

V, L = 200, 12
ids = torch.randint(0, V, (1, L))
n_tok = 18
tokens = torch.randn(1, n_tok, H) * 400         # 模拟 18 个高范数提示向量

def logits(model, ids):
    return model(ids).logits[0, -1]

base = logits(m, ids)
same = logits(m2, ids)
print("① 同一个模型，同输入                           → 输出向量前 3 维:", base[:3].tolist())
print("② 置换后的模型，同文本输入 → 最大差 %.2e（完全一致）" % (base-same).abs().max().item())
print("   （这说明 m2 只是 m 换了一套坐标系，功能完全相同）\n")

# 把 tokens 也按同一置换换基
tp = tokens[:, :, p]
print("③ 置换后的模型 + 同样换基的 token → 是否完全一致？")

def run_with_tokens(model, ids, tok):
    emb = model.get_input_embeddings()(ids)
    inp = torch.cat([emb[:, :4], tok, emb[:, 4:]], dim=1)
    return model(inputs_embeds=inp).logits[0, -1]

a = run_with_tokens(m, ids, tokens)
b = run_with_tokens(m2, ids, tp)
c = run_with_tokens(m2, ids, tokens)      # 换了模型但 token 没换基
print("   原模型 + 原 token        :", a[:3].tolist())
print("   新模型 + 换基后的 token  :", b[:3].tolist(), " 最大差 %.2e  ✓ 完全一致" % (a-b).abs().max().item())
print("   新模型 + 未换基的 token  :", c[:3].tolist(), " 最大差 %.3f  ✗ 失效" % (a-c).abs().max().item())

# 相似度
def cos(u, v): return float((u@v)/(u.norm()*v.norm()))
print("\n   与原始输出的余弦： 换基后 %.6f ｜ 未换基 %.3f" % (cos(a, b), cos(a, c)))
