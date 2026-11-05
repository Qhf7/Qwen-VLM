"""把 18 个提示嵌入和 Vicuna 词表 32000 个词向量做余弦，看它们'最像哪个词'。"""
import os, json
import numpy as np, torch
from transformers import LlamaTokenizer

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
VIC  = os.path.join(ROOT, "code/pretrained_ckpt/vicuna_ckpt/7b_v0")
OUT  = os.path.dirname(os.path.abspath(__file__))
torch.set_grad_enabled(False)

tok = LlamaTokenizer.from_pretrained(VIC, use_fast=False)

# 1) 词表嵌入（只读这一个张量）
idx = json.load(open(os.path.join(VIC, "pytorch_model.bin.index.json")))["weight_map"]
key = "model.embed_tokens.weight"
shard = os.path.join(VIC, idx[key])
sd = torch.load(shard, map_location="cpu", mmap=True, weights_only=True)
E = sd[key].float()                      # [32001, 4096]
del sd
print("vocab embedding:", tuple(E.shape), "norm mean %.2f" % E.norm(dim=-1).mean())
En = E / E.norm(dim=-1, keepdim=True)

# 2) 18 个提示嵌入
P = torch.tensor(np.load(os.path.join(OUT, "embeds_defect.npz"))["prompts"][0]).float()  # [18,4096]
Pn = P / P.norm(dim=-1, keepdim=True)
print("prompt norms:", np.round(P.norm(dim=-1).numpy(), 1))

# 3) 余弦（分块）
sims = []
for i in range(0, Pn.shape[0]):
    sims.append(En @ Pn[i])
S = torch.stack(sims)                    # [18, 32001]
top = S.topk(10, dim=-1)

rows = []
for i in range(18):
    words = []
    for sc, wi in zip(top.values[i], top.indices[i]):
        w = tok.decode([int(wi)]).strip()
        words.append((w, round(float(sc), 3)))
    rows.append(dict(index=i, kind="base" if i < 9 else "map",
                     max_cos=round(float(top.values[i][0]), 3),
                     top10=words))
    print("%2d %-5s max=%.3f | %s" % (i, rows[-1]["kind"], rows[-1]["max_cos"],
          "  ".join("%r:%.3f" % (w, s) for w, s in words[:8])))

# 4) 熵：这个向量在词表上的分布有多"尖"
for T in [1.0]:
    logits = S / T
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-12).log()).sum(-1).numpy()
    print("\n词表分布熵（nats，ln(32001)=10.37）:", np.round(ent, 2))

json.dump(rows, open(os.path.join(OUT, "prompt_neighbors.json"), "w"), ensure_ascii=False, indent=2)

# 5) 图：每个 token 的最大余弦 + 词表 PCA 散点
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(np.arange(18), top.values[:, 0].numpy(), color=["tab:blue"]*9 + ["tab:red"]*9)
ax[0].axhline(0.5, ls="--", c="gray"); ax[0].set_ylim(0, 1)
ax[0].set_xlabel("prompt index (0-8 = base, 9-17 = from M)")
ax[0].set_title("max cosine to any vocabulary word")
ax[0].set_xticks(range(18))

g = torch.Generator().manual_seed(0)
sel = torch.randperm(E.shape[0], generator=g)[:4000]
X = torch.cat([En[sel], Pn], 0)
Xc = X - X.mean(0, keepdim=True)
U, Sv, V = torch.pca_lowrank(Xc, q=2, niter=4)
Z = Xc @ V[:, :2]
ax[1].scatter(Z[:4000, 0], Z[:4000, 1], s=3, c="lightgray", label="vocab (4k sampled)")
ax[1].scatter(Z[4000:, 0], Z[4000:, 1], s=60, c=["tab:blue"]*9 + ["tab:red"]*9,
              edgecolor="k", label="18 prompt embeddings")
ax[1].legend(); ax[1].set_title("PCA of embedding space")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "20_prompt_vocab.png"), dpi=140, bbox_inches="tight")
print("saved")
