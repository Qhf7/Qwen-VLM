"""Vicuna-7B vs Qwen3-8B 的嵌入空间对照：同一批词，落在完全不同的几何里。"""
import os, json
import numpy as np, torch
from transformers import LlamaTokenizer, AutoTokenizer
torch.set_grad_enabled(False)

ROOT="/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
VIC=os.path.join(ROOT,"code/pretrained_ckpt/vicuna_ckpt/7b_v0")

# --- Vicuna 嵌入
idx=json.load(open(os.path.join(VIC,"pytorch_model.bin.index.json")))["weight_map"]
sd=torch.load(os.path.join(VIC,idx["model.embed_tokens.weight"]),map_location="cpu",mmap=True,weights_only=True)
EV=sd["model.embed_tokens.weight"].float(); del sd
tv=LlamaTokenizer.from_pretrained(VIC,use_fast=False)

# --- Qwen3-8B 嵌入（bf16 原始字节）
EQ=torch.empty(151936,4096,dtype=torch.float16)
mm=np.memmap("/tmp/qwen3_embed.bf16",dtype=np.uint16,mode="r",shape=(151936,4096))
for i in range(0,151936,16384):
    EQ[i:i+16384]=torch.from_numpy(np.ascontiguousarray(mm[i:i+16384])).view(torch.bfloat16).float()
del mm
tq=AutoTokenizer.from_pretrained(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots")+"/"+os.listdir(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"))[0], use_fast=False)

WORDS=["bottle","scratch","defect","damaged","yes","no","surface","metal","crack",
       "hole","clean","normal","broken","rust","dent","spot","texture","edge",
       "small","large","left","right","top","bottom"]
def emb_of(tok,E,word):
    ids=tok.encode(" "+word,add_special_tokens=False)
    if not ids: return None
    v=E[torch.tensor(ids)].mean(0)
    return v/v.norm()

WV=torch.stack([emb_of(tv,EV,w) for w in WORDS])
WQ=torch.stack([emb_of(tq,EQ,w) for w in WORDS])
CV=(WV@WV.T).numpy(); CQ=(WQ@WQ.T).numpy()
iu=np.triu_indices(len(WORDS),1)
corr=np.corrcoef(CV[iu],CQ[iu])[0,1]
print("两空间词-词余弦的相关性: %.3f" % corr)
print("平均绝对差: %.3f   Vicuna 均值 %.3f  Qwen3 均值 %.3f" % (np.abs(CV-CQ)[iu].mean(), CV[iu].mean(), CQ[iu].mean()))
for a,b in [("scratch","defect"),("yes","no"),("bottle","metal"),("crack","hole"),("damaged","broken")]:
    i,j=WORDS.index(a),WORDS.index(b)
    print("  cos(%s,%s): Vicuna %+.3f | Qwen3 %+.3f" % (a,b,CV[i,j],CQ[i,j]))

def nn(E,tok,word,k=4):
    v=emb_of(tok,E,word); s=E@v
    val,ix=s.topk(k+1)
    return [ (tok.decode([int(i)]).strip(), round(float(x),2)) for x,i in zip(val[1:],ix[1:]) ]
print("\n'bottle' 的最近邻:")
print("  Vicuna:",nn(EV,tv,"bottle"))
print("  Qwen3 :",nn(EQ,tq,"bottle"))
print("'defect' 的最近邻:")
print("  Vicuna:",nn(EV,tv,"defect"))
print("  Qwen3 :",nn(EQ,tq,"defect"))

# 18 个提示 token 在两个空间里的最大余弦
P=torch.tensor(np.load("vicuna_run/embeds_defect.npz")["prompts"][0]).float()
Pn=P/P.norm(dim=-1,keepdim=True)
mx_v=(EV/EV.norm(dim=-1,keepdim=True))@Pn.T
Enq=EQ/EQ.norm(dim=-1,keepdim=True).clamp(min=1e-6)
mx_q=torch.stack([Enq@Pn[i].half() for i in range(18)]).float()
# 各向异性：随机词对之间的平均余弦
for nm,E_ in [("Vicuna",EV),("Qwen3",EQ)]:
    g=torch.Generator().manual_seed(3); sel=torch.randperm(E_.shape[0],generator=g)[:1500]
    X=E_[sel].float(); X=X/X.norm(dim=-1,keepdim=True)
    C=X@X.T; n=X.shape[0]
    print("%s 随机词对平均余弦 = %.3f" % (nm, float((C.sum()-n)/(n*(n-1)))))
print("\n18 个提示 token 与词表的最大余弦：Vicuna 中位 %.3f | Qwen3 中位 %.3f" %
      (float(mx_v.max(0).values.median()), float(mx_q.max(0).values.median())))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def pca(E,tok,W,n=6000,seed=0):
    g=torch.Generator().manual_seed(seed)
    sel=torch.randperm(E.shape[0],generator=g)[:n]
    En=E[sel].float(); En=En/En.norm(dim=-1,keepdim=True)
    X=torch.cat([En,W],0); Xc=X-X.mean(0,keepdim=True)
    U,S,V=torch.pca_lowrank(Xc,q=2,niter=4)
    return (Xc@V[:,:2]).numpy()
ZV=pca(EV,tv,WV); ZQ=pca(EQ,tq,WQ)
fig,ax=plt.subplots(2,2,figsize=(15,11))
for a,Z,title in [(ax[0,0],ZV,"Vicuna-7B (d=4096)"),(ax[0,1],ZQ,"Qwen3-8B (d=4096)")]:
    a.scatter(Z[:-24,0],Z[:-24,1],s=3,c="lightgray")
    a.scatter(Z[-24:,0],Z[-24:,1],s=55,c="tab:red",edgecolor="k",zorder=3)
    for i,w in enumerate(WORDS): a.annotate(w,(Z[-24+i,0],Z[-24+i,1]),fontsize=9,xytext=(3,3),textcoords="offset points")
    a.set_title(title,fontsize=12); a.set_xticks([]); a.set_yticks([])
ax[1,0].scatter(CV[iu],CQ[iu],s=10,alpha=.5)
ax[1,0].plot([-0.2,1],[-0.2,1],"k--",lw=1)
for a_,b_ in [("scratch","defect"),("yes","no"),("bottle","metal")]:
    i,j=WORDS.index(a_),WORDS.index(b_)
    ax[1,0].annotate(f"{a_}-{b_}",(CV[i,j],CQ[i,j]),fontsize=8,color="tab:red")
ax[1,0].set_xlabel("cosine in Vicuna"); ax[1,0].set_ylabel("cosine in Qwen3")
ax[1,0].set_title("same word pair, similarity: r=%.2f" % corr,fontsize=11)
ax[1,0].grid(alpha=.3)
ax[1,1].boxplot([mx_v.max(0).values.numpy(),mx_q.max(0).values.numpy()],labels=["Vicuna","Qwen3"])
ax[1,1].axhline(0.071,ls="--",c="gray"); ax[1,1].text(1.6,0.075,"random baseline",fontsize=8)
ax[1,1].set_ylabel("max cos of the 18 prompt tokens\nto any vocabulary word")
ax[1,1].set_title("the 18 tokens are 'unreadable' in both spaces",fontsize=11)
fig.tight_layout(); fig.savefig("23_space_compare.png",dpi=140,bbox_inches="tight")
print("saved 23_space_compare.png")
