"""把 prompt_learner 第一层的 4 个 3x3 卷积核画出来，并展示它们对同一张 M 的响应。"""
import os, json
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
OUT = os.path.dirname(os.path.abspath(__file__))
torch.set_grad_enabled(False)

sd = torch.load(os.path.join(ROOT, "code/ckpt/train_mvtec/pytorch_model.pt"),
                map_location="cpu", weights_only=True)
W = sd["prompt_learner.meta_net.0.weight"].float().detach()   # [4,1,3,3]
b = sd["prompt_learner.meta_net.0.bias"].float().detach()     # [4]

d = np.load(os.path.join(OUT, "two_maps_defect.npz"))
M = torch.tensor(d["Mt"], dtype=torch.float32)[None, None]     # [1,1,224,224]
gt = d["gt"]

raw = F.conv2d(M, W, b, padding=1)[0]          # [4,224,224]
relu = torch.relu(raw)

fig, ax = plt.subplots(3, 5, figsize=(17, 10.2))
for c in range(5):
    if c == 0:
        ax[0,0].imshow(M[0,0], cmap="jet", vmin=0, vmax=1); ax[0,0].set_title("input M", fontsize=11)
        ax[1,0].imshow(M[0,0], cmap="jet", vmin=0, vmax=1); ax[1,0].set_title("(same M)", fontsize=9)
        ax[2,0].imshow(M[0,0], cmap="jet", vmin=0, vmax=1); ax[2,0].set_title("(same M)", fontsize=9)
    else:
        k = c - 1
        v = float(W[k,0].abs().max())
        im = ax[0,c].imshow(W[k,0], cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(3):
            for j in range(3):
                ax[0,c].text(j, i, "%.2f" % W[k,0,i,j], ha="center", va="center",
                             fontsize=9, color="black")
        ax[0,c].set_title("kernel %d  (bias=%.2f)" % (k, b[k]), fontsize=11)
        ax[0,c].set_xticks([]); ax[0,c].set_yticks([])
        plt.colorbar(im, ax=ax[0,c], fraction=0.046)
        vv = float(raw[k].abs().max())
        im2 = ax[1,c].imshow(raw[k], cmap="RdBu_r", vmin=-vv, vmax=vv)
        ax[1,c].set_title("channel %d raw response" % k, fontsize=10)
        plt.colorbar(im2, ax=ax[1,c], fraction=0.046)
        im3 = ax[2,c].imshow(relu[k], cmap="Reds", vmin=0, vmax=float(relu[k].max()))
        ax[2,c].set_title("after ReLU (what layer 2 sees)", fontsize=10)
        plt.colorbar(im3, ax=ax[2,c], fraction=0.046)
    if c > 0:
        for r in (1,2):
            g2 = np.ma.masked_where(~gt, gt)
            ax[r,c].imshow(g2, cmap="spring", alpha=0.18)
    for r in range(3):
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
fig.suptitle("4 kernels of prompt_learner.meta_net.0 (3x3, red=+, blue=-) and their responses to the same M", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "16_kernels.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# 统计
stats = []
for k in range(4):
    r = raw[k]; g = torch.tensor(gt)
    stats.append(dict(kernel=k, bias=round(float(b[k]),3),
                      response_min=round(float(r.min()),3), response_max=round(float(r.max()),3),
                      frac_positive=round(float((r>0).float().mean()),3),
                      mean_on_gt=round(float(r[g].mean()),3), mean_off_gt=round(float(r[~g].mean()),3),
                      max_loc=[int(x) for x in np.unravel_index(int(r.argmax()), (224,224))]))
print(json.dumps(stats, indent=2))
json.dump(stats, open(os.path.join(OUT,"kernel_stats.json"),"w"), indent=2)
