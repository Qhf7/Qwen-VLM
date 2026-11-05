"""10x10 玩具例子：缺陷 4 格，逐项算 CE / Focal / Dice。"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 10
gt = np.zeros((N, N), np.float32)
gt[4:6, 4:6] = 1.0                      # 2x2 = 4 个缺陷像素
Y = gt.ravel(); P_pos = Y.sum()          # 4

def ce(p, y, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return -(y*np.log(p) + (1-y)*np.log(1-p))

def focal(p, y, gamma=2.0, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    pt = np.where(y == 1, p, 1-p)
    return -(1-pt)**gamma * np.log(pt)

def dice_loss(p, y, smooth=1.0):
    p, y = p.ravel(), y.ravel()
    inter = (p*y).sum()
    d = (2*inter + smooth) / (p.sum() + y.sum() + smooth)
    return 1 - d, d

scen = {
    "A perfect":        np.where(gt == 1, 0.999, 0.001),
    "B good":      np.where(gt == 1, 0.90, 0.10),
    "C all-normal":    np.full((N,N), 0.01),
    "D all-abnormal":    np.full((N,N), 0.99),
    "E blurry": np.where(gt == 1, 0.55, 0.45),
}

rows = []
for name, pi in scen.items():
    pf = pi.ravel()
    c = ce(pf, Y); f = focal(pf, Y)
    dl, d = dice_loss(pi, gt)
    acc = float(((pf > 0.5).astype(float) == Y).mean())
    rows.append(dict(name=name, acc=acc, ce=float(c.mean()), focal=float(f.mean()), dice_loss=float(dl), dice=float(d),
                     ce_share_from_4=float(c[Y==1].sum()/c.sum()),
                     focal_share_from_4=float(f[Y==1].sum()/f.sum()),
                     sum_p=float(pf.sum()), sum_py=float((pf*Y).sum())))

print("%-14s %7s %9s %10s %10s %8s %10s" % ("场景","像素准确率","CE","Focal","Dice损失","Dice","4格占CE比"))
for r in rows:
    print("%-14s %6.1f%% %9.4f %10.4f %10.4f %8.4f %9.1f%%" %
          (r["name"], r["acc"]*100, r["ce"], r["focal"], r["dice_loss"], r["dice"], r["ce_share_from_4"]*100))

# 逐像素手算（场景 B / C）
print("\n场景 B 单个像素：缺陷格 p=0.90 → CE=%.4f Focal=%.4f" % (ce(np.array([0.9]),np.array([1.]))[0], focal(np.array([0.9]),np.array([1.]))[0]))
print("场景 B 单个像素：正常格 p=0.10 → CE=%.4f Focal=%.4f" % (ce(np.array([0.1]),np.array([0.]))[0], focal(np.array([0.1]),np.array([0.]))[0]))
print("场景 C 单个像素：缺陷格 p=0.01 → CE=%.4f Focal=%.4f" % (ce(np.array([0.01]),np.array([1.]))[0], focal(np.array([0.01]),np.array([1.]))[0]))
print("场景 C 单个像素：正常格 p=0.99 → CE=%.4f Focal=%.4f" % (ce(np.array([0.99]),np.array([0.]))[0], focal(np.array([0.99]),np.array([0.]))[0]))

print("\ngamma 的影响（缺陷格 p=0.9）：")
for g in [0,1,2,3]:
    print("  gamma=%d → %.5f" % (g, focal(np.array([0.9]),np.array([1.]),gamma=g)[0]))

json.dump(rows, open("toy_loss.json","w"), indent=2, ensure_ascii=False)

# ---------------- 图
fig = plt.figure(figsize=(16, 6.4))
gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1])
ax0 = fig.add_subplot(gs[0,0]); ax0.imshow(gt, cmap="gray_r", vmin=0, vmax=1)
ax0.set_title("ground truth\n4 defect pixels"); ax0.set_xticks([]); ax0.set_yticks([])
for i,(name,pi) in enumerate(scen.items()):
    a = fig.add_subplot(gs[0, i+1])
    im = a.imshow(pi, cmap="jet", vmin=0, vmax=1)
    a.set_title(name); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=a, fraction=0.046)

axb = fig.add_subplot(gs[1, :3])
x = np.arange(len(rows)); w = 0.27
axb.bar(x-w, [r["ce"] for r in rows], w, label="CE", color="tab:blue")
axb.bar(x,   [r["focal"] for r in rows], w, label="Focal (γ=2)", color="tab:red")
axb.bar(x+w, [r["dice_loss"] for r in rows], w, label="Dice loss", color="tab:green")
axb.set_xticks(x); axb.set_xticklabels([r["name"] for r in rows], fontsize=9)
axb.set_ylabel("loss (mean over 100 pixels)"); axb.legend(); axb.set_title("three losses on the same prediction")

axc = fig.add_subplot(gs[1, 3:])
pf = scen["C all-normal"].ravel()
c = ce(pf, Y); f = focal(pf, Y)
idx = np.argsort(-c)
axc.stem(np.arange(100), c[idx], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="CE per pixel")
axc.stem(np.arange(100)+0.0, f[idx], linefmt="C3-", markerfmt="C3o", basefmt=" ", label="Focal per pixel")
axc.set_yscale("log"); axc.set_xlabel("pixels sorted by CE"); axc.set_ylabel("loss (log)")
axc.legend(); axc.set_title("'all normal' prediction: 4 pixels dominate")
fig.tight_layout(); fig.savefig("17_toy_loss.png", dpi=140, bbox_inches="tight")
print("saved 17_toy_loss.png")
