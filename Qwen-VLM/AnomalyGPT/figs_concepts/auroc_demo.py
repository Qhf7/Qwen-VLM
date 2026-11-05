"""用真实 M 计算：像素级 AUROC 的中间量 + 图像级判定的两种做法。"""
import os, sys, json
import numpy as np, torch

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
os.chdir(os.path.join(ROOT, "code/code"))
sys.path.insert(0, "model/ImageBind")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from models.imagebind_model import imagebind_huge, ModalityType
from models.multimodal_preprocessors import SimpleTokenizer

torch.set_grad_enabled(False); torch.set_num_threads(8)
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data/mvtec_anomaly_detection")

model, _ = imagebind_huge({"layers": [7, 15, 23, 31]})
model.load_state_dict(torch.load(os.path.join(ROOT, "code/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"),
                                 map_location="cpu", mmap=True, weights_only=True), strict=True)
model.eval()
ag = {k: v.float() for k, v in torch.load(os.path.join(ROOT, "code/ckpt/train_mvtec/pytorch_model.pt"),
                                          map_location="cpu", weights_only=True).items()}
W = [ag["image_decoder.fc.%d.weight" % i] for i in range(4)]
b = [ag["image_decoder.fc.%d.bias" % i] for i in range(4)]

pn = ['{}','flawless {}','perfect {}','unblemished {}','{} without flaw','{} without defect','{} without damage']
pa = ['damaged {}','broken {}','{} with flaw','{} with defect','{} with damage']
pt = ['a photo of a {}.','a photo of the {}.']
tok = SimpleTokenizer(bpe_path="model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
def enc(ss):
    ids = torch.stack([tok(s) for s in ss])
    return model({ModalityType.TEXT: ids})[ModalityType.TEXT][0]
fn = enc([t.format(s.format("bottle")) for s in pn for t in pt]).mean(0, keepdim=True)
fa = enc([t.format(s.format("bottle")) for s in pa for t in pt]).mean(0, keepdim=True)
F = torch.cat([fn/fn.norm(dim=-1,keepdim=True), fa/fa.norm(dim=-1,keepdim=True)], 0)

tf = transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])

def run(img_path):
    x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
    _, snaps = model({ModalityType.VISION: x})[ModalityType.VISION]
    probs = []
    for i, s in enumerate(snaps):
        p = s.transpose(0,1)[:,1:,:]
        p = torch.nn.functional.linear(p, W[i], b[i]); p = p/p.norm(dim=-1,keepdim=True)
        sim = 100.0*(p@F.T)
        probs.append(torch.softmax(sim, dim=-1)[0,:,1].reshape(16,16))
    M16 = torch.stack(probs).mean(0)[None,None]
    return torch.nn.functional.interpolate(M16, size=224, mode="bilinear", align_corners=True)[0,0].numpy()

def auroc(score, label):
    s = score.ravel(); g = label.ravel().astype(bool)
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    P, N = g.sum(), (~g).sum()
    return float((ranks[g].sum() - P*(P+1)/2) / (P*N))

def roc(score, label):
    s = score.ravel(); g = label.ravel().astype(bool)
    ths = np.unique(s)[::-1]
    tpr, fpr = [0.0], [0.0]
    P, N = g.sum(), (~g).sum()
    for t in ths:
        pred = s >= t
        tpr.append((pred & g).sum()/P); fpr.append((pred & ~g).sum()/N)
    return np.array(fpr), np.array(tpr)

cases = {
    "defect": (os.path.join(DATA,"bottle/test/broken_large/000.png"),
               os.path.join(DATA,"bottle/ground_truth/broken_large/000_mask.png")),
    "normal": (os.path.join(DATA,"bottle/test/good/000.png"), None),
}
res = {}
for k,(ip,gp) in cases.items():
    M = run(ip)
    gt = np.zeros((224,224), np.uint8) if gp is None else (np.asarray(Image.open(gp).convert("L").resize((224,224)))>127).astype(np.uint8)
    res[k] = (M, gt)
    print(k, "max=%.6f mean=%.6f" % (M.max(), M.mean()))

# 像素级：两张图合起来算（更接近真实评测）
M_all = np.concatenate([res["defect"][0][None], res["normal"][0][None]], 0)
G_all = np.concatenate([res["defect"][1][None], res["normal"][1][None]], 0)
pix_auc = auroc(M_all, G_all)
print("pooled pixel AUROC =", pix_auc)

# 阈值扫描（只对缺陷图）
Md, Gd = res["defect"]
sweep = []
for t in [0.05,0.1,0.3,0.5,0.7,0.9]:
    pred = Md >= t
    tp = (pred & (Gd==1)).sum(); fp = (pred & (Gd==0)).sum()
    fn_ = (~pred & (Gd==1)).sum(); tn = (~pred & (Gd==0)).sum()
    sweep.append(dict(tau=t, tpr=float(tp/(tp+fn_)), fpr=float(fp/(fp+tn)),
                      precision=float(tp/max(tp+fp,1)), area_ratio=float(pred.mean())))
    print(sweep[-1])

# 图像级：一张图一个分数
img_scores = {k: dict(maxv=float(v[0].max()), meanv=float(v[0].mean()),
                      top1=float(np.sort(v[0].ravel())[-int(0.01*v[0].size):].mean()),
                      top5=float(np.sort(v[0].ravel())[-int(0.05*v[0].size):].mean()))
              for k,v in res.items()}
print(json.dumps(img_scores, indent=2))

fpr, tpr = roc(M_all, G_all)
fig, ax = plt.subplots(1,2, figsize=(11,4.2))
ax[0].plot(fpr, tpr, lw=2); ax[0].plot([0,1],[0,1],'--',c='gray')
ax[0].set_xlabel("FPR (normal pixels flagged)"); ax[0].set_ylabel("TPR (defect pixels found)")
ax[0].set_title("pixel-level ROC, pooled over 2 images\nAUROC = %.4f" % pix_auc)
ax[0].grid(alpha=.3)
ax[1].imshow(Md, cmap="jet", vmin=0, vmax=1); ax[1].set_title("M of the defect image")
ax[1].set_xticks([]); ax[1].set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(OUT,"13_roc.png"), dpi=140, bbox_inches="tight")

json.dump(dict(pixel_auroc=pix_auc, sweep=sweep, image_scores=img_scores),
          open(os.path.join(OUT,"auroc_demo.json"),"w"), indent=2)
print("saved")
