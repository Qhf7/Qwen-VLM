"""同一张图上并排比较两条 M：
   M_text = 文本锚分支（image_decoder + softmax(100)）
   M_mem  = 记忆库分支（1 张正常图 × 4 旋转 = 1024 个 patch，1 - max cos）
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
os.chdir(os.path.join(ROOT, "code/code")); sys.path.insert(0, "model/ImageBind")
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
delta = {k: v.float() for k, v in torch.load(os.path.join(ROOT, "code/ckpt/train_mvtec/pytorch_model.pt"),
                                             map_location="cpu", weights_only=True).items()}
W = [delta["image_decoder.fc.%d.weight" % i] for i in range(4)]
b = [delta["image_decoder.fc.%d.bias" % i] for i in range(4)]

pn = ['{}','flawless {}','perfect {}','unblemished {}','{} without flaw','{} without defect','{} without damage']
pa = ['damaged {}','broken {}','{} with flaw','{} with defect','{} with damage']
pt = ['a photo of a {}.','a photo of the {}.']
tok = SimpleTokenizer(bpe_path="model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
def enc(ss):
    ids = torch.stack([tok(s) for s in ss])
    return model({ModalityType.TEXT: ids})[ModalityType.TEXT][0]
fn = enc([t.format(s.format("bottle")) for s in pn for t in pt]).mean(0, keepdim=True)
fa = enc([t.format(s.format("bottle")) for s in pa for t in pt]).mean(0, keepdim=True)
Ftxt = torch.cat([fn/fn.norm(dim=-1,keepdim=True), fa/fa.norm(dim=-1,keepdim=True)], 0)

tf = transforms.Compose([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])

def patches(path, rot=0):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    if rot: x = torch.rot90(x, rot, dims=[2,3])
    _, snaps = model({ModalityType.VISION: x})[ModalityType.VISION]
    return [s.transpose(0,1)[:,1:,:] for s in snaps]      # 4 × [1,256,1280]

def text_map(path):
    ps = patches(path); probs = []
    for i, p in enumerate(ps):
        q = F.linear(p, W[i], b[i]); q = q/q.norm(dim=-1,keepdim=True)
        probs.append(torch.softmax(100.0*(q@Ftxt.T), dim=-1)[0,:,1].reshape(16,16))
    m16 = torch.stack(probs).mean(0)
    return F.interpolate(m16[None,None], size=224, mode="bilinear", align_corners=True)[0,0].numpy(), m16.numpy()

REF = os.path.join(DATA, "bottle/train/good/000.png")
print("building memory bank from 1 normal image × 4 rotations ...", flush=True)
bank = [[] for _ in range(4)]
for r in range(4):
    for i, p in enumerate(patches(REF, r)):
        bank[i].append(p[0])                      # [256,1280]
bank = [torch.cat(v, 0) for v in bank]            # 4 × [1024,1280]
print("bank size per layer:", [tuple(x.shape) for x in bank], flush=True)

def mem_map(path):
    ps = patches(path); sims = []
    for i, q in enumerate(ps):                    # [1,256,1280]
        c = F.cosine_similarity(q[0][:,None,:], bank[i][None,:,:], dim=-1)   # [256,1024]
        sims.append(c.max(dim=-1)[0].reshape(16,16))
    m16 = 1 - torch.stack(sims).mean(0)
    return F.interpolate(m16[None,None], size=224, mode="bilinear", align_corners=True)[0,0].numpy(), m16.numpy()

cases = {"defect": (os.path.join(DATA,"bottle/test/broken_large/000.png"),
                    os.path.join(DATA,"bottle/ground_truth/broken_large/000_mask.png")),
         "normal": (os.path.join(DATA,"bottle/test/good/000.png"), None)}

def auroc(s, g):
    s, g = s.ravel(), g.ravel().astype(bool)
    r = np.empty(len(s)); r[s.argsort()] = np.arange(1, len(s)+1)
    P, N = g.sum(), (~g).sum()
    return float((r[g].sum() - P*(P+1)/2)/(P*N))

def spearman(a, b):
    ra = np.empty(a.size); ra[a.ravel().argsort()] = np.arange(a.size)
    rb = np.empty(b.size); rb[b.ravel().argsort()] = np.arange(b.size)
    return float(np.corrcoef(ra, rb)[0,1])

stats = {}
for k,(ip,gp) in cases.items():
    print("running", k, flush=True)
    Mt, mt16 = text_map(ip)
    Mm, mm16 = mem_map(ip)
    gt = np.zeros((224,224), bool) if gp is None else (np.asarray(Image.open(gp).convert("L").resize((224,224)))>127)
    s = dict(
        text=dict(min=float(Mt.min()), max=float(Mt.max()), mean=float(Mt.mean()), std=float(Mt.std()),
                  p1=float(np.percentile(Mt,1)), p99=float(np.percentile(Mt,99)),
                  frac_lt_01=float((Mt<0.1).mean()), frac_gt_09=float((Mt>0.9).mean()),
                  frac_mid=float(((Mt>=0.4)&(Mt<=0.6)).mean())),
        mem=dict(min=float(Mm.min()), max=float(Mm.max()), mean=float(Mm.mean()), std=float(Mm.std()),
                 p1=float(np.percentile(Mm,1)), p99=float(np.percentile(Mm,99)),
                 frac_lt_01=float((Mm<0.1).mean()), frac_gt_09=float((Mm>0.9).mean()),
                 frac_mid=float(((Mm>=0.4)&(Mm<=0.6)).mean())),
        spearman_224=spearman(Mt, Mm), spearman_16=spearman(mt16, mm16))
    if gp:
        s["text"]["auroc"] = auroc(Mt, gt); s["mem"]["auroc"] = auroc(Mm, gt)
        s["gt_mean_text"] = float(Mt[gt].mean()); s["gt_mean_mem"] = float(Mm[gt].mean())
        s["bg_mean_text"] = float(Mt[~gt].mean()); s["bg_mean_mem"] = float(Mm[~gt].mean())
    stats[k] = s
    np.savez(os.path.join(OUT, "two_maps_%s.npz" % k), Mt=Mt, Mm=Mm, gt=gt)

print(json.dumps(stats, indent=2))

# ---------------- 图 1：并排
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for r,(k,(ip,gp)) in enumerate(cases.items()):
    d = np.load(os.path.join(OUT, "two_maps_%s.npz" % k)); Mt, Mm = d["Mt"], d["Mm"]
    img = np.asarray(Image.open(ip).convert("RGB").resize((224,224)))
    ax[r,0].imshow(img); ax[r,0].set_title("%s image" % k)
    v = float(np.abs(Mt).max())
    im = ax[r,1].imshow(Mt, cmap="jet", vmin=0, vmax=1)
    ax[r,1].set_title("M_text\nrange [%.3f, %.3f]" % (Mt.min(), Mt.max()))
    plt.colorbar(im, ax=ax[r,1], fraction=0.046)
    im = ax[r,2].imshow(Mm, cmap="jet", vmin=0, vmax=1)
    ax[r,2].set_title("M_mem\nrange [%.3f, %.3f]" % (Mm.min(), Mm.max()))
    plt.colorbar(im, ax=ax[r,2], fraction=0.046)
    if gp:
        g = d["gt"]; ov = img.copy(); ov[g] = (ov[g]*0.4 + np.array([0,255,0])*0.6).astype(np.uint8)
        ax[r,3].imshow(ov); ax[r,3].set_title("ground truth")
    else:
        ax[r,3].axis("off")
    for c in range(4): ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
fig.suptitle("two branches, same image, same colorbar 0~1", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"14_two_maps.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------------- 图 2：直方图 + 散点
d = np.load(os.path.join(OUT, "two_maps_defect.npz")); Mt, Mm = d["Mt"], d["Mm"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].hist(Mt.ravel(), bins=100, alpha=0.65, label="M_text", color="tab:red", log=True)
ax[0].hist(Mm.ravel(), bins=100, alpha=0.65, label="M_mem", color="tab:blue", log=True)
ax[0].set_xlabel("score"); ax[0].set_ylabel("pixel count (log)"); ax[0].legend()
ax[0].set_title("distribution of the two maps (defect image)")
ax[1].scatter(Mt.ravel()[::7], Mm.ravel()[::7], s=2, alpha=0.15)
ax[1].set_xlabel("M_text"); ax[1].set_ylabel("M_mem")
ax[1].set_title("same pixels, two branches\nSpearman = %.3f" % stats["defect"]["spearman_224"])
fig.tight_layout(); fig.savefig(os.path.join(OUT,"15_two_maps_dist.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
json.dump(stats, open(os.path.join(OUT,"two_maps_stats.json"),"w"), indent=2, ensure_ascii=False)
print("saved")
