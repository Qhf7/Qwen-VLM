"""用真实 ImageBind-Huge 权重 + 已发布的 AnomalyGPT(MVTec) delta 权重，
把论文式(1) 从头到尾跑一遍，生成可视化。

仓库里 ckpt/train_mvtec/pytorch_model.pt 就是 image_decoder + prompt_learner，
所以下面得到的 M 是"真模型"的输出，不是模拟。

输出：
  09_real_defect.png / 10_real_normal.png / 11_layers.png / 12_tokens.png
  real_map_stats.json
"""
import os, sys, json
import numpy as np
import torch

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
CODE = os.path.join(ROOT, "code/code")
CKPT = os.path.join(ROOT, "code/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth")
AG_CKPT = os.path.join(ROOT, "code/ckpt/train_mvtec/pytorch_model.pt")
DATA = os.path.join(ROOT, "data/mvtec_anomaly_detection")
OUT = os.path.dirname(os.path.abspath(__file__))

os.chdir(CODE)
sys.path.insert(0, "model/ImageBind")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from models.imagebind_model import imagebind_huge, ModalityType
from models.multimodal_preprocessors import SimpleTokenizer
import torch.nn as nn

torch.set_grad_enabled(False)
torch.set_num_threads(8)

# ---------------- 1. 模型
print("[1/5] building ImageBind-Huge ...", flush=True)
model, _ = imagebind_huge({"layers": [7, 15, 23, 31]})
sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
model.load_state_dict(sd, strict=True)
del sd
model.eval()
print("      ImageBind params = %.1f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

ag = {k: v.float() for k, v in torch.load(AG_CKPT, map_location="cpu", weights_only=True).items()}
W_dec = [ag["image_decoder.fc.%d.weight" % i] for i in range(4)]
b_dec = [ag["image_decoder.fc.%d.bias" % i] for i in range(4)]
print("      AnomalyGPT image_decoder loaded (trained on MVTec)", flush=True)

def build_prompt_learner():
    m = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(16, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(64, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(256, 1024, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(1024, 4096, 5, padding=0),
    )
    s = {k.replace("prompt_learner.meta_net.", ""): v
         for k, v in ag.items() if k.startswith("prompt_learner.meta_net.")}
    m.load_state_dict(s)
    m.eval()
    return m

PL = build_prompt_learner()
BASE = ag["prompt_learner.base_prompts"]

# ---------------- 2. 文本特征（与 openllama.py 完全一致）
prompt_normal = ["{}", "flawless {}", "perfect {}", "unblemished {}",
                 "{} without flaw", "{} without defect", "{} without damage"]
prompt_abnormal = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]
prompt_templates = ["a photo of a {}.", "a photo of the {}."]

def sentences(states, obj):
    return [t.format(s.format(obj)) for s in states for t in prompt_templates]

cls_name = "bottle"
s_norm, s_abn = sentences(prompt_normal, cls_name), sentences(prompt_abnormal, cls_name)
print("[2/5] encoding %d + %d sentences ..." % (len(s_norm), len(s_abn)), flush=True)

tok = SimpleTokenizer(bpe_path="model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")

def encode_texts(ss):
    ids = torch.stack([tok(x) for x in ss])
    return model({ModalityType.TEXT: ids})[ModalityType.TEXT][0]

f_norm = encode_texts(s_norm).mean(0, keepdim=True)
f_abn = encode_texts(s_abn).mean(0, keepdim=True)
f_norm = f_norm / f_norm.norm(dim=-1, keepdim=True)
f_abn = f_abn / f_abn.norm(dim=-1, keepdim=True)
F_text = torch.cat([f_norm, f_abn], dim=0)
print("      F_text:", tuple(F_text.shape), " cos(norm,abn)=%.3f" % float(f_norm @ f_abn.T), flush=True)

# ---------------- 3. 图像
tf = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

cases = {
    "defect": dict(img=os.path.join(DATA, "bottle/test/broken_large/000.png"),
                   gt=os.path.join(DATA, "bottle/ground_truth/broken_large/000_mask.png")),
    "normal": dict(img=os.path.join(DATA, "bottle/test/good/000.png"), gt=None),
}

def compute(path):
    pil = Image.open(path).convert("RGB")
    x = tf(pil).unsqueeze(0)
    out = model({ModalityType.VISION: x})
    _, snaps = out[ModalityType.VISION]
    per_layer = []
    for i, snap in enumerate(snaps):                     # [257,1,1280] 序列在前
        p = snap.transpose(0, 1)[:, 1:, :]               # 丢 CLS → [1,256,1280]
        p = torch.nn.functional.linear(p, W_dec[i], b_dec[i])
        p = p / p.norm(dim=-1, keepdim=True)
        sim = 100.0 * (p @ F_text.T)                     # [1,256,2]
        diff = (sim[0, :, 1] - sim[0, :, 0]).reshape(16, 16).numpy()
        prob = torch.softmax(sim, dim=-1)[0, :, 1].reshape(16, 16).numpy()
        per_layer.append((prob, diff))
    return np.asarray(pil.resize((224, 224))), per_layer

res = {}
for k, cfg in cases.items():
    print("[3/5] running %s ..." % k, flush=True)
    res[k] = compute(cfg["img"])
    print("      done", flush=True)

# ---------------- 4. 画图
def up(a16):
    t = torch.tensor(a16, dtype=torch.float32)[None, None]
    return torch.nn.functional.interpolate(t, size=224, mode="bilinear", align_corners=True)[0, 0].numpy()

def auroc(score, gt):
    s, g = score.ravel(), gt.ravel().astype(bool)
    order = s.argsort()
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = g.sum(), (~g).sum()
    return float((ranks[g].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def draw(key, fname, title):
    img224, per_layer = res[key]
    M16 = np.stack([pl[0] for pl in per_layer], 0).mean(0)
    M224 = up(M16)
    diff32 = per_layer[-1][1]
    gt = cases[key]["gt"]
    cols = 6 if gt else 5
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4.2))
    ax[0].imshow(img224); ax[0].set_title("input 224x224")
    v = float(np.abs(diff32).max())
    im1 = ax[1].imshow(diff32, cmap="coolwarm", vmin=-v, vmax=v)
    ax[1].set_title("16x16 logit diff\n(abn - nor), stage4")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(M16, cmap="jet", vmin=0, vmax=1)
    ax[2].set_title("16x16 P(abnormal)\n4 stages averaged")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    im3 = ax[3].imshow(M224, cmap="jet", vmin=0, vmax=1)
    ax[3].set_title("upsample to 224 = M")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)
    ax[4].imshow(img224)
    ax[4].imshow(M224, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax[4].set_title("overlay")
    if gt:
        g = np.asarray(Image.open(gt).convert("L").resize((224, 224))) > 127
        ax[5].imshow(img224)
        ax[5].imshow(np.ma.masked_where(~g, g), cmap="spring", alpha=0.5)
        ax[5].set_title("ground truth\nmax score = %.3f" % M224.max())
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return M16, M224, diff32, gt

M16_d, M224_d, diff_d, gt_d = draw("defect", "09_real_defect.png",
    "eq.(1) on a real MVTec bottle (break_large): ImageBind-Huge + released AnomalyGPT decoder")
M16_n, M224_n, diff_n, _ = draw("normal", "10_real_normal.png",
    "same pipeline on a normal bottle: nothing should light up")

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    ax[i].imshow(res["defect"][1][i][0], cmap="jet", vmin=0, vmax=1)
    ax[i].set_title("stage %d (block %d)" % (i + 1, [8, 16, 24, 32][i]))
    ax[i].set_xticks([]); ax[i].set_yticks([])
fig.suptitle("the same image, per-stage P(abnormal) at 16x16", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "11_layers.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

M16 = np.stack([pl[0] for pl in res["defect"][1]], 0).mean(0)
M_t = torch.tensor(up(M16), dtype=torch.float32)[None, None]
tok_map = PL(M_t).reshape(1, 4096, 9).transpose(-2, -1)[0]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].imshow(M224_d, cmap="jet", vmin=0, vmax=1); ax[0].set_title("M (input of prompt learner)")
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].bar(np.arange(1, 10), tok_map.norm(dim=-1).numpy(), color="tab:red")
ax[1].set_title("L2 norm of the 9 map tokens"); ax[1].set_xlabel("token index")
ax[2].bar(np.arange(1, 10), torch.nn.functional.cosine_similarity(tok_map, BASE, dim=-1).numpy(), color="tab:blue")
ax[2].set_title("cos(map token i, base prompt i)"); ax[2].set_xlabel("token index"); ax[2].set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "12_tokens.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------------- 5. 统计
stats = {
    "class": cls_name,
    "cos_between_normal_and_abnormal_text": float(f_norm @ f_abn.T),
    "defect": {
        "logit_diff_min_max": [float(diff_d.min()), float(diff_d.max())],
        "prob16_min_max_mean": [float(M16_d.min()), float(M16_d.max()), float(M16_d.mean())],
        "pixels_prob_gt_0.9": float((M224_d > 0.9).mean()),
        "pixels_prob_lt_0.1": float((M224_d < 0.1).mean()),
        "max_score": float(M224_d.max()),
    },
    "normal": {
        "prob16_min_max_mean": [float(M16_n.min()), float(M16_n.max()), float(M16_n.mean())],
        "pixels_prob_gt_0.9": float((M224_n > 0.9).mean()),
        "pixels_prob_lt_0.1": float((M224_n < 0.1).mean()),
        "max_score": float(M224_n.max()),
    },
}
if gt_d:
    g = np.asarray(Image.open(gt_d).convert("L").resize((224, 224))) > 127
    stats["defect"]["gt_area_ratio"] = float(g.mean())
    stats["defect"]["single_image_pixel_auroc"] = auroc(M224_d, g)
    stats["defect"]["recall_at_0.9"] = float((M224_d[g] > 0.9).mean())
    stats["defect"]["precision_at_0.9"] = float(g[M224_d > 0.9].mean()) if (M224_d > 0.9).any() else 0.0

with open(os.path.join(OUT, "real_map_stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("[5/5] done", flush=True)
