"""阶段 1：用 ImageBind + image_decoder + prompt_learner，产出
   E_img [1,4096] 与 18 个提示 token [1,18,4096]，存成 npz，供阶段 2 用。"""
import os, sys, json
import numpy as np, torch
import torch.nn as nn

ROOT = "/Users/qiaohongfei/Desktop/实习/阿里云天池比赛vlm/AnomalyGPT"
os.chdir(os.path.join(ROOT, "code/code")); sys.path.insert(0, "model/ImageBind")
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
pg = {k: v.float() for k, v in torch.load(os.path.join(ROOT, "code/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt"),
                                          map_location="cpu", weights_only=True).items()}
W_proj, b_proj = pg["llama_proj.weight"], pg["llama_proj.bias"]     # [4096,1024]
W = [delta["image_decoder.fc.%d.weight" % i] for i in range(4)]
b = [delta["image_decoder.fc.%d.bias" % i] for i in range(4)]

pl = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
    nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
    nn.Conv2d(16, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
    nn.Conv2d(64, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
    nn.Conv2d(256, 1024, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
    nn.Conv2d(1024, 4096, 5, padding=0))
pl.load_state_dict({k.replace("prompt_learner.meta_net.", ""): v for k, v in delta.items()
                    if k.startswith("prompt_learner.meta_net.")})
pl.eval()
BASE = delta["prompt_learner.base_prompts"]        # [9,4096]

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

DESC = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
cases = {"defect": os.path.join(DATA,"bottle/test/broken_large/000.png"),
         "normal": os.path.join(DATA,"bottle/test/good/000.png")}

for k, p in cases.items():
    x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
    cls_emb, snaps = model({ModalityType.VISION: x})[ModalityType.VISION]   # cls [1,1024]
    probs = []
    for i, s in enumerate(snaps):
        q = s.transpose(0,1)[:,1:,:]
        q = torch.nn.functional.linear(q, W[i], b[i]); q = q/q.norm(dim=-1,keepdim=True)
        probs.append(torch.softmax(100.0*(q@F.T), dim=-1)[0,:,1].reshape(16,16))
    M16 = torch.stack(probs).mean(0)[None,None]
    M = torch.nn.functional.interpolate(M16, size=224, mode="bilinear", align_corners=True)  # [1,1,224,224]
    E_img = torch.nn.functional.linear(cls_emb, W_proj, b_proj)              # [1,4096]
    mp = pl(M).reshape(1, 4096, 9).transpose(-2,-1)                          # [1,9,4096]
    prompts = torch.cat([BASE.unsqueeze(0), mp], dim=1)                      # [1,18,4096]
    np.savez(os.path.join(OUT, "embeds_%s.npz" % k),
             E_img=E_img.numpy(), prompts=prompts.numpy(), M=M[0,0].numpy())
    print(k, "E_img", tuple(E_img.shape), "prompts", tuple(prompts.shape),
          "maxM=%.4f" % float(M.max()))

json.dump({"desc": DESC, "question": "Is there any anomaly in the image?"},
          open(os.path.join(OUT, "prompt_text.json"), "w"), ensure_ascii=False, indent=2)
print("stage1 done")
