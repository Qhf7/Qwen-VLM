"""Pixel AUROC/AP by category and image AUROC from max(M)/p99(M). No LLM."""
import argparse, hashlib, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT/code/code/model")
from ImageBind import imagebind_model, data
from ImageBind.models.imagebind_model import ModalityType
from AnomalyGPT_models import LinearLayer
sys.path.insert(0, "/root/autodl-tmp/AnomalyGPT")
from split_visa import splits
from visa_pipeline import (PIPELINE_VERSION, load_image_tensor, load_mask_tensor,
                           mask_path_for_image, auroc, average_precision)

REPO="/root/autodl-tmp/AnomalyGPT"
data.BPE_PATH=os.path.join(REPO,"code/code/model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
PN=['{}','flawless {}','perfect {}','unblemished {}','{} without flaw','{} without defect','{} without damage']
PA=['damaged {}','broken {}','{} with flaw','{} with defect','{} with damage']
TP=['a photo of a {}.','a photo of the {}.']

def anchors(enc, cls):
    def build(states):
        sents=[t.format(s.format(cls)) for s in states for t in TP]
        toks=data.load_and_transform_text(sents,"cuda")
        with torch.no_grad(): e=enc({ModalityType.TEXT:toks})[ModalityType.TEXT][0]
        e=e.reshape(1,len(sents),1024).mean(1,keepdim=True); return e/e.norm(dim=-1,keepdim=True)
    return torch.cat([build(PN),build(PA)],dim=1)

def ap(y,s):
    """Compatibility alias for callers of the original helper."""
    return average_precision(y, s)

def load_decoder(decoder, checkpoint):
    for name in ("image_decoder", "decoder"):
        if isinstance(checkpoint.get(name), dict):
            decoder.load_state_dict(checkpoint[name], strict=True)
            return name
        prefix = name + "."
        state = {k[len(prefix):]: v for k, v in checkpoint.items()
                 if k.startswith(prefix)}
        if state:
            decoder.load_state_dict(state, strict=True)
            return prefix
    raise KeyError("Checkpoint contains no image_decoder / decoder weights")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.path.join(REPO, "runs/native_clean/best.pt"))
    parser.add_argument("--out", default=os.path.join(REPO, "runs/eval_pixel_corrected"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    enc,_=imagebind_model.imagebind_huge({"layers":[7,15,23,31]})
    enc.load_state_dict(torch.load(os.path.join(REPO,"pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"),map_location="cpu"),strict=True)
    enc.half().cuda().eval()
    dec=LinearLayer(1280,1024,4).float().cuda().eval()
    ck=torch.load(args.ckpt,map_location="cpu")
    loaded_from = load_decoder(dec, ck)
    print("[decoder] strict load %s from %s" % (loaded_from, args.ckpt),flush=True)
    tr,va,te=splits()
    items = va if args.split == "val" else te
    if not items:
        raise ValueError("No images selected for %s" % args.split)
    class_anchors = {c: anchors(enc, c.replace("_", " ")) for c in set(x[1] for x in items)}
    per_cat={}; img_y=[]; img_s=[]; img_max=[]; fg=0; tot=0
    image_predictions = []
    for i,(p,cls,lab) in enumerate(items):
        x=load_image_tensor(p,device="cuda").half()
        with torch.no_grad():
            emb=enc({ModalityType.VISION:x}); A=class_anchors[cls]
            maps=[]
            for t in dec([v.float() for v in emb['vision'][1]]):
                t=t/t.norm(dim=-1,keepdim=True)
                lg=100.0*t@A.transpose(-2,-1).float()
                B,L,C=lg.shape; H=int(L**0.5)
                up=F.interpolate(lg.permute(0,2,1).view(B,C,H,H),size=224,mode='bilinear',align_corners=True)
                maps.append(torch.softmax(up,dim=1)[:,1])
            M=torch.stack(maps).mean(0)[0].cpu().numpy()
        mp=mask_path_for_image(p,lab)
        gt=load_mask_tensor(mp,device="cpu").detach().cpu().numpy().reshape(224,224)
        per_cat.setdefault(cls,{"y":[],"s":[]})
        per_cat[cls]["y"].append(gt.ravel()); per_cat[cls]["s"].append(M.ravel())
        img_y.append(lab); img_s.append(float(np.percentile(M,99)))
        img_max.append(float(M.max()))
        image_predictions.append({"image":p, "class":cls, "label":lab,
                                  "mask":mp, "map_max":img_max[-1],
                                  "map99":img_s[-1]})
        fg+=gt.sum(); tot+=gt.size
        if (i+1)%100==0: print("  %d/%d"%(i+1,len(items)),flush=True)
    aucs, aps, w = [], [], []
    category_metrics = {}
    for c,d in per_cat.items():
        y=np.concatenate(d["y"]); s=np.concatenate(d["s"])
        foreground = int(y.sum())
        if not 0 < foreground < y.size:
            category_metrics[c] = {"pixel_auroc":None,
                                   "pixel_ap":average_precision(y,s) if foreground else None,
                                   "foreground_pixels":foreground, "n_pixels":int(y.size)}
            continue
        aucs.append(auroc(y,s)); aps.append(average_precision(y,s)); w.append(int(y.sum()))
        category_metrics[c] = {"pixel_auroc":aucs[-1], "pixel_ap":aps[-1],
                               "foreground_pixels":w[-1], "n_pixels":int(y.size)}
    if not aucs:
        raise ValueError("No category contains foreground pixels")
    res={"pixel_auroc_macro":float(np.mean(aucs)),"pixel_ap_macro":float(np.mean(aps)),
         "pixel_auroc_weighted":float(np.average(aucs,weights=w)),
         "image_auroc_from_map99":auroc(np.array(img_y),np.array(img_s)),
         "image_auroc_from_map_max":auroc(img_y,img_max),
         "image_ap_from_map_max":average_precision(img_y,img_max),
         "foreground_ratio":float(fg/tot),"n_categories":len(aucs),
         "n_images":len(items), "split":args.split,
         "training_protocol":"supervised_custom_60_20_20",
         "pipeline_version":PIPELINE_VERSION,
         "checkpoint":os.path.abspath(args.ckpt),
         "checkpoint_step":ck.get("step"),
         "checkpoint_pipeline_version":ck.get("pipeline_version"),
         "source_sha256":hashlib.sha256("\n".join(p for p,_,_ in items).encode()).hexdigest(),
         "per_category":category_metrics}
    os.makedirs(args.out,exist_ok=True)
    with open(os.path.join(args.out,"metrics.json"),"w") as f:
        json.dump(res,f,indent=2)
    with open(os.path.join(args.out,"predictions.jsonl"),"w") as f:
        for row in image_predictions:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps(res,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
