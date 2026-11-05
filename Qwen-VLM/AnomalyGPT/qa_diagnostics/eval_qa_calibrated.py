import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = "/root/autodl-tmp/AnomalyGPT"
CODE = os.path.join(ROOT, "code", "code")
os.chdir(CODE)
sys.path.insert(0, CODE)
DATA = os.path.join(ROOT, "code", "data", "VisA")
from model.openllama import OpenLLAMAPEFTModel

CLASS_NAMES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
               "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
DESC = {
    "candle": "This is a photo of 4 candles for anomaly detection, every candle should be round, without any damage, flaw, defect, scratch, hole or broken part.",
    "capsules": "This is a photo of many small capsules for anomaly detection, every capsule is green, should be without any damage, flaw, defect, scratch, hole or broken part.",
    "cashew": "This is a photo of a cashew for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "chewinggum": "This is a photo of a chewinggom for anomaly detection, which should be white, without any damage, flaw, defect, scratch, hole or broken part.",
    "fryum": "This is a photo of a fryum for anomaly detection on green background, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "macaroni1": "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "macaroni2": "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb1": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb2": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb3": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb4": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pipe_fryum": "This is a photo of a pipe fryum for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--per_class", type=int, default=20)
ap.add_argument("--n_ref", type=int, default=5)
ap.add_argument("--margin", type=float, default=1.02)
args = ap.parse_args()

model_args = {
    "imagebind_ckpt_path": f"{ROOT}/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth",
    "vicuna_ckpt_path": f"{ROOT}/pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1",
    "delta_ckpt_path": f"{ROOT}/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt",
    "stage": 2, "max_tgt_len": 16, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
}
model = OpenLLAMAPEFTModel(**model_args)
d = torch.load(args.ckpt, map_location="cpu")
missing, unexpected = model.load_state_dict(d, strict=False)
print("[load] keys=%d missing=%d unexpected=%d" % (len(d), len(missing), len(unexpected)), flush=True)
model = model.eval().half().cuda()

rows = {c: {"train_normal": [], "test_normal": [], "test_anomaly": []} for c in CLASS_NAMES}
with open(os.path.join(DATA, "split_csv", "1cls.csv")) as f:
    for row in csv.reader(f):
        c = row[0]
        if c not in CLASS_NAMES:
            continue
        if row[1] == "train" and "/Normal/" in row[3]:
            rows[c]["train_normal"].append(row[3])
        elif row[1] == "test":
            if "/Anomaly/" in row[3]:
                rows[c]["test_anomaly"].append(row[3])
            else:
                rows[c]["test_normal"].append(row[3])


def map_max(image_rel, normal_rels):
    inputs = {
        "image_paths": [os.path.join(DATA, image_rel)],
        "audio_paths": [], "video_paths": [], "thermal_paths": [],
        "normal_img_paths": [os.path.join(DATA, r) for r in normal_rels],
        "modality_embeds": [],
        "prompt": DESC[image_rel.split("/")[0]] + " Is there any anomaly in the image?",
    }
    with torch.no_grad():
        _, amap = model.extract_multimodal_feature(inputs, False)
    return float(amap.float().max())


k = max(1, args.per_class // 2)
tp = tn = fp = fn = 0
for c in CLASS_NAMES:
    refs = rows[c]["train_normal"][:args.n_ref]
    ref_scores = []
    for i in range(len(refs)):
        # leave-one-out normal reference: query ref i against another normal ref
        ref_scores.append(map_max(refs[i], [refs[(i + 1) % len(refs)]]))
    thr = max(ref_scores) * args.margin
    test_paths = rows[c]["test_normal"][:k] + rows[c]["test_anomaly"][:k]
    c_tp = c_tn = c_fp = c_fn = 0
    for rel in test_paths:
        label = 1 if "/Anomaly/" in rel else 0
        score = map_max(rel, refs)
        pred = int(score > thr)
        if label == 1 and pred == 1:
            c_tp += 1
        elif label == 1:
            c_fn += 1
        elif pred == 0:
            c_tn += 1
        else:
            c_fp += 1
    tp += c_tp; tn += c_tn; fp += c_fp; fn += c_fn
    tpr = c_tp / max(1, c_tp + c_fn)
    tnr = c_tn / max(1, c_tn + c_fp)
    print("CLASS %-12s thr=%.4f ref_max=%.4f TPR=%.3f TNR=%.3f tp=%d fn=%d fp=%d tn=%d" % (
        c, thr, max(ref_scores), tpr, tnr, c_tp, c_fn, c_fp, c_tn), flush=True)

tpr = tp / max(1, tp + fn); tnr = tn / max(1, tn + fp)
print("MEAN TPR=%.3f TNR=%.3f bal=%.3f acc=%.3f tp=%d fn=%d fp=%d tn=%d n=%d" % (
    tpr, tnr, 0.5 * (tpr + tnr), (tp + tn) / max(1, tp + tn + fp + fn),
    tp, fn, fp, tn, tp + fn + fp + tn), flush=True)
print("DONE", flush=True)
