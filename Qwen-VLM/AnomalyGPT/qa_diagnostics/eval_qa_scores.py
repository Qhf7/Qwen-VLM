import argparse
import csv
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = "/root/autodl-tmp/AnomalyGPT"
CODE = os.path.join(ROOT, "code", "code")
os.chdir(CODE)
sys.path.insert(0, CODE)
DATA = os.path.join(ROOT, "code", "data", "VisA")

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
ap.add_argument("--backend", choices=["qwen", "vicuna"], required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--per_class", type=int, default=20)
ap.add_argument("--max_new_tokens", type=int, default=16)
args = ap.parse_args()

if args.backend == "qwen":
    from model.openllama_qwen3vl import OpenLLAMAPEFTModel
    model_args = {
        "imagebind_ckpt_path": f"{ROOT}/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth",
        "vicuna_ckpt_path": "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct",
        "delta_ckpt_path": f"{ROOT}/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt",
        "stage": 2, "max_tgt_len": args.max_new_tokens, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    }
else:
    from model.openllama import OpenLLAMAPEFTModel
    model_args = {
        "imagebind_ckpt_path": f"{ROOT}/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth",
        "vicuna_ckpt_path": f"{ROOT}/pretrained_ckpt/vicuna_ckpt/vicuna-7b-v1.1",
        "delta_ckpt_path": f"{ROOT}/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt",
        "stage": 2, "max_tgt_len": args.max_new_tokens, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    }

print("[load] backend", args.backend, flush=True)
model = OpenLLAMAPEFTModel(**model_args)
d = torch.load(args.ckpt, map_location="cpu")
missing, unexpected = model.load_state_dict(d, strict=False)
print("[load] ckpt keys=%d missing=%d unexpected=%d" % (len(d), len(missing), len(unexpected)), flush=True)
if args.backend == "vicuna":
    model = model.eval().half().cuda()
else:
    model = model.eval()
print("[load] ok", flush=True)

rows = {c: {"Normal": [], "Anomaly": []} for c in CLASS_NAMES}
with open(os.path.join(DATA, "split_csv", "1cls.csv")) as f:
    for row in csv.reader(f):
        if row[0] in CLASS_NAMES and row[1] == "test":
            kind = "Anomaly" if "/Anomaly/" in row[3] else "Normal"
            rows[row[0]][kind].append(row[3])

k = max(1, args.per_class // 2)
records = []
for c in CLASS_NAMES:
    for rel in rows[c]["Normal"][:k] + rows[c]["Anomaly"][:k]:
        path = os.path.join(DATA, rel)
        label = 1 if "/Anomaly/" in rel else 0
        inputs = {
            "image_paths": [path], "audio_paths": [], "video_paths": [], "thermal_paths": [],
            "normal_img_paths": [], "modality_embeds": [],
            "prompt": DESC[c] + " Is there any anomaly in the image?",
            "max_tgt_len": args.max_new_tokens, "top_p": 0.1, "temperature": 0.1,
            "return_score": True,
        }
        with torch.no_grad():
            text, amap, score = model.generate(inputs)
        first = text.split("\n")[0].strip()
        records.append((c, label, float(score), first))
        print("ITEM %-12s label=%d score=%.4f text=%r" % (c, label, float(score), first[:70]), flush=True)

labels = np.array([r[1] for r in records])
scores = np.array([r[2] for r in records])
auc = roc_auc_score(labels, scores)
print("OVERALL auc=%.4f n=%d normal_mean=%.4f anomaly_mean=%.4f" % (
    auc, len(records), scores[labels == 0].mean(), scores[labels == 1].mean()), flush=True)
acc0 = ((scores > 0).astype(int) == labels).mean()
print("ACC_AT_0 %.4f" % acc0, flush=True)

best = (0.0, 0.0)
for th in sorted(set(scores.tolist())):
    pred = (scores >= th).astype(int)
    tpr = pred[labels == 1].mean() if (labels == 1).any() else 0
    tnr = 1 - pred[labels == 0].mean() if (labels == 0).any() else 0
    bal = 0.5 * (tpr + tnr)
    if bal > best[1]:
        best = (th, bal)
print("BEST_BAL th=%.4f bal=%.4f" % best, flush=True)

with open(f"{ROOT}/logs/qa_scores_{args.backend}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "label", "score", "text"])
    w.writerows(records)
print("DONE", flush=True)
