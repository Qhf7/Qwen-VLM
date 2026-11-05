import argparse
import csv
import os
import re
import sys

import torch

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
total = correct = invalid = 0
normal_total = normal_correct = 0
anomaly_total = anomaly_correct = 0
for c in CLASS_NAMES:
    paths = rows[c]["Normal"][:k] + rows[c]["Anomaly"][:k]
    if not paths:
        continue
    c_correct = c_total = 0
    c_nt = c_nc = c_at = c_ac = 0
    for rel in paths:
        path = os.path.join(DATA, rel)
        label = 1 if "/Anomaly/" in rel else 0
        inputs = {
            "image_paths": [path], "audio_paths": [], "video_paths": [], "thermal_paths": [],
            "normal_img_paths": [], "modality_embeds": [],
            "prompt": DESC[c] + " Is there any anomaly in the image?",
            "max_tgt_len": args.max_new_tokens, "top_p": 0.1, "temperature": 0.1,
        }
        with torch.no_grad():
            text, _ = model.generate(inputs)
        first = text.split("\n")[0].strip()
        m = re.search(r"\b(yes|no)\b", first.lower())
        pred = 1 if (m and m.group(1) == "yes") else 0 if (m and m.group(1) == "no") else -1
        if pred < 0:
            invalid += 1
        c_total += 1
        total += 1
        if label == 0:
            c_nt += 1; normal_total += 1
        else:
            c_at += 1; anomaly_total += 1
        if pred == label:
            c_correct += 1; correct += 1
            if label == 0:
                c_nc += 1; normal_correct += 1
            else:
                c_ac += 1; anomaly_correct += 1
    print("CLASS %-12s n=%d acc=%.1f normal=%d/%d anomaly=%d/%d" % (
        c, c_total, 100.0 * c_correct / max(1, c_total), c_nc, c_nt, c_ac, c_at), flush=True)

print("MEAN acc=%.1f normal_acc=%.1f anomaly_acc=%.1f invalid=%d n=%d" % (
    100.0 * correct / max(1, total),
    100.0 * normal_correct / max(1, normal_total),
    100.0 * anomaly_correct / max(1, anomaly_total),
    invalid, total), flush=True)
print("DONE", flush=True)
