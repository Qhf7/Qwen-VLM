"""Render fixed, deterministic test examples using the delivered decoder."""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = "/root/autodl-tmp/AnomalyGPT"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "code/code/model"))
from ImageBind import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
from dec_big import BigDec, patch_features
from split_visa import splits
from visa_pipeline import (PIPELINE_VERSION, load_image_pil, load_image_tensor,
                           load_mask_tensor, mask_path_for_image)


def main():
    output = os.path.join(REPO, "runs/repair_visuals_20261103")
    os.makedirs(output, exist_ok=True)
    test = splits()[2]
    selected = []
    # Median foreground area per chosen category avoids cherry-picking scores.
    for category in ("candle", "capsules", "pcb3"):
        candidates = []
        for row in test:
            p, cls, label = row
            if cls == category and label:
                area = float(load_mask_tensor(mask_path_for_image(p, label), "cpu").sum())
                candidates.append((area, p, row))
        candidates.sort()
        selected.append(candidates[len(candidates)//2][2])
    selected.append(next(row for row in test if row[1] == "candle" and row[2] == 0))
    enc, _ = imagebind_model.imagebind_huge({"layers": [7, 15, 23, 31]})
    enc.load_state_dict(torch.load(os.path.join(REPO, "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth"), map_location="cpu", weights_only=True))
    enc.half().cuda().eval()
    dec = BigDec().cuda().eval()
    ck = torch.load(os.path.join(REPO, "runs/dec_big_fixed_s42/best.pt"), map_location="cpu", weights_only=True)
    assert ck["pipeline_version"] == PIPELINE_VERSION
    dec.load_state_dict(ck["dec"], strict=True)
    canvas = Image.new("RGB", (1056, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((16, 10), "Aligned test examples | probability colors use fixed 0-1 scale", fill="black", font=font)
    for col, title in enumerate(("RGB image", "Binary ground truth", "Anomaly probability", "Probability overlay")):
        draw.text((16+col*260, 44), title, fill="black", font=font)
    rows = []
    with torch.no_grad():
        for row_index, (p, cls, label) in enumerate(selected):
            rgb = load_image_pil(p)
            x = load_image_tensor(p).half()
            raw = enc({ModalityType.VISION: x})["vision"][1]
            prob = dec(patch_features(raw, 1)).sigmoid()[0, 0].cpu().numpy()
            gt = load_mask_tensor(mask_path_for_image(p, label), "cpu")[0].numpy()
            # Simple fixed blue-cyan-yellow-red palette; no per-image scaling.
            anchors = np.array([[0, 20, 90], [0, 180, 220], [255, 230, 0], [220, 0, 20]], dtype=float)
            v = np.clip(prob, 0, 1)*3
            low = np.minimum(v.astype(int), 2)
            fraction = (v-low)[..., None]
            heat = (anchors[low]*(1-fraction)+anchors[low+1]*fraction).astype(np.uint8)
            overlay = (np.asarray(rgb)*0.6+heat*0.4).astype(np.uint8)
            y = 80+row_index*260
            draw.text((16, y), "%s | %s | %s" % (cls, "anomaly" if label else "normal", os.path.basename(p)), fill="black", font=font)
            images = [rgb, Image.fromarray((gt*255).astype(np.uint8)).convert("RGB"), Image.fromarray(heat), Image.fromarray(overlay)]
            for col, image in enumerate(images):
                canvas.paste(image, (16+col*260, y+25))
            rows.append({"image":p, "class":cls, "label":label, "foreground_pixels":int(gt.sum()),
                         "probability_min":float(prob.min()), "probability_max":float(prob.max())})
    canvas.save(os.path.join(output, "aligned_test_examples.png"))
    with open(os.path.join(output, "examples.json"), "w") as f:
        json.dump({"selection":"median foreground area among test anomalies for three predefined categories plus first normal candle",
                   "pipeline_version":PIPELINE_VERSION, "checkpoint_step":ck["step"], "examples":rows}, f, indent=2)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
