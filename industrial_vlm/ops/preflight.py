"""Exercise native Qwen3-VL NF4/LoRA with a labeled interface-only fixture.

This synthetic fixture checks software and gradients, not industrial accuracy.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

base = Path("/root/autodl-tmp/industrial_vlm")
folder = base / "prepared" / "interface_fixture"
folder.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (640, 480), (160, 160, 160))
draw = ImageDraw.Draw(image)
draw.rectangle((240, 150, 390, 300), fill=(30, 30, 30))
path = folder / "synthetic_interface_fixture.png"
image.save(path)
answer = {"is_anomalous": True, "boxes": [[375, 313, 611, 627]]}
row = {"id": "synthetic-interface-fixture", "source_id": "synthetic-interface-fixture",
       "image": str(path), "category": "synthetic_interface_fixture",
       "is_anomalous": True, "boxes": answer["boxes"],
       "prompt": "Return JSON with is_anomalous and boxes. Coordinates are xyxy integers normalized to 0-1000.",
       "answer": json.dumps(answer)}
(folder / "train.jsonl").write_text(json.dumps(row) + "\n")
print(folder / "train.jsonl")
