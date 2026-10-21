#!/usr/bin/env python3
"""HTML → PNG（Chrome headless + 自动裁白）。用法: python3 render.py in.html out.png"""
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageChops
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
src, out = Path(sys.argv[1]), Path(sys.argv[2])
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1540
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=2", f"--window-size={w},6000",
                f"--screenshot={out}", src.resolve().as_uri()], check=True, capture_output=True)
im = Image.open(out).convert("RGB")
bg = Image.new("RGB", im.size, im.getpixel((2, 2)))
bbox = ImageChops.difference(im, bg).getbbox()
if bbox:
    im.crop((0, 0, bbox[2], bbox[3])).save(out, optimize=True)
print(out, Image.open(out).size)
