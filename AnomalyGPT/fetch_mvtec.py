import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "foersben/mvtec-ad"
DEST = "AnomalyGPT/data/mvtec_anomaly_detection"
API = f"https://huggingface.co/api/datasets/{REPO}"
UA = {"User-Agent": "Mozilla/5.0 (research dataset fetch)"}

files = json.load(urllib.request.urlopen(urllib.request.Request(API, headers=UA)))["siblings"]
files = [f["rfilename"] for f in files if not f["rfilename"].startswith(".") and f["rfilename"] != "README.md"]
print("total files:", len(files), flush=True)

def fetch(path):
    out = os.path.join(DEST, path)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(out + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk: break
                    f.write(chunk)
            os.replace(out + ".part", out)
            return os.path.getsize(out)
        except Exception as e:
            time.sleep(2 + attempt * 3)
    print("FAILED", path, flush=True)
    return -1

done = [0]; bytes_ = [0]
def run(p):
    n = fetch(p)
    done[0] += 1
    if n > 0: bytes_[0] += n
    if done[0] % 500 == 0:
        print(f"{done[0]}/{len(files)} files, {bytes_[0]/1e6:.0f} MB", flush=True)

with ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(run, files))
print("DONE", done[0], f"{bytes_[0]/1e6:.0f} MB", flush=True)
