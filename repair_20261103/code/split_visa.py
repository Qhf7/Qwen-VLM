"""按官方 1cls.csv 划分：训练用 official train 正常图 + test 的 60%，
   选模型用 test 的 60-80%，最终测试用 test 的 80-100%（完全不参与训练与选型）。"""
import csv, hashlib, os
VISA = "/root/autodl-tmp/industrial_vlm/datasets/VisA"
def bucket(p): return int(hashlib.md5(p.encode()).hexdigest(), 16) % 100
def rows():
    out = []
    with open(os.path.join(VISA, "split_csv/1cls.csv")) as f:
        for r in csv.DictReader(f):
            p = os.path.join(VISA, r["image"])
            if os.path.exists(p):
                out.append((p, r["object"], 1 if r["label"] == "anomaly" else 0, r["split"]))
    return out
def splits():
    R = rows()
    tr = [(p,c,l) for p,c,l,s in R if s == "train" or (s == "test" and bucket(p) < 60)]
    va = [(p,c,l) for p,c,l,s in R if s == "test" and 60 <= bucket(p) < 80]
    te = [(p,c,l) for p,c,l,s in R if s == "test" and bucket(p) >= 80]
    return tr, va, te
if __name__ == "__main__":
    tr, va, te = splits()
    for n, s in (("train",tr), ("val",va), ("test",te)):
        print("%-5s %5d  (%d 异常 / %d 正常)" % (n, len(s), sum(x[2] for x in s), sum(1-x[2] for x in s)))
    ps = lambda s: set(x[0] for x in s)
    print("train∩test:", len(ps(tr) & ps(te)), " train∩val:", len(ps(tr) & ps(va)), " val∩test:", len(ps(va) & ps(te)))
