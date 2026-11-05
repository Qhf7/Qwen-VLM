"""跨零件异常分割实验包的公共设施。

这里只放**与任务无关**的东西：路径、配置加载、按协议的 split 解析、哈希、
结果台账。模型与指标各自在独立模块里。
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# 冻结资产（数据地基，2026-10-20 冻结，之后不再改）
SPLIT_DIR = ROOT / "artifacts" / "splits"
MANIFEST = ROOT / "artifacts" / "manifest.parquet"
MANIFEST_PIXEL = ROOT / "artifacts" / "seg_cross_part" / "manifest_pixel.parquet"
LEDGER = ROOT / "artifacts" / "seg_cross_part" / "results.csv"

LEDGER_FIELDS = [
    "exp_id", "arm", "protocol", "fold", "backbone", "trainable_modules",
    "split_sha256", "seed", "image_size", "max_steps", "best_select_value",
    "val_macro_apro", "val_macro_aupr", "val_macro_auroc",
    "test_macro_apro", "test_macro_aupr", "test_macro_auroc", "test_macro_f1max",
    "gpu_hours", "latency_ms", "ckpt_source", "notes",
]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def with_defaults(cfg: dict) -> dict:
    cfg.setdefault("protocol", {})
    cfg["protocol"].setdefault("split_dir", str(SPLIT_DIR.relative_to(ROOT)))
    cfg["protocol"].setdefault("fold", None)
    cfg.setdefault("input", {})
    cfg["input"].setdefault("image_size", 448)
    cfg["input"].setdefault("num_workers", 8)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("hidden", 256)
    cfg.setdefault("train", {})
    for k, v in (("seed", 20261020), ("batch", 8), ("eval_batch", 8), ("lr", 1e-3),
                 ("weight_decay", 1e-4), ("max_steps", 6000), ("warmup_steps", 100),
                 ("grad_clip", 1.0), ("log_every", 50)):
        cfg["train"].setdefault(k, v)
    cfg["train"].setdefault("eval_steps", [1000, 2000, 3000, 4000, 5000, 6000])
    cfg.setdefault("data", {})
    cfg["data"].setdefault("source_root", "Dataset")
    cfg.setdefault("output", {})
    cfg["output"].setdefault("root", "artifacts/seg_cross_part")
    return cfg


def resolve_splits(cfg: dict, cli_fold: int | None = None) -> dict[str, Path]:
    """A/B：`{prefix}_{split}.parquet`；C：`protocol_c_fold{fold}/{split}.parquet`。"""
    p = cfg["protocol"]
    sdir = ROOT / p["split_dir"]
    if p["type"] == "c":
        fold = cli_fold if cli_fold is not None else p.get("fold")
        if fold is None:
            raise SystemExit("Protocol C 必须提供 --fold（0..3）")
        fold = int(fold)
        cfg["protocol"]["fold"] = fold
        d = sdir / f"protocol_c_fold{fold}"
        paths = {s: d / f"{s}.parquet" for s in ("train", "val", "test")}
    else:
        prefix = p["split_prefix"]
        paths = {s: sdir / f"{prefix}_{s}.parquet" for s in ("train", "val", "test")}
    for s, pp in paths.items():
        if not pp.exists():
            raise SystemExit(f"找不到 split 文件: {pp}")
    return paths


def check_protocol(cfg: dict, splits: dict[str, Path]) -> dict:
    """硬校验：协议名、type、fold、类别交集必须自洽。

    这是为了防止「用 --protocol 改了名字，但配置文件还是 B」这类静默错误——
    那会让你以为在跑四折整类留出，其实跑的是同类新零件。
    """
    import pandas as pd

    p = cfg["protocol"]
    name, typ, fold = p.get("name", ""), p.get("type"), p.get("fold")
    if "unseen" in name and typ != "c":
        raise SystemExit(
            f"协议名 {name} 表示「未见类别」，但配置 type={typ}。\n"
            f"请用 --config seg_cross_part/protocols/protocol_c_unseen_category.yaml")
    if typ == "c" and fold is None:
        raise SystemExit("Protocol C 必须给 --fold 0..3")
    if typ != "c" and fold is not None:
        raise SystemExit(f"--fold 只对 Protocol C 有效，但当前 type={typ}")

    tr = pd.read_parquet(splits["train"], columns=["class_name"])
    va = pd.read_parquet(splits["val"], columns=["class_name"])
    te = pd.read_parquet(splits["test"], columns=["class_name"])
    a, b_, c_ = set(tr.class_name), set(va.class_name), set(te.class_name)
    overlap = a & c_
    if typ == "c" and overlap:
        raise SystemExit(f"Protocol C 的 train/test 类别有交集: {sorted(overlap)[:5]}")
    if typ != "c" and not overlap:
        raise SystemExit("非 C 协议应含相同类别，但 train/test 类别不相交——配置可能错了")
    return {
        "type": typ, "fold": fold,
        "train_classes": len(a), "val_classes": len(b_), "test_classes": len(c_),
        "train_test_class_overlap": len(overlap),
        "train_views": int(len(tr)), "test_views": int(len(te)),
        "train_sample_classes": sorted(a)[:3], "test_sample_classes": sorted(c_)[:3],
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})
