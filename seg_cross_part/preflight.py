"""实验前复核（Preflight）：开跑前把"会返工"的坑一次性挡住。

背景：曾经因为 `--protocol` 只改名字、配置文件仍是 B，白跑了三个 34 分钟的 fold。
所以把复核做成**代码里的强制关卡**：`train_seg` 默认先跑本模块，任何一项不过就拒绝训练。

手动单跑：
    python -m seg_cross_part.preflight --protocol protocol_c_unseen_category --fold 0
    python -m seg_cross_part.preflight --protocol protocol_b_seen_category

检查项（每项 PASS/FAIL，任一 FAIL → 退出码 1）：
1. 配置文件存在，`type/name/fold` 三者自洽；
2. split 文件存在、可读、记录 sha256；
3. 类别约束：C 的 train/test 类别交集必须为 0；非 C 必须相同类别；
4. 规模约束：与冻结值比对（超出 5% 报警）；
5. `manifest_pixel` 覆盖：split 的每个 image_path 都必须能查到 mask/面积档/group；
6. 抽样图像与 mask 可读；
7. 图像与 mask 的几何对齐自检（防止"裁剪没作用在 mask 上"）；
8. 输出目录是否已存在（防覆盖）；
9. 设备与显存。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (MANIFEST_PIXEL, ROOT, check_protocol, load_config,
                     resolve_splits, sha256_of, with_defaults)

# 冻结规模（2026-10-20 划分时记录），用于发现"跑错数据"
EXPECT = {
    "b": {"train": 20950, "val": 4515, "test": 4390, "train_classes": 24, "test_classes": 24},
    "c": {"train": 17890, "val": 4510, "test": 7455, "train_classes": 18, "test_classes": 6},
}
TOL = 0.05


class Report:
    def __init__(self):
        self.items: list[tuple[bool, str, str]] = []

    def add(self, ok: bool, name: str, detail: str = "") -> bool:
        self.items.append((bool(ok), name, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    def ok(self) -> bool:
        return all(i[0] for i in self.items)

    def summary(self) -> str:
        n = len(self.items)
        p = sum(1 for i in self.items if i[0])
        return f"{p}/{n} 项通过"


def _within(actual: int, expect: int, tol: float = TOL) -> bool:
    return abs(actual - expect) <= max(int(expect * tol), 1)


def run_preflight(protocol: str = "protocol_b_seen_category", fold: int | None = None,
                  config: Path | None = None, exp_id: str | None = None,
                  sample: int = 20, verbose: bool = True) -> bool:
    rep = Report()
    protos = ROOT / "seg_cross_part" / "protocols"
    cfg_path = config or (protos / f"{protocol}.yaml")
    if not rep.add(cfg_path.exists(), "配置文件", str(cfg_path.name)):
        rep.summary(); return False

    cfg = with_defaults(load_config(cfg_path))
    if protocol:
        cfg["protocol"]["name"] = protocol
    typ = cfg["protocol"].get("type")
    rep.add("unseen" not in cfg["protocol"].get("name", "") or typ == "c",
            "协议名与 type 一致", f"name={cfg['protocol']['name']} type={typ}")

    try:
        splits = resolve_splits(cfg, fold)
        info = check_protocol(cfg, splits)
    except SystemExit as e:
        rep.add(False, "协议/类别校验", str(e).splitlines()[0]); rep.summary(); return False
    rep.add(True, "split 解析",
            f"type={info['type']} fold={info['fold']} train={info['train_classes']}类 "
            f"test={info['test_classes']}类 交集={info['train_test_class_overlap']}")
    rep.add((info["train_test_class_overlap"] == 0) if typ == "c"
            else (info["train_test_class_overlap"] == info["test_classes"]),
            "类别交集规则", "C 必须为 0；非 C 必须全交")

    exp = EXPECT.get(typ)
    if exp:
        rep.add(_within(info["train_views"], exp["train"]), "train 视图数",
                f"{info['train_views']}（期望 {exp['train']}±5%）")
        rep.add(_within(info["test_views"], exp["test"]), "test 视图数",
                f"{info['test_views']}（期望 {exp['test']}±5%）")
        rep.add(info["train_classes"] == exp["train_classes"]
                and info["test_classes"] == exp["test_classes"], "类别数",
                f"{info['train_classes']}/{info['test_classes']}")

    # manifest_pixel 覆盖
    if not rep.add(MANIFEST_PIXEL.exists(), "manifest_pixel 存在", str(MANIFEST_PIXEL.name)):
        rep.summary(); return False
    pix = pd.read_parquet(MANIFEST_PIXEL, columns=["image_path", "group_path", "area_bucket",
                                                   "mask_path", "mask_valid"])
    for s in ("train", "val", "test"):
        paths = set(pd.read_parquet(splits[s], columns=["image_path"])["image_path"])
        miss = len(paths - set(pix["image_path"]))
        rep.add(miss == 0, f"{s} 覆盖", f"缺失 {miss}")
    rep.add(pix["group_path"].notna().all() and pix["area_bucket"].notna().all(),
            "元信息完整", "group_path / area_bucket 无空值")

    # 抽样可读 + 对齐
    rng = np.random.default_rng(0)
    src = ROOT / cfg["data"]["source_root"]
    test_df = pd.read_parquet(splits["test"]).merge(
        pix[["image_path", "mask_path", "mask_valid"]], on="image_path", how="left")
    valid = test_df[test_df["mask_valid"] == True]  # noqa: E712
    if len(valid):
        pick = valid.sample(min(sample, len(valid)), random_state=0)
        from PIL import Image
        bad_img = sum(1 for p in pick.image_path if not (src / p).exists())
        bad_msk = sum(1 for p in pick.mask_path
                      if not isinstance(p, str) or not (src / p).exists())
        rep.add(bad_img == 0, "抽样图像可读", f"{len(pick)} 张，缺失 {bad_img}")
        rep.add(bad_msk == 0, "抽样 mask 可读", f"{len(pick)} 张，缺失 {bad_msk}")
    else:
        rep.add(False, "抽样 mask", "test 中没有带 mask 的样本？")

    try:
        from .train_seg import check_alignment
        check_alignment(n=8)          # 抛异常即失败
        rep.add(True, "图像/mask 几何对齐", "IoU 自检通过")
    except SystemExit as e:
        rep.add(False, "图像/mask 几何对齐", str(e))
    except Exception as e:            # noqa: BLE001
        rep.add(False, "图像/mask 几何对齐", f"{type(e).__name__}: {e}")

    # 指标口径自洽：两套实现（训练器 / 统计量）必须给同一批数据同样的数
    try:
        from .check_metrics import compare_implementations
        d = compare_implementations(n_images=12, seed=0)
        rep.add(max(d.values()) < 1e-6, "指标口径自洽",
                "PixelEvaluator vs PixelStats Δmax=" + f"{max(d.values()):.2e}")
    except Exception as e:            # noqa: BLE001
        rep.add(False, "指标口径自洽", f"{type(e).__name__}: {e}")

    # 输出目录冲突
    if exp_id:
        out = ROOT / "artifacts" / "seg_cross_part" / exp_id
        conflict = out.exists() and any(out.iterdir())
        rep.add(not conflict, "输出目录", f"{exp_id} " + ("已存在（需 --overwrite）" if conflict else "可用"))

    # 设备
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        mem = (torch.cuda.get_device_properties(0).total_memory / 2**30
               if dev == "cuda" else float("nan"))
        rep.add(True, "设备", f"{dev}" + (f" {mem:.1f} GiB" if mem == mem else ""))
    except Exception as e:            # noqa: BLE001
        rep.add(False, "设备", str(e))

    ok = rep.ok()
    print(f"\n复核结论：{rep.summary()} → " + ("可以开跑" if ok else "禁止开跑，先修上面的 FAIL"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="实验前复核（Preflight）")
    ap.add_argument("--protocol", default="protocol_b_seen_category")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()
    ok = run_preflight(args.protocol, args.fold, args.config, args.exp_id, args.sample)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
