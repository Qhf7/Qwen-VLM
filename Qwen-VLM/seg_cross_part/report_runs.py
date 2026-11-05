"""扫描 `artifacts/seg_cross_part/*/`，生成运行台账（Markdown 表格）。

目的：让"每次实验都要记录"变成一条命令，而不是靠翻日志回忆。

用法：
    python -m seg_cross_part.report_runs                    # 打印表格
    python -m seg_cross_part.report_runs --md docs/台账.md   # 写入文件（覆盖区块）

每个 run 读取：
    frozen_protocol.json  协议 / fold / 主干 / 设备 / best val
    training_history.csv  最后一个评测点
    test_metrics.json     test 宏平均（没有就标未完成）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import ROOT

ROOT_DIR = ROOT / "artifacts" / "seg_cross_part"
COLS = ["exp_id", "协议", "fold", "状态", "末步", "best val AUPRO",
        "test AUPRO", "test P-AUPR", "test P-AUROC", "test I-AUROC"]


def collect() -> list[dict]:
    rows = []
    for d in sorted(ROOT_DIR.iterdir() if ROOT_DIR.exists() else []):
        if not d.is_dir():
            continue
        row = {"exp_id": d.name, "协议": "?", "fold": "", "状态": "未完成",
               "末步": "", "best val AUPRO": "", "test AUPRO": "",
               "test P-AUPR": "", "test P-AUROC": "", "test I-AUROC": ""}
        fp = d / "frozen_protocol.json"
        if fp.exists():
            try:
                f = json.loads(fp.read_text(encoding="utf-8"))
                p = f.get("protocol", {})
                row["协议"] = p.get("type", "?")
                row["fold"] = p.get("fold") if p.get("fold") is not None else ""
                bv = f.get("best_val_macro_apro")
                row["best val AUPRO"] = f"{bv:.4f}" if isinstance(bv, (int, float)) else ""
            except Exception:
                pass
        th = d / "training_history.csv"
        if th.exists():
            try:
                h = pd.read_csv(th)
                if len(h):
                    row["末步"] = int(h.iloc[-1]["step"])
            except Exception:
                pass
        tm = d / "test_metrics.json"
        if tm.exists():
            try:
                m = json.loads(tm.read_text(encoding="utf-8")).get("macro", {})
                row["状态"] = "✅ 完成"
                row["test AUPRO"] = f"{m.get('apro_0.3', float('nan')):.4f}"
                row["test P-AUPR"] = f"{m.get('aupr', float('nan')):.4f}"
                row["test P-AUROC"] = f"{m.get('auroc', float('nan')):.4f}"
                row["test I-AUROC"] = f"{m.get('i_auroc', float('nan')):.4f}"
            except Exception:
                pass
        elif (d / "best.pt").exists() or (d / "training_history.csv").exists():
            row["状态"] = "⏳ 进行中/中断"
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_（暂无 run）_"
    out = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in COLS) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成运行台账")
    ap.add_argument("--md", type=Path, default=None, help="写入文件（覆盖整份文件）")
    args = ap.parse_args()
    rows = collect()
    table = to_markdown(rows)
    print(table)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("# 运行台账（自动生成）\n\n" + table + "\n", encoding="utf-8")
        print(f"\n已写出 {args.md}")


if __name__ == "__main__":
    main()
