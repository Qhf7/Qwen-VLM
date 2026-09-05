"""Plot a downloaded classification training history and Dev checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/e1_classification_fold1"))
    args = parser.parse_args()
    with (args.run_dir / "training_history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    steps = np.array([int(row["step"]) for row in rows])
    loss = np.array([float(row["loss"]) for row in rows])
    width = min(50, len(rows))
    smooth = lambda values: np.convolve(values, np.ones(width) / width, mode="valid")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(steps, loss, color="#b1c5dc", lw=0.7, alpha=0.65, label="Per-step BCE")
    axes[0].plot(steps[width-1:], smooth(loss), color="#1859a0", lw=2.3, label=f"Trailing {width}-step mean")
    axes[0].axhline(np.log(2), color="#777777", ls=":", label="BCE at p=0.5")
    axes[0].set(title="Classification-only E1: training loss", ylabel="BCE loss")
    for source, label in [("real_anomaly", "Real anomaly"), ("pseudo_anomaly", "Pseudo anomaly"), ("normal", "Normal"), ("hard_normal", "Hard normal")]:
        values = [float(row[f"source_{source}_loss_mean"]) for row in rows]
        axes[1].plot(steps[width-1:], smooth(values), label=label)
        print(f"{label}: last 100 steps mean BCE = {np.mean(values[-100:]):.6f}")
    axes[1].set(title="Loss by training source (trailing 50-step mean)", ylabel="BCE loss")
    metrics = [json.loads(path.read_text()) for path in sorted(args.run_dir.glob("step*_metrics.json"))]
    for key, label in [("macro_aupr", "Macro I-AUPR"), ("macro_f1max", "Macro F1max"), ("small_defect_recall", "Small-defect recall*")]:
        axes[2].plot([int(m["tag"].replace("step", "")) for m in metrics], [m[key] for m in metrics], marker="o", label=label)
    axes[2].set(title="Dev: 720 groups / 3600 views (*checkpoint-specific F1max threshold)", ylabel="Metric", ylim=(0.45, 1.04))
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.18)
        axis.legend(loc="best", fontsize=9)
    destination = args.run_dir / "loss_curve.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination.resolve())


if __name__ == "__main__":
    main()
