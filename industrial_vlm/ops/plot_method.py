"""Draw the implemented pilot as a vector paper figure, without model inference."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(14, 8.1))
    ax.set(xlim=(0, 14), ylim=(0, 8.1))
    ax.axis("off")
    blue, orange, grey, green = "#E7F1FA", "#FFF0DA", "#F2F3F5", "#E8F4ED"
    ink = "#253448"

    def box(x, y, w, h, text, color=grey, size=10, dashed=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                              linewidth=1, edgecolor="#607086", facecolor=color,
                              linestyle="--" if dashed else "-")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=ink, linespacing=1.5)

    def arrow(start, end, dashed=False, color="#4B596B"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                    linewidth=1.2, color=color,
                                    linestyle="--" if dashed else "-"))

    ax.text(.2, 7.85, "(a) Source-disjoint instruction construction", weight="bold", size=12, color=ink)
    box(.25, 6.55, 2.05, .9, "VisA: 10,821 images\nImage + annotation")
    box(2.85, 6.55, 2.5, .9, "Validate + SHA256 split\n7,575 / 1,624 / 1,622")
    box(5.9, 6.55, 2.9, .9, "Train annotations only\nMask → union-box JSON")
    box(9.35, 6.55, 4.15, .9, "Three SFT arms, 768 sources each\nCoverage + region | Random + region\nCoverage + decision", size=9.5)
    for x, end in ((2.3, 2.85), (5.35, 5.9), (8.8, 9.35)):
        arrow((x + .04, 7), (end - .08, 7))
    ax.text(.3, 6.13, "All source partitions fixed before instruction sampling; heldout annotations are used for scoring.",
            size=9.5, color="#556478")

    ax.text(.2, 5.68, "(b) Native Qwen3-VL with language-attention QLoRA", weight="bold", size=12, color=ink)
    box(.25, 4.05, 1.65, 1, "One full image\nNative processor")
    box(2.4, 4.05, 2.05, 1, "Vision encoder\nFrozen BF16", blue)
    box(4.95, 4.05, 2.1, 1, "Merger + DeepStack\nFrozen BF16", blue)
    box(7.6, 3.85, 2.65, 1.4, "Language model\nFrozen NF4 Linear\n+ trainable attention LoRA", orange, size=9.5)
    box(10.8, 4.05, 2.7, 1, "Structured assistant output\nDecision + union box", green, size=9.5)
    for a, b in ((1.9, 2.4), (4.45, 4.95), (7.05, 7.6), (10.25, 10.8)):
        arrow((a + .04, 4.55), (b - .08, 4.55))
    box(7.6, 2.82, 2.65, .58, "Instruction tokens", grey, size=9.5)
    arrow((8.925, 3.48), (8.925, 3.78))
    box(10.8, 2.82, 2.7, .58, "Assistant-only causal CE", orange, size=9.5)
    arrow((12.15, 4.01), (12.15, 3.48), dashed=True)
    box(4.95, 2.82, 2.1, .58, "Training JSON target", grey, size=9.5)
    ax.plot([6, 6, 12.15, 12.15], [2.74, 2.35, 2.35, 2.58],
            color="#4B596B", linewidth=1.2, linestyle="--")
    arrow((12.15, 2.58), (12.15, 2.74), dashed=True)
    ax.text(.3, 3.33, "LoRA: q / k / v / o, r = 16, α = 32\n15.34M trainable parameters\nNative visual grids and positions retained", va="center",
            size=9.5, color="#556478", linespacing=1.6)
    ax.text(.3, 2.47, "Solid: forward inputs\nDashed: training loss only", size=9, color="#556478", linespacing=1.4)

    ax.text(.2, 2.03, "(c) Fixed-budget tests and controlled ablations", weight="bold", size=12, color=ink)
    box(.25, .65, 4.1, .95, "VisA heldout region test: n = 120\nBase vs coverage SFT vs random SFT", green, size=10)
    box(4.95, .65, 4.1, .95, "Common decision prompt: same 120 sources\nBase vs region SFT vs decision SFT", green, size=9.5)
    box(9.65, .65, 3.85, .95, "MVTec external pilot: n = 150\nBase vs coverage SFT", green, size=10)
    ax.text(7, .16, "Single seed · 96 updates × 8 images per arm · max 524,288 pixels · greedy decoding · paired source bootstrap",
            ha="center", size=9.3, color="#556478")
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"method.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
