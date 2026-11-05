"""Plot complete matched three-seed results from final_report/summary.json.

No values are imputed. All nine matched evaluations, training runs, and shared
initialization/order audits must be complete before the main figure is drawn.
An optional historical figure remains a separate artifact and protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)
ARMS = ("qwen_frozen", "qwen_lora", "vicuna")
LABELS = {"qwen_frozen": "Qwen3-VL-8B\nfrozen language",
          "qwen_lora": "Qwen3-VL-8B\nq/k/v/o LoRA",
          "vicuna": "Vicuna-7B-v1.1\nfrozen language",
          "frozen": "Qwen3-VL-8B\nfrozen language", "lora": "Qwen3-VL-8B\nq/k/v/o LoRA"}
COLORS = {"qwen_frozen": "#4477AA", "qwen_lora": "#228833", "vicuna": "#CC6677",
          "frozen": "#4477AA", "lora": "#228833"}
PANELS = (("generation_macro_f1", "Generation Macro-F1"),
          ("generation_anomaly_recall", "Anomaly recall"),
          ("generation_normal_fpr", "Normal false-positive rate"),
          ("category_macro_pixel_ap", "Category-macro pixel AP"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_rate(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= 1


def validate(summary, stage="matched"):
    issues = []
    if summary.get("protocol", {}).get("status") != "verified":
        issues.append("frozen real-source protocol is not verified")
    if summary.get("protocol", {}).get("num_sources") != 2005:
        issues.append("real protocol does not contain exactly 2,005 sources")
    arms = ARMS if stage == "matched" else ("frozen", "lora")
    evaluations = {row.get("name"): row for row in summary.get("real_evaluations", [])}
    groups = summary.get("real_groups", {})
    values = {}
    for arm in arms:
        group = groups.get(f"{stage}_{arm}", {})
        if group.get("status") != "complete" or set(group.get("completed_seeds", [])) != set(SEEDS):
            issues.append(f"{stage}_{arm}: three complete seeds are required")
        for metric, _ in PANELS:
            per_seed = []
            for seed in SEEDS:
                name = f"{stage}_{arm}{seed}"
                row = evaluations.get(name)
                if row is None or row.get("status") != "complete":
                    issues.append(f"{name}: evaluation is missing or incomplete")
                    continue
                value = row.get("metrics", {}).get(metric)
                if not finite_rate(value):
                    issues.append(f"{name}: missing/invalid {metric}")
                else:
                    per_seed.append(float(value))
            if len(per_seed) == 3:
                values[(arm, metric)] = np.asarray(per_seed)
                reported = group.get("metrics", {}).get(metric, {})
                if reported.get("n") != 3 or set(reported.get("seeds", [])) != set(SEEDS):
                    issues.append(f"{stage}_{arm}/{metric}: summary group does not contain the same three seeds")
                mean, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
                if not isinstance(reported.get("mean"), (int, float)) or abs(reported["mean"] - mean) > 1e-12:
                    issues.append(f"{stage}_{arm}/{metric}: reported mean differs from individual seeds")
                if not isinstance(reported.get("sample_sd"), (int, float)) or abs(reported["sample_sd"] - sd) > 1e-12:
                    issues.append(f"{stage}_{arm}/{metric}: reported sample SD differs from individual seeds")
    if stage == "matched":
        training = {(row.get("arm"), row.get("seed")): row for row in summary.get("training_runs", [])}
        audits = {row.get("seed"): row for row in summary.get("training_match_audit", [])}
        for seed in SEEDS:
            if audits.get(seed, {}).get("controlled_comparison_verified") is not True:
                issues.append(f"seed {seed}: shared initialization/order/cache/checkpoint audit is not verified")
            for arm in ARMS:
                row = training.get((arm, seed), {})
                if row.get("status") != "complete" or row.get("steps") != 16:
                    issues.append(f"{arm}_{seed}: 16-step training is not complete")
                if row.get("actual_variant_exposures") != 32 or row.get("max_tgt_len") != 128:
                    issues.append(f"{arm}_{seed}: expected 32 variant exposures and max_tgt_len=128")
    if issues:
        raise ValueError("Cannot plot incomplete or inconsistent results:\n- " + "\n- ".join(sorted(set(issues))))
    return arms, values


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Arial", "DejaVu Sans", "Noto Sans CJK SC", "PingFang SC",
                                  "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei") if name in available]
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": preferred or ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 130,
    })


def figure(summary, out, stage="matched"):
    arms, values = validate(summary, stage)
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 4.35), sharey=True)
    markers = ("o", "s", "^")
    offsets = (-0.16, 0.0, 0.16)
    for panel_index, (axis, (metric, title)) in enumerate(zip(axes, PANELS)):
        for index, arm in enumerate(arms):
            scores = values[(arm, metric)] * 100
            mean, sd = float(scores.mean()), float(scores.std(ddof=1))
            for seed_index, score in enumerate(scores):
                axis.scatter(index + offsets[seed_index], score, s=35, marker=markers[seed_index],
                             color=COLORS[arm], edgecolor="white", linewidth=0.5, zorder=3)
            axis.errorbar(index, mean, yerr=sd, fmt="D", color="#222222", markersize=4.2,
                          capsize=4, elinewidth=1.2, capthick=1.2, zorder=4)
        axis.set_xticks(range(len(arms)), [LABELS[arm] for arm in arms])
        axis.set_xlim(-0.48, len(arms) - 0.52)
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.set_title(f"({chr(97 + panel_index)}) {title}", pad=10)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)
        axis.tick_params(axis="x", length=0, pad=8)
    axes[0].set_ylabel("Score (%)")
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=5,
                      color="#666666", label=f"Seed {seed}") for marker, seed in zip(markers, SEEDS)]
    handles.append(Line2D([], [], color="#222222", marker="D", markersize=4,
                          label="Mean ± sample SD (n=3)", linewidth=1))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4,
               frameon=False, handletextpad=0.5, columnspacing=1.6)
    if stage == "matched":
        note = "Same 2,005-source development diagnostic • three seeds • 16 updates × batch 2"
        title = "Matched language-model migration pilot"
    else:
        note = "Same 2,005-source development diagnostic • three historical seeds • 16 updates; 64-token cap"
        title = "Historical Qwen bridge evaluation (separate protocol)"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.text(0.5, 0.025, note, ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.27, top=0.84, wspace=0.18)
    out.mkdir(parents=True, exist_ok=True)
    stem = "matched_three_seed" if stage == "matched" else "historical_qwen_three_seed"
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    return {"stage": stage, "arms": list(arms), "seeds": list(SEEDS),
            "panels": [{"metric": metric, "title": title,
                        "groups": {arm: {"values": values[(arm, metric)].tolist(),
                                         "mean": float(values[(arm, metric)].mean()),
                                         "sample_sd": float(values[(arm, metric)].std(ddof=1))} for arm in arms}}
                       for metric, title in PANELS],
            "outputs": [str(out / f"{stem}.{suffix}") for suffix in ("png", "pdf")]}


def caption(summary, summary_path, historical):
    protocol = summary["protocol"]
    training = summary.get("training_runs", [])
    source_counts = {seed: next((row.get("actual_unique_sources") for row in training
                               if row.get("seed") == seed and row.get("arm") == "qwen_frozen"), None) for seed in SEEDS}
    lines = ["# Figure caption", "",
             "**Matched language-model migration pilot.** Individual symbols show seeds 42, 43, and 44; black diamonds and error bars show the arithmetic mean ± sample standard deviation (ddof=1, n=3). Panels report free-generation binary-label Macro-F1, anomaly recall, normal false-positive rate (lower is better), and product-category-macro pixel average precision. Axes span 0–100%; SD whiskers beyond this range are clipped by the axes, and full SD values are retained in the results table and manifest. All models use the same 2,005 real VisA evaluation sources; these results are development diagnostics after inspection. Error bars summarize training-seed variation on this fixed source set and are not confidence intervals.", "",
             "The three arms are the Qwen3-VL-8B-Instruct language component with frozen language weights, the same component with q/k/v/o LoRA (rank 8), and frozen Vicuna-7B-v1.1. Vicuna v1.1 differs from the original paper's v0. The same seed shares the bridge initialization and source order across arms. Each arm uses 16 updates at batch size 2 (32 variant exposures) drawn from a cache containing 48 sources and 96 normal/synthetic variants; not all cached sources are exposed. The matched target-token cap is 128. The tokenizer, effective supervised token count, and compute can differ between language backbones.", "",
             f"Actual unique training sources by seed (42/43/44): {source_counts[42]}/{source_counts[43]}/{source_counts[44]}. Frozen real evaluation: {protocol.get('num_sources')} images, {protocol.get('normal_count')} normal and {protocol.get('anomaly_count')} anomalous, across {protocol.get('num_classes')} products. Full images are resized to 224²; category-index masks are binarized before nearest-neighbor resizing. Four layer maps are averaged before scoring.", "",
             "Image Decoder and PromptLearner start from the released VisA-trained AnomalyGPT checkpoint; the language projection is separately seeded. The released PromptLearner was originally paired with Vicuna, so sharing its initial weights does not establish equal semantic adaptation to Qwen. Complete pretraining-source provenance is unavailable; current-stage source isolation does not prove that the entire training chain never saw the evaluation sources. Normal-reference inference and PandaGPT mixed training are omitted. These plots do not establish zero-shot unseen-category generalization or statistical superiority.", "",
             f"Source summary: `{summary_path.resolve()}`. SHA256: `{digest(summary_path)}`.", ""]
    if historical:
        lines += ["The separate historical figure evaluates earlier Qwen frozen/LoRA bridges with max_tgt_len=64. It is not combined with the matched 128-token experiment to claim a controlled training improvement.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--historical", action="store_true", help="Also plot historical frozen/LoRA results separately, requiring all six complete runs")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    # Validate all requested groups before producing any output artifacts.
    validate(summary, "matched")
    if args.historical:
        validate(summary, "historical")
    artifacts = [figure(summary, args.out, "matched")]
    if args.historical:
        artifacts.append(figure(summary, args.out, "historical"))
    (args.out / "caption.md").write_text(caption(summary, args.summary, args.historical))
    manifest = {"summary": str(args.summary.resolve()), "summary_sha256": digest(args.summary),
                "plot_source_sha256": digest(__file__), "figures": artifacts,
                "units": "JSON rates [0,1]; plots percentage [0,100]", "error_bars": "sample SD ddof=1"}
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out.resolve()), "figures": [item["outputs"] for item in artifacts]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
