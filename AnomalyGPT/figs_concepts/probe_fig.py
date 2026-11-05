import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("vicuna_run/probe_results.json"))
labels = ["defect img\nREAL tokens", "defect img\nRANDOM tokens", "defect img\nZERO tokens",
          "defect img\nNORMAL-img tokens", "normal img\nREAL tokens"]
short = ["No, there is no anomaly in the image.",
         "I'm sorry, but I cannot see any image or photo...",
         "Based on the image provided, it appears that\nthere is an anomaly in the bottle.",
         "No, there is no anomaly in the image.",
         "No, there is no anomaly in the image."]
verdict = ["✗ wrong", "✗ broken", "✓ right", "✗ wrong", "✓ right"]
colors = ["#d62728", "#d62728", "#2ca02c", "#d62728", "#2ca02c"]

fig, ax = plt.subplots(figsize=(11.5, 5.4))
y = np.arange(5)[::-1]
for i, r in enumerate(R):
    py, pn = r["p_yes"], r["p_no"]
    po = max(0.0, 1 - py - pn)
    ax.barh(y[i], py, color="#2ca02c", height=.55)
    ax.barh(y[i], pn, left=py, color="#d62728", height=.55)
    ax.barh(y[i], po, left=py+pn, color="#bbbbbb", height=.55)
    if py > .02: ax.text(py/2, y[i], "Yes %.2f" % py, ha="center", va="center", color="w", fontsize=9)
    if pn > .02: ax.text(py+pn/2, y[i], "No %.2f" % pn, ha="center", va="center", color="w", fontsize=9)
    if po > .02: ax.text(py+pn+po/2, y[i], "other %.2f" % po, ha="center", va="center", fontsize=9)
    ax.text(1.02, y[i], short[i], va="center", fontsize=8.5, color=colors[i], clip_on=False)
    ax.text(1.02, y[i]+0.30, verdict[i], va="center", ha="left", fontsize=8.5, color=colors[i],
            fontweight="bold", clip_on=False)

ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 1); ax.set_xlabel("probability of the FIRST generated token")
ax.set_title("Vicuna-7B (int8) first-token distribution under 5 input conditions", fontsize=11)
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in ["#2ca02c","#d62728","#bbbbbb"]],
          labels=["P(Yes)","P(No)","P(other)"], loc="lower right", fontsize=8, framealpha=.9)
ax.grid(axis="x", alpha=.25)
fig.subplots_adjust(right=0.52)
fig.savefig("22_probe.png", dpi=150, bbox_inches="tight")
print("ok")
