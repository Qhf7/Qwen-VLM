import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = 400; warm = 40; lr_max = 1e-3
t = np.arange(T)
warmup = lr_max * t / warm
cos = lr_max * 0.5 * (1 + np.cos(np.pi * (t - warm) / (T - warm)))
lin = lr_max * (1 - (t - warm) / (T - warm))
step = lr_max * (0.5 ** (t // 100))
sch = np.where(t < warm, warmup, cos)
sch_lin = np.where(t < warm, warmup, lin)
const = np.full(T, lr_max)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
ax[0].plot(t, sch, lw=2.6, label="warmup + cosine (paper)")
ax[0].plot(t, sch_lin, "--", lw=2, label="warmup + linear (code WarmupDecayLR)")
ax[0].plot(t, step, ":", lw=2, label="step decay (classic)")
ax[0].plot(t, const, "-.", lw=1.2, alpha=.6, label="constant")
ax[0].axvspan(0, warm, color="orange", alpha=.12)
ax[0].text(warm/2, lr_max*1.03, "warmup", ha="center", fontsize=9)
ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("learning rate")
ax[0].set_title("learning-rate schedules"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.15e-3)

# 本项目真实配置
tot, wr = 22681, 0.1
w = int(tot*wr); tt = np.arange(tot)
s = np.where(tt < w, lr_max*tt/w, lr_max*(1-(tt-w)/(tot-w)))
ax[1].plot(tt, s, lw=1.6, c="tab:red")
ax[1].axvspan(0, w, color="orange", alpha=.12)
ax[1].axhline(lr_max, ls=":", c="gray")
ax[1].annotate("peak 1e-3 @ step %d" % w, (w, lr_max), textcoords="offset points", xytext=(10, -14), fontsize=9)
ax[1].annotate("→ 0 @ step %d" % tot, (tot, 0), textcoords="offset points", xytext=(-90, 12), fontsize=9)
ax[1].set_xlabel("step (50 epochs x 2 batches/step)"); ax[1].set_ylabel("lr")
ax[1].set_title("this project: 22681 steps, warmup 10%")
fig.tight_layout(); fig.savefig("21_lr_schedule.png", dpi=140, bbox_inches="tight")
print("ok")
