import csv, json, statistics

PATH = "training_history.csv"
rows = []
with open(PATH) as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append({
            "step": int(d["step"]),
            "loss": float(d["loss"]),
            "focal": float(d["focal_loss"]),
            "dice": float(d["dice_loss"]),
            "grad": float(d["grad_norm"]),
            "lr": float(d["lr"]),
            "step_s": float(d["step_s"]),
            "mix": d["input_mix"],
        })

n = len(rows)
last = rows[-1]
first = rows[0]
best = min(rows, key=lambda x: x["loss"])

def mean(seq, key):
    return sum(x[key] for x in seq) / len(seq) if seq else float("nan")

def window(lo, hi, key):
    sub = [x for x in rows if lo <= x["step"] < hi]
    return mean(sub, key) if sub else float("nan")

stats = {
    "n_steps": n,
    "first_step": first["step"], "first_loss": first["loss"],
    "last_step": last["step"], "last_loss": last["loss"],
    "last_focal": last["focal"], "last_dice": last["dice"],
    "best_step": best["step"], "best_loss": best["loss"],
    "mean_loss_0_1000": window(0, 1000, "loss"),
    "mean_loss_1000_4000": window(1000, 4000, "loss"),
    "mean_loss_4000_8000": window(4000, 8000, "loss"),
    "mean_loss_8000_12000": window(8000, 12000, "loss"),
    "mean_dice_last200": window(n-200, n+1, "dice"),
    "mean_focal_last200": window(n-200, n+1, "focal"),
    "grad_min": min(x["grad"] for x in rows),
    "grad_max": max(x["grad"] for x in rows),
    "grad_last200_mean": window(n-200, n+1, "grad"),
    "lr_first": first["lr"], "lr_last": last["lr"],
    "lr_max": max(x["lr"] for x in rows),
    "lr_peak_step": max(rows, key=lambda x: x["lr"])["step"],
    "step_s_mean": statistics.mean(x["step_s"] for x in rows),
    "step_s_min": min(x["step_s"] for x in rows),
    "step_s_max": max(x["step_s"] for x in rows),
    "mix_unique": sorted(set(x["mix"] for x in rows)),
}
with open("training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
