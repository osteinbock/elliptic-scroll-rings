"""Observed vs predicted half-period of the axis-switching oscillation, every half-cycle
of every run (period_table.csv from period_analysis.py). Prediction: linear filament
equation with alpha, beta, R measured in the same half-cycle."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("period_table.csv")))
PAL = {"a=1.1 b=0.15": "#2a78d6", "a=1.1 b=0.16": "#1baf7a", "a=1.1 b=0.17 (8x5)": "#eb6834",
       "a=1.1 b=0.17 (12x7.5)": "#eda100", "a=1.1 b=0.18": "#e87ba4", "a=0.8 b=0.09 (meander)": "#6b6a63"}
NAME = {"a=1.1 b=0.15": "b = 0.15", "a=1.1 b=0.16": "b = 0.16", "a=1.1 b=0.17 (8x5)": "b = 0.17, small ring",
        "a=1.1 b=0.17 (12x7.5)": "b = 0.17, large ring", "a=1.1 b=0.18": "b = 0.18",
        "a=0.8 b=0.09 (meander)": "a = 0.8, b = 0.09 (meandering)"}
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
lim = [0, 45]
ax.plot(lim, lim, color="#6b6a63", lw=1, label="theory, constant β")
ax.plot(lim, [1.32 * v for v in lim], color="#1f1f1e", lw=1, ls="--", label="1.32 × theory")
for run, col in PAL.items():
    rr = [r for r in rows if r["run"] == run]
    if not rr:
        continue
    mk = "^" if "meander" in run else "o"
    ax.scatter([float(r["T2"]) for r in rr], [float(r["Th"]) for r in rr], s=30, marker=mk, color=col,
               edgecolor="white", linewidth=0.8, zorder=3, label=NAME[run])
ax.set(xlim=lim, ylim=lim, xlabel="predicted half-period", ylabel="observed half-period")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.grid(color="#e4e3dd", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("period_obs_vs_pred.png")
print("wrote period_obs_vs_pred.png")
