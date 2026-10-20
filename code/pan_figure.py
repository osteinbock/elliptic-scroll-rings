"""Fig. S1: elliptic rings of the Panfilov-Rudenko model (kinetics B).
(a) g_f = 0.775 (zero tension) and (b) g_f = 0.8: half-widths, RD (period-smoothed,
single-curve frames) vs the filament equation with |beta| of circular rings, following the
measured size change and started at the first planar moment; (c) observed / predicted half-period of every
half-cycle (halfcycle.py conventions), from halfcycles_panfilov.csv.
    python pan_figure.py [datadir] [out.png] [barkley_period_table.csv]    (run halfcycle_tables.py first)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import halfcycle as H
import pan_alpha as PA

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"
D = sys.argv[1] if len(sys.argv) > 1 else "v2"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "figS1.png")
rows = H.read_csv(os.path.join(D, "halfcycles_panfilov.csv"))
# Barkley reference: mean ratio of Fig. 5 (rigid spirals), from that analysis's table
BARKLEY = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(D)), "v2b", "period_table_r2.csv")
bark, _ = H.barkley_reference(BARKLEY)
print(f"Barkley reference {bark:.3f} from {BARKLEY}")

fig = plt.figure(figsize=(10, 7.2), dpi=130)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.35)
ax_a = fig.add_subplot(gs[0, :2])
ax_s = fig.add_subplot(gs[1, :2], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 2])


def halfwidths(ax, fn, gf, lab):
    beta = [r["beta"] for r in rows if r["file"] == fn][0]
    S = H.smoothed(os.path.join(D, fn), PA.PERIOD[gf])
    M = H.model_series_grown(S, beta)   # follows the measured size change of the ring
    ax.plot(S["t"], S["hx"], color=C1, lw=2, label=r"$h_x$")
    ax.plot(S["t"], S["hy"], color=C2, lw=2, label=r"$h_y$")
    ax.plot(M["t"], M["hx"], color=C1, lw=1.2, ls="--")
    ax.plot(M["t"], M["hy"], color=C2, lw=1.2, ls="--")
    ax.text(0.01, 0.95, lab, transform=ax.transAxes, va="top", fontsize=9, color=INK)


halfwidths(ax_a, "pell_gf0.775_20x12.npz", 0.775, r"(a) $g_f = 0.775$ (zero tension): half-widths")
ax_a.plot([], [], color=MUTED, ls="--", label="filament eq.")
ax_a.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
ax_a.set_ylabel("half-width")
halfwidths(ax_s, "pell_gf0.8_24x15.npz", 0.8, r"(b) $g_f = 0.8$ (weak positive tension): half-widths")
ax_s.set_ylabel("half-width")
ax_s.set_xlabel("time")

# (c) one point per run: mean +- sd over its half-cycles; small dots: single half-cycles
order = [("pell_gf0.775_20x12.npz", "0.775\n0.6"), ("pell_gf0.8_24x15.npz", "0.8\n0.6"),
         ("pellBIG_gf0.8_24x15.npz", "0.8\n0.6, big"), ("pellHR_gf0.8_24x15.npz", "0.8\n0.4"),
         ("pellXR_gf0.8_24x15.npz", "0.8\n0.3")]
for i, (fn, lab) in enumerate(order):
    q = np.array([r["ratio"] for r in rows if r["file"] == fn])
    if not len(q):
        continue
    col = INK if i == 0 else MUTED
    ax_c.plot(np.full(len(q), i) + 0.12, q, ".", color=col, ms=4)
    ax_c.errorbar(i, q.mean(), yerr=q.std() if len(q) > 1 else None, fmt="o", color=col, ms=7, capsize=3)
ax_c.axhline(1.0, color=INK, lw=1)
ax_c.axhline(bark, color=C2, lw=1, ls=":")
ax_c.text(4.3, bark - 0.015, f"Barkley ({bark:.2f})", fontsize=7, color=C2, ha="right", va="top")
ax_c.set_xticks(range(len(order)))
ax_c.set_xticklabels([o[1] for o in order], fontsize=7)
ax_c.set_xlim(-0.6, len(order) - 0.4)
ax_c.set_ylim(0.85, 1.4)
ax_c.set_xlabel(r"$g_f$ / grid spacing $\Delta x$", fontsize=8)
ax_c.set_ylabel("observed / predicted half-period")
ax_c.set_title("(c) period ratio", fontsize=9, loc="left")
for ax in (ax_a, ax_s, ax_c):
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
