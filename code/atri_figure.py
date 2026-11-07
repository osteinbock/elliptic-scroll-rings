"""Earlier version of Fig. S5 (now drawn by make_figs.py): elliptic rings of the Atri Ca2+ model (P = 0.1 uM).
(a) half-widths and (b) saddle amplitude of the 100 x 60 um ring (period-smoothed,
single-curve frames) vs the filament equation with |beta| of circular rings, started at the
first planar moment and following the measured growth of the ring (halfcycle.model_series_grown);
(c) observed / predicted half-period of every half-cycle versus mean radius, from
halfcycles_atri.csv (seeds 100 x 60 and 130 x 80, weakly elliptic 100 x 88 and 120 x 105,
and 100 x 60 at dx = 2 um);
reference lines from period_table_r2.csv (Barkley) and halfcycles_panfilov.csv.
    python atri_figure.py [datadir] [out.png] [barkley_period_table.csv]   (run halfcycle_tables.py first)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import halfcycle as H
from halfcycle_tables import ATRI_PERIOD

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"
D = sys.argv[1] if len(sys.argv) > 1 else "v2"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "atri_figure.png")
allrows = H.read_csv(os.path.join(D, "halfcycles_atri.csv"))
rows = [r for r in allrows if r["dx"] == 3.0]
pan = [r["ratio"] for r in H.read_csv(os.path.join(D, "halfcycles_panfilov.csv"))]
# Barkley reference: mean ratio of Fig. 5 (rigid spirals), from that analysis's table
BARKLEY = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(D)), "v2b", "period_table_r2.csv")
bark, _ = H.barkley_reference(BARKLEY)
print(f"Barkley reference {bark:.3f} from {BARKLEY}")

fn = "aell2_P0.1_100x60.npz"
beta = [r["beta"] for r in rows if r["file"] == fn][0]
S = H.smoothed(os.path.join(D, fn), ATRI_PERIOD)
M = H.model_series_grown(S, beta)   # follows the measured growth of the ring (alpha < 0)
# the saddle amplitude's sign depends on the ring's orientation (sign of beta); the model
# is run with |beta|, so the RD curve is shown with the sign that makes the initial slopes agree
x0 = H.C.crossings(S["t"], S["z2c"])[0]
k = (S["t"] > x0) & (S["t"] < x0 + 3 * ATRI_PERIOD) & np.isfinite(S["z2c"])
sgn = np.sign(np.polyfit(S["t"][k], S["z2c"][k], 1)[0] * np.polyfit(M["t"][:60], M["z2c"][:60], 1)[0])

fig = plt.figure(figsize=(10, 6.4), dpi=130)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75], hspace=0.3, wspace=0.4)
ax_a = fig.add_subplot(gs[0, :2])
ax_b = fig.add_subplot(gs[1, :2], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 2])

ax_a.plot(S["t"], S["hx"], color=C1, lw=2, label=r"$h_x$")
ax_a.plot(S["t"], S["hy"], color=C2, lw=2, label=r"$h_y$")
ax_a.plot(M["t"], M["hx"], color=C1, lw=1.2, ls="--")
ax_a.plot(M["t"], M["hy"], color=C2, lw=1.2, ls="--")
ax_a.plot([], [], color=MUTED, ls="--", label="filament eq.")
ax_a.set_ylabel(r"half-width ($\mu$m)")
ax_a.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax_a.set_title("(a)", fontsize=9, loc="left")
ax_b.plot(S["t"], sgn * S["z2c"], color=INK, lw=2, label="simulation")
ax_b.plot(M["t"], M["z2c"], color=MUTED, lw=1.2, ls="--", label="filament eq.")
ax_b.axhline(0, color=MUTED, lw=0.6)
ax_b.set_ylabel(r"saddle amplitude ($\mu$m)")
ax_b.set_xlabel("time (s)")
ax_b.legend(frameon=False, fontsize=8, loc="upper right")
ax_b.set_title("(b)", fontsize=9, loc="left")

for f_, mk, col, mfc, lab in (("aell2_P0.1_100x60.npz", "o", INK, INK, r"$100\times60$"),
                              ("aell2_P0.1_130x80.npz", "s", INK, "white", r"$130\times80$"),
                              ("aellW_P0.1_100x88.npz", "^", C1, C1, r"$100\times88$"),
                              ("aellW_P0.1_120x105.npz", "v", C1, "white", r"$120\times105$"),
                              ("aell_dx2_P0.1_100x60.npz", "D", MUTED, "white", r"$100\times60$, $\Delta x = 2\,\mu$m")):
    R = [r["Rbar"] for r in allrows if r["file"] == f_]
    q = [r["ratio"] for r in allrows if r["file"] == f_]
    if R:
        ax_c.plot(R, q, mk, color=col, ms=6, mfc=mfc, label=lab)
Rall = [r["Rbar"] for r in allrows]
xr = (min(Rall) - 8, max(Rall) + 8)
ax_c.axhline(1.0, color=INK, lw=1)
ax_c.axhline(bark, color=C2, lw=1, ls=":")
ax_c.text(xr[1] - 1, bark - 0.008, "Barkley", fontsize=7, color=C2, ha="right", va="top")
ax_c.axhspan(min(pan), max(pan), color=C1, alpha=0.12, lw=0)
ax_c.text(xr[1] - 1, min(pan) - 0.008, "Panfilov–Rudenko", fontsize=7, color=C1, ha="right", va="top")
ax_c.set_xlim(*xr)
ax_c.set_ylim(0.85, 1.5)
ax_c.set_xlabel(r"mean radius $\bar R$ ($\mu$m)")
ax_c.set_ylabel("observed / predicted half-period")
ax_c.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.0), title="seed (µm)", title_fontsize=7)
ax_c.set_title("(c)", fontsize=9, loc="left")
for ax in (ax_a, ax_b, ax_c):
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
