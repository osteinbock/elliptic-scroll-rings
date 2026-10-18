"""SI figure for the Atri Ca2+ model (P = 0.1 uM):
(a) half-widths and (b) saddle amplitude of the 100 x 60 um ring, simulation vs filament
equation with |beta| of circular rings (alpha = 0); (c) observed / predicted half-period of
all half-cycles of the 100 x 60 and 130 x 80 rings versus mean radius (finite-amplitude
prediction; values from atri_compare.py).
    python atri_figure.py out.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"
out = sys.argv[1] if len(sys.argv) > 1 else "atri_figS2.png"
D = np.load("aell2_P0.1_100x60_cmp_data.npz")
# (Rbar, ratio) per half-cycle, from atri_compare.py (finite-amplitude prediction)
small = [(93.2, 1.160), (104.9, 1.183), (119.0, 1.242)]
large = [(121.4, 1.140), (134.8, 1.228)]

fig = plt.figure(figsize=(10, 6.4), dpi=130)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75], hspace=0.3, wspace=0.4)
ax_a = fig.add_subplot(gs[0, :2])
ax_b = fig.add_subplot(gs[1, :2], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 2])

ax_a.plot(D["t"], D["hx"], color=C1, lw=2, label=r"$h_x$")
ax_a.plot(D["t"], D["hy"], color=C2, lw=2, label=r"$h_y$")
ax_a.plot(D["mt"], D["mhx"], color=C1, lw=1.2, ls="--")
ax_a.plot(D["mt"], D["mhy"], color=C2, lw=1.2, ls="--")
ax_a.plot([], [], color=MUTED, ls="--", label="filament eq.")
ax_a.set_ylabel(r"half-width ($\mu$m)")
ax_a.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax_a.set_title("(a)", fontsize=9, loc="left")
ax_b.plot(D["t"], -D["z2c"], color=INK, lw=2, label="simulation")   # sign matched to the model's chirality
ax_b.plot(D["mt"], D["mh"], color=MUTED, lw=1.2, ls="--", label="filament eq.")
ax_b.axhline(0, color=MUTED, lw=0.6)
ax_b.set_ylabel(r"saddle amplitude ($\mu$m)")
ax_b.set_xlabel("time (s)")
ax_b.legend(frameon=False, fontsize=8, loc="upper right")
ax_b.set_title("(b)", fontsize=9, loc="left")

for pts, mk, lab in ((small, "o", r"seed $100\times60$"), (large, "s", r"seed $130\times80$")):
    R, q = np.array(pts).T
    ax_c.plot(R, q, mk, color=INK, ms=7, mfc=INK if mk == "o" else "white", label=lab)
ax_c.axhline(1.0, color=INK, lw=1)
ax_c.axhline(1.33, color=C2, lw=1, ls=":")
ax_c.text(137, 1.335, "Barkley", fontsize=7, color=C2, ha="right", va="bottom")
ax_c.axhspan(1.01, 1.08, color=C1, alpha=0.12, lw=0)
ax_c.text(137, 1.045, "Panfilov–Rudenko", fontsize=7, color=C1, ha="right", va="center")
ax_c.set_xlim(85, 140)
ax_c.set_ylim(0.9, 1.45)
ax_c.set_xlabel(r"mean radius $\bar R$ ($\mu$m)")
ax_c.set_ylabel("observed / predicted half-period")
ax_c.legend(frameon=False, fontsize=8, loc="lower left")
ax_c.set_title("(c)", fontsize=9, loc="left")
for ax in (ax_a, ax_b, ax_c):
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
