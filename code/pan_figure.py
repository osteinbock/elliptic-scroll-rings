"""Second-model figure: Panfilov kinetics B elliptic rings.
(a) zero tension (gf = 0.775): half-widths and saddle amplitude, RD vs filament eq.
(b) weak positive tension (gf = 0.8): half-widths, RD vs filament eq.
(c) observed vs predicted half-period for every half-cycle (prediction from the
    circular-ring coefficients at the half-cycle's mean radius).
Reads pan_compare_gf*_data.npz written by pan_compare.py.
    python pan_figure.py out.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"
out = sys.argv[1] if len(sys.argv) > 1 else "pan_second_model.png"
A = np.load("pan_compare_gf0.775_data.npz")
B_ = np.load("pan_compare_gf0.8_data.npz")

fig = plt.figure(figsize=(10, 7.2), dpi=130)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.35)
ax_a = fig.add_subplot(gs[0, :2])
ax_s = fig.add_subplot(gs[1, :2], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 2])


def halfwidths(ax, D, lab):
    ax.plot(D["t"], D["hx"], color=C1, lw=2, label=r"$h_x$")
    ax.plot(D["t"], D["hy"], color=C2, lw=2, label=r"$h_y$")
    ax.plot(D["mt"], D["mhx"], color=C1, lw=1.2, ls="--")
    ax.plot(D["mt"], D["mhy"], color=C2, lw=1.2, ls="--")
    ax.text(0.01, 0.95, lab, transform=ax.transAxes, va="top", fontsize=9, color=INK)


halfwidths(ax_a, A, r"(a) $g_f = 0.775$ (zero tension): half-widths")
ax_a.plot([], [], color=MUTED, ls="--", label="filament eq.")
ax_a.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
ax_a.set_ylabel("half-width")
halfwidths(ax_s, B_, r"(b) $g_f = 0.8$ (weak positive tension): half-widths")
ax_s.set_ylabel("half-width")
ax_s.set_xlabel("time")

# (c) observed / predicted half-period from zero crossings of the saddle amplitude
#     (prediction pi Rbar^2 / (2 sqrt3 |beta|), beta from circular rings at the same dx)
runs = [("0.775\n0.6", 1.039, 0.033, INK), ("0.8\n0.6", 1.013, 0.011, MUTED),
        ("0.8\n0.6, big", 1.039, 0.0, MUTED), ("0.8\n0.4", 1.080, 0.012, MUTED),
        ("0.8\n0.3", 1.057, 0.0, MUTED)]
for i, (lab, q, e, col) in enumerate(runs):
    ax_c.errorbar(i, q, yerr=e if e else None, fmt="o", color=col, ms=7, capsize=3)
ax_c.axhline(1.0, color=INK, lw=1)
ax_c.axhline(1.33, color=C2, lw=1, ls=":")
ax_c.text(4.3, 1.315, "Barkley (1.33)", fontsize=7, color=C2, ha="right", va="top")
ax_c.set_xticks(range(len(runs)))
ax_c.set_xticklabels([r[0] for r in runs], fontsize=7)
ax_c.set_xlim(-0.6, len(runs) - 0.4)
ax_c.set_ylim(0.9, 1.4)
ax_c.set_xlabel(r"$g_f$ / grid spacing $\Delta x$", fontsize=8)
ax_c.set_ylabel("observed / predicted half-period")
ax_c.set_title("(c) period ratio", fontsize=9, loc="left")
for ax in (ax_a, ax_s, ax_c):
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
# fig.suptitle("Elliptic scroll rings in a second model (piecewise-linear kinetics of Panfilov & Rudenko)",
#             fontsize=10, color=INK)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
