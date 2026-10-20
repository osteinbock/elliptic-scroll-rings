"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Circular scroll rings (Barkley a=1.1 b=0.17 eps=0.02, periodic boxes with Lz >= 16):
ring radius vs time for different initial radii. Rings converge to a stable radius R*
from both sides; rings formed below an unstable radius collapse."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import ellipse_compare as C

FILES = [("circHB_R0_4.0.npz", 4.0), ("circS_R0_4.5.npz", 4.5), ("circHB_R0_5.0.npz", 5.0),
         ("circS_R0_5.5.npz", 5.5), ("circL_R0_6.npz", 6.0), ("circS_R0_6.5.npz", 6.5),
         ("circS_R0_7.5.npz", 7.5), ("circL_R0_9.npz", 9.0), ("circL_R0_13.npz", 13.0)]
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
cmap = plt.get_cmap("viridis")
for i, (f, R0) in enumerate(FILES):
    d = np.load(f)
    ok = np.isfinite(d["Rm"])
    t, R = d["t"][ok], d["Rm"][ok]
    g = np.abs(R - median_filter(R, 25, mode="nearest")) < 0.4
    t, R = t[g], R[g]
    Rs = C.smooth(t, R, 4.44)
    m = t <= 200
    ax.plot(t[m], Rs[m], color=cmap(i / (len(FILES) - 1)), lw=1.8, label=f"seed R0 = {R0:g}")
ax.axhspan(4.5, 5.1, color="#e4e3dd", zorder=0)
ax.text(198, 4.8, "R* ≈ 4.8", ha="right", va="center", fontsize=8, color="#1f1f1e")
ax.set(xlabel="time", ylabel="ring radius R", xlim=(0, 200), ylim=(0, 12))
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.grid(color="#e4e3dd", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Circular scroll rings, Barkley a=1.1 b=0.17 eps=0.02", fontsize=9)
fig.tight_layout()
fig.savefig("radius_scan.png")
print("wrote radius_scan.png")
