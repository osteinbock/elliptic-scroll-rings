"""Plot both RD elliptic rings against the filament model (eta = 0 and best eta)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fit_eta_scan as S
import ellipse_compare as C

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"
best = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SRC = sys.argv[3] if len(sys.argv) > 3 else "eta_scan.npy"
out = np.load(SRC, allow_pickle=True).item()
fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=120, sharex="col")
for j, key in enumerate(S.RINGS):
    t, hx, hy, h, L = S.rd(S.RINGS[key])
    series = [("RD", t, (hx - hy) / 2, h, L, INK, "-", 2.0)]
    for eta, col, ls in ((0, C2, "--"), (best, C1, "--")):
        r = out[f"{key}_eta{eta:g}_lam{LAM:g}"]
        series.append((f"model eta={eta:g}, lam={LAM:g}", r["t"], (r["hx"] - r["hy"]) / 2, r["h2c"], r["length"], col, ls, 1.5))
    for lab, tt, e, hh, LL, col, ls, lw in series:
        axes[0, j].plot(tt, e, color=col, ls=ls, lw=lw, label=lab)
        axes[1, j].plot(tt, hh, color=col, ls=ls, lw=lw)
        axes[2, j].plot(tt, LL, color=col, ls=ls, lw=lw)
    rx = np.round(C.crossings(t, hx - hy), 1)
    for eta in (0, best):
        r = out[f"{key}_eta{eta:g}_lam{LAM:g}"]
        print(key, f"eta={eta}", "model crossings", np.round(C.crossings(r["t"], r["hx"] - r["hy"]), 1))
    print(key, "RD crossings   ", rx)
    axes[0, j].set_title(f"seed {key}", fontsize=10, color=INK)
    axes[0, j].legend(fontsize=8, frameon=False, ncol=3)
    axes[2, j].set_xlabel("time", fontsize=9)
for ax, lab in zip(axes[:, 0], ["(hx − hy)/2", "saddle amplitude", "filament length"]):
    ax.set_ylabel(lab, fontsize=9)
for ax in axes.flat:
    ax.grid(color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
fig.suptitle("Barkley a=1.1 b=0.17 eps=0.02: RD elliptic rings vs filament model " +
             ("(local alpha(K), beta(K) from circular rings; rigidity eta)" if "local" in SRC else "(alpha, beta from circular rings, set by ring size; rigidity eta)"), fontsize=10)
fig.tight_layout()
fig.savefig(SRC.replace(".npy", "_compare.png"))
print("wrote", SRC.replace(".npy", "_compare.png"))
