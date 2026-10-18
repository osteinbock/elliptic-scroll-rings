"""Compare an RD elliptic scroll ring (ellipse_ring.py output) with the
filament-equation prediction (filament_flow.py) for the measured alpha, beta.

    python ellipse_compare.py ell_b0.09_8x5.npz 0.0 -2.4 4.12 ellipse_compare.png
(args: RD npz, alpha, beta, spiral rotation period used for smoothing, output)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import filament_flow as F

C1, C2, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"


def smooth(t, y, period):
    """Moving average over one rotation period (removes the tip-circle wobble)."""
    dt = np.median(np.diff(t))
    w = max(1, int(round(period / dt)))
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="same")
    half = w // 2
    ys[:half] = np.nan
    ys[len(ys) - half:] = np.nan
    return ys


def crossings(t, e):
    return [t[i] - e[i] * (t[i + 1] - t[i]) / (e[i + 1] - e[i])
            for i in range(len(e) - 1) if np.isfinite(e[i]) and np.isfinite(e[i + 1]) and e[i] * e[i + 1] < 0]


def main():
    path, alpha, beta, period, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    d = np.load(path)
    t = d["t"]
    ok = np.isfinite(d["hx"])
    t = t[ok]
    hx, hy, h2c, L, z = (smooth(t, d[k][ok], period) for k in ("hx", "hy", "h2c", "length", "z"))
    Ax, Ay = float(d["Ax"]), float(d["Ay"])
    # optional: start the filament equation from the RD ring's formed shape at time t0,
    # and add a second curve with a fitted beta
    t0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    Ax0 = float(sys.argv[7]) if len(sys.argv) > 7 else Ax
    Ay0 = float(sys.argv[8]) if len(sys.argv) > 8 else Ay
    beta2 = float(sys.argv[9]) if len(sys.argv) > 9 else None
    lia = F.run(Ax0, Ay0, alpha, beta, t[-1] - t0, n=64, dt=0.01)
    lia["t"] = lia["t"] + t0
    lia2 = None
    if beta2 is not None:
        lia2 = F.run(Ax0, Ay0, alpha, beta2, t[-1] - t0, n=64, dt=0.01)
        lia2["t"] = lia2["t"] + t0

    rd_x = crossings(t, hx - hy)
    li_x = crossings(lia["t"], lia["hx"] - lia["hy"])
    print("RD axis crossings :", np.round(rd_x, 1))
    print("LIA axis crossings:", np.round(li_x, 1))
    if len(rd_x) > 2:
        print("RD period ~", 2 * np.mean(np.diff(rd_x)))
    if len(li_x) > 2:
        print("LIA period ~", 2 * np.mean(np.diff(li_x)))
    m = np.isfinite(z)
    print("RD mean dz/dt =", np.polyfit(t[m], z[m], 1)[0])

    fig = plt.figure(figsize=(10, 9), dpi=120)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.2, 1, 1, 1.1], hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    for ax in (ax1, ax2, ax3):
        ax.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)

    ax1.plot(t, hx, color=C1, lw=2, label="RD  half-extent x")
    ax1.plot(t, hy, color=C2, lw=2, label="RD  half-extent y")
    ax1.plot(lia["t"], lia["hx"], color=C1, lw=1.4, ls="--", label="filament eq.  x")
    ax1.plot(lia["t"], lia["hy"], color=C2, lw=1.4, ls="--", label="filament eq.  y")
    if lia2 is not None:
        ax1.plot(lia2["t"], lia2["hx"], color=C1, lw=1.0, ls=":", label=f"beta={beta2:g}  x")
        ax1.plot(lia2["t"], lia2["hy"], color=C2, lw=1.0, ls=":", label=f"beta={beta2:g}  y")
        print("LIA(beta2) axis crossings:", np.round(crossings(lia2["t"], lia2["hx"] - lia2["hy"]), 1))
    ax1.set_ylabel("half-extent", color=INK, fontsize=9)
    ax1.legend(ncol=6 if lia2 is not None else 4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax1.set_title(f"Elliptic scroll ring, Barkley a={float(d['a'])} b={float(d['b'])} eps={float(d['eps'])}, "
                  f"seed {Ax:g} x {Ay:g};  filament eq. from {Ax0:g} x {Ay0:g} at t={t0:g}, alpha={alpha:g}, beta={beta:g}",
                  fontsize=9, color=INK, pad=28)

    ax2.plot(t, h2c, color=INK, lw=2, label="RD")
    ax2.plot(lia["t"], lia["h2c"], color=MUTED, lw=1.4, ls="--", label=f"filament eq. beta={beta:g}")
    if lia2 is not None:
        ax2.plot(lia2["t"], lia2["h2c"], color=MUTED, lw=1.0, ls=":", label=f"filament eq. beta={beta2:g}")
    ax2.axhline(0, color=MUTED, lw=0.6)
    ax2.set_ylabel("saddle amplitude\n(cos 2θ part of z)", color=INK, fontsize=9)
    ax2.legend(fontsize=8, frameon=False, loc="upper right")

    ax3.plot(t, L, color=INK, lw=2, label="RD")
    ax3.plot(lia["t"], lia["length"], color=MUTED, lw=1.4, ls="--", label="filament eq.")
    ax3.set_ylabel("filament length", color=INK, fontsize=9)
    ax3.set_xlabel("time", color=INK, fontsize=9)
    ax3.legend(fontsize=8, frameon=False, loc="lower left")

    # filament snapshots (top view, coloured by height relative to the ring centre)
    snaps = sorted({float(k.split("_")[1]) for k in d.files if k.startswith("snap_")})
    want = ([float(x) for x in sys.argv[10].split(",")] if len(sys.argv) > 10
            else (np.linspace(snaps[0], snaps[-1], 4) if len(snaps) >= 4 else snaps))
    pick = [min(snaps, key=lambda s: abs(s - w)) for w in want]
    lim = max(Ax, Ay) + 1.5
    for i, s in enumerate(pick):
        ax = fig.add_subplot(gs[3, i])
        curves = [d[k] for k in d.files if k.startswith(f"snap_{s:g}_")]
        p = max(curves, key=len)
        c = p.mean(0)
        sc = ax.scatter(p[:, 0] - c[0], p[:, 1] - c[1], c=p[:, 2] - c[2], s=4, cmap="coolwarm", vmin=-1.2, vmax=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"t = {s:g}", fontsize=8, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
    cb = fig.colorbar(sc, ax=fig.axes[-4:], shrink=0.8, pad=0.02)
    cb.set_label("z − z̄", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
