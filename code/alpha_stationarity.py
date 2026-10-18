"""When does circular-scroll-ring tension become a function of curvature alone?

Long circular rings (circL_R0_*.npz, Barkley a=1.1 b=0.17 eps=0.02): alpha and
beta in sliding 30-unit windows. Left: alpha(t) per ring. Right: alpha(K), points
coloured by window time. If alpha = alpha(K) after some t_s, late windows of
different rings fall on one curve. Saves late-time (t >= t_s) points for refits.

    python alpha_stationarity.py [t_s]
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import ellipse_compare as C

PERIOD, WIN, STEP = 4.44, 30.0, 10.0
INK, MUTED = "#1f1f1e", "#6b6a63"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def windows(path):
    d = np.load(path)
    ok = np.isfinite(d["Rm"]) & (d["ncurves"] >= 1)
    t, R, z = d["t"][ok], d["Rm"][ok], d["z"][ok]
    good = np.abs(R - median_filter(R, 25, mode="nearest")) < 0.4
    t, R, z = t[good], R[good], z[good]
    out = []
    for t0 in np.arange(10, t[-1] - WIN + 1e-9, STEP):
        m = (t >= t0) & (t < t0 + WIN)
        if m.sum() < 30 or R[m].min() < 1.5:
            continue
        out.append((t0 + WIN / 2, 1 / R[m].mean(), -0.5 * np.polyfit(t[m], R[m] ** 2, 1)[0],
                    np.polyfit(t[m], z[m], 1)[0] * R[m].mean()))
    return np.array(out)


if __name__ == "__main__":
    ts = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    files = sorted(glob.glob("circL_R0_*.npz"), key=lambda f: float(f.split("_")[-1][:-4]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), dpi=120)
    late = []
    for f, col in zip(files, COLS):
        w = windows(f)
        lab = "R0=" + f.split("_")[-1][:-4]
        print(lab)
        for tc, K, al, be in w:
            print(f"   t={tc:5.0f}  R={1/K:5.2f}  K={K:.3f}  alpha={al:+.3f}  beta={be:+.2f}")
        ax[0].plot(w[:, 0], w[:, 2], "o-", ms=3, color=col, label=lab)
        sc = ax[1].scatter(w[:, 1], w[:, 2], c=w[:, 0], cmap="viridis", s=18, vmin=0, vmax=300)
        ax[2].scatter(w[:, 1], w[:, 3], c=w[:, 0], cmap="viridis", s=18, vmin=0, vmax=300)
        late.append(w[w[:, 0] >= ts])
    ax[0].axvline(ts, color=MUTED, ls=":", lw=1)
    ax[0].set(xlabel="window centre time", ylabel="tension alpha")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xlabel="curvature K = 1/R", ylabel="tension alpha")
    ax[2].set(xlabel="curvature K = 1/R", ylabel="binormal beta")
    fig.colorbar(sc, ax=ax[1:], label="window time", shrink=0.85)
    for a in ax:
        a.axhline(0, color=MUTED, lw=0.6)
        a.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Circular scroll rings, Barkley a=1.1 b=0.17 eps=0.02: tension vs time and curvature",
                 fontsize=10, color=INK)
    fig.savefig("alpha_stationarity.png", bbox_inches="tight")
    late = np.concatenate([l for l in late if len(l)])
    np.save("alpha_late.npy", late)
    print(f"late (t>={ts:g}) points: {len(late)}  K range {late[:, 1].min():.3f}-{late[:, 1].max():.3f}")
    print("wrote alpha_stationarity.png, alpha_late.npy")
