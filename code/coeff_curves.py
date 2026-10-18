"""alpha(K) and beta(K) from circular scroll rings (local slopes in sliding windows).

For each run: smooth R(t), z(t) over one rotation period, then in 10-unit windows
alpha = -1/2 d(R^2)/dt, beta = R dz/dt, K = 1/<R>. Pools all runs and fits
alpha(K), beta(K) as quadratics in K (least squares), plus a plot.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C

PERIOD, WIN, TSTART = 4.44, 30.0, 15.0
FILES = sorted(glob.glob("circ_R0_*.npz")) + ["circ_a1.1_b0.17.npz", "circ_a1.1_b0.17_R9.npz"]


def windows(path):
    d = np.load(path)
    from scipy.ndimage import median_filter
    ok = np.isfinite(d["Rm"]) & (d["ncurves"] >= 1)
    t, Rraw, zraw = d["t"][ok], d["Rm"][ok], d["z"][ok]
    good = np.abs(Rraw - median_filter(Rraw, 25, mode="nearest")) < 0.4    # drop wrong-curve frames
    t, Rraw, zraw = t[good], Rraw[good], zraw[good]
    R = C.smooth(t, Rraw, PERIOD)
    z = C.smooth(t, zraw, PERIOD)
    out = []
    for t0 in np.arange(TSTART, t[-1] - WIN + 1e-9, WIN / 2):
        m = (t >= t0) & (t < t0 + WIN) & np.isfinite(R)
        if m.sum() < 40 or np.nanmin(R[m]) < 1.5:
            continue
        al = -0.5 * np.polyfit(t[m], R[m] ** 2, 1)[0]
        be = np.polyfit(t[m], z[m], 1)[0] * R[m].mean()
        out.append((1 / R[m].mean(), al, be))
    return np.array(out)


def fit():
    pts = np.concatenate([w for w in (windows(f) for f in FILES) if len(w)])
    K, A, Bt = pts.T
    pa = np.polyfit(K, A, 2)
    pb = np.polyfit(K, Bt, 2)
    return pts, pa, pb


if __name__ == "__main__":
    pts, pa, pb = fit()
    for f in FILES:
        w = windows(f)
        if len(w):
            print(f"{f:28s} K {w[0,0]:.3f}->{w[-1,0]:.3f}  alpha {w[0,1]:+.3f}->{w[-1,1]:+.3f}  beta {w[:,2].mean():+.2f}")
    print("alpha(K) quadratic coeffs (K^2, K, 1):", np.round(pa, 3))
    print("beta(K)  quadratic coeffs (K^2, K, 1):", np.round(pb, 3))
    print("K range", pts[:, 0].min(), pts[:, 0].max())
    np.save("coeff_curves.npy", dict(pts=pts, pa=pa, pb=pb), allow_pickle=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    kk = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
    for a_, j, lab in ((ax[0], 1, "tension alpha"), (ax[1], 2, "binormal beta")):
        a_.scatter(pts[:, 0], pts[:, j], s=12, color="#2a78d6", label="RD circle windows")
        a_.plot(kk, np.polyval(pa if j == 1 else pb, kk), color="#1f1f1e", lw=1.5, label="quadratic fit")
        a_.axhline(0, color="#6b6a63", lw=0.6)
        a_.set_xlabel("curvature K = 1/R")
        a_.set_ylabel(lab)
        a_.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Barkley a=1.1 b=0.17 eps=0.02: filament coefficients vs ring curvature", fontsize=10)
    fig.tight_layout()
    fig.savefig("coeff_curves.png")
    print("wrote coeff_curves.png")
