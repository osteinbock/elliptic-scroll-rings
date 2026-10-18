"""Panfilov kinetics B: elliptic ring vs prediction from circular-ring coefficients.

alpha(K), beta(K): linear fits to period-averaged windows of all circular rings
with the same gf (pan_alpha.windows). Predictions:
  (1) per half-cycle: T_half = pi / Im(lambda), lambda from Eq. (modes) with m = 2,
      alpha and beta at the half-cycle's mean radius;
  (2) filament equation with ring-size-dependent alpha(Rbar), beta(Rbar), started
      from the RD ring's period-averaged half-widths at t0.
    python pan_compare.py ELLIPSE.npz gf period t0 out.png
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C
import filament_flow as F
import pan_alpha as PA

INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"


def merge(x, gap):
    """Merge clusters of crossings closer than gap (jitter while h_x ~ h_y) into their mean."""
    out, cur = [], []
    for v in x:
        if cur and v - cur[-1] > gap:
            out.append(np.mean(cur)); cur = []
        cur.append(v)
    if cur:
        out.append(np.mean(cur))
    return out


def coeff_fits(gf):
    pts = []
    for f in glob.glob(f"pcirc_gf{gf:g}_R*.npz"):
        _, w = PA.windows(f)
        if len(w):
            pts.append(w)
    w = np.concatenate(pts)
    K = 1 / w[:, 1]
    pa = np.polyfit(K, w[:, 2], 1)
    pb = np.polyfit(K, w[:, 3], 1)
    return pa, pb, K, w


def main():
    path, gf, per, t0, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    pa, pb, K, w = coeff_fits(gf)
    print(f"circle fits (K range {K.min():.3f}-{K.max():.3f}): alpha = {pa[1]:+.3f} {pa[0]:+.3f} K ;"
          f" beta = {pb[1]:+.3f} {pb[0]:+.3f} K  (n={len(K)})")
    d = np.load(path)
    ok = np.isfinite(d["hx"]) & (d["ncurves"] >= 1)
    t = d["t"][ok]
    hx, hy, h, L = (C.smooth(t, d[k][ok], per) for k in ("hx", "hy", "h2c", "length"))
    Rb = L / (2 * np.pi)
    x = merge(C.crossings(t, hx - hy), per)
    print("RD axis switches:", np.round(x, 1))
    # (1) per half-cycle linear prediction
    rows = []
    for a, b in zip(x[:-1], x[1:]):
        m = (t >= a) & (t <= b) & np.isfinite(Rb)
        R = Rb[m].mean()
        al, be = np.polyval(pa, 1 / R), np.polyval(pb, 1 / R)
        im = np.sqrt(max(48 * be ** 2 - al ** 2, 1e-12)) / (2 * R ** 2)
        rows.append((a, b, b - a, R, al, be, np.pi / im))
        print(f"  half-cycle {a:6.1f}-{b:6.1f}: T={b-a:6.1f}  Rbar={R:5.2f}  alpha={al:+.3f} beta={be:+.3f}"
              f"  predicted T={np.pi/im:6.1f}  ratio={(b-a)/(np.pi/im):.3f}")
    rows = np.array(rows)
    if len(rows):
        q = rows[:, 2] / rows[:, 6]
        print(f"observed/predicted half-period: mean {q.mean():.3f}, sd {q.std():.3f}, n={len(q)}")
    # (2) filament equation from the RD shape at t0
    i0 = np.nanargmin(np.abs(t - t0))
    # negative tension makes the filament equation ill-posed (no rigidity term):
    # clip alpha at zero (the circle data at gf = 0.775 are consistent with alpha = 0)
    f = lambda R: (max(np.polyval(pa, 1 / R), 0.0), np.polyval(pb, 1 / R))
    r = F.run(hx[i0], hy[i0], 0.0, 0.0, t[-1] - t[i0], n=48, dt=0.2, coeff_of_R=f)
    r["t"] = r["t"] + t[i0]
    xm = merge(C.crossings(r["t"], r["hx"] - r["hy"]), per)
    print("model switches:", np.round(xm, 1))
    np.savez(out.replace(".png", "_data.npz"), t=t, hx=hx, hy=hy, h=h, Rb=Rb, rows=rows,
             mt=r["t"], mhx=r["hx"], mhy=r["hy"], mh=r["h2c"], mL=r["length"], pa=pa, pb=pb)

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), dpi=120, sharex=True)
    ax[0].plot(t, hx, color=C1, lw=2, label="RD  $h_x$")
    ax[0].plot(t, hy, color=C2, lw=2, label="RD  $h_y$")
    ax[0].plot(r["t"], r["hx"], color=C1, lw=1.3, ls="--", label="filament eq.")
    ax[0].plot(r["t"], r["hy"], color=C2, lw=1.3, ls="--")
    ax[0].set_ylabel("half-width")
    ax[0].legend(frameon=False, fontsize=8, ncol=3)
    ax[1].plot(t, h, color=INK, lw=2, label="RD")
    ax[1].plot(r["t"], r["h2c"], color=MUTED, lw=1.3, ls="--", label="filament eq.")
    ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_ylabel("saddle amplitude")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(t, Rb, color=INK, lw=2, label="RD")
    ax[2].plot(r["t"], r["length"] / (2 * np.pi), color=MUTED, lw=1.3, ls="--", label="filament eq.")
    ax[2].set_ylabel(r"mean radius $\bar R$")
    ax[2].set_xlabel("time")
    ax[2].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.suptitle(f"Panfilov kinetics B, $g_f$ = {gf:g}: elliptic ring vs filament equation with "
                 "circular-ring coefficients", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
