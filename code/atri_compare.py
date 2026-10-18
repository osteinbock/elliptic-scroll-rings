"""Atri Ca2+ model: elliptic scroll ring vs prediction from circular rings.

Frames with a single filament curve are averaged over one rotation period (17.5 s).
Half-periods from zero crossings of the saddle amplitude zeta_2 (z2c), normalized by
the period-averaged mean radius; prediction pi Rbar^2 / (2 sqrt3 |beta|) with alpha
included via Im(lambda) of Eq. (modes). Also: filament equation with alpha = 0,
|beta| from circular rings, started from the averaged ring shape.
    python atri_compare.py aell_130x80.npz beta alpha out.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C
import filament_flow as F

import os
PER = float(os.environ.get("ATRI_PER", "17.5"))
INK, MUTED, C1, C2 = "#1f1f1e", "#6b6a63", "#2a78d6", "#eb6834"


def binned(d, keys):
    ok = (d["ncurves"] == 1) & np.isfinite(d["Rm"]) & (d["length"] > 300)
    t = d["t"][ok]
    edges = np.arange(0, t[-1] + PER, PER)
    out = {k: [] for k in ["t"] + keys}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 5:
            out["t"].append(t[m].mean())
            for k in keys:
                out[k].append(d[k][ok][m].mean())
    return {k: np.array(v) for k, v in out.items()}


def main():
    path, beta, alpha, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    d = np.load(path)
    B = binned(d, ["hx", "hy", "Rm", "z2c", "r2c", "length"])
    t = B["t"]
    xz = C.crossings(t, B["z2c"])
    xr = C.crossings(t, B["hx"] - B["hy"])
    print("saddle zero crossings:", np.round(xz, 0))
    print("h_x = h_y crossings  :", np.round(xr, 0))
    q = []
    for a, b in zip(xz[:-1], xz[1:]):
        m = (t >= a) & (t <= b)
        R = B["Rm"][m].mean()
        im = np.sqrt(max(48 * beta ** 2 - alpha ** 2, 1e-12)) / (2 * R ** 2)
        q.append((a, b, b - a, R, np.pi / im))
        print(f"  half-cycle {a:6.0f}-{b:6.0f}: T={b-a:6.1f}  Rbar={R:6.1f}  predicted={np.pi/im:6.1f}  ratio={(b-a)/(np.pi/im):.3f}")
    q = np.array(q)
    if len(q):
        r = q[:, 2] / q[:, 4]
        print(f"ratio observed/predicted (linear): mean {r.mean():.3f}, sd {r.std():.3f}, n={len(r)}")
    # finite-amplitude prediction: start the filament equation (alpha = 0) from the planar
    # ellipse measured at each saddle zero crossing; predicted half-period = time to the
    # model's next planar moment (next zero crossing of its saddle amplitude)
    # The model supplies only the relative finite-amplitude factor f = T_model / T_lin at the
    # measured aspect ratio (radius-independent); the prediction is f times the linear
    # half-period at the mean radius Rbar used for the beta calibration.
    fa = []
    for (a, b, Tobs, R, Tlin_R), _ in zip(q, range(len(q))):
        hx0, hy0 = np.interp(a, t, B["hx"]), np.interp(a, t, B["hy"])
        h0 = (hx0 + hy0) / 2
        Tlin_h = np.pi * h0 ** 2 / (2 * np.sqrt(3) * beta)
        r_ = F.run(hx0, hy0, 0.0, beta, 2.5 * Tlin_h, n=48, dt=1.0)
        zc = [x for x in C.crossings(r_["t"], r_["h2c"]) if x > 0.2 * Tlin_h]
        if zc:
            f = zc[0] / Tlin_h
            fa.append((a, b, Tobs, f * Tlin_R, f))
            print(f"  finite-amplitude: start {a:6.0f} (aspect {max(hx0,hy0)/min(hx0,hy0):.2f}, factor {f:.3f}): "
                  f"observed {Tobs:6.1f}, predicted {f*Tlin_R:6.1f}, ratio {Tobs/(f*Tlin_R):.3f}")
    if fa:
        fa = np.array(fa)
        rr = fa[:, 2] / fa[:, 3]
        print(f"ratio observed/predicted (finite amplitude): mean {rr.mean():.3f}, sd {rr.std():.3f}, n={len(rr)}")
    i0 = 1
    mod = F.run(B["hx"][i0], B["hy"][i0], 0.0, beta, t[-1] - t[i0], n=48, dt=1.0)
    mod["t"] = mod["t"] + t[i0]
    print("model h_x = h_y crossings:", np.round(C.crossings(mod["t"], mod["hx"] - mod["hy"]), 0))
    np.savez(out.replace(".png", "_data.npz"), **B, q=q, mt=mod["t"], mhx=mod["hx"], mhy=mod["hy"], mh=mod["h2c"])

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), dpi=120, sharex=True)
    ax[0].plot(t, B["hx"], color=C1, lw=2, label=r"$h_x$")
    ax[0].plot(t, B["hy"], color=C2, lw=2, label=r"$h_y$")
    ax[0].plot(mod["t"], mod["hx"], color=C1, lw=1.2, ls="--")
    ax[0].plot(mod["t"], mod["hy"], color=C2, lw=1.2, ls="--")
    ax[0].plot([], [], color=MUTED, ls="--", label="filament eq.")
    ax[0].set_ylabel(r"half-width ($\mu$m)")
    ax[0].legend(frameon=False, fontsize=8, ncol=3)
    ax[1].plot(t, B["z2c"], color=INK, lw=2, label="simulation")
    ax[1].plot(mod["t"], mod["h2c"], color=MUTED, lw=1.2, ls="--", label="filament eq.")
    ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_ylabel(r"saddle amplitude ($\mu$m)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(t, B["Rm"], color=INK, lw=2)
    ax[2].set_ylabel(r"mean radius ($\mu$m)")
    ax[2].set_xlabel("time (s)")
    for a_ in ax:
        a_.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
