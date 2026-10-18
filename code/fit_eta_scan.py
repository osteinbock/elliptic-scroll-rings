"""Rigidity scan with circle-calibrated alpha(K), beta(K) (Barkley a=1.1 b=0.17 eps=0.02).

Circular rings measured at R = 5.1 and 7.28:  alpha_c = 0.010, 0.100;  beta_c = -1.68, -1.88.
Model: alpha(Rbar) = A0 + (C - eta)/Rbar^2, beta(Rbar) = B0 + B2/Rbar^2 (Rbar = L/2pi, uniform along
the loop) plus rigidity -eta P(X_ssss), whose own circle shrinkage eta/Rbar^2 is subtracted so circles match.
Only eta is scanned; each value is scored against BOTH elliptic RD rings.
"""
import numpy as np
from multiprocessing import Pool
import filament_flow as F
import ellipse_compare as C

R1, R2 = 5.1, 7.28
a1, a2, b1, b2 = 0.010, 0.100, -1.68, -1.88
Cc = (a1 - a2) / (1 / R1 ** 2 - 1 / R2 ** 2)
A0 = a1 - Cc / R1 ** 2
B2 = (b1 - b2) / (1 / R1 ** 2 - 1 / R2 ** 2)
B0 = b1 - B2 / R1 ** 2
T0 = 6.0
RINGS = {"8x5": "ell_a1.1_b0.17_8x5.npz", "12x7.5": "ell_a1.1_b0.17_12x7.5.npz"}


def rd(path):
    d = np.load(path)
    ok = np.isfinite(d["hx"])
    t = d["t"][ok]
    hx, hy, h, L = (C.smooth(t, d[k][ok], 4.44) for k in ("hx", "hy", "h2c", "length"))
    return t, hx, hy, h, L


def one(args):
    eta, key, lam = args if len(args) == 3 else (*args, 1.0)
    t, hx, hy, h, L = rd(RINGS[key])
    i = np.nanargmin(abs(t - T0))
    # ring-size-dependent coefficients; subtract rigidity's own circle shrinkage eta/R^2
    f = lambda R: (A0 + (Cc - eta) / R ** 2, lam * (B0 + B2 / R ** 2))
    r = F.run(hx[i], hy[i], A0, B0, t[-1] - T0, n=32, dt=0.04, eta=eta, coeff_of_R=f)
    r["t"] = r["t"] + T0
    m = (t >= T0) & np.isfinite(hx) & np.isfinite(h)
    e = (hx - hy)[m] / 2
    em = np.interp(t[m], r["t"], (r["hx"] - r["hy"]) / 2)
    hm = np.interp(t[m], r["t"], r["h2c"])
    Lm = np.interp(t[m], r["t"], r["length"])
    rms = np.sqrt(np.mean((em - e) ** 2) + np.mean((hm - h[m]) ** 2))
    return eta, key, lam, rms, np.sqrt(np.mean((Lm - L[m]) ** 2)), r


if __name__ == "__main__":
    import sys
    print(f"A0={A0:.4f} C=A2+eta={Cc:.3f} B0={B0:.3f} B2={B2:.2f}")
    lams = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1.0]
    etas = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 0.5, 1, 1.5, 2, 3]
    jobs = [(e, k, l) for l in lams for e in etas for k in RINGS]
    with Pool(18) as p:
        res = p.map(one, jobs)
    out, tab = {}, {}
    for eta, key, lam, rms, rmsL, r in res:
        out[f"{key}_eta{eta:g}_lam{lam:g}"] = r
        tab.setdefault((lam, eta), {})[key] = (rms, rmsL)
    print("lam   eta   rms(8x5) rms(12x7.5)  combined   lenrms(8x5) lenrms(12x7.5)")
    for (lam, eta), v in sorted(tab.items()):
        a, b = v["8x5"], v["12x7.5"]
        print(f"{lam:4.2f} {eta:5.2f}   {a[0]:.3f}    {b[0]:.3f}      {np.hypot(a[0], b[0]):.3f}      {a[1]:.2f}        {b[1]:.2f}")
    np.save("eta_scan.npy", out, allow_pickle=True)
