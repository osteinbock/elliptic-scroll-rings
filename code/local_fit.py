"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Refit rigidity eta with LOCAL coefficients alpha(K), beta(K) from circular rings
(Barkley a=1.1 b=0.17 eps=0.02). The circle data give alpha_c(K) = alpha(K) + eta K^2
(the projected rigidity term alone shrinks a circle like a tension eta K^2), so the
model uses alpha(K) = alpha_c(K) - eta K^2. K is clamped to the measured range."""
import sys
import numpy as np
from multiprocessing import Pool
import filament_flow as F
import fit_eta_scan as S

late = np.load("alpha_late.npy")                                   # (t, K, alpha, beta), long rings, t >= 50
pts = late[:, 1:]
extra = np.array([[0.39, 0.10, -1.7]])                             # R0=4 ring (collapsing, rough)
W = np.r_[np.ones(len(pts)), 3.0]
P = np.r_[pts, extra]
PA = np.polyfit(P[:, 0], P[:, 1], 3, w=W)
PB = np.polyfit(P[:, 0], P[:, 2], 2, w=W)
KMIN, KMAX = 0.085, 0.42


def coeffs(eta, lam=1.0):
    def f(K):
        Kc = np.clip(K, KMIN, KMAX)
        return np.polyval(PA, Kc) - eta * K ** 2, lam * np.polyval(PB, Kc)
    return f


def one(args):
    eta, key, lam = args
    t, hx, hy, h, L = S.rd(S.RINGS[key])
    i = np.nanargmin(abs(t - S.T0))
    r = F.run(hx[i], hy[i], 0.0, 0.0, t[-1] - S.T0, n=32, dt=0.04, eta=eta, local=coeffs(eta, lam))
    r["t"] = r["t"] + S.T0
    m = (t >= S.T0) & np.isfinite(hx) & np.isfinite(h)
    e = (hx - hy)[m] / 2
    em = np.interp(t[m], r["t"], (r["hx"] - r["hy"]) / 2)
    hm = np.interp(t[m], r["t"], r["h2c"])
    Lm = np.interp(t[m], r["t"], r["length"])
    bad = not np.all(np.isfinite(r["length"]))
    rms = np.inf if bad else np.sqrt(np.mean((em - e) ** 2) + np.mean((hm - h[m]) ** 2))
    return eta, key, lam, rms, (np.inf if bad else np.sqrt(np.mean((Lm - L[m]) ** 2))), r


if __name__ == "__main__":
    kk = np.array([0.09, 0.12, 0.15, 0.19, 0.22, 0.3, 0.39])
    print("alpha_c(K):", dict(zip(kk, np.round(np.polyval(PA, kk), 3))))
    print("beta(K):   ", dict(zip(kk, np.round(np.polyval(PB, kk), 3))))
    lams = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1.0]
    etas = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 0.5, 1, 2, 3, 5]
    jobs = [(e, k, l) for l in lams for e in etas for k in S.RINGS]
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
    np.save("local_scan.npy", out, allow_pickle=True)
