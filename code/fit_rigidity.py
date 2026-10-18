"""Fit beta and rigidity eta of the filament equation to an RD elliptic ring.

alpha is tied to the measured circular-ring tension: alpha = alpha_circ - eta/R^2
(the projected -eta X_ssss term alone shrinks a circle like a tension eta/R^2).
Fitted: beta, eta, and the semi-axes of the starting ellipse at time t0.
Residual: smoothed RD (hx-hy)/2 and saddle amplitude h2c vs the model.

    python fit_rigidity.py RD.npz period t0 Ax0 Ay0 beta0 eta0 [alpha_circ]
"""
import sys
import numpy as np
from scipy.optimize import minimize
import filament_flow as F
import ellipse_compare as C


def load(path, period, t0):
    d = np.load(path)
    ok = np.isfinite(d["hx"])
    t = d["t"][ok]
    e = (C.smooth(t, d["hx"][ok], period) - C.smooth(t, d["hy"][ok], period)) / 2
    h = C.smooth(t, d["h2c"][ok], period)
    L = C.smooth(t, d["length"][ok], period)
    m = (t >= t0) & np.isfinite(e) & np.isfinite(h)
    return t[m], e[m], h[m], L[m]


def model(p, t0, T, alpha_circ, n=32):
    beta, eta, Ax0, Ay0 = p
    R = F.ellipse(Ax0, Ay0, 256)
    Rbar = np.linalg.norm(np.diff(np.r_[R, R[:1]], axis=0), axis=1).sum() / (2 * np.pi)
    alpha = alpha_circ - eta / Rbar ** 2
    r = F.run(Ax0, Ay0, alpha, beta, T - t0, n=n, eta=max(eta, 0.0), dt=0.04)
    r["t"] = r["t"] + t0
    return r, alpha


def resid(p, data, t0, alpha_circ):
    t, e, h, _ = data
    if p[1] < 0 or p[2] <= p[3] * 0.5:
        return 1e3
    r, _ = model(p, t0, t[-1], alpha_circ)
    em = np.interp(t, r["t"], (r["hx"] - r["hy"]) / 2)
    hm = np.interp(t, r["t"], r["h2c"])
    return np.mean((em - e) ** 2) + np.mean((hm - h) ** 2)


if __name__ == "__main__":
    path, period, t0, Ax0, Ay0, b0, eta0 = sys.argv[1], *map(float, sys.argv[2:8])
    alpha_circ = float(sys.argv[8]) if len(sys.argv) > 8 else 0.01
    data = load(path, period, t0)
    if eta0 > 0:      # full fit: beta, eta, start shape
        res = minimize(resid, [b0, eta0, Ax0, Ay0], args=(data, t0, alpha_circ), method="Nelder-Mead",
                       options=dict(xatol=2e-3, fatol=1e-5, maxiter=160))
        beta, eta, Ax, Ay = res.x
    else:             # rigidity off: beta and start shape only
        res = minimize(lambda q: resid([q[0], 0.0, q[1], q[2]], data, t0, alpha_circ),
                       [b0, Ax0, Ay0], method="Nelder-Mead", options=dict(xatol=2e-3, fatol=1e-5, maxiter=120))
        (beta, Ax, Ay), eta = res.x, 0.0
    _, alpha = model([beta, eta, Ax, Ay], t0, t0 + 1, alpha_circ)
    print(f"FIT beta={beta:.3f} eta={eta:.3f} alpha={alpha:.4f} start {Ax:.2f}x{Ay:.2f}  rms={np.sqrt(res.fun):.4f}  nfev={res.nfev}")
    np.save(path.replace(".npz", f"_fit_eta{'free' if eta0 > 0 else '0'}.npy"), [beta, eta, Ax, Ay, alpha, np.sqrt(res.fun)])
