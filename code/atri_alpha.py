"""Tension alpha and drift coefficient beta (um^2/s) of circular Atri Ca2+ scroll rings.
Single-curve frames are averaged over one rotation period; alpha and beta are fitted in
400-s windows (alpha = -1/2 dR^2/dt, beta = R dz/dt).
    python atri_alpha.py file.npz period [file.npz period ...]
"""
import sys
import numpy as np


def binned(path, per):
    d = np.load(path)
    ok = (d["ncurves"] == 1) & np.isfinite(d["Rm"]) & (d["length"] > 300)
    t, R, z = d["t"][ok], d["Rm"][ok], d["z"][ok]
    edges = np.arange(0, t[-1] + per, per)
    tc, Rc, zc = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 5:
            tc.append(t[m].mean()); Rc.append(R[m].mean()); zc.append(z[m].mean())
    return np.array(tc), np.array(Rc), np.array(zc), ok.mean()


def fit(path, per, win=400.0, t0=100.0):
    tc, Rc, zc, frac = binned(path, per)
    rows = []
    for a in np.arange(t0, tc[-1] - win + 1e-9, win / 2):
        m = (tc >= a) & (tc < a + win)
        if m.sum() > 5:
            al = -0.5 * np.polyfit(tc[m], Rc[m] ** 2, 1)[0]
            be = np.polyfit(tc[m], zc[m], 1)[0] * Rc[m].mean()
            rows.append((a + win / 2, Rc[m].mean(), al, be))
    return np.array(rows), frac, (tc, Rc)


if __name__ == "__main__":
    args = sys.argv[1:]
    for path, per in zip(args[::2], map(float, args[1::2])):
        rows, frac, (tc, Rc) = fit(path, per)
        print(f"{path}: single-curve fraction {frac:.2f}, R {Rc[0]:.0f} -> {Rc[-1]:.0f} um")
        for tm, R, al, be in rows:
            print(f"   t={tm:5.0f}  R={R:6.1f}  alpha={al:+6.2f}  beta={be:+6.2f}")
        if len(rows):
            print(f"   MEAN alpha={rows[:,2].mean():+.2f} +- {rows[:,2].std():.2f}   |beta|={abs(rows[:,3]).mean():.2f} +- {rows[:,3].std():.2f}")
