"""Tension alpha and binormal coefficient beta of circular Panfilov kinetics-B
scroll rings (pcirc_gf*_R*.npz): period-averaged R(t), z(t); sliding windows of
two rotation periods; alpha = -1/2 dR^2/dt, beta = R dz/dt.
    python pan_alpha.py
"""
import glob
import numpy as np
import ellipse_compare as C

PERIOD = {0.8: 36.9, 0.825: 34.3, 0.85: 32.0, 0.9: 28.2, 0.95: 25.3, 0.775: 40.0}


def windows(path, tstart=None, single=False):
    d = np.load(path)
    gf = float(d["gf"])
    per = PERIOD[round(gf, 3)]
    ok = np.isfinite(d["Rm"]) & ((d["ncurves"] == 1) if single else (d["ncurves"] >= 1)) & (d["Rm"] > 2)
    t, R, z = d["t"][ok], d["Rm"][ok], d["z"][ok]
    Rs, zs = C.smooth(t, R, per), C.smooth(t, z, per)
    out = []
    t0s = tstart if tstart is not None else 2 * per
    for a in np.arange(t0s, t[-1] - 2 * per + 1e-9, per):
        m = (t >= a) & (t < a + 2 * per) & np.isfinite(Rs)
        if m.sum() < 20:
            continue
        al = -0.5 * np.polyfit(t[m], Rs[m] ** 2, 1)[0]
        be = np.polyfit(t[m], zs[m], 1)[0] * Rs[m].mean()
        out.append((a + per, Rs[m].mean(), al, be))
    return gf, np.array(out)


if __name__ == "__main__":
    for f in sorted(glob.glob("pcirc_gf*_R*.npz")):
        gf, w = windows(f)
        if not len(w):
            print(f, "no windows"); continue
        print(f"{f}: R {w[0,1]:.1f}->{w[-1,1]:.1f}")
        for tc, R, al, be in w[::2]:
            print(f"   t={tc:5.0f} R={R:5.2f} alpha={al:+.3f} beta={be:+.3f} |beta/alpha|={abs(be/al) if al else np.inf:.1f}")
        print(f"   MEAN alpha={w[:,2].mean():+.3f}+-{w[:,2].std():.3f}  beta={w[:,3].mean():+.3f}+-{w[:,3].std():.3f}")
