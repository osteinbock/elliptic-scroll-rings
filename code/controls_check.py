"""Controls of the revision runs (referee comments A6, C4b): straight filaments outside the b = 0.17 band,
a larger domain for a mode run, a halved time step and a seed turned by 45 degrees.

    python controls_check.py          (reads BARKLEY and REVISION, see folders.py; writes REVISION/controls.json)
"""
import os
import json
import numpy as np

import ellipse_compare as C
from sf2_analysis import csmooth, local_rates
from mode_theory2 import fit_mode

import folders as FD


def straight_rates(path):
    """Zero-amplitude gamma/k^2 and |beta| as in sf2_linear.py (10-unit windows, kA < 0.45)."""
    d = np.load(path)
    t, k = d["t"], float(d["k"])
    s = csmooth(t, d["uk"])
    r = local_rates(t, s, width=10, step=2.5, floor=0.02)
    r = r[(r[:, 0] - 5 >= 12) & (k * r[:, 3] < 0.45)]
    x = (k * r[:, 3]) ** 2
    G, Bt = r[:, 1] / k ** 2, np.abs(r[:, 2]) / k ** 2
    if len(r) >= 3 and np.ptp(x) > 1e-3:
        return k, float(np.polyfit(x, G, 1)[1]), float(np.polyfit(x, Bt, 1)[1]), len(r)
    return k, float(G.mean()), float(Bt.mean()), len(r)


def mode_fit(path, m=2, t1=125.0):
    d = np.load(path)
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    S = {k: C.smooth(t, d[k][ok], 4.44) for k in (f"r{m}c", f"z{m}c", "Rm", "z")}
    sel = (t >= 6) & (t <= t1) & np.isfinite(S[f"r{m}c"]) & np.isfinite(S[f"z{m}c"])
    R = S["Rm"][sel].mean()
    f = fit_mode(t[sel], S[f"r{m}c"][sel], S[f"z{m}c"][sel], m * np.sqrt(m * m - 1) * 1.2 / R ** 2)
    W = f["omega"] * R ** 2 / (m * np.sqrt(m * m - 1))
    Q = f["ratio"] / np.sqrt((m * m - 1) / m ** 2)
    bd = abs(np.polyfit(t[sel], S["z"][sel], 1)[0]) * R
    return dict(R=float(R), omega=float(f["omega"]), q=float(f["ratio"]), gamma=float(f["gamma"]),
                bz_bd=float(W / Q / bd), br_bd=float(W * Q / bd))


def switches(path, key="r2c", T=80.0):
    d = np.load(path)
    ok = np.isfinite(d[key]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    y = C.smooth(t, d[key][ok], 4.44)
    L = C.smooth(t, d["length"][ok], 4.44)
    z = C.smooth(t, d["z"][ok], 4.44)
    x = [c for c in C.crossings(t, y) if c <= T]
    m = (t > 10) & (t < T) & np.isfinite(z) & np.isfinite(L)
    v = np.polyfit(t[m], z[m], 1)[0]
    return np.array(x), float(abs(v)), float(np.nanmean(L[m]) / (2 * np.pi))


def main():
    out = {}
    print("straight filaments (Lxy = 32, A = 0.5; zero-amplitude extrapolation):")
    st = []
    for lab, path in (("b=0.17 k=0.35", f"{FD.BARKLEY}/sf2/L32_k035.npz"), ("b=0.17 k=0.45", f"{FD.BARKLEY}/sf2/L32_k045.npz"),
                      ("b=0.17 k=0.60", f"{FD.BARKLEY}/sf2/L32_k060.npz"), ("b=0.15 k=0.35", f"{FD.REVISION}/sf2/b015_k035.npz"),
                      ("b=0.15 k=0.45", f"{FD.REVISION}/sf2/b015_k045.npz"), ("b=0.15 k=0.60", f"{FD.REVISION}/sf2/b015_k060.npz"),
                      ("b=0.16 k=0.45", f"{FD.REVISION}/sf2/b016_k045.npz")):
        k, g, bb, n = straight_rates(path)
        st.append(dict(label=lab, k=k, gamma_over_k2=g, beta=bb, windows=n))
        print(f"  {lab}: gamma/k^2 = {g:+.3f} ({'decays' if g > 0 else 'grows'}), growth rate {-g * k * k:+.4f},"
              f" |beta| = {bb:.3f}  ({n} windows)")
    out["straight"] = st
    a, bbx = mode_fit(f"{FD.BARKLEY}/modeR_m2_a0.08.npz"), mode_fit(f"{FD.REVISION}/modeR_m2_a0.08_bigbox.npz")
    out["bigbox"] = dict(standard=a, bigbox=bbx)
    print(f"\nmode m=2 amp 0.08, 26^2 x 18 vs 36^2 x 24: omega {a['omega']:.4f} vs {bbx['omega']:.4f}; q {a['q']:.4f} vs {bbx['q']:.4f};"
          f" bz/bd {a['bz_bd']:.3f} vs {bbx['bz_bd']:.3f}; br/bd {a['br_bd']:.3f} vs {bbx['br_bd']:.3f};"
          f" decay {a['gamma']:.4f} vs {bbx['gamma']:.4f}; R {a['R']:.3f} vs {bbx['R']:.3f}")
    ref = switches(f"{FD.BARKLEY}/rerun_r2/ell_a1.1_b0.17_8x5.npz")
    dt4 = switches(f"{FD.REVISION}/ell_8x5_dt04.npz")
    rot = switches(f"{FD.REVISION}/ell_8x5_rot45.npz", key="r2s")
    n = min(len(ref[0]), len(dt4[0]), len(rot[0]))
    out["switches"] = dict(reference=ref[0].tolist(), dt_half=dt4[0].tolist(), rot45=rot[0].tolist(),
                           drift=[ref[1], dt4[1], rot[1]], R=[ref[2], dt4[2], rot[2]])
    print(f"\n8x5 ellipse, switch times (r2 sign changes) up to t = 80:\n  reference  {np.round(ref[0], 2).tolist()}\n"
          f"  dt / 2     {np.round(dt4[0], 2).tolist()}  max difference {np.max(np.abs(dt4[0][:n] - ref[0][:n])):.3f}\n"
          f"  turned 45  {np.round(rot[0], 2).tolist()}  max difference {np.max(np.abs(rot[0][:n] - ref[0][:n])):.3f}\n"
          f"  drift speed {ref[1]:.4f} / {dt4[1]:.4f} / {rot[1]:.4f}; mean L/2pi {ref[2]:.3f} / {dt4[2]:.3f} / {rot[2]:.3f}")
    json.dump(out, open(os.path.join(FD.REVISION, "controls.json"), "w"), indent=1)
    print("wrote", os.path.join(FD.REVISION, "controls.json"))


if __name__ == "__main__":
    main()
