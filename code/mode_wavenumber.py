"""Dependence of the out-of-plane coupling beta_zeta on the wavenumber k = m/R (Fig. 6b, Table S11).

1. Large rings at b = 0.17 (m = 2, 3, 4; R about 11 to 15), with the fits in tau of the larger ring in
   mode_refit.py: the whole interval and its halves.
2. Rings at b = 0.15 and 0.16, which shrink quickly: fits in tau in windows of about one oscillation,
   overlapping by half, keeping windows with an amplitude of at least AMP_MIN.
3. Comparison of all rings at b = 0.17 with the circular-ring drift coefficient at K = k, and with the
   binormal coefficient of straight filaments at equal k.

    python mode_wavenumber.py        (REVISION holds the runs; writes REVISION/mode_wavenumber.json)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ellipse_compare as C
import halfcycle_uniform as HU
import folders as FD
from mode_refit import tau_fit, local_law

PERIOD = {0.15: 4.13, 0.16: 4.318, 0.17: 4.44}     # rotation periods of the planar spirals
LARGE = ((2, "modeXL_m2.npz", 150.0), (3, "modeXL_m3.npz", 110.0), (4, "modeXL_m4.npz", 110.0))
WINDOWS = (("modeB15X_m3.npz", 3, 0.15, 30.0), ("modeB15X_m4.npz", 4, 0.15, 30.0), ("modeB16L_m2.npz", 2, 0.16, 50.0),
           ("modeB16X_m3.npz", 3, 0.16, 35.0), ("modeB16X_m4.npz", 4, 0.16, 30.0))
AMP_MIN = 0.05
T0_LARGE, T0_FAST = 15.0, 10.0


def beta_log(K):
    """The logarithmic form of the b = 0.17 calibration (SI Sec. III)."""
    return 0.69 * np.log(2.8 / np.asarray(K, float))


def signals(name, m, b):
    d = np.load(os.path.join(FD.REVISION, name))
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    return t, {k: C.smooth(t, d[k][ok], PERIOD[b]) for k in (f"r{m}c", f"z{m}c", "Rm", "z")}


def fit(t, S, m, t0, t1):
    return tau_fit(t, S[f"r{m}c"], S[f"z{m}c"], S["Rm"], S["z"], m, t0, t1)


def large_rings():
    cal = HU.barkley_calibration(0.17)
    out = []
    for m, name, t1 in LARGE:
        t, S = signals(name, m, 0.17)
        mid = 0.5 * (T0_LARGE + t1)
        for t0w, t1w in ((T0_LARGE, t1), (T0_LARGE, mid), (mid, t1)):
            r = fit(t, S, m, t0w, t1w)
            r.update(local_law(cal, r["R"], m))
            r["run"] = name
            r["whole"] = t1w == t1 and t0w == T0_LARGE
            out.append(r)
    return out


def fast_windows(t_first=T0_FAST):
    cal = {b: HU.barkley_calibration(b) for b in (0.15, 0.16)}
    out = []
    for name, m, b, W in WINDOWS:
        t, S = signals(name, m, b)
        t0 = t_first
        while t0 + W <= t[-1] + 1:
            r = fit(t, S, m, t0, t0 + W)
            sel = (t >= t0) & (t <= t0 + W)
            amp = float(np.nanmin(np.hypot(S[f"r{m}c"][sel], S[f"z{m}c"][sel])))
            K = 1 / r["R"]
            lo, hi = cal[b].Krange
            r.update(run=name, b=b, amp_min=amp, beta_c=float(np.polyval(cal[b].pb, K)), K_inside=bool(lo <= K <= hi),
                     beta_log=float(beta_log(r["k"])), used=bool(amp >= AMP_MIN and r["ok"]))
            out.append(r)
            t0 += W / 2
    return out


def main():
    res = dict(large=large_rings(), windows=fast_windows())
    json.dump(res, open(os.path.join(FD.REVISION, "mode_wavenumber.json"), "w"), indent=1)
    print("large rings at b = 0.17:")
    for r in res["large"]:
        print(f"  {r['run']:<14} t {r['t0']:5.1f}-{r['t1']:5.1f} R {r['R_start']:.2f}->{r['R_end']:.2f} R_m={r['R']:.2f} k={r['k']:.3f} "
              f"bz={r['bz']:.3f} br={r['br']:.3f} bz/bc={r['bz'] / r['beta_c']:.3f} br/bc={r['br'] / r['beta_c']:.3f} "
              f"Eq5/bc={r['beta_r_loc'] / r['beta_c']:.3f} br/Eq5={r['br'] / r['beta_r_loc']:.3f} bz/log(k)={r['bz'] / beta_log(r['k']):.3f} "
              f"az={r['alpha_zeta']:+.2f} ar={r['alpha_r']:+.2f} ok={r['ok']}")
    for label, rows in (("start 10", res["windows"]), ("start 15 (sensitivity)", fast_windows(15.0))):
        print(f"\nwindows at b = 0.15 and 0.16, {label}:")
        for r in rows:
            print(f"  {r['run']:<14} b={r['b']} t {r['t0']:5.1f}-{r['t1']:5.1f} R_m={r['R']:5.2f} k={r['k']:.3f} bz={r['bz']:.3f} br={r['br']:.3f} "
                  f"bz/log(k)={r['bz'] / r['beta_log']:.3f} bz/bc={r['bz'] / r['beta_c']:.3f}{'' if r['K_inside'] else '*'} amp={r['amp_min']:.3f} "
                  f"{'used' if r['used'] else 'skipped'}")
        for b in (0.15, 0.16):
            u = [r for r in rows if r["b"] == b and r["used"]]
            q = [r["bz"] / r["beta_log"] for r in u]
            c = [r["bz"] / r["beta_c"] for r in u]
            print(f"  b={b}: {len(u)} windows, k {min(r['k'] for r in u):.2f}-{max(r['k'] for r in u):.2f}, bz/log(k) {min(q):.2f}-{max(q):.2f} "
                  f"mean {np.mean(q):.2f}; bz/bc {min(c):.2f}-{max(c):.2f}")


if __name__ == "__main__":
    main()
