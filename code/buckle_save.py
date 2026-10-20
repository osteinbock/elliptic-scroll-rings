"""Buckling of the R0 = 38 ring: mode amplitudes, growth rates, and frequencies -> sf2/buckle.npz.

Growth rate from a log-linear fit of sqrt(|r_m|^2 + |z_m|^2) for t >= 90; frequency from the
exchange between radial and vertical parts, s = (|r_m|^2 - |z_m|^2)/A^2 ~ cos(2 omega t); the
binormal coefficient follows from omega = m sqrt(m^2 - 1) sqrt(beta_zeta beta_r) / R^2.

The fit of s is restricted to the principal Nyquist band, 0 < omega < pi/(2 dt) (dt: snapshot
spacing); boundary solutions are rejected. This selects the admissible sampled frequency but cannot
by itself exclude a higher-frequency component of the continuous signal aliased into that band. The
frequency is also refitted with the window starting at t = 80 and 100 (sensitivity). The
complex mode series fr_m(t), fz_m(t) are saved with the fits, so that
    python buckle_save.py --from-series sf2/buckle.npz
repeats the fits from the saved series without the filament snapshots.
"""
import re
import sys
import numpy as np
from scipy.optimize import curve_fit

MODES = (4, 8, 12, 16, 20)
FIT_MODES = (12, 16, 20)


def series_from_snapshots(path="sf2/circXXL_R0_38_snaps.npz"):
    from ring_spectra import spectra
    d = np.load(path)
    snaps = {}
    for key in d.files:
        m = re.match(r"snap_([\d.]+)_(\d+)$", key)
        if m:
            snaps.setdefault(float(m.group(1)), []).append(d[key])
    T = np.array(sorted(snaps))
    res = [spectra(snaps[t]) for t in T]
    R = np.array([r[0] for r in res])
    fr = {m: np.array([r[1][m] for r in res]) for m in MODES}
    fz = {m: np.array([r[2][m] for r in res]) for m in MODES}
    last = np.concatenate([l for l in snaps[190.0] if len(l) > 20])
    return T, R, fr, fz, last


def fit_frequency(t, s):
    """Bounded fit of s = a cos(2 w t + ph) + c over a scan of starting frequencies."""
    dts = np.median(np.diff(t))
    wmax = np.pi / (2 * dts)                       # 2 w below the Nyquist limit pi/dts
    f = lambda t, a, w, ph, c: a * np.cos(2 * w * t + ph) + c
    best, sols = None, []
    for w0 in np.linspace(0.05, 0.95 * wmax, 111):
        try:
            p, cov = curve_fit(f, t, s, p0=[0.8, w0, 0, 0],
                               bounds=([0, 1e-3, -np.inf, -np.inf], [np.inf, 0.999 * wmax, np.inf, np.inf]))
        except (RuntimeError, ValueError):
            continue
        if not (1e-3 * 1.01 < p[1] < 0.999 * wmax * 0.999):     # ended on a bound: rejected
            continue
        r2 = np.sum((f(t, *p) - s) ** 2)
        sols.append((r2, p[1]))
        if best is None or r2 < best[0]:
            best = (r2, p, cov)
    if best is None:
        raise RuntimeError("no admissible frequency fit")
    w = best[1][1]
    werr = float(np.sqrt(max(best[2][1, 1], 0))) if np.all(np.isfinite(best[2])) else np.nan
    other = [r for r, ww in sols if abs(ww - w) > 0.1 * w]
    contrast = (min(other) / best[0]) if other else np.inf
    return w, werr, contrast, 2 * w * dts / np.pi, best[1], f


def main(from_series=None):
    if from_series:
        d = np.load(from_series)
        T, R, last = d["T"], d["R"], d["last"]
        fr = {m: d[f"fr{m}"] for m in MODES}
        fz = {m: d[f"fz{m}"] for m in MODES}
    else:
        T, R, fr, fz, last = series_from_snapshots()
    amp = {m: np.hypot(abs(fr[m]), abs(fz[m])) for m in MODES}
    fits, fit_series = [], {}
    for m in FIT_MODES:
        ws = {}
        for t0 in (80.0, 90.0, 100.0):
            sel = T >= t0
            t = T[sel]
            A = np.hypot(abs(fr[m][sel]), abs(fz[m][sel]))
            s = (abs(fr[m][sel]) ** 2 - abs(fz[m][sel]) ** 2) / A ** 2
            ws[t0] = fit_frequency(t, s) + (t, s, A)
        w, werr, contrast, nyq, p, f, t, s, A = ws[90.0]
        g = np.polyfit(t, np.log(A), 1)[0]
        spread = max(abs(ws[t0][0] - w) for t0 in ws)
        print(f"   m={m}: omega {w:.3f} +- {werr:.3f} (descriptive), 2 omega / Nyquist = {nyq:.2f}, next distinct "
              f"solution has {contrast:.1f}x the residual, window start 80/100 changes omega by <= {spread:.3f}")
        Rm = R[T >= 90].mean()
        k = m / Rm
        fits.append((m, Rm, k, g, -g / k ** 2, w, w * Rm ** 2 / (m * np.sqrt(m * m - 1)), werr, contrast, spread))
        fit_series[f"fit_t{m}"], fit_series[f"fit_s{m}"], fit_series[f"fit_model{m}"] = t, s, f(t, *p)
        print(f"m={m}: R={Rm:.2f} k={k:.3f} growth={g:.4f} gamma/k^2={-g / k**2:+.3f} omega={w:.3f} "
              f"sqrt(bz br)={fits[-1][6]:.2f}")
    # fits columns: m, R, k, growth, gamma/k^2, omega, sqrt(bz br), omega err, residual contrast, window spread
    np.savez("sf2/buckle.npz", T=T, R=R, fits=np.array(fits), last=last,
             **{f"amp{m}": amp[m] for m in MODES}, **{f"fr{m}": fr[m] for m in MODES},
             **{f"fz{m}": fz[m] for m in MODES}, **fit_series)
    print("wrote sf2/buckle.npz")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--from-series") + 1] if "--from-series" in sys.argv else None)
