"""Buckling of the R0 = 38 ring: mode amplitudes, growth rates, and frequencies -> sf2/buckle.npz.

Growth rate from a log-linear fit of sqrt(|r_m|^2 + |z_m|^2) for t >= 90; frequency from the
exchange between radial and vertical parts, (|r_m|^2 - |z_m|^2)/A^2 ~ cos(2 omega t); the
binormal coefficient follows from omega = m sqrt(m^2 - 1) sqrt(beta_zeta beta_r) / R^2.
"""
import re
import numpy as np
from scipy.optimize import curve_fit
from buckle_modes import spectra

MODES = (4, 8, 12, 16, 20)


def main():
    d = np.load("sf2/circXXL_R0_38_snaps.npz")
    snaps = {}
    for key in d.files:
        m = re.match(r"snap_([\d.]+)_(\d+)$", key)
        if m:
            snaps.setdefault(float(m.group(1)), []).append(d[key])
    T = np.array(sorted(snaps))
    res = [spectra(snaps[t]) for t in T]
    R = np.array([r[0] for r in res])
    amp = {m: np.array([np.hypot(abs(r[1][m]), abs(r[2][m])) for r in res]) for m in MODES}
    fits = []
    sel = T >= 90
    for m in (12, 16, 20):
        fr = np.array([r[1][m] for r in res])[sel]
        fz = np.array([r[2][m] for r in res])[sel]
        t = T[sel]
        A = np.hypot(abs(fr), abs(fz))
        g = np.polyfit(t, np.log(A), 1)[0]
        s = (abs(fr) ** 2 - abs(fz) ** 2) / A ** 2
        best = None
        for w0 in np.linspace(0.05, 0.6, 111):
            try:
                p, _ = curve_fit(lambda t, a, w, ph, c: a * np.cos(2 * w * t + ph) + c, t, s, p0=[0.8, w0, 0, 0])
            except RuntimeError:
                continue
            r2 = np.sum((p[0] * np.cos(2 * p[1] * t + p[2]) + p[3] - s) ** 2)
            if best is None or r2 < best[0]:
                best = (r2, p)
        w = abs(best[1][1])
        Rm = R[sel].mean()
        k = m / Rm
        fits.append((m, Rm, k, g, -g / k ** 2, w, w * Rm ** 2 / (m * np.sqrt(m * m - 1))))
        print(f"m={m}: R={Rm:.2f} k={k:.3f} growth={g:.4f} gamma/k^2={-g / k**2:+.3f} omega={w:.3f} "
              f"sqrt(bz br)={fits[-1][-1]:.2f}")
    # filament at the last snapshot, for the figure
    last = [l for l in snaps[190.0] if len(l) > 20]
    np.savez("sf2/buckle.npz", T=T, R=R, fits=np.array(fits), last=np.concatenate(last),
             **{f"amp{m}": amp[m] for m in MODES})
    print("wrote sf2/buckle.npz")


if __name__ == "__main__":
    main()
