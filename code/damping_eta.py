"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Rigidity-type damping of m = 2 and m = 3 shape modes at the stable radius.

Solves the filament equation X_t = alpha K N + beta K B - eta P(X_ssss) for weakly deformed
m-fold rings (R = 4.85, beta = -1.23 so that the m = 2 frequency matches the RD ring) and fits
damped sinusoids (mode_theory2.fit_mode) to the signed mode amplitudes, exactly as for the
reaction-diffusion runs. Two conventions:
  (i)  alpha = 0                     -> ring shrinks slowly (rigidity acts like a tension eta/R^2);
                                        decay rate of the relative amplitude r_m / R
  (ii) alpha = -eta / R^2            -> stationary ring, as for the RD ring at R*;
                                        decay rate of the absolute amplitude
For each convention: eta matched to the measured gamma_2 range, predicted gamma_3, and
the solver ratio gamma_3 / gamma_2.
"""
import numpy as np
from scipy.optimize import brentq
import filament_flow as F
from mode_theory2 import fit_mode

R0, BETA = 4.85, -1.23
G2_MEAS = (0.0097, 0.0113, 0.0121)            # min, central, max from the RD window scan
G3_MEAS = (0.052, 0.057, 0.071)


def gamma(m, eta, conv):
    alpha = 0.0 if conv == "i" else -eta / R0 ** 2
    w = m * np.sqrt(m * m - 1) * abs(BETA) / R0 ** 2
    T = 3 * 2 * np.pi / w
    r = F.run(0, 0, alpha, BETA, T, n=64, dt=0.01, eta=eta, X0=F.mfold(R0, m, 0.03, n=64), every=0.05)
    rc, zc = r[f"r{m}c"], r[f"z{m}c"]
    if conv == "i":                            # relative amplitude
        Rt = r["length"] / (2 * np.pi)
        rc, zc = rc / Rt, zc / Rt
    if not (np.all(np.isfinite(rc)) and np.all(np.isfinite(zc))):
        raise RuntimeError(f"solver unstable at eta = {eta}")
    return fit_mode(r["t"], rc, zc, w)["gamma"]


if __name__ == "__main__":
    for conv in ("i", "ii"):
        g2 = lambda e: gamma(2, e, conv)
        etas = {lab: brentq(lambda e: g2(e) - g, 0.05, 0.8, xtol=2e-3) for lab, g in zip(("min", "mid", "max"), G2_MEAS)}
        g3 = {lab: gamma(3, e, conv) for lab, e in etas.items()}
        ratio = g3["mid"] / G2_MEAS[1]
        print(f"convention ({conv}): eta = {etas['mid']:.3f} (range {etas['min']:.3f}-{etas['max']:.3f});"
              f" predicted gamma_3 = {g3['mid']:.4f} (range {g3['min']:.4f}-{g3['max']:.4f});"
              f" solver ratio gamma_3/gamma_2 = {ratio:.2f}; measured gamma_3 {G3_MEAS[0]}-{G3_MEAS[2]}")
