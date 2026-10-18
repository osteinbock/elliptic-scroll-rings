"""Corrected linear theory of shape modes with curvature-dependent coefficients.

For X_t = alpha(K) K N + beta(K) K B, a radial deformation r_m changes |K| while a
vertical deformation zeta_m only tilts N and B. Linearizing around a ring of radius R:
    R^2 dr/dt   = -alpha_eff (m^2-1) r - beta m^2 zeta
    R^2 dzeta/dt =  beta_eff (m^2-1) r - alpha m^2 zeta
with alpha_eff = d(alpha K)/dK, beta_eff = d(beta K)/dK. For alpha = 0:
    omega_m = m sqrt(m^2-1) sqrt(beta beta_eff) / R^2
    |zeta_m| / |r_m| = sqrt((m^2-1)/m^2) sqrt(beta_eff / beta)
This script (1) checks both statements against the full nonlinear filament solver with a
prescribed beta(K), and (2) provides fit_mode(), which extracts omega, decay rate, and the
amplitude ratio from signed r_mc(t), zeta_mc(t) series.
"""
import numpy as np
from scipy.optimize import least_squares
import filament_flow as F


def fit_mode(t, r, z, omega0):
    """Fit r = A e^{-g t} cos(w t + p) + cr,  z = B e^{-g t} sin(w t + p + d) + cz."""
    def res(q):
        A, B, g, w, p, d, cr, cz = q
        e = np.exp(-g * (t - t[0]))
        return np.r_[A * e * np.cos(w * (t - t[0]) + p) + cr - r,
                     B * e * np.sin(w * (t - t[0]) + p + d) + cz - z]
    best = None
    for p0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        q0 = [np.ptp(r) / 2, np.ptp(z) / 2, 0.01, omega0, p0, 0.0, np.mean(r), np.mean(z)]
        s = least_squares(res, q0)
        if best is None or s.cost < best.cost:
            best = s
    A, B, g, w, p, d, cr, cz = best.x
    return dict(A=abs(A), B=abs(B), ratio=abs(B / A), gamma=g, omega=abs(w), dphase=(d + np.pi) % (2 * np.pi) - np.pi,
                rms=np.sqrt(np.mean(best.fun ** 2)))


def solver_check(R=4.85, beta0=-1.76, slope=0.0, m=2, amp=0.02, T=None):
    """Run the filament solver with beta(K) = beta0 + slope (K - 1/R), alpha = 0."""
    K0 = 1 / R
    local = lambda K: (np.zeros_like(K), beta0 + slope * (K - K0))
    X0 = F.mfold(R, m, amp, n=64)
    om_guess = m * np.sqrt(m * m - 1) * abs(beta0) / R ** 2
    T = T or 3 * 2 * np.pi / om_guess
    r = F.run(0, 0, 0.0, beta0, T, n=64, dt=0.01, X0=X0, every=0.05, local=local)
    fit = fit_mode(r["t"], r[f"r{m}c"], r[f"z{m}c"], om_guess)
    beta_eff = beta0 + K0 * slope
    pred_w = m * np.sqrt(m * m - 1) * np.sqrt(beta0 * beta_eff) / R ** 2
    pred_ratio = np.sqrt((m * m - 1) / m ** 2) * np.sqrt(beta_eff / beta0)
    return fit, pred_w, pred_ratio, om_guess


if __name__ == "__main__":
    print("solver check (alpha = 0, R = 4.85, beta(K) = -1.76 + slope (K - K0)):")
    print(" m  slope   beta_eff/beta | omega: fit   sqrt-theory  old(beta_eff)  | ratio: fit   theory")
    for m in (2, 3):
        for slope in (0.0, 3.0, 6.0):
            fit, pw, pr, w0 = solver_check(m=m, slope=slope)
            be = (-1.76 + slope / 4.85) / -1.76
            old = m * np.sqrt(m * m - 1) * abs(-1.76 * be) / 4.85 ** 2
            print(f" {m}  {slope:4.1f}     {be:.3f}        | {fit['omega']:.4f}   {pw:.4f}       {old:.4f}        "
                  f"| {fit['ratio']:.3f}   {pr:.3f}")
