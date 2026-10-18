"""Closed-curve flow X_t = alpha(K) K N + beta(K) K B - eta P(X_ssss),  alpha(K) = alpha + alpha2 K^2,
                       beta(K) = beta + beta2 K^2
                       (= alpha X_ss + beta X_s x X_ss - eta [X_ssss]_normal).

The eta term is filament rigidity (Dierckx et al., PRL 109, 174102 (2012);
Nakouzi et al., PRE 89, 042902 (2014), Eq. 4), projected onto the normal plane.

Pseudo-spectral in a uniform parameter, RK4, with tangential reparametrisation
toward equal arc-length spacing. Used to predict what an RD scroll ring with
measured (alpha, beta) should do, starting from a flat ellipse.

    python filament_flow.py 8 5 0.0 -2.4 120
"""
import sys
import numpy as np


def derivs(X):
    n = len(X)
    k = 1j * np.fft.fftfreq(n, 1.0 / n)
    F = np.fft.fft(X, axis=0)
    Xu = np.fft.ifft(k[:, None] * F, axis=0).real
    Xuu = np.fft.ifft((k ** 2)[:, None] * F, axis=0).real
    return Xu, Xuu


def d_s(f, s, k):
    """Arc-length derivative of the columns of f, with s = |X_u|."""
    return np.fft.ifft(k[:, None] * np.fft.fft(f, axis=0), axis=0).real / s[:, None]


def rhs(X, alpha, beta, eta=0.0, relax=1.0, alpha2=0.0, beta2=0.0, local=None):
    Xu, Xuu = derivs(X)
    g = (Xu ** 2).sum(1)
    T = Xu / np.sqrt(g)[:, None]
    Xss = (Xuu - (Xuu * T).sum(1)[:, None] * T) / g[:, None]      # = K N
    if local is not None:                                          # local alpha(K), beta(K)
        al, be = local(np.sqrt((Xss ** 2).sum(1)))
        V = al[:, None] * Xss + be[:, None] * np.cross(T, Xss)
    elif alpha2 or beta2:                                          # curvature-dependent coefficients
        K2 = (Xss ** 2).sum(1)[:, None]
        V = (alpha + alpha2 * K2) * Xss + (beta + beta2 * K2) * np.cross(T, Xss)
    else:
        V = alpha * Xss + beta * np.cross(T, Xss)
    if eta:
        n_ = len(X)
        kk = 1j * np.fft.fftfreq(n_, 1.0 / n_)
        D4 = d_s(d_s(Xss, np.sqrt(g), kk), np.sqrt(g), kk)           # X_ssss
        V = V - eta * (D4 - (D4 * T).sum(1)[:, None] * T)
    # tangential velocity that pushes |X_u| toward its mean (keeps spacing even)
    s = np.sqrt(g)
    w = s - s.mean()
    n = len(X)
    k = 1j * np.fft.fftfreq(n, 1.0 / n)
    k[0] = 1
    W = np.fft.fft(w)
    W[0] = 0
    phi = np.fft.ifft(W / k).real                                 # d(phi)/du = w
    return V - relax * phi[:, None] * T


def mfold(R, m, amp, n=192):
    """Planar m-fold curve r = R (1 + amp cos(m phi))."""
    u = 2 * np.pi * np.arange(n) / n
    r = R * (1 + amp * np.cos(m * u))
    return np.c_[r * np.cos(u), r * np.sin(u), np.zeros(n)]


def modes(X):
    """Polar-angle Fourier amplitudes |r_m|, |z_m| (m = 2, 3), angle-weighted."""
    c = X.mean(0)
    th = np.arctan2(X[:, 1] - c[1], X[:, 0] - c[0])
    rho = np.hypot(X[:, 0] - c[0], X[:, 1] - c[1])
    dz = X[:, 2] - c[2]
    dth = np.abs(np.diff(np.unwrap(np.r_[th, th[:1]])))
    wt = 0.5 * (dth + np.roll(dth, 1))
    wt = wt / wt.sum()
    out = {}
    for m in (2, 3):
        e = np.exp(-1j * m * th)
        cr, cz = 2 * (wt * rho * e).sum(), 2 * (wt * dz * e).sum()
        out[f"r{m}"], out[f"z{m}"] = abs(cr), abs(cz)
        out[f"r{m}c"], out[f"z{m}c"] = cr.real, cz.real      # signed cos(m theta) parts
    return out


def ellipse(Ax, Ay, n=192):
    u = 2 * np.pi * np.arange(n) / n
    return np.c_[Ax * np.cos(u), Ay * np.sin(u), np.zeros(n)]


def run(Ax, Ay, alpha, beta, T, dt=None, n=192, every=0.25, eta=0.0, alpha2=0.0, beta2=0.0, coeff_of_R=None, local=None, X0=None):
    """coeff_of_R: optional f(Rbar) -> (alpha, beta), Rbar = length / 2 pi, applied
    uniformly along the loop (ring-size-dependent rather than local coefficients)."""
    """Integrating-factor RK4: the stiff -eta d^4/ds^4 part (with the current mean
    spacing) is integrated exactly in Fourier space, the rest explicitly."""
    X = ellipse(Ax, Ay, n) if X0 is None else X0.copy()
    n = len(X)
    ds = np.linalg.norm(np.diff(np.r_[X, X[:1]], axis=0), axis=1).sum() / n   # mean spacing
    kmax = np.pi / ds
    if dt is None:
        dt = 0.4 / (max(np.hypot(alpha, beta), 1e-9) * kmax ** 2)
        if eta:
            dt = min(dt, 4.0 / (eta * kmax ** 4) + 0.2 / (eta * kmax ** 3 / ds))
    kint = np.fft.fftfreq(n, 1.0 / n)
    out, t, nxt = [], 0.0, 0.0
    while t <= T + 1e-9:
        if t >= nxt - 1e-12:
            c = X.mean(0)
            th = np.arctan2(X[:, 1] - c[1], X[:, 0] - c[0])
            out.append(dict(t=t, hx=np.ptp(X[:, 0]) / 2, hy=np.ptp(X[:, 1]) / 2, z=c[2],
                            h2c=2 * np.mean((X[:, 2] - c[2]) * np.cos(2 * th)),
                            length=np.linalg.norm(np.diff(np.r_[X, X[:1]], axis=0), axis=1).sum(),
                            **modes(X)))
            nxt += every
        sbar = np.linalg.norm(np.diff(np.r_[X, X[:1]], axis=0), axis=1).sum() / (2 * np.pi)
        if coeff_of_R is not None:
            alpha, beta = coeff_of_R(sbar)
        Lk = (-eta * (kint / sbar) ** 4)[:, None]
        E, E2 = np.exp(Lk * dt / 2), np.exp(Lk * dt)

        def N(Xh):
            Xr = np.fft.ifft(Xh, axis=0).real
            return np.fft.fft(rhs(Xr, alpha, beta, eta, alpha2=alpha2, beta2=beta2, local=local), axis=0) - Lk * Xh
        Xh = np.fft.fft(X, axis=0)
        a = N(Xh)
        b = N(E * (Xh + dt / 2 * a))
        c_ = N(E * Xh + dt / 2 * b)
        d = N(E2 * Xh + dt * E * c_)
        Xh = E2 * Xh + dt / 6 * (E2 * a + 2 * E * (b + c_) + d)
        X = np.fft.ifft(Xh, axis=0).real
        t += dt
    return {k: np.array([r[k] for r in out]) for k in out[0]}


if __name__ == "__main__":
    Ax, Ay, al, be, T = map(float, sys.argv[1:6])
    d = run(Ax, Ay, al, be, T)
    for i in range(0, len(d["t"]), 8):
        print(f"t={d['t'][i]:6.1f} hx={d['hx'][i]:.3f} hy={d['hy'][i]:.3f} z={d['z'][i]:+.2f} "
              f"h2c={d['h2c'][i]:+.3f} L={d['length'][i]:.3f}")
