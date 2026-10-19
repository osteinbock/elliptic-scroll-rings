"""Atri, Amundson, Clapham & Sneyd (Biophys. J. 65, 1727, 1993) Ca2+ model for the
Xenopus oocyte at constant [IP3] (their spiral-wave setup, Fig. 11):

    dc/dt = Dc lap(c) + kflux mu n (b + V1 c/(k1 + c)) - gamma c/(kgamma + c) + beta
    tau_n dn/dt = k2^2/(k2^2 + c^2) - n                       (n does not diffuse)

c: [Ca2+] in uM; n: fraction of IP3 receptors not inactivated by Ca2+.
mu = mu0 + mu1 P/(kmu + P), P = [IP3] = 0.095 uM -> mu = 0.577 (excitable, just
below the oscillatory range 0.578-0.988). Units: um, s.
"""
import numpy as np
import numba as nb

import barkley3d as B

KF, BB, V1, K1, GAM, KG, K2, TAU, DC = 8.1, 0.111, 0.889, 0.7, 2.0, 0.1, 0.7, 2.0, 20.0


def mu_of_P(P):
    return 0.567 + 0.433 * P / (4.0 + P)


@nb.njit(parallel=True, cache=True)
def step(u, v, un, vn, mu, beta, dt, dx, p0, p1, p2):
    n0, n1, n2 = u.shape
    c = DC * dt / (dx * dx)
    for i in nb.prange(n0):
        if i > 0:
            im = i - 1
        else:
            im = n0 - 1 if p0 else 0
        if i < n0 - 1:
            ip = i + 1
        else:
            ip = 0 if p0 else n0 - 1
        for j in range(n1):
            if j > 0:
                jm = j - 1
            else:
                jm = n1 - 1 if p1 else 0
            if j < n1 - 1:
                jp = j + 1
            else:
                jp = 0 if p1 else n1 - 1
            for k in range(n2):
                if k > 0:
                    km = k - 1
                else:
                    km = n2 - 1 if p2 else 0
                if k < n2 - 1:
                    kp = k + 1
                else:
                    kp = 0 if p2 else n2 - 1
                cc = u[i, j, k]
                nn = v[i, j, k]
                lu = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                      + u[i, j, km] + u[i, j, kp] - 6.0 * cc)
                J = KF * mu * nn * (BB + V1 * cc / (K1 + cc)) - GAM * cc / (KG + cc) + beta
                un[i, j, k] = cc + dt * J + c * lu
                vn[i, j, k] = nn + dt / TAU * (K2 * K2 / (K2 * K2 + cc * cc) - nn)


def rest(mu, beta=0.0):
    from scipy.optimize import brentq
    f = lambda c: KF * mu * (K2 ** 2 / (K2 ** 2 + c ** 2)) * (BB + V1 * c / (K1 + c)) - GAM * c / (KG + c) + beta
    cs = brentq(f, 1e-6, 5.0)
    return cs, K2 ** 2 / (K2 ** 2 + cs ** 2)


class Model:
    """Atri Ca2+ plug-in for ellipse_ring.run (periodic in all axes). The filament
    is where c = cstar and n = nstar (the 'tip' of the Ca2+ spiral)."""

    def __init__(self, P=0.095, beta=0.0, cstar=0.4, nstar=None, dt=0.01,
                 c_exc=0.9, n_wake=0.05, slab=40.0, wake=120.0):
        self.mu, self.beta = mu_of_P(P), beta
        self.cstar, self.nstar = cstar, nstar
        if nstar is None:
            raise ValueError("nstar (filament threshold of the slow variable) must be given")
        self.dt_fixed = dt
        self.c0, self.n0 = rest(self.mu, beta)
        self.c_exc, self.n_wake, self.slab, self.wake = c_exc, n_wake, slab, wake

    def dt(self, dx):
        return min(self.dt_fixed, 0.8 * dx * dx / (6 * DC))

    def step(self, u, v, un, vn, dt, dx):
        step(u, v, un, vn, self.mu, self.beta, dt, dx, 1, 1, 1)

    def ic(self, shape, dx, Ax, Ay, z0, flip=False, **kw):
        x = [(np.arange(m) + 0.5) * dx for m in shape]
        X, Y, Z = np.meshgrid(*x, indexing="ij")
        L = np.array(shape) * dx
        inside = ((X - L[0] / 2) / Ax) ** 2 + ((Y - L[1] / 2) / Ay) ** 2 < 1.0
        zeta = (Z - z0) * (-1 if flip else 1)
        u = np.full(shape, self.c0)
        v = np.full(shape, self.n0)
        u[inside & (zeta > -self.slab) & (zeta <= 0)] = self.c_exc
        v[inside & (zeta > 0) & (zeta < self.wake)] = self.n_wake
        return u, v

    def find(self, u, v, dx):
        return B.find_curves(u - self.cstar, v - self.nstar, dx, stitch_tol=2.0)
