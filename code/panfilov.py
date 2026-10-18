"""Piecewise-linear excitable kinetics of Panfilov & Rudenko, Physica D 28, 215 (1987)
("kinetics B" of Nandapurkar & Winfree, Physica D 35, 277 (1989)):

    du/dt = lap(u) - F(u) - v
    dv/dt = h(u) (gs u - v)                      (v does not diffuse)

    F(u) = 4 u                for u < u1,
           -gf (u - ga)       for u1 <= u <= u2,
           15 (u - gb)        for u > u2,
    u1 = gf ga / (4 + gf),  u2 = (15 gb + gf ga) / (15 + gf)   (F continuous),
    h(u) = 1.82 (u < 0.01), 0.06 (0.01 <= u <= 0.95), 2.0 (u > 0.95),
    ga = 0.1, gb = 1, gs = 1.

gf sets the excitability; rings contract for gf > ~0.825 and expand below it
(Panfilov & Rudenko). Forward Euler, 7-point Laplacian, periodic or no-flux per
axis. Model plugs into ellipse_ring.run.
"""
import numpy as np
import numba as nb

import barkley3d as B

GA, GB, GS = 0.1, 1.0, 1.0


@nb.njit(parallel=True, cache=True)
def step(u, v, un, vn, gf, eps_mid, dt, dx, p0, p1, p2):
    n0, n1, n2 = u.shape
    c = dt / (dx * dx)
    u1 = gf * GA / (4.0 + gf)
    u2 = (15.0 * GB + gf * GA) / (15.0 + gf)
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
                uc = u[i, j, k]
                vc = v[i, j, k]
                if uc < u1:
                    F = 4.0 * uc
                elif uc <= u2:
                    F = -gf * (uc - GA)
                else:
                    F = 15.0 * (uc - GB)
                if uc < 0.01:
                    h = 1.82
                elif uc <= 0.95:
                    h = eps_mid
                else:
                    h = 2.0
                lu = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                      + u[i, j, km] + u[i, j, kp] - 6.0 * uc)
                un[i, j, k] = uc + dt * (-F - vc) + c * lu
                vn[i, j, k] = vc + dt * h * (GS * uc - vc)


class Model:
    """Panfilov kinetics plug-in for ellipse_ring.run (periodic in all axes)."""

    def __init__(self, gf, eps_mid=0.06, ustar=0.5, vstar=None, dt=None,
                 u_exc=0.95, v_wake=0.8, slab=4.0, wake=12.0):
        self.gf, self.eps_mid = gf, eps_mid
        self.ustar, self.vstar = ustar, vstar
        self.dt_fixed = dt
        self.u_exc, self.v_wake, self.slab, self.wake = u_exc, v_wake, slab, wake

    def dt(self, dx):
        return self.dt_fixed or min(0.05, 0.8 * dx * dx / 6)

    def step(self, u, v, un, vn, dt, dx):
        step(u, v, un, vn, self.gf, self.eps_mid, dt, dx, 1, 1, 1)

    def ic(self, shape, dx, Ax, Ay, z0, flip=False, mode=0, amp=0.0, **kw):
        """Ellipse with semi-axes Ax, Ay, or, for mode > 0, the m-fold seed
        r < Ax (1 + amp cos(mode phi))."""
        x = [(np.arange(m) + 0.5) * dx for m in shape]
        X, Y, Z = np.meshgrid(*x, indexing="ij")
        L = np.array(shape) * dx
        if mode:
            r = np.hypot(X - L[0] / 2, Y - L[1] / 2)
            phi = np.arctan2(Y - L[1] / 2, X - L[0] / 2)
            inside = r < Ax * (1 + amp * np.cos(mode * phi))
        else:
            inside = ((X - L[0] / 2) / Ax) ** 2 + ((Y - L[1] / 2) / Ay) ** 2 < 1.0
        zeta = (Z - z0) * (-1 if flip else 1)
        u = np.zeros(shape)
        v = np.zeros(shape)
        u[inside & (zeta > -self.slab) & (zeta <= 0)] = self.u_exc
        v[inside & (zeta > 0) & (zeta < self.wake)] = self.v_wake
        return u, v

    def find(self, u, v, dx):
        return B.find_curves(u - self.ustar, v - self.vstar, dx)
