"""Two-variable Oregonator (Tyson-Fife / Jahnke-Winfree form), as in Alonso,
Sagues & Mikhailov, J. Phys. Chem. A 110, 12063 (2006):

    du/dt = (1/eps) [u (1 - u) - f v (u - q) / (u + q)] + lap(u)
    dv/dt = u - v + Dv lap(v)

u ~ HBrO2, v ~ oxidized catalyst. Dv = 0: catalyst immobilized in a gel.
Forward Euler, 7-point Laplacian; boundaries periodic (pbc=1) or no-flux (pbc=0)
per axis. Plugs into ellipse_ring.run via the Model object below.
"""
import numpy as np
import numba as nb

import barkley3d as B


@nb.njit(parallel=True, cache=True)
def step(u, v, un, vn, f, q, eps, Dv, dt, dx, p0, p1, p2):
    n0, n1, n2 = u.shape
    c = dt / (dx * dx)
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
                lu = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                      + u[i, j, km] + u[i, j, kp] - 6.0 * uc)
                un[i, j, k] = uc + dt / eps * (uc * (1.0 - uc) - f * vc * (uc - q) / (uc + q)) + c * lu
                if Dv != 0.0:
                    lv = (v[im, j, k] + v[ip, j, k] + v[i, jm, k] + v[i, jp, k]
                          + v[i, j, km] + v[i, j, kp] - 6.0 * vc)
                    vn[i, j, k] = vc + dt * (uc - vc) + Dv * c * lv
                else:
                    vn[i, j, k] = vc + dt * (uc - vc)


def rest_state(f, q):
    """Homogeneous steady state (u0 = v0): u(1-u) = f u (u-q)/(u+q)."""
    # u^2 + (f + q - 1) u - q (1 + f) = 0 -> positive root
    b_ = f + q - 1.0
    u0 = 0.5 * (-b_ + np.sqrt(b_ * b_ + 4 * q * (1 + f)))
    return u0


class Model:
    """Oregonator plug-in for ellipse_ring.run (periodic in all three axes)."""

    def __init__(self, f, q=0.002, eps=0.05, Dv=0.0, ustar=0.3, vstar=None, dt=None,
                 u_exc=0.9, v_wake=None, slab=3.0, wake=8.0):
        self.f, self.q, self.eps, self.Dv = f, q, eps, Dv
        self.u0 = rest_state(f, q)
        self.ustar = ustar
        self.vstar = vstar
        if vstar is None:
            raise ValueError("vstar (filament threshold of v) must be given")
        self.dt_fixed = dt
        self.u_exc, self.v_wake, self.slab, self.wake = u_exc, v_wake, slab, wake

    def dt(self, dx):
        if self.dt_fixed:
            return self.dt_fixed
        return min(0.001, 0.8 * dx * dx / (6 * max(1.0, self.Dv)))

    def step(self, u, v, un, vn, dt, dx):
        step(u, v, un, vn, self.f, self.q, self.eps, self.Dv, dt, dx, 1, 1, 1)

    def ic(self, shape, dx, Ax, Ay, z0, flip=False, **kw):
        x = [(np.arange(m) + 0.5) * dx for m in shape]
        X, Y, Z = np.meshgrid(*x, indexing="ij")
        L = np.array(shape) * dx
        inside = ((X - L[0] / 2) / Ax) ** 2 + ((Y - L[1] / 2) / Ay) ** 2 < 1.0
        zeta = (Z - z0) * (-1 if flip else 1)
        u = np.full(shape, self.u0)
        v = np.full(shape, self.u0)
        exc = inside & (zeta > -self.slab) & (zeta <= 0)
        u[exc] = self.u_exc
        vw = self.v_wake if self.v_wake is not None else 0.5
        v[inside & (zeta > 0) & (zeta < self.wake)] = vw
        return u, v

    def find(self, u, v, dx):
        return B.find_curves(u - self.ustar, v - self.vstar, dx, stitch_tol=2.0)
