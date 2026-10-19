"""Straight scroll-wave filament with a sinusoidal deformation (Barkley model).

A single scroll wave rotates around a straight filament along z, in a box with no-flux
side walls and periodic boundaries in z (height Lz = Nz dx). The initial filament is
displaced by A cos(k z), k = 2 pi / Lz. In linear filament theory, the complex deformation
u_k (x + i y component at wavenumber k) evolves as
    d u_k / dt = -(alpha(k) + i beta(k)) k^2 u_k ,
so that |u_k| decays at gamma = alpha(k) k^2 and its phase rotates at omega = beta(k) k^2.
A fourth-order term -eta X_ssss adds eta k^4 to gamma. The filament has no curvature in the
unperturbed state, so this measures the response to deformations of wavenumber k without the
ring curvature of the elliptic scroll rings.

    python straight_filament.py Nz A T out.npz [Lxy] [a b eps dx] [--shift S]

Saves both helical components u_{+k} (uk) and u_{-k} (ukm), plus the full filament profile
(prof) at every output time. --shift S displaces the whole initial filament by S in x
(with A = 0 this is a k = 0 control for wall-induced drift).
"""
import sys
import time
import numpy as np
import numba as nb

import barkley3d as B


@nb.njit(parallel=True, cache=True)
def step_mixed(u, v, un, vn, a, b, eps, dt, dx):
    """No-flux in x and y, periodic in z."""
    n0, n1, n2 = u.shape
    c = dt / (dx * dx)
    for i in nb.prange(n0):
        im = i - 1 if i > 0 else 0
        ip = i + 1 if i < n0 - 1 else n0 - 1
        for j in range(n1):
            jm = j - 1 if j > 0 else 0
            jp = j + 1 if j < n1 - 1 else n1 - 1
            for k in range(n2):
                km = k - 1 if k > 0 else n2 - 1
                kp = k + 1 if k < n2 - 1 else 0
                uc = u[i, j, k]
                vc = v[i, j, k]
                lap = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                       + u[i, j, km] + u[i, j, kp] - 6.0 * uc)
                un[i, j, k] = uc + dt / eps * uc * (1.0 - uc) * (uc - (vc + b) / a) + c * lap
                vn[i, j, k] = vc + dt * (uc - vc)


def filament_profile(u, v, a, b, dx, prev):
    """Filament position (x + i y) in each z layer, from crossings on faces normal to z."""
    idx, pos, _ = B.face_points(u - 0.5, v - (a / 2 - b), 2)
    nz = u.shape[2]
    prof = np.full(nz, np.nan + 1j * np.nan)
    if len(idx) == 0:
        return prof
    zc = (pos[:, 0] + 0.5) * dx + 1j * (pos[:, 1] + 0.5) * dx
    lay = idx[:, 2]
    for kz in np.unique(lay):
        cand = zc[lay == kz]
        ref = prev[kz] if np.isfinite(prev[kz]) else np.nanmean(prev) if np.any(np.isfinite(prev)) else cand[0]
        prof[kz] = cand[np.argmin(np.abs(cand - ref))]
    return prof


def helical_parts(prof, k, z):
    """Coefficients A, B of prof(z) = c + A e^{ikz} + B e^{-ikz} (complex least squares).
    With all layers present this equals the Fourier projections mean(prof e^{-ikz}) and
    mean(prof e^{ikz}); with missing layers it keeps the constant centre out of A and B."""
    M = np.c_[np.ones_like(z), np.exp(1j * k * z), np.exp(-1j * k * z)]
    c = np.linalg.lstsq(M, prof.astype(complex), rcond=None)[0]
    return c[1], c[2]


def main():
    shift = 0.0
    if "--shift" in sys.argv:
        i = sys.argv.index("--shift")
        shift = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    Nz, A, T, out = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    Lxy = float(sys.argv[5]) if len(sys.argv) > 5 else 32.0
    a, b, eps, dx = (map(float, sys.argv[6:10]) if len(sys.argv) > 9 else (1.1, 0.17, 0.02, 0.2))
    N = int(round(Lxy / dx))
    Lz = Nz * dx
    k = 2 * np.pi / Lz
    x = (np.arange(N) + 0.5) * dx
    z = (np.arange(Nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    X0 = Lxy / 2 + shift + A * np.cos(k * Z)
    u = np.where(X < X0, 1.0, 0.0)
    v = np.where(Y < Lxy / 2, a / 2, 0.0)
    un, vn = np.empty_like(u), np.empty_like(v)
    dt = 0.8 * dx * dx / 6
    every = max(1, int(round(0.1 / dt)))
    zz = np.arange(Nz) * dx               # face layers sit at the lower cell index
    prev = np.full(Nz, Lxy / 2 + 1j * Lxy / 2)
    rows = []
    t0 = time.time()
    for n in range(1, int(T / dt) + 1):
        step_mixed(u, v, un, vn, a, b, eps, dt, dx)
        u, un = un, u
        v, vn = vn, v
        if n % every:
            continue
        prof = filament_profile(u, v, a, b, dx, prev)
        ok = np.isfinite(prof)
        if ok.sum() < 0.8 * Nz:
            continue
        prev = np.where(ok, prof, prev)
        uk, ukm = helical_parts(prof[ok], k, zz[ok] + 0.5 * dx)
        rows.append((n * dt, prof[ok].mean(), 2 * uk, ok.sum(), 2 * ukm, prof.copy()))
        if len(rows) % 50 == 0:
            print(f"t={n*dt:6.1f}  |u_k|={abs(2*uk):.3f}  arg={np.angle(uk):+.2f}  layers={ok.sum()}  [{time.time()-t0:.0f}s]",
                  flush=True)
    r = np.array(rows, dtype=object)
    np.savez(out, t=r[:, 0].astype(float), u0=r[:, 1].astype(complex), uk=r[:, 2].astype(complex),
             nlay=r[:, 3].astype(int), ukm=r[:, 4].astype(complex), prof=np.stack(r[:, 5]),
             shift=shift, k=k, Lz=Lz, Nz=Nz, A=A, a=a, b=b, eps=eps, dx=dx, Lxy=Lxy)
    print("wrote", out, " k =", round(k, 4))


if __name__ == "__main__":
    main()
