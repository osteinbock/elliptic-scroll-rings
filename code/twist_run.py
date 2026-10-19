"""Twist of an elliptic scroll ring during its axis-switching oscillation.

Runs the Barkley ellipse (periodic box, as ellipse_ring.py) and at each sample:
  1. takes the filament (tip line), resamples it uniformly in arc length, low-pass
     filters it in Fourier space,
  2. builds a parallel-transported (Bishop) frame (U, V) along it, closed around the
     loop by spreading the holonomy angle uniformly,
  3. measures the rotation phase theta(s) = angle of grad u (projected onto the normal
     plane) in that frame, and the Frenet angle psi(s) of the principal normal,
  4. reports twist w = d theta/ds (physical twist, Bishop frame), torsion tau = d psi/ds,
     and the twist relative to the Frenet frame, w - tau.
Also records the shape (hx, hy, saddle amplitude) so twist can be read against phase.

    python twist_run.py a b Ax Ay T out.npz
"""
import sys
import time
import numpy as np
from scipy.ndimage import map_coordinates

import barkley3d as B
import ellipse_ring as E

KEEP = 10            # Fourier modes kept when smoothing the filament
M = 256              # resampling points


def smooth_loop(p):
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    # orient counter-clockwise seen from +z, so the sign of d theta/ds is consistent
    xy = p[:, :2] - p[:, :2].mean(0)
    if np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) < 0:
        p = p[::-1]
    seg = np.linalg.norm(np.diff(np.r_[p, p[:1]], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]
    q = np.linspace(0, L, M, endpoint=False)
    P = np.c_[[np.interp(q, s, np.r_[p[:, k], p[0, k]]) for k in range(3)]].T
    F = np.fft.fft(P, axis=0)
    k = np.fft.fftfreq(M, 1.0 / M)
    F[np.abs(k) > KEEP] = 0
    return np.fft.ifft(F, axis=0).real, L


def frames(P, L):
    k = 1j * np.fft.fftfreq(M, 1.0 / M) * (2 * np.pi / L)      # d/ds on the uniform grid
    D = lambda f: np.fft.ifft(k[:, None] * np.fft.fft(f, axis=0), axis=0).real
    Xs = D(P)
    T = Xs / np.linalg.norm(Xs, axis=1)[:, None]
    Ts = D(T)
    Kv = Ts - (Ts * T).sum(1)[:, None] * T                        # curvature vector K N
    N = Kv / np.linalg.norm(Kv, axis=1)[:, None]
    # parallel transport
    U = np.zeros((M, 3))
    u0 = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(u0) < 1e-3:
        u0 = np.cross(T[0], [1.0, 0, 0])
    U[0] = u0 / np.linalg.norm(u0)
    for i in range(1, M + 1):
        j = i % M
        w = U[i - 1] - (U[i - 1] @ T[j]) * T[j]
        w /= np.linalg.norm(w)
        if i < M:
            U[i] = w
        else:
            Uend = w
    V0 = np.cross(T[0], U[0])
    chi = np.arctan2(Uend @ V0, Uend @ U[0])                      # holonomy angle
    s = np.arange(M) * L / M
    rot = -chi * s / L
    V = np.cross(T, U)
    U, V = U * np.cos(rot)[:, None] + V * np.sin(rot)[:, None], -U * np.sin(rot)[:, None] + V * np.cos(rot)[:, None]
    psi = np.unwrap(np.arctan2((N * V).sum(1), (N * U).sum(1)))
    return T, U, V, N, psi, chi


def twist_sample(u, P, L, dx, T, U, V):
    g = np.stack(np.gradient(u, dx), 0)
    idx = (P / dx - 0.5).T                                         # grid coordinates of the points
    G = np.stack([map_coordinates(g[c], idx, order=1, mode="wrap") for c in range(3)], 1)
    Gp = G - (G * T).sum(1)[:, None] * T
    theta = np.unwrap(np.arctan2((Gp * V).sum(1), (Gp * U).sum(1)))
    return theta


def deriv_periodic(f, L, wind=0.0):
    """d f / d s for a periodic profile with total winding `wind` (radians)."""
    s = np.arange(M) * L / M
    g = f - wind * s / L
    k = 1j * np.fft.fftfreq(M, 1.0 / M) * (2 * np.pi / L)
    G = np.fft.fft(g)
    G[np.abs(np.fft.fftfreq(M, 1.0 / M)) > KEEP] = 0
    return np.fft.ifft(k * G).real + wind / L


def main():
    a, b, Ax, Ay, T_end = map(float, sys.argv[1:6])
    out = sys.argv[6]
    eps, dx, Lxy, Lz = 0.02, 0.2, 28.0, 16.0
    shape = (int(Lxy / dx), int(Lxy / dx), int(Lz / dx))
    dt = 0.8 * dx * dx / 6
    u, v = E.initial_ring(shape, dx, Ax, Ay, Lz / 2)
    un, vn = np.empty_like(u), np.empty_like(v)
    every = int(round(0.5 / dt))
    rows = []
    t0 = time.time()
    for n in range(1, int(T_end / dt) + 1):
        E.step_periodic(u, v, un, vn, a, b, eps, dt, dx)
        u, un = un, u
        v, vn = vn, v
        if n % every or n * dt < 4:
            continue
        lines = B.find_filaments(u, v, a, b, dx)
        if not lines:
            break
        p = max(lines, key=len)
        if len(p) < 40 or not np.allclose(p[0], p[-1]):
            continue
        k = int(round((Lz / 2 - p[:, 2].mean()) / dx))             # keep ring mid-box
        if k:
            u = np.roll(u, k, axis=2); v = np.roll(v, k, axis=2)
            un = np.empty_like(u); vn = np.empty_like(v)
            p = p + np.array([0, 0, k * dx])
        P, L = smooth_loop(p)
        Tt, U, V, N, psi, chi = frames(P, L)
        theta = twist_sample(u, P, L, dx, Tt, U, V)
        nw_t = np.round((theta[-1] - theta[0]) / (2 * np.pi))     # integer phase winding
        nw_p = np.round((psi[-1] - psi[0]) / (2 * np.pi))
        # theta and psi are angles in the closed frame, which turns by -chi s / L relative to
        # the parallel-transport (Bishop) frame; add that connection to get twist and torsion
        w = deriv_periodic(theta, L, 2 * np.pi * nw_t) - chi / L
        tau = deriv_periodic(psi, L, 2 * np.pi * nw_p) - chi / L
        c = P.mean(0)
        th = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        h2c = 2 * np.mean((P[:, 2] - c[2]) * np.cos(2 * th))
        rows.append(dict(t=n * dt, L=L, hx=np.ptp(P[:, 0]) / 2, hy=np.ptp(P[:, 1]) / 2, h2c=h2c,
                         w_rms=np.sqrt(np.mean((w - w.mean()) ** 2)), w_mean=w.mean(), w_max=np.abs(w).max(),
                         tau_rms=np.sqrt(np.mean((tau - tau.mean()) ** 2)), tau_max=np.abs(tau).max(),
                         wF_rms=np.sqrt(np.mean((w - tau - (w - tau).mean()) ** 2)),
                         corr=np.corrcoef(w, tau)[0, 1], chi=chi, n_theta=nw_t, n_psi=nw_p))
        r = rows[-1]
        print(f"t={r['t']:5.1f} hx={r['hx']:.2f} hy={r['hy']:.2f} h2c={r['h2c']:+.2f}  "
              f"w_rms={r['w_rms']:.4f} tau_rms={r['tau_rms']:.4f} (w-tau)_rms={r['wF_rms']:.4f} "
              f"corr(w,tau)={r['corr']:+.2f} wind={nw_t:.0f}/{nw_p:.0f}  [{time.time() - t0:.0f}s]", flush=True)
    np.savez(out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print("wrote", out)


if __name__ == "__main__":
    main()
