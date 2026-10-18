"""2D spiral tip at given b (a=0.8, eps=0.02): rigid vs meander vs wall drift.
No-flux box (thin 3D slab), tip tracked in a window around the previous tip."""
import sys
import numpy as np
import barkley3d as B
a, eps, dx = 0.8, 0.02, 0.2
b = float(sys.argv[1]); N = int(sys.argv[2]); Tend = float(sys.argv[3])
Lb = N * dx
x = (np.arange(N) + 0.5) * dx
X, Y = np.meshgrid(x, x, indexing="ij")
u = np.where(X < Lb / 2, 1.0, 0.0)[:, :, None].repeat(3, 2)
v = np.where(Y < Lb / 2, a / 2, 0.0)[:, :, None].repeat(3, 2)
un, vn = np.empty_like(u), np.empty_like(v)
dt = 0.8 * dx * dx / 6
prev = np.array([Lb / 2, Lb / 2]); tips = []; W = 25
for n in range(1, int(Tend / dt)):
    B.step(u, v, un, vn, a, b, eps, dt, dx); u, un = un, u; v, vn = vn, v
    if n * dt > 20 and n % 19 == 0:
        i0, j0 = W, W
        L = B.find_filaments(u, v, a, b, dx)
        if L:
            P = np.array([l[:, :2].mean(0) for l in L]) + np.array([i0 - W, j0 - W]) * dx
            k = np.argmin(np.hypot(*(P - prev).T)); prev = P[k]; tips.append((n * dt, *P[k]))
T = np.array(tips); np.save(f"tip2d_b{b}_N{N}.npy", T)
for t0 in np.arange(20, T[-1, 0] - 9, 10):
    m = (T[:, 0] >= t0) & (T[:, 0] < t0 + 8.2)
    q = T[m, 1:]; A = np.c_[2 * q, np.ones(len(q))]
    s, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None); R = np.sqrt(s[2] + s[:2] @ s[:2])
    dev = np.abs(np.hypot(*(q - s[:2]).T) - R).max()
    print(f"t={t0:5.0f} centre=({s[0]:6.2f},{s[1]:6.2f}) R={R:.3f} maxdev={dev:.3f} wall_dist={min(s.min(), Lb - s.max()):.1f}", flush=True)
