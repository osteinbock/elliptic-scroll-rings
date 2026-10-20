"""2D Panfilov kinetics-B spiral (thin no-flux slab): ranges, tip thresholds, rigidity.
    python pan2d.py gf eps_mid unused [L] [T] [dx] [dt]"""
import sys
import time
import numpy as np
import panfilov as PK
import barkley3d as B

f, eps, Dv = map(float, sys.argv[1:4])
L = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
T = float(sys.argv[5]) if len(sys.argv) > 5 else 150.0
dx = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
dt = float(sys.argv[7]) if len(sys.argv) > 7 else 0.04
q = 0.002
N = int(round(L / dx))
u0 = 0.0
x = (np.arange(N) + 0.5) * dx
X, Y = np.meshgrid(x, x, indexing="ij")
u = np.where((X < L / 2) & (Y < L / 2 + 6) & (Y > L / 2 - 6), 0.95, 0.0)[:, :, None].repeat(3, 2)
v = np.where((X < L / 2) & (Y >= L / 2 + 6) & (Y < L / 2 + 40), 0.8, 0.0)[:, :, None].repeat(3, 2)
un, vn = np.empty_like(u), np.empty_like(v)
t0 = time.time()
tips, prev, vstar, ustar = [], np.array([L / 2, L / 2]), None, 0.3
for n in range(1, int(T / dt) + 1):
    PK.step(u, v, un, vn, f, eps, dt, dx, 0, 0, 1)
    u, un = un, u
    v, vn = vn, v
    if n % int(round(10 / dt)) == 0:
        if not np.isfinite(u).all():
            print("BLOWUP at t", n * dt); sys.exit(1)
        uu, vv = u[:, :, 1], v[:, :, 1]
        on = np.abs(uu - 0.5) < 0.05
        vst = np.median(vv[on]) if on.any() else np.nan
        print(f"t={n*dt:5.0f} u[{uu.min():.4f},{uu.max():.3f}] v[{vv.min():.4f},{vv.max():.3f}] "
              f"frac_exc={np.mean(uu > 0.3):.3f} v_on_u0.3(median)={vst:.4f} [{time.time()-t0:.0f}s]", flush=True)
        if vstar is None and n * dt >= 80:
            vstar = 0.5 * (vv.min() + vv.max())
            ustar = 0.5 * (uu.min() + uu.max())
    if vstar is not None and n * dt > 120 and n % max(1, int(round(0.2 / dt))) == 0:
        Ls = B.find_curves(u - ustar, v - vstar, dx)
        if Ls:
            P = np.array([l[:, :2].mean(0) for l in Ls])
            k = np.argmin(np.hypot(*(P - prev).T)); prev = P[k]
            tips.append((n * dt, *P[k], len(Ls)))
T_ = np.array(tips)
np.savez(f"pan2d_fields_gf{f}_e{eps}.npz", u=u[:, :, 1], v=v[:, :, 1], vstar=vstar if vstar is not None else np.nan, dx=dx)
np.save(f"pan2d_gf{f}_e{eps}.npy", T_)
if len(T_) > 50:
    t, z = T_[:, 0], T_[:, 1] + 1j * T_[:, 2]
    zz = (z - z.mean()) * np.hanning(len(z))
    fr = np.fft.fftfreq(len(z), np.diff(t).mean()); P = np.abs(np.fft.fft(zz)) ** 2
    P[np.abs(fr) < 3 / (t[-1] - t[0])] = 0
    imax = np.argmax(P)
    per = 1 / abs(fr[imax])
    # meander index: strongest peak away from the rotation line and its neighbours
    P2 = P.copy(); P2[np.abs(fr - fr[imax]) < 0.03] = 0; P2[np.abs(fr + fr[imax]) < 0.03] = 0
    P2[np.abs(np.abs(fr) - 2 * abs(fr[imax])) < 0.03] = 0
    mi = np.sqrt(P2.max() / P[imax]); fsec = fr[np.argmax(P2)]
    cs, rs = [], []
    for a in np.arange(t[0], t[-1] - 2 * per, per):
        m = (t >= a) & (t < a + 2 * per); qq = np.c_[z[m].real, z[m].imag]
        A = np.c_[2 * qq, np.ones(len(qq))]; s, *_ = np.linalg.lstsq(A, (qq ** 2).sum(1), rcond=None)
        cs.append(s[:2]); rs.append(np.sqrt(s[2] + s[:2] @ s[:2]))
    cs, rs = np.array(cs), np.array(rs)
    print(f"RESULT f={f} eps={eps} Dv={Dv}: ustar={ustar:.3f} vstar={vstar:.4f} period={per:.3f} tipR={np.median(rs):.3f} "
          f"Rstd={rs.std():.3f} centre_speed={np.median(np.hypot(*np.diff(cs, axis=0).T))/per:.4f} ntips={np.median(T_[:,3]):.0f} "
          f"meander_index={mi:.3f} (f2={fsec:+.3f}, f1={fr[imax]:+.3f}) tipR_range={rs.min():.3f}-{rs.max():.3f}")
else:
    print(f"RESULT f={f} eps={eps} Dv={Dv}: too few tips ({len(T_)})")
