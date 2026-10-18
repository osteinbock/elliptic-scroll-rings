"""For one Barkley parameter set (a, b, eps; Dv = 0):
  1. 2D spiral: rotation period, tip-circle radius, and rigidity (does the
     rotation centre wander / does the tip radius change from period to period?)
  2. 3D circular scroll ring (periodic box): filament tension alpha and beta.

    python param_probe.py a b eps [T2d] [T3d]
Prints one summary line starting with RESULT.
"""
import sys
import numpy as np
import barkley3d as B
import ellipse_ring as E
import tension_scan as S


def spiral2d(a, b, eps, Tend, N=200, dx=0.2):
    Lb = N * dx
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.where(X < Lb / 2, 1.0, 0.0)[:, :, None].repeat(3, 2)
    v = np.where(Y < Lb / 2, a / 2, 0.0)[:, :, None].repeat(3, 2)
    un, vn = np.empty_like(u), np.empty_like(v)
    dt = 0.8 * dx * dx / 6
    every = max(1, int(round(0.05 / dt)))
    prev, tips = np.array([Lb / 2, Lb / 2]), []
    for n in range(1, int(Tend / dt)):
        B.step(u, v, un, vn, a, b, eps, dt, dx)
        u, un = un, u
        v, vn = vn, v
        if n * dt > 25 and n % every == 0:
            L = B.find_filaments(u, v, a, b, dx)
            if L:
                P = np.array([l[:, :2].mean(0) for l in L])
                k = np.argmin(np.hypot(*(P - prev).T))
                prev = P[k]
                tips.append((n * dt, *P[k]))
    T = np.array(tips)
    if len(T) < 50:
        return None
    t, z = T[:, 0], T[:, 1] + 1j * T[:, 2]
    # rotation period from the dominant spectral line
    zz = (z - z.mean()) * np.hanning(len(z))
    f = np.fft.fftfreq(len(z), np.diff(t).mean())
    P = np.abs(np.fft.fft(zz)) ** 2
    P[np.abs(f) < 3.0 / (t[-1] - t[0])] = 0           # ignore the slowest bins (drift)
    per = 1 / abs(f[np.argmax(P)])
    # circle fits over 2-period windows -> radius and centre per window
    cs, rs, ts = [], [], []
    for t0 in np.arange(t[0], t[-1] - 2 * per, per):
        m = (t >= t0) & (t < t0 + 2 * per)
        q = np.c_[z[m].real, z[m].imag]
        A = np.c_[2 * q, np.ones(len(q))]
        s, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        cs.append(s[:2]); rs.append(np.sqrt(s[2] + s[:2] @ s[:2])); ts.append(t0)
    cs, rs, ts = np.array(cs), np.array(rs), np.array(ts)
    speed = np.median(np.hypot(*np.diff(cs, axis=0).T)) / per       # centre wander per time
    return dict(period=per, R=np.median(rs), Rstd=rs.std(), cspeed=speed)


def main():
    a, b, eps = map(float, sys.argv[1:4])
    T2d = float(sys.argv[4]) if len(sys.argv) > 4 else 120
    T3d = float(sys.argv[5]) if len(sys.argv) > 5 else 60
    s = spiral2d(a, b, eps, T2d)
    if s is None:
        print(f"RESULT a={a} b={b} eps={eps}: no sustained 2D spiral", flush=True)
        return
    print(f"2D a={a} b={b} eps={eps}: period={s['period']:.3f} R={s['R']:.3f} "
          f"Rstd={s['Rstd']:.3f} centre_speed={s['cspeed']:.4f}", flush=True)
    rows, _ = E.run(a, b, eps, 6, 6, 24, 16, 0.2, T3d, every_t=0.25, verbose=False)
    al, be, d = S.fit(rows, 15)
    ok = np.isfinite(d["Rm"])
    Rend = d["Rm"][ok][-1] if ok.any() else np.nan
    print(f"RESULT a={a} b={b} eps={eps}: period={s['period']:.3f} tipR={s['R']:.3f} "
          f"Rstd={s['Rstd']:.3f} centre_speed={s['cspeed']:.4f} | alpha={al:+.3f} beta={be:+.3f} "
          f"R(end)={Rend:.2f} t_end={d['t'][-1]:.0f}", flush=True)


if __name__ == "__main__":
    main()
