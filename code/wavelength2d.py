"""Rotation period, core radius and wavelength of two-dimensional spirals (revision: referee comments
A1 and C6, dimensionless comparison of the three models).

Barkley model: a broken wave in a thin no-flux slab (as param_probe.spiral2d), run to T; the tip is the
u = 1/2, v = a/2 - b crossing. Panfilov-Rudenko model: the fields saved by pan2d.py are read instead.
The period comes from the tip spectrum, the core radius from circle fits to the tip path over two
periods, and the wavelength from the spacing of successive u = 1/2 up-crossings along 48 rays from the
centre of the tip path, at distances between 2 core radii plus one estimated wavelength and the domain
edge minus 2 (median over rays and arms).

    python wavelength2d.py barkley a b eps T N dx OUT.json
    python wavelength2d.py panfilov FIELDS.npz TIPS.npy OUT.json
"""
import sys
import json
import numpy as np
from scipy.ndimage import map_coordinates


def tip_stats(t, z):
    zz = (z - z.mean()) * np.hanning(len(z))
    f = np.fft.fftfreq(len(z), np.diff(t).mean())
    P = np.abs(np.fft.fft(zz)) ** 2
    P[np.abs(f) < 3.0 / (t[-1] - t[0])] = 0
    per = 1 / abs(f[np.argmax(P)])
    cs, rs = [], []
    for t0 in np.arange(t[0], t[-1] - 2 * per, per):
        m = (t >= t0) & (t < t0 + 2 * per)
        q = np.c_[z[m].real, z[m].imag]
        A = np.c_[2 * q, np.ones(len(q))]
        s, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        cs.append(s[:2]); rs.append(np.sqrt(s[2] + s[:2] @ s[:2]))
    return per, float(np.median(rs)), np.array(cs)[-3:].mean(0)


def wavelength(u, dx, centre, rcore, lam_guess, nray=48):
    N = u.shape[0]
    Lb = N * dx
    edge = min(centre[0], centre[1], Lb - centre[0], Lb - centre[1]) - 2.0
    r = np.arange(0, edge, dx / 4)
    gaps = []
    for ph in np.linspace(0, 2 * np.pi, nray, endpoint=False):
        X = centre[0] + r * np.cos(ph)
        Y = centre[1] + r * np.sin(ph)
        val = map_coordinates(u, np.vstack([X / dx - 0.5, Y / dx - 0.5]), order=1, mode="nearest")
        s = val - 0.5
        idx = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
        rc = r[idx] - s[idx] * (r[idx + 1] - r[idx]) / (s[idx + 1] - s[idx])
        rc = rc[rc > 2 * rcore + lam_guess]
        gaps += list(np.diff(rc))
    gaps = np.array(gaps)
    gaps = gaps[(gaps > 0.5 * lam_guess) & (gaps < 2 * lam_guess)]
    return float(np.median(gaps)), float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75)), len(gaps), float(edge)


def barkley(a, b, eps, T, N, dx):
    import barkley3d as B
    Lb = N * dx
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.where(X < Lb / 2, 1.0, 0.0)[:, :, None].repeat(3, 2)
    v = np.where(Y < Lb / 2, a / 2, 0.0)[:, :, None].repeat(3, 2)
    un, vn = np.empty_like(u), np.empty_like(v)
    dt = 0.8 * dx * dx / 6
    every = max(1, int(round(0.05 / dt)))
    prev, tips = np.array([Lb / 2, Lb / 2]), []
    for n in range(1, int(T / dt)):
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
    Tt = np.array(tips)
    per, rcore, c = tip_stats(Tt[:, 0], Tt[:, 1] + 1j * Tt[:, 2])
    return per, rcore, c, u[:, :, 1]


def main():
    if sys.argv[1] == "barkley":
        a, b, eps, T = map(float, sys.argv[2:6])
        N, dx, out = int(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
        per, rcore, c, u = barkley(a, b, eps, T, N, dx)
        lam_guess = 12.0
        label = dict(model="barkley", a=a, b=b, eps=eps)
    else:
        d = np.load(sys.argv[2])
        tips = np.load(sys.argv[3])
        out = sys.argv[4]
        u, dx = d["u"], float(d["dx"])
        per, rcore, c = tip_stats(tips[:, 0], tips[:, 1] + 1j * tips[:, 2])
        lam_guess = 35.0
        label = dict(model="panfilov", fields=sys.argv[2])
    lam, q1, q3, n, edge = wavelength(u, dx, c, rcore, lam_guess)
    lam2, *_ = wavelength(u, dx, c, rcore, lam)              # second pass with the measured value
    res = dict(**label, period=per, core_radius=rcore, wavelength=lam2, wavelength_first_pass=lam,
               wavelength_iqr=[q1, q3], gaps=n, edge=edge, speed=lam2 / per)
    print(json.dumps(res))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
