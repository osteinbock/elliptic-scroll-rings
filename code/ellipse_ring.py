"""Elliptic scroll rings in the 3D Barkley model with fully periodic boundaries.

    du/dt = (1/eps) u (1-u) (u - (v+b)/a) + lap(u)
    dv/dt = u - v                                   (Dv = 0)

Same scheme as barkley3d.py (forward Euler, 7-point Laplacian, no u-update
trick), but periodic in x, y and z so the ring can drift along z for as long as
we like. The ring lies in a plane z = const with semi-axes (Ax, Ay) along x, y.

Initial condition: a broken plane pulse confined to the inside of the ellipse
(excited slab just below the plane, refractory wake just above it). Its free
edge is the ellipse, which curls up into a single scroll ring; the wake stops
the slab from also launching a wave upwards.

Filament = u = 1/2 and v = a/2 - b (barkley3d.find_filaments). Before each
detection the arrays are rolled along z so the ring sits mid-box and never
straddles the periodic seam.

    python ellipse_ring.py --b 0.05 --Ax 6 --Ay 6 --T 40          # circular ring
    python ellipse_ring.py --b 0.09 --Ax 8 --Ay 5 --T 300 --out e.npz
"""
import argparse
import time

import numpy as np
import numba as nb

import barkley3d as B


@nb.njit(parallel=True, cache=True)
def step_periodic(u, v, un, vn, a, b, eps, dt, dx):
    n0, n1, n2 = u.shape
    c = dt / (dx * dx)
    for i in nb.prange(n0):
        im = i - 1 if i > 0 else n0 - 1
        ip = i + 1 if i < n0 - 1 else 0
        for j in range(n1):
            jm = j - 1 if j > 0 else n1 - 1
            jp = j + 1 if j < n1 - 1 else 0
            for k in range(n2):
                km = k - 1 if k > 0 else n2 - 1
                kp = k + 1 if k < n2 - 1 else 0
                uc = u[i, j, k]
                vc = v[i, j, k]
                lap = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                       + u[i, j, km] + u[i, j, kp] - 6.0 * uc)
                un[i, j, k] = uc + dt / eps * uc * (1.0 - uc) * (uc - (vc + b) / a) + c * lap
                vn[i, j, k] = vc + dt * (uc - vc)


def initial_ring(shape, dx, Ax, Ay, z0, slab=1.5, wake=4.0, flip=False, mode=0, amp=0.0, tilt=0.0):
    """Broken pulse inside the ellipse (x/Ax)^2 + (y/Ay)^2 < 1 at height z0, or,
    with mode > 0, inside r < Ax (1 + amp cos(mode phi)) (an m-fold seed).
    tilt (radians) rotates the ring plane about the x axis (an m = 1 seed).
    flip=True mirrors it in z (opposite chirality, opposite drift)."""
    x = [(np.arange(m) + 0.5) * dx for m in shape]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    L = np.array(shape) * dx
    if tilt:
        Yc, Zc = Y - L[1] / 2, Z - z0
        Y = L[1] / 2 + Yc * np.cos(tilt) + Zc * np.sin(tilt)
        Z = z0 - Yc * np.sin(tilt) + Zc * np.cos(tilt)
    if mode:
        r = np.hypot(X - L[0] / 2, Y - L[1] / 2)
        phi = np.arctan2(Y - L[1] / 2, X - L[0] / 2)
        inside = r < Ax * (1 + amp * np.cos(mode * phi))
    else:
        inside = ((X - L[0] / 2) / Ax) ** 2 + ((Y - L[1] / 2) / Ay) ** 2 < 1.0
    zeta = (Z - z0) * (-1 if flip else 1)
    u = np.where(inside & (zeta > -slab) & (zeta <= 0), 1.0, 0.0)
    v = np.where(inside & (zeta > -slab) & (zeta <= 0), 0.0, 0.0)
    v = np.where(inside & (zeta > 0) & (zeta < wake), 1.0, v)
    return u, v


def measure(lines, Lxy):
    """Ellipse diagnostics of the longest closed curve: centre, in-plane half
    extents along x and y, and the cos(2 theta) / sin(2 theta) parts of z."""
    if not lines:
        return None
    p = max(lines, key=len)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    w = np.r_[seg, 0] + np.r_[0, seg]                    # arc-length weights
    c = (p * w[:, None]).sum(0) / w.sum()
    th = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    dz = p[:, 2] - c[2]
    h2c = 2 * (w * dz * np.cos(2 * th)).sum() / w.sum()
    h2s = 2 * (w * dz * np.sin(2 * th)).sum() / w.sum()
    rho = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    out = dict(cx=c[0], cy=c[1], zc=c[2],
               hx=(p[:, 0].max() - p[:, 0].min()) / 2, hy=(p[:, 1].max() - p[:, 1].min()) / 2,
               Rm=(w * rho).sum() / w.sum(), h2c=h2c, h2s=h2s,
               length=seg.sum(), npts=len(p), ncurves=len(lines))
    # Fourier modes in the polar angle (angle-weighted): radius and height
    dth = np.abs(np.diff(np.unwrap(th)))
    wt = np.r_[dth, 0] + np.r_[0, dth]
    wt = wt / wt.sum()
    for mm in (1, 2, 3):
        cm, sm = np.cos(mm * th), np.sin(mm * th)
        out[f"r{mm}c"] = 2 * (wt * rho * cm).sum()
        out[f"r{mm}s"] = 2 * (wt * rho * sm).sum()
        out[f"z{mm}c"] = 2 * (wt * dz * cm).sum()
        out[f"z{mm}s"] = 2 * (wt * dz * sm).sum()
    return out


def run(a, b, eps, Ax, Ay, Lxy, Lz, dx, T, every_t=0.5, flip=False, verbose=True, snaps=(), mode=0, amp=0.0, tilt=0.0, fields=None, probe=None, model=None):
    """model: optional kinetics plug-in (e.g. oregonator.Model) with dt(dx), ic(...),
    step(u, v, un, vn, dt, dx) and find(u, v, dx); default is the Barkley model."""
    shape = (int(round(Lxy / dx)), int(round(Lxy / dx)), int(round(Lz / dx)))
    dt = model.dt(dx) if model else 0.8 * dx * dx / 6
    nsteps = int(round(T / dt))
    every = max(1, int(round(every_t / dt)))
    if model:
        u, v = model.ic(shape, dx, Ax, Ay, Lz / 2, flip=flip, mode=mode, amp=amp)
    else:
        u, v = initial_ring(shape, dx, Ax, Ay, Lz / 2, flip=flip, mode=mode, amp=amp, tilt=tilt)
    un, vn = np.empty_like(u), np.empty_like(v)
    zoff = 0.0                                  # accumulated z shift from rolling
    rows, frames = [], {}
    snap_steps = {int(round(s / dt)): s for s in snaps}
    t0 = time.time()
    pc = None                                   # probe voxel at the ring center (co-moving)
    for n in range(1, nsteps + 1):
        if model:
            model.step(u, v, un, vn, dt, dx)
        else:
            step_periodic(u, v, un, vn, a, b, eps, dt, dx)
        u, un = un, u
        v, vn = vn, v
        if probe is not None and pc is not None:
            probe.append((n * dt, float(u[pc])))
        if n % every == 0 or n in snap_steps:
            lines = model.find(u, v, dx) if model else B.find_filaments(u, v, a, b, dx)
            m = measure(lines, Lxy)
            t = n * dt
            if m is None:
                if verbose:
                    print(f"t={t:7.1f}  no filament")
                if t > (20 if model else 5):
                    rows.append(dict(t=t))
                    break
                continue
            # recentre the ring in z (roll the fields) so it never hits the seam
            k = int(round((Lz / 2 - m["zc"]) / dx))
            if k:
                u = np.roll(u, k, axis=2); v = np.roll(v, k, axis=2)
                un = np.empty_like(u); vn = np.empty_like(v)
                zoff -= k * dx
                lines = [l + np.array([0, 0, k * dx]) for l in lines]
            m["z"] = m["zc"] + zoff
            m["t"] = t
            pc = (int(round(m["cx"] / dx)) % shape[0], int(round(m["cy"] / dx)) % shape[1],
                  int(round((m["zc"] + k * dx) / dx)) % shape[2])
            rows.append(m)
            if n in snap_steps:
                frames[snap_steps[n]] = [l.copy() + np.array([0, 0, zoff]) for l in lines]
                if fields is not None:
                    fields[snap_steps[n]] = (u.astype(np.float16), None if fields.get("_uonly") else v.astype(np.float16), zoff)
            if verbose and (n % (every * 4) == 0):
                print(f"t={t:7.1f}  ncurves={m['ncurves']}  hx={m['hx']:.3f} hy={m['hy']:.3f} "
                      f"Rm={m['Rm']:.3f}  z={m['z']:.3f}  h2c={m['h2c']:+.3f}  L={m['length']:.2f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    return rows, frames


def table(rows):
    keys = ["t", "hx", "hy", "Rm", "z", "zc", "cx", "cy", "h2c", "h2s", "length", "ncurves",
            "r1c", "r1s", "z1c", "z1s", "r2c", "r2s", "z2c", "z2s", "r3c", "r3s", "z3c", "z3s"]
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--b", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--Ax", type=float, default=6.0)
    ap.add_argument("--Ay", type=float, default=6.0)
    ap.add_argument("--Lxy", type=float, default=24.0)
    ap.add_argument("--Lz", type=float, default=16.0)
    ap.add_argument("--dx", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--snaps", type=float, nargs="*", default=())
    ap.add_argument("--m", type=int, default=0, help="m-fold seed r = Ax (1 + amp cos m phi); 0 = ellipse")
    ap.add_argument("--amp", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0, help="ring tilt about x (degrees)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fields", action="store_true", help="also save u, v (float16) at the snapshot times")
    ap.add_argument("--uonly", action="store_true", help="with --fields: save only u")
    ap.add_argument("--probe", action="store_true", help="record u at the ring center every time step")
    args = ap.parse_args()
    flds = ({"_uonly": True} if args.uonly else {}) if args.fields else None
    prb = [] if args.probe else None
    rows, frames = run(args.a, args.b, args.eps, args.Ax, args.Ay, args.Lxy, args.Lz, args.dx,
                       args.T, args.every, snaps=args.snaps, mode=args.m, amp=args.amp,
                       tilt=np.radians(args.tilt), fields=flds, probe=prb)
    if args.out:
        d = table(rows)
        for s, ls in frames.items():
            for i, l in enumerate(ls):
                d[f"snap_{s:g}_{i}"] = l
        for s, val in (flds or {}).items():
            if s == "_uonly":
                continue
            uu, vv, zo = val
            d[f"u_{s:g}"], d[f"zoff_{s:g}"] = uu, zo
            if vv is not None:
                d[f"v_{s:g}"] = vv
        if prb:
            pa = np.array(prb)
            d["probe_t"], d["probe_u"] = pa[:, 0], pa[:, 1]
        np.savez_compressed(args.out, **d, **{k: getattr(args, k) for k in
                                               ("a", "b", "eps", "Ax", "Ay", "Lxy", "Lz", "dx")})
        print("wrote", args.out)


if __name__ == "__main__":
    main()
