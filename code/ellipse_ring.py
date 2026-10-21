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

Filament = u = 1/2 and v = a/2 - b (barkley3d.find_filaments). After each
detection the arrays are rolled along z by whole cells so that the ring sits mid-box at
the next detection and does not reach the periodic z seam; the x and y seams are not
stitched, so the lateral box must keep the ring away from them. Each row records the
extraction diagnostics (rejected faces, number and length of bridges, refused bridges).

Optional measurements (columns added to the same table):
  --levels u1,v1 u2,v2 ...  the same ring measures for other isosurface pairs u = u_i, v = v_i
                            (columns L1_*, L2_*, ...), to test the dependence on the filament definition;
  --twist                   rotation phase of the wave along the filament in a parallel-transported
                            frame (twist_run.py), its Fourier parts tw{m}c, tw{m}s in the polar angle
                            (m = 1-4), the rms local twist tw_rms, and the net twist tw_net.

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


def initial_ring(shape, dx, Ax, Ay, z0, slab=1.5, wake=4.0, flip=False, mode=0, amp=0.0, tilt=0.0, rot=0.0):
    """Broken pulse inside the ellipse (x/Ax)^2 + (y/Ay)^2 < 1 at height z0, or,
    with mode > 0, inside r < Ax (1 + amp cos(mode phi)) (an m-fold seed).
    tilt (radians) rotates the ring plane about the x axis (an m = 1 seed).
    rot (radians) turns the seed about the vertical axis (grid-orientation control).
    flip=True mirrors it in z (opposite chirality, opposite drift)."""
    x = [(np.arange(m) + 0.5) * dx for m in shape]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    L = np.array(shape) * dx
    if tilt:
        Yc, Zc = Y - L[1] / 2, Z - z0
        Y = L[1] / 2 + Yc * np.cos(tilt) + Zc * np.sin(tilt)
        Z = z0 - Yc * np.sin(tilt) + Zc * np.cos(tilt)
    if rot:
        Xc, Yc = X - L[0] / 2, Y - L[1] / 2
        X = L[0] / 2 + Xc * np.cos(rot) + Yc * np.sin(rot)
        Y = L[1] / 2 - Xc * np.sin(rot) + Yc * np.cos(rot)
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
    """Ellipse diagnostics of the longest curve (by arc length; closed = 1 if it closes): centre, in-plane half
    extents along x and y, and the cos(2 theta) / sin(2 theta) parts of z."""
    if not lines:
        return None
    arc = lambda l: np.linalg.norm(np.diff(l, axis=0), axis=1).sum()
    p = max(lines, key=arc)                       # longest curve by arc length; see 'closed'
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
               length=seg.sum(), npts=len(p), ncurves=len(lines), closed=float(B.is_closed(p)))
    # Fourier modes in the polar angle (angle-weighted): radius and height
    dth = np.abs(np.diff(np.unwrap(th)))
    wt = np.r_[dth, 0] + np.r_[0, dth]
    wt = wt / wt.sum()
    for mm in (1, 2, 3, 4):
        cm, sm = np.cos(mm * th), np.sin(mm * th)
        out[f"r{mm}c"] = 2 * (wt * rho * cm).sum()
        out[f"r{mm}s"] = 2 * (wt * rho * sm).sum()
        out[f"z{mm}c"] = 2 * (wt * dz * cm).sum()
        out[f"z{mm}s"] = 2 * (wt * dz * sm).sum()
    return out


LEVEL_KEYS = ("hx", "hy", "Rm", "zc", "cx", "cy", "length", "ncurves", "closed",
              *(f"{q}{mm}{cs}" for mm in (1, 2, 3, 4) for q in "rz" for cs in "cs"))


def measure_levels(u, v, levels, dx, Lxy):
    """measure() for the extra isosurface pairs (u*, v*): keys L1_*, L2_*, ... (missing if no curve)."""
    out = {}
    for i, (us, vs) in enumerate(levels, 1):
        m = measure(B.find_curves(u - us, v - vs, dx, stitch_tol=2.0), Lxy)
        if m is not None:
            out.update({f"L{i}_{k}": m[k] for k in LEVEL_KEYS})
    return out


def twist_phase(u, p, dx, modes=(1, 2, 3, 4)):
    """Rotation phase theta(s) of the wave along the closed filament p, measured as the angle of
    grad u in the normal plane relative to a parallel-transported frame (twist_run.py). Returns
    the Fourier parts of theta (linear winding and mean removed) in the polar angle about the
    filament centroid (angle-weighted, as r_m and z_m), the circular mean phase tw0, the rms
    local twist tw_rms = rms(d theta/ds - mean), and the net twist tw_net = int (d theta/ds) ds."""
    import twist_run as TW
    if len(p) < 40 or not B.is_closed(p):
        return {}
    P, L = TW.smooth_loop(p)
    T, U, V, _, _, chi = TW.frames(P, L)
    th = TW.twist_sample(u, P, L, dx, T, U, V)
    M = len(P)
    s = np.arange(M) * L / M
    n = np.round((th[-1] - th[0]) / (2 * np.pi))
    w = TW.deriv_periodic(th, L, 2 * np.pi * n) - chi / L
    thl = th - 2 * np.pi * n * s / L
    c = P.mean(0)
    ph = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    d = np.abs(np.angle(np.exp(1j * (np.roll(ph, -1) - ph))))
    wt = 0.5 * (d + np.roll(d, 1))
    wt = wt / wt.sum()
    thc = thl - (wt * thl).sum()
    out = dict(tw0=float(np.angle((wt * np.exp(1j * thl)).sum())), tw_rms=float(np.sqrt(np.mean((w - w.mean()) ** 2))),
               tw_net=float(2 * np.pi * n - chi), tw_wind=float(n))
    for mm in modes:
        out[f"tw{mm}c"] = 2 * (wt * thc * np.cos(mm * ph)).sum()
        out[f"tw{mm}s"] = 2 * (wt * thc * np.sin(mm * ph)).sum()
    return out


def run(a, b, eps, Ax, Ay, Lxy, Lz, dx, T, every_t=0.5, flip=False, verbose=True, snaps=(), mode=0, amp=0.0, tilt=0.0, fields=None, probe=None, model=None,
        levels=(), twist=False, dtfac=0.8, rot=0.0):
    """model: optional kinetics plug-in (e.g. oregonator.Model) with dt(dx), ic(...),
    step(u, v, un, vn, dt, dx) and find(u, v, dx); default is the Barkley model.
    levels: extra isosurface pairs (u*, v*) measured at every sample (Barkley model only);
    twist: also measure the rotation phase along the filament; dtfac: dt = dtfac dx^2 / 6;
    rot: turns the seed about the vertical axis (radians)."""
    shape = (int(round(Lxy / dx)), int(round(Lxy / dx)), int(round(Lz / dx)))
    dt = model.dt(dx) if model else dtfac * dx * dx / 6
    nsteps = int(round(T / dt))
    every = max(1, int(round(every_t / dt)))
    if model:
        u, v = model.ic(shape, dx, Ax, Ay, Lz / 2, flip=flip, mode=mode, amp=amp)
    else:
        u, v = initial_ring(shape, dx, Ax, Ay, Lz / 2, flip=flip, mode=mode, amp=amp, tilt=tilt, rot=rot)
    un, vn = np.empty_like(u), np.empty_like(v)
    zoff = 0.0                                  # accumulated z shift from rolling
    misses = 0                                  # consecutive samples without a filament
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
            diag = dict(B.LAST) if B.LAST else {}
            if m is None:
                misses += 1
                if verbose:
                    print(f"t={t:7.1f}  no filament")
                # end only after three consecutive samples without a filament
                if t > (20 if model else 5) and misses >= 3:
                    # end of the run: distinguish a diverged field from a vanished filament
                    finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
                    rows.append(dict(t=t, end=1.0 if finite else 2.0))   # 1: no filament, 2: non-finite field
                    if verbose:
                        print("end of run:", "filament vanished (ring collapsed or left the domain)" if finite
                              else "non-finite field (numerical instability)")
                    break
                continue
            misses = 0
            m.update(rejected=float(diag.get("rejected", 0)), bridges=float(diag.get("bridges", 0)),
                     max_bridge=float(diag.get("max_bridge", 0.0)), refused=float(diag.get("refused", 0)))
            # optional measures on the same (pre-roll) fields; the diagnostics above are already copied
            if twist:
                arc = lambda l: np.linalg.norm(np.diff(l, axis=0), axis=1).sum()
                m.update(twist_phase(u, max(lines, key=arc), dx))
            if levels:
                m.update(measure_levels(u, v, levels, dx, Lxy))
                for i in range(1, len(levels) + 1):
                    if f"L{i}_zc" in m:
                        m[f"L{i}_z"] = m[f"L{i}_zc"] + zoff
            # physical height uses the offset BEFORE this frame's roll (zc is the pre-roll centre)
            m["z"] = m["zc"] + zoff
            # recentre the ring in z (roll the fields) so it never hits the seam
            k = int(round((Lz / 2 - m["zc"]) / dx))
            if k:
                u = np.roll(u, k, axis=2); v = np.roll(v, k, axis=2)
                un = np.empty_like(u); vn = np.empty_like(v)
                zoff -= k * dx
                lines = [l + np.array([0, 0, k * dx]) for l in lines]
            m["zfix"] = 1.0                         # marks output written after the z fix
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
            "r1c", "r1s", "z1c", "z1s", "r2c", "r2s", "z2c", "z2s", "r3c", "r3s", "z3c", "z3s", "zfix", "closed", "end",
            "rejected", "bridges", "max_bridge", "refused"]
    extra = sorted({k for r in rows for k in r} - set(keys) - {"npts"})    # r4/z4, L*_ and tw* columns
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys + extra}


def code_revision():
    """'git:<commit>' of the code directory, or 'tree:<digest>' of provenance.SIMULATION_CODE."""
    import provenance
    return provenance.code_revision()


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
    ap.add_argument("--levels", nargs="*", default=(), help="extra isosurface pairs u*,v* (columns L1_*, L2_*, ...)")
    ap.add_argument("--twist", action="store_true", help="measure the rotation phase along the filament")
    ap.add_argument("--dtfac", type=float, default=0.8, help="time step dt = dtfac dx^2 / 6")
    ap.add_argument("--rot", type=float, default=0.0, help="turn the seed about the vertical axis (degrees)")
    args = ap.parse_args()
    flds = ({"_uonly": True} if args.uonly else {}) if args.fields else None
    prb = [] if args.probe else None
    levels = [tuple(float(x) for x in s.split(",")) for s in args.levels]
    rows, frames = run(args.a, args.b, args.eps, args.Ax, args.Ay, args.Lxy, args.Lz, args.dx,
                       args.T, args.every, snaps=args.snaps, mode=args.m, amp=args.amp,
                       tilt=np.radians(args.tilt), fields=flds, probe=prb, levels=levels, twist=args.twist,
                       dtfac=args.dtfac, rot=np.radians(args.rot))
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
        meta = {k: getattr(args, k) for k in ("a", "b", "eps", "Ax", "Ay", "Lxy", "Lz", "dx", "T", "every",
                                              "m", "amp", "tilt", "dtfac", "rot")}
        meta.update(dt=args.dtfac * args.dx ** 2 / 6, snaps_requested=np.array(args.snaps, float),
                    levels=np.array(levels, float).reshape(-1, 2), twist=args.twist,
                    snaps_saved=np.array(sorted(frames), float), code_revision=code_revision())
        np.savez_compressed(args.out, **d, **meta)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
