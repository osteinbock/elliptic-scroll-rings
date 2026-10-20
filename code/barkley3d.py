"""Barkley model in 3D: a single scroll wave in a cube with no-flux boundaries.

    du/dt = (1/eps) u (1-u) (u - (v+b)/a) + lap(u)
    dv/dt = u - v                                   (Dv = 0)

Forward Euler, 7-point Laplacian, plain update everywhere (no u-update trick).
Default a=0.8, b=0.05, eps=0.02 gives rigidly rotating spirals (tip circle
R ~ 0.44, period ~ 3.41, pitch ~ 10.6), so L = 21 is about two pitches.

The filament is the intersection of the isosurfaces u = 1/2 and v = a/2 - b,
found on cell faces (bilinear interpolation + Newton) and chained cube by cube
into oriented polylines. Output: interactive HTML with the u = 1/2 isosurface
and the filament superposed.

    python barkley3d.py                 # mildly curved initial filament
    python barkley3d.py --perturb 0     # perfectly straight scroll
"""
import argparse
import time

import numpy as np
import numba as nb


# ---------------------------------------------------------------- simulation

@nb.njit(parallel=True, cache=True)
def step(u, v, un, vn, a, b, eps, dt, dx):
    n0, n1, n2 = u.shape
    c = dt / (dx * dx)
    for i in nb.prange(n0):
        im = i - 1 if i > 0 else 0              # no-flux: mirror ghost = edge value
        ip = i + 1 if i < n0 - 1 else n0 - 1
        for j in range(n1):
            jm = j - 1 if j > 0 else 0
            jp = j + 1 if j < n1 - 1 else n1 - 1
            for k in range(n2):
                km = k - 1 if k > 0 else 0
                kp = k + 1 if k < n2 - 1 else n2 - 1
                uc = u[i, j, k]
                vc = v[i, j, k]
                lap = (u[im, j, k] + u[ip, j, k] + u[i, jm, k] + u[i, jp, k]
                       + u[i, j, km] + u[i, j, kp] - 6.0 * uc)
                un[i, j, k] = uc + dt / eps * uc * (1.0 - uc) * (uc - (vc + b) / a) + c * lap
                vn[i, j, k] = vc + dt * (uc - vc)


def initial_scroll(N, dx, a, perturb):
    """Cross-field initial condition extruded along z. The edge of the excited
    half-space is displaced by perturb*sin(pi z / L) so the initial filament is
    curved; perturb = 0 gives a straight scroll along z through the centre."""
    x = (np.arange(N) + 0.5) * dx
    L = N * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    shift = perturb * np.sin(np.pi * Z / L)
    u = np.where(X < L / 2 + shift, 1.0, 0.0)
    v = np.where(Y < L / 2, a / 2, 0.0)
    return u, v


# ---------------------------------------------------------- filament finding

def face_points(f, g, n):
    """Points on faces normal to axis n where bilinear f = g = 0.
    Returns lower-corner indices (m,3), positions (m,3) in grid units, and the
    sign of (grad f x grad g) . e_n."""
    p, q = (n + 1) % 3, (n + 2) % 3             # (p, q, n) right-handed
    N = f.shape
    # corner slices: face spans [i, i+1] in p and q, fixed index in n
    def corner(dp, dq):
        sl = [slice(None)] * 3
        sl[p] = slice(dp, N[p] - 1 + dp)
        sl[q] = slice(dq, N[q] - 1 + dq)
        return sl
    F = [f[tuple(corner(dp, dq))] for dp, dq in ((0, 0), (1, 0), (0, 1), (1, 1))]
    G = [g[tuple(corner(dp, dq))] for dp, dq in ((0, 0), (1, 0), (0, 1), (1, 1))]
    Fs, Gs = np.stack(F), np.stack(G)
    # exact zeros count as tiny positive values (symbolic perturbation): an intersection that
    # lies exactly on a grid plane is then found once, on one face, instead of not at all
    tf = 1e-12 * max(np.abs(Fs).max(), 1e-300)
    tg = 1e-12 * max(np.abs(Gs).max(), 1e-300)
    Fs = np.where(Fs == 0, tf, Fs)
    Gs = np.where(Gs == 0, tg, Gs)
    cand = ((Fs.min(0) < 0) & (Fs.max(0) > 0) & (Gs.min(0) < 0) & (Gs.max(0) > 0))
    idx = np.argwhere(cand)
    if len(idx) == 0:
        return np.empty((0, 3), int), np.empty((0, 3)), np.empty(0)
    f00, f10, f01, f11 = (Fs[c][cand] for c in range(4))
    g00, g10, g01, g11 = (Gs[c][cand] for c in range(4))
    fscale = np.max(np.abs([f00, f10, f01, f11]), axis=0)
    gscale = np.max(np.abs([g00, g10, g01, g11]), axis=0)

    def newton(s, t):
        for _ in range(30):
            fv = f00 * (1 - s) * (1 - t) + f10 * s * (1 - t) + f01 * (1 - s) * t + f11 * s * t
            gv = g00 * (1 - s) * (1 - t) + g10 * s * (1 - t) + g01 * (1 - s) * t + g11 * s * t
            fs = (f10 - f00) * (1 - t) + (f11 - f01) * t
            ft = (f01 - f00) * (1 - s) + (f11 - f10) * s
            gs = (g10 - g00) * (1 - t) + (g11 - g01) * t
            gt = (g01 - g00) * (1 - s) + (g11 - g10) * s
            det = fs * gt - ft * gs
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            s = s - (gt * fv - ft * gv) / det
            t = t - (-gs * fv + fs * gv) / det
        fv = f00 * (1 - s) * (1 - t) + f10 * s * (1 - t) + f01 * (1 - s) * t + f11 * s * t
        gv = g00 * (1 - s) * (1 - t) + g10 * s * (1 - t) + g01 * (1 - s) * t + g11 * s * t
        fs = (f10 - f00) * (1 - t) + (f11 - f01) * t
        ft = (f01 - f00) * (1 - s) + (f11 - f10) * s
        gs = (g10 - g00) * (1 - t) + (g11 - g01) * t
        gt = (g01 - g00) * (1 - s) + (g11 - g10) * s
        e = 1e-9
        good = ((s >= -e) & (s <= 1 + e) & (t >= -e) & (t <= 1 + e) & np.isfinite(s) & np.isfinite(t)
                & (np.abs(fv) <= RES_TOL * fscale) & (np.abs(gv) <= RES_TOL * gscale))
        return np.clip(s, 0, 1), np.clip(t, 0, 1), np.sign(fs * gt - ft * gs), good

    # several starting points; the first converged in-face root is accepted, and faces whose
    # bilinear equations have no real common zero in the face are rejected (residual test)
    s = np.full(len(idx), np.nan); t = s.copy(); orient = np.zeros(len(idx)); ok = np.zeros(len(idx), bool)
    for s0, t0 in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):   # diverging starts are rejected
            sn, tn, on, gn = newton(np.full(len(idx), s0), np.full(len(idx), t0))
        take = gn & ~ok
        s[take], t[take], orient[take] = sn[take], tn[take], on[take]
        ok |= gn
    REJECTED[n] = int((~ok).sum())
    # cand keeps all three array axes, so idx is already the face's lower corner
    pos = idx.astype(float)
    pos[:, p] += s
    pos[:, q] += t
    return idx[ok], pos[ok], orient[ok]


RES_TOL = 1e-8          # accepted root: |f|, |g| below RES_TOL times the face's corner scale
REJECTED = {0: 0, 1: 0, 2: 0}   # candidate faces without an accepted root, per axis (last call)


def find_filaments(u, v, a, b, dx):
    """Return a list of polylines (arrays of xyz in space units)."""
    return find_curves(u - 0.5, v - (a / 2 - b), dx, stitch_tol=2.0)


def find_curves(f, g, dx, stitch_tol=None):
    """Oriented polylines along the intersection of f = 0 and g = 0 (any box shape).
    With stitch_tol (in grid cells), pieces broken by a single missed face crossing are
    joined (see stitch)."""
    lines = chain_faces([face_points(f, g, n) for n in range(3)], f.shape, dx)
    LAST.update(rejected=sum(REJECTED.values()), bridges=0, max_bridge=0.0, refused=0)
    return stitch(lines, stitch_tol * dx, fields=(f, g, dx)) if stitch_tol else lines


LAST = {}   # diagnostics of the last find_curves call: rejected faces, bridges, longest bridge


def _off_surface(fields, P):
    """Distances (in grid cells) of points P from the f = 0 and g = 0 isosurfaces, estimated
    as |value| / |gradient| from trilinear interpolation of the fields."""
    from scipy.ndimage import map_coordinates
    f, g, dx = fields
    out = []
    for F in (f, g):
        idx = (P / dx - 0.5).T
        val = map_coordinates(F, idx, order=1, mode="nearest")
        grad = np.zeros(len(P))
        for ax in range(3):
            e = np.zeros((3, 1)); e[ax] = 0.5
            d = map_coordinates(F, idx + e, order=1, mode="nearest") - map_coordinates(F, idx - e, order=1, mode="nearest")
            grad += d ** 2
        out.append(np.abs(val) / np.maximum(np.sqrt(grad), 1e-300))
    return np.maximum(*out)


def is_closed(p, tol=1e-9):
    return len(p) > 3 and np.linalg.norm(p[0] - p[-1]) <= tol


def stitch(lines, tol, fields=None):
    """Join oriented open polylines whose end lies within tol of another's start (or of
    its own start, which closes it). A face crossing missed by the root finder breaks a
    filament into such pieces, with a gap of about one grid cell. With fields = (f, g, dx),
    a bridge is accepted only if its midpoint lies within half a grid cell of both
    isosurfaces; the number of bridges and the longest one are recorded in LAST."""
    closed = [l for l in lines if is_closed(l)]
    open_ = [l for l in lines if not is_closed(l)]

    def bridge_ok(p0, p1):
        if fields is None:
            return True
        ok = _off_surface(fields, 0.5 * (p0 + p1)[None, :])[0] <= 0.5
        if ok:
            LAST["bridges"] = LAST.get("bridges", 0) + 1
            LAST["max_bridge"] = max(LAST.get("max_bridge", 0.0), float(np.linalg.norm(p1 - p0)))
        else:
            LAST["refused"] = LAST.get("refused", 0) + 1
        return ok
    changed = True
    while changed and open_:
        changed = False
        for i, a in enumerate(open_):
            # own start first (closing), then the nearest other start
            if len(a) > 3 and np.linalg.norm(a[-1] - a[0]) < tol and bridge_ok(a[-1], a[0]):
                closed.append(np.vstack([a, a[:1]]))
                open_.pop(i); changed = True
                break
            dist = [(np.linalg.norm(a[-1] - b[0]), j) for j, b in enumerate(open_) if j != i]
            if dist:
                dmin, j = min(dist)
                if dmin < tol and bridge_ok(a[-1], open_[j][0]):
                    b = open_[j]
                    joined = np.vstack([a, b])
                    open_ = [l for k, l in enumerate(open_) if k not in (i, j)] + [joined]
                    changed = True
                    break
    return closed + open_


def chain_faces(per_axis, shape, dx):
    """Chain oriented face crossings into polylines. per_axis[n] = (lower corners,
    positions in grid units, orientation sign) for the faces normal to axis n."""
    ncube = np.array(shape) - 1
    pts, cube_lists = [], {}
    for n, (low, pos, orient) in enumerate(per_axis):
        e = np.zeros(3, int)
        e[n] = 1
        for c, x, o in zip(low, pos, orient):
            pid = len(pts)
            pts.append(x)
            # face is the lower n-face of cube c (outward normal -e_n) and the
            # upper n-face of cube c - e_n (outward normal +e_n)
            for cube, out in ((tuple(c), -o), (tuple(c - e), o)):
                if all(0 <= ci < m for ci, m in zip(cube, ncube)):
                    cube_lists.setdefault(cube, []).append((pid, out))
    nxt = {}
    for lst in cube_lists.values():
        ins = [pid for pid, out in lst if out < 0]
        outs = [pid for pid, out in lst if out > 0]
        for pi, po in zip(ins, outs):           # filament enters at pi, leaves at po
            nxt[pi] = po
    has_prev = set(nxt.values())
    lines, used = [], set()
    starts = [p for p in range(len(pts)) if p not in has_prev] + list(range(len(pts)))
    for p0 in starts:
        if p0 in used:
            continue
        chain, p = [], p0
        while p is not None and p not in used:
            used.add(p)
            chain.append(p)
            p = nxt.get(p)
        if p == p0:
            chain.append(p0)                    # closed loop
        if len(chain) > 1:
            lines.append((np.array([pts[i] for i in chain]) + 0.5) * dx)
    return lines


# -------------------------------------------------------------- visualising

# six tetrahedra sharing the cube diagonal 0-7; corner index = di + 2 dj + 4 dk
_TETS = np.array([(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)])
_CORNERS = np.array([(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)])


def isosurface(vol, level, spacing):
    """Marching tetrahedra. Returns welded vertices (m,3) in space units and
    triangles (k,3). spacing = grid step along each axis."""
    n = np.array(vol.shape) - 1
    base = np.stack(np.meshgrid(*[np.arange(m) for m in n], indexing="ij"), -1).reshape(-1, 3)
    tris = []
    for tet in _TETS:
        P = base[:, None, :] + _CORNERS[tet][None]                 # (cells, 4, 3)
        F = vol[P[..., 0], P[..., 1], P[..., 2]]
        inside = F > level
        cnt = inside.sum(1)
        keep = (cnt > 0) & (cnt < 4)
        P, F, inside, cnt = P[keep], F[keep], inside[keep], cnt[keep]
        order = np.argsort(~inside, axis=1, kind="stable")          # inside vertices first
        P = np.take_along_axis(P, order[..., None], 1).astype(float)
        F = np.take_along_axis(F, order, 1)

        def edge(i, j, m):
            t = ((level - F[m, i]) / (F[m, j] - F[m, i]))[:, None]
            return P[m, i] + t * (P[m, j] - P[m, i])
        m = cnt == 1
        tris.append(np.stack([edge(0, 1, m), edge(0, 2, m), edge(0, 3, m)], 1))
        m = cnt == 3
        tris.append(np.stack([edge(3, 0, m), edge(3, 1, m), edge(3, 2, m)], 1))
        m = cnt == 2
        e02, e03, e13, e12 = edge(0, 2, m), edge(0, 3, m), edge(1, 3, m), edge(1, 2, m)
        tris.append(np.stack([e02, e03, e13], 1))
        tris.append(np.stack([e02, e13, e12], 1))
    T = np.concatenate(tris) * np.asarray(spacing)
    verts, faces = np.unique(np.round(T.reshape(-1, 3), 4), axis=0, return_inverse=True)
    return verts, faces.reshape(-1, 3)


def surface_mesh(u, dx, level=0.5, sxy=None, sz=None):
    """Isosurface u = level on a subsampled grid. Default strides: fine in x,y,
    coarse along z (a straight scroll is nearly z-invariant), ~70 x 70 x 15 samples."""
    sxy = max(1, u.shape[0] // 70) if sxy is None else sxy
    sz = max(1, u.shape[2] // 15) if sz is None else sz
    us = u[::sxy, ::sxy, ::sz]
    h = np.array([sxy * dx, sxy * dx, sz * dx])
    verts, faces = isosurface(us, level, h)
    # orient every triangle so its normal points towards decreasing u (clean shading)
    grad = np.stack(np.gradient(us, *h), -1)
    cen = verts[faces].mean(1)
    gi = np.clip(np.round(cen / h).astype(int), 0, np.array(us.shape) - 1)
    nrm = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    flip = (nrm * grad[gi[:, 0], gi[:, 1], gi[:, 2]]).sum(1) > 0
    faces[flip] = faces[flip][:, ::-1]
    return verts + 0.5 * dx, faces


def render_html(verts, faces, lines, L, path, title):
    import plotly.graph_objects as go          # only needed for interactive renderings
    traces = [go.Mesh3d(
        x=np.round(verts[:, 0], 2), y=np.round(verts[:, 1], 2), z=np.round(verts[:, 2], 2),
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#f28e2b", opacity=0.35, name="u = 1/2", showlegend=True,
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1))]
    for k, ln in enumerate(lines):
        traces.append(go.Scatter3d(
            x=np.round(ln[:, 0], 3), y=np.round(ln[:, 1], 3), z=np.round(ln[:, 2], 3), mode="lines",
            line=dict(color="#c0182a", width=12),
            name="filament" if k == 0 else None, showlegend=(k == 0)))
    ax = dict(range=[0, L])
    fig = go.Figure(traces)
    fig.update_layout(title=title, scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube"),
                      margin=dict(l=0, r=0, t=40, b=0), autosize=True)
    fig.write_html(path, include_plotlyjs="cdn", default_width="100%", default_height="95vh",
                   config={"responsive": True})


def render_png(verts, faces, lines, L, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(8, 8), dpi=110)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(
        verts[faces], facecolors="#f28e2b", alpha=0.35,
        shade=True, lightsource=LightSource(azdeg=225, altdeg=45), zorder=1))
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], color="#c0182a", lw=3.5, zorder=10)
    ax.set(xlim=(0, L), ylim=(0, L), zlim=(0, L), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render(u, lines, dx, path, title):
    """Write path (.html, interactive) and the same name with .png (static)."""
    verts, faces = surface_mesh(u, dx)
    L = u.shape[0] * dx
    render_html(verts, faces, lines, L, path, title)
    render_png(verts, faces, lines, L, path.rsplit(".", 1)[0] + ".png", title)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--b", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--L", type=float, default=21.0, help="cube side (~2 pitches)")
    ap.add_argument("--dx", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=60.0, help="total time")
    ap.add_argument("--perturb", type=float, default=1.5, help="initial filament curvature amplitude")
    ap.add_argument("--out", default="scroll3d.html")
    ap.add_argument("--replot", metavar="NPZ", help="skip the simulation and re-render a saved run")
    args = ap.parse_args()

    if args.replot:
        d = np.load(args.replot)
        lines = [d[k] for k in sorted(d.files) if k.startswith("filament")]
        dx = float(d["dx"]) if "dx" in d.files else args.dx
        out = args.replot.rsplit(".", 1)[0] + ".html"
        render(d["u"], lines, dx, out, f"Barkley scroll wave  L={d['u'].shape[0] * dx:g}  (u = 1/2, filament)")
        print("wrote", out)
        return

    a, b, eps, dx = args.a, args.b, args.eps, args.dx
    N = int(round(args.L / dx))
    dt = 0.8 * dx * dx / 6                      # 80% of explicit stability limit
    nsteps = int(round(args.T / dt))
    every = max(1, int(round(1.0 / dt)))        # filament diagnostics once per time unit
    print(f"N={N}^3  dx={dx}  dt={dt:.5f}  steps={nsteps}")

    u, v = initial_scroll(N, dx, a, args.perturb)
    un, vn = np.empty_like(u), np.empty_like(v)
    t0 = time.time()
    for n in range(1, nsteps + 1):
        step(u, v, un, vn, a, b, eps, dt, dx)
        u, un = un, u
        v, vn = vn, v
        if n % every == 0 or n == nsteps:
            lines = find_filaments(u, v, a, b, dx)
            length = sum(np.linalg.norm(np.diff(l, axis=0), axis=1).sum() for l in lines)
            mid = np.concatenate(lines).mean(0) if lines else np.full(3, np.nan)
            print(f"t={n * dt:6.1f}  filaments={len(lines)}  length={length:6.2f}  "
                  f"centroid=({mid[0]:.2f},{mid[1]:.2f},{mid[2]:.2f})  [{time.time() - t0:.0f}s]")

    render(u, lines, dx, args.out,
           f"Barkley scroll wave  a={a} b={b} eps={eps}  t={args.T}  (u = 1/2, filament)")
    np.savez_compressed(args.out.replace(".html", ".npz"), u=u, v=v, dx=dx,
                        **{f"filament{k}": l for k, l in enumerate(lines)})
    print("wrote", args.out, "and", args.out.replace(".html", ".png"))


if __name__ == "__main__":
    main()
