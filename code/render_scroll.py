"""Volume rendering of a scroll ring: excited region (u > 1/2) as a shaded, translucent surface,
the filament as an opaque tube colored by height, with an optional cutaway.

Orthographic ray marching with front-to-back alpha compositing; no external renderer needed.

    from render_scroll import render
    img = render(u, lines, dx, azim=45, elev=24, ...)   # RGBA float image
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import cKDTree


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def render(u, lines, dx, azim=45.0, elev=24.0, size=(700, 560), extent=None, depth_step=0.05,
           center=None, cut=None, wave_color="#eb6834", wave_alpha=0.55, tube=0.32,
           fil_low="#2a78d6", fil_high="#d6452a", zrange=None, fil_cmap=None, light=(-0.4, 0.5, 0.75), crop=None, solid=False, tmax=None):
    """u: (nx, ny, nz) field on a grid with spacing dx (index * dx = position).
    lines: list of (n, 3) filament arrays in the same coordinates.
    center: point the camera looks at; extent: half-widths (horizontal, vertical) of the view.
    cut: function(x, y, z) -> bool mask of points removed (cutaway), applied to the wave only.
    crop: (lo, hi) arrays; points outside this box are transparent."""
    nx, ny, nz = u.shape
    L = np.array([nx, ny, nz]) * dx
    if center is None:
        center = L / 2
    a, e = np.radians(azim), np.radians(elev)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # view direction
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(right, fwd)
    W, H = size
    if extent is None:
        extent = (0.6 * np.linalg.norm(L), 0.6 * np.linalg.norm(L) * H / W)
    xs = np.linspace(-extent[0], extent[0], W)
    ys = np.linspace(extent[1], -extent[1], H)
    X, Y = np.meshgrid(xs, ys)
    depth = np.linalg.norm(L)
    half = depth / 2 if tmax is None else min(depth / 2, tmax)
    ts = np.arange(-half, half, depth_step)   # march only where content can be
    us = gaussian_filter(u.astype(np.float32), 0.7)   # light smoothing (0.7 cells) of the shell only
    gx, gy, gz = np.gradient(us, dx)
    # filament: distance field and height color
    pts = np.concatenate([l for l in lines if len(l) > 5])
    tree = cKDTree(pts)
    zlo, zhi = zrange if zrange is not None else (pts[:, 2].min(), pts[:, 2].max())
    clow, chigh, cw = _hex(fil_low), _hex(fil_high), _hex(wave_color)
    lightv = np.array(light) / np.linalg.norm(light)
    col = np.zeros((H, W, 3))
    acc = np.zeros((H, W))
    base = center[None, None, :] + X[..., None] * right + Y[..., None] * up
    for t in ts:
        p = base + t * fwd
        q = p.reshape(-1, 3)
        idx = (q / dx - 0.5).T                # fields are cell-centred at (i + 1/2) dx
        inside = np.all((q >= 0.5 * dx) & (q <= L - 0.5 * dx), axis=1)
        if crop is not None:
            inside &= np.all((q >= crop[0]) & (q <= crop[1]), axis=1)
        if not inside.any():
            continue
        live = acc.reshape(-1) < 0.995
        sel = inside & live
        if not sel.any():
            continue
        qi = idx[:, sel]
        val = map_coordinates(us, qi, order=1, mode="nearest")
        # wave: thin shell at u = 1/2; the integral of alpha across one crossing equals wave_alpha
        g = np.stack([map_coordinates(gg, qi, order=1, mode="nearest") for gg in (gx, gy, gz)], 1)
        gn = np.linalg.norm(g, axis=1) + 1e-9
        sig = 0.06
        aw = wave_alpha * gn * np.exp(-0.5 * ((val - 0.5) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) * depth_step
        aw = 1 - np.exp(-aw)
        if solid:                                                 # opaque excited region u > 1/2
            aw = np.clip((val - 0.5) / 0.03, 0, 1)
        cap = np.zeros(len(val), bool)
        if cut is not None:
            qs = q[sel]
            incut = cut(*qs.T)
            aw[incut] = 0.0
            if solid:                                             # cut faces of the solid region
                back = qs - 1.5 * depth_step * fwd
                cap = ~incut & cut(*back.T) & (val > 0.5)
        nrm = -g / gn[:, None]                                   # outward normal of the excited region
        facing = np.sign(-(nrm @ fwd))[:, None]                  # two-sided lighting
        nrm = nrm * np.where(facing == 0, 1, facing)
        lam = np.clip(nrm @ lightv, 0, 1)
        spec = np.clip((nrm @ (lightv - fwd) / np.linalg.norm(lightv - fwd)), 0, 1) ** 24
        shade_w = (0.45 + 0.55 * lam)[:, None] * cw + 0.3 * spec[:, None]
        if solid and cap.any():
            shade_w[cap] = 0.78 * cw + 0.08
        # filament tube (opaque)
        dist, j = tree.query(q[sel], distance_upper_bound=tube)
        hit = np.isfinite(dist)
        af = np.where(hit, 1.0, 0.0)
        zz = np.clip((pts[np.minimum(j, len(pts) - 1), 2] - zlo) / max(zhi - zlo, 1e-6), 0, 1)
        fcol = (1 - zz)[:, None] * clow + zz[:, None] * chigh
        if fil_cmap is not None:
            fcol = fil_cmap(zz)[:, :3]
        # tube shading from the radial offset
        sh = np.where(hit, 1.0 - 0.35 * (dist / tube) ** 2, 1.0)
        fcol = fcol * sh[:, None]
        alpha = np.where(hit, af, aw)
        c = np.where(hit[:, None], fcol, shade_w)
        A = acc.reshape(-1)
        Cc = col.reshape(-1, 3)
        w = (1 - A[sel]) * alpha
        Cc[sel] += w[:, None] * c
        A[sel] += w
        acc = A.reshape(H, W)
        col = Cc.reshape(H, W, 3)
    rgba = np.dstack([col / np.maximum(acc, 1e-6)[..., None], acc])
    rgba[..., :3] = np.clip(rgba[..., :3], 0, 1)
    return rgba
