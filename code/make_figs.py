"""Publication figures for the elliptic-scroll-ring paper (Chaos, AIP).

Writes chaos_paper/figures/fig1.pdf ... fig6.pdf as vector PDFs:
single column 3.37 in, double column 6.69 in, all text >= 8 pt, panel letters,
no internal titles. Data come from the simulation outputs in this folder.

    python make_figs.py            # all figures
    python make_figs.py 3 5        # selected figures
"""
import csv
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import median_filter

import ellipse_compare as C
import filament_flow as F

OUT = "chaos_paper/figures"
W1, W2 = 3.37, 6.69
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
})


# circ_R0_{6,9,13} are identical (deterministic) to the first 90 time units of circL_R0_{6,9,13} and are
# left out to avoid double counting, as is circHB_R0_4.0 (same as circ_R0_4); circ_R0_11_Lz32 is the
# box-size control for circ_R0_11.
VALID_CIRCLES = ["circ_R0_4.npz", "circ_R0_11.npz", "circL_R0_6.npz", "circL_R0_9.npz", "circL_R0_13.npz",
                 "circS_R0_4.5.npz", "circS_R0_5.5.npz", "circS_R0_6.5.npz", "circS_R0_7.5.npz",
                 "circHB_R0_5.0.npz", "circ_a1.1_b0.17.npz", "circ_a1.1_b0.17_R9.npz"]


LARGE_CIRCLES = ["sf2/circXL_R0_16.npz", "sf2/circXL_R0_20.npz"]


def label(ax, s, x=-0.17, y=1.0):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def grid(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def smooth_run(path, period, keys):
    d = np.load(path)
    ok = np.isfinite(d["hx"])
    t = d["t"][ok]
    return t, {k: C.smooth(t, d[k][ok], period) for k in keys}, d


def model_until_valid(r, t0, Rmin):
    """Shift model time and cut it where the ring becomes too small for the solver."""
    t = r["t"] + t0
    Rb = r["length"] / (2 * np.pi)
    stuck = np.r_[False, np.abs(np.diff(r["length"])) < 1e-4]
    keep = (Rb > Rmin) & ~np.maximum.accumulate(stuck) & np.isfinite(Rb)
    return t[keep], {k: v[keep] for k, v in r.items()}


# ----------------------------------------------------------------------------- Fig. 1
def _frame(f, th, d=1e-4):
    """Position, Frenet frame (T, N, B) and curvature of a parametric curve f(theta)."""
    X = f(th)
    X1 = (f(th + d) - f(th - d)) / (2 * d)
    X2 = (f(th + d) - 2 * X + f(th - d)) / d ** 2
    T = X1 / np.linalg.norm(X1)
    Kv = (X2 - (X2 @ T) * T) / (X1 @ X1)
    K = np.linalg.norm(Kv)
    N = Kv / K
    return X, T, N, np.cross(T, N), K


def _proj(P, azim=12.0, elev=26.0):
    """Orthographic projection of 3D points onto the drawing plane (front = lower)."""
    P = np.atleast_2d(P)
    a, e = np.radians(azim), np.radians(elev)
    x = P[:, 0] * np.cos(a) + P[:, 1] * np.sin(a)
    depth = -P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)
    return np.c_[x, P[:, 2] * np.cos(e) + depth * np.sin(e)]


def _arrow(ax, p0, p1, col, lw=1.2, ms=8):
    a, b = _proj(p0)[0], _proj(p1)[0]
    ax.annotate("", xy=b, xytext=a, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def _schematic(ax, f, points, velocity):
    th = np.linspace(0, 2 * np.pi, 400)
    C = np.array([f(t) for t in th])
    S = C.copy(); S[:, 2] = -0.85
    ax.plot(*_proj(S).T, color="#d4d3cc", lw=0.8)                                   # projection
    ax.plot(*_proj(np.array([[0, 0, -0.85], [0, 0, 1.15]])).T, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.plot(*_proj(C).T, color=INK, lw=1.7)
    L = 0.6
    for th0, offs in points:
        X, T, N, B, K = _frame(f, th0)
        ax.plot(*_proj(X)[0], "o", color=INK, ms=2.5)
        for v, col, lab in ((T, INK, "T"), (N, ORANGE, "N"), (B, BLUE, "B")):
            _arrow(ax, X, X + L * v, col)
            tx, ty = _proj(X + L * v)[0] + np.array(offs[lab])
            ax.text(tx, ty, lab, color=col, fontsize=8, fontweight="bold", ha="center", va="center")
        if velocity == "speed":            # binormal velocity beta K B, length proportional to K
            out = 0.42 * np.array([np.cos(th0), np.sin(th0), 0.0])
            _arrow(ax, X + out, X + out + 0.45 * K * B, "#a9a8a1", lw=2.6, ms=9)
        else:                              # horizontal part of B: inward / outward motion
            h = np.array([B[0], B[1], 0.0])
            base = X + np.array([0, 0, -0.3])
            _arrow(ax, base, base + 0.75 * h, "#a9a8a1", lw=2.6, ms=9)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.45, 1.35)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _seg_crossings(A, B):
    """Index pairs (i, j) where segment A[i]A[i+1] crosses segment B[j]B[j+1] (2D polylines)."""
    out = []
    for i in range(len(A) - 1):
        p, r = A[i], A[i + 1] - A[i]
        q, s = B[:-1], B[1:] - B[:-1]
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(den) > 1e-12
        qp = q - p
        t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(ok, den, 1), -1)
        for j in np.where(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))[0]:
            out.append((i, j))
    return out


def _mech_draw(ax):
    """Sketch: planar elliptic filament in orthographic view; T, N, B at one end of the long axis;
    sense of rotation of the scroll wave around the filament at the other end; drift of the ring.
    Orientation: right-hand rule (fingers along the rotation, thumb along T). The loop then runs
    counterclockwise seen from above, B = T x N points up, and the ring drifts along +B (beta > 0),
    as in our simulations (fig1_fields/handed_loop.py)."""
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe
    th_c, el = np.radians(-62.0), np.radians(20.0)
    cam = np.array([np.cos(el) * np.cos(th_c), np.cos(el) * np.sin(th_c), np.sin(el)])   # toward viewer
    rgt = np.array([-np.sin(th_c), np.cos(th_c), 0.0])
    upv = np.cross(cam, rgt)
    P = lambda X: np.array([np.dot(X, rgt), np.dot(X, upv)])
    a_, b_ = 1.75, 1.0
    th = np.linspace(0, 2 * np.pi, 600)
    E3 = np.c_[a_ * np.cos(th), b_ * np.sin(th), np.zeros_like(th)]
    E = np.array([P(X) for X in E3])
    # rotation circle around the filament at the left end, in the N-B (x-z) plane; increasing psi is
    # counterclockwise seen from -y: rotation vector along -y = T there (loop counterclockwise from above)
    XL = np.array([-a_, 0.0, 0.0])
    r = 0.42
    psi = np.linspace(-np.pi, np.pi, 401)                              # closed circle
    C3 = np.array([XL + r * np.array([np.cos(p), 0.0, np.sin(p)]) for p in psi])
    C2 = np.array([P(X) for X in C3])
    ax.plot(*E.T, color=INK, lw=1.6, zorder=2, solid_capstyle="round")
    ax.plot(*C2.T, color=AQUA, lw=1.2, zorder=2, solid_capstyle="round")
    # arrowhead mid-arc at the top (psi = pi/2), along the tangent of increasing psi (counterclockwise)
    # arrowhead: triangle centered on the projected circle at its topmost point, along the tangent of
    # increasing psi (counterclockwise rotation)
    from matplotlib.patches import Polygon
    k0 = int(np.argmax(C2[:, 1]))
    c0 = C2[k0]
    tg = C2[min(k0 + 2, len(C2) - 1)] - C2[max(k0 - 2, 0)]
    tg /= np.linalg.norm(tg)
    nv = np.array([-tg[1], tg[0]])
    Lh, Wh = 0.2, 0.085
    ax.add_patch(Polygon([c0 + 0.55 * Lh * tg, c0 - 0.45 * Lh * tg + Wh * nv, c0 - 0.45 * Lh * tg - Wh * nv],
                         closed=True, facecolor=AQUA, edgecolor="none", zorder=6))
    # over/under at each crossing: redraw a short piece of whichever curve is in front, with a white halo
    halo = lambda w: [pe.Stroke(linewidth=w + 3.0, foreground="white"), pe.Normal()]
    for i, j in _seg_crossings(C2, E):
        if np.dot(C3[i], cam) > np.dot(E3[j], cam):
            k = slice(max(i - 10, 0), min(i + 11, len(C2)))
            k2 = slice(max(i - 16, 0), min(i + 17, len(C2)))
            ax.plot(*C2[k].T, color=AQUA, lw=1.2, zorder=4, path_effects=halo(1.2), solid_capstyle="butt")
            ax.plot(*C2[k2].T, color=AQUA, lw=1.2, zorder=4.5, solid_capstyle="butt")
        else:
            k = slice(max(j - 6, 0), min(j + 7, len(E)))
            k2 = slice(max(j - 10, 0), min(j + 11, len(E)))
            ax.plot(*E[k].T, color=INK, lw=1.6, zorder=4, path_effects=halo(1.6), solid_capstyle="butt")
            ax.plot(*E[k2].T, color=INK, lw=1.6, zorder=4.5, solid_capstyle="butt")
    # ring axis and drift
    ax.plot(*np.array([P([0, 0, -0.3]), P([0, 0, 0.7])]).T, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.add_patch(FancyArrowPatch(P([0, 0, 0.78]), P([0, 0, 1.45]), arrowstyle="-|>", mutation_scale=10,
                                 color="#8e8d86", lw=2.4, shrinkA=0, shrinkB=0))
    ax.text(*(P([0, 0, 1.1]) + np.array([0.1, 0])), "drift", fontsize=8, color="#6b6a63", va="center")
    # local frame at the right end: T = +y (loop counterclockwise from above), N = -x, B = T x N = +z
    X0 = np.array([a_, 0.0, 0.0])
    L = 0.62
    for v, col, lab, off in (((0, 1, 0), INK, r"$\mathbf{T}$", (0.12, 0.04)),
                             ((-1, 0, 0), ORANGE, r"$\mathbf{N}$", (0.0, 0.13)),
                             ((0, 0, 1), BLUE, r"$\mathbf{B}$", (-0.13, 0.04))):
        tip = P(X0 + L * np.array(v, float) * (1.6 if v == (0, 1, 0) else 1.0))
        ax.add_patch(FancyArrowPatch(P(X0), tip, arrowstyle="-|>", mutation_scale=8, color=col, lw=1.2,
                                     shrinkA=0, shrinkB=0, zorder=5))
        ax.text(*(tip + np.array(off)), lab, color=col, fontsize=9, ha="center", va="center")
    ax.plot(*P(X0), "o", color=INK, ms=3, zorder=6)
    ax.plot(*P(XL), "o", color=INK, ms=3, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(min(E[:, 0].min(), C2[:, 0].min()) - 0.15, E[:, 0].max() + 0.6)
    ax.set_ylim(E[:, 1].min() - 0.2, P([0, 0, 1.5])[1])
    ax.set_axis_off()


def figM():
    fig, ax = plt.subplots(figsize=(W1, 1.85))
    _mech_draw(ax)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f"{OUT}/fig_mech.pdf")
    plt.close(fig)


def _cam(azim=40.0, elev=26.0):
    """Orthographic camera matching render_scroll.render: returns projection and depth functions."""
    a, e = np.radians(azim), np.radians(elev)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(right, fwd)
    return (lambda X: np.c_[np.atleast_2d(X) @ right, np.atleast_2d(X) @ up]), (lambda X: -np.atleast_2d(X) @ fwd)


def fig1():
    from matplotlib.collections import LineCollection
    fig = plt.figure(figsize=(W2, 3.0))
    # (a): sketch of the local frame, rotation sense, and drift
    am = fig.add_axes([0.0, 0.66, 0.44, 0.33])
    _mech_draw(am)
    fig.text(0.004, 0.995, "(a)", fontsize=8, fontweight="bold", va="top")
    # (b): rendered wave field (u = 1/2 surface, cut open) with the filament at t = 12
    img = np.load("fig1_fields/render_t12.npy")
    rows = np.where(img[..., 3].max(1) > 0.02)[0]
    cols = np.where(img[..., 3].max(0) > 0.02)[0]
    img = img[max(rows[0] - 6, 0):rows[-1] + 6, max(cols[0] - 6, 0):cols[-1] + 6]
    a0 = fig.add_axes([0.0, 0.0, 0.44, 0.63])
    a0.imshow(img, interpolation="lanczos")
    a0.set_axis_off()
    fig.text(0.004, 0.655, "(b)", fontsize=8, fontweight="bold", va="top")
    fig.text(0.037, 0.655, "$t = 12$", fontsize=8, va="top")
    # (c)-(f): filament through one axis-switching cycle, same camera as (b)
    d = np.load("fig1_fields/ell8x5_fields.npz")
    times = [4, 12, 20, 28]
    norm = plt.Normalize(-1.0, 1.0)
    cmap = plt.get_cmap("coolwarm")
    P, depth = _cam()
    zf = -2.2                                                     # floor for the projection
    curves = []
    for s_ in times:
        p = max((d[k] for k in d.files if k.startswith(f"snap_{s_}_")), key=len)
        q = p - p.mean(0)
        curves.append(np.vstack([q, q[:1]]))
    allp = np.vstack([P(np.c_[c[:, 0], c[:, 1], np.where(True, c[:, 2], 0)]) for c in curves] +
                     [P(np.c_[c[:, 0], c[:, 1], np.full(len(c), zf)]) for c in curves])
    lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
    w, h = 0.235, 0.475
    pos = [(0.455, 0.505), (0.69, 0.505), (0.455, 0.0), (0.69, 0.0)]
    for i, (q, (x0, y0)) in enumerate(zip(curves, pos)):
        ax = fig.add_axes([x0, y0, w, h])
        floor = np.c_[q[:, 0], q[:, 1], np.full(len(q), zf)]
        ax.plot(*P(floor).T, color="#c9c8c1", lw=0.7, zorder=1)
        for xa, ya in (((-6.5, 6.5), (0, 0)), ((0, 0), (-6.5, 6.5))):
            ax.plot(*P(np.c_[xa, ya, [zf, zf]]).T, color="#e9e8e2", lw=0.5, zorder=0)
        seg3 = np.stack([q[:-1], q[1:]], 1)
        order = np.argsort(depth(seg3.mean(1)))                    # far segments first
        seg2 = np.stack([P(seg3[:, 0]), P(seg3[:, 1])], 1)[order]
        zc = seg3.mean(1)[order, 2]
        ax.add_collection(LineCollection(seg2, colors=cmap(norm(zc)), linewidths=2.4, capstyle="round", zorder=2))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1] + 0.9)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.text(0.0, 1.0, f"({'cdef'[i]})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(0.13, 1.0, f"$t = {times[i]}$", transform=ax.transAxes, fontsize=8, va="top")
    cax = fig.add_axes([0.94, 0.3, 0.009, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1, 0, 1])
    cb.set_label("height", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2)
    fig.savefig(f"{OUT}/fig1.pdf", dpi=400)
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 2
def fig2():
    t, S, _ = smooth_run("ell_a1.1_b0.17_8x5.npz", 4.44, ("hx", "hy", "h2c"))
    # filament equation with alpha = 0 and the drift coefficient of a circular ring of the same size
    m1 = F.run(6.5, 4.0, 0.0, -1.80, t[-1] - 5, n=64, dt=0.01)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=True)
    for k, col in (("hx", BLUE), ("hy", ORANGE)):
        ax[0].plot(t, S[k], color=col, lw=1.4)
        ax[0].plot(m1["t"] + 5, m1[k], color=col, lw=0.9, ls="--")
    ax[0].set_ylabel("half-width")
    ax[1].plot(t, S["h2c"], color=INK, lw=1.4)
    ax[1].plot(m1["t"] + 5, m1["h2c"], color=MUTED, lw=0.9, ls="--")
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set(ylabel="saddle amplitude", xlabel="time", xlim=(0, 120), ylim=(-1.2, 1.2))
    for a_, s in zip(ax, "ab"):
        grid(a_)
        label(a_, s, x=-0.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=ORANGE, lw=1.4),
         Line2D([], [], color=INK, lw=1.4), Line2D([], [], color=MUTED, lw=0.9, ls="--")]
    fig.legend(h, [r"$h_x$", r"$h_y$", "simulation", r"filament eq., $\beta=1.80$"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0), columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.87))
    fig.savefig(f"{OUT}/fig2.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3
def theory_crossings(alpha, beta, xmax):
    """Axis-switch positions x = ln(R0/R) from the linear m=2 equations in log-time."""
    from scipy.integrate import solve_ivp
    rhs = lambda tau, y: [-3 * alpha * y[0] - 4 * beta * y[1], 3 * beta * y[0] - 4 * alpha * y[1]]
    tau = np.linspace(0, xmax / alpha, 20001)
    sol = solve_ivp(rhs, (0, tau[-1]), [1.0, 0.0], t_eval=tau, rtol=1e-9, atol=1e-12)
    a = sol.y[0]
    x = alpha * tau
    return [x[i] - a[i] * (x[i + 1] - x[i]) / (a[i + 1] - a[i]) for i in range(len(a) - 1) if a[i] * a[i + 1] < 0]


def fig3():
    P, t0 = 4.13, 6.0
    t, S, d = smooth_run("ellP_a1.1_b0.15_12x7.5.npz", P, ("hx", "hy", "h2c", "length"))
    r = F.run(10.76, 6.73, 0.566, -1.484, t[-1] - t0, n=64, dt=0.01)
    tm, M = model_until_valid(r, t0, 3.6)
    fig = plt.figure(figsize=(W1, 3.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.45)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharex=a0); a2 = fig.add_subplot(gs[2])
    for k, col, nm in (("hx", BLUE, r"$h_x$"), ("hy", ORANGE, r"$h_y$")):
        a0.plot(t, S[k], color=col, lw=1.4, label=nm)
        a0.plot(tm, M[k], color=col, lw=0.9, ls="--")
    a0.set(ylabel="half-width", ylim=(0, 12))
    a0.legend(loc="upper right", ncol=2)
    a1.plot(t, S["h2c"], color=INK, lw=1.4, label="simulation")
    a1.plot(tm, M["h2c"], color=MUTED, lw=0.9, ls="--", label="filament eq.")
    a1.axhline(0, color=MUTED, lw=0.5)
    a1.set(ylabel="saddle ampl.", xlabel="time", xlim=(0, 92), yticks=[-1, 0, 1])
    a1.legend(loc="lower right")
    # (c) axis switches vs ln(R0/R)
    cases = [("ellP_a1.1_b0.15_12x7.5.npz", 4.13, 0.566, -1.484, BLUE, r"$a=1.1,\ b=0.15$"),
             ("ellP_a0.8_b0.05_12x7.5.npz", 3.39, 1.362, -1.024, ORANGE, r"$a=0.8,\ b=0.05$")]
    from matplotlib.lines import Line2D
    for path, Pp, al, be, col, nm in cases:
        tt, SS, _ = smooth_run(path, Pp, ("hx", "hy", "length"))
        R = SS["length"] / (2 * np.pi)
        R0 = R[np.nanargmin(abs(tt - t0))]
        xs = [c for c in C.crossings(tt, SS["hx"] - SS["hy"]) if c > t0]
        Rx = [np.interp(c, tt, R) for c in xs]
        good = [(np.log(R0 / rr), k + 1) for k, rr in enumerate(Rx) if rr > 3.0]
        th = theory_crossings(al, abs(be), 1.4)
        a2.plot(th, np.arange(1, len(th) + 1), "o", mfc="white", mec=col, ms=6, mew=1.0, zorder=2)
        if good:
            gx, gk = zip(*good)
            a2.plot(gx, gk, "o", color=col, ms=4.2, zorder=3)
    h = [Line2D([], [], ls="", marker="s", color=BLUE, ms=5), Line2D([], [], ls="", marker="s", color=ORANGE, ms=5),
         Line2D([], [], ls="", marker="o", color=INK, ms=4.2),
         Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=6, mew=1.0)]
    a2.legend(h, [r"$a=1.1,\ b=0.15$", r"$a=0.8,\ b=0.05$", "simulation", "linear theory"], loc="upper left",
              ncol=2, fontsize=7, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    a2.set(xlabel=r"$\ln(R_0/\bar R)$", ylabel="switch number", xlim=(0, 1.1), ylim=(0.4, 4.2),
           yticks=[1, 2, 3])
    for a_, s in zip((a0, a1, a2), "abc"):
        grid(a_)
        label(a_, s, x=-0.25)
    plt.setp(a0.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.98, bottom=0.11)
    fig.savefig(f"{OUT}/fig3.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 4
def fig4():
    files = [("circHB_R0_4.0.npz", 4.0), ("circS_R0_4.5.npz", 4.5), ("circHB_R0_5.0.npz", 5.0),
             ("circS_R0_5.5.npz", 5.5), ("circL_R0_6.npz", 6.0), ("circS_R0_6.5.npz", 6.5),
             ("circS_R0_7.5.npz", 7.5), ("circL_R0_9.npz", 9.0), ("circL_R0_13.npz", 13.0)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    cm = lambda x: plt.get_cmap("Greys")(0.35 + 0.65 * x)    # encodes R0, distinct from the time map
    for i, (f, R0) in enumerate(files):
        d = np.load(f)
        ok = np.isfinite(d["Rm"])
        t, R = d["t"][ok], d["Rm"][ok]
        g = np.abs(R - median_filter(R, 25, mode="nearest")) < 0.4
        t, R = t[g], R[g]
        Rs = C.smooth(t, R, 4.44)
        m = (t <= 200) & np.isfinite(Rs)
        ax[0].plot(t[m], Rs[m], color=cm(i / (len(files) - 1)), lw=1.1)
    ax[0].axhspan(4.5, 4.9, color=GRID, zorder=0)
    ax[0].text(197, 3.75, r"$R^\ast$", ha="right", va="center", fontsize=8,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].set(xlabel="time", ylabel=r"ring radius $\overline{R}$", xlim=(0, 200), ylim=(0, 12.5))
    # alpha, beta vs K from sliding windows of every circular run with a vertical box >= 16
    # (the Lz = 14 runs circH_* are excluded: that box height produced spurious ring growth)
    import alpha_stationarity as A
    pts = [A.windows(f) for f in VALID_CIRCLES + LARGE_CIRCLES]
    big = A.windows("sf2/circXXL_R0_38.npz")
    pts.append(big[big[:, 0] <= 145])        # later windows are affected by the buckling instability
    pts = np.concatenate([p for p in pts if len(p)])
    sc = ax[1].scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap="viridis", s=9, vmin=0, vmax=300, lw=0)
    ax[2].scatter(pts[:, 1], -pts[:, 3], c=pts[:, 0], cmap="viridis", s=9, vmin=0, vmax=300, lw=0)
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set(xlabel=r"curvature $K = 1/\bar R$", ylabel=r"tension $\alpha$")
    ax[2].set(xlabel=r"curvature $K = 1/\bar R$", ylabel=r"drift coefficient $\beta$")
    for a_, s in zip(ax, "abc"):
        grid(a_)
        label(a_, s, x=-0.36, y=1.12)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.905, 0.25, 0.009, 0.6])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("time", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2)
    fig.savefig(f"{OUT}/fig4.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 5
def fig5():
    rows = list(csv.DictReader(open("period_table_r2.csv")))
    sets = [("a=1.1 b=0.15", BLUE, r"$b=0.15$"), ("a=1.1 b=0.16", AQUA, r"$b=0.16$"),
            ("a=1.1 b=0.17 (8x5)", ORANGE, r"$b=0.17$, small"),
            ("a=1.1 b=0.17 (12x7.5)", YELLOW, r"$b=0.17$, large"), ("a=1.1 b=0.18", MAGENTA, r"$b=0.18$"),
            ("a=0.8 b=0.09 (meander)", MUTED, "meandering")]
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    lim = [0, 45]
    ax.plot(lim, lim, color=MUTED, lw=0.9)
    ax.plot(lim, [1.33 * v for v in lim], color=INK, lw=0.9, ls="--")
    ax.text(44.5, 38.6, "1 : 1", fontsize=8, color=MUTED, ha="right", va="top")
    ax.text(31.5, 43.2, "1.33 : 1", fontsize=8, color=INK, ha="right")
    for run, col, nm in sets:
        rr = [r for r in rows if r["run"] == run]
        ax.scatter([float(r["T2"]) for r in rr], [float(r["Th"]) for r in rr], s=18,
                   marker="^" if "meander" in run else "o", color=col, edgecolor="white", linewidth=0.5,
                   zorder=3, label=nm)
    ax.set(xlim=lim, ylim=lim, xlabel="predicted half-period", ylabel="observed half-period")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.1, labelspacing=0.25, borderaxespad=0.3,
              markerscale=0.9, frameon=False)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 6
def _beta_curve_runs(tmin=50.0):
    """Late-time (K, |beta|) windows of all valid circular runs, grouped by run."""
    import alpha_stationarity as A
    groups = []
    for f in VALID_CIRCLES:
        w = A.windows(f)
        if len(w):
            w = w[w[:, 0] >= tmin]
            if len(w):
                groups.append(np.c_[w[:, 1], np.abs(w[:, 3])])
    return groups


def _local_ratio(groups, K, deg=2):
    """1 + K beta'/beta from a polynomial fit of beta(K) to the pooled windows."""
    P = np.concatenate(groups)
    p = np.polyfit(P[:, 0], P[:, 1], deg)
    return 1 + K * np.polyval(np.polyder(p), K) / np.polyval(p, K)


def fig6():
    from matplotlib.lines import Line2D
    dec = np.load("mode_decompose.npy")    # m, amp, R, omega, ratio, W, Q, beta_zeta, beta_r, beta_drift, gamma, rms
    nt = np.load("new_tests.npz")
    tl, lg = nt["tilt"], nt["large"]
    sl = np.load("sf2/sf2_linear.npy")     # k, gamma/k^2, err, |beta|, err, ...
    bk = np.load("sf2/buckle.npz")["fits"]  # m, R, k, growth, gamma/k^2, omega, sqrt(bz br)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.0), gridspec_kw=dict(height_ratios=[1.15, 1]))
    # (a) ring couplings relative to the drift coefficient, versus k = m / R (bars: fit-choice ranges)
    a = ax[0]
    a.axhline(1, color=MUTED, lw=0.6)
    a.scatter([1 / tl[0]], [tl[4] / tl[5]], marker="o", s=30, color=BLUE, zorder=4)
    for m, mk in ((2, "s"), (3, "D")):
        d = dec[dec[:, 0] == m]
        k = m / d[:, 2].mean()
        for col, j, dk in ((BLUE, 7, -0.006), (ORANGE, 8, 0.006)):
            y = d[:, j] / d[:, 9]
            a.errorbar([k + dk], [y.mean()], yerr=[[y.mean() - y.min()], [y.max() - y.mean()]], fmt=mk, ms=5,
                       color=col, elinewidth=0.8, capsize=2, zorder=4)
        L = lg[(lg[:, 0] == m) & np.isfinite(lg[:, 8])]
        kL = L[:, 3].mean()
        for col, j, dk in ((BLUE, 6, -0.006), (ORANGE, 7, 0.006)):
            y = L[:, j] / L[:, 8]
            a.errorbar([kL + dk], [y.mean()], yerr=[[y.mean() - y.min()], [y.max() - y.mean()]], fmt=mk, ms=5,
                       mfc="white", mec=col, color=col, mew=1.1, elinewidth=0.8, capsize=2, zorder=4)
    a.text(0.30, 0.88, r"$\beta_\zeta$", color=BLUE, fontsize=9, ha="center")
    a.text(0.30, 0.50, r"$\beta_r$", color=ORANGE, fontsize=9, ha="center")
    a.text(0.225, 1.09, r"$m=1$", fontsize=7, color=MUTED, va="center")
    a.set(xlabel=r"wavenumber $k = m/\bar R$", ylabel=r"coupling $/\,\beta_\mathrm{drift}$",
          xlim=(0.15, 0.68), ylim=(0.0, 1.3))
    # (b) binormal coefficient: straight filaments (small amplitude) and ring modes
    b_ = ax[1]
    b_.errorbar(sl[:, 0], sl[:, 3], yerr=sl[:, 4], fmt="o-", color=BLUE, ms=4.5, lw=1.0, )
    for m, mk in ((2, "s"), (3, "D")):
        d = dec[dec[:, 0] == m]
        b_.scatter([m / d[:, 2].mean()], [d[:, 5].mean()], marker=mk, s=26, color=INK, zorder=4)
        L = lg[lg[:, 0] == m]
        b_.scatter([L[:, 3].mean()], [L[:, 4].mean()], marker=mk, s=26, facecolor="white", edgecolor=INK,
                   linewidth=1.1, zorder=4)
    b_.scatter(bk[:, 2], bk[:, 6], marker="^", s=30, color=AQUA, zorder=4)
    b_.text(0.80, 1.17, "straight\nfilament", color=BLUE, fontsize=7, ha="center", va="bottom")
    b_.text(0.59, 1.36, r"$\bar R \approx 34$", color=AQUA, fontsize=7, ha="left")
    b_.text(0.30, 1.22, "ring modes", color=INK, fontsize=7, ha="center")
    b_.set(xlabel=r"wavenumber $k$", ylabel=r"binormal coefficient", xlim=(0.15, 0.85), ylim=(0.8, 2.45))
    for a_, s_ in zip(ax, "ab"):
        grid(a_)
        label(a_, s_, x=-0.22, y=1.08)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(f"{OUT}/fig6.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 7
def fig7():
    bk = np.load("sf2/buckle.npz")
    T = bk["T"]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.6), gridspec_kw=dict(height_ratios=[1, 1.15]))
    # (a) radial and vertical deviation along the ring at t = 190
    p = bk["last"]
    c = p.mean(0)
    th = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    rho = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    keep = np.abs(rho - np.median(rho)) < 6
    th, rho, z = th[keep], rho[keep], p[keep, 2]
    nb = 360
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    n = np.bincount(b, minlength=nb)
    ok = n > 0
    ang = np.degrees((np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi)
    rb = np.bincount(b, rho, nb)[ok] / n[ok]
    zb = np.bincount(b, z, nb)[ok] / n[ok]
    ax[0].plot(ang[ok], rb - rb.mean(), color=BLUE, lw=0.9, label=r"radial")
    ax[0].plot(ang[ok], zb - zb.mean(), color=ORANGE, lw=0.9, label=r"vertical")
    ax[0].set(xlabel=r"polar angle $\theta$ (degrees)", ylabel="deviation", xlim=(-180, 180), ylim=(-1.4, 1.8))
    ax[0].set_xticks([-180, -90, 0, 90, 180])
    ax[0].legend(loc="upper right", ncol=2, fontsize=8, handletextpad=0.3, columnspacing=0.8)
    # (b) mode amplitudes versus time
    cols = {4: MUTED, 8: YELLOW, 12: BLUE, 16: ORANGE, 20: AQUA}
    for m, col in cols.items():
        ax[1].semilogy(T, bk[f"amp{m}"], color=col, lw=1.0, label=rf"$m={m}$")
    ax[1].set(xlabel="time", ylabel=r"mode amplitude", xlim=(40, 200), ylim=(3e-3, 1.5))
    ax[1].legend(loc="upper left", ncol=3, fontsize=8, handletextpad=0.3, columnspacing=0.8)
    for a_, s_ in zip(ax, "ab"):
        grid(a_)
        label(a_, s_, x=-0.2, y=1.08)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(f"{OUT}/fig7.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = [x if x == "M" else int(x) for x in sys.argv[1:]] or ["M", 1, 2, 3, 4, 5, 6, 7]
    for n in which:
        globals()[f"fig{n}" if str(n) != "M" else "figM"]()
        print(f"wrote {OUT}/fig{n}.pdf")
