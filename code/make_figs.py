"""Figures of the elliptic-scroll-ring paper (Chaos, AIP).

Writes fig1.pdf ... fig6.pdf (main text) and figS1.pdf ... figS4.pdf (supplementary material) as vector
PDFs: single column 3.37 in, double column 6.69 in, all text >= 8 pt, panel letters, no internal titles.
Run in the BARKLEY folder (runs_barkley_v2.sh outputs and their analysis products); the supplementary
models and the revision runs are read from SUPP and REVISION (folders.py). FIG_OUT sets the output
directory (default figs).

    python make_figs.py            # all figures
    python make_figs.py 3 5 S2     # selected figures
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402

import ellipse_compare as C                     # noqa: E402
import filament_flow as F                       # noqa: E402
import halfcycle as H                           # noqa: E402
import halfcycle_uniform as HU                  # noqa: E402
import period_analysis as PA                    # noqa: E402
import folders as FD                            # noqa: E402

OUT = os.environ.get("FIG_OUT", "figs")         # output directory (created when figures are written)
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


# ----------------------------------------------------------------------------- data helpers
def smooth_run(path, period, keys):
    d = np.load(path)
    ok = np.isfinite(d["hx"]) & (d["ncurves"] == 1)     # single closed filaments (mask_frames.py)
    t = d["t"][ok]
    return t, {k: C.smooth(t, d[k][ok], period) for k in keys}, d


def theory_crossings(alpha, beta, xmax):
    """Axis-switch positions x = ln(R0/R) from the linear m=2 equations in log-time."""
    from scipy.integrate import solve_ivp
    rhs = lambda tau, y: [-3 * alpha * y[0] - 4 * beta * y[1], 3 * beta * y[0] - 4 * alpha * y[1]]
    tau = np.linspace(0, xmax / alpha, 20001)
    sol = solve_ivp(rhs, (0, tau[-1]), [1.0, 0.0], t_eval=tau, rtol=1e-9, atol=1e-12)
    a = sol.y[0]
    x = alpha * tau
    return [x[i] - a[i] * (x[i + 1] - x[i]) / (a[i + 1] - a[i]) for i in range(len(a) - 1) if a[i] * a[i + 1] < 0]


def fit_shrink_drift(path, period, t0=6.0, Rmin=3.6):
    """Constant alpha = -1/2 dR^2/dt and beta = dz/d(int dt/R) of a shrinking ring, fitted from t0
    while Rbar = L/2pi > Rmin (single closed filaments, period-averaged); also returns the
    period-averaged half-widths at t0."""
    d = np.load(path)
    ok = np.isfinite(d["hx"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    R = C.smooth(t, d["length"][ok], period) / (2 * np.pi)
    z = C.smooth(t, d["z"][ok], period)
    m = (t >= t0) & np.isfinite(R) & np.isfinite(z)
    m &= np.cumsum(m & ~(R > Rmin)) == 0                 # stop at the first R <= Rmin
    alpha = -0.5 * np.polyfit(t[m], R[m] ** 2, 1)[0]
    s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[m][1:] + 1 / R[m][:-1]) * np.diff(t[m]))])
    beta = np.polyfit(s, z[m], 1)[0]
    i = np.argmin(abs(t - t0))
    hx, hy = (C.smooth(t, d[k][ok], period)[i] for k in ("hx", "hy"))
    return alpha, beta, hx, hy


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


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


# ----------------------------------------------------------------------------- Fig. 1
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
        if i == 3:      # scale bar, 5 space units; the projection is orthographic, so horizontal lengths are true
            x0, y0 = lo[0] + 0.4, lo[1] + 0.35
            ax.plot([x0, x0 + 5.0], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.text(x0 + 2.5, y0 + 0.25, "5", fontsize=8, ha="center", va="bottom")
    cax = fig.add_axes([0.94, 0.3, 0.009, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1, 0, 1])
    cb.set_label("height", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2)
    fig.savefig(f"{OUT}/fig1.pdf", dpi=400)
    plt.close(fig)


# ----------------------------------------------------------------------------- shared helpers
def measured_shape(t, S, t0, n=96):
    """Closed curve with the measured m = 2 content of the reaction-diffusion ring at t0: polar radius
    R0 + r2c cos 2th + r2s sin 2th and height z2c cos 2th + z2s sin 2th (angle-weighted Fourier parts),
    with R0 = R_m - |r2|^2/(2 R_m) (second-order relation between R_m and the polar mean radius); the
    curve is scaled to the measured filament length."""
    g = lambda k: float(np.interp(t0, t, S[k]))
    Rm, r2c, r2s, z2c, z2s, L = (g(k) for k in ("Rm", "r2c", "r2s", "z2c", "z2s", "length"))
    R0 = Rm - (r2c ** 2 + r2s ** 2) / (2 * Rm)
    th = 2 * np.pi * np.arange(n) / n
    rho = R0 + r2c * np.cos(2 * th) + r2s * np.sin(2 * th)
    X = np.c_[rho * np.cos(th), rho * np.sin(th), z2c * np.cos(2 * th) + z2s * np.sin(2 * th)]
    LX = np.linalg.norm(np.diff(np.r_[X, X[:1]], axis=0), axis=1).sum()
    return X * (L / LX)


def smooth_all(path, period, keys=("hx", "hy", "Rm", "length", "z", "r2c", "r2s", "z2c", "z2s")):
    d = np.load(path)
    ok = np.isfinite(d["hx"]) & PA.single_closed(d)
    t = d["t"][ok]
    return t, {k: C.smooth(t, d[k][ok], period) for k in keys}


def follow_size(t, S, t0, X0, cal, beta_sign=-1.0, n=96):
    """Lowest-order law with the circular-ring drift coefficient at the instantaneous size: binormal
    flow is scale invariant, so a ring of length L(t) with coefficient beta_c(L / 2 pi) runs on the
    clock s = int (beta_c / beta_ref) (L0 / L)^2 dt with lengths scaled by L / L0 (L: period-averaged
    length of the reaction-diffusion ring, L0 = L(t0)). alpha = 0."""
    g = (t >= t0) & np.isfinite(S["length"])
    tt, LL = t[g], S["length"][g]
    L0 = LL[0]
    bref = float(cal.beta(L0 / (2 * np.pi)))
    rate = cal.beta(LL / (2 * np.pi)) / bref * (L0 / LL) ** 2
    s = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tt))])
    r = F.run(0, 0, 0.0, beta_sign * bref, s[-1], n=n, every=s[-1] / 1200, X0=X0)
    tm = np.interp(r["t"], s, tt)
    sc = np.interp(tm, tt, LL) / L0
    return dict(t=tm, hx=r["hx"] * sc, hy=r["hy"] * sc, z2c=r["z2c"] * sc, bref=bref)


# ----------------------------------------------------------------------------- Fig. 2
def fig2():
    path, per, t0 = "rerun_r2/ell_a1.1_b0.17_8x5.npz", 4.44, 4.0
    t, S = smooth_all(path, per)
    cal = HU.barkley_calibration(0.17)
    X0 = measured_shape(t, S, t0)
    M = follow_size(t, S, t0, X0, cal)
    print(f"fig2: start t0 = {t0}, L/2pi = {np.interp(t0, t, S['length']) / (2 * np.pi):.3f}, beta_c(start) = {M['bref']:.3f}")
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=True)
    for k, col in (("hx", BLUE), ("hy", ORANGE)):
        ax[0].plot(t, S[k], color=col, lw=1.4)
        ax[0].plot(M["t"], M[k], color=col, lw=0.9, ls="--")
    ax[0].set_ylabel("half-width")
    ax[1].plot(t, S["z2c"], color=INK, lw=1.4)
    ax[1].plot(M["t"], M["z2c"], color=MUTED, lw=0.9, ls="--")
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set(ylabel="saddle amplitude", xlabel="time", xlim=(0, 120), ylim=(-1.2, 1.2))
    for a_, s in zip(ax, "ab"):
        grid(a_)
        label(a_, s, x=-0.25)
    h = [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], color=ORANGE, lw=1.4),
         Line2D([], [], color=INK, lw=1.4), Line2D([], [], color=MUTED, lw=0.9, ls="--")]
    fig.legend(h, [r"$h_x$", r"$h_y$", "simulation", "lowest-order law"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.55, 1.0), columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.87))
    fig.savefig(f"{OUT}/fig2.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3
def lowest_order_run(X0, cal, T, n=96, dt=0.002, beta_sign=-1.0):
    """Lowest-order law dX/dt = alpha K N + beta K B with the coefficients of circular rings of the same
    size, alpha_c(Rbar) and beta_c(Rbar), Rbar = L / 2 pi of the model ring, uniform along the ring
    (well posed for alpha_c > 0; the curvature-only law with the measured alpha(K) is not, because
    d(alpha K)/dK < 0 for 2.7 < R < 5.3 at b = 0.15)."""
    K0, K1 = cal.Krange
    co = lambda Rb: (float(np.polyval(cal.pa, np.clip(1 / Rb, K0, K1))), beta_sign * float(np.polyval(cal.pb, np.clip(1 / Rb, K0, K1))))
    return F.run(0, 0, 0.0, beta_sign, T, n=n, dt=dt, every=0.05, coeff_of_R=co, X0=X0)


def fig3():
    path, per, t0 = "rerun_r2/ellP_a1.1_b0.15_12x7.5.npz", 4.13, 4.0
    t, S = smooth_all(path, per)
    cal = HU.barkley_calibration(0.15)
    X0 = measured_shape(t, S, t0)
    r = lowest_order_run(X0, cal, 88.0)
    RLm = r["length"] / (2 * np.pi)
    keep = np.cumsum(~(RLm > 2.5)) == 0             # until the model ring first drops below R = 2.5
    tm = r["t"][keep] + t0
    RL = S["length"] / (2 * np.pi)
    R0 = float(np.interp(t0, t, RL))
    print(f"fig3: start t0 = {t0}, L/2pi = {R0:.3f}; model R at t0 + 20, 40, 60: "
          f"{np.interp([20, 40, 60], r['t'], RLm).round(3).tolist()} vs simulation {np.interp([t0 + 20, t0 + 40, t0 + 60], t, RL).round(3).tolist()}")
    fig = plt.figure(figsize=(W1, 3.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.45)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharex=a0); a2 = fig.add_subplot(gs[2])
    for k, col, nm in (("hx", BLUE, r"$h_x$"), ("hy", ORANGE, r"$h_y$")):
        a0.plot(t, S[k], color=col, lw=1.4, label=nm)
        a0.plot(tm, r[k][keep], color=col, lw=0.9, ls="--")
    a0.set(ylabel="half-width", ylim=(0, 12))
    a0.legend(loc="upper right", ncol=2)
    a1.plot(t, S["z2c"], color=INK, lw=1.4, label="simulation")
    a1.plot(tm, r["z2c"][keep], color=MUTED, lw=0.9, ls="--", label="lowest-order law")
    a1.axhline(0, color=MUTED, lw=0.5)
    a1.set(ylabel="saddle ampl.", xlabel="time", xlim=(0, 92), yticks=[-1, 0, 1])
    a1.legend(loc="lower right", fontsize=7)
    # (c) switches against ln(R0 / Rbar)
    xs = [c for c in C.crossings(t, S["hx"] - S["hy"]) if c > t0]
    sim = [np.log(R0 / np.interp(c, t, RL)) for c in xs if np.interp(c, t, RL) > 3.0]
    tvalid = r["t"][keep][-1]                       # model ring above R = 2.5 (before its collapse)
    ms = [c for c in C.crossings(r["t"], r["hx"] - r["hy"]) if 0.5 < c < tvalid]
    low = [np.log(RLm[0] / np.interp(c, r["t"], RLm)) for c in ms if np.interp(c, r["t"], RLm) > 3.0]
    sim_t = [c for c in xs if np.interp(c, t, RL) > 3.0]
    low_t = [c + t0 for c in ms if np.interp(c, r["t"], RLm) > 3.0]
    print(f"fig3(c): simulation  ln(R0/R) {np.round(sim, 3).tolist()} at t {np.round(sim_t, 2).tolist()}")
    print(f"         lowest-order ln(R0/R) {np.round(low, 3).tolist()} at t {np.round(low_t, 2).tolist()}")
    for xx, kw in ((low, dict(marker="o", mfc="white", mec=MUTED, ms=6.5, mew=1.0)),
                   (sim, dict(marker="o", color=INK, ms=4.2))):
        a2.plot(xx, np.arange(1, len(xx) + 1), ls="", zorder=3, **kw)
    h = [Line2D([], [], ls="", marker="o", color=INK, ms=4.2),
         Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, ms=6.5, mew=1.0)]
    a2.legend(h, ["simulation", "lowest-order law"], loc="upper left", fontsize=7, handletextpad=0.2, borderaxespad=0.2)
    a2.set(xlabel=r"$\ln(R_0/\bar R)$", ylabel="switch number", xlim=(0, 1.1), ylim=(0.4, 4.4), yticks=[1, 2, 3, 4])
    for a_, s in zip((a0, a1, a2), "abc"):
        grid(a_)
        label(a_, s, x=-0.25)
    plt.setp(a0.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.98, bottom=0.11)
    fig.savefig(f"{OUT}/fig3.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 4
def seed_radius(path):
    d = np.load(path)
    return float(d["R0"]) if "R0" in d.files else float(d["Ax"])


def fig4():
    import matplotlib.colors as mcolors
    files = [("circHB_R0_4.0.npz", 4.0), ("circS_R0_4.5.npz", 4.5), ("circHB_R0_5.0.npz", 5.0),
             ("circS_R0_5.5.npz", 5.5), ("circL_R0_6.npz", 6.0), ("circS_R0_6.5.npz", 6.5),
             ("circS_R0_7.5.npz", 7.5), ("circL_R0_9.npz", 9.0), ("circL_R0_13.npz", 13.0)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    norm = mcolors.LogNorm(4, 40)
    cmap = plt.get_cmap("viridis")
    for f, R0 in files:
        d = np.load(f)
        ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
        t, R = d["t"][ok], d["Rm"][ok]
        Rs = C.smooth(t, R, 4.44)
        m = (t <= 200) & np.isfinite(Rs)
        ax[0].plot(t[m], Rs[m], color=cmap(norm(R0)), lw=1.1)
    ax[0].axhspan(4.5, 4.9, color=GRID, zorder=0)
    ax[0].text(197, 3.75, r"$R^\ast$", ha="right", va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax[0].set(xlabel="time", ylabel=r"ring radius $\bar R$", xlim=(0, 200), ylim=(0, 12.5))
    # (b, c): the calibration windows (10-unit windows, t >= 20) colored by seed radius; R0 = 38 open
    paths = [f for f in VALID_CIRCLES] + LARGE_CIRCLES + [os.path.join(FD.REVISION, "circ17_R0_26.npz")]
    for p in paths:
        w = HU.circle_windows(p, 4.44, 10.0, 5.0, 20.0)
        if len(w):
            c = cmap(norm(seed_radius(p)))
            ax[1].scatter(w[:, 0], w[:, 1], color=c, s=7, lw=0)
            ax[2].scatter(w[:, 0], w[:, 2], color=c, s=7, lw=0)
    w = HU.circle_windows("sf2/circXXL_R0_38_snaps.npz", 4.44, 10.0, 5.0, 20.0)
    w = w[: int(np.searchsorted(np.arange(20, 200, 5.0) + 5, 145))]           # before the buckling (t <= 145)
    for j, a_ in ((1, ax[1]), (2, ax[2])):
        a_.scatter(w[:, 0], w[:, j], s=9, facecolor="white", edgecolor=cmap(norm(38)), lw=0.7, zorder=3)
    cal = HU.barkley_calibration(0.17)
    K = np.linspace(*cal.Krange, 200)
    ax[1].plot(K, np.polyval(cal.pa, K), color=INK, lw=0.8)
    ax[2].plot(K, np.polyval(cal.pb, K), color=INK, lw=0.8)
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set(xlabel=r"curvature $K = 1/R_m$", ylabel=r"tension $\alpha$", xlim=(0, 0.32))
    ax[2].set(xlabel=r"curvature $K = 1/R_m$", ylabel=r"drift coefficient $\beta$", xlim=(0, 0.32))
    for a_, s in zip(ax, "abc"):
        grid(a_)
        label(a_, s, x=-0.36, y=1.12)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.87, 1))
    cax = fig.add_axes([0.885, 0.25, 0.009, 0.6])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, ticks=[4, 6, 10, 20, 40])
    from matplotlib.ticker import NullFormatter, NullLocator
    cb.ax.yaxis.set_minor_locator(NullLocator())
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.set_yticklabels(["4", "6", "10", "20", "40"])
    cb.set_label("seed radius", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2)
    fig.savefig(f"{OUT}/fig4.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 5
LAMBDA = {}       # filled from REVISION/spiral2d/*.json by load_lambda()


def load_lambda():
    import glob
    for f in glob.glob(os.path.join(FD.REVISION, "spiral2d", "*.json")):
        d = json.load(open(f))
        key = ("barkley", round(d["b"], 2)) if d["model"] == "barkley" else ("panfilov", 0.775 if "0.775" in d["fields"] else 0.8)
        LAMBDA[key] = d
    LAMBDA[("atri", 0.1)] = dict(wavelength=120.0, period=16.6)      # SI Sec. III


def run_lambda(r):
    if r["model"] == "barkley":
        b = float(r["run"].split("b=")[1][:4])
        return LAMBDA[("barkley", round(b, 2))]["wavelength"]
    if r["model"] == "panfilov":
        return LAMBDA[("panfilov", 0.775 if "0.775" in r["run"] else 0.8)]["wavelength"]
    return LAMBDA[("atri", 0.1)]["wavelength"]


def fig5():
    load_lambda()
    rows = read_csv(os.path.join(FD.REVISION, "uniform_halfcycles.csv"))
    sets = [("a=1.1 b=0.15", BLUE, r"$b=0.15$"), ("a=1.1 b=0.16", AQUA, r"$b=0.16$"),
            ("a=1.1 b=0.17 (8x5)", ORANGE, r"$b=0.17$, small"), ("a=1.1 b=0.17 (12x7.5)", YELLOW, r"$b=0.17$, large"),
            ("a=1.1 b=0.18", MAGENTA, r"$b=0.18$")]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    r2 = [r for r in rows if r["model"] == "barkley" and r["event"] == "r2"]
    top = max(max(r["T_obs"] for r in r2), max(r["T_obs"] / r["ratio_c"] for r in r2))
    lim = [0, 5 * np.ceil(1.05 * top / 5)]
    q17 = np.mean([r["ratio_c"] for r in r2 if "b=0.17" in r["run"]])
    a.plot(lim, lim, color=MUTED, lw=0.9)
    a.plot(lim, [q17 * v for v in lim], color=INK, lw=0.9, ls="--")
    a.text(0.99 * lim[1], 0.86 * lim[1], "1 : 1", fontsize=8, color=MUTED, ha="right", va="top")
    a.text(0.66 * lim[1], 0.96 * lim[1], f"{q17:.2f} : 1 ($b=0.17$)", fontsize=8, color=INK, ha="right")
    for run, col, nm in sets:
        rr = [r for r in r2 if r["run"] == run]
        a.scatter([r["T_obs"] / r["ratio_c"] for r in rr], [r["T_obs"] for r in rr], s=18, color=col,
                  edgecolor="white", linewidth=0.5, zorder=3, label=nm)
    a.set(xlim=lim, ylim=lim, xlabel="predicted half-period", ylabel="observed half-period")
    a.set_aspect("equal")
    a.legend(loc="lower right", fontsize=7, handletextpad=0.1, labelspacing=0.25, borderaxespad=0.3)
    # (b) ratio against the ring diameter in wavelengths, all three models (planar-moment half-cycles, R_m)
    b_ = ax[1]
    z2 = [r for r in rows if r["event"] == "z2" and r["radius"] == "Rm"
          and not (r["model"] == "barkley" and r["Rbar"] < HU.RBAR_MIN_CAL)]      # beta(K) poorly constrained there
    groups = [("barkley", (lambda run: lambda r: r["run"] == run)(run), col, "o", None) for run, col, _ in sets]
    groups += [("barkley", lambda r: "mode" in r["run"], ORANGE, "s", "weak $m=2$"),
               ("panfilov", lambda r: True, INK, "D", "Panfilov–Rudenko"),
               ("atri", lambda r: True, MUTED, "^", "Atri")]
    b_.axhline(1, color=MUTED, lw=0.6)
    for model, sel, col, mk, nm in groups:
        rr = [r for r in z2 if r["model"] == model and sel(r)]
        x = [2 * r["Rbar"] / run_lambda(r) for r in rr]
        b_.scatter(x, [r["ratio_c"] for r in rr], s=16, marker=mk, color=col, edgecolor="white", lw=0.4, zorder=3)
        res = np.array([r["residual"] for r in rr])
        alt = np.array([r["ratio_c"] / r["eq5_global"] for r in rr])
        lo, hi = np.minimum(res, alt), np.maximum(res, alt)
        b_.vlines(x, lo, hi, color=col, lw=0.7, zorder=1)
        b_.scatter(x, res, s=16, marker=mk, facecolor="white", edgecolor=col, lw=0.8, zorder=2)
    named = [g for g in groups if g[4]]
    h = [Line2D([], [], ls="", marker="o", color=MUTED, ms=4.5)] + \
        [Line2D([], [], ls="", marker=mk, color=col, ms=4.5) for _, _, col, mk, _ in named]
    h += [Line2D([], [], ls="", marker="o", color=MUTED, ms=4.5),
          Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, ms=4.5)]
    b_.legend(h, ["Barkley (as in a)"] + [g[4] for g in named] + ["frequency ratio", "after Eq. (5)"], loc="upper right",
              fontsize=7, ncol=1, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.3)
    b_.set(xlabel=r"ring diameter $2R_m/\lambda$", ylabel="predicted / observed frequency", ylim=(0.9, 1.65), xlim=(0, 2.6))
    for a_, s in zip(ax, "ab"):
        grid(a_)
        label(a_, s, x=-0.24, y=1.03)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig5.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 6
def mode_ranges():
    """Couplings of the ring modes relative to beta_c of circular rings of the same radius, with their
    full ranges (mode_summary.py)."""
    import mode_summary
    return mode_summary.modes()


def fig6():
    import mode_summary
    import mode_wavenumber as MWN
    rows = mode_ranges() + mode_summary.large_modes()
    for r in rows:
        print(f"fig6: {r['ring']:6s} m={r['m']} k={r['k']:.3f}  bz/bd {r['bz']:.3f} [{r['bz_rng'][0]:.3f}, {r['bz_rng'][1]:.3f}]"
              f"  br/bd {r['br']:.3f} [{r['br_rng'][0]:.3f}, {r['br_rng'][1]:.3f}]  sqrt(bz br) {r['W']:.3f}")
    tl = np.load("new_tests.npz")["tilt"]
    sl = np.load("sf2/sf2_linear.npy")
    bk = np.load("sf2/buckle.npz")["fits"]
    ctl = json.load(open(os.path.join(FD.REVISION, "controls.json")))["straight"]
    s15 = [c for c in ctl if c["label"].startswith("b=0.15")]
    win = [w for w in json.load(open(os.path.join(FD.REVISION, "mode_wavenumber.json")))["windows"] if w["used"]]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.5), gridspec_kw=dict(height_ratios=[1, 1.25]))
    a = ax[0]
    a.axhline(1, color=MUTED, lw=0.6)
    cal = HU.barkley_calibration(0.17)
    a.scatter([1 / tl[0]], [tl[4] / tl[5]], marker="o", s=26, facecolor="white", edgecolor=MUTED, lw=1.0, zorder=4)
    a.text(1 / tl[0] + 0.015, tl[4] / tl[5] + 0.05, "tilt ($m=1$)", fontsize=7, color=MUTED, va="bottom")
    mk = {2: "s", 3: "D", 4: "v"}
    for r in rows:
        filled = r["ring"] == "R*"
        a.plot([r["k"] - 0.022, r["k"] + 0.022], [r["br_loc"]] * 2, color=ORANGE, lw=1.0, ls=(0, (2, 1)), zorder=3)   # Eq. (5)
        for col, key, dk in ((BLUE, "bz", -0.006), (ORANGE, "br", 0.006)):
            lo, hi = r[key + "_rng"]
            a.errorbar([r["k"] + dk], [r[key]], yerr=[[r[key] - lo], [hi - r[key]]], fmt=mk[r["m"]], ms=5, color=col,
                       mfc=col if filled else "white", mec=col, mew=1.1, elinewidth=0.8, capsize=2, zorder=4)
    a.text(0.66, 0.37, r"$\beta_\zeta$", color=BLUE, fontsize=9, ha="center")
    a.text(0.66, 0.86, r"$\beta_r$", color=ORANGE, fontsize=9, ha="center")
    a.set(xlabel=r"wavenumber $k = m/R_m$", ylabel=r"coupling $/\,\beta_\mathrm{c}$", xlim=(0.13, 0.88), ylim=(0.0, 1.3))
    b_ = ax[1]
    Kc, Kx = np.linspace(0.13, cal.Krange[1], 40), np.linspace(cal.Krange[1], 0.88, 40)
    b_.plot(Kc, MWN.beta_log(Kc), color=INK, lw=1.0, zorder=2)
    b_.plot(Kx, MWN.beta_log(Kx), color=INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    b_.errorbar(sl[:, 0], sl[:, 3], yerr=sl[:, 4], fmt="o-", color=BLUE, ms=4.5, lw=1.0)
    b_.plot([c["k"] for c in s15], [c["beta"] for c in s15], marker="o", mfc="none", mec=MUTED, ms=6.5, mew=1.1, ls=":", color=MUTED, lw=1.1,
            zorder=5)                                 # on top: at k = 0.45 and 0.6 it coincides with b = 0.17
    b_.scatter(bk[:, 2], bk[:, 6], marker="^", s=30, color=AQUA, zorder=4)
    for w in win:
        col = MUTED if w["b"] == 0.15 else YELLOW
        b_.scatter([w["k"]], [w["bz"]], marker=mk[w["m"]], s=15, facecolor="white", edgecolor=col, linewidth=0.9, zorder=5)
    b_.scatter([1 / tl[0]], [float(cal.beta(tl[0])) * tl[4] / tl[5]], marker="o", s=26, facecolor="white", edgecolor=MUTED, lw=1.0, zorder=6)
    for r in rows:
        c = float(cal.beta(r["R"]))
        b_.errorbar([r["k"]], [r["bz"] * c], yerr=[[(r["bz"] - r["bz_rng"][0]) * c], [(r["bz_rng"][1] - r["bz"]) * c]], fmt=mk[r["m"]],
                    ms=5, color=INK, mfc=INK if r["ring"] == "R*" else "white", mec=INK, mew=1.1, elinewidth=0.8, capsize=1.5, zorder=6)
    b_.text(0.785, 1.14, "straight\nfilament", color=BLUE, fontsize=7, ha="center", va="bottom")
    b_.text(0.40, 1.74, r"straight, $b=0.15$", color=MUTED, fontsize=7, ha="left")
    b_.text(0.60, 1.42, r"$\bar R \approx 34$", color=AQUA, fontsize=7, ha="left")
    b_.text(0.46, 0.93, r"$\beta_\mathrm{c}(K = k)$", color=INK, fontsize=7, ha="left")
    b_.legend(handles=[Line2D([], [], marker="s", ls="", ms=4.5, mfc=INK, mec=INK, label=r"rings, $b = 0.17$"),
                       Line2D([], [], marker="s", ls="", ms=3.5, mfc="white", mec=YELLOW, mew=0.9, label=r"rings, $b = 0.16$"),
                       Line2D([], [], marker="s", ls="", ms=3.5, mfc="white", mec=MUTED, mew=0.9, label=r"rings, $b = 0.15$")],
              loc="upper right", fontsize=7, handletextpad=0.3)
    b_.set(xlabel=r"wavenumber $k$", ylabel=r"$\beta_\zeta$ or binormal coefficient", xlim=(0.13, 0.88), ylim=(0.6, 2.6))
    for a_, s_ in zip(ax, "ab"):
        grid(a_)
        label(a_, s_, x=-0.22, y=1.08)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(f"{OUT}/fig6.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. S1: calibrations at b = 0.15-0.18
def figS1():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    cols = {0.15: BLUE, 0.16: AQUA, 0.17: ORANGE, 0.18: MAGENTA}
    for b, col in cols.items():
        cal = HU.barkley_calibration(b)
        P = cal.P
        ax[0].scatter(P[:, 0], P[:, 1], s=6, color=col, lw=0, alpha=0.8)
        ax[1].scatter(P[:, 0], P[:, 2], s=6, color=col, lw=0, alpha=0.8, label=f"$b={b}$")
        K = np.linspace(*cal.Krange, 200)
        ax[0].plot(K, np.polyval(cal.pa, K), color=col, lw=0.9)
        ax[1].plot(K, np.polyval(cal.pb, K), color=col, lw=0.9)
    ax[0].axhline(0, color=MUTED, lw=0.5)
    ax[0].set(xlabel=r"curvature $K = 1/R_m$", ylabel=r"tension $\alpha$", xlim=(0, 0.5))
    ax[1].set(xlabel=r"curvature $K = 1/R_m$", ylabel=r"drift coefficient $\beta$", xlim=(0, 0.5))
    ax[1].legend(loc="upper right", fontsize=7, markerscale=2)
    for a_, s in zip(ax, "ab"):
        grid(a_)
        label(a_, s, x=-0.2, y=1.08)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{OUT}/figS1.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. S2: a mode fit and the induced twist
def figS2():
    from mode_theory2 import fit_mode
    d = np.load(os.path.join(FD.REVISION, "modeR_m2_a0.15_lv.npz"))
    m = 2
    ok = np.isfinite(d["Rm"]) & np.isfinite(d["r2c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    S = {k: C.smooth(t, d[k][ok], 4.44) for k in ("r2c", "z2c", "Rm", "tw2c")}
    sel = (t >= 6) & (t <= 125) & np.isfinite(S["r2c"]) & np.isfinite(S["z2c"])
    R = S["Rm"][sel].mean()
    f = fit_mode(t[sel], S["r2c"][sel], S["z2c"][sel], m * np.sqrt(m * m - 1) * 1.2 / R ** 2)
    tt = t[sel] - t[sel][0]
    e = np.exp(-f["gamma"] * tt)
    # model curves from the fitted parameters (fit_mode canonical form)
    from scipy.optimize import least_squares
    def res(q):
        A, B, g, w, p, dd, cr, cz = q
        ee = np.exp(-g * tt)
        return np.r_[A * ee * np.cos(w * tt + p) + cr - S["r2c"][sel], B * ee * np.sin(w * tt + p + dd) + cz - S["z2c"][sel]]
    best = min((least_squares(res, [np.ptp(S["r2c"][sel]) / 2, np.ptp(S["z2c"][sel]) / 2, f["gamma"], f["omega"], p0, 0, 0, 0])
                for p0 in np.linspace(0, 2 * np.pi, 8, endpoint=False)), key=lambda s: s.cost)
    A, B, g, w, p, dd, cr, cz = best.x
    ee = np.exp(-g * tt)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.2), sharex=True)
    ax[0].plot(t, S["r2c"], color=BLUE, lw=1.3, label=r"$r_2$")
    ax[0].plot(t, S["z2c"], color=ORANGE, lw=1.3, label=r"$\zeta_2$")
    ax[0].plot(t[sel], A * ee * np.cos(w * tt + p) + cr, color=INK, lw=0.7, ls="--")
    ax[0].plot(t[sel], B * ee * np.sin(w * tt + p + dd) + cz, color=INK, lw=0.7, ls="--", label="fit")
    ax[0].axhline(0, color=MUTED, lw=0.5)
    ax[0].set(ylabel="mode amplitude")
    ax[0].legend(loc="upper right", ncol=3, fontsize=7)
    ax[1].plot(t, S["tw2c"], color=INK, lw=1.2, label=r"rotation phase, $\cos 2\theta$ part")
    ax[1].plot(t, 0.17 * S["r2c"], color=BLUE, lw=0.8, ls=":", label=r"$0.17\,r_2$")
    ax[1].axhline(0, color=MUTED, lw=0.5)
    ax[1].set(ylabel="phase (rad)", xlabel="time", xlim=(0, 130))
    ax[1].legend(loc="upper right", fontsize=7)
    for a_, s in zip(ax, "ab"):
        grid(a_)
        label(a_, s, x=-0.22)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(f"{OUT}/figS2.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- Figs. S3, S4: the other two models

def _stacked_halfwidths(ax, path, period, beta, lab, unit=""):
    S = H.smoothed(path, period)
    M = H.model_series_grown(S, beta)          # follows the measured size change of the ring
    ax.plot(S["t"], S["hx"], color=BLUE, lw=1.3)
    ax.plot(S["t"], S["hy"], color=ORANGE, lw=1.3)
    ax.plot(M["t"], M["hx"], color=BLUE, lw=0.8, ls="--")
    ax.plot(M["t"], M["hy"], color=ORANGE, lw=0.8, ls="--")
    ax.text(0.01, 0.96, lab, transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.set_ylabel("half-width" + unit)
    return S, M


def figS3():
    pan = H.read_csv(os.path.join(FD.SUPP, "halfcycles_panfilov.csv"))
    rows = [r for r in read_csv(os.path.join(FD.REVISION, "uniform_halfcycles.csv")) if r["model"] == "panfilov" and r["radius"] == "Rm"]
    import pan_alpha
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.12, wspace=0.32)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); a2 = fig.add_subplot(gs[:, 1])
    for ax, fn, gf, lab in ((a0, "pell_gf0.775_20x12.npz", 0.775, r"$g_f = 0.775$ (slightly negative tension)"),
                            (a1, "pell_gf0.8_24x15.npz", 0.8, r"$g_f = 0.8$ (weakly positive tension)")):
        beta = [r["beta"] for r in pan if r["file"] == fn][0]
        _stacked_halfwidths(ax, os.path.join(FD.SUPP, fn), pan_alpha.PERIOD[gf], beta, lab)
    a0.set_ylim(11.5, 22.5)                    # headroom: the panel titles must not touch the curves
    a0.set_yticks([12.5, 15.0, 17.5, 20.0])
    a1.set_ylim(7.0, 27.0)
    a1.set_xlabel("time")
    plt.setp(a0.get_xticklabels(), visible=False)
    h = [Line2D([], [], color=BLUE, lw=1.3), Line2D([], [], color=ORANGE, lw=1.3), Line2D([], [], color=MUTED, lw=0.8, ls="--")]
    a1.legend(h, [r"$h_x$", r"$h_y$", "filament eq."], loc="lower left", ncol=3, fontsize=8)
    runs = [("gf=0.775 dx=0.6 80^2 x 50", "0.775\n0.6"), ("gf=0.8 dx=0.6 88^2 x 50", "0.8\n0.6"),
            ("gf=0.8 dx=0.6 120^2 x 80", "0.8\n0.6,\nlarge"), ("gf=0.8 dx=0.4 88^2 x 50", "0.8\n0.4"),
            ("gf=0.8 dx=0.3 76^2 x 40", "0.8\n0.3")]
    for i, (run, lab) in enumerate(runs):
        q = np.array([r["ratio_c"] for r in rows if r["run"] == run])
        res = np.array([r["residual"] for r in rows if r["run"] == run])
        a2.plot(np.full(len(q), i) - 0.1, q, "o", color=INK, ms=3.5)
        a2.plot(np.full(len(res), i) + 0.1, res, "o", mfc="white", mec=INK, ms=3.5, mew=0.8)
    a2.axhline(1.0, color=MUTED, lw=0.8)
    a2.set_xticks(range(len(runs)))
    a2.set_xticklabels([r[1] for r in runs], fontsize=7)
    a2.set(xlim=(-0.6, len(runs) - 0.4), ylim=(0.9, 1.1), xlabel=r"$g_f$ / grid spacing $\Delta x$",
           ylabel="predicted / observed frequency")
    a2.legend([Line2D([], [], ls="", marker="o", color=INK, ms=3.5), Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=3.5)],
              ["frequency ratio", "after Eq. (5)"], loc="upper left", fontsize=7)
    for a_, s_ in zip((a0, a1, a2), "abc"):
        grid(a_)
        label(a_, s_, x=-0.14 if a_ is not a2 else -0.42, y=1.06)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.17)
    fig.savefig(f"{OUT}/figS3.pdf")
    plt.close(fig)


def figS4():
    import halfcycle_tables as HT
    allrows = H.read_csv(os.path.join(FD.SUPP, "halfcycles_atri.csv"))
    rows = [r for r in read_csv(os.path.join(FD.REVISION, "uniform_halfcycles.csv")) if r["model"] == "atri" and r["radius"] == "Rm"]
    fn = "aell2_P0.1_100x60.npz"
    beta = [r["beta"] for r in allrows if r["file"] == fn][0]
    S = H.smoothed(os.path.join(FD.SUPP, fn), HT.ATRI_PERIOD)
    M = H.model_series_grown(S, beta)
    x0 = C.crossings(S["t"], S["z2c"])[0]
    k = (S["t"] > x0) & (S["t"] < x0 + 3 * HT.ATRI_PERIOD) & np.isfinite(S["z2c"])
    sgn = np.sign(np.polyfit(S["t"][k], S["z2c"][k], 1)[0] * np.polyfit(M["t"][:60], M["z2c"][:60], 1)[0])
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.12, wspace=0.32)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); a2 = fig.add_subplot(gs[:, 1])
    a0.plot(S["t"], S["hx"], color=BLUE, lw=1.3); a0.plot(S["t"], S["hy"], color=ORANGE, lw=1.3)
    a0.plot(M["t"], M["hx"], color=BLUE, lw=0.8, ls="--"); a0.plot(M["t"], M["hy"], color=ORANGE, lw=0.8, ls="--")
    a0.set_ylabel(r"half-width ($\mu$m)")
    a0.legend([Line2D([], [], color=BLUE, lw=1.3), Line2D([], [], color=ORANGE, lw=1.3), Line2D([], [], color=MUTED, lw=0.8, ls="--")],
              [r"$h_x$", r"$h_y$", "filament eq."], loc="upper left", ncol=3, fontsize=8)
    a1.plot(S["t"], sgn * S["z2c"], color=INK, lw=1.3)
    a1.plot(M["t"], M["z2c"], color=MUTED, lw=0.8, ls="--")
    a1.axhline(0, color=MUTED, lw=0.5)
    a1.set(ylabel=r"saddle ampl. ($\mu$m)", xlabel="time (s)")
    plt.setp(a0.get_xticklabels(), visible=False)
    for run, mk, mfc, lab in (("dx=3.0 600^2 x 280 seed 100 x 60", "o", INK, r"$100\times60$"),
                              ("dx=3.0 660^2 x 280 seed 130 x 80", "s", "white", r"$130\times80$"),
                              ("dx=3.0 600^2 x 280 seed 100 x 88", "^", BLUE, r"$100\times88$"),
                              ("dx=3.0 660^2 x 280 seed 120 x 105", "v", "white", r"$120\times105$"),
                              ("dx=2.0 600^2 x 280 seed 100 x 60", "D", "white", r"$100\times60$, $\Delta x=2\,\mu$m")):
        rr = [r for r in rows if r["run"] == run]
        col = BLUE if "88" in run or "105" in run else (MUTED if "dx=2.0" in run else INK)
        a2.plot([r["Rbar"] for r in rr], [r["ratio_c"] for r in rr], mk, color=col, mfc=mfc if mfc != BLUE else col, mec=col, ms=5, label=lab)
    a2.axhline(1.0, color=MUTED, lw=0.8)
    a2.set(xlabel=r"mean radius $\bar R_m$ ($\mu$m)", ylabel="predicted / observed frequency", ylim=(0.92, 1.10))
    a2.legend(loc="lower left", fontsize=6.5, title=r"seed ($\mu$m)", title_fontsize=7, handletextpad=0.2)
    for a_, s_ in zip((a0, a1, a2), "abc"):
        grid(a_)
        label(a_, s_, x=-0.14 if a_ is not a2 else -0.42, y=1.06)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14)
    fig.savefig(f"{OUT}/figS4.pdf")
    plt.close(fig)


ALL = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6,
       "S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
        print(f"wrote {OUT}/fig{k}.pdf")
