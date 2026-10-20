"""Helper module: the paper uses its RUNS, amplitude_factor, intervals and predict through
period_analysis_r2.py (Fig. 5); its own __main__ table and figure are exploratory.

Axis-switching half-periods of RD elliptic scroll rings vs filament-equation predictions.

For every interval between consecutive axis crossings (hx = hy) of a run, measure
  T_half (observed), Rbar = <L/2pi>, beta = d z / d(int dt/R), alpha = -1/2 d(R^2)/dt,
  eps = max |hx-hy|/(hx+hy) (amplitude)
and compare with
  linear, no tension:   T0 = pi Rbar^2 / (2 sqrt3 |beta|)
  with tension:         T1 = 2 pi Rbar^2 / sqrt(48 beta^2 - alpha^2)
  with tension + amplitude: T2 = T1 * f(eps), f from the alpha = 0 filament equation.
No fitted parameters: alpha and beta come from the same interval of the same run.

    python period_analysis.py          -> period_analysis.png, period_table.csv
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C
import filament_flow as F

EPS_MAX = 0.36          # largest relative ellipticity of amplitude_factor's grid

# file, a, b, label, spiral rotation period (for smoothing; from param_probe.py), rigid?
RUNS = [
    ("ellP_a1.1_b0.15_12x7.5.npz", 1.1, 0.15, "a=1.1 b=0.15", 4.13, True),
    ("ellB_a1.1_b0.16_12x7.5.npz", 1.1, 0.16, "a=1.1 b=0.16", 4.318, True),
    ("ell_a1.1_b0.17_8x5.npz",     1.1, 0.17, "a=1.1 b=0.17 (8x5)", 4.44, True),
    ("ell_a1.1_b0.17_12x7.5.npz",  1.1, 0.17, "a=1.1 b=0.17 (12x7.5)", 4.44, True),
    ("ellB_a1.1_b0.18_12x7.5.npz", 1.1, 0.18, "a=1.1 b=0.18", 4.750, True),
    ("ellP_a0.8_b0.05_12x7.5.npz", 0.8, 0.05, "a=0.8 b=0.05", 3.39, True),
    ("ell_b0.09_8x5.npz",          0.8, 0.09, "a=0.8 b=0.09 (meander)", 4.130, False),
]


def amplitude_factor(eps_grid=np.linspace(0.02, 0.36, 9)):
    """f(eps) = T_half(LIA, finite amplitude) / T_half(linear) for alpha = 0, beta = 1."""
    cache = "amp_factor.npy"
    if os.path.exists(cache):
        return np.load(cache)
    out = []
    for e in eps_grid:
        r = F.run(1 + e, 1 - e, 0.0, 1.0, 6.0, n=48, dt=0.0005, every=0.01)   # converged (checked n=64, dt/3)
        x = C.crossings(r["t"], r["hx"] - r["hy"])
        Rb = r["length"][0] / (2 * np.pi)
        Th = np.mean(np.diff(x)) if len(x) > 1 else 2 * x[0]
        out.append((e, Th / (np.pi * Rb ** 2 / (2 * np.sqrt(3)))))
    out = np.array(out)
    np.save(cache, out)
    return out


def single_closed(d):
    """Frames whose filament is one closed curve (all frames for files without the flag)."""
    g = d["ncurves"] == 1
    return g & (d["closed"] == 1) if "closed" in d.files else g


def intervals(path, period):
    d = np.load(path)
    ok = np.isfinite(d["hx"]) & np.isfinite(d["length"])
    t = d["t"][ok]
    good = single_closed(d)[ok]                 # other frames are bridged by the smoothing
    hx, hy, L, z = (C.smooth(t, np.where(good, d[k][ok], np.nan), period) for k in ("hx", "hy", "length", "z"))
    R = L / (2 * np.pi)
    x = C.crossings(t, hx - hy)
    rows = []
    for t0, t1 in zip(x[:-1], x[1:]):
        m = (t >= t0) & (t <= t1) & np.isfinite(R) & np.isfinite(z)
        if m.sum() < 12 or t1 - t0 < 2 * period:
            continue
        Rb = R[m].mean()
        s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[m][1:] + 1 / R[m][:-1]) * np.diff(t[m]))])
        beta = np.polyfit(s, z[m], 1)[0]
        alpha = -0.5 * np.polyfit(t[m], R[m] ** 2, 1)[0]
        eps = np.max(np.abs(hx[m] - hy[m]) / (hx[m] + hy[m]))
        nc = d["ncurves"][ok][m]
        # skip intervals near collapse, with tiny amplitude, beyond the range of the
        # finite-amplitude factor (eps <= 0.36), or after the ring broke up
        gap = C.max_gap(t[good], t0, t1)
        if Rb < 3.2 or eps < 0.04 or eps > EPS_MAX or np.mean(nc > 1) > 0.2 or gap > period / 2:
            continue
        rows.append(dict(t0=t0, t1=t1, Th=t1 - t0, R=Rb, beta=beta, alpha=alpha, eps=eps, max_gap=gap))
    return rows


def predict(r, fac):
    T0 = np.pi * r["R"] ** 2 / (2 * np.sqrt(3) * abs(r["beta"]))
    disc = 48 * r["beta"] ** 2 - r["alpha"] ** 2
    T1 = 2 * np.pi * r["R"] ** 2 / np.sqrt(disc) if disc > 0 else np.nan
    f = np.interp(r["eps"], fac[:, 0], fac[:, 1])
    return T0, T1, T1 * f


if __name__ == "__main__":
    fac = amplitude_factor()
    print("amplitude factor f(eps):", np.round(fac, 3).tolist())
    table = []
    for path, a, b, lab, P, rigid in RUNS:
        if not os.path.exists(path):
            print("missing", path)
            continue
        for r in intervals(path, P):
            T0, T1, T2 = predict(r, fac)
            table.append(dict(run=lab, a=a, b=b, rigid=rigid, **r, T0=T0, T1=T1, T2=T2))
            print(f"{lab:24s} t={r['t0']:6.1f}-{r['t1']:6.1f}  Th={r['Th']:5.1f}  R={r['R']:.2f}  "
                  f"beta={r['beta']:+.2f} alpha={r['alpha']:+.3f} eps={r['eps']:.2f}  "
                  f"T0={T0:5.1f} T1={T1:5.1f} T2={T2:5.1f}  obs/T2={r['Th'] / T2:.2f}")
    keys = ["run", "a", "b", "rigid", "t0", "t1", "Th", "R", "beta", "alpha", "eps", "T0", "T1", "T2", "max_gap"]
    with open("period_table.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in table:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---------------------------------------------------------------- figure
    INK, MUTED = "#1f1f1e", "#6b6a63"
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6b6a63"]
    labs = [r[3] for r in RUNS if os.path.exists(r[0])]
    col = {l: PAL[i % len(PAL)] for i, l in enumerate(labs)}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), dpi=120)

    # (a) observed vs predicted half-period (tension + amplitude)
    for l in labs:
        rr = [r for r in table if r["run"] == l]
        if not rr:
            continue
        mk = "o" if rr[0]["rigid"] else "^"
        ax[0].scatter([r["T2"] for r in rr], [r["Th"] for r in rr], s=36, color=col[l], marker=mk,
                      edgecolor="white", linewidth=1, label=l, zorder=3)
    lim = [0, 1.1 * max(max(r["Th"] for r in table), max(r["T2"] for r in table))]
    ax[0].plot(lim, lim, color=MUTED, lw=1)
    ax[0].fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="#e4e3dd", zorder=0)
    ax[0].set(xlim=lim, ylim=lim, xlabel="predicted half-period (tension + amplitude)",
              ylabel="observed half-period (RD)")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0].set_title("(a) every half-cycle, all runs  (band: ±10%)", fontsize=9, color=INK)

    # (b) normalized half-period vs model parameter b (a = 1.1 runs)
    sub = [r for r in table if r["a"] == 1.1]
    bs = np.array([r["b"] for r in sub])
    jit = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.003
    theta = np.array([r["Th"] * abs(r["beta"]) / r["R"] ** 2 for r in sub])
    th2 = np.array([r["T2"] * abs(r["beta"]) / r["R"] ** 2 for r in sub])
    ax[1].scatter(bs + jit, theta, s=34, color="#2a78d6", edgecolor="white", zorder=3, label="RD (each half-cycle)")
    ax[1].scatter(bs + jit, th2, s=30, facecolor="none", edgecolor="#eb6834", zorder=3,
                  label="prediction incl. tension + amplitude")
    ax[1].axhline(np.pi / (2 * np.sqrt(3)), color=INK, lw=1.2, label="linear theory  π/(2√3)")
    ax[1].set(xlabel="Barkley parameter b  (a = 1.1, ε = 0.02)",
              ylabel="normalized half-period  T½·|β| / R̄²")
    ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("(b) scaled oscillation time vs model parameter", fontsize=9, color=INK)

    # (c) raw half-period vs b at a reference radius (scaled by (R_ref/R)^2), with prediction
    Rref = 6.0
    traw = np.array([r["Th"] * (Rref / r["R"]) ** 2 for r in sub])
    tpred = np.array([r["T2"] * (Rref / r["R"]) ** 2 for r in sub])
    ax[2].scatter(bs + jit, traw, s=34, color="#2a78d6", edgecolor="white", zorder=3, label="RD")
    ax[2].scatter(bs + jit, tpred, s=30, facecolor="none", edgecolor="#eb6834", zorder=3, label="prediction")
    # analytical curve from a linear fit of the measured beta(b) at R ~ R_ref
    bb = np.array([r["b"] for r in sub]); be = np.array([abs(r["beta"]) for r in sub])
    pb = np.polyfit(bb, be, 1)
    xs = np.linspace(bb.min() - 0.005, bb.max() + 0.005, 50)
    ax[2].plot(xs, np.pi * Rref ** 2 / (2 * np.sqrt(3) * np.polyval(pb, xs)), color=INK, lw=1.2,
               label="π R²/(2√3 |β(b)|), linear")
    ax[2].set(xlabel="Barkley parameter b  (a = 1.1, ε = 0.02)",
              ylabel=f"half-period scaled to R = {Rref:g}")
    ax[2].legend(frameon=False, fontsize=7)
    ax[2].set_title("(c) oscillation time vs model parameter", fontsize=9, color=INK)
    for a_ in ax:
        a_.grid(color="#e4e3dd", lw=0.6)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig("period_analysis.png")
    print("wrote period_analysis.png, period_table.csv")
