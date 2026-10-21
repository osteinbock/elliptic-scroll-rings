"""One half-cycle analysis for all three models (revision: referee comments A7a, B-M1/M2, C2).

For every elliptic run of the Barkley, Panfilov-Rudenko and Atri models:
  * frames with a single closed filament, smoothed over one rotation period (gap-aware);
  * half-cycles between zero crossings of the saddle amplitude z2c (planar moments, event "z2"; the
    convention of halfcycle.py) and, for the Barkley runs, between sign changes of r2c (axis switches,
    event "r2"; the intervals of period_analysis_r2.py, with its filters);
  * radius conventions: for z2 events R_m (arc-length-weighted horizontal distance from the centre) and
    L / 2 pi, each with the finite-amplitude factor f of the filament equation (alpha = 0) computed in the
    same convention from the planar ellipse at the start of the half-cycle (halfcycle.model_factor); for
    r2 events L / 2 pi with f(eps) of period_analysis.amplitude_factor;
  * two references for the drift coefficient: circular rings of the same radius, beta_c(R), from the
    model's own calibration (Calibration), and the elliptic ring's own drift during the half-cycle
    (slope of z against int dt / R, as period_analysis_r2.py);
  * predictions as phase integrals over the measured radius R(t) of the half-cycle: with
    omega(t) = sqrt(48 beta^2 - alpha^2) / (2 R(t)^2) (Eq. 2 for m = 2), the ratio of observed to
    predicted half-period is  ratio = int omega dt / (pi f).  For a ring of constant size this is
    T_obs / (f T_lin); for a shrinking ring it avoids the bias of a half-cycle mean of R^2 (Referee B, M1c);
  * the curvature-only local law, Eq. (5): its frequency is lower by sqrt(beta_r / beta), beta_r =
    d(beta K)/dK, so eq5_factor = int omega dt / int omega sqrt(beta_r/beta) dt and residual =
    ratio / eq5_factor. beta_r / beta comes from local quadratic fits of beta(K) (Calibration.ratio_r).

    python halfcycle_uniform.py OUT.csv          (folders SUPP, BARKLEY, REVISION as in folders.py)

Rows of OUT.csv: one per half-cycle, event and radius convention; columns are listed in COLUMNS.
"""
import os
import sys
import csv
import numpy as np

import ellipse_compare as C
import halfcycle as H
import period_analysis as PA

HERE = os.path.dirname(os.path.abspath(__file__))
import folders as FD          # noqa: E402  (SUPP, BARKLEY, REVISION)
ASPECT_MIN = 1.05             # z2 events: nearly circular rings have ill-defined planar moments
RBAR_MIN_CAL = 3.5            # Barkley b = 0.15, 0.16: beta(K) poorly constrained below (flagged in Tables S1-S2, Fig. 5b)


# ----------------------------------------------------------------------------- calibration
def circle_windows(path, period, win, step, tmin, Rmin=1.5):
    """(K, alpha, |beta|) in windows of a circular ring: alpha = -1/2 d(R^2)/dt, beta = dz/ds with
    s = int dt / R (exact for a circle of constant beta), K = mean of 1/R; R = R_m, single
    closed curves only, smoothed over one rotation period."""
    d = np.load(path)
    ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
    if "closed" in d.files:
        ok &= d["closed"] == 1
    t, R, z = d["t"][ok], d["Rm"][ok], d["z"][ok]
    R, z = C.smooth(t, R, period), C.smooth(t, z, period)
    out = []
    for t0 in np.arange(tmin, t[-1] - win + 1e-9, step):
        m = (t >= t0) & (t < t0 + win) & np.isfinite(R) & np.isfinite(z)
        if m.sum() < 10 or R[m].min() < Rmin:
            continue
        tt, RR, zz = t[m], R[m], z[m]
        s = np.concatenate([[0], np.cumsum(0.5 * (1 / RR[1:] + 1 / RR[:-1]) * np.diff(tt))])
        out.append((np.mean(1 / RR), -0.5 * np.polyfit(tt, RR ** 2, 1)[0], abs(np.polyfit(s, zz, 1)[0])))
    return np.array(out).reshape(-1, 3)


class Calibration:
    """alpha(K) and |beta|(K) of circular rings from pooled windows: global polynomial fits for the
    values, and local quadratic fits (|K - K0| < dK, at least nmin windows) for beta_r / beta = 1 + K
    beta'/beta, which is sensitive to the global fit near the minimum of beta(K)."""

    def __init__(self, points, deg=3, label="", dK=0.05, nmin=8):
        self.P = np.asarray(points)
        self.deg = min(deg, len(np.unique(np.round(self.P[:, 0], 4))) - 1)
        self.pa = np.polyfit(self.P[:, 0], self.P[:, 1], self.deg)
        self.pb = np.polyfit(self.P[:, 0], self.P[:, 2], self.deg)
        self.Krange = (self.P[:, 0].min(), self.P[:, 0].max())
        self.label, self.dK, self.nmin = label, dK, nmin
        self.Kg = np.linspace(*self.Krange, 120)
        self.rg = np.array([self._local_ratio(K) for K in self.Kg])

    def _local_ratio(self, K):
        m = np.abs(self.P[:, 0] - K) < self.dK
        if m.sum() < self.nmin:
            return 1 + K * np.polyval(np.polyder(self.pb), K) / np.polyval(self.pb, K)
        p = np.polyfit(self.P[m, 0] - K, self.P[m, 2], 2)
        return 1 + K * p[1] / p[2]

    def beta(self, R):
        return np.polyval(self.pb, 1 / np.asarray(R, float))

    def alpha(self, R):
        return np.polyval(self.pa, 1 / np.asarray(R, float))

    def beta_r_over_beta(self, R):
        """Global-fit value of d(beta K)/dK / beta (Eq. 5)."""
        K = 1 / np.asarray(R, float)
        return 1 + K * np.polyval(np.polyder(self.pb), K) / np.polyval(self.pb, K)

    def ratio_r(self, R):
        """Local-fit value of beta_r / beta, interpolated (held constant outside the calibrated range)."""
        return np.interp(1 / np.asarray(R, float), self.Kg, self.rg)

    def inside(self, R):
        return self.Krange[0] <= 1 / R <= self.Krange[1]


def barkley_calibration(b, deg=3, win=10.0, step=5.0):
    """Circular Barkley rings (a = 1.1): b = 0.17 from the rings of Fig. 4 (BARKLEY) and the R0 = 26 ring
    (REVISION), other b from REVISION."""
    period = {0.15: 4.13, 0.16: 4.318, 0.17: 4.44, 0.18: 4.75}[b]
    if b == 0.17:
        import make_figs as MF
        files = [os.path.join(FD.BARKLEY, f) for f in MF.VALID_CIRCLES] + [os.path.join(FD.BARKLEY, f) for f in MF.LARGE_CIRCLES]
        files += [f for f in [os.path.join(FD.REVISION, "circ17_R0_26.npz")] if os.path.exists(f)]
    else:
        import glob
        files = sorted(glob.glob(os.path.join(FD.REVISION, f"circ{int(round(100 * b))}_R0_*.npz")))
    pts = [circle_windows(f, period, win, step, 20.0) for f in files if os.path.exists(f)]
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    cal = Calibration(np.concatenate(pts), deg, f"Barkley b={b}")
    cal.nrings = len(pts)                     # rings that contribute at least one window
    return cal


def panfilov_calibration(gf, dx=0.6, deg=1):
    """Circular Panfilov-Rudenko rings: at dx = 0.6 the seeds 14 and 22 (SUPP) and 18 (REVISION)."""
    paths = {g: [os.path.join(FD.SUPP, f"pcirc_gf{g}_R14.npz"), os.path.join(FD.SUPP, f"pcirc_gf{g}_R22.npz"),
                 os.path.join(FD.REVISION, f"pcirc_gf{g}_R18.npz")] for g in (0.775, 0.8)}[gf]
    if dx == 0.4:
        paths = [os.path.join(FD.SUPP, "pcircHR_gf0.8_R14.npz")]
    elif dx == 0.3:
        paths = [os.path.join(FD.SUPP, "pcircXR_gf0.8_R14.npz")]
    import pan_alpha
    per = pan_alpha.PERIOD[gf]
    pts = [circle_windows(p, per, 2 * per, per, 2 * per, Rmin=2) for p in paths]
    return Calibration(np.concatenate(pts), deg, f"Panfilov gf={gf} dx={dx}", dK=1.0, nmin=10 ** 9)


def atri_calibration(dx=3.0, deg=1):
    import halfcycle_tables as HT
    files = ["acirc3_P0.1.npz", "acirc4_P0.1_c052.npz"] if dx == 3.0 else ["acirc_dx2_P0.1.npz"]
    pts = [circle_windows(os.path.join(FD.SUPP, f), HT.ATRI_PERIOD, 400.0, 200.0, 100.0, Rmin=20) for f in files]
    return Calibration(np.concatenate(pts), deg, f"Atri dx={dx}", dK=1.0, nmin=10 ** 9)


# ----------------------------------------------------------------------------- half-cycles
def phase(t, R, beta, alpha):
    """int omega dt over the samples, omega = sqrt(48 beta^2 - alpha^2) / (2 R^2); beta, alpha arrays or scalars."""
    w = np.sqrt(np.maximum(48 * np.asarray(beta) ** 2 - np.asarray(alpha) ** 2, 0)) / (2 * R ** 2)
    return np.trapz(w * np.ones_like(t), t)


def own_coefficients(t, R, z):
    """Own drift coefficient (dz/ds, s = int dt / R) and shrinkage (-1/2 dR^2/dt) over the samples."""
    s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[1:] + 1 / R[:-1]) * np.diff(t))])
    return abs(np.polyfit(s, z, 1)[0]), -0.5 * np.polyfit(t, R ** 2, 1)[0]


def predictions(t, R, z, f, calib):
    """Ratios of observed to predicted half-period for one half-cycle (samples t, R(t), z(t))."""
    bown, aown = own_coefficients(t, R, z)
    Rb = R.mean()
    row = dict(Rbar=Rb, R_end_over_start=R[-1] / R[0], beta_own=bown, alpha_own=aown, factor=f)
    row["ratio_own"] = phase(t, R, bown, aown) / (np.pi * f)
    if calib is not None:
        bc, ac = calib.beta(R), calib.alpha(R)
        Pc = phase(t, R, bc, ac)
        Ploc = phase(t, R, bc * np.sqrt(np.clip(calib.ratio_r(R), 1e-6, None)), ac)
        disc = 48 * float(calib.beta(Rb)) ** 2 - float(calib.alpha(Rb)) ** 2
        row.update(beta_c=float(calib.beta(Rb)), alpha_c=float(calib.alpha(Rb)), own_over_c=bown / float(calib.beta(Rb)),
                   ratio_c=Pc / (np.pi * f), ratio_c_avg=(t[-1] - t[0]) / (f * 2 * np.pi * Rb ** 2 / np.sqrt(disc)),
                   eq5_factor=Pc / Ploc, eq5_global=1 / np.sqrt(float(calib.beta_r_over_beta(Rb))),
                   residual=Ploc / (np.pi * f), calib_inside=float(calib.inside(Rb)))
        # ratio_c is a ratio of mean frequencies; the literal ratio of durations uses the time at which the
        # predicted phase reaches pi f (held at the last frequency if that lies beyond the half-cycle)
        w = np.sqrt(np.maximum(48 * bc ** 2 - ac ** 2, 0)) / (2 * R ** 2)
        cum = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
        tp = np.interp(np.pi * f, cum, t) if cum[-1] >= np.pi * f else t[-1] + (np.pi * f - cum[-1]) / w[-1]
        row["ratio_c_duration"] = (t[-1] - t[0]) / (tp - t[0])
    return row


def z2_half_cycles(path, period, calib, model, run, t_contact=np.inf, radii=("Rm", "L")):
    """Half-cycles between zero crossings of z2c (planar moments), for both radius conventions."""
    S = H.smoothed(path, period)
    t = S["t"]
    x = C.crossings(t, S["z2c"])
    rows = []
    for a, b in zip(x[:-1], x[1:]):
        gap = C.max_gap(S["t_valid"], a, b)
        if gap > period / 2 or b > t_contact or b - a < 2 * period:
            continue
        hx0, hy0 = np.interp(a, t, S["hx"]), np.interp(a, t, S["hy"])
        if max(hx0, hy0) / min(hx0, hy0) < ASPECT_MIN:
            continue
        for rad in radii:
            R = S["Rm"] if rad == "Rm" else S["RL"]
            m = (t >= a) & (t <= b) & np.isfinite(R) & np.isfinite(S["z"]) & np.isfinite(S["RL"])
            if m.sum() < 5:
                continue
            sc = np.sqrt(hx0 * hy0) / 5.0                       # f depends on the shape only; solve at R ~ 5
            f = H.model_factor(hx0 / sc, hy0 / sc, 1.0, radius=rad)
            # own drift from the centre height against int dt / (L / 2 pi), as period_analysis_r2.py
            row = predictions(t[m], R[m], S["z"][m], f, calib)
            if rad == "Rm":                                     # own coefficients always from L / 2 pi
                bo, ao = own_coefficients(t[m], S["RL"][m], S["z"][m])
                row.update(beta_own=bo, alpha_own=ao, ratio_own=phase(t[m], R[m], bo, ao) / (np.pi * f))
                if calib is not None:
                    row["own_over_c"] = bo / row["beta_c"]
            row.update(model=model, run=run, event="z2", radius=rad, t_start=a, t_end=b, T_obs=b - a,
                       aspect=max(hx0, hy0) / min(hx0, hy0), max_gap=gap)
            rows.append(row)
    return rows


def r2_half_cycles(path, period, calib, model, run):
    """Half-cycles between sign changes of r2c (axis switches) with the intervals and filters of
    period_analysis_r2.intervals_r2 and f(eps) in the L / 2 pi convention."""
    import period_analysis_r2 as P2
    fac = PA.amplitude_factor() if os.path.exists("amp_factor.npy") else np.load(os.path.join(FD.BARKLEY, "amp_factor.npy"))
    d = np.load(path)
    ok = np.isfinite(d["r2c"]) & np.isfinite(d["length"])
    t = d["t"][ok]
    good = PA.single_closed(d)[ok]
    L, z = (C.smooth(t, np.where(good, d[k][ok], np.nan), period) for k in ("length", "z"))
    RL = L / (2 * np.pi)
    rows = []
    for r in P2.intervals_r2(path, period):
        m = (t >= r["t0"]) & (t <= r["t1"]) & np.isfinite(RL) & np.isfinite(z)
        f = float(np.interp(r["eps"], fac[:, 0], fac[:, 1]))
        row = predictions(t[m], RL[m], z[m], f, calib)
        row.update(model=model, run=run, event="r2", radius="L", t_start=r["t0"], t_end=r["t1"], T_obs=r["Th"],
                   eps=r["eps"], max_gap=r["max_gap"])
        rows.append(row)
    return rows


COLUMNS = ["model", "run", "event", "radius", "t_start", "t_end", "T_obs", "Rbar", "R_end_over_start", "aspect",
           "eps", "factor", "beta_c", "alpha_c", "beta_own", "alpha_own", "own_over_c", "ratio_c", "ratio_c_avg", "ratio_c_duration",
           "ratio_own", "eq5_factor", "eq5_global", "residual", "calib_inside", "max_gap"]


def barkley_runs():
    """(path, period, b, label, image-contact time) of the rigid a = 1.1 Barkley elliptic runs and the R* mode runs."""
    src = {"ellP_a1.1_b0.15_12x7.5.npz": "rerun_r2/ellP_a1.1_b0.15_12x7.5.npz",
           "ell_a1.1_b0.17_8x5.npz": "rerun_r2/ell_a1.1_b0.17_8x5.npz",
           "ell_a1.1_b0.17_12x7.5.npz": "rerun_r2/ell_a1.1_b0.17_12x7.5.npz"}
    out = []
    for fn, a, b, lab, per, rigid in PA.RUNS:
        if not rigid or a != 1.1:
            continue
        path = os.path.join(FD.BARKLEY, src.get(fn, fn))
        out.append((path, per, b, lab, PA.image_contact_time(np.load(path), per), True))
    for amp in (0.08, 0.15):
        out.append((os.path.join(FD.BARKLEY, f"modeR_m2_a{amp}.npz"), 4.44, 0.17, f"a=1.1 b=0.17 mode m=2 amp {amp}", np.inf, False))
    return out


def summarize(rows):
    for ev, rad in (("z2", "Rm"), ("z2", "L"), ("r2", "L")):
        sel = [r for r in rows if r["event"] == ev and r["radius"] == rad]
        if not sel:
            continue
        print(f"\nevent {ev}, radius {rad}:")
        for (model, run) in dict.fromkeys((r["model"], r["run"]) for r in sel):
            rr = [r for r in sel if r["model"] == model and r["run"] == run]
            g = lambda k: np.round([r.get(k, np.nan) for r in rr], 3).tolist()
            print(f"  {model:8s} {run:32s} n={len(rr)}  circle {g('ratio_c')}  own {g('ratio_own')}  Eq.5 {g('eq5_factor')}"
                  f"  residual {g('residual')}  f {g('factor')}  R {g('Rbar')}")


def main(out_csv):
    os.chdir(FD.BARKLEY)                # period_analysis_r2 and amplitude_factor read relative paths
    rows, cal = [], {}
    for path, per, b, lab, tc, with_r2 in barkley_runs():
        if b not in cal:
            cal[b] = barkley_calibration(b)
            c = cal[b]
            print(f"Barkley b={b}: " + ("no calibration" if c is None else
                  f"{len(c.P)} windows, K {c.Krange[0]:.3f}-{c.Krange[1]:.3f}, deg {c.deg}"))
        rows += z2_half_cycles(path, per, cal[b], "barkley", lab, tc)
        if with_r2:
            rows += r2_half_cycles(path, per, cal[b], "barkley", lab)
    import pan_alpha
    import halfcycle_tables as HT
    for fn, gf, dx, dom, seed, circ in HT.PAN:
        rows += z2_half_cycles(os.path.join(FD.SUPP, fn), pan_alpha.PERIOD[gf], panfilov_calibration(gf, dx), "panfilov",
                               f"gf={gf} dx={dx} {dom}")
    for fn, dx, dom, seed, circ in HT.ATRI:
        rows += z2_half_cycles(os.path.join(FD.SUPP, fn), HT.ATRI_PERIOD, atri_calibration(dx), "atri", f"dx={dx} {dom} seed {seed}")
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    os.replace(tmp, out_csv)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    summarize(rows)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(FD.REVISION, "uniform_halfcycles.csv"))
