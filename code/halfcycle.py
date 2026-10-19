"""Axis-switching half-cycles of an RD elliptic ring and their prediction (Panfilov, Atri).

One analysis for both supplementary models, with the conventions of the main text:
  * frames: single closed filament curve (fragmented or open frames are dropped; the
    gap-aware smoothing of ellipse_compare.smooth bridges them);
  * all ring measures are smoothed over one rotation period;
  * half-cycles run between zero crossings of the saddle amplitude z2c (planar moments);
  * Rbar = mean over the half-cycle of the arc-length-weighted radius Rm (default) or of
    L/(2 pi); Rm is insensitive to small-scale wiggles of the filament, which lengthen L
    (by 1-3% in the Panfilov-Rudenko model);
  * linear prediction T_lin = 2 pi Rbar^2 / sqrt(48 beta^2 - alpha^2), Eq. (modes);
  * finite-amplitude factor f: the filament equation (alpha = 0, same beta) started from
    the planar ellipse measured at the start of the half-cycle; f = its half-period
    divided by its own linear half-period pi R^2 / (2 sqrt3 |beta|), with
    R the model's radius in the chosen convention, i.e. the same
    radius convention as for the RD ring (Rm: mean over the model half-cycle);
  * ratio = T_obs / (f T_lin).
"""
import numpy as np
import ellipse_compare as C
import filament_flow as F


def load_single(path):
    d = np.load(path)
    if "zfix" not in d.files:
        raise RuntimeError(f"{path}: z not repaired (run zfix_repair.py)")
    ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
    if "closed" in d.files:
        ok &= d["closed"] == 1
    return d, ok


def smoothed(path, period):
    d, ok = load_single(path)
    t = d["t"][np.isfinite(d["Rm"])]
    out = {"t": t}
    for k in ("hx", "hy", "z2c", "r2c", "length", "Rm", "z"):
        y = np.where(ok, d[k], np.nan)[np.isfinite(d["Rm"])]
        out[k] = C.smooth(t, y, period)
    out["RL"] = out["length"] / (2 * np.pi)
    out["frac_used"] = ok[np.isfinite(d["Rm"])].mean()
    return out


def model_factor(hx0, hy0, beta, n=48, radius="Rm"):
    """Finite-amplitude factor f for a planar ellipse with half-widths hx0, hy0, relative to
    the linear half-period at the model's own radius: radius = "Rm" (arc-length-weighted
    radius, mean over the model half-cycle) or "L" (L / 2 pi, conserved by binormal flow)."""
    beta = abs(beta)
    R0 = np.sqrt(hx0 * hy0)                       # time scale only
    T0 = np.pi * R0 ** 2 / (2 * np.sqrt(3) * beta)
    r = F.run(hx0, hy0, 0.0, beta, 2.5 * T0, n=n, dt=None, every=T0 / 400)   # stable default dt
    x = [c for c in C.crossings(r["t"], r["h2c"]) if c > 0.2 * T0]
    if not x:
        return np.nan
    if radius == "L":
        Rr = r["length"][0] / (2 * np.pi)
    else:
        Rr = r["Rm"][r["t"] <= x[0]].mean()
    return x[0] / (np.pi * Rr ** 2 / (2 * np.sqrt(3) * beta))


def half_cycles(path, period, beta, alpha=0.0, radius="Rm"):
    S = smoothed(path, period)
    S["R"] = S["Rm"] if radius == "Rm" else S["RL"]
    t = S["t"]
    x = C.crossings(t, S["z2c"])
    rows = []
    for a, b in zip(x[:-1], x[1:]):
        m = (t >= a) & (t <= b) & np.isfinite(S["R"])
        if m.sum() < 5:
            continue
        Rb = S["R"][m].mean()
        disc = 48 * beta ** 2 - alpha ** 2
        Tlin = 2 * np.pi * Rb ** 2 / np.sqrt(disc)
        hx0, hy0 = np.interp(a, t, S["hx"]), np.interp(a, t, S["hy"])
        f = model_factor(hx0, hy0, beta, radius=radius)
        rows.append(dict(t_start=a, t_end=b, T_obs=b - a, Rbar=Rb, aspect=max(hx0, hy0) / min(hx0, hy0),
                         T_lin=Tlin, factor=f, T_pred=f * Tlin, ratio=(b - a) / (f * Tlin),
                         ratio_linear=(b - a) / Tlin))
    return rows, S


def model_series(S, beta, t_end=None):
    """Filament equation (alpha = 0, |beta|) started from the smoothed RD ring at its first
    planar moment (first zero crossing of z2c); returns t, hx, hy, h2c on the RD clock."""
    x = C.crossings(S["t"], S["z2c"])
    t0 = x[0]
    hx0, hy0 = np.interp(t0, S["t"], S["hx"]), np.interp(t0, S["t"], S["hy"])
    T = (t_end if t_end is not None else np.nanmax(S["t"])) - t0
    r = F.run(hx0, hy0, 0.0, abs(beta), T, n=48, dt=None, every=T / 600)
    return dict(t=r["t"] + t0, hx=r["hx"], hy=r["hy"], h2c=r["h2c"])


def read_csv(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def barkley_reference(path="period_table_r2.csv"):
    """Mean and sd of the Barkley half-cycle ratio (rigid rotation, b = 0.15-0.17), as in Fig. 5."""
    q = [r["Th"] / r["T2"] for r in read_csv(path)
         if str(r["rigid"]) == "True" and r["a"] == 1.1 and 0.15 <= r["b"] <= 0.17]
    return float(np.mean(q)), float(np.std(q, ddof=1))


def model_series_grown(S, beta, t_end=None):
    """As model_series, but following the measured size change of the RD ring: binormal flow
    is scale invariant, so a ring of radius R(t) runs on the clock s = int dt R0^2 / R(t)^2 and
    its lengths scale with R(t)/R0 (R: smoothed Rm of the RD ring; R0 = R at the start)."""
    t = S["t"]
    x = C.crossings(t, S["z2c"])
    t0 = x[0]
    g = np.isfinite(S["Rm"]) & (t >= t0)
    tt, RR = t[g], S["Rm"][g]
    R0 = RR[0]
    s = np.concatenate([[0], np.cumsum(0.5 * ((R0 / RR[1:]) ** 2 + (R0 / RR[:-1]) ** 2) * np.diff(tt))])
    hx0, hy0 = np.interp(t0, t, S["hx"]), np.interp(t0, t, S["hy"])
    r = F.run(hx0, hy0, 0.0, abs(beta), s[-1], n=48, dt=None, every=s[-1] / 600)
    tm = np.interp(r["t"], s, tt)
    sc = np.interp(tm, tt, RR) / R0
    return dict(t=tm, hx=r["hx"] * sc, hy=r["hy"] * sc, h2c=r["h2c"] * sc)
