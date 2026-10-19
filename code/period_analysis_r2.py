"""Axis-switching half-periods from the Fourier amplitude r_2 instead of the bounding box.

Switches are the zero crossings of r_2c (the cos 2 theta part of the radius, which uses all
filament points; the seeds are aligned with x, so r_2c > 0 <=> long axis along x). The amplitude
eps = max |r_2c| / R. Everything else (R, alpha, beta, predictions, filters) as in
period_analysis.py. Prints both methods side by side and writes period_table_r2.csv.

    python period_analysis_r2.py
"""
import os
import numpy as np
from scipy.ndimage import median_filter
import ellipse_compare as C
import period_analysis as PA

SRC = {"ellP_a1.1_b0.15_12x7.5.npz": "rerun_r2/ellP_a1.1_b0.15_12x7.5.npz",
       "ell_a1.1_b0.17_8x5.npz": "rerun_r2/ell_a1.1_b0.17_8x5.npz",
       "ell_a1.1_b0.17_12x7.5.npz": "rerun_r2/ell_a1.1_b0.17_12x7.5.npz",
       "ellP_a0.8_b0.05_12x7.5.npz": "rerun_r2/ellP_a0.8_b0.05_12x7.5.npz",
       "ell_b0.09_8x5.npz": "rerun_r2/ell_b0.09_8x5.npz"}


def intervals_r2(path, period):
    d = np.load(path)
    ok = np.isfinite(d["r2c"]) & np.isfinite(d["length"])
    t = d["t"][ok]
    good = PA.single_closed(d)[ok]
    r2 = np.where(good, d["r2c"][ok], np.nan)
    r2 = np.where(np.abs(r2 - median_filter(np.nan_to_num(r2, nan=np.nanmedian(r2)), 9, mode="nearest")) > 0.5, np.nan, r2)   # wrong-curve frames
    r2, L, z = (C.smooth(t, y, period) for y in (r2, np.where(good, d["length"][ok], np.nan), np.where(good, d["z"][ok], np.nan)))
    R = L / (2 * np.pi)
    x = C.crossings(t, r2)
    rows = []
    for t0, t1 in zip(x[:-1], x[1:]):
        m = (t >= t0) & (t <= t1) & np.isfinite(R) & np.isfinite(z)
        if m.sum() < 12 or t1 - t0 < 2 * period:
            continue
        Rb = R[m].mean()
        s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[m][1:] + 1 / R[m][:-1]) * np.diff(t[m]))])
        beta = np.polyfit(s, z[m], 1)[0]
        alpha = -0.5 * np.polyfit(t[m], R[m] ** 2, 1)[0]
        eps = np.max(np.abs(r2[m]) / R[m])
        nc = d["ncurves"][ok][m]
        if Rb < 3.2 or eps < 0.04 or eps > PA.EPS_MAX or np.mean(nc > 1) > 0.2:
            continue
        rows.append(dict(t0=t0, t1=t1, Th=t1 - t0, R=Rb, beta=beta, alpha=alpha, eps=eps))
    return rows


def stats(table, sel):
    v = np.array([r["Th"] / r["T2"] for r in table if sel(r)])
    return v


if __name__ == "__main__":
    fac = PA.amplitude_factor()
    out = {"hx": [], "r2": []}
    for path, a, b, lab, P, rigid in PA.RUNS:
        for meth, fn, src in (("hx", PA.intervals, path), ("r2", intervals_r2, SRC.get(path, path))):
            for r in fn(src, P):
                T0, T1, T2 = PA.predict(r, fac)
                out[meth].append(dict(run=lab, a=a, b=b, rigid=rigid, **r, T0=T0, T1=T1, T2=T2))
    for lab in [r[3] for r in PA.RUNS]:
        A = [r for r in out["hx"] if r["run"] == lab]
        B = [r for r in out["r2"] if r["run"] == lab]
        print(f"{lab:24s} hx: n={len(A):2d} ratios {np.round([r['Th']/r['T2'] for r in A], 2).tolist()}")
        print(f"{'':24s} r2: n={len(B):2d} ratios {np.round([r['Th']/r['T2'] for r in B], 2).tolist()}")
        if A and B:
            sw_a = np.array([r["t0"] for r in A] + [A[-1]["t1"]])
            sw_b = np.array([r["t0"] for r in B] + [B[-1]["t1"]])
            dd = [np.min(np.abs(sw_b - s)) for s in sw_a]
            print(f"{'':24s} switch times differ by median {np.median(dd):.2f}, max {np.max(dd):.2f}")
    for meth in ("hx", "r2"):
        tab = out[meth]
        rig = stats(tab, lambda r: r["rigid"] and r["a"] == 1.1 and 0.15 <= r["b"] <= 0.17)
        mea = stats(tab, lambda r: not r["rigid"])
        b18 = stats(tab, lambda r: r["b"] == 0.18)
        allp = len(tab)
        print(f"[{meth}] {allp} half-cycles; rigid b=0.15-0.17: n={len(rig)} mean {rig.mean():.3f} sd {rig.std(ddof=1):.3f} "
              f"range {rig.min():.2f}-{rig.max():.2f}; b=0.18: {np.round(b18, 2).tolist()}; meander: n={len(mea)} "
              f"mean {mea.mean():.3f} sd {mea.std(ddof=1):.3f}")
    keys = ["run", "a", "b", "rigid", "t0", "t1", "Th", "R", "beta", "alpha", "eps", "T0", "T1", "T2"]
    with open("period_table_r2.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in out["r2"]:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print("wrote period_table_r2.csv")
