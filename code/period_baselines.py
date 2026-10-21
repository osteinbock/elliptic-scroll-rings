"""Barkley half-cycles of Fig. 5 with both references for the drift coefficient (revision: A3, B-M1, C2).

Reads period_table_r2.csv (half-cycles between sign changes of r_2, Rbar = L / 2 pi, the ring's own
alpha and beta during the half-cycle, finite-amplitude factor f in the L / 2 pi convention) and adds,
from the circular rings of the same b (halfcycle_uniform.barkley_calibration):
  beta_c, alpha_c   circular-ring coefficients at Rbar,
  T_pred_c, ratio_c the prediction of Eqs. (2) with beta_c, alpha_c and the same f, and T_obs / T_pred_c,
  own_over_c        own drift coefficient / beta_c,
  eq5_factor        sqrt(beta_c / beta_r), the lengthening that the curvature-only law gives (Eq. 5),
  residual          ratio_c / eq5_factor.
Writes period_table_baselines.csv and prints per-run summaries.

    python period_baselines.py            (in BARKLEY; REVISION as in folders.py)
"""
import csv
import numpy as np

import halfcycle_uniform as HU

KEYS = ["run", "a", "b", "rigid", "t0", "t1", "Th", "R", "beta", "alpha", "eps", "T2", "ratio_own", "beta_c",
        "alpha_c", "own_over_c", "T_pred_c", "ratio_c", "eq5_factor", "residual", "calib_inside"]


def main():
    rows = list(csv.DictReader(open("period_table_r2.csv")))
    fac = np.load("amp_factor.npy")
    cal = {}
    out = []
    for r in rows:
        for k in ("a", "b", "t0", "t1", "Th", "R", "beta", "alpha", "eps", "T0", "T1", "T2"):
            r[k] = float(r[k])
        r["ratio_own"] = r["Th"] / r["T2"]
        b = r["b"]
        if r["a"] == 1.1 and b not in cal:
            cal[b] = HU.barkley_calibration(b)
        c = cal.get(b) if r["a"] == 1.1 else None
        if c is not None:
            R = r["R"]
            bc, ac = float(c.beta(R)), float(c.alpha(R))
            f = np.interp(r["eps"], fac[:, 0], fac[:, 1])
            Tc = 2 * np.pi * R ** 2 / np.sqrt(48 * bc ** 2 - ac ** 2) * f
            loc = float(c.beta_r_over_beta(R))
            r.update(beta_c=bc, alpha_c=ac, own_over_c=abs(r["beta"]) / bc, T_pred_c=Tc, ratio_c=r["Th"] / Tc,
                     eq5_factor=1 / np.sqrt(loc), residual=r["Th"] / Tc * np.sqrt(loc), calib_inside=float(c.inside(R)))
        out.append(r)
    with open("period_table_baselines.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in KEYS})
    for b, c in cal.items():
        print(f"b={b}: " + ("no calibration" if c is None else f"{len(c.P)} windows, K {c.Krange[0]:.3f}-{c.Krange[1]:.3f}"))
    for run in dict.fromkeys(r["run"] for r in out):
        rr = [r for r in out if r["run"] == run]
        f = lambda k: np.round([r.get(k, np.nan) for r in rr], 3).tolist()
        print(f"{run:24s} R {f('R')}\n{'':24s} own {f('ratio_own')}  circle {f('ratio_c')}  own/c {f('own_over_c')}"
              f"  Eq.5 {f('eq5_factor')}  residual {f('residual')}  inside {f('calib_inside')}")
    rig = [r for r in out if r["rigid"] == "True" and r["a"] == 1.1 and 0.15 <= r["b"] <= 0.17 and "ratio_c" in r]
    if rig:
        q = np.array([r["ratio_c"] for r in rig])
        e = np.array([r["eq5_factor"] for r in rig])
        print(f"\nrigid b = 0.15-0.17 with a calibration: n = {len(q)}, circle ratio {q.mean():.3f} +- {q.std(ddof=1):.3f} "
              f"(range {q.min():.2f}-{q.max():.2f}); Eq. 5 factor {e.mean():.3f} ({e.min():.3f}-{e.max():.3f}); "
              f"residual {np.mean(q / e):.3f} ({np.min(q / e):.3f}-{np.max(q / e):.3f})")
    print("wrote period_table_baselines.csv")


if __name__ == "__main__":
    main()
