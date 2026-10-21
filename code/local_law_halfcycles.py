"""Half-cycle by half-cycle test of the curvature-only local law at finite amplitude (revision:
referee comment C1(iii)).

For each planar-to-planar half-cycle of a reaction-diffusion ring (zero crossings of z2c), the filament
equation is started from a planar ellipse with the half-widths measured at the start of the half-cycle
and run to its own next planar moment. Each point moves with alpha K N + beta(K) K B, with beta(K)
from the circular rings (halfcycle_uniform.barkley_calibration; continued as a constant ("clamp") or
linearly ("linear") outside the calibrated curvature range) and with the uniform tension alpha equal
to the shrinkage -1/2 d(Rbar^2)/dt of the reaction-diffusion ring during that half-cycle, so that both
rings shrink alike. The reference "const" uses beta = beta_c(Rbar) constant along the ring (the
lowest-order law with the circular coefficient). A local alpha(K) is not used: the curvature-only
law with alpha(K) < 0 at the strongly curved ends is ill-posed (backward curve shortening).

    python local_law_halfcycles.py OUT.csv          (in BARKLEY; runs listed in RUNS; folders as in folders.py)
"""
import sys
import csv
import numpy as np

import ellipse_compare as C
import filament_flow as F
import halfcycle as H
import halfcycle_uniform as HU

RUNS = [("rerun_r2/ell_a1.1_b0.17_8x5.npz", 0.17, 4.44, "b=0.17 small (8x5)"),
        ("rerun_r2/ell_a1.1_b0.17_12x7.5.npz", 0.17, 4.44, "b=0.17 large (12x7.5)"),
        ("modeR_m2_a0.15.npz", 0.17, 4.44, "b=0.17 m=2 mode, amp 0.15"),
        ("modeR_m2_a0.08.npz", 0.17, 4.44, "b=0.17 m=2 mode, amp 0.08")]


def beta_law(cal, mode):
    K0, K1 = cal.Krange
    p, dp = cal.pb, np.polyder(cal.pb)

    def b(K):
        Kc = np.clip(K, K0, K1)
        v = np.polyval(p, Kc)
        return v + np.polyval(dp, Kc) * (K - Kc) if mode == "linear" else v
    return b


def model_halfcycle(hx0, hy0, alpha, law, T_guess, n=64, dt=0.004):
    """First planar-to-planar interval of the filament equation from a planar ellipse."""
    r = F.run(hx0, hy0, 0.0, -1.0, 2.0 * T_guess, n=n, dt=dt, every=0.02,
              local=lambda K: (np.full_like(K, alpha), -law(K)))
    x = [c for c in C.crossings(r["t"], r["z2c"]) if c > 0.3 * T_guess]
    return x[0] if x else np.nan, float(np.max(np.abs(r["z2c"]))), r["length"][0] / (2 * np.pi)


def main(out):
    cal = HU.barkley_calibration(0.17)
    rows = []
    for path, b, per, lab in RUNS:
        S = H.smoothed(path, per)
        t = S["t"]
        x = C.crossings(t, S["z2c"])
        for a, e in zip(x[:-1], x[1:]):
            gap = C.max_gap(S["t_valid"], a, e)
            hx0, hy0 = np.interp(a, t, S["hx"]), np.interp(a, t, S["hy"])
            if gap > per / 2 or e - a < 2 * per or max(hx0, hy0) / min(hx0, hy0) < HU.ASPECT_MIN:
                continue
            m = (t >= a) & (t <= e) & np.isfinite(S["RL"])
            RL = S["RL"][m].mean()
            alpha = max(0.0, -0.5 * np.polyfit(t[m], S["RL"][m] ** 2, 1)[0])
            bc = float(cal.beta(RL))
            T_guess = np.pi * RL ** 2 / (2 * np.sqrt(3) * bc)
            row = dict(run=lab, t_start=a, t_end=e, T_obs=e - a, RL=RL, aspect=max(hx0, hy0) / min(hx0, hy0), alpha=alpha, beta_c=bc)
            for mode in ("clamp", "linear"):
                Tm, zmax, R0 = model_halfcycle(hx0, hy0, alpha, beta_law(cal, mode), T_guess)
                row[f"T_{mode}"], row[f"ratio_{mode}"] = Tm, (e - a) / Tm
            Tc, _, _ = model_halfcycle(hx0, hy0, alpha, lambda K: np.full_like(K, bc), T_guess)
            row["T_const"], row["ratio_const"] = Tc, (e - a) / Tc
            row["local_over_const"] = Tc / row["T_clamp"]
            rows.append(row)
            print(f"{lab:28s} t {a:6.1f}-{e:6.1f} T_obs {e - a:5.2f}  R {RL:.2f} aspect {row['aspect']:.2f}:"
                  f"  RD/const {row['ratio_const']:.3f}  RD/local(clamp) {row['ratio_clamp']:.3f}  RD/local(linear) {row['ratio_linear']:.3f}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local_law_halfcycles.csv")
