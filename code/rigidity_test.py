"""Test of the rigidity-extended local filament law against the ring modes and straight filaments
(revision: referee comments A2, B-M5, C4a).

Linearizing the filament equation of Dierckx et al., PRL 109, 174102 (2012), Eq. (8), about a ring of
radius R = 1/K gives for the mode m (derivation in the SI):
    beta_zeta = beta_c + (m^2 - 1) e2 K^2,    beta_r = d(beta_c K)/dK + m^2 e2 K^2,
    alpha_zeta = alpha_c + (m^2 - 1) e1 K^2,  alpha_r = d(alpha_c K)/dK + m^2 e1 K^2,
where alpha_c(K), beta_c(K) are the coefficients of circular rings, and for a straight filament
    omega / k^2 = gamma2 + e2 k^2,   growth / k^2 = -(gamma1 + e1 k^2).
A single (e1, e2) must describe every mode. For each ring mode we infer e2 three ways (from beta_zeta,
from beta_r, and from the frequency alone, which fixes the product beta_zeta beta_r), and e1 from the
two diagonal rates; for the straight filaments we fit omega/k^2 against k^2.

    python rigidity_test.py        (in BARKLEY, after mode_refit.py and mode_definitions.py; writes rigidity_test.json)
"""
import json
import numpy as np


def e2_from_product(m, K, bz, br, bc, br_loc):
    """Solve (bc + (m^2-1) x)(br_loc + m^2 x) = bz br for x = e2 K^2 (root nearest zero)."""
    a2, a1, a0 = (m * m - 1) * m * m, (m * m - 1) * br_loc + m * m * bc, bc * br_loc - bz * br
    roots = np.roots([a2, a1, a0]).real
    x = roots[np.argmin(np.abs(roots))]
    return x / K ** 2, bc + (m * m - 1) * x, br_loc + m * m * x


def m4_row():
    """The m = 4 mode at R* (standard filament, time fit of mode_definitions.py) with the curvature-only
    coefficients of mode_refit.local_law at its radius."""
    import os
    import folders as FD
    import halfcycle_uniform as HU
    from mode_refit import local_law, rigidity
    r = [x for x in json.load(open(os.path.join(FD.REVISION, "mode_definitions.json")))["rstar"]
         if x["m"] == 4 and "standard" in x["definition"]][0]
    row = dict(run="R* m=4", m=4, R=r["R"], k=4 / r["R"], bz=r["bz"], br=r["br"], alpha_r=r["alpha_r"], alpha_zeta=r["alpha_zeta"])
    law = local_law(HU.barkley_calibration(0.17), r["R"], 4)
    row.update(law)
    row.update(rigidity(row, law, 4, r["R"]))
    return row


def main():
    R = json.load(open("mode_refit.json"))
    rows = []
    # R*: the time fits of Table S5 (mode_refit "rstar"); larger ring: the fits in tau over the whole run
    sel = [dict(r, run=f"R* amp {r['amp']}") for r in R["rstar"]] + [m4_row()]
    sel += [r for r in R["tau_fits"] if r["run"] == "larger ring" and r["t0"] == 15.0 and r["t1"] in (150.0, 100.0)]
    print("ring modes (R*: fits in t; larger ring: fits in tau; circular-ring coefficients beta_c, alpha_c at the same R):")
    for r in sel:
        m, K = r["m"], 1 / r["R"]
        e2p, bz_p, br_p = e2_from_product(m, K, r["bz"], r["br"], r["beta_c"], r["beta_r_loc"])
        row = dict(run=r["run"], m=m, R=r["R"], k=r["k"], e2_zeta=r["e2_from_zeta"], e2_r=r["e2_from_r"], e2_freq=float(e2p),
                   bz_pred_freq=float(bz_p / r["beta_c"]), br_pred_freq=float(br_p / r["beta_c"]),
                   bz=r["bz"] / r["beta_c"], br=r["br"] / r["beta_c"], e1_zeta=r["e1_from_zeta"], e1_r=r["e1_from_r"])
        rows.append(row)
        print(f"  {r['run']:14s} m={m} R={r['R']:.2f} k={r['k']:.3f}:  e2 from beta_zeta {row['e2_zeta']:+.2f}, from beta_r {row['e2_r']:+.2f},"
              f" from frequency {row['e2_freq']:+.2f} (would give beta_zeta/beta_c {row['bz_pred_freq']:.2f}, beta_r/beta_c {row['br_pred_freq']:.2f};"
              f" measured {row['bz']:.2f}, {row['br']:.2f});  e1 from alpha_zeta {row['e1_zeta']:+.2f}, from alpha_r {row['e1_r']:+.2f}")
    # straight filaments
    sl = np.load("sf2/sf2_linear.npy")          # k, gamma/k^2, err, |beta|, err, ...
    k, wk, werr, gk = sl[:, 0], sl[:, 3], sl[:, 4], sl[:, 1]
    p1 = np.polyfit(k ** 2, wk, 1)
    p2 = np.polyfit(k ** 2, wk, 2)
    res1 = wk - np.polyval(p1, k ** 2)
    print("\nstraight filaments: omega/k^2 =", np.round(wk, 3).tolist(), "at k =", np.round(k, 3).tolist())
    print(f"  fit gamma2 + e2 k^2:          gamma2 = {p1[1]:.2f}, e2 = {p1[0]:+.2f}; residuals {np.round(res1, 3).tolist()} "
          f"(measurement errors {np.round(werr, 3).tolist()})")
    print(f"  fit gamma2 + e2 k^2 + e3 k^4: gamma2 = {p2[2]:.2f}, e2 = {p2[1]:+.2f}, e3 = {p2[0]:+.2f}")
    loc = [(ki, (wi - p1[1]) / ki ** 2) for ki, wi in zip(k, wk)]
    q1 = np.polyfit(k ** 2, -gk, 1)
    print(f"  growth/k^2 = -(gamma1 + e1 k^2): gamma1 = {-q1[1]:+.3f}, e1 = {-q1[0]:+.3f}")
    out = dict(rings=rows, straight=dict(k=k.tolist(), omega_over_k2=wk.tolist(), fit_linear=p1.tolist(),
                                          fit_quadratic=p2.tolist(), residuals_linear=res1.tolist(),
                                          growth_fit=q1.tolist()))
    json.dump(out, open("rigidity_test.json", "w"), indent=1)
    print("wrote rigidity_test.json")


if __name__ == "__main__":
    main()
