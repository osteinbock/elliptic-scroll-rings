"""Revision analyses of the ring shape modes (referee comments A2, B-M4/M5, C4c, C5).

1. Diagonal rates. The damped-sinusoid fit (mode_theory2.fit_mode) of r_m and zeta_m gives the decay
   rate g, the frequency omega and the phase offset d from quadrature. With A = alpha_r (m^2-1)/R^2 and
   D = alpha_zeta m^2/R^2 (Eq. 4), g = (A + D)/2 and tan d = (D - A)/(2 omega) (SI Sec. I F), so
       alpha_r = (g - omega tan d) R^2/(m^2 - 1),   alpha_zeta = (g + omega tan d) R^2/m^2.
   These are compared with the curvature-only local law, alpha_r = d(alpha K)/dK and alpha_zeta = alpha
   (circular-ring calibration, halfcycle_uniform.barkley_calibration), and with its extension by the
   rigidity terms of Dierckx et al. (2012), for which
       alpha_zeta - alpha_c = (m^2-1) e1 K^2,   alpha_r - d(alpha_c K)/dK = m^2 e1 K^2,
       beta_zeta - beta_c = (m^2-1) e2 K^2,     beta_r - d(beta_c K)/dK = m^2 e2 K^2.
2. Uncertainty of the couplings at R*: the full range over seed amplitude, fit window and smoothing
   (mode_decompose.analyze, as mode_sweep.npy) and the dx = 0.15 runs (v5r).
3. Larger ring (R about 7, shrinking): fits in the clock tau = int dt / R^2, in which Eq. (4) has
   constant coefficients while R changes; compared with the constant-frequency windows of
   new_tests_analysis.py.

    python mode_refit.py            (run in BARKLEY; also reads REFINE and REVISION, see folders.py; writes mode_refit.json)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ellipse_compare as C
import mode_decompose as MD
import halfcycle_uniform as HU
import folders as FD
from mode_theory2 import fit_mode

PERIOD = 4.44


def diag_rates(m, R, omega, gamma, dphase):
    """alpha_r, alpha_zeta from a fit in time t (R: mean radius)."""
    tq = np.tan(dphase)
    return (gamma - omega * tq) * R ** 2 / (m * m - 1), (gamma + omega * tq) * R ** 2 / (m * m)


def tau_fit(t, r, z, Rs, zz, m, t0, t1):
    """Fit r_m, zeta_m in tau = int dt / R^2 over t0 <= t <= t1. Returns couplings, diagonal rates,
    the own drift coefficient (dz/ds, s = int dt / R) and the mean radius and wavenumber."""
    sel = (t >= t0) & (t <= t1) & np.isfinite(r) & np.isfinite(z) & np.isfinite(Rs) & np.isfinite(zz)
    tt, RR = t[sel], Rs[sel]
    tau = np.concatenate([[0], np.cumsum(0.5 * (1 / RR[1:] ** 2 + 1 / RR[:-1] ** 2) * np.diff(tt))])
    s = np.concatenate([[0], np.cumsum(0.5 * (1 / RR[1:] + 1 / RR[:-1]) * np.diff(tt))])
    f = fit_mode(tau, r[sel], z[sel], m * np.sqrt(m * m - 1) * 1.2)
    W = f["omega"] / (m * np.sqrt(m * m - 1))
    Q = f["ratio"] / np.sqrt((m * m - 1) / m ** 2)
    tq = np.tan(f["dphase"])
    ar = (f["gamma"] - f["omega"] * tq) / (m * m - 1)
    az = (f["gamma"] + f["omega"] * tq) / (m * m)
    bd = abs(np.polyfit(s, zz[sel], 1)[0])
    Rb = RR.mean()
    return dict(m=m, t0=float(t0), t1=float(t1), R=float(Rb), R_start=float(RR[0]), R_end=float(RR[-1]),
                k=float(m / Rb), W=float(W), Q=float(Q), bz=float(W / Q), br=float(W * Q), bdrift=float(bd),
                alpha_r=float(ar), alpha_zeta=float(az), dphase_deg=float(np.degrees(f["dphase"])),
                ok=bool(f["success"] and not f["at_bound"]), rms=float(f["rms"]))


def signals(path, m, smooth=1):
    d = np.load(path)
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    S = {k: C.smooth(t, d[k][ok], smooth * PERIOD) for k in (f"r{m}c", f"z{m}c", "Rm", "z")}
    return t, S


def local_law(cal, R, m):
    """Curvature-only predictions (normalized by the circular beta_c at R) and the circular values."""
    K = 1 / R
    pa, pb = cal.pa, cal.pb
    ac, bc = np.polyval(pa, K), np.polyval(pb, K)
    dac = ac + K * np.polyval(np.polyder(pa), K)          # d(alpha K)/dK
    dbc = bc + K * np.polyval(np.polyder(pb), K)          # d(beta K)/dK
    return dict(alpha_c=float(ac), beta_c=float(bc), alpha_r_loc=float(dac), alpha_zeta_loc=float(ac),
                beta_r_loc=float(dbc), beta_zeta_loc=float(bc))


def rigidity(row, law, m, R):
    K2 = 1 / R ** 2
    return dict(e1_from_zeta=(row["alpha_zeta"] - law["alpha_c"]) / ((m * m - 1) * K2),
                e1_from_r=(row["alpha_r"] - law["alpha_r_loc"]) / (m * m * K2),
                e2_from_zeta=(row["bz"] - law["beta_c"]) / ((m * m - 1) * K2),
                e2_from_r=(row["br"] - law["beta_r_loc"]) / (m * m * K2))


def main():
    cal = HU.barkley_calibration(0.17)
    out = {"calibration": dict(K_range=list(cal.Krange), deg=cal.deg, windows=len(cal.P))}
    # ---------------------------------------------------------------- 1. R* modes (time fits, as in the paper)
    rstar = []
    for m in (2, 3):
        for amp in (0.08, 0.15):
            r = MD.analyze(m, amp, MD.T1[m])
            ar, az = diag_rates(m, r["R"], r["omega"], r["gamma"], np.radians(r["dphase"]))
            row = dict(m=m, amp=amp, R=float(r["R"]), k=float(m / r["R"]), omega=float(r["omega"]), gamma=float(r["gamma"]),
                       dphase_deg=float(r["dphase"]), bz=float(r["bz"]), br=float(r["br"]), bdrift=float(r["bdrift"]),
                       alpha_r=float(ar), alpha_zeta=float(az))
            law = local_law(cal, r["R"], m)
            row.update(law)
            row.update(rigidity(row, law, m, r["R"]))
            # growth rate that the curvature-only law predicts: -(A + D)/2
            row["growth_loc"] = float(-(law["alpha_r_loc"] * (m * m - 1) + law["alpha_zeta_loc"] * m * m) / (2 * r["R"] ** 2))
            rstar.append(row)
    out["rstar"] = rstar
    print("R* modes (time fits):")
    for r in rstar:
        print(f"  m={r['m']} amp={r['amp']}: R={r['R']:.3f} k={r['k']:.3f} decay {r['gamma']:+.4f} (local law: growth "
              f"{r['growth_loc']:+.4f})  phase {r['dphase_deg']:+.1f} deg  alpha_r={r['alpha_r']:+.3f} (loc {r['alpha_r_loc']:+.3f})"
              f"  alpha_zeta={r['alpha_zeta']:+.3f} (loc {r['alpha_zeta_loc']:+.3f})  e1 from zeta/r {r['e1_from_zeta']:+.2f}/{r['e1_from_r']:+.2f}"
              f"  e2 from zeta/r {r['e2_from_zeta']:+.2f}/{r['e2_from_r']:+.2f}")
    # ---------------------------------------------------------------- 2. full ranges at R*
    ranges = {}
    ref = json.load(open(os.path.join(FD.REFINE, "refine_compare.json")))["modes"]
    for m in (2, 3):
        vals = []
        for amp in (0.08, 0.15):
            for fr in (0.6, 0.8, 1.0):
                for sm in (1, 2):
                    r = MD.analyze(m, amp, MD.T1[m] * fr, sm)
                    if r["ok"]:
                        vals.append((r["bz"] / r["bdrift"], r["br"] / r["bdrift"]))
        vals = np.array(vals)
        fine = ref[f"m{m}"]["dx0.15"]
        lo_z, hi_z = min(vals[:, 0].min(), fine["bz_over_drift_range"][0]), max(vals[:, 0].max(), fine["bz_over_drift_range"][1])
        lo_r, hi_r = min(vals[:, 1].min(), fine["br_over_drift_range"][0]), max(vals[:, 1].max(), fine["br_over_drift_range"][1])
        main_z = np.mean([r["bz"] / r["bdrift"] for r in rstar if r["m"] == m])
        main_r = np.mean([r["br"] / r["bdrift"] for r in rstar if r["m"] == m])
        ranges[m] = dict(bz_main=float(main_z), bz_range=[float(lo_z), float(hi_z)], br_main=float(main_r),
                         br_range=[float(lo_r), float(hi_r)], n_fits=len(vals))
        print(f"m={m}: beta_zeta/beta_drift {main_z:.3f} [{lo_z:.3f}, {hi_z:.3f}]   beta_r/beta_drift {main_r:.3f} [{lo_r:.3f}, {hi_r:.3f}]"
              f"   ({len(vals)} dx = 0.2 fits + dx = 0.15 range)")
    out["rstar_ranges"] = {str(k): v for k, v in ranges.items()}
    # ---------------------------------------------------------------- 3. tau fits: R* and larger ring
    taufits = []
    for m, amp in ((2, 0.08), (2, 0.15), (3, 0.08), (3, 0.15)):
        t, S = signals(f"modeR_m{m}_a{amp}.npz", m)
        row = tau_fit(t, S[f"r{m}c"], S[f"z{m}c"], S["Rm"], S["z"], m, 6.0, MD.T1[m])
        row.update(run=f"R* amp {amp}")
        taufits.append(row)
    for m, f, t1 in ((2, "modeL_m2.npz", 150.0), (3, "modeL_m3.npz", 100.0)):
        t, S = signals(f, m)
        for t0, t1w in ((15.0, t1), (15.0, 0.5 * (15 + t1)), (0.5 * (15 + t1), t1)):
            row = tau_fit(t, S[f"r{m}c"], S[f"z{m}c"], S["Rm"], S["z"], m, t0, t1w)
            row.update(run="larger ring")
            taufits.append(row)
    for r in taufits:
        law = local_law(cal, r["R"], r["m"])
        r.update({f"{k}": v for k, v in law.items()})
        r.update(rigidity(r, law, r["m"], r["R"]))
    out["tau_fits"] = taufits
    print("\nfits in tau = int dt / R^2:")
    for r in taufits:
        print(f"  {r['run']:14s} m={r['m']} t {r['t0']:5.1f}-{r['t1']:5.1f}: R {r['R_start']:.2f}->{r['R_end']:.2f} k={r['k']:.3f}"
              f"  bz/bd={r['bz'] / r['bdrift']:.3f} br/bd={r['br'] / r['bdrift']:.3f} sqrt={r['W'] / r['bdrift']:.3f}"
              f"  bz/bc={r['bz'] / r['beta_c']:.3f} br/bc={r['br'] / r['beta_c']:.3f} (Eq.5 {r['beta_r_loc'] / r['beta_c']:.3f})"
              f"  alpha_r={r['alpha_r']:+.3f} (loc {r['alpha_r_loc']:+.3f}) alpha_zeta={r['alpha_zeta']:+.3f} (loc {r['alpha_zeta_loc']:+.3f})"
              f"  phase {r['dphase_deg']:+.1f}  ok={r['ok']}")
    json.dump(out, open("mode_refit.json", "w"), indent=1)
    print("wrote mode_refit.json")


if __name__ == "__main__":
    main()
