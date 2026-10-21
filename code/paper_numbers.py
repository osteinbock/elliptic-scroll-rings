"""Every number quoted in the manuscript and SI that follows from the simulations, recomputed
from the production data and analysis products; written to <BARKLEY>/paper_numbers.json.
    python paper_numbers.py SUPP BARKLEY [REFINE [REVISION]]   (after build_all.sh has produced the tables)
The four folders are also passed to folders.py (environment variables SUPP, BARKLEY, REFINE, REVISION).
"""
import os
import sys
import csv
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ellipse_compare as C          # noqa: E402
import filament_flow as F            # noqa: E402
import halfcycle as H                # noqa: E402

S, B = (os.path.abspath(p) for p in sys.argv[1:3])
RF = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
RV = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else None
for _name, _path in (("SUPP", S), ("BARKLEY", B), ("REFINE", RF), ("REVISION", RV)):
    if _path:
        os.environ[_name] = _path
OUT = {}


def put(key, value, note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    OUT[key] = value
    print(f"{key:58s} {value!r:>24}  {note}")


def single(d):
    g = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
    return g & (d["closed"] == 1) if "closed" in d.files else g


def smoothed(path, P, keys):
    d = np.load(path)
    ok = single(d)
    t = d["t"][ok]
    return t, {k: C.smooth(t, d[k][ok], P) for k in keys}, d


def section(name, fn):
    print(f"--- {name}")
    try:
        fn()
    except Exception as e:                      # keep going, but make the failure visible
        put(f"ERROR {name}", f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- Barkley
def methods():
    import glob
    fr = []
    for f in sorted(glob.glob(os.path.join(B, "*.npz")) + glob.glob(os.path.join(B, "*", "*.npz"))):
        d = np.load(f)
        if "closed" in d.files and "raw_Rm" in d.files:
            ok = np.isfinite(d["raw_Rm"]) & (d["t"] > 10)
            if ok.any():
                fr.append(single(d)[ok].mean())
    put("barkley.single_closed_fraction.min", round(min(fr), 4))
    rows = list(csv.DictReader(open(os.path.join(B, "period_table_r2.csv"))))
    import period_analysis as PA
    import period_analysis_r2 as P2
    cl, gaps = [], []
    for r in rows:
        f = [x for x in PA.RUNS if x[3] == r["run"]][0][0]
        d = np.load(os.path.join(B, P2.SRC.get(f, f)))
        k = np.isfinite(d["hx"]) & (d["t"] >= float(r["t0"])) & (d["t"] <= float(r["t1"]))
        cl.append(float(d["Lxy"]) - 2 * np.maximum(d["hx"][k], d["hy"][k]).max())
        gaps.append(float(r["max_gap"]))
    put("barkley.analysed.min_image_clearance", round(min(cl), 2))
    put("barkley.analysed.max_gap", round(max(gaps), 2))


def radii():
    import make_figs as MF
    cwd = os.getcwd(); os.chdir(B)
    try:
        circ, mode = [], []
        for f in MF.VALID_CIRCLES + MF.LARGE_CIRCLES:
            t, s, _ = smoothed(f, 4.44, ("Rm", "length"))
            g = np.isfinite(s["Rm"]) & np.isfinite(s["length"]) & (t > 6)
            circ.append(np.max(np.abs(s["length"][g] / (2 * np.pi * s["Rm"][g]) - 1)))
        for f, t1 in (("modeR_m2_a0.08.npz", 125), ("modeR_m2_a0.15.npz", 125), ("modeR_m3_a0.08.npz", 45),
                      ("modeR_m3_a0.15.npz", 45), ("modeL_m2.npz", 1e9), ("modeL_m3.npz", 1e9)):
            t, s, _ = smoothed(f, 4.44, ("Rm", "length"))
            g = np.isfinite(s["Rm"]) & np.isfinite(s["length"]) & (t > 6) & (t <= t1)
            mode.append(np.median(s["length"][g] / (2 * np.pi * s["Rm"][g]) - 1))
        put("barkley.circles.L2pi_vs_Rm.max_percent", round(100 * max(circ), 3))
        put("barkley.modes.L2pi_vs_Rm.median_percent.range", [round(100 * min(mode), 2), round(100 * max(mode), 2)])
    finally:
        os.chdir(cwd)


def fig2():
    t, s, d = smoothed(os.path.join(B, "ell_a1.1_b0.17_8x5.npz"), 4.44, ("hx", "hy", "z", "length", "r2c", "z2c"))
    i = np.argmin(abs(t - 5))
    put("fig2.halfwidths_t5", [round(s["hx"][i], 2), round(s["hy"][i], 2)])
    put("fig2.planar_moments", [round(x, 1) for x in C.crossings(t, s["z2c"])[:2]])
    sw = C.crossings(t, s["r2c"])
    put("fig2.switches_within_120", sum(1 for x in sw if x <= 120))
    k = (t > 8) & (t < 40) & np.isfinite(s["z"])
    put("fig2.drift_speed", round(abs(np.polyfit(t[k], s["z"][k], 1)[0]), 3))
    R = s["length"] / (2 * np.pi)
    rv = []
    for a in np.arange(8, 110, 10):
        m = (t >= a) & (t < a + 10) & np.isfinite(s["z"]) & np.isfinite(R)
        rv.append(abs(np.polyfit(t[m], s["z"][m], 1)[0]) * R[m].mean())
    put("fig2.Rv_10unit_windows.first_last", [round(rv[0], 2), round(rv[-1], 2)])
    z2 = np.abs(s["z2c"])
    put("fig2.saddle_amplitude_decrease", round(np.nanmax(z2[(t > 5) & (t < 25)]) / np.nanmax(z2[(t > 95) & (t < 120)]), 2))
    m1 = F.run(6.5, 4.0, 0.0, -1.80, 115, n=64, dt=0.01)
    xm = [x + 5 for x in C.crossings(m1["t"], m1["hx"] - m1["hy"])]
    xs = C.crossings(t, s["hx"] - s["hy"])
    n = min(len(xs), len(xm))
    put("fig2.model_faster_factor", round(np.diff(xs[:n]).mean() / np.diff(xm[:n]).mean(), 3))
    # circles at the same curvature (Fig. 4 windows)
    import alpha_stationarity as A
    import make_figs as MF
    cwd = os.getcwd(); os.chdir(B)
    try:
        P = np.concatenate([w for w in (A.windows(f) for f in MF.VALID_CIRCLES + MF.LARGE_CIRCLES) if len(w)])
    finally:
        os.chdir(cwd)
    bc = []
    for a in (8, 108):
        m = (t >= a) & (t < a + 10) & np.isfinite(R)
        K = 1 / R[m].mean()
        near = np.abs(P[:, 1] - K) < 0.01
        bc.append(np.median(np.abs(P[near, 3])))
    put("fig2.circle_beta_same_size.first_last", [round(bc[0], 2), round(bc[1], 2)])
    put("fig2.slowdown_first_last_percent", [round(100 * (1 - rv[0] / bc[0]), 1), round(100 * (1 - rv[-1] / bc[1]), 1)])


def fig3():
    import make_figs as MF
    cwd = os.getcwd(); os.chdir(B)
    try:
        al, be, hx0, hy0 = MF.fit_shrink_drift("ellP_a1.1_b0.15_12x7.5.npz", 4.13)
        put("fig3.b015.alpha_beta", [round(al, 3), round(abs(be), 3)])
        al5, be5, _, _ = MF.fit_shrink_drift("ellP_a0.8_b0.05_12x7.5.npz", 3.39)
        put("fig3.b005.abs_beta_over_alpha", round(abs(be5) / al5, 3))
        t, s, d = MF.smooth_run("ellP_a1.1_b0.15_12x7.5.npz", 4.13, ("hx", "hy", "length"))
        R = s["length"] / (2 * np.pi)
        xs = [c for c in C.crossings(t, s["hx"] - s["hy"]) if c > 6]
        put("fig3.b015.R_at_t6", round(np.interp(6, t, R), 2))
        put("fig3.b015.last_filament_t", round(float(d["t"][np.isfinite(d["Rm"])][-1]), 1))
        Rx = [np.interp(c, t, R) for c in xs]
        put("fig3.b015.switches_with_R_above_2.9", sum(1 for r in Rx if r >= 2.9))
        put("fig3.b015.first_two_intervals", [round(v, 1) for v in np.diff(np.r_[6, xs])[1:4]])
        R0 = R[np.nanargmin(abs(t - 6))]
        obs = [np.log(R0 / r) for r in Rx if r > 3]
        th = MF.theory_crossings(al, abs(be), 1.4)[:len(obs)]
        put("fig3.b015.max_switch_position_error_lnR", round(max(abs(a - b) for a, b in zip(obs, th)), 3))
        # b = 0.18 ring until it approaches its images
        t, s, d = MF.smooth_run("ellB_a1.1_b0.18_12x7.5.npz", 4.75, ("hx", "hy", "length", "r2c"))
        cl = float(d["Lxy"]) - 2 * np.maximum(s["hx"], s["hy"])
        tc = t[np.flatnonzero(np.isfinite(cl) & (cl < 12))[0]]
        R = s["length"] / (2 * np.pi)
        put("b018.time_clearance_below_12", round(tc, 1))
        g = np.isfinite(R)
        put("b018.R_at_t10_and_at_tc", [round(np.interp(10, t[g], R[g]), 1), round(np.interp(tc, t[g], R[g]), 1)])
        put("b018.switches_before_tc", sum(1 for x in C.crossings(t, s["r2c"]) if x < tc))
        e = np.abs(s["hx"] - s["hy"]) / (s["hx"] + s["hy"])
        put("b018.max_ellipticity_before_tc", round(np.nanmax(e[t < tc]), 2))
    finally:
        os.chdir(cwd)


def fig4():
    import alpha_stationarity as A
    import make_figs as MF
    cwd = os.getcwd(); os.chdir(B)
    try:
        fs = MF.VALID_CIRCLES + MF.LARGE_CIRCLES + ["sf2/circXXL_R0_38_snaps.npz"]
        ws = [A.windows(f) for f in fs]
        put("fig4.rings_with_windows", sum(1 for w in ws if len(w)))
        P = np.concatenate([w for w in ws if len(w)])
        K, al, be = P[:, 1], P[:, 2], np.abs(P[:, 3])
        p = np.polyfit(K, al, 3)
        put("fig4.alpha_zero_K", [round(r.real, 3) for r in np.roots(p) if abs(r.imag) < 1e-9 and 0.1 < r.real < 0.3])
        put("fig4.alpha_beta_at_K_below_0.04", [round(np.median(al[K < 0.04]), 2), round(np.median(be[K < 0.04]), 2)])
        put("fig4.beta_at_K_above_0.24", round(np.median(be[K > 0.24]), 2))
        fin = []
        for f in ("circS_R0_4.5.npz", "circHB_R0_5.0.npz", "circS_R0_5.5.npz", "circL_R0_6.npz", "circS_R0_6.5.npz"):
            t, s, _ = smoothed(f, 4.44, ("Rm",))
            fin.append(s["Rm"][np.isfinite(s["Rm"])][-1])
        put("fig4.final_radii_near_Rstar", [round(min(fin), 2), round(max(fin), 2)])
        t, s, _ = smoothed("ctrl_circ_R0_6.5_dx015.npz", 4.44, ("Rm",))
        g = np.isfinite(s["Rm"])
        put("dx015.ring_R_at_120_and_145", [round(np.interp(120, t[g], s["Rm"][g]), 2), round(np.interp(145, t[g], s["Rm"][g]), 2)])
    finally:
        os.chdir(cwd)


def fig5():
    rows = list(csv.DictReader(open(os.path.join(B, "period_table_r2.csv"))))
    q = lambda sel: np.array([float(r["Th"]) / float(r["T2"]) for r in rows if sel(r)])
    rig = q(lambda r: r["rigid"] == "True" and float(r["a"]) == 1.1 and 0.15 <= float(r["b"]) <= 0.17)
    put("fig5.n_halfcycles", len(rows))
    put("fig5.rigid.mean_sd_n", [round(rig.mean(), 3), round(rig.std(ddof=1), 3), len(rig)])
    put("fig5.rigid.range", [round(rig.min(), 2), round(rig.max(), 2)])
    put("fig5.b018", [round(v, 2) for v in q(lambda r: float(r["b"]) == 0.18)])
    me = q(lambda r: r["rigid"] == "False")
    put("fig5.meander.mean_sd", [round(me.mean(), 2), round(me.std(ddof=1), 2)])
    T0 = np.array([float(r["T0"]) for r in rows]); T1 = np.array([float(r["T1"]) for r in rows])
    T2 = np.array([float(r["T2"]) for r in rows])
    put("fig5.max_tension_correction_percent", round(100 * np.max(np.abs(T1 / T0 - 1)), 2))
    put("fig5.max_amplitude_correction_percent", round(100 * np.max(np.abs(T2 / T1 - 1)), 2))


def modes():
    d = np.load(os.path.join(B, "mode_decompose.npy"))
    for m in (2, 3):
        r = d[d[:, 0] == m]
        put(f"modes.m{m}.beta_zeta_beta_r_beta_drift", [round(r[:, 7].mean(), 2), round(r[:, 8].mean(), 2), round(r[:, 9].mean(), 2)])
        put(f"modes.m{m}.freq_difference_percent", round(100 * abs(r[1, 3] / r[0, 3] - 1), 1))
        put(f"modes.m{m}.decay_rate", round(r[:, 10].mean(), 3))
    sw = np.load(os.path.join(B, "mode_sweep.npy"))
    nt = np.load(os.path.join(B, "new_tests.npz"))
    L = nt["large"]
    qd = lambda a: np.minimum(np.abs(a) % 180, 180 - np.abs(a) % 180)
    put("modes.max_departure_from_quadrature_deg", round(max(qd(sw[sw[:, 7] == 1, 6]).max(), L[:, 11].max()), 1))
    for m in (2, 3):
        s = sw[(sw[:, 0] == m) & (sw[:, 7] == 1)]
        put(f"modes.m{m}.beta_zeta_over_drift.range", [round(s[:, 4].min(), 2), round(s[:, 4].max(), 2)])
        put(f"modes.m{m}.beta_r_over_drift.range", [round(s[:, 5].min(), 2), round(s[:, 5].max(), 2)])
    import make_figs as MF
    cwd = os.getcwd(); os.chdir(B)
    try:
        g = MF._beta_curve_runs()
        put("modes.local_law_beta_r_over_drift_at_Rstar", round(float(MF._local_ratio(g, 1 / 4.87)), 2))
    finally:
        os.chdir(cwd)
    tl = nt["tilt"]
    put("tilt.angle_deg_and_ratio", [round(np.degrees(np.arctan(tl[1] / tl[0])), 1), round(tl[4] / tl[5], 3)])
    for m in (2, 3):
        x = L[L[:, 0] == m]
        put(f"large.m{m}.beta_zeta_over_drift_mean_k", [round(np.mean(x[:, 6] / x[:, 8]), 2), round(np.mean(x[:, 3]), 2)])
    sl = np.load(os.path.join(B, "sf2", "sf2_linear.npy"))
    put("straight.beta_at_k_first_last", [round(sl[0, 3], 2), round(sl[0, 0], 2), round(sl[-1, 3], 2), round(sl[-1, 0], 2)])
    gr = -sl[:, 1] * sl[:, 0] ** 2
    put("straight.growth_rates_k_le_0.6", [round(gr[sl[:, 0] <= 0.61].min(), 3), round(gr[sl[:, 0] <= 0.61].max(), 3)])
    put("straight.growth_rate_k_0.79", round(gr[-1], 4))
    ib = lambda k: np.interp(k, sl[:, 0], sl[:, 3])
    ring = [(np.mean(d[d[:, 0] == m, 5]), m / np.mean(d[d[:, 0] == m, 2])) for m in (2, 3)]
    ring += [(np.mean(L[L[:, 0] == m, 4]), np.mean(L[L[:, 0] == m, 3])) for m in (2, 3)]
    below = [100 * (1 - W / ib(k)) for W, k in ring]
    put("ringmodes_below_straight_percent.range", [round(min(below), 0), round(max(below), 0)])
    bk = np.load(os.path.join(B, "sf2", "buckle.npz"))["fits"]
    put("buckle.max_deviation_from_straight_percent", round(max(100 * abs(bk[:, 6] / ib(bk[:, 2]) - 1)), 1))
    tw = np.load(os.path.join(B, "twist_8x5.npz"))
    put("twist.median_rms_per_length", round(np.median(tw["w_rms"]), 3))
    dr = np.load(os.path.join(B, "diffrot.npz"))
    put("twist.tipphase.spread_max_rad", round(np.max(np.ptp(dr["rel"], 1)), 2))
    put("twist.tipphase.median_gradient", round(np.median(np.abs(dr["grad"])), 3))


# ------------------------------------------------------------------- supplementary models
def supp():
    P = H.read_csv(os.path.join(S, "halfcycles_panfilov.csv"))
    q = np.array([r["ratio"] for r in P])
    put("panfilov.ratio.mean_sd_n", [round(q.mean(), 3), round(q.std(ddof=1), 3), len(q)])
    for fn in dict.fromkeys(r["file"] for r in P):
        x = np.array([r["ratio"] for r in P if r["file"] == fn])
        put(f"tableS2.{fn}", [round(x.mean(), 3), round(x.std(), 3) if len(x) > 1 else None, len(x)])
    A = H.read_csv(os.path.join(S, "halfcycles_atri.csv"))
    a3 = [r for r in A if r["dx"] == 3.0]
    q3 = np.array([r["ratio"] for r in a3])
    put("atri.dx3.ratio.mean_sd_n", [round(q3.mean(), 3), round(q3.std(ddof=1), 3), len(q3)])
    q2 = np.array([r["ratio"] for r in A if r["dx"] == 2.0])
    put("atri.dx2.ratio.mean_sd_n", [round(q2.mean(), 3), round(q2.std(ddof=1), 3), len(q2)])
    put("atri.dx3.Rbar_range", [round(min(r["Rbar"] for r in a3)), round(max(r["Rbar"] for r in a3))])
    put("atri.dx3.aspect_range", [round(min(r["aspect"] for r in a3), 2), round(max(r["aspect"] for r in a3), 2)])
    cl = []
    for r in a3:
        d = np.load(os.path.join(S, r["file"]))
        k = single(d) & (d["t"] >= r["t_start"]) & (d["t"] <= r["t_end"])
        cl.append(float(d["Lxy"]) - 2 * np.maximum(d["hx"][k], d["hy"][k]).max())
    clean = q3[np.array(cl) >= 260]
    put("atri.dx3.min_clearance_um", round(min(cl)))
    put("atri.dx3.clearance_ge_260.mean_sd_n", [round(clean.mean(), 3), round(clean.std(ddof=1), 3), len(clean)])
    put("supp.max_gap.panfilov_atri", [max(r["max_gap"] for r in P), max(r["max_gap"] for r in A)])
    fa = [r["factor"] for r in P + A]
    put("supp.factor.range", [round(min(fa), 3), round(max(fa), 3)])
    for asp in (1.05, 1.5, 2.4):
        put(f"supp.factor.at_aspect_{asp}", round(H.model_factor(asp, 1.0, 1.0), 3))
    put("tableS4.rows", [[r["file"], r["dx"], round(r["t_start"]), round(r["t_end"]), round(r["Rbar"]),
                          round(r["aspect"], 2), round(r["T_obs"]), round(r["ratio"], 3)] for r in A])
    co = H.read_csv(os.path.join(S, "coefficients_circles.csv"))
    put("circles.used_for_tables", [[r["model"], r["file"], round(r["alpha"], 4), round(r["beta"], 4)] for r in co])
    # Table S1 (S10 of the SI): pooled over the seeds 14 and 22 and, for g_f = 0.775 and 0.8, 18 (REVISION)
    import pan_alpha as PA
    for gf in (0.775, 0.8, 0.825, 0.85, 0.9):
        paths = [os.path.join(S, f"pcirc_gf{gf}_R{R}.npz") for R in (14, 22)]
        if RV and gf in (0.775, 0.8):
            paths.append(os.path.join(RV, f"pcirc_gf{gf}_R18.npz"))
        w = np.vstack([PA.windows(p, single=True)[1] for p in paths])
        put(f"tableS1.gf{gf}", [round(w[:, 1].min()), round(w[:, 1].max()), round(w[:, 2].mean(), 3),
                                 round(w[:, 2].std(), 3), round(abs(w[:, 3].mean()), 2), round(w[:, 3].std(), 2)])
    # Table S3 and Atri controls
    import atri_alpha as AA
    for lab, files, per in (("0.085", ["acirc3_P0.085.npz"], 20.0), ("0.095", ["acirc2_R80.npz", "acirc2_R130.npz"], 17.5),
                            ("0.100", ["acirc4_P0.1_c052.npz"], 16.58), ("0.100.dx2", ["acirc_dx2_P0.1.npz"], 16.58),
                            ("0.100.cstar0.566", ["acirc3_P0.1.npz"], 16.58)):
        rows = [AA.fit(os.path.join(S, f), per)[0] for f in files]
        R = np.vstack(rows)
        tc = [AA.fit(os.path.join(S, f), per)[2] for f in files]
        put(f"tableS3.P{lab}", [round(min(x[1].min() for x in tc)), round(max(x[1].max() for x in tc)),
                                round(R[:, 2].mean(), 2), round(R[:, 2].std(), 2), round(abs(R[:, 3].mean()), 2), round(R[:, 3].std(), 2)])
    # P = 0.095 ellipse: first switch, observed vs filament equation following the measured size
    Sx = H.smoothed(os.path.join(S, "aell2_P0.095_130x80.npz"), 17.5)
    t = Sx["t"]; g = np.isfinite(Sx["Rm"])
    obs = C.crossings(t, Sx["hx"] - Sx["hy"])[0]
    w = np.vstack([AA.fit(os.path.join(S, f), 17.5)[0] for f in ("acirc2_R80.npz", "acirc2_R130.npz")])
    t0 = t[g][0]
    k = g & (t >= t0)
    tt, LL = t[k], Sx["length"][k]
    s = np.concatenate([[0], np.cumsum(0.5 * ((LL[0] / LL[1:]) ** 2 + (LL[0] / LL[:-1]) ** 2) * np.diff(tt))])
    r = F.run(Sx["hx"][g][0], Sx["hy"][g][0], 0.0, abs(w[:, 3].mean()), s[-1], n=48, dt=None, every=s[-1] / 2000)
    pred = np.interp(C.crossings(r["t"], r["hx"] - r["hy"])[0], s, tt)
    put("atri.P0.095.first_switch_obs_pred", [round(obs), round(pred)])


def refine():
    import refine_compare as RC
    res = {"m2": None, "m3": None}
    mc, mf = RC.modes(B), RC.modes(RF)
    for m in (2, 3):
        c, f = mc[m], mf[m]
        put(f"refine.m{m}.beta_zeta_change_percent", round(100 * (f["beta_zeta"] / c["beta_zeta"] - 1), 1))
        put(f"refine.m{m}.beta_r_change_percent", round(100 * (f["beta_r"] / c["beta_r"] - 1), 1))
        put(f"refine.m{m}.beta_zeta_over_drift_fine.range", [round(v, 2) for v in f["bz_over_drift_range"]])
        put(f"refine.m{m}.R_fine", round(f["R"], 2))
    sc = RC.straight(os.path.join(B, "sf2", "L32_k035.npz"))
    sf = RC.straight(os.path.join(RF, "sf2", "L32_k035_dx015.npz"))
    put("refine.straight_k0.35.beta_change_percent", round(100 * (sf["beta"] / sc["beta"] - 1), 1))


# ----------------------------------------------------------------------------- revision (referee reports of 20 October 2026)
def _rows(path):
    return H.read_csv(path)


def _ms(v, nd=3):
    v = np.asarray(v, float)
    return [round(v.mean(), nd), round(v.std(ddof=1), nd) if len(v) > 1 else None, len(v)]


def _rng(v, nd=2):
    v = np.asarray(v, float)
    return [round(v.min(), nd), round(v.max(), nd)]


def _uniform():
    return _rows(os.path.join(RV, "uniform_halfcycles.csv"))


def rev_halfcycles():
    """Sec. IV C, the abstract, Sec. V, SI Sec. II: the half-cycle analysis with the circular-ring baseline."""
    import halfcycle_uniform as HU
    U = _uniform()
    r2 = [r for r in U if r["model"] == "barkley" and r["event"] == "r2"]
    put("rev.fig5a.n_halfcycles_and_runs", [len(r2), len({r["run"] for r in r2})])
    b17 = [r for r in r2 if "b=0.17" in r["run"]]
    q = np.array([r["ratio_c"] for r in b17])
    put("rev.b017.ratio_c.mean_sd_n", _ms(q))
    put("rev.b017.ratio_c.range", _rng(q))
    put("rev.b017.ratio_c.percent_longer", round(100 * (q.mean() - 1)))
    put("rev.b017.Rbar.range", _rng([r["Rbar"] for r in b17], 1))
    put("rev.b017.eq5_factor.mean_sd", _ms([r["eq5_factor"] for r in b17])[:2])
    put("rev.b017.residual.mean_sd", _ms([r["residual"] for r in b17])[:2])
    cal = HU.barkley_calibration(0.17)
    put("rev.b017.beta_r_over_beta_at_Rbar.range", _rng([float(cal.ratio_r(r["Rbar"])) for r in b17]))
    put("rev.b017.beta_r_over_beta_from_eq5.range", _rng([1 / r["eq5_factor"] ** 2 for r in b17]))
    put("rev.b017.ratio_own.mean_sd", _ms([r["ratio_own"] for r in b17], 2)[:2])
    put("rev.b017.drift_per_switch_over_Rbar", round(float(np.mean([r["ratio_own"] for r in b17])) * np.pi / (2 * np.sqrt(3)), 2))
    deficit = lambda rows: _rng([100 * (1 - r["own_over_c"]) for r in rows], 0)
    put("rev.own_drift_deficit_percent.b017.range", deficit(b17))
    b1516 = [r for r in r2 if "b=0.15" in r["run"] or "b=0.16" in r["run"]]
    b18 = [r for r in r2 if "b=0.18" in r["run"]]
    put("rev.own_drift_deficit_percent.b015_b016.range", deficit(b1516))
    put("rev.own_drift_deficit_percent.b018", deficit(b18))
    put("rev.b015_b016.ratio_c.range", _rng([r["ratio_c"] for r in b1516]))
    put("rev.b018.ratio_c", [round(r["ratio_c"], 2) for r in b18])
    big = [r for r in b1516 if r["Rbar"] >= 3.6]
    small = [r for r in b1516 if r["Rbar"] < 3.6]
    put("rev.b015_b016.Rbar_ge_3.6.eq5_factor.range", _rng([r["eq5_factor"] for r in big]))
    put("rev.b015_b016.Rbar_ge_3.6.residual.range", _rng([r["residual"] for r in big]))
    put("rev.b015_b016.smallest.residual_and_Rbar", [_rng([r["residual"] for r in small]), _rng([r["Rbar"] for r in small], 1)])
    put("rev.b015_b016.eq5_local_vs_global.max_abs_difference", round(max(abs(r["eq5_factor"] - r["eq5_global"]) for r in b1516), 3))
    put("rev.fig5a.T_obs.range", [round(min(r["T_obs"] for r in r2), 1), round(max(r["T_obs"] for r in r2))])
    put("rev.fig5a.max_shrink_within_halfcycle_percent", round(100 * max(1 - r["R_end_over_start"] for r in r2)))
    put("rev.fig5a.factor.range", _rng([r["factor"] for r in r2]))
    put("rev.fig5a.max_amplitude_correction_percent", round(100 * max(abs(1 - r["factor"]) for r in r2), 1))
    put("rev.fig5a.max_tension_correction_percent",
        round(100 * max(1 - np.sqrt(1 - r["alpha_c"] ** 2 / (48 * r["beta_c"] ** 2)) for r in r2), 3))
    # planar-moment half-cycles (cross-model comparison)
    z2 = lambda model, sel, rad="Rm": [r for r in U if r["model"] == model and r["event"] == "z2" and r["radius"] == rad and sel(r)]
    weak = lambda r: "mode" in r["run"]
    b17z = lambda r: "b=0.17" in r["run"] and "mode" not in r["run"]
    for rad in ("Rm", "L"):
        put(f"rev.b017.all.z2.{rad}.ratio_c.mean_sd_n", _ms([r["ratio_c"] for r in z2("barkley", lambda r: "b=0.17" in r["run"], rad)], 3))
        put(f"rev.b017.weak_modes.z2.{rad}.ratio_c.mean_sd_n", _ms([r["ratio_c"] for r in z2("barkley", weak, rad)], 3))
        put(f"rev.b017.elliptic.z2.{rad}.ratio_c.mean_sd_n", _ms([r["ratio_c"] for r in z2("barkley", b17z, rad)], 3))
    for gf in ("0.775", "0.8"):
        rows = z2("panfilov", lambda r: f"gf={gf} " in r["run"])
        put(f"rev.panfilov.gf{gf}.ratio_c.mean_sd_n_runs", _ms([r["ratio_c"] for r in rows]) + [len({r["run"] for r in rows})])
        put(f"rev.panfilov.gf{gf}.eq5_factor.mean", round(float(np.mean([r["eq5_factor"] for r in rows])), 3))
        put(f"rev.panfilov.gf{gf}.residual.mean", round(float(np.mean([r["residual"] for r in rows])), 3))
    rm, rl = z2("panfilov", lambda r: True), z2("panfilov", lambda r: True, "L")
    put("rev.panfilov.L_vs_Rm_ratio_change_percent.range",
        _rng([100 * (b["ratio_c"] / a["ratio_c"] - 1) for a, b in zip(rm, rl)], 1))
    A = z2("atri", lambda r: True)
    AL = z2("atri", lambda r: True, "L")
    qa = np.array([r["ratio_c"] for r in A])
    put("rev.atri.excess_percent.mean_sd_n_runs", [round(100 * (qa.mean() - 1), 1), round(100 * qa.std(ddof=1), 1), len(qa),
                                                     len({r["run"] for r in A})])
    low = np.array([r["ratio_c"] for r in A if r["aspect"] < 1.3])
    put("rev.atri.aspect_below_1.3.excess_percent.mean_sd_n", [round(100 * (low.mean() - 1), 1), round(100 * low.std(ddof=1), 1), len(low)])
    put("rev.atri.Rbar.range_um", _rng([r["Rbar"] for r in A], 0))
    i0, i1 = int(np.argmin([r["Rbar"] for r in A])), int(np.argmax([r["Rbar"] for r in A]))
    put("rev.atri.ratio_at_smallest_and_largest_Rbar", [round(A[i0]["ratio_c"], 2), round(A[i1]["ratio_c"], 2)])
    dconv = [abs(b["ratio_c"] / a["ratio_c"] - 1) * 100 for a, b in zip(A, AL)]
    put("rev.atri.L_vs_Rm.max_change_percent.eccentric_and_aspect_below_1.3",
        [round(max(d for d, a in zip(dconv, A) if a["aspect"] >= 1.3), 1), round(max(d for d, a in zip(dconv, A) if a["aspect"] < 1.3), 1)])
    put("rev.atri.factor.max_deviation_percent.aspect_below_1.3", round(100 * max(abs(r["factor"] - 1) for r in A if r["aspect"] < 1.3), 1))
    m3 = np.mean([r["ratio_c"] for r in A if r["run"] == "dx=3.0 600^2 x 280 seed 100 x 60"])
    m2 = np.mean([r["ratio_c"] for r in A if r["run"] == "dx=2.0 600^2 x 280 seed 100 x 60"])
    put("rev.atri.dx2_vs_dx3.ratio_change_percent", round(100 * (m2 / m3 - 1), 1))
    # clearance of the Atri half-cycles from their periodic images (as in supp())
    import halfcycle_tables as HT
    files = {f"dx={dx} {dom} seed {seed}": fn for fn, dx, dom, seed, circ in HT.ATRI}
    cl = []
    for r in A:
        d = np.load(os.path.join(S, files[r["run"]]))
        k = single(d) & (d["t"] >= r["t_start"]) & (d["t"] <= r["t_end"])
        cl.append(float(d["Lxy"]) - 2 * np.maximum(d["hx"][k], d["hy"][k]).max())
    cl = np.array(cl)
    dx3 = np.array(["dx=3.0" in r["run"] for r in A])
    keep = dx3 & (cl >= 260)
    put("rev.atri.dx3.clearance_ge_260.n_mean_vs_others_mean",
        [int(keep.sum()), round(qa[keep].mean(), 3), round(qa[~keep].mean(), 3)])
    put("rev.atri.min_clearance_um", round(cl.min()))
    import atri_alpha as AA
    beta = abs(float(AA.fit(os.path.join(S, "acirc4_P0.1_c052.npz"), 16.58)[0][:, 3].mean()))
    put("rev.atri.switch_interval_min_for_R_95_145_um",
        [round(qa.mean() * np.pi * R ** 2 / (2 * np.sqrt(3) * beta) / 60, 1) for R in (95.0, 145.0)])
    # the Atri ring at P = 0.095 uM: closest approach to its periodic images
    d = np.load(os.path.join(S, "aell2_P0.095_130x80.npz"))
    ok = single(d)
    c = C.smooth(d["t"][ok], float(d["Lxy"]) - 2 * np.maximum(d["hx"][ok], d["hy"][ok]), 17.5)
    put("rev.atri.P0.095.min_clearance_um", round(float(np.nanmin(c))))


def rev_procedures():
    """SI Sec. II A: finite-amplitude factors, eccentric binormal flow, the bias of a mean R^2."""
    put("rev.factor.L.at_aspect_1.17_1.5_1.77_2.06_2.43",
        [round(H.model_factor(5 * np.sqrt(a), 5 / np.sqrt(a), 1.0, radius="L"), 3) for a in (1.17, 1.5, 1.77, 2.06, 2.43)])
    put("rev.factor.Rm.at_aspect_1.17_1.5_1.77_2.06_2.43",
        [round(H.model_factor(5 * np.sqrt(a), 5 / np.sqrt(a), 1.0, radius="Rm"), 3) for a in (1.17, 1.5, 1.77, 2.06, 2.43)])
    a = 2.4
    T0 = np.pi * 25 / (2 * np.sqrt(3))
    r = F.run(5 * np.sqrt(a), 5 / np.sqrt(a), 0.0, 1.0, 6.2 * T0, n=48, dt=None, every=T0 / 400)
    x = np.r_[0.0, [c for c in C.crossings(r["t"], r["h2c"]) if c > 0.2 * T0]]
    hc = np.diff(x)                                     # planar moment to planar moment, from the planar start
    put("rev.eccentric_2.4.later_halfcycles_shorter_percent.range", _rng(100 * (1 - hc[1:] / hc[0]), 1))
    s = 1 - 0.7 ** 2                                    # R^2 linear in time, shrinking by 30%
    exact = -np.log(1 - s) / s                          # int dt / R^2 (units R0 = 1, half-cycle = 1)
    put("rev.mean_R2_bias_percent_for_30_percent_shrink", round(100 * abs(1 / (1 - s / 2) / exact - 1), 1))


def rev_barkley_rings():
    """Secs. IV A, B: Figs. 2 and 3, circular rings, calibrations, spirals."""
    import make_figs as MF
    import halfcycle_uniform as HU
    cwd = os.getcwd(); os.chdir(B)
    try:
        t, S2 = MF.smooth_all("rerun_r2/ell_a1.1_b0.17_8x5.npz", 4.44)
        put("rev.fig2.halfwidths_t4", [round(float(np.interp(4.0, t, S2["hx"])), 1), round(float(np.interp(4.0, t, S2["hy"])), 1)])
        put("rev.fig2.planar_moments_after_t6", [round(x, 1) for x in C.crossings(t, S2["z2c"]) if x > 6][:2])
        put("rev.fig2.circle_beta_c_at_first_last_window",
            [round(float(HU.barkley_calibration(0.17).beta(np.mean(S2["length"][(t >= a) & (t < a + 10)]) / (2 * np.pi))), 2) for a in (8, 108)])
        cal = HU.barkley_calibration(0.17)
        X0 = MF.measured_shape(t, S2, 4.0)
        M = MF.follow_size(t, S2, 4.0, X0, cal)
        sw = [x for x in C.crossings(t, S2["r2c"]) if x > 4]
        sm = [x for x in C.crossings(M["t"], M["hx"] - M["hy"]) if x > 4]
        n = min(len(sw), len(sm)) - 1
        put("rev.fig2.model_halfcycle_ratios.range", _rng(np.diff(sw)[:n] / np.diff(sm)[:n]))
        r = F.run(0, 0, 0.0, -1.0, 20.0, n=96, every=0.05, X0=X0)       # constant beta = 1 from the measured shape
        RL = r["length"] / (2 * np.pi)
        f = []
        for a in (0, 5, 10):
            m = (r["t"] >= a) & (r["t"] < a + 10)
            f.append(abs(np.polyfit(r["t"][m], r["z"][m], 1)[0]) * RL[m].mean())
        put("rev.fig2.model_drift_deficit_percent.windows_0_5_10", [round(100 * (1 - v), 1) for v in f])
        # Fig. 3
        t, S3 = MF.smooth_all("rerun_r2/ellP_a1.1_b0.15_12x7.5.npz", 4.13)
        cal15 = HU.barkley_calibration(0.15)
        X0 = MF.measured_shape(t, S3, 4.0)
        rr = MF.lowest_order_run(X0, cal15, 88.0)
        RLm = rr["length"] / (2 * np.pi)
        keep = np.cumsum(~(RLm > 2.5)) == 0
        RL = S3["length"] / (2 * np.pi)
        R0 = float(np.interp(4.0, t, RL))
        put("rev.fig3.R_at_t4", round(R0, 2))
        xs = [c for c in C.crossings(t, S3["hx"] - S3["hy"]) if c > 4.0]
        sim = [(c, np.log(R0 / np.interp(c, t, RL))) for c in xs if np.interp(c, t, RL) > 3.0]
        tvalid = rr["t"][keep][-1]
        ms_ = [c for c in C.crossings(rr["t"], rr["hx"] - rr["hy"]) if 0.5 < c < tvalid]
        low = [(c + 4.0, np.log(RLm[0] / np.interp(c, rr["t"], RLm))) for c in ms_ if np.interp(c, rr["t"], RLm) > 3.0]
        put("rev.fig3.sim_lnR_spacings", [round(sim[i + 1][1] - sim[i][1], 2) for i in range(len(sim) - 1)])
        put("rev.fig3.model_vs_sim_lnR_deviation_first_two", [round(abs(low[i][1] - sim[i][1]), 2) for i in range(2)])
        put("rev.fig3.sim_over_model_time_to_switch_first_two", [round((sim[i][0] - 4.0) / (low[i][0] - 4.0), 2) for i in range(2)])
        put("rev.fig3.third_switch_lnR_model_sim", [round(low[2][1], 2), round(sim[2][1], 2)])
        put("rev.fig3.R_at_t44_model_sim", [round(float(np.interp(40.0, rr["t"], RLm)), 1), round(float(np.interp(44.0, t, RL)), 1)])
        al, be, _, _ = MF.fit_shrink_drift("ellP_a0.8_b0.05_12x7.5.npz", 3.39, 6.0)
        put("rev.fig3.b005.alpha_abs_beta", [round(al, 2), round(abs(be), 2)])
        # circular rings at b = 0.17: the calibration of Fig. 4
        files = [f for f in MF.VALID_CIRCLES + MF.LARGE_CIRCLES] + [os.path.join(RV, "circ17_R0_26.npz")]
        used = [f for f in files if len(HU.circle_windows(f, 4.44, 10.0, 5.0, 20.0))]
        put("rev.fig4.calibration_rings_and_seed_radii", [len(used), _rng([MF.seed_radius(f) for f in used], 1)])
        put("rev.fig4.calibration_K_range", _rng(cal.Krange))
        K0, K1 = cal.Krange
        Ks = np.linspace(K0, K1, 4001)
        put("rev.fig4.alpha_at_K_range_ends_and_zero", [round(float(np.polyval(cal.pa, K0)), 2), round(float(np.polyval(cal.pa, K1)), 2),
                                                        round(float(Ks[np.argmin(np.abs(np.polyval(cal.pa, Ks)))]), 2)])
        put("rev.fig4.beta_at_K_range_ends", [round(float(np.polyval(cal.pb, K0)), 1), round(float(np.polyval(cal.pb, K1)), 1)])
        w = HU.circle_windows("sf2/circXXL_R0_38_snaps.npz", 4.44, 10.0, 5.0, 20.0)
        w = w[: int(np.searchsorted(np.arange(20, 200, 5.0) + 5, 145))]
        put("rev.fig4.R0_38_windows_K.range", _rng(w[:, 0], 3))
        fin = {}
        for f in ("circS_R0_4.5.npz", "circHB_R0_5.0.npz", "circS_R0_5.5.npz", "circL_R0_6.npz", "circS_R0_6.5.npz"):
            tt, s, _ = smoothed(f, 4.44, ("Rm", "length"))
            g = np.isfinite(s["Rm"]) & np.isfinite(s["length"])
            fin[f] = (s["Rm"][g][-1], s["length"][g][-1] / (2 * np.pi))
        put("rev.fig4.final_radii_near_Rstar.Rm_and_L2pi", [_rng([v[0] for v in fin.values()]), _rng([v[1] for v in fin.values()])])
        slope, ends = [], []
        for f in ("circS_R0_7.5.npz", "circL_R0_9.npz", "circL_R0_13.npz"):
            tt, s, d = smoothed(f, 4.44, ("Rm",))
            g = np.isfinite(s["Rm"])
            m = g & (tt > tt[g][-1] - 20)
            slope.append(round(float(np.polyfit(tt[m], s["Rm"][m], 1)[0]), 4))
        put("rev.fig4.final_dR_dt.R0_7.5_9_13", slope)
        d = np.load("circ_R0_4.npz")
        put("rev.fig4.R0_4.last_filament_t_and_run_end", [round(float(d["t"][np.isfinite(d["Rm"])][-1]), 1), round(float(d["t"][-1]), 1)])
        Tmax = max(float(np.load(f)["t"][-1]) for f in MF.VALID_CIRCLES)
        put("rev.fig4.longest_circle_run", round(Tmax))
    finally:
        os.chdir(cwd)
    # calibrations at b = 0.15, 0.16, 0.18 (SI Sec. III, Table S3)
    c15 = HU.barkley_calibration(0.15)
    P = c15.P[(c15.P[:, 0] >= 0.13) & (c15.P[:, 0] <= 0.33)]
    put("rev.b015.beta_windows_K_0.13_0.33.range", _rng(P[:, 2]))
    put("rev.b015.beta_fit_K_0.13_0.33.range", _rng(np.polyval(c15.pb, np.linspace(0.13, 0.33, 201))))
    c16 = HU.barkley_calibration(0.16)
    Ks = np.linspace(*c16.Krange, 2000)
    put("rev.b016.Rstar_from_fit", round(float(1 / Ks[np.argmin(np.abs(np.polyval(c16.pa, Ks)))]), 1))
    fin16 = []
    for R0 in ("5", "8", "10"):
        d = np.load(os.path.join(RV, f"circ16_R0_{R0}.npz"))
        ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
        Rs = C.smooth(d["t"][ok], d["Rm"][ok], 4.318)
        fin16.append(Rs[np.isfinite(Rs)][-1])
    put("rev.b016.final_Rm.R0_5_8_10", _rng(fin16, 1))
    coll = []
    for f in sorted(glob.glob(os.path.join(RV, "circ15_R0_*.npz"))):
        d = np.load(f)
        ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
        Rs = C.smooth(d["t"][ok], d["Rm"][ok], 4.13)
        coll.append(bool(Rs[np.isfinite(Rs)][-1] < 2.5 and d["t"][-1] < float(d["T"]) - 1) if "T" in d.files else bool(Rs[np.isfinite(Rs)][-1] < 2.5))
    put("rev.b015.all_rings_collapse", bool(all(coll)))
    c18 = HU.barkley_calibration(0.18)
    put("rev.b018.max_window_alpha", round(float(c18.P[:, 1].max()), 3))
    grow = []
    for f in sorted(glob.glob(os.path.join(RV, "circ18_R0_*.npz"))):
        d = np.load(f)
        ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1)
        Rs = C.smooth(d["t"][ok], d["Rm"][ok], 4.75)
        g = np.isfinite(Rs)
        grow.append(bool(Rs[g][-1] > Rs[g][0]))
    put("rev.b018.all_rings_expand", bool(all(grow)))
    # two-dimensional spirals (Table S4)
    sp = {}
    for f in glob.glob(os.path.join(RV, "spiral2d", "*.json")):
        d = json.load(open(f))
        sp[d["model"] + (f"{d['b']}" if d["model"] == "barkley" else ("0.775" if "0.775" in d["fields"] else "0.8"))] = d
    put("rev.spirals.barkley.core_radius_b0.15_0.16_0.17_0.18", [round(sp[f"barkley{b}"]["core_radius"], 2) for b in (0.15, 0.16, 0.17, 0.18)])
    put("rev.spirals.barkley.wavelength_b0.15_0.16_0.17_0.18", [round(sp[f"barkley{b}"]["wavelength"], 1) for b in (0.15, 0.16, 0.17, 0.18)])
    put("rev.spirals.panfilov.core_radius_wavelength_gf0.775_0.8",
        [[round(sp[f"panfilov{g}"]["core_radius"], 2), round(sp[f"panfilov{g}"]["wavelength"], 1)] for g in ("0.775", "0.8")])


def rev_modes():
    """Sec. IV D, SI Sec. IV: ring modes, filament definitions, twist, rigidity, straight filaments."""
    import make_figs as MF
    import halfcycle_uniform as HU
    MR = json.load(open(os.path.join(B, "mode_refit.json")))
    MDf = json.load(open(os.path.join(RV, "mode_definitions.json")))
    RT = json.load(open(os.path.join(B, "rigidity_test.json")))
    quad = lambda a: float(min(abs(a) % 180, 180 - abs(a) % 180))     # departure from quadrature
    std4 = [r for r in MDf["rstar"] if r["m"] == 4 and "standard" in r["definition"]][0]
    put("rev.modes.Rm_at_Rstar.range", _rng([r["R"] for r in MR["rstar"]] + [std4["R"]], 2))
    full = [r for r in MR["tau_fits"] if r["run"] == "larger ring" and r["t0"] == 15.0 and r["t1"] in (150.0, 100.0)]
    put("rev.modes.larger_ring.R_mean_m2_m3", [round(r["R"], 1) for r in full])
    put("rev.modes.larger_ring.m2.R_start_end", [round(full[0]["R_start"], 1), round(full[0]["R_end"], 1)])
    put("rev.modes.quadrature_deg.standard_max", round(max([quad(r["dphase_deg"]) for r in MR["rstar"]] + [quad(r["dphase_deg"]) for r in full]), 1))
    other = [r for r in MDf["rstar"] + MDf["large"] if r["m"] in (2, 3) and "standard" not in r["definition"]]
    put("rev.modes.quadrature_deg.other_definitions_max", round(max(quad(r["dphase_deg"]) for r in other), 1))
    put("rev.modes.quadrature_deg.m4.range", _rng([quad(r["dphase_deg"]) for r in MDf["rstar"] if r["m"] == 4], 0))
    cwd = os.getcwd(); os.chdir(B)
    try:
        rows = MF.mode_ranges()
    finally:
        os.chdir(cwd)
    for r in rows:
        key = f"rev.modes.{'Rstar' if r['ring'] == 'R*' else 'larger'}.m{r['m']}"
        put(key + ".k", round(r["k"], 2))
        put(key + ".bz_over_drift.main_range", [round(r["bz"], 2)] + _rng(r["bz_rng"]))
        put(key + ".br_over_drift.main_range", [round(r["br"], 2)] + _rng(r["br_rng"]))
    m2 = [r for r in rows if r["m"] == 2]
    put("rev.modes.m2.halfperiod_ratio_from_couplings.Rstar_larger", [round(1 / np.sqrt(r["bz"] * r["br"]), 3) for r in m2])
    put("rev.modes.m2.halfperiod_ratio_vs_circle.Rstar_larger",
        [round(float(HU.barkley_calibration(0.17).beta(R)) / r["W"], 3) for r, R in zip(m2, (np.mean([x["R"] for x in MR["rstar"] if x["m"] == 2]), full[0]["R"]))])
    st = sorted(rows, key=lambda r: r["k"])
    sl = np.load(os.path.join(B, "sf2", "sf2_linear.npy"))
    ib = lambda k: np.interp(k, sl[:, 0], sl[:, 3])
    put("rev.modes.ringmodes_below_straight_percent.range_all_six", _rng([100 * (1 - r["W"] / ib(r["k"])) for r in st], 0))
    put("rev.modes.ringmodes_below_straight_percent.range_m2_m3", _rng([100 * (1 - r["W"] / ib(r["k"])) for r in st if r["m"] < 4], 0))
    bk = np.load(os.path.join(B, "sf2", "buckle.npz"))["fits"]
    put("rev.buckle.m_k_R", [[int(bk[:, 0].min()), int(bk[:, 0].max())], _rng(bk[:, 2]), round(float(bk[:, 1].mean()))])
    lam = [2 * np.pi * r["R"] / r["m"] for r in MR["rstar"]] + [2 * np.pi * std4["R"] / 4] + [2 * np.pi * r["R_start"] / r["m"] for r in full]
    put("rev.modes.deformation_wavelengths.range", _rng(lam, 0))
    # the local (curvature-only) law at R*
    R2 = [r for r in MR["rstar"] if r["m"] == 2]
    R3 = [r for r in MR["rstar"] if r["m"] == 3]
    put("rev.modes.local_law.alpha_c_and_alpha_r_at_Rstar", [round(float(np.mean([r["alpha_zeta_loc"] for r in MR["rstar"]])), 3),
                                                           round(float(np.mean([r["alpha_r_loc"] for r in MR["rstar"]])), 2)])
    put("rev.modes.local_law.growth_m2_m3", [round(float(np.mean([r["growth_loc"] for r in R2])), 3), round(float(np.mean([r["growth_loc"] for r in R3])), 3)])
    put("rev.modes.decay_m2_m3", [round(float(np.mean([r["gamma"] for r in R2])), 3), round(float(np.mean([r["gamma"] for r in R3])), 3)])
    put("rev.modes.local_law.beta_r_over_drift_at_Rstar", round(float(HU.barkley_calibration(0.17).ratio_r(np.mean([r["R"] for r in R2]))), 2))
    for m in (2, 3):
        a = [r for r in MR["rstar"] if r["m"] == m]
        dfn = [r for r in MDf["rstar"] if r["m"] == m]
        put(f"rev.modes.m{m}.alpha_r.range_amps", _rng([r["alpha_r"] for r in a]))
        put(f"rev.modes.m{m}.alpha_zeta.range_amps", _rng([r["alpha_zeta"] for r in a]))
        put(f"rev.modes.m{m}.alpha_r.range_definitions", _rng([r["alpha_r"] for r in dfn]))
        put(f"rev.modes.m{m}.alpha_zeta.range_definitions", _rng([r["alpha_zeta"] for r in dfn]))
    d2 = [r for r in MDf["rstar"] if r["m"] == 2]
    put("rev.defs.m2.bz_over_drift.range", _rng([r["bz"] / r["bdrift"] for r in d2], 3))
    put("rev.defs.m2.br_over_drift.max_spread", round(max(np.ptp([r["br"] / r["bdrift"] for r in d2 if r["amp"] == a]) for a in (0.08, 0.15)), 2))
    put("rev.defs.m2.Rm.range", _rng([r["R"] for r in d2]))
    put("rev.defs.Rstar.omega_spread_percent", round(max(100 * np.ptp([r["omega"] for r in MDf["rstar"] if r["m"] == m and r["amp"] == a])
                                                             / np.mean([r["omega"] for r in MDf["rstar"] if r["m"] == m and r["amp"] == a])
                                                             for m, a in ((2, 0.08), (2, 0.15), (3, 0.08), (3, 0.15))), 2))
    put("rev.defs.Rstar.max_bz_change", round(max(np.ptp([r["bz"] / r["bdrift"] for r in MDf["rstar"] if r["m"] == m and r["amp"] == a])
                                                  for m, a in ((2, 0.08), (2, 0.15), (3, 0.08), (3, 0.15))), 3))
    for m in (2, 3):
        put(f"rev.defs.larger.m{m}.bz_over_drift.range", _rng([r["bz"] / r["bdrift"] for r in MDf["large"] if r["m"] == m], 3))
    tw = MDf["twist"]
    put("rev.twist.amp_over_r.range", _rng([t["rel"][f"tw{t['m']}c"]["amp_over_r"] for t in tw]))
    put("rev.twist.abs_phase_deg.range", _rng([abs(t["rel"][f"tw{t['m']}c"]["phase_deg"]) for t in tw], 0))
    put("rev.twist.m2.max_abs_phase_deg", round(max(abs(t["rel"]["tw2c"]["phase_deg"]) for t in tw if t["m"] == 2)))
    put("rev.twist.sin_part.max_amp_over_r", round(max(t["rel"][f"tw{t['m']}s"]["amp_over_r"] for t in tw), 3))
    put("rev.twist.local_median.range", _rng([t["tw_rms_median"] for t in tw], 3))
    put("rev.twist.net.max", round(max(t["tw_net_max"] for t in tw), 3))
    s2 = [t for t in tw if t["m"] == 2 and "0.15" in t["run"] and "modeR" in t["run"]]
    put("rev.twist.figS2_run.amp_over_r", round(s2[0]["rel"]["tw2c"]["amp_over_r"], 2) if s2 else None)
    put("rev.twist.m2.max_departure_alpha_r_beta_r_from_local",
        [round(max(abs(r["alpha_r"] - r["alpha_r_loc"]) for r in R2), 2), round(max(abs(r["br"] - r["beta_r_loc"]) for r in R2), 2)])
    put("rev.twist.m2.shift_alpha_zeta_and_beta_zeta",
        [round(float(np.mean([r["alpha_zeta"] - r["alpha_zeta_loc"] for r in R2])), 2), round(float(np.mean([r["beta_zeta_loc"] - r["bz"] for r in R2])), 2)])
    # rigidity-extended local law
    ring = RT["rings"]
    put("rev.rigidity.e2_from_freq.range", _rng([r["e2_freq"] for r in ring], 1))
    put("rev.rigidity.implied_split.bz_br.ranges", [_rng([r["bz_pred_freq"] for r in ring]), _rng([r["br_pred_freq"] for r in ring])])
    put("rev.rigidity.e2_from_zeta.range", _rng([r["e2_zeta"] for r in ring], 1))
    put("rev.rigidity.e2_from_r.range", _rng([r["e2_r"] for r in ring], 2))
    put("rev.rigidity.e1_from_zeta_and_r.ranges", [_rng([r["e1_zeta"] for r in ring], 1), _rng([r["e1_r"] for r in ring], 1)])
    stf = RT["straight"]
    put("rev.rigidity.straight.linear_fit_gamma2_e2", [round(stf["fit_linear"][1], 2), round(stf["fit_linear"][0], 1)])
    put("rev.rigidity.straight.linear_residuals_abs.range", _rng(np.abs(stf["residuals_linear"])))
    put("rev.rigidity.straight.omega_over_k2_uncertainty.range", _rng(sl[:, 4], 3))
    put("rev.rigidity.straight.quadratic_fit_e2_e3", [round(stf["fit_quadratic"][1], 1), round(stf["fit_quadratic"][0], 1)])
    # straight filaments at b = 0.15 and 0.16 (controls)
    CT = json.load(open(os.path.join(RV, "controls.json")))["straight"]
    s15 = [c for c in CT if c["label"].startswith("b=0.15")]
    put("rev.straight.b015.all_decay_and_beta", [bool(all(c["gamma_over_k2"] > 0 for c in s15)), [round(c["beta"], 2) for c in s15]])
    put("rev.straight.b016.k045.gamma_over_k2", round([c for c in CT if c["label"].startswith("b=0.16")][0]["gamma_over_k2"], 3))


def rev_controls():
    """Sec. III: resolution, domain size, time step, seed orientation; Fig. 2 tilt estimator; solver checks."""
    import halfcycle_uniform as HU
    import period_analysis as PA
    import ellipse_ring as E
    import period_analysis_r2 as P2
    import controls_check as CC
    from mode_theory2 import solver_check
    CT = json.load(open(os.path.join(RV, "controls.json")))
    bb = CT["bigbox"]
    put("rev.controls.mode_bigbox.max_relative_change", float("%.1g" % max(abs(bb["bigbox"][k] / bb["standard"][k] - 1) for k in ("omega", "q", "gamma"))))
    fac = np.load(os.path.join(B, "amp_factor.npy"))
    cal = HU.barkley_calibration(0.17)

    def halfcycles(path, key, per=4.44, T=80.0):
        d = np.load(path)
        ok = np.isfinite(d[key]) & np.isfinite(d["length"])
        t = d["t"][ok]; good = PA.single_closed(d)[ok]
        y, L, z = (C.smooth(t, np.where(good, d[k][ok], np.nan), per) for k in (key, "length", "z"))
        R = L / (2 * np.pi)
        x = [c for c in C.crossings(t, y) if c <= T]
        out = []
        for a, b in zip(x[:-1], x[1:]):
            m = (t >= a) & (t <= b) & np.isfinite(R) & np.isfinite(z)
            eps = np.max(np.abs(y[m]) / R[m]); f = float(np.interp(eps, fac[:, 0], fac[:, 1]))
            row = HU.predictions(t[m], R[m], z[m], f, cal)
            out.append((row["ratio_own"], row["ratio_c"]))
        return np.array(out)
    ref = halfcycles(os.path.join(B, "rerun_r2", "ell_a1.1_b0.17_8x5.npz"), "r2c")
    dt4 = halfcycles(os.path.join(RV, "ell_8x5_dt04.npz"), "r2c")
    rot = halfcycles(os.path.join(RV, "ell_8x5_rot45.npz"), "r2s")
    n = min(len(ref), len(dt4), len(rot))
    for j, nm in ((1, "ratio_c"), (0, "ratio_own")):
        ch = np.r_[dt4[:n, j] / ref[:n, j] - 1, rot[:n, j] / ref[:n, j] - 1] * 100
        put(f"rev.controls.dt_half_and_rot45.{nm}_change_percent.range", _rng(ch, 1))
        put(f"rev.controls.dt_half_and_rot45.{nm}_mean_change_percent",
            [round(100 * (dt4[:n, j].mean() / ref[:n, j].mean() - 1), 2), round(100 * (rot[:n, j].mean() / ref[:n, j].mean() - 1), 2)])
    sw = CT["switches"]
    hp = lambda x: np.diff(x)
    put("rev.controls.dt_half_and_rot45.halfperiod_change_percent.range",
        _rng(np.r_[hp(sw["dt_half"]) / hp(sw["reference"])[:len(sw["dt_half"]) - 1] - 1, hp(sw["rot45"]) / hp(sw["reference"])[:len(sw["rot45"]) - 1] - 1] * 100, 1))
    # elliptic ring in the larger domain: switch times (intervals of period_analysis_r2)
    cwd = os.getcwd(); os.chdir(B)
    try:
        a = P2.intervals_r2("rerun_r2/ell_a1.1_b0.17_12x7.5.npz", 4.44)
        b = P2.intervals_r2("conv/ell_12x7.5_bigbox.npz", 4.44)
        sa = np.array([r["t0"] for r in a] + [a[-1]["t1"]]); sb = np.array([r["t0"] for r in b] + [b[-1]["t1"]])
        k = min(len(sa), len(sb))
        put("rev.controls.ellipse_bigbox.max_switch_time_change", round(float(np.max(np.abs(sa[:k] - sb[:k]))), 2))
        # dx = 0.1 and 0.15 ellipses: mean ratio (own drift, as conv_compare.py; and circular-ring baseline)
        own = lambda rows: np.array([r["Th"] / PA.predict(r, fac)[2] for r in rows])
        r0 = P2.intervals_r2("rerun_r2/ell_a1.1_b0.17_8x5.npz", 4.44)
        for lab, f in (("dx0.1", "conv/ell_8x5_dx0.1.npz"), ("dx0.15", "ctrl_ell_8x5_dx015.npz")):
            r1 = P2.intervals_r2(f, 4.44)
            nn = min(len(r0), len(r1))
            c0, c1 = halfcycles(os.path.join(B, "rerun_r2", "ell_a1.1_b0.17_8x5.npz"), "r2c", T=1e9), halfcycles(os.path.join(B, f), "r2c", T=1e9)
            mm = min(len(c0), len(c1))
            put(f"rev.controls.{lab}.mean_ratio_change_percent.own_and_c_and_n",
                [round(100 * (own(r1)[:nn].mean() / own(r0)[:nn].mean() - 1), 2), round(100 * (c1[:mm, 1].mean() / c0[:mm, 1].mean() - 1), 2), nn, mm])
        # circular ring in a larger domain, and dx = 0.15 circles, against the calibration windows
        wa = HU.circle_windows("circS_R0_6.5.npz", 4.44, 10.0, 5.0, 20.0)
        wb = HU.circle_windows("ctrl_circ_R0_6.5_bigbox.npz", 4.44, 10.0, 5.0, 20.0)
        k = min(len(wa), len(wb))
        put("rev.controls.circle_bigbox.max_change_alpha_beta", [round(float(np.max(np.abs(wa[:k, 1] - wb[:k, 1]))), 4),
                                                                 round(float(np.max(np.abs(wa[:k, 2] - wb[:k, 2]))), 4)])
        dal, dbe = [], []
        for f in ("ctrl_circ_R0_6.5_dx015.npz", "ctrl_circ_R0_9_dx015.npz"):
            w = HU.circle_windows(f, 4.44, 10.0, 5.0, 20.0)
            w = w[(w[:, 0] >= cal.Krange[0]) & (w[:, 0] <= cal.Krange[1])]
            dal += list(w[:, 1] - np.polyval(cal.pa, w[:, 0]))
            dbe += list(w[:, 2] / np.polyval(cal.pb, w[:, 0]) - 1)
        put("rev.controls.dx015.alpha_shift.range", _rng(dal))
        put("rev.controls.dx015.beta_change_percent.max_abs", round(100 * float(np.max(np.abs(dbe))), 1))
        med_a, med_b = [], []
        for f in ("ctrl_circ_R0_6.5_dx015.npz", "ctrl_circ_R0_9_dx015.npz"):
            w = HU.circle_windows(f, 4.44, 10.0, 5.0, 20.0)
            med_a.append(float(np.median(w[:, 1] - np.polyval(cal.pa, w[:, 0]))))
            med_b.append(float(np.median(w[:, 2] / np.polyval(cal.pb, w[:, 0]) - 1)))
        put("rev.controls.dx015.median_alpha_shift.R0_6.5_9", [round(v, 2) for v in med_a])
        put("rev.controls.dx015.median_beta_change_percent.R0_6.5_9", [round(100 * v, 1) for v in med_b])
        # straight filament: side walls 32 vs 64 apart
        g32, g64 = CC.straight_rates("sf2/L32_k035.npz"), CC.straight_rates("sf2/L64_k035.npz")
        kk = g32[0]
        put("rev.controls.straight_walls_32_vs_64.rate_changes", [round(abs((g32[1] - g64[1]) * kk ** 2), 6), round(abs((g32[2] - g64[2]) * kk ** 2), 6)])
        # time of formation: first frame with a single closed filament in the elliptic runs
        first = []
        for path, per, b_, lab, tc, with_r2 in HU.barkley_runs():
            d = np.load(path)
            first.append(float(d["t"][PA.single_closed(d) & np.isfinite(d["Rm"])][0]))
        put("rev.methods.formation_time.max", round(max(first), 1))
        # axis switches by r2 against h_x = h_y
        dd, per_run = [], []
        for path, a_, b_, lab, P, rigid in PA.RUNS:
            if not rigid or a_ != 1.1:
                continue
            A_ = PA.intervals(path, P)
            B_ = P2.intervals_r2(P2.SRC.get(path, path), P)
            if not A_ or not B_:
                continue
            sw_a = np.array([r["t0"] for r in A_] + [A_[-1]["t1"]])
            sw_b = np.array([r["t0"] for r in B_] + [B_[-1]["t1"]])
            d_ = [float(np.min(np.abs(sw_b - s))) for s in sw_a]
            dd += d_; per_run.append(np.median(d_))
        put("rev.methods.switch_r2_vs_hx.median_pooled_and_per_run_max", [round(float(np.median(dd)), 3), round(float(max(per_run)), 3)])
    finally:
        os.chdir(cwd)
    # tilt estimator on an exactly tilted circle (SI Sec. I E)
    th = 2 * np.pi * np.arange(4000) / 4000; R = 4.96; vt = np.radians(12.0)
    Pc = np.c_[R * np.cos(th), R * np.sin(th) * np.cos(vt), R * np.sin(th) * np.sin(vt)]
    m = E.measure([np.r_[Pc, Pc[:1]]], 50.0)
    put("rev.tilt.expected_ratio_for_estimators", round(float(m["Rm"] * np.tan(vt) / np.hypot(m["z1s"], m["z1c"])), 4))
    # solver: weak ellipse against Eq. (4), and the inversion check of Sec. II
    r = F.run(1.02, 0.98, 0.0, 1.0, 6.0, n=48, dt=0.0005, every=0.01)
    x = C.crossings(r["t"], r["hx"] - r["hy"])
    Rb = r["length"][0] / (2 * np.pi)
    put("rev.solver.weak_ellipse_period_error_percent", round(100 * (np.mean(np.diff(x)) / (np.pi * Rb ** 2 / (2 * np.sqrt(3))) - 1), 2))
    ew, ea = [], []
    for mm in (2, 3):
        for s_ in (0.0, 3.0):
            fit, pw, pr, w0 = solver_check(m=mm, slope=s_)
            ew.append(abs(fit["omega"] / pw - 1)); ea.append(abs(fit["ratio"] / pr - 1))
    put("rev.solver.inversion_check.max_error_percent_freq_ratio", [round(100 * max(ew), 2), round(100 * max(ea), 2)])


def rev_supp_models():
    """Sec. V and SI Secs. V, VI: switching, sizes, wiggles, and Eq. (5) for the Panfilov-Rudenko rings."""
    import halfcycle_uniform as HU
    import pan_alpha
    for fn, gf in (("pell_gf0.775_20x12.npz", 0.775), ("pell_gf0.8_24x15.npz", 0.8)):
        Sx = H.smoothed(os.path.join(S, fn), pan_alpha.PERIOD[gf])
        t = Sx["t"]
        sw = C.crossings(t, Sx["r2c"])
        pm = C.crossings(t, Sx["z2c"])
        g = np.isfinite(Sx["Rm"])
        put(f"rev.panfilov.gf{gf}.switches_and_Rm_first_last", [len(sw), round(float(Sx["Rm"][g][0]), 1), round(float(Sx["Rm"][g][-1]), 1)])
        ell = [abs(np.interp(x, t, Sx["hx"]) - np.interp(x, t, Sx["hy"])) / (np.interp(x, t, Sx["hx"]) + np.interp(x, t, Sx["hy"])) for x in pm]
        e2 = [abs(np.interp(x, t, Sx["r2c"])) / np.interp(x, t, Sx["Rm"]) for x in pm]
        Rp = [np.interp(x, t, Sx["Rm"]) for x in pm]
        put(f"rev.panfilov.gf{gf}.planar_moments.ellipticity_hx_hy_first_last", [round(ell[0], 2), round(ell[-1], 2)])
        put(f"rev.panfilov.gf{gf}.planar_moments.r2_over_Rm_first_last", [round(e2[0], 2), round(e2[-1], 2)])
        put(f"rev.panfilov.gf{gf}.planar_moments.Rm_first_last", [round(Rp[0], 1), round(Rp[-1], 1)])
        put(f"rev.panfilov.gf{gf}.tension_growth_of_relative_ellipticity", round(float((Rp[-1] / Rp[0]) ** 2.5), 2))
        U = [r for r in _uniform() if r["model"] == "panfilov" and r["radius"] == "Rm" and f"gf={gf} dx=0.6" in r["run"]]
        cal = HU.panfilov_calibration(gf)
        Rmean = float(np.mean([r["Rbar"] for r in U]))
        br = float(cal.beta_r_over_beta(Rmean))
        put(f"rev.panfilov.gf{gf}.beta_r_over_beta_at_mean_Rbar_and_shortening_percent", [round(br, 2), round(100 * (1 - 1 / np.sqrt(br)))])
    wig = []
    for p in [os.path.join(S, f"pcirc_gf{g}_R{R}.npz") for g in (0.775, 0.8) for R in (14, 22)] + \
             [os.path.join(RV, f"pcirc_gf{g}_R18.npz") for g in (0.775, 0.8)]:
        d = np.load(p)
        ok = single(d) & (d["t"] > 2 * 40.0)
        wig.append(np.median(d["length"][ok] / (2 * np.pi * d["Rm"][ok]) - 1))
    put("rev.panfilov.circles.L_over_2piRm_minus_1.median_percent.range", _rng(100 * np.array(wig), 1))
    import atri
    put("rev.atri.rest_state_c_nM.P0.095", round(1000 * atri.rest(atri.mu_of_P(0.095))[0]))
    ref, dx2, cst = OUT["tableS3.P0.100"], OUT["tableS3.P0.100.dx2"], OUT["tableS3.P0.100.cstar0.566"]
    put("rev.atri.circles.dx2_and_cstar_change_percent.alpha_beta",
        [[round(100 * abs(x[2] / ref[2] - 1), 1), round(100 * abs(x[4] / ref[4] - 1), 2)] for x in (dx2, cst)])


def rev_gaps_and_clearance():
    """SI Sec. II A: half-cycles rejected only because of a gap; SI Sec. VI: clearance of the Atri circles."""
    import halfcycle_uniform as HU
    import period_analysis as PA
    import period_analysis_r2 as P2
    import pan_alpha
    import halfcycle_tables as HT
    from scipy.ndimage import median_filter
    n_r2 = 0
    cwd = os.getcwd(); os.chdir(B)
    try:
        for fn, a, b, lab, per, rigid in PA.RUNS:
            if not rigid or a != 1.1:
                continue
            path = P2.SRC.get(fn, fn)
            d = np.load(path)
            ok = np.isfinite(d["r2c"]) & np.isfinite(d["length"])
            t = d["t"][ok]
            good = PA.single_closed(d)[ok]
            r2 = np.where(good, d["r2c"][ok], np.nan)
            r2 = np.where(np.abs(r2 - median_filter(np.nan_to_num(r2, nan=np.nanmedian(r2)), 9, mode="nearest")) > 0.5, np.nan, r2)
            raw = r2.copy()
            r2, L, z = (C.smooth(t, y, per) for y in (r2, np.where(good, d["length"][ok], np.nan), np.where(good, d["z"][ok], np.nan)))
            R = L / (2 * np.pi)
            tc = PA.image_contact_time(d, per)
            for t0, t1 in zip(C.crossings(t, r2)[:-1], C.crossings(t, r2)[1:]):
                m = (t >= t0) & (t <= t1) & np.isfinite(R) & np.isfinite(z)
                if m.sum() < 12 or t1 - t0 < 2 * per:
                    continue
                eps = np.max(np.abs(r2[m]) / R[m])
                other = (R[m].mean() < 3.2 or eps < 0.04 or eps > PA.EPS_MAX or np.mean(d["ncurves"][ok][m] > 1) > 0.2 or t1 > tc)
                if not other and C.max_gap(t[good & np.isfinite(raw)], t0, t1) > per / 2:
                    n_r2 += 1
        runs = [(p, per, tc) for p, per, b, lab, tc, w in HU.barkley_runs()]
    finally:
        os.chdir(cwd)
    runs += [(os.path.join(S, fn), pan_alpha.PERIOD[gf], np.inf) for fn, gf, dx, dom, seed, circ in HT.PAN]
    runs += [(os.path.join(S, fn), HT.ATRI_PERIOD, np.inf) for fn, dx, dom, seed, circ in HT.ATRI]
    n_z2 = 0
    for path, per, tc in runs:
        Sx = H.smoothed(path, per)
        x = C.crossings(Sx["t"], Sx["z2c"])
        for a, b in zip(x[:-1], x[1:]):
            if b > tc or b - a < 2 * per:
                continue
            hx0, hy0 = np.interp(a, Sx["t"], Sx["hx"]), np.interp(a, Sx["t"], Sx["hy"])
            if max(hx0, hy0) / min(hx0, hy0) >= HU.ASPECT_MIN and C.max_gap(Sx["t_valid"], a, b) > per / 2:
                n_z2 += 1
    put("rev.halfcycles.rejected_only_for_gaps.r2_z2", [n_r2, n_z2])
    import atri_alpha as AA
    cl = {}
    for P, f, per in (("0.085", "acirc3_P0.085.npz", 20.0), ("0.095", "acirc2_R80.npz", 17.5), ("0.095", "acirc2_R130.npz", 17.5),
                      ("0.100", "acirc4_P0.1_c052.npz", 16.58), ("0.100", "acirc_dx2_P0.1.npz", 16.58), ("0.100", "acirc3_P0.1.npz", 16.58)):
        rows, frac, (tc, Rc) = AA.fit(os.path.join(S, f), per)          # binned radii over the fitted windows
        m = (tc >= 100) & (tc <= rows[:, 0].max() + 200)
        c = float(np.load(os.path.join(S, f))["Lxy"]) - 2 * Rc[m].max()
        cl[P] = min(cl.get(P, np.inf), c)
    put("rev.atri.circles.min_clearance_um.P0.085_0.095_0.100", [round(cl[P]) for P in ("0.085", "0.095", "0.100")])


def round2_halfcycles():
    """Second-round revision, Sec. IV C and the abstract: the frequency ratio, the decomposition and its ranges."""
    import halfcycle_uniform as HU
    U = _uniform()
    small = lambda r: r["model"] == "barkley" and r["Rbar"] < HU.RBAR_MIN_CAL
    b17 = lambda r: r["model"] == "barkley" and "b=0.17 (" in r["run"]
    weak = lambda r: r["model"] == "barkley" and "mode" in r["run"]
    b1516 = lambda r: r["model"] == "barkley" and ("b=0.15" in r["run"] or "b=0.16" in r["run"])
    d = lambda rr: [100 * (1 - r["ratio_c_duration"] / r["ratio_c"]) for r in rr]
    put("round2.duration_ratio_smaller_percent.b015_b016", _rng(d([r for r in U if b1516(r) and r["event"] == "r2"]), 1))
    put("round2.duration_ratio_smaller_percent.b017", _rng(d([r for r in U if b17(r) and r["event"] == "r2"]), 1))
    strong = [r for r in U if b17(r) and (r["event"], r["radius"]) in (("r2", "L"), ("z2", "Rm"), ("z2", "L"))]
    put("round2.b017.strong.eq5_linear_local_and_global.range", _rng([r["eq5_factor"] for r in strong] + [r["eq5_global"] for r in strong]))
    put("round2.b017.strong.residual_linear_local_and_global.range",
        _rng([r["residual"] for r in strong] + [r["ratio_c"] / r["eq5_global"] for r in strong]))
    wk = [r for r in U if weak(r) and r["event"] == "z2" and r["radius"] == "Rm"]
    put("round2.b017.weak.ratio_eq5_residual", [round(np.mean([r[k] for r in wk]), 3) for k in ("ratio_c", "eq5_factor", "residual")])
    for run in ("a=1.1 b=0.17 (8x5)", "a=1.1 b=0.17 (12x7.5)"):
        for rad in ("Rm", "L"):
            rr = [r for r in U if r["run"] == run and r["event"] == "z2" and r["radius"] == rad]
            put(f"round2.b017.{run.split('(')[1][:-1]}.z2_{rad}.ratio_mean", round(np.mean([r["ratio_c"] for r in rr]), 3))
    sm = {rad: [np.mean([r["ratio_c"] for r in U if r["run"] == run and r["event"] == "z2" and r["radius"] == rad])
                for run in ("a=1.1 b=0.17 (8x5)", "a=1.1 b=0.17 (12x7.5)")] for rad in ("Rm", "L")}
    wm = {rad: np.mean([r["ratio_c"] for r in U if weak(r) and r["event"] == "z2" and r["radius"] == rad]) for rad in ("Rm", "L")}
    put("round2.b017.strong_below_weak_percent.range",
        _rng([100 * (1 - s / wm[rad]) for rad in ("Rm", "L") for s in sm[rad]], 1))
    # the curvature-only law at finite amplitude: T_local / T_const of two filament-equation runs from the same shape
    LL = _rows(os.path.join(B, "local_law_halfcycles.csv"))
    runmap = {"b=0.17 small (8x5)": "a=1.1 b=0.17 (8x5)", "b=0.17 large (12x7.5)": "a=1.1 b=0.17 (12x7.5)",
              "b=0.17 m=2 mode, amp 0.15": "a=1.1 b=0.17 mode m=2 amp 0.15", "b=0.17 m=2 mode, amp 0.08": "a=1.1 b=0.17 mode m=2 amp 0.08"}
    fa_s, fa_w, res_s = [], [], []
    for r in LL:
        fa = [r["T_clamp"] / r["T_const"], r["T_linear"] / r["T_const"]]
        if "mode" in r["run"]:
            fa_w += fa
            continue
        fa_s += fa
        match = [u for u in U if u["run"] == runmap[r["run"]] and u["event"] == "z2" and u["radius"] == "L"
                 and abs(u["t_start"] - r["t_start"]) < 0.3]
        if match:
            res_s += [match[0]["ratio_c"] / x for x in fa]
    put("round2.b017.finite_amplitude_eq5.strong_and_weak.range", [_rng(fa_s), _rng(fa_w)])
    put("round2.b017.finite_amplitude_residual.strong.range_n", _rng(res_s) + [len(res_s)])
    put("round2.b017.weak.z2_L.ratio_mean", round(float(wm["L"]), 3))
    # SI II D: L/2pi of the half-width ellipse that starts each integration, against the ring's own L/2pi then
    import local_law_halfcycles as LLH
    import halfcycle_uniform as HU
    ell = lambda a, b: 0.5 * (a + b) * (1 + 3 * ((a - b) / (a + b)) ** 2 / (10 + np.sqrt(4 - 3 * ((a - b) / (a + b)) ** 2)))
    bias = {"strong": [], "weak": []}
    cwd = os.getcwd(); os.chdir(B)
    try:
        for path, b, per, lab in LLH.RUNS:
            Sm = H.smoothed(path, per)
            t, ok = Sm["t"], np.isfinite(Sm["RL"])
            x = C.crossings(t, Sm["z2c"])
            for a, e in zip(x[:-1], x[1:]):
                hx0, hy0 = np.interp(a, t, Sm["hx"]), np.interp(a, t, Sm["hy"])
                if C.max_gap(Sm["t_valid"], a, e) > per / 2 or e - a < 2 * per or max(hx0, hy0) / min(hx0, hy0) < HU.ASPECT_MIN:
                    continue
                bias["weak" if "mode" in lab else "strong"].append(100 * (ell(hx0, hy0) / np.interp(a, t[ok], Sm["RL"][ok]) - 1))
    finally:
        os.chdir(cwd)
    put("round2.siIID.halfwidth_ellipse_L_minus_ring_L_percent.strong_and_weak", [_rng(bias["strong"], 1), _rng(bias["weak"], 1)])
    ok1516 = [r for r in U if b1516(r) and not small(r) and (r["event"], r["radius"]) in (("r2", "L"), ("z2", "Rm"))]
    put("round2.b015_b016.eq5_local_and_global.range", _rng([r["eq5_factor"] for r in ok1516] + [r["eq5_global"] for r in ok1516]))
    put("round2.b015_b016.residual_local_and_global.range",
        _rng([r["residual"] for r in ok1516] + [r["ratio_c"] / r["eq5_global"] for r in ok1516], 1))
    pan = [r for r in U if r["model"] == "panfilov" and "gf=0.8 " in r["run"] and r["event"] == "z2" and r["radius"] == "Rm"]
    put("round2.panfilov.gf0.8.eq5_factor_by_dx.0.6_0.4_0.3",
        [_rng([r["eq5_factor"] for r in pan if f"dx={dx} " in r["run"]], 3) for dx in ("0.6", "0.4", "0.3")])
    put("round2.small_ring_halfcycles_excluded_n", sum(1 for r in U if small(r) and r["radius"] in ("Rm", "L") and r["event"] in ("z2", "r2")
                                                        and not (r["event"] == "z2" and r["radius"] == "L")))
    # drift per half-cycle (r2 events, b = 0.17) and the lowest-order reference at finite amplitude
    r2 = [r for r in U if b17(r) and r["event"] == "r2"]
    dz = [r["beta_own"] * r["T_obs"] / r["Rbar"] ** 2 for r in r2]
    put("round2.drift_per_halfcycle_over_R.b017.range_mean", _rng(dz) + [round(np.mean(dz), 2)])
    put("round2.drift_per_halfcycle.b017.eps_range", _rng([r["eps"] for r in r2]))
    ref = []
    for e in (0.02, 0.3):
        m = F.run(1 + e, 1 - e, 0.0, 1.0, 8.0, n=64, dt=0.0005, every=0.005)
        x = C.crossings(m["t"], m["hx"] - m["hy"])
        ref.append(round(float(np.mean(np.abs(np.diff(np.interp(x, m["t"], m["z"])))) / (m["length"][0] / (2 * np.pi))), 2))
    put("round2.lowest_order_drift_per_switch_over_R.eps_0.02_0.3", ref)


def round2_modes():
    """Second-round revision, Sec. IV D (Table I), SI IV: shape modes relative to beta_c, twist bound, rigidity, dx."""
    import json
    import halfcycle_uniform as HU
    import mode_summary
    for r in mode_summary.modes():
        key = f"round2.modes.{'Rstar' if r['ring'] == 'R*' else 'larger'}.m{r['m']}"
        put(key + ".bz_br_brloc", [round(r["bz"], 2), round(r["br"], 2), round(r["br_loc"], 2)])
        put(key + ".bz_rng_br_rng", [_rng(r["bz_rng"]), _rng(r["br_rng"])])
        put(key + ".az_ar_arloc", [round(r["az"], 2), round(r["ar"], 2), round(r["ar_loc"], 2)])
        put(key + ".az_rng_ar_rng", [_rng(r["az_rng"]), _rng(r["ar_rng"])])
        put(key + ".phi_deg", [round(p, 1) for p in r["phi"]])
        put(key + ".br_over_eq5_minus_1_percent", round(100 * (r["br"] / r["br_loc"] - 1)))
    MR = json.load(open(os.path.join(B, "mode_refit.json")))
    MD = json.load(open(os.path.join(RV, "mode_definitions.json")))
    phi = lambda dd: (dd % 360.0) - 180.0
    put("round2.modes.m4.phase_windows_and_definitions.range",
        [_rng([phi(w["dphase_deg"]) for w in MD["m4_windows"]], 1), _rng([phi(x["dphase_deg"]) for x in MD["rstar"] if x["m"] == 4], 1)])
    std4 = [x for x in MD["rstar"] if x["m"] == 4 and "standard" in x["definition"]][0]
    put("round2.modes.decay_rates.m2_m3_m4", [round(np.mean([r["gamma"] for r in MR["rstar"] if r["m"] == m]), 3) for m in (2, 3)]
        + [round(std4["gamma"], 3)])
    put("round2.modes.curvature_only_growth.m2_m3", [round(np.mean([r["growth_loc"] for r in MR["rstar"] if r["m"] == m]), 3) for m in (2, 3)])
    cal = HU.barkley_calibration(0.17)
    Ks = 1 / 4.87
    loc = []
    for dK in (0.03, 0.05, 0.07):
        mk = np.abs(cal.P[:, 0] - Ks) < dK
        p = np.polyfit(cal.P[mk, 0] - Ks, cal.P[mk, 1], 2)
        loc.append(p[2] + Ks * p[1])
    put("round2.modes.alpha_r_curvature_only_at_Rstar.global_and_local_range",
        [round(float(np.polyval(cal.pa, Ks) + Ks * np.polyval(np.polyder(cal.pa), Ks)), 2), _rng(loc)])
    grow = {m: [r["growth_loc"] for r in MR["rstar"] if r["m"] == m] for m in (2, 3)}
    for r in MR["rstar"]:                    # the same growth rate with the local fits of d(alpha_c K)/dK
        grow[r["m"]] += [-(a * (r["m"] ** 2 - 1) + r["alpha_c"] * r["m"] ** 2) / (2 * r["R"] ** 2) for a in loc]
    put("round2.modes.curvature_only_growth.m2_m3.global_and_local_range", [_rng(grow[2], 3), _rng(grow[3], 3)])
    put("round2.modes.alpha_c.larger_m2_m3", [round(r["ac"], 2) for r in mode_summary.modes() if r["ring"] == "larger"])
    # Sec. III: beta_drift = R_m v (mode fits) against beta_own = slope of z against int dt / (L/2pi), over the fit windows
    dev = []
    for f, t0, t1 in (("modeR_m2_a0.08_lv.npz", 6, 125), ("modeR_m2_a0.15_lv.npz", 6, 125), ("modeR_m3_a0.08_lv.npz", 6, 45),
                      ("modeR_m3_a0.15_lv.npz", 6, 45), ("modeR_m4_a0.10_lv.npz", 6, 30), ("modeL_m2_lv.npz", 15, 150),
                      ("modeL_m3_lv.npz", 15, 100)):
        d = np.load(os.path.join(RV, f))
        ok = np.isfinite(d["Rm"]) & (d["ncurves"] == 1) & np.isfinite(d["z"])
        t = d["t"][ok]
        Rm, z, RL = (C.smooth(t, x, 4.44) for x in (d["Rm"][ok], d["z"][ok], d["length"][ok] / (2 * np.pi)))
        s = (t >= t0) & (t <= t1) & np.isfinite(Rm) & np.isfinite(z) & np.isfinite(RL)
        clock = lambda R: np.concatenate([[0], np.cumsum(np.diff(t[s]) / (0.5 * (R[s][1:] + R[s][:-1])))])
        bd = abs(np.polyfit(clock(Rm), z[s], 1)[0]) if f.startswith("modeL") else abs(np.polyfit(t[s], z[s], 1)[0]) * Rm[s].mean()
        dev.append(100 * (bd / abs(np.polyfit(clock(RL), z[s], 1)[0]) - 1))
    put("round2.modes.beta_drift_vs_beta_own_percent.max_abs", round(float(np.max(np.abs(dev))), 1))
    for m in (2, 3):                         # the five filament definitions at R*: relative to beta_c(R_m) and to their own drift
        rows = [x for x in MD["rstar"] if x["m"] == m]
        put(f"round2.defs.Rstar.m{m}.bz_over_betac_range_br_spread_bz_over_drift_spread",
            [_rng([x["bz"] / float(cal.beta(x["R"])) for x in rows]),
             round(float(np.ptp([x["br"] / float(cal.beta(x["R"])) for x in rows])), 2),
             round(max(float(np.ptp([x["bz"] / x["bdrift"] for x in rows if x["amp"] == a])) for a in {x["amp"] for x in rows}), 2)])
    fr = [2 * np.sqrt(3) * r["beta_c"] / (r["omega"] * r["R"] ** 2) for r in MR["rstar"] if r["m"] == 2]
    big = [x for x in MR["tau_fits"] if x["run"] == "larger ring" and x["m"] == 2 and x["t0"] == 15.0 and x["t1"] == 150.0][0]
    put("round2.modes.m2_frequency_ratio.Rstar_range_and_larger", [_rng(fr), round(2 * np.sqrt(3) * big["beta_c"] / big["omega_tau"], 2)])
    put("round2.modes.larger.m2.Rm_start_end_and_betac_change_percent",
        [round(big["R_start"], 1), round(big["R_end"], 1), round(100 * (float(cal.beta(big["R_end"])) / float(cal.beta(big["R_start"])) - 1), 1)])
    RT = json.load(open(os.path.join(B, "rigidity_test.json")))
    qs = []
    for r in [x for x in MR["rstar"] if x["m"] == 2]:
        t = [x for x in RT["rings"] if x["run"] == f"R* amp {r['amp']}"][0]
        f = np.sqrt(3 / 4)
        qs.append([f * np.sqrt(r["br"] / r["bz"]), f * np.sqrt(r["beta_r_loc"] / r["beta_c"]), f * np.sqrt(t["br_pred_freq"] / t["bz_pred_freq"])])
    put("round2.modes.m2_amplitude_ratio.measured_curvature_only_rigidity", [round(float(np.mean([q[i] for q in qs])), 2) for i in range(3)])
    rr = RT["rings"]
    put("round2.rigidity.e2_from_zeta_r_freq.ranges", [_rng([x["e2_zeta"] for x in rr], 1), _rng([x["e2_r"] for x in rr]),
                                                       _rng([x["e2_freq"] for x in rr], 1)])
    put("round2.rigidity.freq_matched_bz_br_over_betac.ranges", [_rng([x["bz_pred_freq"] for x in rr]), _rng([x["br_pred_freq"] for x in rr])])
    put("round2.rigidity.e1_from_zeta_and_r.ranges", [_rng([x["e1_zeta"] for x in rr], 1), _rng([x["e1_r"] for x in rr], 1)])
    # twist: |d beta_zeta| <= |d alpha_r| (m^2 - 1) tan(psi) / (m^2 q), with the twist lag psi behind r_m (SI IV C)
    from mode_refit import local_law
    bounds, shifts = {}, {}
    for t in MD["twist"]:
        m = t["m"]
        psi = np.radians(abs(t["rel"][f"tw{m}c"]["phase_deg"]))
        amp = float(os.path.basename(t["run"]).split("_a")[1].split("_")[0])
        if m == 4:
            law = local_law(cal, std4["R"], 4)
            r = dict(std4, beta_c=law["beta_c"], alpha_r_loc=law["alpha_r_loc"])
        else:
            r = [x for x in MR["rstar"] if x["m"] == m and abs(x["amp"] - amp) < 1e-9][0]
        q = np.sqrt((m * m - 1) / m ** 2) * np.sqrt(r["br"] / r["bz"])
        bounds.setdefault(m, []).append(abs(r["alpha_r"] - r["alpha_r_loc"]) * (m * m - 1) * np.tan(psi) / (m * m * q))
        shifts.setdefault(m, []).append(r["beta_c"] - r["bz"])
    put("round2.twist.beta_zeta_bound_max.m2_m3_m4", [round(max(bounds[m]), 3) for m in (2, 3, 4)])
    put("round2.twist.observed_beta_zeta_shift.m2_m3_m4", [_rng(shifts[m]) for m in (2, 3)] + [round(shifts[4][0], 2)])
    put("round2.twist.lag_deg.range", _rng([abs(t["rel"][f"tw{t['m']}c"]["phase_deg"]) for t in MD["twist"]], 0))
    # the dx = 0.15 mode runs
    RC = json.load(open(os.path.join(RF, "refine_compare.json")))["modes"] if RF else None
    if RC:
        for m in ("m2", "m3"):
            x = RC[m]["dx0.15"]
            put(f"round2.dx015.{m}.R_az_ar", [round(x["R"], 2), _rng(x["alpha_zeta"]), _rng(x["alpha_r"])])
    # beta_c(K): logarithmic and even-polynomial fits to the 426 windows at b = 0.17
    K, be = cal.P[:, 0], cal.P[:, 2]
    p = np.polyfit(np.log(K), be, 1)
    pe = np.polyfit(K ** 2, be, 2)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    put("round2.betac.log_fit.a_b_rms_n", [round(p[1], 2), round(p[0], 2), round(rms(be - np.polyval(p, np.log(K))), 3), len(K)])
    put("round2.betac.log_fit.K0", round(float(np.exp(-p[1] / p[0])), 1))
    put("round2.betac.even_poly_K2_and_cubic_rms", [round(rms(be - np.polyval(pe, K ** 2)), 3), round(rms(be - np.polyval(cal.pb, K)), 3)])
    put("round2.betac.at_K_0.04", round(float(np.polyval(cal.pb, 0.04)), 1))


SECTIONS = [("methods", methods), ("radius conventions", radii), ("Fig. 2", fig2), ("Fig. 3 and b = 0.18", fig3),
                 ("Fig. 4", fig4), ("Fig. 5", fig5), ("shape modes", modes), ("supplementary models", supp)]
if RF:
    SECTIONS.append(("grid refinement", refine))
if RV:
    SECTIONS += [("revision: half-cycles", rev_halfcycles), ("revision: procedures", rev_procedures),
                 ("revision: Barkley rings", rev_barkley_rings), ("revision: shape modes", rev_modes),
                 ("revision: controls", rev_controls), ("revision: gaps and clearance", rev_gaps_and_clearance),
                 ("revision: other models", rev_supp_models), ("second round: half-cycles", round2_halfcycles),
                 ("second round: shape modes", round2_modes)]
for name, fn in SECTIONS:
    section(name, fn)
errors = [k for k in OUT if k.startswith("ERROR")]
target = os.path.join(B, "paper_numbers.json")
if errors:                          # never replace a complete result with a partial one
    with open(target + ".partial", "w", newline=chr(10)) as f:
        json.dump(OUT, f, indent=1)
    sys.exit(f"{len(errors)} section(s) failed: {errors}; wrote {target}.partial, kept {target}")
with open(target + ".tmp", "w", newline=chr(10)) as f:
    json.dump(OUT, f, indent=1)
os.replace(target + ".tmp", target)
print("wrote", target)
