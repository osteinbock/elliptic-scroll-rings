"""Every number quoted in the manuscript and SI that follows from the simulations, recomputed
from the production data and analysis products; written to <BARKLEY>/paper_numbers.json.
    python paper_numbers.py SUPP BARKLEY        (after build_all.sh has produced the tables)
"""
import os
import sys
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ellipse_compare as C          # noqa: E402
import filament_flow as F            # noqa: E402
import halfcycle as H                # noqa: E402

S, B = sys.argv[1], sys.argv[2]
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
    # Table S1: pooled over both seeds
    import pan_alpha as PA
    for gf in (0.775, 0.8, 0.825, 0.85, 0.9):
        w = np.vstack([PA.windows(os.path.join(S, f"pcirc_gf{gf}_R{R}.npz"), single=True)[1] for R in (14, 22)])
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


for name, fn in (("methods", methods), ("radius conventions", radii), ("Fig. 2", fig2), ("Fig. 3 and b = 0.18", fig3),
                 ("Fig. 4", fig4), ("Fig. 5", fig5), ("shape modes", modes), ("supplementary models", supp)):
    section(name, fn)
with open(os.path.join(B, "paper_numbers.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote", os.path.join(B, "paper_numbers.json"))
