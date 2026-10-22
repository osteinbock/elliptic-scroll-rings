"""LaTeX table bodies of the supplementary material, generated from the analysis outputs
(REVISION/uniform_halfcycles.csv, BARKLEY/mode_refit.json, REVISION/mode_definitions.json,
BARKLEY/rigidity_test.json, REVISION/controls.json, REVISION/spiral2d/*.json, the circular-ring
calibrations; folders as in folders.py).

    python si_tables.py OUTDIR
"""
import os
import sys
import csv
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import folders as FD               # noqa: E402


def rows_csv(path):
    out = list(csv.DictReader(open(path)))
    for r in out:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def write(outdir, name, lines):
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name, len(lines), "rows")


def fmt(x, nd=3):
    return "---" if x == "" or x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def phi(dphase_deg):
    """Signed offset from quadrature (degrees) of the fitted relative phase; positive for D > A."""
    return (dphase_deg % 360.0) - 180.0


def main(outdir):
    import halfcycle_uniform as HU
    os.makedirs(outdir, exist_ok=True)
    U = rows_csv(os.path.join(FD.REVISION, "uniform_halfcycles.csv"))
    small = lambda r: r["model"] == "barkley" and r["Rbar"] < HU.RBAR_MIN_CAL      # beta(K) poorly constrained
    # ---------------------------------------------------------------- Barkley half-cycles of Fig. 5(a)
    labels = {"a=1.1 b=0.15": r"0.15, $12\times7.5$", "a=1.1 b=0.16": r"0.16, $12\times7.5$",
              "a=1.1 b=0.17 (8x5)": r"0.17, $8\times5$", "a=1.1 b=0.17 (12x7.5)": r"0.17, $12\times7.5$",
              "a=1.1 b=0.18": r"0.18, $12\times7.5$"}
    L = []
    for r in [r for r in U if r["model"] == "barkley" and r["event"] == "r2"]:
        L.append(f"{labels[r['run']]} & {r['t_start']:.1f}--{r['t_end']:.1f} & {r['T_obs']:.2f} & {r['Rbar']:.2f}"
                 f"{'$^\\dagger$' if small(r) else ''} & "
                 f"{r['eps']:.2f} & {r['alpha_own']:+.3f} & {r['beta_own']:.3f} & {r['beta_c']:.3f} & {r['factor']:.3f} & "
                 f"{r['ratio_c']:.3f} & {r['ratio_own']:.3f} & {r['eq5_factor']:.3f} & {r['residual']:.3f} \\\\")
    write(outdir, "tab_barkley_halfcycles.tex", L)
    # ---------------------------------------------------------------- per-run summary, all models (planar moments)
    L = []
    order = [("barkley", "a=1.1 b=0.15", r"Barkley, $b = 0.15$, $12\times7.5$"),
             ("barkley", "a=1.1 b=0.16", r"Barkley, $b = 0.16$, $12\times7.5$"),
             ("barkley", "a=1.1 b=0.17 (8x5)", r"Barkley, $b = 0.17$, $8\times5$"),
             ("barkley", "a=1.1 b=0.17 (12x7.5)", r"Barkley, $b = 0.17$, $12\times7.5$"),
             ("barkley", "a=1.1 b=0.17 mode m=2 amp 0.08", r"Barkley, $b = 0.17$, $m = 2$, $\delta = 0.08$"),
             ("barkley", "a=1.1 b=0.17 mode m=2 amp 0.15", r"Barkley, $b = 0.17$, $m = 2$, $\delta = 0.15$"),
             ("barkley", "a=1.1 b=0.18", r"Barkley, $b = 0.18$, $12\times7.5$"),
             ("panfilov", "gf=0.775 dx=0.6 80^2 x 50", r"P--R, $g_f = 0.775$, $\Delta x = 0.6$"),
             ("panfilov", "gf=0.8 dx=0.6 88^2 x 50", r"P--R, $g_f = 0.8$, $\Delta x = 0.6$"),
             ("panfilov", "gf=0.8 dx=0.6 120^2 x 80", r"P--R, $g_f = 0.8$, $\Delta x = 0.6$, large"),
             ("panfilov", "gf=0.8 dx=0.4 88^2 x 50", r"P--R, $g_f = 0.8$, $\Delta x = 0.4$"),
             ("panfilov", "gf=0.8 dx=0.3 76^2 x 40", r"P--R, $g_f = 0.8$, $\Delta x = 0.3$"),
             ("atri", "dx=3.0 600^2 x 280 seed 100 x 60", r"Atri, $100\times60~\mu$m"),
             ("atri", "dx=3.0 660^2 x 280 seed 130 x 80", r"Atri, $130\times80~\mu$m"),
             ("atri", "dx=3.0 600^2 x 280 seed 100 x 88", r"Atri, $100\times88~\mu$m"),
             ("atri", "dx=3.0 660^2 x 280 seed 120 x 105", r"Atri, $120\times105~\mu$m"),
             ("atri", "dx=2.0 600^2 x 280 seed 100 x 60", r"Atri, $100\times60~\mu$m, $\Delta x = 2~\mu$m")]
    for model, run, lab in order:
        rm = [r for r in U if r["model"] == model and r["run"] == run and r["event"] == "z2" and r["radius"] == "Rm" and not small(r)]
        rl = [r for r in U if r["model"] == model and r["run"] == run and r["event"] == "z2" and r["radius"] == "L" and not small(r)]
        if not rm:
            continue
        ms = lambda v: f"{np.mean(v):.3f}" + (f" $\\pm$ {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")
        L.append(f"{lab} & {len(rm)} & {min(r['aspect'] for r in rm):.2f}--{max(r['aspect'] for r in rm):.2f} & "
                 f"{ms([r['ratio_c'] for r in rm])} & {ms([r['ratio_c'] for r in rl])} & {ms([r['ratio_own'] for r in rm])} & "
                 f"{ms([r['eq5_factor'] for r in rm])} & {ms([r['residual'] for r in rm])} \\\\")
    write(outdir, "tab_crossmodel.tex", L)
    # ---------------------------------------------------------------- Atri half-cycles (planar moments, R_m)
    L = []
    for r in [r for r in U if r["model"] == "atri" and r["radius"] == "Rm"]:
        seed = r["run"].split("seed ")[1].replace(" x ", r"\times")
        dx = r["run"].split("dx=")[1][:3]
        L.append(f"${seed}$ & {float(dx):.0f} & {r['t_start']:.0f}--{r['t_end']:.0f} & {r['Rbar']:.0f} & {r['aspect']:.2f} & "
                 f"{r['T_obs']:.0f} & {r['factor']:.3f} & {r['ratio_c']:.3f} & {r['ratio_own']:.3f} \\\\")
    write(outdir, "tab_atri_halfcycles.tex", L)
    # ---------------------------------------------------------------- Panfilov half-cycles (planar moments, R_m)
    L = []
    for r in [r for r in U if r["model"] == "panfilov" and r["radius"] == "Rm"]:
        g = r["run"].split()[0].split("=")[1]
        dx = r["run"].split()[1].split("=")[1]
        dom = r["run"].split(" ", 2)[2].replace("^2 x ", r"^2\times")
        L.append(f"{g} & {dx} & ${dom}$ & {r['t_start']:.0f}--{r['t_end']:.0f} & {r['Rbar']:.1f} & {r['aspect']:.2f} & "
                 f"{r['factor']:.3f} & {r['ratio_c']:.3f} & {r['eq5_factor']:.3f} & {r['residual']:.3f} \\\\")
    write(outdir, "tab_pan_halfcycles.tex", L)
    # ---------------------------------------------------------------- calibrations at b = 0.15-0.18
    L = []
    for b in (0.15, 0.16, 0.17, 0.18):
        c = HU.barkley_calibration(b)
        nring = c.nrings
        Ks = np.linspace(*c.Krange, 2000)
        a_ = np.polyval(c.pa, Ks)
        if a_.min() < 0 < a_.max():
            Kz = Ks[np.argmin(np.abs(a_))]
            star = f"{1 / Kz:.1f}"
        else:
            star = "none (collapse)" if a_.min() > 0 else "none (expansion)"
        cells = []
        for R in (4.0, 6.0, 10.0):
            if c.inside(R):
                cells.append(f"{float(c.alpha(R)):+.2f} & {float(c.beta(R)):.2f} & {float(c.ratio_r(R)):.2f}")
            else:
                cells.append("--- & --- & ---")
        L.append(f"{b} & {nring} & {len(c.P)} & {c.Krange[0]:.3f}--{c.Krange[1]:.3f} & {star} & " + " & ".join(cells) + " \\\\")
    write(outdir, "tab_calibration.tex", L)
    # ---------------------------------------------------------------- two-dimensional spirals
    sp = {}
    for f in glob.glob(os.path.join(FD.REVISION, "spiral2d", "*.json")):
        d = json.load(open(f))
        sp[(d["model"], d.get("b", "0.775" if "0.775" in d.get("fields", "") else "0.8"))] = d
    # Periods, core radii and the Atri row are measured constants of earlier two-dimensional runs (Barkley:
    # the periods of period_analysis.RUNS; Panfilov-Rudenko: 600-unit pan2d.py runs in a no-flux square of
    # side 160; Atri: atri2d.py); the wavelengths and the Barkley core radii come from spiral2d/*.json.
    per = {0.15: 4.13, 0.16: 4.318, 0.17: 4.44, 0.18: 4.75}
    L = []
    for b in (0.15, 0.16, 0.17, 0.18):
        d = sp[("barkley", b)]
        L.append(f"Barkley, $b = {b}$ & {per[b]:.2f} & {d['core_radius']:.2f} & {d['wavelength']:.1f} \\\\")
    for g, T, rc in (("0.775", 40.0, 3.5), ("0.8", 36.9, 3.3)):
        d = sp[("panfilov", g)]
        L.append(f"Panfilov--Rudenko, $g_f = {g}$ & {T:.1f} & {rc:.1f} & {d['wavelength']:.0f} \\\\")
    L.append(r"Atri, $P = 0.1~\mu$M & 16.6~s & 20~$\mu$m & 120~$\mu$m \\")
    write(outdir, "tab_spirals.tex", L)
    # ---------------------------------------------------------------- shape modes
    MR = json.load(open(os.path.join(FD.BARKLEY, "mode_refit.json")))
    MD = json.load(open(os.path.join(FD.REVISION, "mode_definitions.json")))
    cal = HU.barkley_calibration(0.17)
    bc = lambda R: float(cal.beta(R))                      # circular-ring drift coefficient at R
    L = []
    for r in MR["rstar"]:
        L.append(f"$R^\\ast$ & {r['m']} & {r['amp']:.2f} & {r['R']:.2f} & {r['k']:.3f} & {r['omega']:.4f} & {r['gamma']:.4f} & "
                 f"{phi(r['dphase_deg']):+.1f} & {r['bz'] / r['beta_c']:.3f} & {r['br'] / r['beta_c']:.3f} & "
                 f"{r['alpha_zeta']:+.2f} & {r['alpha_r']:+.2f} \\\\")
    for r in [x for x in MD["rstar"] if x["m"] == 4 and "standard" in x["definition"]]:
        L.append(f"$R^\\ast$ & 4 & 0.10 & {r['R']:.2f} & {4 / r['R']:.3f} & {r['omega']:.4f} & {r['gamma']:.4f} & "
                 f"{phi(r['dphase_deg']):+.1f} & {r['bz'] / bc(r['R']):.3f} & {r['br'] / bc(r['R']):.3f} & "
                 f"{r['alpha_zeta']:+.2f} & {r['alpha_r']:+.2f} \\\\")
    for r in [x for x in MR["tau_fits"] if x["run"] == "larger ring" and x["t0"] == 15.0 and x["t1"] in (150.0, 100.0)]:
        L.append(f"larger & {r['m']} & {0.08 if r['m'] == 2 else 0.10:.2f} & {r['R']:.2f} & {r['k']:.3f} & "
                 f"{r['omega_tau']:.3f}$^\\tau$ & {r['gamma_tau']:.3f}$^\\tau$ & "
                 f"{phi(r['dphase_deg']):+.1f} & {r['bz'] / r['beta_c']:.3f} & {r['br'] / r['beta_c']:.3f} & "
                 f"{r['alpha_zeta']:+.2f} & {r['alpha_r']:+.2f} \\\\")
    write(outdir, "tab_modes.tex", L)
    L = []
    for r in MD["rstar"]:
        d = r["definition"].replace("u=", "$u = ").replace(", v=", "$, $v = ").replace(" (standard)", "$ (standard)")
        if "standard" not in d:
            d += "$"
        L.append(f"{r['m']} & {r['amp']:.2f} & {d} & {r['R']:.3f} & {r['omega']:.4f} & {r['q']:.4f} & {phi(r['dphase_deg']):+.1f} & "
                 f"{r['bz'] / bc(r['R']):.3f} & {r['br'] / bc(r['R']):.3f} & {r['alpha_zeta']:+.2f} & {r['alpha_r']:+.2f} \\\\")
    write(outdir, "tab_definitions.tex", L)
    # ---------------------------------------------------------------- Table S5: summary of the shape modes
    import mode_summary
    f2 = lambda x: f"{x:.2f}".replace("-", "$-$")
    span = lambda v: f"{f2(v[0])}--{f2(v[1])}" if v[0] >= 0 else f"{f2(v[0])} to {f2(v[1])}"
    L = []
    for r in mode_summary.modes():
        ring = "$R^\\ast$" if r["ring"] == "R*" else "larger"
        L.append(f"{ring} & {r['m']} & {r['k']:.2f} & {f2(r['bz'])} ({span(r['bz_rng'])}) & {f2(r['br'])} ({span(r['br_rng'])}) & "
                 f"{f2(r['br_loc'])} & {f2(r['az'])} ({span(r['az_rng'])}) & {f2(r['ar'])} ({span(r['ar_rng'])}) & "
                 f"{f2(r['ar_loc'])} & {'/'.join(f'{p:+.0f}'.replace('-', '$-$') for p in r['phi'])} \\\\")
    write(outdir, "tab_main_modes.tex", L)
    RT = json.load(open(os.path.join(FD.BARKLEY, "rigidity_test.json")))
    L = []
    for r in RT["rings"]:
        ring = "$R^\\ast$" if r["run"].startswith("R*") else "larger"
        L.append(f"{ring} & {r['m']} & {r['k']:.3f} & {r['e2_zeta']:+.2f} & {r['e2_r']:+.2f} & {r['e2_freq']:+.2f} & "
                 f"{r['bz_pred_freq']:.2f} & {r['br_pred_freq']:.2f} & {r['bz']:.2f} & {r['br']:.2f} & {r['e1_zeta']:+.2f} & {r['e1_r']:+.2f} \\\\")
    write(outdir, "tab_rigidity.tex", L)
    CT = json.load(open(os.path.join(FD.REVISION, "controls.json")))
    L = []
    for c in CT["straight"]:
        b, k = c["label"].split()
        L.append(f"{b.split('=')[1]} & {c['k']:.3f} & {c['gamma_over_k2']:+.3f} & {-c['gamma_over_k2'] * c['k'] ** 2:+.4f} & {c['beta']:.3f} \\\\")
    write(outdir, "tab_straight.tex", L)
    tw = MD["twist"]
    L = []
    for t in tw:
        m = t["m"]
        rel = t["rel"][f"tw{m}c"]
        amp = os.path.basename(t["run"]).split("_a")[1].split("_")[0]          # seed amplitude delta of the mode run
        L.append(f"{float(amp):.2f} & {m} & {rel['amp_over_r']:.3f} & "
                 f"{rel['phase_deg']:+.0f} & {t['rel'][f'tw{m}s']['amp_over_r']:.3f} & {t['tw_rms_median']:.4f} & {t['tw_net_max']:.3f} \\\\")
    write(outdir, "tab_twist.tex", L)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1])
