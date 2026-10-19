"""Tables S2 and S4 (and Figs. S1c, S2c): half-cycles of the Panfilov and Atri elliptic rings.

Circular rings give alpha and beta (pan_alpha.windows, atri_alpha.fit; single-curve frames);
halfcycle.half_cycles gives the observed and predicted half-periods. Writes
halfcycles_panfilov.csv, halfcycles_atri.csv and coefficients_circles.csv into the data
directory and prints the per-run summaries (mean +- sd over half-cycles).
    python halfcycle_tables.py [datadir]          (default v2)
"""
import os
import sys
import csv
import numpy as np
import halfcycle as H
import pan_alpha as PA
import atri_alpha as AA

RADIUS = os.environ.get("HC_RADIUS", "Rm")   # "Rm" (default) or "L"; see halfcycle.py
ATRI_PERIOD = 16.58          # rotation period of the Atri spiral at P = 0.1 uM (tip angle)

# ellipse file, g_f, dx, domain, seed, circle file for alpha and beta
PAN = [
    ("pell_gf0.775_20x12.npz", 0.775, 0.6, "80^2 x 50", "20 x 12", "pcirc_gf0.775_R14.npz"),
    ("pell_gf0.8_24x15.npz", 0.8, 0.6, "88^2 x 50", "24 x 15", "pcirc_gf0.8_R14.npz"),
    ("pellBIG_gf0.8_24x15.npz", 0.8, 0.6, "120^2 x 80", "24 x 15", "pcirc_gf0.8_R14.npz"),
    ("pellHR_gf0.8_24x15.npz", 0.8, 0.4, "88^2 x 50", "24 x 15", "pcircHR_gf0.8_R14.npz"),
    ("pellXR_gf0.8_24x15.npz", 0.8, 0.3, "76^2 x 40", "24 x 15", "pcircXR_gf0.8_R14.npz"),
]
ATRI = [
    ("aell2_P0.1_100x60.npz", 3.0, "600^2 x 280", "100 x 60", "acirc4_P0.1_c052.npz"),
    ("aell2_P0.1_130x80.npz", 3.0, "660^2 x 280", "130 x 80", "acirc4_P0.1_c052.npz"),
    ("aell_dx2_P0.1_100x60.npz", 2.0, "600^2 x 280", "100 x 60", "acirc_dx2_P0.1.npz"),
    ("aellW_P0.1_100x88.npz", 3.0, "600^2 x 280", "100 x 88", "acirc4_P0.1_c052.npz"),
    ("aellW_P0.1_120x105.npz", 3.0, "660^2 x 280", "120 x 105", "acirc4_P0.1_c052.npz"),
]


def pan_coeffs(path):
    _, w = PA.windows(path, single=True)
    return w[:, 2].mean(), w[:, 2].std(), abs(w[:, 3].mean()), w[:, 3].std(), len(w)


def atri_coeffs(path):
    rows, _, _ = AA.fit(path, ATRI_PERIOD)
    return rows[:, 2].mean(), rows[:, 2].std(), abs(rows[:, 3].mean()), rows[:, 3].std(), len(rows)


def write(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def summary(label, rows):
    q = np.array([r["ratio"] for r in rows])
    ql = np.array([r["ratio_linear"] for r in rows])
    sd = f"+- {q.std():.3f}" if len(q) > 1 else "        "
    print(f"  {label}  ratio {q.mean():.3f} {sd}  n={len(q)}  (linear only {ql.mean():.3f})"
          f"   per half-cycle {np.round(q, 3).tolist()}")


def main(D):
    p = lambda f: os.path.join(D, f)
    coeffs, out = [], []
    print("Panfilov-Rudenko:")
    for fn, gf, dx, dom, seed, circ in PAN:
        if not os.path.exists(p(fn)):
            print("  missing", fn); continue
        al, sal, be, sbe, nw = pan_coeffs(p(circ))
        coeffs.append(dict(model="panfilov", file=circ, gf=gf, dx=dx, alpha=al, alpha_sd=sal, beta=be, beta_sd=sbe, windows=nw))
        rows, S = H.half_cycles(p(fn), PA.PERIOD[gf], be, al, RADIUS)
        summary(f"gf {gf:<5} dx {dx} {dom:11s} seed {seed} |beta| {be:.4f} alpha {al:+.3f} used {S['frac_used']:.2f}", rows)
        out += [dict(file=fn, gf=gf, dx=dx, circle=circ, beta=be, alpha=al, **r) for r in rows]
    write(p("halfcycles_panfilov.csv"), out)
    out = []
    print("Atri (P = 0.1 uM):")
    for fn, dx, dom, seed, circ in ATRI:
        if not os.path.exists(p(fn)):
            print("  missing", fn); continue
        al, sal, be, sbe, nw = atri_coeffs(p(circ))
        coeffs.append(dict(model="atri", file=circ, gf=0.1, dx=dx, alpha=al, alpha_sd=sal, beta=be, beta_sd=sbe, windows=nw))
        rows, S = H.half_cycles(p(fn), ATRI_PERIOD, be, al, RADIUS)
        summary(f"dx {dx} {dom} seed {seed} |beta| {be:.2f} alpha {al:+.2f} used {S['frac_used']:.2f}", rows)
        out += [dict(file=fn, dx=dx, circle=circ, beta=be, alpha=al, **r) for r in rows]
    write(p("halfcycles_atri.csv"), out)
    write(p("coefficients_circles.csv"), coeffs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
