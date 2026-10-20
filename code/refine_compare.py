"""Grid refinement of the mode couplings and of the straight-filament dispersion (re-audit 3).

Applies the production analysis (mode_decompose.analyze; the zero-amplitude extrapolation of
sf2_linear.py) to the runs at dx = 0.2 and dx = 0.15 and compares the couplings
beta_zeta/beta_drift and beta_r/beta_drift of the m = 2, 3 modes at R*, and the straight-filament
coefficient |beta| = omega/k^2 at k = 0.349 (box 32).
    python refine_compare.py COARSE FINE       (folders with modeR_*.npz and sf2/L32_k035*.npz)
Writes FINE/refine_compare.json.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def modes(folder):
    import mode_decompose as MD
    cwd = os.getcwd(); os.chdir(folder)
    try:
        out = {}
        for m in (2, 3):
            rows = []
            for amp in (0.08, 0.15):
                for fr in (0.6, 0.8, 1.0):
                    for sm in (1, 2):
                        r = MD.analyze(m, amp, MD.T1[m] * fr, sm)
                        if r["ok"]:
                            rows.append((r["bz"] / r["bdrift"], r["br"] / r["bdrift"], r["bz"], r["br"], r["bdrift"], r["R"]))
            a = np.array(rows)
            main = [MD.analyze(m, amp, MD.T1[m]) for amp in (0.08, 0.15)]
            out[m] = dict(beta_zeta=np.mean([r["bz"] for r in main]), beta_r=np.mean([r["br"] for r in main]),
                          beta_drift=np.mean([r["bdrift"] for r in main]), R=np.mean([r["R"] for r in main]),
                          bz_over_drift_range=[a[:, 0].min(), a[:, 0].max()], br_over_drift_range=[a[:, 1].min(), a[:, 1].max()])
        return out
    finally:
        os.chdir(cwd)


def straight(path):
    from sf2_analysis import csmooth, local_rates
    d = np.load(path)
    t, k = d["t"], float(d["k"])
    s = csmooth(t, d["uk"])
    r = local_rates(t, s, width=10, step=2.5, floor=0.02)
    r = r[(r[:, 0] - 5 >= 12) & (k * r[:, 3] < 0.45)]
    x = (k * r[:, 3]) ** 2
    G, Bt = r[:, 1] / k ** 2, np.abs(r[:, 2]) / k ** 2
    sub = [np.polyfit(x[j::4], Bt[j::4], 1)[1] for j in range(4) if len(x[j::4]) >= 2 and np.ptp(x[j::4]) > 1e-3]
    return dict(k=k, beta=float(np.polyfit(x, Bt, 1)[1]), beta_err=float(0.5 * np.ptp(sub)) if len(sub) > 1 else None,
                gamma_over_k2=float(np.polyfit(x, G, 1)[1]), windows=int(len(r)))


def main(coarse, fine):
    res = {"modes": {}, "straight": {}}
    mc, mf = modes(coarse), modes(fine)
    for m in (2, 3):
        res["modes"][f"m{m}"] = {"dx0.2": mc[m], "dx0.15": mf[m]}
        c, f = mc[m], mf[m]
        print(f"m={m}: R* {c['R']:.2f} -> {f['R']:.2f}   beta_drift {c['beta_drift']:.3f} -> {f['beta_drift']:.3f}   "
              f"beta_zeta {c['beta_zeta']:.3f} -> {f['beta_zeta']:.3f}   beta_r {c['beta_r']:.3f} -> {f['beta_r']:.3f}")
        print(f"      beta_zeta/beta_drift {np.round(c['bz_over_drift_range'], 2)} -> {np.round(f['bz_over_drift_range'], 2)}   "
              f"beta_r/beta_drift {np.round(c['br_over_drift_range'], 2)} -> {np.round(f['br_over_drift_range'], 2)}")
    sc = straight(os.path.join(coarse, "sf2", "L32_k035.npz"))
    sf = straight(os.path.join(fine, "sf2", "L32_k035_dx015.npz"))
    res["straight"] = {"dx0.2": sc, "dx0.15": sf}
    print(f"straight k={sc['k']:.3f}/{sf['k']:.3f}: |beta| {sc['beta']:.3f} -> {sf['beta']:.3f} "
          f"({100 * (sf['beta'] / sc['beta'] - 1):+.1f}%), gamma/k^2 {sc['gamma_over_k2']:+.3f} -> {sf['gamma_over_k2']:+.3f}")
    conv = lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else o
    with open(os.path.join(fine, "refine_compare.json"), "w", newline=chr(10)) as fh:
        json.dump(res, fh, indent=1, default=conv)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
