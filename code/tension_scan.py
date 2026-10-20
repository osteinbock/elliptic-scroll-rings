"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Filament tension alpha and binormal coefficient beta from circular scroll
rings (Barkley, periodic box): R^2 = R0^2 - 2 alpha t, dz/dt = beta / R."""
import sys
import numpy as np
import ellipse_ring as E

def fit(rows, tmin=4.0):
    d = E.table(rows)
    ok = (d["t"] > tmin) & np.isfinite(d["Rm"])
    t, R, z = d["t"][ok], d["Rm"][ok], d["z"][ok]
    if len(t) < 10:
        return np.nan, np.nan, d
    alpha = -0.5 * np.polyfit(t, R ** 2, 1)[0]
    # z vs  integral dt/R  -> slope beta
    s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[1:] + 1 / R[:-1]) * np.diff(t))])
    beta = np.polyfit(s, z, 1)[0]
    return alpha, beta, d

if __name__ == "__main__":
    a = float(sys.argv[1]); bs = [float(x) for x in sys.argv[2].split(",")]
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 30
    for b in bs:
        rows, _ = E.run(a, b, 0.02, 6, 6, 22, 14, 0.2, T, every_t=0.25, verbose=False)
        al, be, d = fit(rows)
        ok = np.isfinite(d["Rm"])
        print(f"a={a} b={b}: alpha={al:+.3f} beta={be:+.3f}  R: {d['Rm'][ok][min(8,ok.sum()-1)]:.2f} -> {d['Rm'][ok][-1]:.2f} "
              f"at t={d['t'][ok][-1]:.1f}  ncurves(max)={np.nanmax(d['ncurves']):.0f}", flush=True)
