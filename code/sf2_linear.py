"""Zero-amplitude extrapolation of straight-filament rates (sf2/, A = 0.5, Lxy = 64).

Local rates in 10-unit windows after the formation transient (window start >= 12) are fitted as
gamma/k^2 = g0 + c (k|u_k|)^2 and |beta| = b0 + d (k|u_k|)^2; g0, b0 are the linear (small-amplitude)
coefficients. Fits use windows while k|u_k| < 0.45. k = 0.79 decays: its rate comes from a
direct log-linear fit with a lower amplitude floor.

Uncertainties: the 10-unit windows are 2.5 apart and overlap by 75%, so regression covariance
overstates the precision. Windows four steps apart do not overlap; the intercepts are fitted
separately on the four interleaved non-overlapping subsets, and half their spread is reported
(a descriptive uncertainty of the extrapolation, not an independent-run error).
"""
import glob
import numpy as np
from sf2_analysis import csmooth, local_rates

rows = []
for f in sorted(glob.glob("sf2/L64_k0[1-9]*.npz")):
    d = np.load(f)
    t, k = d["t"], float(d["k"])
    s = csmooth(t, d["uk"])
    r = local_rates(t, s, width=10, step=2.5, floor=0.02)
    r = r[(r[:, 0] - 5 >= 12) & (k * r[:, 3] < 0.45)]
    x = (k * r[:, 3]) ** 2
    G = r[:, 1] / k**2
    Bt = np.abs(r[:, 2]) / k**2
    if len(r) >= 3 and np.ptp(x) > 1e-3:
        g0 = np.polyfit(x, G, 1)[1]
        b0 = np.polyfit(x, Bt, 1)[1]
        sub_g, sub_b = [], []
        for j in range(4):                      # interleaved subsets of non-overlapping windows
            xs, Gs, Bs = x[j::4], G[j::4], Bt[j::4]
            if len(xs) >= 2 and np.ptp(xs) > 1e-3:
                sub_g.append(np.polyfit(xs, Gs, 1)[1]); sub_b.append(np.polyfit(xs, Bs, 1)[1])
        eg = 0.5 * np.ptp(sub_g) if len(sub_g) >= 2 else np.nan
        eb = 0.5 * np.ptp(sub_b) if len(sub_b) >= 2 else np.nan
    else:
        g0, eg, b0, eb = G.mean(), G.std(), Bt.mean(), Bt.std()
    rows.append((k, g0, eg, b0, eb, len(r), (k * r[:, 3]).min(), (k * r[:, 3]).max()))
    print(f"k={k:.3f}  linear gamma/k^2 = {g0:+.3f} +- {eg:.3f}   |beta| = {b0:.3f} +- {eb:.3f}"
          f"   ({len(r)} windows, kA {rows[-1][6]:.2f}-{rows[-1][7]:.2f})")
R = np.array(rows)
p = np.polyfit(R[:, 0] ** 2, R[:, 1], 1, w=1 / np.maximum(np.nan_to_num(R[:, 2], nan=0.01), 0.01))
print(f"fit gamma/k^2 = alpha0 + eta k^2 (all k): alpha0 = {p[1]:.3f}, eta = {p[0]:.3f}")
for drop in range(len(R)):
    m = np.arange(len(R)) != drop
    q = np.polyfit(R[m, 0] ** 2, R[m, 1], 1)
    print(f"  without k={R[drop,0]:.2f}: alpha0 = {q[1]:.3f}, eta = {q[0]:.3f}")
np.save("sf2/sf2_linear.npy", R)
