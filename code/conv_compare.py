"""Resolution (dx = 0.1) and box-size (60 x 60 x 30) checks of the elliptic rings at a = 1.1, b = 0.17.

Compares switch times (sign changes of r_2) and observed/predicted half-period ratios with the
production runs (dx = 0.2; 28 x 28 x 16 and 40 x 40 x 20), using period_analysis_r2.intervals_r2.
"""
import numpy as np
import period_analysis as PA
from period_analysis_r2 import intervals_r2

PAIRS = [("8x5: dx 0.2 vs 0.1", "rerun_r2/ell_a1.1_b0.17_8x5.npz", "conv/ell_8x5_dx0.1.npz"),
         ("12x7.5: box 40x40x20 vs 60x60x30", "rerun_r2/ell_a1.1_b0.17_12x7.5.npz", "conv/ell_12x7.5_bigbox.npz")]

fac = PA.amplitude_factor()
for lab, ref, new in PAIRS:
    out = []
    for path in (ref, new):
        rows = intervals_r2(path, 4.44)
        out.append([(r["t0"], r["t1"], r["Th"], r["R"], r["Th"] / PA.predict(r, fac)[2], r["beta"]) for r in rows])
    a, b = (np.array(o) for o in out)
    print(f"\n{lab}")
    n = min(len(a), len(b))
    for i in range(n):
        print(f"  half-cycle {i+1}: t0 {a[i,0]:6.1f} vs {b[i,0]:6.1f}   T1/2 {a[i,2]:5.2f} vs {b[i,2]:5.2f}   "
              f"R {a[i,3]:.2f} vs {b[i,3]:.2f}   ratio {a[i,4]:.3f} vs {b[i,4]:.3f}")
    print(f"  mean ratio {a[:n,4].mean():.3f} vs {b[:n,4].mean():.3f}  (difference {100*(b[:n,4].mean()/a[:n,4].mean()-1):+.1f}%)")
