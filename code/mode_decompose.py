"""Fit-free decomposition of the RD shape modes at the stable radius.

From each run (modeR_m{2,3}_a{0.08,0.15}.npz) we fit damped sinusoids to the signed
cos(m theta) parts r_mc(t), zeta_mc(t) and obtain the frequency omega and the amplitude
ratio |zeta|/|r|. The corrected linear theory (mode_theory2.py) gives
    W = omega R^2 / (m sqrt(m^2-1)) = sqrt(beta * beta_eff)
    Q = (|zeta|/|r|) / sqrt((m^2-1)/m^2) = sqrt(beta_eff / beta)
so that beta = W/Q (the coefficient the vertical->radial coupling sees) and
beta_eff = W*Q (the radial->vertical coupling) follow without any fitted beta(K) curve.
"""
import numpy as np
from scipy.ndimage import median_filter
import ellipse_compare as C
from mode_theory2 import fit_mode

PERIOD = 4.44
rows = []
for m, t1 in ((2, 125.0), (3, 45.0)):
    for amp in (0.08, 0.15):
        d = np.load(f"modeR_m{m}_a{amp}.npz")
        ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"])
        t = d["t"][ok]
        sig = {}
        for k in (f"r{m}c", f"z{m}c", "Rm", "z"):
            y = d[k][ok]
            y = np.where(np.abs(y - median_filter(y, 15, mode="nearest")) > 0.4, np.nan, y)  # wrong-curve frames
            good = np.isfinite(y)
            sig[k] = C.smooth(t[good], y[good], PERIOD), good
        tt = t[sig[f"r{m}c"][1] & sig[f"z{m}c"][1]]
        r = np.interp(tt, t[sig[f"r{m}c"][1]], sig[f"r{m}c"][0])
        z = np.interp(tt, t[sig[f"z{m}c"][1]], sig[f"z{m}c"][0])
        sel = (tt >= 6) & (tt <= t1) & np.isfinite(r) & np.isfinite(z)
        R = np.nanmean(sig["Rm"][0][(t[sig["Rm"][1]] >= 6) & (t[sig["Rm"][1]] <= t1)])
        w0 = m * np.sqrt(m * m - 1) * 1.2 / R ** 2
        f = fit_mode(tt[sel], r[sel], z[sel], w0)
        # drift coefficient of the same ring
        zz, gz = sig["z"]; RR, gR = sig["Rm"]
        tz = t[gz]; mz = (tz >= 6) & (tz <= t1) & np.isfinite(zz)
        bdrift = abs(np.polyfit(tz[mz], zz[mz], 1)[0]) * R
        W = f["omega"] * R ** 2 / (m * np.sqrt(m * m - 1))
        Q = f["ratio"] / np.sqrt((m * m - 1) / m ** 2)
        rows.append((m, amp, R, f["omega"], f["ratio"], W, Q, W / Q, W * Q, bdrift, f["gamma"], f["rms"]))
        print(f"m={m} amp={amp}: R={R:.2f} omega={f['omega']:.4f} |z|/|r|={f['ratio']:.3f} (const-beta {np.sqrt((m*m-1)/m**2):.3f})"
              f"  W=sqrt(b*b_eff)={W:.3f}  Q=sqrt(b_eff/b)={Q:.3f}  -> beta={W/Q:.3f} beta_eff={W*Q:.3f}"
              f"  beta_drift={bdrift:.3f}  gamma={f['gamma']:.4f}  fit rms={f['rms']:.3f}")
rows = np.array(rows)
np.save("mode_decompose.npy", rows)
print()
for m in (2, 3):
    rr = rows[rows[:, 0] == m]
    print(f"m={m}: beta (vertical->radial) = {rr[:,7].mean():.3f}, beta_eff = {rr[:,8].mean():.3f}, "
          f"beta_eff/beta = {(rr[:,8]/rr[:,7]).mean():.3f}, beta_drift = {rr[:,9].mean():.3f}")
