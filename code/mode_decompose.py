"""Fit-free decomposition of the RD shape modes at the stable radius.

From each run (modeR_m{2,3}_a{0.08,0.15}.npz) we fit damped sinusoids to the signed
cos(m theta) parts r_mc(t), zeta_mc(t) and obtain the frequency omega and the amplitude
ratio |zeta|/|r|. The corrected linear theory (mode_theory2.py) gives
    W = omega R^2 / (m sqrt(m^2-1)) = sqrt(beta * beta_eff)
    Q = (|zeta|/|r|) / sqrt((m^2-1)/m^2) = sqrt(beta_eff / beta)
so that beta = W/Q (the coefficient the vertical->radial coupling sees) and
beta_eff = W*Q (the radial->vertical coupling) follow without any fitted beta(K) curve.

Frames without a single closed filament are nan (mask_frames.py) and are bridged by the
period average. The main values use the window 6 <= t <= t1 (t1 = 125 for m = 2, 45 for
m = 3) and one rotation period of smoothing; the sensitivity sweep varies the window end
(t1 x 0.6, 0.8, 1) and the smoothing (1 or 2 periods) and is saved as mode_sweep.npy.
"""
import numpy as np
import ellipse_compare as C
from mode_theory2 import fit_mode

PERIOD = 4.44


def quad_dev(dphase_deg):
    """Departure from quadrature: with B > 0, d = 0 and d = 180 deg are both quadrature."""
    a = np.abs(np.asarray(dphase_deg)) % 180.0
    return np.minimum(a, 180.0 - a)


T1 = {2: 125.0, 3: 45.0}


def analyze(m, amp, t1, smooth=1):
    d = np.load(f"modeR_m{m}_a{amp}.npz")
    if "masked" not in d.files:
        raise RuntimeError("run mask_frames.py first")
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    s = {k: C.smooth(t, d[k][ok], smooth * PERIOD) for k in (f"r{m}c", f"z{m}c", "Rm", "z")}
    sel = (t >= 6) & (t <= t1) & np.isfinite(s[f"r{m}c"]) & np.isfinite(s[f"z{m}c"])
    R = np.nanmean(s["Rm"][sel])
    w0 = m * np.sqrt(m * m - 1) * 1.2 / R ** 2
    f = fit_mode(t[sel], s[f"r{m}c"][sel], s[f"z{m}c"][sel], w0)
    mz = sel & np.isfinite(s["z"])
    bdrift = abs(np.polyfit(t[mz], s["z"][mz], 1)[0]) * R
    W = f["omega"] * R ** 2 / (m * np.sqrt(m * m - 1))
    Q = f["ratio"] / np.sqrt((m * m - 1) / m ** 2)
    return dict(m=m, amp=amp, R=R, omega=f["omega"], ratio=f["ratio"], W=W, Q=Q, bz=W / Q, br=W * Q,
                bdrift=bdrift, gamma=f["gamma"], rms=f["rms"], dphase=np.degrees(f["dphase"]),
                ok=f["success"] and not f["at_bound"], t1=t1, smooth=smooth)


if __name__ == "__main__":
    rows = []
    for m in (2, 3):
        for amp in (0.08, 0.15):
            r = analyze(m, amp, T1[m])
            rows.append([r[k] for k in ("m", "amp", "R", "omega", "ratio", "W", "Q", "bz", "br", "bdrift", "gamma", "rms")])
            print(f"m={m} amp={amp}: R={r['R']:.2f} omega={r['omega']:.4f} |z|/|r|={r['ratio']:.3f} "
                  f"(const-beta {np.sqrt((m*m-1)/m**2):.3f})  -> beta_zeta={r['bz']:.3f} beta_r={r['br']:.3f}  "
                  f"beta_drift={r['bdrift']:.3f}  gamma={r['gamma']:.4f}  off quadrature {quad_dev(r['dphase']):.1f} deg  "
                  f"fit ok={r['ok']} rms={r['rms']:.3f}")
    rows = np.array(rows)
    np.save("mode_decompose.npy", rows)
    print()
    for m in (2, 3):
        rr = rows[rows[:, 0] == m]
        print(f"m={m}: beta_zeta = {rr[:,7].mean():.3f}, beta_r = {rr[:,8].mean():.3f}, "
              f"beta_r/beta_zeta = {(rr[:,8]/rr[:,7]).mean():.3f}, beta_drift = {rr[:,9].mean():.3f}, "
              f"omega(amp .08 / .15) = {rr[0,3]:.4f} / {rr[1,3]:.4f} ({100*abs(rr[1,3]/rr[0,3]-1):.1f}%)")
    # sensitivity to fitting window and smoothing
    sweep = []
    for m in (2, 3):
        for amp in (0.08, 0.15):
            for fr in (0.6, 0.8, 1.0):
                for sm in (1, 2):
                    r = analyze(m, amp, T1[m] * fr, sm)
                    sweep.append([m, amp, fr, sm, r["bz"] / r["bdrift"], r["br"] / r["bdrift"], r["dphase"], r["ok"]])
    sweep = np.array(sweep, dtype=float)
    np.save("mode_sweep.npy", sweep)
    print("\nsweep (window end x 0.6/0.8/1, smoothing 1/2 periods; converged fits only):")
    for m in (2, 3):
        s = sweep[(sweep[:, 0] == m) & (sweep[:, 7] == 1)]
        print(f"  m={m}: beta_zeta/beta_drift {s[:,4].min():.2f}-{s[:,4].max():.2f}   "
              f"beta_r/beta_drift {s[:,5].min():.2f}-{s[:,5].max():.2f}   departure from quadrature max "
              f"{np.max(quad_dev(s[:,6])):.1f} deg   (n={len(s)})")
