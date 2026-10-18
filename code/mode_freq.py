"""Frequencies of m = 2 and m = 3 shape modes of scroll rings near the stable radius
(modeR_m{2,3}_a{0.08,0.15}.npz, Barkley a=1.1 b=0.17 eps=0.02).

beta_eff(m) = omega_m Rbar^2 / (m sqrt(m^2-1)) is the binormal coefficient the mode
'sees'; beta_drift = Rbar dz/dt is the one the circular translation sees. With
constant or local beta(K), beta_eff is the same for m = 2 and 3 (omega3/omega2 = 2.449
at equal R); an m-dependent beta_eff points to a dispersive higher-order binormal term.
"""
import numpy as np
import ellipse_compare as C
import mode_damping as MD

PERIOD = 4.44
rows = []
for m in (2, 3):
    for amp in (0.08, 0.15):
        path = f"modeR_m{m}_a{amp}.npz"
        r = MD.analyse(path, m, t_start=6.0, floor=0.008)
        if m == 3:
            r["t1"] = min(r["t1"], 45.0)          # m = 3 decays below the noise floor after ~2.5 cycles
        d = np.load(path)
        ok = np.isfinite(d["Rm"])
        t = d["t"][ok]
        R = C.smooth(t, d["Rm"][ok], PERIOD)
        z = C.smooth(t, d["z"][ok], PERIOD)
        sel = (t >= r["t0"]) & (t <= r["t1"]) & np.isfinite(R) & np.isfinite(z)
        s = np.concatenate([[0], np.cumsum(0.5 * (1 / R[sel][1:] + 1 / R[sel][:-1]) * np.diff(t[sel]))])
        bdrift = abs(np.polyfit(s, z[sel], 1)[0])
        # frequency from zero crossings of the smoothed z_m cos-component (robust; the
        # phase of r_m + i c z_m is corrupted by single wrong-curve frames in r_m)
        zc = C.smooth(t, d[f"z{m}c"][ok], PERIOD)
        x = [c for c in C.crossings(t, zc) if r["t0"] <= c <= r["t1"]]
        r["omega"] = np.pi / np.mean(np.diff(x))
        beff = r["omega"] * r["Rmean"] ** 2 / (m * np.sqrt(m * m - 1))
        A0 = r["rel"][np.isfinite(r["rel"])][:20].mean()
        rows.append((m, amp, A0, r["Rmean"], r["omega"], beff, bdrift, r["gamma"], r["t0"], r["t1"]))
        print(f"m={m} amp={amp}: A0/R={A0:.3f} <R>={r['Rmean']:.2f} fit t={r['t0']:.0f}-{r['t1']:.0f}  "
              f"omega={r['omega']:.4f}  gamma={r['gamma']:.4f}  beta_eff={beff:.3f}  beta_drift={bdrift:.3f}  "
              f"ratio={beff / bdrift:.3f}")
rows = np.array(rows)
print()
for m in (2, 3):
    rr = rows[rows[:, 0] == m]
    # linear extrapolation of beta_eff/beta_drift in (initial relative amplitude)^2 to zero
    x, y = rr[:, 2] ** 2, rr[:, 5] / rr[:, 6]
    p = np.polyfit(x, y, 1)
    print(f"m={m}: beta_eff/beta_drift at amplitudes {np.round(rr[:, 2], 3)} = {np.round(y, 3)};  "
          f"extrapolated to zero amplitude: {p[1]:.3f}")
b2 = rows[rows[:, 0] == 2][:, 5].mean()
b3 = rows[rows[:, 0] == 3][:, 5].mean()
print(f"beta_eff(3)/beta_eff(2) = {b3 / b2:.3f}   (equal-coefficient prediction 1.000)")
