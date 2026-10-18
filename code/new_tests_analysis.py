"""Analysis of the round-2 test simulations (Barkley a=1.1 b=0.17 eps=0.02).

1. Straight filaments with a sinusoidal deformation (sf_k*.npz): decay rate gamma and
   precession rate omega of the k-component -> alpha(k) = gamma/k^2, |beta(k)| = omega/k^2;
   fit gamma/k^2 = alpha_0 + eta k^2 across k.
2. Tilted ring (tilt_m1.npz): beta_zeta(m=1) = R^2 (dc_y/dt) / zeta_1 vs beta_drift = R dz/dt.
3. Larger rings with m = 2, 3 deformations (modeL_m{2,3}.npz): sliding damped-sinusoid fits
   -> local R, sqrt(beta_zeta beta_r), amplitude ratio, beta_drift.
Writes new_tests.npz with all results.
"""
import glob
import numpy as np
from scipy.ndimage import median_filter
import ellipse_compare as C
from mode_theory2 import fit_mode

PERIOD = 4.44


def csmooth(t, z):
    return C.smooth(t, z.real, PERIOD) + 1j * C.smooth(t, z.imag, PERIOD)


def straight():
    res = []
    for f in sorted(glob.glob("sf_k0*.npz")):
        d = np.load(f)
        t, uk, k = d["t"], d["uk"], float(d["k"])
        s = csmooth(t, uk)
        ok = (t >= 8) & np.isfinite(s) & (np.abs(s) > 0.08)
        # stop at the first time the amplitude falls below the noise floor
        if (~ok & (t >= 8)).any():
            ok &= t < t[(t >= 8) & ~ok][0]
        g = -np.polyfit(t[ok], np.log(np.abs(s[ok])), 1)[0]
        w = np.polyfit(t[ok], np.unwrap(np.angle(s[ok])), 1)[0]
        res.append((k, g, w, g / k ** 2, abs(w) / k ** 2, t[ok][0], t[ok][-1], np.abs(s[ok][0])))
        print(f"{f}: k={k:.3f}  fit t={t[ok][0]:.0f}-{t[ok][-1]:.0f}  gamma={g:.4f}  omega={w:+.4f}"
              f"  alpha(k)=gamma/k^2={g/k**2:.3f}  |beta(k)|=omega/k^2={abs(w)/k**2:.3f}")
    res = np.array(res)
    if len(res) >= 2:
        p = np.polyfit(res[:, 0] ** 2, res[:, 3], 1)
        print(f"  fit gamma/k^2 = alpha_0 + eta k^2: alpha_0 = {p[1]:.3f}, eta = {p[0]:.3f}")
    return res


def tilt():
    d = np.load("tilt_m1.npz")
    ok = np.isfinite(d["Rm"]) & np.isfinite(d["z1s"])
    t = d["t"][ok]
    S = {k: C.smooth(t, d[k][ok], PERIOD) for k in ("Rm", "cy", "cx", "z", "z1s", "z1c")}
    m = (t >= 8) & np.isfinite(S["Rm"])
    R = S["Rm"][m].mean()
    vy = np.polyfit(t[m], S["cy"][m], 1)[0]
    vx = np.polyfit(t[m], S["cx"][m], 1)[0]
    vz = np.polyfit(t[m], S["z"][m], 1)[0]
    z1 = np.hypot(S["z1s"][m], S["z1c"][m]).mean()
    vh = np.hypot(vx, vy)
    beta_zeta = R ** 2 * vh / z1
    beta_drift = R * abs(vz)
    print(f"tilt: R={R:.2f}  zeta_1={z1:.3f} (tilt {np.degrees(np.arctan(z1/R)):.1f} deg)  v_horizontal={vh:.4f}"
          f"  v_axial={vz:+.4f}  beta_zeta(m=1)={beta_zeta:.3f}  beta_drift={beta_drift:.3f}  ratio={beta_zeta/beta_drift:.3f}")
    return np.array([R, z1, vh, vz, beta_zeta, beta_drift])


def large_modes():
    out = []
    for m, f in ((2, "modeL_m2.npz"), (3, "modeL_m3.npz")):
        d = np.load(f)
        ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"])
        t = d["t"][ok]
        sig = {}
        for key in (f"r{m}c", f"z{m}c", "Rm", "z"):
            y = d[key][ok]
            y = np.where(np.abs(y - median_filter(y, 15, mode="nearest")) > 0.4, np.nan, y)
            g = np.isfinite(y)
            sig[key] = np.interp(t, t[g], C.smooth(t[g], y[g], PERIOD))
        R_all = sig["Rm"]
        for tc in np.arange(15, t[-1] - 15, 10.0):
            Rloc = np.nanmean(R_all[np.abs(t - tc) < 5])
            w0 = m * np.sqrt(m * m - 1) * 1.2 / Rloc ** 2
            span = 0.75 * 2 * np.pi / w0
            sel = (np.abs(t - tc) <= span) & np.isfinite(sig[f"r{m}c"]) & np.isfinite(sig[f"z{m}c"])
            if sel.sum() < 40 or t[sel][0] < 6:
                continue
            fm = fit_mode(t[sel], sig[f"r{m}c"][sel], sig[f"z{m}c"][sel], w0)
            Rw = R_all[sel].mean()
            W = fm["omega"] * Rw ** 2 / (m * np.sqrt(m * m - 1))
            Q = fm["ratio"] / np.sqrt((m * m - 1) / m ** 2)
            zz = sig["z"][sel]
            bd = abs(np.polyfit(t[sel], zz, 1)[0]) * Rw
            amp = np.hypot(sig[f"r{m}c"][sel], sig[f"z{m}c"][sel]).max()
            out.append((m, tc, Rw, m / Rw, W, Q, W / Q, W * Q, bd, fm["gamma"], amp / Rw))
            print(f"m={m} t~{tc:.0f}: R={Rw:.2f} k={m/Rw:.3f} sqrt(bz*br)={W:.3f} Q=sqrt(br/bz)={Q:.3f}"
                  f" beta_zeta={W/Q:.3f} beta_r={W*Q:.3f} beta_drift={bd:.3f} ratio={W/bd:.3f} rel.amp={amp/Rw:.3f}")
    return np.array(out)


if __name__ == "__main__":
    sf = straight()
    tl = tilt()
    lm = large_modes()
    np.savez("new_tests.npz", straight=sf, tilt=tl, large=lm)
    print("wrote new_tests.npz")
