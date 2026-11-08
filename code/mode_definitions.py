"""Filament definition and induced twist in the shape-mode runs (revision: referee comments A4a-b, C5).

The runs modeR_m{2,3}_a{0.08,0.15}_lv.npz and modeL_m{2,3}_lv.npz of REVISION repeat the mode simulations of
BARKLEY bit for bit (same seeds; the standard columns are identical) and also record:
  * the ring measures for four other isosurface pairs (u*, v*) = (0.3, 0.16), (0.7, 0.60), (0.5, 0.28),
    (0.5, 0.48), against the standard (0.5, a/2 - b = 0.38) (columns L1_ ... L4_);
  * the rotation phase of the wave along the standard filament in a parallel-transported frame and its
    Fourier parts tw{m}c, tw{m}s in the polar angle (ellipse_ring.twist_phase).
For every definition the mode fit of mode_decompose.py (R*: time fit, window 6 <= t <= t1) or of
mode_refit.py (larger ring: fit in tau = int dt / R^2) is repeated, giving the frequency, the amplitude
ratio q, the couplings beta_zeta and beta_r, and the diagonal rates alpha_r and alpha_zeta. For the twist,
the m-th phase component is fitted with the frequency and decay rate of the deformation, giving its
amplitude relative to r_m and its phase. The m = 4 run (modeR_m4_a0.10_lv.npz) decays within about two
periods, so its couplings are also fitted over the windows M4_WINDOWS.

    python mode_definitions.py         (in REVISION; writes mode_definitions.json)
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ellipse_compare as C
from mode_theory2 import fit_mode
from mode_refit import tau_fit, diag_rates
import mode_decompose as MD

PERIOD = 4.44
T1 = {2: 125.0, 3: 45.0, 4: 30.0}
M4_WINDOWS = ((6.0, 25.0), (6.0, 30.0), (6.0, 35.0), (8.0, 30.0), (10.0, 40.0))   # (t0, t1) of the m = 4 range
DEFS = [("", "u=0.5, v=0.38 (standard)"), ("L1_", "u=0.3, v=0.16"), ("L2_", "u=0.7, v=0.60"),
        ("L3_", "u=0.5, v=0.28"), ("L4_", "u=0.5, v=0.48")]


def columns(d, pre, m):
    """Period-averaged r_mc, z_mc, R_m and z of one filament definition (single closed curves only)."""
    ok = np.isfinite(d[pre + "Rm"]) & np.isfinite(d[f"{pre}r{m}c"]) & (d[pre + "ncurves"] == 1) & (d[pre + "closed"] == 1)
    t = d["t"][ok]
    zkey = "z" if pre == "" else pre + "z"
    S = {k: C.smooth(t, d[pre + k if k != "z" else zkey][ok], PERIOD) for k in (f"r{m}c", f"z{m}c", "Rm", "z")}
    return t, S, ok.mean()


def time_fit(t, S, m, t0, t1):
    sel = (t >= t0) & (t <= t1) & np.isfinite(S[f"r{m}c"]) & np.isfinite(S[f"z{m}c"]) & np.isfinite(S["Rm"]) & np.isfinite(S["z"])
    R = S["Rm"][sel].mean()
    f = fit_mode(t[sel], S[f"r{m}c"][sel], S[f"z{m}c"][sel], m * np.sqrt(m * m - 1) * 1.2 / R ** 2)
    W = f["omega"] * R ** 2 / (m * np.sqrt(m * m - 1))
    Q = f["ratio"] / np.sqrt((m * m - 1) / m ** 2)
    bd = abs(np.polyfit(t[sel], S["z"][sel], 1)[0]) * R
    tq = np.tan(f["dphase"])
    return dict(R=float(R), omega=float(f["omega"]), q=float(f["ratio"]), gamma=float(f["gamma"]),
                dphase_deg=float(np.degrees(f["dphase"])), bz=float(W / Q), br=float(W * Q), bdrift=float(bd),
                alpha_r=float((f["gamma"] - f["omega"] * tq) * R ** 2 / (m * m - 1)),
                alpha_zeta=float((f["gamma"] + f["omega"] * tq) * R ** 2 / (m * m)),
                ok=bool(f["success"] and not f["at_bound"]), rms=float(f["rms"]))


def twist_fit(d, m, t0, t1, omega, gamma):
    """Fit tw_mc and tw_ms with the deformation's frequency and decay; amplitudes and phases relative to r_mc."""
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"tw{m}c"]) & (d["ncurves"] == 1)
    t = d["t"][ok]
    S = {k: C.smooth(t, d[k][ok], PERIOD) for k in (f"r{m}c", f"tw{m}c", f"tw{m}s", "tw_rms", "tw_net")}
    sel = (t >= t0) & (t <= t1) & np.isfinite(S[f"r{m}c"]) & np.isfinite(S[f"tw{m}c"])
    tt = t[sel] - t[sel][0]
    out = {}
    for key in (f"r{m}c", f"tw{m}c", f"tw{m}s"):
        y = S[key][sel]
        res = lambda p: p[0] * np.exp(-gamma * tt) * np.cos(omega * tt + p[1]) + p[2] - y
        best = min((least_squares(res, [np.ptp(y) / 2, p0, y.mean()]) for p0 in np.linspace(0, 2 * np.pi, 6, endpoint=False)),
                   key=lambda s: s.cost)
        A, ph, c = best.x
        if A < 0:
            A, ph = -A, ph + np.pi
        out[key] = (float(A), float(ph), float(np.sqrt(np.mean(best.fun ** 2))), float(np.ptp(y)))
    Ar, pr = out[f"r{m}c"][:2]
    rel = {k: dict(amp_over_r=v[0] / Ar, phase_deg=float(np.degrees((v[1] - pr + np.pi) % (2 * np.pi) - np.pi)),
                   rms=v[2], ptp=v[3]) for k, v in out.items() if k != f"r{m}c"}
    w = S["tw_rms"][sel]
    return dict(rel=rel, tw_rms_median=float(np.nanmedian(w)), tw_net_max=float(np.nanmax(np.abs(S["tw_net"][sel]))))


def main():
    out = {"rstar": [], "large": [], "twist": []}
    for m, amp in ((2, 0.08), (2, 0.15), (3, 0.08), (3, 0.15), (4, 0.10)):
        path = f"modeR_m{m}_a{amp:.2f}_lv.npz" if m == 4 else f"modeR_m{m}_a{amp}_lv.npz"
        d = np.load(path)
        print(f"{path}:")
        ref = None
        for pre, lab in DEFS:
            t, S, frac = columns(d, pre, m)
            r = time_fit(t, S, m, 6.0, T1[m])
            r.update(m=m, amp=amp, definition=lab, frac_single_closed=float(frac))
            out["rstar"].append(r)
            if pre == "":
                ref = r
            print(f"  {lab:26s} R={r['R']:.3f} omega={r['omega']:.4f} q={r['q']:.4f} bz/bd={r['bz'] / r['bdrift']:.3f} "
                  f"br/bd={r['br'] / r['bdrift']:.3f} alpha_r={r['alpha_r']:+.3f} alpha_zeta={r['alpha_zeta']:+.3f} "
                  f"decay {r['gamma']:+.4f} phase {r['dphase_deg']:+.1f}  (single closed {frac:.3f}, ok {r['ok']})")
        tw = twist_fit(d, m, 6.0, T1[m], ref["omega"], ref["gamma"])
        tw.update(run=path, m=m)
        out["twist"].append(tw)
        print(f"  twist: |tw{m}c|/|r{m}c| = {tw['rel'][f'tw{m}c']['amp_over_r']:.3f} rad per unit (phase {tw['rel'][f'tw{m}c']['phase_deg']:+.0f} deg"
              f" from r{m}c), |tw{m}s|/|r{m}c| = {tw['rel'][f'tw{m}s']['amp_over_r']:.3f}; median rms local twist {tw['tw_rms_median']:.4f} rad/unit;"
              f" max |net twist| {tw['tw_net_max']:.3f} rad")
    for m, t1 in ((2, 150.0), (3, 100.0)):
        d = np.load(f"modeL_m{m}_lv.npz")
        print(f"modeL_m{m}_lv.npz (fits in tau, 15 <= t <= {t1:.0f}):")
        for pre, lab in DEFS:
            t, S, frac = columns(d, pre, m)
            r = tau_fit(t, S[f"r{m}c"], S[f"z{m}c"], S["Rm"], S["z"], m, 15.0, t1)
            r.update(definition=lab, frac_single_closed=float(frac))
            out["large"].append(r)
            print(f"  {lab:26s} R={r['R']:.3f} q={r['Q'] * np.sqrt((m * m - 1) / m ** 2):.4f} bz/bd={r['bz'] / r['bdrift']:.3f} "
                  f"br/bd={r['br'] / r['bdrift']:.3f} sqrt/bd={r['W'] / r['bdrift']:.3f} alpha_r={r['alpha_r']:+.3f} "
                  f"alpha_zeta={r['alpha_zeta']:+.3f}  (single closed {frac:.3f}, ok {r['ok']})")
    # m = 4 decays within about two periods: its range also covers the fit window (fits as mode_decompose.analyze)
    out["m4_windows"] = []
    for t0, t1 in M4_WINDOWS:
        r = MD.analyze(4, 0.10, t1, 1, path="modeR_m4_a0.10_lv.npz", t0=t0)
        ar, az = diag_rates(4, r["R"], r["omega"], r["gamma"], np.radians(r["dphase"]))
        out["m4_windows"].append(dict(t0=t0, t1=t1, bz_over_drift=float(r["bz"] / r["bdrift"]),
                                     br_over_drift=float(r["br"] / r["bdrift"]), dphase_deg=float(r["dphase"]),
                                     alpha_r=float(ar), alpha_zeta=float(az), R=float(r["R"]), ok=bool(r["ok"]),
                                     bz=float(r["bz"]), br=float(r["br"]), bdrift=float(r["bdrift"])))
        print(f"m=4 window {t0:g}-{t1:g}: bz/bd={r['bz'] / r['bdrift']:.3f} br/bd={r['br'] / r['bdrift']:.3f} (ok {r['ok']})")
    json.dump(out, open("mode_definitions.json", "w"), indent=1)
    print("wrote mode_definitions.json")


if __name__ == "__main__":
    main()
