"""Damping and frequency of m = 2 and m = 3 shape modes of RD scroll rings
(mode_m{2,3}_{s,L}.npz from ellipse_ring.py --m ... --amp ...).

Amplitude invariant A_m = sqrt(r_m^2 + m^2/(m^2-1) z_m^2) (the binormal term only
exchanges r_m and z_m); damping gamma_m = -d ln(A_m/R)/dt. Frequency from the
phase of q = r_mc + i m/sqrt(m^2-1) z_mc. Compares gamma_3/gamma_2 and the size
dependence with tension-like (ratio 3, R^-2) and rigidity (ratio 5, R^-4) predictions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ellipse_compare as C

PERIOD = 4.44
INK, MUTED = "#1f1f1e", "#6b6a63"
COL = {2: "#2a78d6", 3: "#eb6834"}


def analyse(path, m, t_start=6.0, floor=0.012):
    d = np.load(path)
    ok = np.isfinite(d["Rm"]) & np.isfinite(d[f"r{m}c"])
    t = d["t"][ok]
    c = m / np.sqrt(m * m - 1)
    rmc, rms_, zmc, zms = (d[f"{k}"][ok] for k in (f"r{m}c", f"r{m}s", f"z{m}c", f"z{m}s"))
    A = np.sqrt(rmc ** 2 + rms_ ** 2 + c ** 2 * (zmc ** 2 + zms ** 2))
    R = C.smooth(t, d["Rm"][ok], PERIOD)
    As = C.smooth(t, A, PERIOD)
    rel = As / R
    fitm = (t >= t_start) & np.isfinite(rel) & (rel > floor)
    # stop at the first time the relative amplitude drops below the floor
    if (rel[t >= t_start] < floor).any():
        tstop = t[t >= t_start][np.argmax(rel[t >= t_start] < floor)]
        fitm &= t < tstop
    gamma = -np.polyfit(t[fitm], np.log(rel[fitm]), 1)[0]
    q = C.smooth(t, rmc, PERIOD) + 1j * c * C.smooth(t, zmc, PERIOD)
    qm = fitm & np.isfinite(q)
    ph = np.unwrap(np.angle(q[qm]))
    omega = abs(np.polyfit(t[qm], ph, 1)[0])
    return dict(t=t, rel=rel, fit=fitm, gamma=gamma, omega=omega, Rmean=np.nanmean(R[fitm]),
                t0=t[fitm][0], t1=t[fitm][-1])


if __name__ == "__main__":
    res = {}
    for size in ("s", "L"):
        for m in (2, 3):
            r = analyse(f"mode_m{m}_{size}.npz", m)
            res[(m, size)] = r
            print(f"m={m} size={size}: <R>={r['Rmean']:.2f}  fit t={r['t0']:.0f}-{r['t1']:.0f}  "
                  f"gamma={r['gamma']:.4f}  omega={r['omega']:.3f}  "
                  f"beta_eff=omega R^2/(m sqrt(m^2-1))={r['omega'] * r['Rmean'] ** 2 / (m * np.sqrt(m * m - 1)):.2f}")
    for size in ("s", "L"):
        print(f"size {size}: gamma3/gamma2 = {res[(3, size)]['gamma'] / res[(2, size)]['gamma']:.2f}  "
              f"(tension 3.0, rigidity 5.0)")
    for m in (2, 3):
        s, L = res[(m, "s")], res[(m, "L")]
        p = np.log(s["gamma"] / L["gamma"]) / np.log(L["Rmean"] / s["Rmean"])
        print(f"m={m}: gamma ~ R^-{p:.2f}  (tension 2, rigidity 4)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=120, sharey=True)
    for j, size in enumerate(("s", "L")):
        for m in (2, 3):
            r = res[(m, size)]
            ax[j].semilogy(r["t"], r["rel"], color=COL[m], lw=1.6, label=f"m={m}  gamma={r['gamma']:.4f}")
            tt = r["t"][r["fit"]]
            ax[j].semilogy(tt, np.exp(np.polyval(np.polyfit(tt, np.log(r["rel"][r["fit"]]), 1), tt)),
                           color=COL[m], ls="--", lw=1)
        ax[j].set_title(f"ring <R> ≈ {res[(2, size)]['Rmean']:.1f}", fontsize=10, color=INK)
        ax[j].set_xlabel("time")
        ax[j].legend(frameon=False, fontsize=8)
        ax[j].grid(color="#e4e3dd", lw=0.6, which="both")
        for s_ in ("top", "right"):
            ax[j].spines[s_].set_visible(False)
    ax[0].set_ylabel("relative mode amplitude A_m / R")
    fig.suptitle("Barkley a=1.1 b=0.17 eps=0.02: damping of m=2 and m=3 filament shape modes", fontsize=10)
    fig.tight_layout()
    fig.savefig("mode_damping.png")
    print("wrote mode_damping.png")
