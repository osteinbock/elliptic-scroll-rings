"""Test of the curvature-dependent-drift explanation for the slow axis-switching oscillation.

y = beta_eff / beta_drift, where beta_eff is the binormal coefficient implied by the
observed oscillation (half-cycles: beta_eff = pi R^2 / (2 sqrt3 T_half); fixed-radius
modes: beta_eff = omega R^2 / (m sqrt(m^2-1))) and beta_drift = R dz/dt of the same ring.
x = K = 1/R.
Prediction (no free parameter): linearizing a local law v_B = beta(K) K gives the shape
modes the coefficient d(beta K)/dK = beta + K beta', so the ratio is 1 + K beta'/beta,
with beta(K) from the circular rings (alpha_late.npy, b = 0.17). Band: bootstrap over
the circle windows (quadratic fits); dashed beyond the measured K range.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#1f1f1e", "#6b6a63"
COL = {0.15: "#2a78d6", 0.16: "#1baf7a", 0.17: "#eb6834", 0.18: "#e87ba4"}

late = np.load("alpha_late.npy")
Kc, Bc = late[:, 1], np.abs(late[:, 3])
kmin, kmax = Kc.min(), Kc.max()
kk = np.linspace(0.08, 0.34, 200)


def ratio_curve(K, B, deg=2):
    p = np.polyfit(K, B, deg)
    return 1 + kk * np.polyval(np.polyder(p), kk) / np.polyval(p, kk)


rng = np.random.default_rng(1)
boot = np.array([ratio_curve(Kc[i], Bc[i]) for i in (rng.integers(0, len(Kc), len(Kc)) for _ in range(400))])
lo, hi = np.percentile(boot, [16, 84], axis=0)
mid = ratio_curve(Kc, Bc)
lin = ratio_curve(Kc, Bc, 1)

rows = [r for r in csv.DictReader(open("period_table.csv")) if r["a"] == "1.1"]
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=140)
inside = (kk >= kmin) & (kk <= kmax)
ax.fill_between(kk[inside], lo[inside], hi[inside], color="#e4e3dd", zorder=0,
                label="prediction 1 + Kβ′/β  (β(K) from circles, b = 0.17)")
ax.plot(kk[inside], mid[inside], color=INK, lw=1.4)
ax.plot(kk[kk >= kmax], mid[kk >= kmax], color=INK, lw=1.0, ls="--")
ax.plot(kk[kk <= kmin], mid[kk <= kmin], color=INK, lw=1.0, ls="--")
ax.plot(kk[inside], lin[inside], color=MUTED, lw=1.0, ls=":", label="same, linear fit of β(K)")
for b in sorted(COL):
    rr = [r for r in rows if abs(float(r["b"]) - b) < 1e-9]
    if not rr:
        continue
    K = np.array([1 / float(r["R"]) for r in rr])
    beff = np.array([np.pi * float(r["R"]) ** 2 / (2 * np.sqrt(3) * float(r["Th"])) for r in rr])
    y = beff / np.array([abs(float(r["beta"])) for r in rr])
    filled = abs(b - 0.17) < 1e-9
    ax.scatter(K, y, s=34, color=COL[b] if filled else "white", edgecolor=COL[b], linewidth=1.4, zorder=3,
               label=f"ellipse half-cycles, b = {b:g}" + ("" if filled else "  (β(K) not measured)"))
# fixed-radius mode runs (b = 0.17, R ~ 4.85), from mode_freq.py
for m, y, mk in ((2, (0.681 + 0.690) / 2, "s"), (3, (0.612 + 0.636) / 2, "D")):
    ax.scatter([1 / 4.85], [y], marker=mk, s=60, color="#1f1f1e", zorder=4, label=f"m = {m} mode at R*, b = 0.17")
ax.axhline(1, color=MUTED, lw=0.8)
ax.text(0.335, 1.01, "constant β", ha="right", va="bottom", fontsize=7, color=MUTED)
ax.set(xlabel="ring curvature K = 1/R̄", ylabel="β_eff / β_drift", xlim=(0.08, 0.34), ylim=(0.3, 1.15))
ax.grid(color="#e4e3dd", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.set_title("Barkley a = 1.1, ε = 0.02: oscillation-implied vs drift binormal coefficient", fontsize=8.5)
fig.tight_layout()
fig.savefig("beta_ratio.png")
print("circle K range %.3f-%.3f" % (kmin, kmax))
for k0 in (0.10, 0.15, 0.20, 0.206, 0.22):
    i = np.argmin(abs(kk - k0))
    print(f"K={k0:.3f}: predicted ratio {mid[i]:.3f}  (68% band {lo[i]:.3f}-{hi[i]:.3f}; linear-fit {lin[i]:.3f})")
print("wrote beta_ratio.png")
