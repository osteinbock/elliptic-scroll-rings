"""Damping and frequency of m-fold shape modes (Panfilov kinetics, seeded rings).

E_m = sqrt(r_m^2 + m^2/(m^2-1) zeta_m^2) is conserved by the lowest-order law with
alpha = 0; tension changes it at rate -alpha (2m^2-1)/(2 R^2), rigidity at about
-eta m^2 (m^2-1) / R^4 (pure rigidity: m = 2 rate R^4 / eta = 24, validated).
The phase of (r_m, sqrt(m^2/(m^2-1)) zeta_m) rotates with omega_m.
    python pan_modes.py file.npz m period alpha
"""
import sys
import numpy as np


def analyze(path, m, per, alpha, tmin=100.0, floor=0.15):
    d = np.load(path)
    ok = (d["ncurves"] == 1) & np.isfinite(d[f"r{m}c"])
    t = d["t"][ok]
    r = np.hypot(d[f"r{m}c"][ok], d[f"r{m}s"][ok]) * np.sign(d[f"r{m}c"][ok] + 1e-12)
    rc, zc = d[f"r{m}c"][ok], d[f"z{m}c"][ok]
    rs, zs = d[f"r{m}s"][ok], d[f"z{m}s"][ok]
    R = d["Rm"][ok]
    w = np.sqrt(m * m / (m * m - 1.0))
    E = np.sqrt(rc ** 2 + rs ** 2 + w ** 2 * (zc ** 2 + zs ** 2))
    edges = np.arange(0, t[-1] + per, per)
    tb, Eb, Rb, ph = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (t >= a) & (t < b)
        if k.sum() >= 5:
            tb.append(t[k].mean()); Eb.append(E[k].mean()); Rb.append(R[k].mean())
            ph.append(np.arctan2(w * zc[k].mean(), rc[k].mean()))
    tb, Eb, Rb, ph = map(np.array, (tb, Eb, Rb, ph))
    sel = (tb > tmin) & (Eb > floor)
    rate = -np.polyfit(tb[sel], np.log(Eb[sel]), 1)[0]
    om = abs(np.polyfit(tb[sel], np.unwrap(ph[sel]), 1)[0])
    R2 = np.mean(Rb[sel] ** 2)
    tens = alpha * (2 * m * m - 1) / (2 * R2)
    print(f"{path}: m={m}  Rbar={np.sqrt(R2):.2f}  window {tb[sel][0]:.0f}-{tb[sel][-1]:.0f}  "
          f"E {Eb[sel][0]:.2f}->{Eb[sel][-1]:.2f}")
    print(f"   decay rate {rate:.5f}   tension part {tens:+.5f}   extra {rate - tens:.5f}   "
          f"omega {om:.5f}  -> beta_eff = omega R^2/(m sqrt(m^2-1)) = {om * R2 / (m * np.sqrt(m * m - 1)):.3f}")
    return rate, tens, om, R2


if __name__ == "__main__":
    analyze(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
