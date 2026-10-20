"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Linear damping of m = 2 and m = 3 shape modes on a ring, filament equation
X_t = alpha X_ss + beta X_s x X_ss - eta P(X_ssss), for two hypotheses:
  tension-like damping (alpha > 0, eta = 0)  vs  rigidity (alpha = 0, eta > 0).
Amplitude invariant A_m = sqrt(r_m^2 + m^2/(m^2-1) z_m^2) (exchanged by the binormal term)."""
import numpy as np
import filament_flow as F


def rate(m, R, alpha, beta, eta, T):
    X0 = F.mfold(R, m, 0.03, n=64)
    r = F.run(0, 0, alpha, beta, T, n=64, dt=0.01, eta=eta, X0=X0, every=0.1)
    A = np.sqrt(r[f"r{m}"] ** 2 + m ** 2 / (m ** 2 - 1) * r[f"z{m}"] ** 2)
    # remove the trivial shrink of the ring itself (relative amplitude A/R)
    Rt = r["length"] / (2 * np.pi)
    y = np.log(A / Rt)
    return -np.polyfit(r["t"], y, 1)[0], r


if __name__ == "__main__":
    beta = -1.85
    print("hypothesis        R    gamma_2    gamma_3   ratio 3/2")
    for R in (5.3, 8.0):
        for lab, al, eta in (("tension a=0.1", 0.1, 0.0), ("rigidity eta=1", 0.0, 1.0)):
            T = 30 * (R / 5.3) ** 2
            g2, _ = rate(2, R, al, beta, eta, T)
            g3, _ = rate(3, R, al, beta, eta, T)
            print(f"{lab:16s} {R:4.1f}   {g2:.5f}   {g3:.5f}   {g3 / g2:.2f}")
