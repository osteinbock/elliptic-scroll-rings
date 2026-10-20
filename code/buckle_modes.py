"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Buckling mode analysis of the R0 = 38 scroll ring (sf2/circXXL_R0_38_snaps.npz).

For each filament snapshot: radial and vertical deviations from the mean circle, binned in the
polar angle, and their Fourier spectra |r_m|, |z_m|. Reports the dominant m, the growth of the
leading modes, their precession (phase rotation of r_m + i z_m), and k = m / R.
Writes sf2/buckle_spectra.png.

    python buckle_modes.py [file]
"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = 1024


def spectra(lines):
    lines = [l for l in lines if len(l) > 20]
    p = np.concatenate(lines)
    c = p.mean(0)
    th = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    rho = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    R = np.median(rho)
    keep = np.abs(rho - R) < 6
    th, rho, z = th[keep], rho[keep], p[keep, 2]
    b = ((th + np.pi) / (2 * np.pi) * NB).astype(int) % NB
    cnt = np.bincount(b, minlength=NB)
    rb = np.bincount(b, rho, NB)
    zb = np.bincount(b, z, NB)
    ok = cnt > 0
    ang = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi
    rb = np.interp(ang, ang[ok], rb[ok] / cnt[ok], period=2 * np.pi)
    zb = np.interp(ang, ang[ok], zb[ok] / cnt[ok], period=2 * np.pi)
    fr = np.fft.rfft(rb - rb.mean()) * 2 / NB
    fz = np.fft.rfft(zb - zb.mean()) * 2 / NB
    return R, fr, fz, ok.mean()


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else "sf2/circXXL_R0_38_snaps.npz"
    d = np.load(f)
    snaps = {}
    for key in d.files:
        m = re.match(r"snap_([\d.]+)_(\d+)$", key)
        if m:
            snaps.setdefault(float(m.group(1)), []).append(d[key])
    times = sorted(snaps)
    res = [(t, *spectra(snaps[t])) for t in times]
    mm = np.arange(len(res[0][2]))
    A = np.array([np.sqrt(np.abs(r[2]) ** 2 + np.abs(r[3]) ** 2) for r in res])   # combined amplitude
    print(" t      R     cover  top modes (m: |r_m|, |z_m|)")
    for (t, R, fr, fz, cov), a in zip(res, A):
        top = np.argsort(a[1:80])[::-1][:4] + 1
        print(f"{t:5.1f} {R:6.2f}  {cov:.2f}  " +
              "  ".join(f"{m}:{abs(fr[m]):.3f},{abs(fz[m]):.3f}" for m in top))
    late = A[-1]
    mdom = int(np.argmax(late[1:80]) + 1)
    Rl = res[-1][1]
    print(f"\ndominant m at t = {times[-1]:g}: {mdom}  (k = m/R = {mdom / Rl:.3f}, wavelength {2 * np.pi * Rl / mdom:.1f})")
    # growth rates of the leading modes (log-linear over the window where they are above noise)
    t = np.array(times)
    for m in np.argsort(late[1:80])[::-1][:6] + 1:
        y = A[:, m]
        sel = (y > 3 * np.median(A[:3, 1:80])) & (y < 0.15 * Rl / m * 2)
        if sel.sum() >= 4:
            g = np.polyfit(t[sel], np.log(y[sel]), 1)[0]
            Rm = np.mean([res[i][1] for i in np.where(sel)[0]])
            # precession: phase of the complex mode r_m + i z_m is not meaningful; use phase of z_m
            ph = np.unwrap(np.angle([res[i][3][m] for i in np.where(sel)[0]]))
            w = np.polyfit(t[sel], ph, 1)[0] / m      # angular speed of the pattern
            ratio = np.median([abs(res[i][3][m]) / abs(res[i][2][m]) for i in np.where(sel)[0]])
            print(f"m={m:3d}  k={m / Rm:.3f}  growth={g:+.4f}  -gamma/k^2={g / (m / Rm) ** 2:+.3f}"
                  f"  pattern rotation={w:+.4f}  |z_m|/|r_m|={ratio:.2f}  ({sel.sum()} snaps)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i in range(0, len(res), max(1, len(res) // 8)):
        ax[0].semilogy(mm[1:80], A[i, 1:80] + 1e-5, label=f"t={times[i]:g}")
    ax[0].set(xlabel="m", ylabel="|(r_m, z_m)|")
    ax[0].legend(fontsize=7)
    for m in np.argsort(late[1:80])[::-1][:6] + 1:
        ax[1].semilogy(times, A[:, m], label=f"m={m}")
    ax[1].set(xlabel="t", ylabel="mode amplitude")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("sf2/buckle_spectra.png", dpi=110)


if __name__ == "__main__":
    main()
