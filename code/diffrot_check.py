"""Differential rotation check: local rotation phase of the scroll wave around the elliptic ring.

The tip line (u = 1/2, v = a/2 - b) circles the true filament once per rotation. At each polar
angle theta of the ring, its offset from the period-averaged filament, (d_rho, d_z), gives the
local rotation phase phi(theta, t) = atan2(d_z, d_rho). A twist w along the filament appears as a
phase gradient, w ~ (1/R) d phi / d theta; differential rotation appears as a phase difference
between ends and sides that varies with the switching cycle.

Data: movie/ell8x5_movie.npz (8 x 5 ring, a = 1.1, b = 0.17; snapshots every 0.2 for t = 3-40).
"""
import re
import numpy as np
from scipy.ndimage import uniform_filter1d

d = np.load("movie/ell8x5_movie.npz")
times = np.array(sorted(float(m.group(1)) for k in d.files if (m := re.match(r"snap_([\d.]+)_0$", k))))
NB = 36
th_c = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi
RHO = np.full((len(times), NB), np.nan)
Z = np.full((len(times), NB), np.nan)
for i, t in enumerate(times):
    ls = [d[k] for k in d.files if k.startswith(f"snap_{t:g}_")]
    p = max(ls, key=len)
    c = p.mean(0)
    th = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    rho = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    b = ((th + np.pi) / (2 * np.pi) * NB).astype(int) % NB
    n = np.bincount(b, minlength=NB)
    ok = n > 0
    RHO[i, ok] = np.bincount(b, rho, NB)[ok] / n[ok]
    Z[i, ok] = np.bincount(b, p[:, 2] - c[2], NB)[ok] / n[ok]
# fill rare empty bins by interpolation in theta
for A in (RHO, Z):
    for i in range(len(times)):
        g = np.isfinite(A[i])
        A[i] = np.interp(th_c, th_c[g], A[i, g], period=2 * np.pi)
P = 4.8                                         # ring rotation period (from the strobe probe)
w = int(round(P / 0.2))
Rb, Zb = uniform_filter1d(RHO, w, axis=0, mode="nearest"), uniform_filter1d(Z, w, axis=0, mode="nearest")
dr, dz = RHO - Rb, Z - Zb
amp = np.hypot(dr, dz)
phi = np.unwrap(np.arctan2(dz, dr), axis=0)          # unwrap in time
phi = phi - 2 * np.pi * np.round((phi[:1] - phi[:1, :1]) / (2 * np.pi))   # common branch at t0
rel = phi - phi.mean(1, keepdims=True)                # phase relative to the ring average
sel = (times > times[0] + P) & (times < times[-1] - P)   # avoid smoothing edges
Rbar = Rb[sel].mean()
grad = np.gradient(rel[sel], th_c, axis=1) / Rbar     # twist estimate (rad per unit length)
print(f"tip-wobble amplitude: median {np.median(amp[sel]):.2f} (core radius ~0.5)")
print(f"ring-average rotation rate: {np.polyfit(times[sel], phi[sel].mean(1), 1)[0]:+.3f} rad/time "
      f"(2 pi/4.8 = {2*np.pi/4.8:.3f})")
print(f"phase spread around the ring (max-min of relative phase): median {np.median(np.ptp(rel[sel], 1)):.2f} rad, "
      f"max {np.max(np.ptp(rel[sel], 1)):.2f} rad")
print(f"twist |dphi/ds|: median {np.median(np.abs(grad)):.3f}, 90th pct {np.percentile(np.abs(grad), 90):.3f} rad/length")
# ends vs sides: theta = 0 / pi are the initial long-axis ends (x), +-pi/2 the sides (y)
ends = rel[sel][:, [np.argmin(abs(th_c)), np.argmin(abs(abs(th_c) - np.pi))]].mean(1)
sides = rel[sel][:, [np.argmin(abs(th_c - np.pi / 2)), np.argmin(abs(th_c + np.pi / 2))]].mean(1)
diff = ends - sides
hxy = None
print("t, phase(ends)-phase(sides) [rad]:")
for tt, dd in list(zip(times[sel], diff))[::10]:
    print(f"  t={tt:5.1f}  {dd:+.2f}")
np.savez("diffrot.npz", t=times[sel], rel=rel[sel], th=th_c, diff=diff, grad=grad)
