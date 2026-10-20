"""EXPLORATORY: not used for any figure, table, or number of the paper; kept for the record.

Phase-locked strobe times from the ring-center probe u(t): band-pass around the rotation
frequency, Hilbert phase, and the times at which the unwrapped phase passes 2 pi n.

    python strobe_triggers.py pass1.npz [t_start] -> prints times, writes strobe/triggers.txt
"""
import sys
import numpy as np
from scipy.signal import hilbert

d = np.load(sys.argv[1])
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
t, u = d["probe_t"], d["probe_u"]
dt = np.median(np.diff(t))
x = u - u.mean()
F = np.fft.rfft(x)
f = np.fft.rfftfreq(len(x), dt)
F[(f < 0.12) | (f > 0.32)] = 0                       # rotation frequency ~ 0.21
xf = np.fft.irfft(F, len(x))
ph = np.unwrap(np.angle(hilbert(xf)))
m = t >= t0
n0 = np.ceil(ph[m][0] / (2 * np.pi))
targets = 2 * np.pi * np.arange(n0, ph[m][-1] / (2 * np.pi))
tt = np.interp(targets, ph[m], t[m])
tt = tt[tt <= t[-1] - 2.0]
P = np.diff(tt)
print(f"{len(tt)} strobe times, period median {np.median(P):.3f} (range {P.min():.3f}-{P.max():.3f})")
np.savetxt("strobe/triggers.txt", np.round(tt, 4), fmt="%.4f")
