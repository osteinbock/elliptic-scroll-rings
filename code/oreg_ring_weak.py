"""Oregonator scroll ring at weak excitability, with free tip thresholds, seed wake, and time step.
    python oreg_ring_weak.py f eps Dv ustar vstar vwake dt Ax Ay Lxy Lz T out.npz
The ring of the supplementary material:
    python oreg_ring_weak.py 4.2 0.05 0.0 0.199 0.0408 0.25 0.0004 30 20 100 48 100 out.npz"""
import sys
import numpy as np
import ellipse_ring as E
import oregonator as O

f, eps, Dv, ustar, vstar, vwake, dt, Ax, Ay, Lxy, Lz, T = map(float, sys.argv[1:13])
out = sys.argv[13]
M = O.Model(f, eps=eps, Dv=Dv, ustar=ustar, vstar=vstar, dt=dt, v_wake=vwake)
rows, frames = E.run(0, 0, 0, Ax, Ay, Lxy, Lz, 0.5, T, every_t=0.25, verbose=True, snaps=[], model=M)
d = E.table(rows)
np.savez(out, **d, f=f, eps=eps, Dv=Dv, Ax=Ax, Ay=Ay, Lxy=Lxy, Lz=Lz, dx=0.5, dt=dt)
ok = np.isfinite(d["Rm"])
print(f"DONE {out}: R {d['Rm'][ok][min(20, ok.sum() - 1)]:.2f} -> {d['Rm'][ok][-1]:.2f} at t={d['t'][ok][-1]:.0f}")
