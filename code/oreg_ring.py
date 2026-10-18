"""Oregonator scroll ring (circle or ellipse) in a periodic box; saves the ring table.
    python oreg_ring.py f eps Dv vstar Ax Ay Lxy Lz T out.npz [snaps...]"""
import sys
import numpy as np
import ellipse_ring as E
import oregonator as O

f, eps, Dv, vstar, Ax, Ay, Lxy, Lz, T = map(float, sys.argv[1:10])
out = sys.argv[10]
snaps = [float(s) for s in sys.argv[11:]]
M = O.Model(f, eps=eps, Dv=Dv, ustar=0.3, vstar=vstar, dt=0.0005)
rows, frames = E.run(0, 0, 0, Ax, Ay, Lxy, Lz, 0.5, T, every_t=0.25, verbose=True, snaps=snaps, model=M)
d = E.table(rows)
for s, ls in frames.items():
    for i, l in enumerate(ls):
        d[f"snap_{s:g}_{i}"] = l
np.savez(out, **d, f=f, eps=eps, Dv=Dv, Ax=Ax, Ay=Ay, Lxy=Lxy, Lz=Lz, dx=0.5)
ok = np.isfinite(d["Rm"])
print(f"DONE {out}: R {d['Rm'][ok][min(20, ok.sum()-1)]:.2f} -> {d['Rm'][ok][-1]:.2f} at t={d['t'][ok][-1]:.0f}")
