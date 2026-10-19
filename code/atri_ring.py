"""Atri Ca2+ scroll ring (um, s) (circle or ellipse) in a periodic box.
    ATRI_DX=3.0 ATRI_CSTAR=0.52 python atri_ring.py P_uM nstar Ax Ay Lxy Lz T out.npz [snaps...]
(the paper uses ATRI_DX = 3 and ATRI_CSTAR = 0.52 at P = 0.1; see runs_v2.sh)"""
import os
import sys
import numpy as np
import ellipse_ring as E
import atri as AT

gf, vstar, Ax, Ay, Lxy, Lz, T = map(float, sys.argv[1:8])
out = sys.argv[8]
snaps = [float(s) for s in sys.argv[9:]]
DX = float(os.environ.get('ATRI_DX', '2.0'))
M = AT.Model(P=gf, cstar=float(os.environ.get('ATRI_CSTAR', '0.566')), nstar=vstar, dt=0.01)
rows, frames = E.run(0, 0, 0, Ax, Ay, Lxy, Lz, DX, T, every_t=2.0, verbose=True, snaps=snaps, model=M)
d = E.table(rows)
for s, ls in frames.items():
    for i, l in enumerate(ls):
        d[f"snap_{s:g}_{i}"] = l
np.savez(out, **d, gf=gf, P=gf, Ax=Ax, Ay=Ay, Lxy=Lxy, Lz=Lz, dx=DX, dt=M.dt(DX), cstar=M.cstar,
         nstar=M.nstar, every_t=2.0, T=T)   # gf: legacy name of P
ok = np.isfinite(d["Rm"])
print(f"DONE {out}: R {d['Rm'][ok][min(20, ok.sum()-1)]:.2f} -> {d['Rm'][ok][-1]:.2f} at t={d['t'][ok][-1]:.0f}")
