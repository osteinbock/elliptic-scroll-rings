"""Panfilov kinetics-B scroll ring (circle or ellipse) in a periodic box.
    python pan_ring.py gf vstar Ax Ay Lxy Lz T out.npz [snaps...]"""
import os
import sys
import numpy as np
import ellipse_ring as E
import panfilov as PK

gf, vstar, Ax, Ay, Lxy, Lz, T = map(float, sys.argv[1:8])
out = sys.argv[8]
snaps = [float(s) for s in sys.argv[9:]]
DX = float(os.environ.get('PAN_DX', '0.6'))
M = PK.Model(gf, ustar=0.47, vstar=vstar, dt=min(0.04, 0.8 * DX * DX / 6))
MODE = int(os.environ.get('PAN_MODE', '0')); AMP = float(os.environ.get('PAN_AMP', '0'))
rows, frames = E.run(0, 0, 0, Ax, Ay, Lxy, Lz, DX, T, every_t=1.0, verbose=True, snaps=snaps, model=M, mode=MODE, amp=AMP)
d = E.table(rows)
for s, ls in frames.items():
    for i, l in enumerate(ls):
        d[f"snap_{s:g}_{i}"] = l
np.savez(out, **d, gf=gf, Ax=Ax, Ay=Ay, Lxy=Lxy, Lz=Lz, dx=DX)
ok = np.isfinite(d["Rm"])
print(f"DONE {out}: R {d['Rm'][ok][min(20, ok.sum()-1)]:.2f} -> {d['Rm'][ok][-1]:.2f} at t={d['t'][ok][-1]:.0f}")
