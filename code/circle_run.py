"""One circular scroll ring (Barkley, periodic box); saves the R(t), z(t) table.
    python circle_run.py a b eps R0 Lxy Lz T out.npz [dx]"""
import sys
import numpy as np
import ellipse_ring as E

a, b, eps, R0, Lxy, Lz, T = map(float, sys.argv[1:8])
dx = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2
rows, _ = E.run(a, b, eps, R0, R0, Lxy, Lz, dx, T, every_t=0.25, verbose=False)
d = E.table(rows)
np.savez(sys.argv[8], **d, a=a, b=b, eps=eps, R0=R0, Lxy=Lxy, Lz=Lz, dx=dx, T=T, every_t=0.25,
         dt=0.8 * dx * dx / 6, code_revision=E.code_revision())
ok = np.isfinite(d["Rm"])
if ok.sum() == 0:
    sys.exit(f"R0={R0}: no filament found")
print(f"R0={R0}: R {d['Rm'][ok][min(20, ok.sum() - 1)]:.2f} -> {d['Rm'][ok][-1]:.2f} at t={d['t'][ok][-1]:.0f}, "
      f"max ncurves after t=10: {np.nanmax(d['ncurves'][d['t'] > 10]):.0f}")
