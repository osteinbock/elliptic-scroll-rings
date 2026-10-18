"""Draft of Fig. 1(b): one axis-switching cycle of an elliptic scroll-ring filament
(Barkley a=1.1 b=0.17 eps=0.02, seed 8x5), filament coloured by height."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("ell_a1.1_b0.17_8x5.npz")
times = [4, 12, 20, 28, 36, 44]
fig = plt.figure(figsize=(13, 2.9), dpi=130)
for i, s in enumerate(times):
    curves = [d[k] for k in d.files if k.startswith(f"snap_{s}_")]
    p = max(curves, key=len)
    c = p.mean(0)
    q = p - c
    ax = fig.add_subplot(1, len(times), i + 1, projection="3d")
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=q[:, 2], cmap="coolwarm", vmin=-1.2, vmax=1.2, s=3)
    ax.plot(q[:, 0], q[:, 1], np.full(len(q), -3.0), color="#b8b7b0", lw=0.8)   # shadow
    ax.set(xlim=(-7, 7), ylim=(-7, 7), zlim=(-3, 3))
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=28, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"t = {s}", fontsize=9)
fig.suptitle("Draft Fig. 1(b): one axis-switching cycle (filament coloured by height; grey = projection)", fontsize=9)
fig.tight_layout()
fig.savefig("fig1_draft.png")
print("wrote fig1_draft.png")
