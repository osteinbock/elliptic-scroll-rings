"""Fig. 1(b): rendered wave field (u = 1/2 shell, cut open) with the filament at t = 12.

Reads fig1_fields/ell8x5_fields.npz (ellipse_ring.py --fields run of the 8 x 5 ring) and writes
fig1_fields/render_t12.npy, which make_figs.py fig1() reads. Settings as for the published
panel (wave_alpha = 0.42).
    python fig1_render.py            (run from the data directory)
"""
import numpy as np
import matplotlib
from render_scroll import render

T = "12"
d = np.load("fig1_fields/ell8x5_fields.npz")
u = d[f"u_{T}"].astype(np.float32)
zo = float(d[f"zoff_{T}"])
lines = [d[k] - np.array([0, 0, zo]) for k in d.files if k.startswith(f"snap_{T}_")]
c = np.concatenate(lines).mean(0)
cut = lambda x, y, z: (x > c[0]) & (y > c[1])
img = render(u, lines, 0.2, azim=40, elev=26, fil_cmap=matplotlib.colormaps["coolwarm"],
             center=c + np.array([0, 0, -0.5]), extent=(12.5, 10.2), size=(1100, 900), cut=cut,
             crop=(np.array([0, 0, c[2] - 6.5]), np.array([99, 99, c[2] + 5.5])),
             wave_alpha=0.42, depth_step=0.035, tube=0.42, fil_low="#1f6fd6", fil_high="#e0402a",
             zrange=(c[2] - 1.0, c[2] + 1.0), light=(-0.3, 0.6, 0.75))
np.save("fig1_fields/render_t12.npy", img.astype(np.float32))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imsave("fig1_fields/render_t12.png", np.clip(img, 0, 1))
print("wrote fig1_fields/render_t12.npy")
