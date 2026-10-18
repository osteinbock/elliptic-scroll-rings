"""Frames for the supplementary movie (Fig. 1 multimedia): elliptic scroll ring, a = 1.1, b = 0.17.

Each frame is rendered as in Fig. 1(b): translucent u = 1/2 surface (facing quadrant cut away,
clipped to a cylinder around the ring), filament tube colored by height, camera azim 40, elev 26,
co-moving frame. The filament is also projected onto a horizontal plane below the wave (gray).

    python movie_frames.py i0 i1 [step]     -> movie/frames/f_XXXX.png for snapshot indices i0..i1-1
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_scroll import render

SRC = os.environ.get("FRAMES_SRC", "movie/ell8x5_movie.npz")
OUT = os.environ.get("FRAMES_OUT", "movie/frames")
WAVE_ALPHA = float(os.environ.get("WAVE_ALPHA", "0.42"))
AZIM, ELEV = 40.0, 26.0
EXT = (12.5, 10.2)
SIZE = (720, 588)


def camera():
    a, e = np.radians(AZIM), np.radians(ELEV)
    fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    return right, np.cross(right, fwd)


def to_pixels(P, center):
    right, up = camera()
    X, Y = (P - center) @ right, (P - center) @ up
    W, H = SIZE
    return (X + EXT[0]) / (2 * EXT[0]) * (W - 1), (EXT[1] - Y) / (2 * EXT[1]) * (H - 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(SRC)
    times = sorted(float(m.group(1)) for k in d.files if (m := re.match(r"u_([\d.]+)$", k)))
    i0, i1 = int(sys.argv[1]), int(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    cmap = matplotlib.colormaps["coolwarm"]
    for i in range(i0, min(i1, len(times)), step):
        T = times[i]
        key = f"{T:g}"
        u = d[f"u_{key}"].astype(np.float32)
        zo = float(d[f"zoff_{key}"])
        lines = [d[k] - np.array([0, 0, zo]) for k in d.files if k.startswith(f"snap_{key}_")]
        c = np.concatenate(lines).mean(0)
        center = c + np.array([0, 0, -0.5])
        cut = lambda x, y, z, c=c: (((x > c[0]) & (y > c[1])) | (np.hypot(x - c[0], y - c[1]) > 10.5)
                                    | (np.abs(z - c[2] + 0.3) > 5.2))
        img = render(u, lines, 0.2, azim=AZIM, elev=ELEV, fil_cmap=cmap, center=center, extent=EXT,
                     size=SIZE, cut=cut, crop=(np.array([0, 0, c[2] - 6.5]), np.array([99, 99, c[2] + 5.5])),
                     wave_alpha=WAVE_ALPHA, depth_step=0.06, tube=0.42, zrange=(c[2] - 1.0, c[2] + 1.0),
                     light=(-0.3, 0.6, 0.75), tmax=13.0)
        W, H = SIZE
        fig = plt.figure(figsize=(W / 120, H / 120), dpi=120)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_axis_off()
        # projection of the filament onto a horizontal plane below the wave
        p = max(lines, key=len)
        floor = np.c_[p[:, 0], p[:, 1], np.full(len(p), c[2] - 6.2)]
        px, py = to_pixels(np.vstack([floor, floor[:1]]), center)
        ax.plot(px, py, color="#a9a8a1", lw=1.6, zorder=1, solid_capstyle="round")
        ax.imshow(img, interpolation="lanczos", zorder=2)
        ax.text(0.03, 0.96, f"$t = {T:.1f}$", transform=ax.transAxes, fontsize=13, va="top", zorder=3)
        fig.savefig(f"{OUT}/f_{i:04d}.png", dpi=120, facecolor="white")
        plt.close(fig)
        print("frame", i, T, flush=True)


if __name__ == "__main__":
    main()
