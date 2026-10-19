"""Validation of the treatment of fragmented frames (audit item 4).

In frames with ncurves > 1 the old extractor broke the ring into open pieces separated by
about one grid cell (one missed face crossing), and measure() used the longest piece. For
every saved snapshot with several pieces, the pieces are stitched (barkley3d.stitch) and
measured; the result is compared with (i) the old row of that frame and (ii) the value
interpolated in time from the neighbouring single-curve frames, which is what the
analyses use after dropping fragmented frames.
    python stitch_validate.py file.npz dx [file.npz dx ...]
"""
import sys
import collections
import numpy as np
import barkley3d as B
import ellipse_ring as E

KEYS = ("Rm", "length", "r2c", "z2c", "z")


def validate(path, dx):
    d = np.load(path)
    t, nc = d["t"], d["ncurves"]
    good = np.isfinite(d["Rm"]) & (nc == 1)
    snaps = collections.defaultdict(list)
    for k in d.files:
        if k.startswith("snap_"):
            s, i = k[5:].rsplit("_", 1)
            snaps[s].append((int(i), d[k]))
    res = collections.defaultdict(list)
    nclosed = nmulti = 0
    for s, ls in snaps.items():
        if len(ls) < 2:
            continue
        nmulti += 1
        lines = B.stitch([l for _, l in sorted(ls)], 2.0 * dx)
        m = E.measure(lines, None)
        if m is None or not m["closed"]:
            continue
        nclosed += 1
        i = np.nanargmin(np.abs(t - float(s)))
        zoff = d["z"][i] - d["zc"][i]                 # snapshots carry the offset already
        m["z"] = m["zc"]
        for k in KEYS:
            interp = np.interp(t[i], t[good], d[k][good])
            res[k].append((m[k], d[k][i], interp))
    print(f"{path}: multi-piece snapshots {nmulti}, stitched to one closed ring {nclosed}")
    for k in KEYS:
        if not res[k]:
            continue
        a = np.array(res[k])
        scale = np.nanmean(np.abs(d[k][good])) if k in ("z2c", "r2c") else np.nanmean(d["Rm"][good])
        print(f"   {k:6s} |old row - stitched| median {np.median(abs(a[:,1]-a[:,0])):.3f}   "
              f"|interpolated - stitched| median {np.median(abs(a[:,2]-a[:,0])):.3f} "
              f"max {np.max(abs(a[:,2]-a[:,0])):.3f}   (scale {scale:.2f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    for p, dx in zip(a[::2], map(float, a[1::2])):
        validate(p, dx)
