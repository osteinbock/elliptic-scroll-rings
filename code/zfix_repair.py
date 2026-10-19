"""Repair the saved ring height z in tables written by ellipse_ring.run before the z fix.

The old writer stored z = zc + zoff with zoff already updated by that frame's roll, so
each frame with a roll k carried an error of -k dx. With that writer z - zc is the
cumulative offset after the frame; its step at a roll frame equals -k dx, hence
    z_fixed = z - diff([0, z - zc])        (over the finite rows, in order).
The original column is kept as z_raw, and zfix = 1 marks repaired (or new) files.
Each repaired file with filament snapshots is checked: the arc-length-weighted mean
height of the snapshot curve must equal z_fixed at the matching row.
    python zfix_repair.py [--dry] file.npz [file.npz ...]
"""
import sys
import numpy as np


def repaired(d):
    z, zc = d["z"].astype(float), d["zc"].astype(float)
    ok = np.isfinite(z) & np.isfinite(zc)
    idx = np.flatnonzero(ok)
    if len(idx) and idx[-1] - idx[0] + 1 != len(idx):
        raise ValueError("non-finite rows inside the record; ordering not guaranteed")
    zf = z.copy()
    zf[ok] = z[ok] - np.diff(np.r_[0.0, (z - zc)[ok]])
    return zf


def snapshot_check(d, zf):
    t = d["t"]
    errs = []
    for k in d.files:
        if not (k.startswith("snap_") and k.endswith("_0")):
            continue
        s = float(k[5:].rsplit("_", 1)[0])
        if f"snap_{k[5:].rsplit('_', 1)[0]}_1" in d.files:
            continue                                   # several curves: skip
        p = d[k]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        w = np.r_[seg, 0] + np.r_[0, seg]
        zs = (p[:, 2] * w).sum() / w.sum()
        i = np.nanargmin(np.abs(t - s))
        if abs(t[i] - s) < 0.5 and np.isfinite(zf[i]):
            errs.append(zs - zf[i])
    return np.array(errs)


def main(paths, dry=False):
    for p in paths:
        try:
            d = np.load(p)
        except Exception as e:
            print(f"{p}: unreadable ({e})"); continue
        if "z" not in d.files or "zc" not in d.files:
            continue
        if "zfix" in d.files:
            print(f"{p}: already fixed"); continue
        try:
            zf = repaired(d)
        except ValueError as e:
            print(f"{p}: NOT repaired ({e})"); continue
        e = snapshot_check(d, zf)
        e_old = snapshot_check(d, d["z"].astype(float))
        tag = ""
        if len(e):
            tag = f"snapshots {len(e)}: max|err| old {np.abs(e_old).max():.2e}  new {np.abs(e).max():.2e}"
            if np.abs(e).max() > 1e-6:
                print(f"{p}: NOT repaired, snapshot mismatch ({tag})"); continue
        nfix = int((np.abs(zf - d["z"]) > 1e-12).sum())
        print(f"{p}: {nfix} rows changed  {tag}")
        if not dry:
            out = {k: d[k] for k in d.files}
            out["z_raw"] = out["z"]
            out["z"] = zf
            out["zfix"] = np.ones_like(zf)
            d.close()
            np.savez(p, **out)


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry" in args
    main([a for a in args if a != "--dry"], dry)
