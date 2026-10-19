"""Mark frames whose filament is not a single closed curve as invalid (audit item 4).

For tables written by ellipse_ring.run with the 'closed' flag: in frames with ncurves > 1 or
closed = 0 the measured quantities describe only the longest piece, so they are set to nan;
the original values are kept as raw_<key>, and masked = 1 marks the file. All analyses drop
non-finite frames; ellipse_compare.smooth bridges the gaps (up to half a rotation period).
The last row of a run (no filament at all) is left alone. Idempotent.
    python mask_frames.py file.npz [file.npz ...]
"""
import sys
import numpy as np

KEEP = {"t", "ncurves", "closed", "zfix"}


def mask(path):
    d = np.load(path)
    if "closed" not in d.files or "ncurves" not in d.files or "masked" in d.files:
        return None
    out = {k: d[k] for k in d.files}
    d.close()
    n = len(out["t"])
    bad = np.isfinite(out["Rm"]) & ~((out["ncurves"] == 1) & (out["closed"] == 1))
    for k, v in list(out.items()):
        if k in KEEP or k.startswith("snap_") or np.ndim(v) != 1 or len(v) != n or v.dtype.kind != "f":
            continue
        out["raw_" + k] = v.copy()
        w = v.copy()
        w[bad] = np.nan
        out[k] = w
    out["masked"] = np.array(1)
    np.savez_compressed(path, **out)
    return bad.sum(), np.isfinite(out["raw_Rm"]).sum()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = mask(p)
        if r:
            print(f"{p}: masked {r[0]} of {r[1]} frames")
