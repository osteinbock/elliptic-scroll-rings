"""Compact data archive for the repository: per-frame tables of all production runs and the
analysis products behind every figure and table.

For each run listed in runs_v2.sh / runs_barkley_v2.sh, the per-frame ring measures (masked by
mask_frames.py, with flags and extraction diagnostics) and the run metadata are written to
data/<group>/<name>.npz; filament snapshots, fields and straight-filament profiles are omitted
(the complex amplitudes u_k are kept). Analysis products are copied. data/MANIFEST.json lists
every file with its SHA-256, the command that produced it, and the code revision.
    python export_data.py outdir [ROOT_SUPP ROOT_BARKLEY]    (data folders, default v2 v2b)
"""
import os
import re
import sys
import json
import shutil
import hashlib
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DROP = ("snap_", "u_", "v_", "zoff_", "raw_")
PRODUCTS = {
    "barkley": ["period_table_r2.csv", "mode_decompose.npy", "mode_sweep.npy", "new_tests.npz",
            "alpha_late.npy", "sf2/sf2_linear.npy", "sf2/buckle.npz", "amp_factor.npy", "diffrot.npz",
                "paper_numbers.json"],
    "supp": ["halfcycles_panfilov.csv", "halfcycles_panfilov.csv.meta.json", "halfcycles_atri.csv",
           "halfcycles_atri.csv.meta.json", "coefficients_circles.csv", "coefficients_circles.csv.meta.json"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commands(roots):
    """Output file (relative to its run directory) -> command line, from the run lists. The
    run lists write to v2/ and v2b/; roots maps these to the folders actually used."""
    out = {}
    for lst, listed, root in (("runs_v2.sh", "v2", roots[0]), ("runs_barkley_v2.sh", "v2b", roots[1])):
        for line in open(os.path.join(HERE, lst), encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"cd (\S+) && (.*)", line)
            sub = os.path.relpath(m.group(1), listed) if m else "."
            for name in re.findall(r"([\w.\-]+\.npz)", line):
                out[os.path.normpath(os.path.join(root, sub, name))] = line
    return out


def main(outdir, roots=("v2", "v2b")):
    rev = subprocess.run(["git", "-C", os.path.join(HERE), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip() or "see repository history"
    cmd = commands(roots)
    manifest = dict(code_revision=rev, runs=[], products=[])
    for src, line in sorted(cmd.items()):
        path = os.path.join(HERE, src)
        if not os.path.exists(path):
            raise FileNotFoundError(f"run output missing: {src}")
        d = np.load(path)
        keep = {k: d[k] for k in d.files if not k.startswith(DROP) and k != "prof"}
        dst = os.path.join(outdir, "runs", src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, **keep)
        manifest["runs"].append(dict(file=os.path.relpath(dst, outdir).replace("\\", "/"), source=src.replace("\\", "/"),
                                     command=line, source_sha256=sha256(path), sha256=sha256(dst),
                                     frames=int(len(keep["t"])) if "t" in keep else None))
    for group, files in PRODUCTS.items():
        root = roots[1] if group == "barkley" else roots[0]
        for f in files:
            src = os.path.join(HERE, root, f)
            if not os.path.exists(src):
                raise FileNotFoundError(f"analysis product missing: {root}/{f}")
            dst = os.path.join(outdir, "analysis", group, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            manifest["products"].append(dict(file=os.path.relpath(dst, outdir).replace("\\", "/"), sha256=sha256(dst)))
    with open(os.path.join(outdir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest['runs'])} run tables and {len(manifest['products'])} analysis products to {outdir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], tuple(sys.argv[2:4]) if len(sys.argv) >= 4 else ("v2", "v2b"))
