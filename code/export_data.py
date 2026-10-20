"""Compact data archive for the repository: per-frame tables of all production runs and the
analysis products behind every figure, table, and quoted number.

For each run listed in runs_v2.sh (group "supp") and runs_barkley_v2.sh (group "barkley"), the
per-frame ring measures (masked by mask_frames.py, with validity flags and extraction diagnostics),
the pre-mask radius raw_Rm, and the run metadata are written to <outdir>/runs/<group>/<path>.npz;
filament snapshots, fields, and straight-filament profiles are omitted (the complex amplitudes
u_k are kept). Analysis products are copied to <outdir>/analysis/<group>/. <outdir>/MANIFEST.json
lists every file with its SHA-256, the command that produced it, and the source digest.

The input folders are only read: archive paths are built from the run-list paths relative to
each folder, every destination must lie inside <outdir> and differ from its source, source
hashes are taken before anything is written, and files are written atomically.
    python export_data.py OUTDIR SUPP BARKLEY REFINE [--sim-commit COMMIT]
(SUPP, BARKLEY, REFINE: folders of the runs_v2.sh / runs_barkley_v2.sh / runs_refine.sh outputs; COMMIT: the repository
commit whose simulation modules produced the runs, recorded in the manifest)
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
DROP = ("snap_", "u_", "v_", "zoff_")
KEEP_RAW = ("raw_Rm",)                      # pre-mask validity (methods statistics)
# analysis products: group -> [(path relative to the data folder, producing command)]
PRODUCTS = {
    "barkley": [
        ("period_table_r2.csv", "python period_analysis_r2.py"),
        ("mode_decompose.npy", "python mode_decompose.py"),
        ("mode_sweep.npy", "python mode_decompose.py"),
        ("new_tests.npz", "python new_tests_analysis.py"),
        ("alpha_late.npy", "python alpha_stationarity.py"),
        ("sf2/sf2_linear.npy", "python sf2_linear.py"),
        ("sf2/buckle.npz", "python buckle_save.py"),
        ("amp_factor.npy", "python period_analysis_r2.py (amplitude_factor cache)"),
        ("diffrot.npz", "python diffrot_check.py"),
        ("paper_numbers.json", "python paper_numbers.py SUPP BARKLEY"),
    ],
    "supp": [
        ("halfcycles_panfilov.csv", "python halfcycle_tables.py SUPP"),
        ("halfcycles_panfilov.csv.meta.json", "python halfcycle_tables.py SUPP"),
        ("halfcycles_atri.csv", "python halfcycle_tables.py SUPP"),
        ("halfcycles_atri.csv.meta.json", "python halfcycle_tables.py SUPP"),
        ("coefficients_circles.csv", "python halfcycle_tables.py SUPP"),
        ("coefficients_circles.csv.meta.json", "python halfcycle_tables.py SUPP"),
    ],
    "refine": [
        ("refine_compare.json", "python refine_compare.py BARKLEY REFINE"),
    ],
}
SIMULATION_CODE = ("barkley3d.py", "ellipse_ring.py", "panfilov.py", "atri.py", "oregonator.py",
                   "pan_ring.py", "atri_ring.py", "circle_run.py", "straight_filament.py", "twist_run.py",
                   "movie_frames.py", "param_probe.py", "meander_check.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_digest(files=None):
    """SHA-256 over the named source files (name and bytes with LF line ends, sorted): identifies
    the code whether or not it lies in a git checkout, independent of line-ending conversion."""
    h = hashlib.sha256()
    for f in sorted(files or [f for f in os.listdir(HERE) if f.endswith(".py")]):
        h.update(f.encode())
        h.update(open(os.path.join(HERE, f), "rb").read().replace(bytes([13, 10]), bytes([10])))
    return h.hexdigest()


def git_revision():
    try:
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def commands():
    """[(group, path relative to the run folder, command)] from the run lists."""
    out = []
    for lst, listed, group in (("runs_v2.sh", "v2", "supp"), ("runs_barkley_v2.sh", "v2b", "barkley"),
                               ("runs_refine.sh", "v5r", "refine")):
        for line in open(os.path.join(HERE, lst), encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"cd (\S+) && (.*)", line)
            sub = os.path.relpath(m.group(1), listed) if m else "."
            for name in re.findall(r"([\w.\-]+\.npz)", line):
                rel = os.path.normpath(os.path.join(sub, name))
                if os.path.isabs(rel) or rel.startswith(".."):
                    raise ValueError(f"run list entry leaves its folder: {line}")
                out.append((group, rel, line))
    return out


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def safe_dest(outdir, *parts):
    dst = os.path.join(outdir, *parts)
    if not inside(dst, outdir):
        raise ValueError(f"destination escapes the archive: {dst}")
    return dst


def atomic_savez(dst, arrays):
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dst)


def main(outdir, supp, barkley, refine, sim_commit=None):
    roots = {"supp": os.path.abspath(supp), "barkley": os.path.abspath(barkley), "refine": os.path.abspath(refine)}
    outdir = os.path.abspath(outdir)
    for g, r in roots.items():
        if inside(outdir, r) or inside(r, outdir):
            raise ValueError(f"archive {outdir} and input folder {r} must not contain each other")
    manifest = dict(git_revision=git_revision(), source_tree_sha256=source_digest(),
                    simulation_code_sha256=source_digest(SIMULATION_CODE),
                    simulation_code_files=list(SIMULATION_CODE),
                    production_simulation_commit=sim_commit,
                    note="hashes are of the bytes as written; data/ is stored with -text in git", runs=[], products=[])
    for group, rel, line in commands():
        src = os.path.join(roots[group], rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"run output missing: {src}")
        src_hash = sha256(src)                                  # before anything is written
        dst = safe_dest(outdir, "runs", group, rel)
        if os.path.realpath(dst) == os.path.realpath(src):
            raise ValueError(f"destination equals source: {src}")
        d = np.load(src)
        keep = {k: d[k] for k in d.files if (not k.startswith(DROP) and not k.startswith("raw_") and k != "prof")
                or k in KEEP_RAW}
        d.close()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        atomic_savez(dst, keep)
        if sha256(src) != src_hash:
            raise RuntimeError(f"source changed during export: {src}")
        manifest["runs"].append(dict(file=os.path.relpath(dst, outdir).replace("\\", "/"), group=group,
                                     source=rel.replace("\\", "/"), command=line, source_sha256=src_hash,
                                     sha256=sha256(dst), frames=int(len(keep["t"])) if "t" in keep else None))
    for group, items in PRODUCTS.items():
        for rel, cmd in items:
            src = os.path.join(roots[group], rel)
            if not os.path.exists(src):
                raise FileNotFoundError(f"analysis product missing: {src}")
            dst = safe_dest(outdir, "analysis", group, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
            manifest["products"].append(dict(file=os.path.relpath(dst, outdir).replace("\\", "/"), group=group,
                                             command=f"cd {group.upper()} && {cmd}", sha256=sha256(dst)))
    with open(os.path.join(outdir, "MANIFEST.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest['runs'])} run tables and {len(manifest['products'])} analysis products to {outdir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    commit = None
    if "--sim-commit" in args:
        i = args.index("--sim-commit")
        commit = args[i + 1]
        del args[i:i + 2]
    if len(args) != 4:
        sys.exit(__doc__)
    main(*args, sim_commit=commit)
