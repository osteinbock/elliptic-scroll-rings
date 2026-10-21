"""Recompute every quoted number from the compact data archive alone (no simulations, no snapshots).

Copies the per-frame run tables of <archive>/runs/ into working folders, reruns all analyses that
do not need filament snapshots or fields (the buckling and tip-phase checks run from their saved
series; the two-dimensional spirals enter through their archived analysis, spiral2d/*.json),
recomputes paper_numbers.json, and compares it with the archived one.
    python rebuild_from_archive.py ARCHIVE [WORKDIR]      (exit status 1 on any difference)
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd, folders):
    env = dict(os.environ, PYTHONPATH=HERE, **folders)
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed in {cwd}: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def close(a, b, tol=1e-9):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def main(archive, work=None):
    archive = os.path.abspath(archive)
    work = os.path.abspath(work or tempfile.mkdtemp(prefix="rebuild_"))
    S, B, RF, RV = (os.path.join(work, g) for g in ("supp", "barkley", "refine", "revision"))
    fold = dict(SUPP=S, BARKLEY=B, REFINE=RF, REVISION=RV)
    for group, dst in (("supp", S), ("barkley", B), ("refine", RF), ("revision", RV)):
        shutil.copytree(os.path.join(archive, "runs", group), dst, dirs_exist_ok=True)
    shutil.copytree(os.path.join(archive, "analysis", "revision", "spiral2d"), os.path.join(RV, "spiral2d"), dirs_exist_ok=True)
    # saved series for the two snapshot-based checks
    os.makedirs(os.path.join(B, "sf2"), exist_ok=True)
    shutil.copyfile(os.path.join(archive, "analysis", "barkley", "sf2", "buckle.npz"), os.path.join(B, "sf2", "buckle_series.npz"))
    shutil.copyfile(os.path.join(archive, "analysis", "barkley", "diffrot.npz"), os.path.join(B, "diffrot_bins.npz"))
    for cmd in (["alpha_stationarity.py"], ["period_analysis_r2.py"], ["mode_decompose.py"], ["new_tests_analysis.py"],
                ["sf2_linear.py"], ["buckle_save.py", "--from-series", "sf2/buckle_series.npz"],
                ["diffrot_check.py", "--from-bins", "diffrot_bins.npz"]):
        run([os.path.join(HERE, cmd[0])] + cmd[1:], B, fold)
    run([os.path.join(HERE, "halfcycle_tables.py"), S], work, fold)
    run([os.path.join(HERE, "refine_compare.py"), B, RF], work, fold)
    run([os.path.join(HERE, "mode_definitions.py")], RV, fold)
    run([os.path.join(HERE, "controls_check.py")], work, fold)
    run([os.path.join(HERE, "halfcycle_uniform.py"), os.path.join(RV, "uniform_halfcycles.csv")], work, fold)
    for cmd in (["period_baselines.py"], ["mode_refit.py"], ["rigidity_test.py"],
                ["local_law_halfcycles.py", "local_law_halfcycles.csv"]):
        run([os.path.join(HERE, cmd[0])] + cmd[1:], B, fold)
    run([os.path.join(HERE, "paper_numbers.py"), S, B, RF, RV], work, fold)
    new = json.load(open(os.path.join(B, "paper_numbers.json")))
    old = json.load(open(os.path.join(archive, "analysis", "barkley", "paper_numbers.json")))
    diff = [k for k in sorted(set(old) | set(new)) if not close(old.get(k), new.get(k))]
    print(f"rebuilt {len(new)} numbers in {work}; {len(diff)} differ from the archive")
    for k in diff:
        print(f"  {k}: archive {old.get(k)!r}  rebuilt {new.get(k)!r}")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
