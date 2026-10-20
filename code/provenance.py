"""Source identification shared by the simulation writers and the data exporter.

SIMULATION_CODE lists the modules that determine the simulated dynamics and the saved
measurements. digest() hashes file names and bytes with LF line ends, so the value does not
depend on git's line-ending conversion; code_revision() returns 'git:<commit>' inside a git
checkout and 'tree:<digest of SIMULATION_CODE>' otherwise.
"""
import os
import hashlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SIMULATION_CODE = ("barkley3d.py", "ellipse_ring.py", "panfilov.py", "atri.py", "oregonator.py",
                   "pan_ring.py", "atri_ring.py", "circle_run.py", "straight_filament.py", "twist_run.py",
                   "movie_frames.py", "param_probe.py", "meander_check.py")


def digest(contents):
    """contents: iterable of (file name, bytes); SHA-256 over sorted names and LF-normalized bytes."""
    h = hashlib.sha256()
    for name, data in sorted(contents):
        h.update(name.encode())
        h.update(data.replace(bytes([13, 10]), bytes([10])))
    return h.hexdigest()


def files_digest(names, folder=HERE):
    return digest((n, open(os.path.join(folder, n), "rb").read()) for n in names
                  if os.path.exists(os.path.join(folder, n)))


def git(*args, folder=HERE):
    try:
        r = subprocess.run(["git", "-C", folder, *args], capture_output=True, timeout=20)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def commit_digest(commit, names=SIMULATION_CODE, prefix="code/"):
    """Digest of the named files as stored in a git commit (None if unavailable)."""
    out = []
    for n in names:
        data = git("show", f"{commit}:{prefix}{n}")
        if data is None:
            return None
        out.append((n, data))
    return digest(out)


def code_revision():
    rev = git("rev-parse", "HEAD")
    if rev:
        return "git:" + rev.decode().strip()
    return "tree:" + files_digest(SIMULATION_CODE)
