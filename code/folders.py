"""Data folders of the analyses, read from environment variables that build_all.sh and
rebuild_from_archive.py set:

    SUPP       outputs of runs_v2.sh (Panfilov-Rudenko and Atri models)
    BARKLEY    outputs of runs_barkley_v2.sh
    REFINE     outputs of runs_refine.sh (grid refinement)
    REVISION   outputs of runs_revision.sh

There are no defaults, so that an analysis never reads a folder of older runs by accident.
    import folders; folders.BARKLEY      # absolute path; RuntimeError if the variable is unset
"""
import os

RUN_LISTS = dict(SUPP="runs_v2.sh", BARKLEY="runs_barkley_v2.sh", REFINE="runs_refine.sh",
                 REVISION="runs_revision.sh")


def __getattr__(name):
    if name not in RUN_LISTS:
        raise AttributeError(name)
    path = os.environ.get(name)
    if not path:
        raise RuntimeError(f"set {name} to the folder of the {RUN_LISTS[name]} outputs "
                           "(build_all.sh sets SUPP, BARKLEY, REFINE and REVISION)")
    if not os.path.isdir(path):
        raise RuntimeError(f"{name} = {path} is not a folder")
    return os.path.abspath(path)
