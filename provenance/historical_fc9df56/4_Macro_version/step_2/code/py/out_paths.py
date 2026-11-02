#!/usr/bin/env python3
"""Where a finished exhibit is written, and why it is written twice.

The project keeps code in a Git checkout outside Dropbox and keeps the shared
data and paper files in Dropbox. The co-authors read Dropbox; Git holds the
history. An exhibit needs both, so `write_exhibit` writes it to both in one call.

DIRECTION MATTERS, AND IT IS OPPOSITE FOR THE TWO KINDS OF FILE.

    Files the code makes (tables, figures):   Git  ->  Dropbox
    Files people write (paper, slides):       Dropbox  ->  Git

Nothing is ever written in both directions, so a producer cannot overwrite
something a co-author typed, and a co-author cannot silently revert a regenerated
table. `sync_from_dropbox.py` handles the other direction.

If the Dropbox project is not mounted, the write to Git still happens and the
caller is told the Dropbox copy was skipped, rather than the run failing on a
laptop that is offline.

RUNNING WITH DROPBOX HELD READ-ONLY. Set `INVARIANCE_SCRATCH_ROOT` to a local
directory and the whole pipeline stops writing into the shared project: the cell
cache and the Stata export are redirected under it (`mpgr_disagg.py`,
`41_export_cells_stata.py`, `stata/00_launcher.do`) and `write_exhibit` stops
mirroring, so a finished exhibit lands in the Git `Results/` folder and nowhere
else. Dropbox is still READ from - the employment cells and the certified panel
live there and are the inputs - so this is a write barrier, not an isolation
mode. It exists because a table produced on a branch that has not been reconciled
with a co-author's should not appear in the folder the co-authors read. Unset,
every path is exactly what it was.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
sys.path.insert(0, str(STEP1_CODE))

MACRO_ROOT = HERE.parents[3]   # py -> code -> step_2 -> 4_Macro_version
GIT_RESULTS = MACRO_ROOT / "Results"

SCRATCH_ENV = "INVARIANCE_SCRATCH_ROOT"


def scratch_root():
    """The local stand-in for the Dropbox data tree, or None if unset.

    Resolved here rather than in each consumer so that the cell cache, the Stata
    export and the exhibit mirror cannot disagree about whether a run is writing
    into Dropbox. Every caller asks this one function.
    """
    v = os.environ.get(SCRATCH_ENV, "").strip()
    return Path(v).expanduser().resolve() if v else None


def dropbox_results():
    """The mirror folder in the shared Dropbox project, or None if unmounted.

    Also None under `INVARIANCE_SCRATCH_ROOT`, which is what makes the write
    barrier cover finished exhibits and not just intermediates.
    """
    if scratch_root() is not None:
        return None
    try:
        import dcads_common as dc
        d = dc.DROPBOX_PROJECT_ROOT / MACRO_ROOT.name / "Results"
    except Exception:
        return None
    return d if d.parent.is_dir() else None


def write_exhibit(name, text, quiet=False, subdir=None):
    """Write one exhibit to the Git Results folder and mirror it into Dropbox.

    `subdir` puts the file in a folder under Results/ on BOTH sides, so the Git
    tree and the Dropbox mirror keep the same shape. Default None writes to the
    Results/ root, which is where every exhibit written before 2026-09 lives and
    where they stay.

    Returns (git_path, dropbox_path_or_None).
    """
    rel = Path(name) if subdir is None else Path(subdir) / name
    git_path = GIT_RESULTS / rel
    git_path.parent.mkdir(parents=True, exist_ok=True)
    git_path.write_text(text)

    drop = dropbox_results()
    drop_path = None
    if drop is not None:
        drop_path = drop / rel
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(git_path, drop_path)

    if not quiet:
        root = MACRO_ROOT.parent
        print(f"  wrote {git_path.relative_to(root)}")
        # The two reasons the mirror was skipped are not the same problem and must
        # not print the same line: one is a laptop that is offline, the other is a
        # deliberate write barrier.
        if drop_path is not None:
            tail = f"and mirrored to Dropbox/{MACRO_ROOT.name}/Results"
        elif scratch_root() is not None:
            tail = (f"{SCRATCH_ENV} is set, so the Dropbox mirror was skipped "
                    f"on purpose")
        else:
            tail = "DROPBOX NOT MOUNTED, the shared copy was not updated"
        print("        " + tail)
    return git_path, drop_path
