#!/usr/bin/env python3
"""Copy hand-written documents from Dropbox into the Git checkout, one way.

THE PAPER AND THE SLIDES ARE WRITTEN IN DROPBOX, because that is where the
co-authors work. Git holds their history. This script moves them in that
direction and only that direction: Dropbox is the live copy, Git is the record.
Nothing here ever writes into Dropbox, so no edit of yours or a co-author's can
be overwritten by a script.

Generated exhibits travel the other way, Git to Dropbox, and are handled by
`out_paths.write_exhibit`. Keeping one direction per kind of file is what stops
the two copies from fighting.

    python3 sync_from_dropbox.py                    # the default folders, dry run
    python3 sync_from_dropbox.py --apply            # actually copy
    python3 sync_from_dropbox.py --folder 4_Macro_version/paper --apply

LaTeX build artefacts are skipped: they are regenerated on every compile, they
are already gitignored, and copying them would put noise in every diff.
"""
import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[3] / "step_1" / "code" / "py"))
import dcads_common as dc  # noqa: E402

# Folders written by people, relative to the project root. A folder that does not
# exist on either side is reported and skipped rather than created.
DEFAULT_FOLDERS = ("4_Macro_version/document", "4_Macro_version/paper")

SKIP_SUFFIX = {".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".synctex.gz",
               ".fls", ".fdb_latexmk", ".nav", ".snm", ".vrb"}
SKIP_NAME = {".DS_Store"}

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--folder", nargs="+", default=list(DEFAULT_FOLDERS))
a.add_argument("--apply", action="store_true",
               help="perform the copy; without it the script only reports")
g = a.parse_args()

print(f"\n  DROPBOX -> GIT, {'applying' if g.apply else 'dry run'}")
print(f"  from {dc.DROPBOX_PROJECT_ROOT}")
print(f"  to   {REPO_ROOT}\n")

new = changed = same = 0
for rel in g.folder:
    src_dir = dc.DROPBOX_PROJECT_ROOT / rel
    dst_dir = REPO_ROOT / rel
    if not src_dir.is_dir():
        print(f"  [skip] {rel} does not exist in Dropbox")
        continue
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        if src.suffix in SKIP_SUFFIX or src.name in SKIP_NAME:
            continue
        dst = dst_dir / src.relative_to(src_dir)
        if not dst.exists():
            state, new = "NEW    ", new + 1
        elif not filecmp.cmp(src, dst, shallow=False):
            state, changed = "CHANGED", changed + 1
        else:
            same += 1
            continue
        print(f"  [{state}] {dst.relative_to(REPO_ROOT)}")
        if g.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

print(f"\n  {new} new, {changed} changed, {same} already identical")
if not g.apply and (new or changed):
    print("  nothing was copied. Rerun with --apply to copy, then commit.")
