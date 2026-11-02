#!/usr/bin/env python3
"""The NS journey ladder as a Beamer slide fragment.

Renders from producer 52's saved ladder file and adds no estimate. One line per
row, the standard error in parentheses beside the coefficient, because the
twelve rows of the full table do not fit a slide at two lines each.

The two scaling blocks keep their own unit headers: the coefficient's unit
changes where the scaling denominator does, and the blocks are not comparable
in level.
"""
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
LADDER = dc.LOG_DIR / "journey_ladder.csv"
# Exhibits live in the shared Dropbox project, not in the checkout, like every
# other repointed producer. See dcads_common.MACRO_RESULTS_ROOT.
OUT = dc.MACRO_RESULTS_ROOT / "ns14_diagnostics" / \
    "Table_journey_ladder_slides.tex"

d = pd.read_csv(LADDER)
single = d[d["instrument"] == "single"].set_index("row")
comp = d[d["instrument"] == "composition"].set_index("row")
rows = sorted(single.index)
assert rows == sorted(comp.index), "the two instrument columns disagree on rows"

# The unit switches where the scaling denominator does; block the table there.
blocks = []
for r in rows:
    lab = single.loc[r, "unit_label"]
    if not blocks or blocks[-1][0] != lab:
        blocks.append((lab, []))
    blocks[-1][1].append(r)


def cell(s):
    return f"{s['beta_scaled']:.3f} ({s['se_scaled']:.3f})"


lines = []
for lab, rr in blocks:
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{6}{l}{\textit{Coefficient in %s}} \\"
                 % lab.replace("%", r"\%").replace("$", r"\$"))
    for r in rr:
        s, c = single.loc[r], comp.loc[r]
        lines.append(
            f"{r}. {s['change']} & {int(s['n_obs']):,} & {cell(s)} & "
            f"{s['first_stage_F']:.1f} & {cell(c)} & "
            f"{c['first_stage_F']:.1f} \\\\")

tex = "\n".join([
    r"% Slide table. From Nakamura and Steinsson's specification to ours.",
    r"% Rendered by 67_journey_ladder_slide.py from producer 52's ladder file."
    r" Do not edit.",
    # Both caps together fit the frame: twelve rows plus two block headers run
    # taller than a slide leaves above its bullets when scaled to full width.
    r"\adjustbox{max width=0.99\linewidth, max totalheight=0.55\textheight}{",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r" & & \multicolumn{2}{c}{Single shift} & \multicolumn{2}{c}{Composition} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"\textit{Change from the row above} & $N$ & $\hat\beta$ (SE) & $F$"
    r" & $\hat\beta$ (SE) & $F$ \\",
    r"\midrule",
    *lines,
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    "",
])
OUT.write_text(tex)
print(f"wrote {OUT.name} -> {OUT.parent}  ({len(rows)} rows, {len(blocks)} blocks)")
