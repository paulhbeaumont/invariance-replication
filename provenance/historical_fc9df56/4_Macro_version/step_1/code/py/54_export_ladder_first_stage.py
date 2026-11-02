#!/usr/bin/env python3
"""Render the aggregation-ladder first stages.

Estimates nothing. Reads 52's 72-row grid and writes the exhibit plus the two
CSVs that later runs read instead of re-estimating.

FOUR PANELS, BECAUSE THERE ARE TWO ADMISSIBLE ABSORPTIONS

Each era appears twice, once under each arm. `sxy` absorbs unit + state x year at
the county and commuting zone and clones the candidate package; `uy` absorbs
unit + year, which is what step_2's Tables 1-5 estimate on. Both are admissible
under protocol A.1.4 - the treatment varies at county x start year and each arm is
strictly coarser - and on identical rows they give a different F. Reporting only
the first left step_1's single typeset exhibit unable to be compared with the
application built on top of it, which is the gap these panels close.

WHY F LEADS AND BETA FOLLOWS

The request is the F at every rung, and F is also the only number comparable
across the winsorization arms. Capping the instrument at five interquartile
ranges removes most of its variance, so beta = cov(y, z) / var(z) rises by more
than an order of magnitude between the arms on the same data. Putting the two
betas side by side without the F above them would invite a reading of that ratio
as an economic magnitude when it is a property of the cap. F is invariant to any
positive rescaling of the instrument and is therefore the honest headline; beta
and its standard error are reported underneath so that neither arm is quoted
without its scale.

WHERE THE OUTPUTS GO

step_1 has never had a `result/` folder - every table in section 1.8 of its
README is hand-rendered markdown off a gitignored CSV. This file creates one,
because the request asks for the numbers in .tex under step_1 rather than only
in prose. The CSVs are mirrored to the Dropbox project as 45 and 49 do, so a
co-author with the data and not the checkout can still read them.

Outputs:
    step_1/result/TableF1_ladder_first_stage.tex
    step_1/result/_preview.tex
    batch_logs/macro_version/ladder_first_stage_headline.csv
    batch_logs/macro_version/54_export_ladder_first_stage.log
    Dropbox project: 4_Macro_version/data/temp/ladder_first_stage_headline.csv
    Dropbox project: 4_Macro_version/data/temp/ladder_first_stage.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


SOURCE = dc.LOG_DIR / "ladder_first_stage.csv"
RESULT_DIR = Path(__file__).resolve().parents[2] / "result"
TABLE = RESULT_DIR / "TableF1_ladder_first_stage.tex"
PREVIEW = RESULT_DIR / "_preview.tex"
HEADLINE = dc.LOG_DIR / "ladder_first_stage_headline.csv"
HEADLINE_DROPBOX = dc.MACRO_VERSION_TEMP_DIR / "ladder_first_stage_headline.csv"
GRID_DROPBOX = dc.MACRO_VERSION_TEMP_DIR / "ladder_first_stage.csv"
LOG = dc.LOG_DIR / "54_export_ladder_first_stage.log"

HEADLINE_HORIZON = 4
HORIZONS = (4, 3, 2)  # headline first, left to right
ARMS = ("none", "iq5")
FE_ARMS = ("sxy", "uy")
RUNGS = (("county", "County"), ("cz", "Commuting zone"), ("state", "State"))
# Four panels: each era under each absorption arm. The two arms are reported as
# co-equal panels rather than one headline and one robustness, because they answer
# different questions - `sxy` is the arm this ladder was frozen on and clones the
# candidate package, `uy` is the arm step_2's Tables 1-5 estimate on. A reader
# comparing step_1's F to step_2's needs the second.
ARM_TITLES = {
    "sxy": r"unit $+$ state $\times$ year",
    "uy": r"unit $+$ year",
}
PANELS = (
    ("pre", "sxy", "Panel A. Pre-1998 (1979--1997)"),
    ("post", "sxy", "Panel B. Post-1998 (1998--2018)"),
    ("pre", "uy", "Panel A. Pre-1998 (1979--1997)"),
    ("post", "uy", "Panel B. Post-1998 (1998--2018)"),
)
EXPECTED_CELLS = 72


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, value):
        self.stdout.write(value)
        self.file.write(value)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def load():
    grid = pd.read_csv(SOURCE)
    if len(grid) != EXPECTED_CELLS:
        raise AssertionError(f"grid has {len(grid)} rows, expected {EXPECTED_CELLS}")
    if not grid["status"].eq("ok").all():
        raise AssertionError("grid contains failed regressions")
    key = ["rung", "period", "horizon", "fe_arm", "winsor_rule"]
    if grid.duplicated(key).any():
        raise AssertionError(
            "grid is not unique on rung x period x horizon x fe_arm x rule"
        )
    return grid.set_index(key).sort_index()


def cell(grid, rung, period, horizon, fe_arm, arm):
    try:
        return grid.loc[(rung, period, horizon, fe_arm, arm)]
    except KeyError as error:
        raise AssertionError(
            f"missing cell: {rung} {period} h={horizon} fe={fe_arm} {arm}"
        ) from error


def header():
    """The column stub, repeated in each arm's tabular."""
    return [
        r"\begin{tabular}{@{\extracolsep{\fill}}l*{6}{c}}",
        r"\toprule",
        "                &\\multicolumn{2}{c}{$h=4$}&\\multicolumn{2}{c}{$h=3$}"
        "&\\multicolumn{2}{c}{$h=2$}\\\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        "                "
        + "".join(
            r"&\multicolumn{1}{c}{Uncapped}&\multicolumn{1}{c}{IQR5}"
            for _ in HORIZONS
        )
        + r"\\",
        "                "
        + "".join(
            f"&\\multicolumn{{1}}{{c}}{{({2 * i + 1})}}"
            f"&\\multicolumn{{1}}{{c}}{{({2 * i + 2})}}"
            for i in range(len(HORIZONS))
        )
        + r"\\",
        r"\midrule",
    ]


def arm_block(grid, panels):
    """One tabular, holding both eras under a single absorption.

    Split by arm rather than emitted as one four-panel tabular because a tabular
    cannot break across a page: four panels overflowed the landscape body by
    251pt and silently ran off it. Two tabulars of two panels each are the same
    numbers, laid out so they can actually be read.
    """
    lines = list(header())
    for index, (period, fe_arm, title) in enumerate(panels):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}}\\\\")
        lines.append(r"\addlinespace")
        for rung, label in RUNGS:
            units = int(
                cell(grid, rung, period, HEADLINE_HORIZON, fe_arm, "none")["n_units"]
            )
            lines.append(
                f"\\textbf{{{label}}} ({units:,} units)"
                + "&" * 6
                + r"\\"
            )
            statistic = ["\\quad First-stage $F$"]
            estimate = ["\\quad $\\beta$"]
            error = ["\\quad "]
            counts = ["\\quad Observations"]
            for horizon in HORIZONS:
                for arm in ARMS:
                    row = cell(grid, rung, period, horizon, fe_arm, arm)
                    statistic.append(f"{row['fs_F']:9.2f}")
                    estimate.append(f"{row['fs_coef']:9.1f}")
                    error.append(f"({row['fs_se']:.1f})")
                observations = int(
                    cell(grid, rung, period, horizon, fe_arm, "none")["n_obs"]
                )
                counts.append(f"\\multicolumn{{2}}{{c}}{{{observations:,}}}")
            lines.append("&".join(statistic) + r"\\")
            lines.append("&".join(estimate) + r"\\")
            lines.append("&".join(error) + r"\\")
            lines.append("&".join(counts) + r"\\")
            lines.append(r"\addlinespace")
        windows = [
            int(cell(grid, "county", period, horizon, fe_arm, "none")["n_windows"])
            for horizon in HORIZONS
        ]
        lines.append(
            "Start years"
            + "".join(f"&\\multicolumn{{2}}{{c}}{{{w}}}" for w in windows)
            + r"\\"
        )
        last = index == len(panels) - 1
        lines.append(r"\bottomrule" if last else r"\midrule")

    lines.append(r"\end{tabular}")
    return lines


NOTE = (
    r"$\beta$ is the \emph{first-stage} coefficient, the shock on the instrument, "
    r"not a 2SLS employment response. Both absorptions are admissible: the "
    r"treatment varies at county $\times$ start year and each is strictly coarser "
    r"than that. At the \textbf{state} rung the unit is the state, so "
    r"state $\times$ year is not defined and the two arms are the same regression "
    r"--- the State block is identical between the two tables by construction, and "
    r"the pipeline asserts it at $0$. Step 2's Tables 1--5 estimate on the "
    r"unit $+$ year arm."
)


def table(grid):
    lines = [
        "% Table F1. First-stage F at every rung, on the candidate income instrument.",
        "% Rendered by 54_export_ladder_first_stage.py from "
        "batch_logs/macro_version/ladder_first_stage.csv. Do not edit.",
        "% h=4 headline, h=3 and h=2 robustness; exposure fixed5; nonoverlapping "
        "start years; no dollar floor.",
        "% TWO TABULARS, one per absorption arm. A tabular cannot break across a "
        "page, so all four panels in one ran 251pt off the body.",
        r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
    ]
    for position, fe_arm in enumerate(FE_ARMS):
        panels = [entry for entry in PANELS if entry[1] == fe_arm]
        spec = ARM_TITLES[fe_arm]
        lines.append("%")
        lines.append(f"% ---- absorbing {spec} ----")
        lines.append(
            r"\begin{minipage}{\textwidth}\centering"
            if position == 0
            else r"\begin{minipage}{\textwidth}\centering"
        )
        lines.append(
            f"\\textbf{{Table F1{'ab'[position]}. First-stage $F$ at every rung, "
            f"absorbing {spec}}}\\\\[0.4em]"
        )
        lines.extend(arm_block(grid, panels))
        lines.append(r"\end{minipage}")
        if position == 0:
            lines.append(r"\vspace{1.2em}")
    lines.append(r"\begin{minipage}{\textwidth}\vspace{0.4em}\footnotesize")
    lines.append(NOTE)
    lines.append(r"\end{minipage}")
    return "\n".join(lines) + "\n"


def preview():
    return "\n".join(
        [
            "% Throwaway wrapper so the table can be compiled on its own.",
            "% Rendered by 54_export_ladder_first_stage.py. Do not edit.",
            r"\documentclass[10pt]{article}",
            r"\usepackage{booktabs}",
            r"\usepackage{geometry}",
            r"\geometry{landscape,margin=1.5cm}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\begin{center}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\input{TableF1_ladder_first_stage}",
            r"\end{center}",
            r"\end{document}",
        ]
    ) + "\n"


def headline_table(grid):
    """The h=4 block, wide over the winsorization arm.

    This is the file later runs read instead of re-estimating: one row per rung
    and era, both arms beside each other, so the winsorization axis is a column
    rather than something to be reconstructed from a filter.
    """
    block = grid.reset_index()
    block = block[block["horizon"].eq(HEADLINE_HORIZON)]
    wide = block.pivot(
        index=[
            "rung", "period", "horizon", "fe_arm", "fe_spec",
            "share_label", "n_units", "n_windows", "n_obs",
        ],
        columns="winsor_rule",
        values=["fs_coef", "fs_se", "fs_F", "fs_rho"],
    )
    wide.columns = [f"{stat}_{arm}" for stat, arm in wide.columns]
    wide = wide.reset_index()
    order = {rung: i for i, (rung, _) in enumerate(RUNGS)}
    fe_order = {fe_arm: i for i, fe_arm in enumerate(FE_ARMS)}
    wide["_order"] = wide["rung"].map(order)
    wide["_fe"] = wide["fe_arm"].map(fe_order)
    wide = wide.sort_values(["_order", "period", "_fe"]).drop(
        columns=["_order", "_fe"]
    )
    return wide


def main():
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        grid = load()
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        TABLE.write_text(table(grid), encoding="utf-8")
        PREVIEW.write_text(preview(), encoding="utf-8")

        headline = headline_table(grid)
        headline.to_csv(HEADLINE, index=False)
        HEADLINE_DROPBOX.parent.mkdir(parents=True, exist_ok=True)
        headline.to_csv(HEADLINE_DROPBOX, index=False)
        grid.reset_index().to_csv(GRID_DROPBOX, index=False)

        expected_headline = len(RUNGS) * len({p for p, _, _ in PANELS}) * len(FE_ARMS)
        if len(headline) != expected_headline:
            raise AssertionError(
                f"headline has {len(headline)} rows, expected {expected_headline}"
            )

        flat = grid.reset_index()
        print("first-stage F, by absorption arm and winsorization arm")
        for fe_arm in FE_ARMS:
            for arm in ARMS:
                side = flat[
                    flat["fe_arm"].eq(fe_arm) & flat["winsor_rule"].eq(arm)
                ]
                print(
                    f"  fe={fe_arm:<4s} {arm:<5s} F from {side['fs_F'].min():8.2f} to "
                    f"{side['fs_F'].max():8.2f}, median {side['fs_F'].median():8.2f}"
                )
        print()
        print(f"h={HEADLINE_HORIZON} headline")
        for _, row in headline.iterrows():
            print(
                f"  {row['rung']:<7s} {row['period']:<5s} fe={row['fe_arm']:<4s} "
                f"{int(row['n_units']):>5,} units  "
                f"F uncapped {row['fs_F_none']:7.2f}   F IQR5 {row['fs_F_iq5']:8.2f}"
            )
        print()
        # The state rung is where the two arms coincide by construction. Printed
        # so the exhibit's own footnote can be checked against the numbers.
        state = headline[headline["rung"].eq("state")]
        for period in sorted(state["period"].unique()):
            pair = state[state["period"].eq(period)]
            gap = float(pair["fs_F_iq5"].max() - pair["fs_F_iq5"].min())
            print(f"  state {period}: |F(sxy) - F(uy)| = {gap:.3e} (must be 0)")
        print()
        print(f"table -> {TABLE}")
        print(f"preview wrapper -> {PREVIEW}")
        print(f"headline -> {HEADLINE}")
        print(f"headline mirror -> {HEADLINE_DROPBOX}")
        print(f"grid mirror -> {GRID_DROPBOX}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
