#!/usr/bin/env python3
"""Report the symmetric first-stage F across the two shifter denominators.

The 39_export_award_first_stages_above9.py analogue for the symmetric
nonoverlapping grid. Like 39 this file estimates nothing: it reads 42's
72-row result CSV and reshapes it.

WHAT fs_F IS HERE

The single-instrument Wald (b/se)^2 under county clustering, the same statistic
39 reports and the same one every other step-1 script reports. There is no
Kleibergen-Paap, no effective F and no weak-instrument-robust inference in this
file, because there is none anywhere in step 1 - README register item 9 is open
and this file does not close it.

fs_F is the only column comparable across the denominator axis. The two
denominators put the instrument on scales four or five orders of magnitude
apart, so fs_coef and fs_se move with the units; F does not, and 43's check 19
is the standing proof of that.

WHY THE STRUCTURAL FILE COVERS EVERY FAMILY

39's compact output is built from the specifications that clear F > 9, so a
family with nothing above the threshold does not appear in it at all. README
register item 1 is about exactly that reading - selection on first-stage
strength, over a winsorization axis that transforms the dependent variable.
This file writes the structural table for ALL 36 families, above the threshold
or not, and keeps 39's F_min_all / F_median_all / F_no_winsor columns beside
best_F so the maximum can never be read on its own. The above-9 file is kept
because it is what was asked for, not because 9 selects anything here.

With only two winsorization cells per family, F_min_all and F_median_all carry
much less information than they do in 39's ten-cell families. The console
summary says so rather than letting the column names imply otherwise.

WHERE THE STRUCTURAL TABLE GOES

batch_logs/ is the git checkout, so a co-author who has the Dropbox project and
not the repo cannot see anything written only there. The structural table is the
one meant to be read outside the checkout, so it is mirrored to the macro
application's Dropbox scratch folder as well. The repo copy stays authoritative
and the mirror is rewritten on every run; nothing reads the mirror back.

Outputs:
    batch_logs/macro_version/symmetric_first_stage_above9.csv
    batch_logs/macro_version/symmetric_first_stage_structural.csv
    batch_logs/macro_version/symmetric_first_stage_denominator_comparison.csv
    batch_logs/macro_version/45_export_symmetric_first_stages.log
    Dropbox project: 4_Macro_version/data/temp/symmetric_first_stage_structural.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


SOURCE = dc.LOG_DIR / "symmetric_nonoverlap_first_stage.csv"
DETAIL = dc.LOG_DIR / "symmetric_first_stage_above9.csv"
STRUCTURAL = dc.LOG_DIR / "symmetric_first_stage_structural.csv"
STRUCTURAL_DROPBOX = (
    dc.MACRO_VERSION_TEMP_DIR / "symmetric_first_stage_structural.csv"
)
COMPARISON = dc.LOG_DIR / "symmetric_first_stage_denominator_comparison.csv"
LOG = dc.LOG_DIR / "45_export_symmetric_first_stages.log"

# Inherited from 39, which uses 9 rather than the conventional 10. The value is
# undocumented there; it is carried here only so the two exports read the same
# way, and nothing in this pipeline is selected on it.
F_THRESHOLD = 9

STRUCTURAL_KEY = ["period", "horizon", "share_label", "denominator"]
DENOMINATORS = ("county_income", "national_income")
EXPECTED_CELLS = 72
EXPECTED_FAMILIES = 36


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
        raise AssertionError(
            f"result grid has {len(grid)} rows, expected {EXPECTED_CELLS}"
        )
    if not grid["status"].eq("ok").all():
        raise AssertionError("result grid contains failed regressions")
    if sorted(grid["denominator"].unique()) != sorted(DENOMINATORS):
        raise AssertionError(
            f"result grid denominators are {sorted(grid['denominator'].unique())}"
        )
    if grid.duplicated(STRUCTURAL_KEY + ["winsor_rule", "winsor_group"]).any():
        raise AssertionError("result grid is not unique on the reporting key")
    return grid


def structural_table(grid: pd.DataFrame):
    """One row per family, written whether or not anything clears the threshold."""
    overall = (
        grid.groupby(STRUCTURAL_KEY, sort=False)["fs_F"]
        .agg(F_min_all="min", F_median_all="median", F_max="max")
        .reset_index()
    )
    no_winsor = grid[grid["winsor_rule"].eq("none")][STRUCTURAL_KEY + ["fs_F"]]
    no_winsor = no_winsor.rename(columns={"fs_F": "F_no_winsor"})
    above = grid[grid["fs_F"] > F_THRESHOLD].sort_values("fs_F", ascending=False)
    counts = (
        above.groupby(STRUCTURAL_KEY, sort=False)
        .size()
        .rename("winsor_cells_above9")
        .reset_index()
    )
    best = above.drop_duplicates(STRUCTURAL_KEY)[
        STRUCTURAL_KEY
        + ["fs_F", "winsor_rule", "winsor_group", "fs_coef", "fs_se"]
    ].rename(
        columns={
            "fs_F": "best_F",
            "winsor_rule": "best_winsor_rule",
            "winsor_group": "best_winsor_group",
            "fs_coef": "best_fs_coef",
            "fs_se": "best_fs_se",
        }
    )
    # Left merges, unlike 39's inner ones: a family with nothing above the
    # threshold keeps its row and reports zero rather than disappearing.
    table = overall.merge(no_winsor, on=STRUCTURAL_KEY, validate="one_to_one")
    table = table.merge(counts, on=STRUCTURAL_KEY, how="left", validate="one_to_one")
    table = table.merge(best, on=STRUCTURAL_KEY, how="left", validate="one_to_one")
    table["winsor_cells_above9"] = (
        table["winsor_cells_above9"].fillna(0).astype(int)
    )
    ordered = STRUCTURAL_KEY + [
        "winsor_cells_above9",
        "F_min_all",
        "F_median_all",
        "F_max",
        "F_no_winsor",
        "best_F",
        "best_winsor_rule",
        "best_winsor_group",
        "best_fs_coef",
        "best_fs_se",
    ]
    return above, table[ordered].sort_values("F_max", ascending=False)


def comparison_table(grid: pd.DataFrame):
    """The payoff table: the same cell under both denominators, side by side.

    fs_coef and fs_se are deliberately absent. They differ by the county-specific
    factor Y_ct / (Y^nat_t - Y_ct) and putting them in adjacent columns would
    invite a comparison that means nothing.
    """
    cell = ["period", "horizon", "share_label", "winsor_rule", "winsor_group"]
    wide = grid.pivot(index=cell, columns="denominator", values="fs_F").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "county_income": "F_county_income",
            "national_income": "F_national_income",
        }
    )
    if wide[["F_county_income", "F_national_income"]].isna().any().any():
        raise AssertionError("a cell is missing one of the two denominators")
    wide["F_ratio_national_over_county"] = (
        wide["F_national_income"] / wide["F_county_income"]
    )
    return wide.sort_values(
        ["winsor_rule", "period", "horizon", "share_label"]
    ).reset_index(drop=True)


def summarize(structural: pd.DataFrame, comparison: pd.DataFrame):
    print("first-stage F by denominator, all 36 families")
    for denominator in DENOMINATORS:
        block = structural[structural["denominator"].eq(denominator)]
        print(
            f"  {denominator:<16s} "
            f"F_no_winsor {block['F_no_winsor'].min():8.3f} to "
            f"{block['F_no_winsor'].max():8.3f}   "
            f"F_max {block['F_max'].min():8.3f} to {block['F_max'].max():8.3f}   "
            f"{int(block['winsor_cells_above9'].gt(0).sum())}/18 families "
            f"with a cell above {F_THRESHOLD}"
        )
    for rule in ("none", "tail1"):
        block = comparison[comparison["winsor_rule"].eq(rule)]
        print(
            f"  winsor={rule:<6s} ratio national/county: "
            f"min {block['F_ratio_national_over_county'].min():.4f}, "
            f"median {block['F_ratio_national_over_county'].median():.4f}, "
            f"max {block['F_ratio_national_over_county'].max():.4f}"
        )
    print(
        "  each family holds two winsorization cells, so F_min_all and "
        "F_median_all are the two cells themselves, not a distribution"
    )


def main():
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        grid = load()
        above, structural = structural_table(grid)
        comparison = comparison_table(grid)

        above.to_csv(DETAIL, index=False)
        structural.to_csv(STRUCTURAL, index=False)
        comparison.to_csv(COMPARISON, index=False)
        STRUCTURAL_DROPBOX.parent.mkdir(parents=True, exist_ok=True)
        structural.to_csv(STRUCTURAL_DROPBOX, index=False)

        assert above["fs_F"].gt(F_THRESHOLD).all()
        assert len(structural) == EXPECTED_FAMILIES
        assert not structural.duplicated(STRUCTURAL_KEY).any()
        assert len(comparison) == EXPECTED_FAMILIES
        assert structural["F_no_winsor"].notna().all()

        summarize(structural, comparison)
        print(f"{len(above)} specifications above {F_THRESHOLD} -> {DETAIL}")
        print(f"{len(structural)} structural families -> {STRUCTURAL}")
        print(f"{len(structural)} structural families -> {STRUCTURAL_DROPBOX}")
        print(f"{len(comparison)} paired cells -> {COMPARISON}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
