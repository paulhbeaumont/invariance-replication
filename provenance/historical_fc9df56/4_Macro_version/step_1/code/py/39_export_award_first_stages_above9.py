#!/usr/bin/env python3
"""Export award-year first-stage specifications with F-statistics above nine."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


SOURCE = dc.POST_GRID_LABELED
DETAIL = dc.LOG_DIR / "post_first_stage_award_above9.csv"
COMPACT = dc.LOG_DIR / "post_first_stage_award_structural_above9.csv"


def main():
    grid = pd.read_csv(SOURCE)
    award = grid[grid["flow"].eq("award")].copy()
    detail = award[award["fs_F"] > 9].sort_values("fs_F", ascending=False)

    structural = [
        "rung",
        "window_mode",
        "treatment",
        "horizon",
        "growth",
        "share_label",
    ]
    overall = (
        award.groupby(structural, sort=False)["fs_F"]
        .agg(F_min_all="min", F_median_all="median", F_max="max")
        .reset_index()
    )
    no_winsor = award[award["winsor_rule"].eq("none")][structural + ["fs_F"]]
    no_winsor = no_winsor.rename(columns={"fs_F": "F_no_winsor"})
    counts = (
        detail.groupby(structural, sort=False)
        .size()
        .rename("winsor_cells_above9")
        .reset_index()
    )
    best = detail.drop_duplicates(structural)[
        structural + ["fs_F", "winsor_rule", "winsor_group", "fs_coef", "fs_se"]
    ].rename(
        columns={
            "fs_F": "best_F",
            "winsor_rule": "best_winsor_rule",
            "winsor_group": "best_winsor_group",
            "fs_coef": "best_fs_coef",
            "fs_se": "best_fs_se",
        }
    )
    compact = counts.merge(overall, on=structural, validate="one_to_one")
    compact = compact.merge(no_winsor, on=structural, validate="one_to_one")
    compact = compact.merge(best, on=structural, validate="one_to_one")
    compact = compact.sort_values("best_F", ascending=False)

    detail.to_csv(DETAIL, index=False)
    compact.to_csv(COMPACT, index=False)

    assert detail["flow"].eq("award").all()
    assert detail["fs_F"].gt(9).all()
    assert not detail.duplicated(structural + ["winsor_rule", "winsor_group"]).any()
    assert not compact.duplicated(structural).any()
    print(f"{len(detail)} individual specifications -> {DETAIL}")
    print(f"{len(compact)} structural families -> {COMPACT}")


if __name__ == "__main__":
    main()
