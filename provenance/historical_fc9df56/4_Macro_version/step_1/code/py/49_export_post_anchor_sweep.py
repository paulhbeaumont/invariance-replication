#!/usr/bin/env python3
"""Reshape the post-1998 anchor sweep into the three tables that answer it.

Estimates nothing. 47 produced 162 first stages and a 126-row
leave-one-window-out band; this file turns them into the headline the request
asked for, the decomposition that says which lever moved the instrument, and
the verdict table that applies the rule 46 froze before 47 ran.

    post_anchor_sweep_headline.csv        27 rows, the coupled diagonal
    post_anchor_sweep_decomposition.csv   18 rows, the 3x3 per family
    post_anchor_sweep_verdict.csv         18 rows, the pre-committed rule

Two things drive the shape of these tables.

fs_F is not comparable across anchors. n_obs is 30,800/27,720/27,720 at h=2 and
15,400/12,320/12,320 at h=4, and F scales with N at a fixed partial
correlation, so a 20 percent F gap at h=4 could be nothing but the 20 percent
of the sample that anchor 1999 does not have. fs_rho carries the comparison and
fs_F answers the separate question of whether the cell clears the
weak-instrument bar. h=3 is the only horizon where all three anchors have six
windows, so it is the one to read first.

The unwinsorized cell is not a footnote. The anchor changes the F > 9 verdict
under rule none and does not under rule tail1, so the finding is the
interaction and both rules are reported side by side rather than one being the
headline and the other an audit.

Outputs, mirrored to the Dropbox macro scratch folder so a co-author without
the git checkout can read them:
    batch_logs/macro_version/post_anchor_sweep_{headline,decomposition,verdict}.csv
    batch_logs/macro_version/49_export_post_anchor_sweep.log
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


RESULTS = dc.LOG_DIR / "post_anchor_sweep_first_stage.csv"
LOWO = dc.LOG_DIR / "post_anchor_sweep_lowo.csv"
PAIRING_AUDIT = dc.LOG_DIR / "post_anchor_sweep_pairing_audit.csv"
PANEL_AUDIT = dc.LOG_DIR / "post_anchor_sweep_panel_audit.csv"
HEADLINE_OUT = dc.LOG_DIR / "post_anchor_sweep_headline.csv"
DECOMPOSITION_OUT = dc.LOG_DIR / "post_anchor_sweep_decomposition.csv"
VERDICT_OUT = dc.LOG_DIR / "post_anchor_sweep_verdict.csv"
LOG = dc.LOG_DIR / "49_export_post_anchor_sweep.log"

F_THRESHOLD = 9
SCHEDULE_ANCHORS = (1998, 1999, 2000)
EXPOSURE_ANCHORS = (1998, 1999, 2000)
SHARE_LENGTHS = (4, 5, 10)
HORIZONS = (2, 3, 4)
RULES = ("none", "tail1")
PINNED_ANCHOR = 1998
EXPECTED_CELLS = 162
EXPECTED_HEADLINE = 27
EXPECTED_FAMILIES = 18


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


def build_headline(results: pd.DataFrame, panel_audit: pd.DataFrame):
    """One row per coupled family: exposure anchor == schedule anchor."""
    diagonal = results[results["is_diagonal"].eq(1)]
    key = ["sched_anchor", "horizon", "share_len"]
    wide = diagonal.pivot_table(
        index=key,
        columns="winsor_rule",
        values=["fs_F", "fs_rho", "fs_coef", "fs_se"],
        aggfunc="first",
    )
    wide.columns = [f"{statistic}_{rule}" for statistic, rule in wide.columns]
    wide = wide.reset_index()
    context = diagonal.drop_duplicates(key)[
        key + ["n_windows", "n_obs", "df_absorbed", "first_outcome_year", "last_outcome_year"]
    ]
    headline = context.merge(wide, on=key, validate="one_to_one")
    seam = panel_audit[
        ["sched_anchor", "horizon", "frac_outcome_years_post_seam", "n_unused_tail_years"]
    ]
    headline = headline.merge(seam, on=["sched_anchor", "horizon"], validate="many_to_one")
    for rule in RULES:
        headline[f"above9_{rule}"] = headline[f"fs_F_{rule}"] > F_THRESHOLD
    # h=3 is the only horizon where the three anchors share a window count, so
    # it is the only one where an F gap cannot be a sample-size artifact.
    balanced = (
        headline.groupby("horizon")["n_windows"].transform("nunique").eq(1)
    )
    headline["n_windows_balanced"] = balanced
    return headline.sort_values(["horizon", "share_len", "sched_anchor"]).reset_index(
        drop=True
    )


def build_decomposition(results: pd.DataFrame):
    """The 3x3 per family: schedule down the rows, exposure across the columns.

    Reading down a column moves the start-year phase with the exposure window
    held fixed. Reading across a row moves the exposure window with the phase
    held. The coupled anchor moves both at once and cannot tell them apart.
    """
    rows = []
    for horizon in HORIZONS:
        for length in SHARE_LENGTHS:
            for rule in RULES:
                block = results[
                    results["horizon"].eq(horizon)
                    & results["share_len"].eq(length)
                    & results["winsor_rule"].eq(rule)
                ]
                if len(block) != len(SCHEDULE_ANCHORS) * len(EXPOSURE_ANCHORS):
                    raise AssertionError(
                        f"h={horizon} fixed{length} {rule} has {len(block)} cells"
                    )
                row = {"horizon": horizon, "share_len": length, "winsor_rule": rule}
                grid_f, grid_rho = {}, {}
                for sched in SCHEDULE_ANCHORS:
                    for exposure in EXPOSURE_ANCHORS:
                        cell = block[
                            block["sched_anchor"].eq(sched)
                            & block["exposure_anchor"].eq(exposure)
                        ].iloc[0]
                        grid_f[(sched, exposure)] = float(cell["fs_F"])
                        grid_rho[(sched, exposure)] = float(cell["fs_rho"])
                        row[f"F_s{sched}_e{exposure}"] = float(cell["fs_F"])
                        row[f"rho_s{sched}_e{exposure}"] = float(cell["fs_rho"])

                # Schedule leg: vary sched with exposure held, worst case over
                # the three exposures. Exposure leg: the mirror image.
                schedule_f = max(
                    max(grid_f[(s, e)] for s in SCHEDULE_ANCHORS)
                    - min(grid_f[(s, e)] for s in SCHEDULE_ANCHORS)
                    for e in EXPOSURE_ANCHORS
                )
                exposure_f = max(
                    max(grid_f[(s, e)] for e in EXPOSURE_ANCHORS)
                    - min(grid_f[(s, e)] for e in EXPOSURE_ANCHORS)
                    for s in SCHEDULE_ANCHORS
                )
                schedule_rho = max(
                    max(grid_rho[(s, e)] for s in SCHEDULE_ANCHORS)
                    - min(grid_rho[(s, e)] for s in SCHEDULE_ANCHORS)
                    for e in EXPOSURE_ANCHORS
                )
                exposure_rho = max(
                    max(grid_rho[(s, e)] for e in EXPOSURE_ANCHORS)
                    - min(grid_rho[(s, e)] for e in EXPOSURE_ANCHORS)
                    for s in SCHEDULE_ANCHORS
                )
                coupled_f = [grid_f[(a, a)] for a in SCHEDULE_ANCHORS]
                coupled_rho = [grid_rho[(a, a)] for a in SCHEDULE_ANCHORS]
                row.update(
                    {
                        "schedule_leg_F_range": schedule_f,
                        "exposure_leg_F_range": exposure_f,
                        "schedule_leg_rho_range": schedule_rho,
                        "exposure_leg_rho_range": exposure_rho,
                        "coupled_F_range": max(coupled_f) - min(coupled_f),
                        "coupled_rho_range": max(coupled_rho) - min(coupled_rho),
                        "schedule_share_of_coupled_rho": (
                            schedule_rho / (schedule_rho + exposure_rho)
                            if schedule_rho + exposure_rho > 0
                            else np.nan
                        ),
                        "dominant_leg": (
                            "schedule" if schedule_rho > exposure_rho else "exposure"
                        ),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_verdict(
    results: pd.DataFrame, lowo: pd.DataFrame, pairing_audit: pd.DataFrame
):
    """Apply the rule 46 froze into the design JSON before 47 ran."""
    rows = []
    for horizon in HORIZONS:
        for length in SHARE_LENGTHS:
            for rule in RULES:
                coupled = results[
                    results["is_diagonal"].eq(1)
                    & results["horizon"].eq(horizon)
                    & results["share_len"].eq(length)
                    & results["winsor_rule"].eq(rule)
                ]
                if len(coupled) != len(SCHEDULE_ANCHORS):
                    raise AssertionError(
                        f"h={horizon} fixed{length} {rule} has {len(coupled)} "
                        "coupled cells"
                    )
                band = lowo[
                    lowo["horizon"].eq(horizon)
                    & lowo["share_len"].eq(length)
                    & lowo["winsor_rule"].eq(rule)
                ]
                if band.empty:
                    raise AssertionError(
                        f"no leave-one-window-out band for h={horizon} "
                        f"fixed{length} {rule}"
                    )
                anchor_rho_range = float(
                    coupled["fs_rho"].max() - coupled["fs_rho"].min()
                )
                lowo_rho_range = float(band["fs_rho"].max() - band["fs_rho"].min())
                above = coupled["fs_F"] > F_THRESHOLD
                f9_flips = bool(above.nunique() > 1)
                sign_flips = bool(np.sign(coupled["fs_coef"]).nunique() > 1)
                outside = bool(anchor_rho_range > lowo_rho_range)
                # The pairing that reaches one year into its own outcome window
                # is (sched 1998, exposure 2000); it is off the diagonal, so it
                # never touches this table. Recorded so the reader knows.
                overlap = int(
                    pairing_audit[
                        pairing_audit["is_diagonal"]
                        & pairing_audit["share_len"].eq(length)
                    ]["exposure_overlap_years"].max()
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "share_len": length,
                        "winsor_rule": rule,
                        "n_windows_balanced": bool(
                            coupled["n_windows"].nunique() == 1
                        ),
                        "n_windows_min": int(coupled["n_windows"].min()),
                        "n_windows_max": int(coupled["n_windows"].max()),
                        "F_min": float(coupled["fs_F"].min()),
                        "F_max": float(coupled["fs_F"].max()),
                        "cells_above9": int(above.sum()),
                        "rho_min": float(coupled["fs_rho"].min()),
                        "rho_max": float(coupled["fs_rho"].max()),
                        "anchor_rho_range": anchor_rho_range,
                        "lowo_rho_range": lowo_rho_range,
                        "lowo_F_min": float(band["fs_F"].min()),
                        "lowo_F_max": float(band["fs_F"].max()),
                        "a_f9_verdict_flips": f9_flips,
                        "b_coef_sign_flips": sign_flips,
                        "c_rho_outside_lowo_band": outside,
                        "anchor_moves_first_stage": bool(
                            f9_flips or sign_flips or outside
                        ),
                        "verdict": (
                            "outside_lowo_band" if outside else "inside_lowo_band"
                        ),
                        "diagonal_exposure_overlap_years": overlap,
                    }
                )
    verdict = pd.DataFrame(rows)
    # At h=2 the anchor-2000 schedule is the anchor-1998 schedule minus the
    # single start year 1998, so that one contrast moves the exposure with the
    # phase held. It is the cleanest cell in the grid and is named on the row.
    verdict["h2_anchor2000_shares_1998_phase"] = verdict["horizon"].eq(2)
    return verdict


def mirror(frame: pd.DataFrame, name: str):
    try:
        dc.MACRO_VERSION_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        target = dc.MACRO_VERSION_TEMP_DIR / name
        frame.to_csv(target, index=False)
        print(f"  mirrored -> {target}")
    except OSError as error:
        print(f"  mirror skipped ({error})")


def main():
    required = [RESULTS, LOWO, PAIRING_AUDIT, PANEL_AUDIT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required export inputs missing: " + ", ".join(missing))

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        results = pd.read_csv(RESULTS)
        lowo = pd.read_csv(LOWO)
        pairing_audit = pd.read_csv(PAIRING_AUDIT)
        panel_audit = pd.read_csv(PANEL_AUDIT)
        if len(results) != EXPECTED_CELLS:
            raise AssertionError(
                f"first-stage grid has {len(results)} rows, expected {EXPECTED_CELLS}"
            )
        if not results["status"].eq("ok").all():
            raise AssertionError("first-stage grid contains failed regressions")

        headline = build_headline(results, panel_audit)
        if len(headline) != EXPECTED_HEADLINE:
            raise AssertionError(
                f"headline has {len(headline)} rows, expected {EXPECTED_HEADLINE}"
            )
        decomposition = build_decomposition(results)
        if len(decomposition) != EXPECTED_FAMILIES:
            raise AssertionError(
                f"decomposition has {len(decomposition)} rows, "
                f"expected {EXPECTED_FAMILIES}"
            )
        verdict = build_verdict(results, lowo, pairing_audit)
        if len(verdict) != EXPECTED_FAMILIES:
            raise AssertionError(
                f"verdict has {len(verdict)} rows, expected {EXPECTED_FAMILIES}"
            )

        headline.to_csv(HEADLINE_OUT, index=False)
        print(f"{len(headline)} coupled families -> {HEADLINE_OUT}")
        mirror(headline, HEADLINE_OUT.name)
        decomposition.to_csv(DECOMPOSITION_OUT, index=False)
        print(f"{len(decomposition)} decompositions -> {DECOMPOSITION_OUT}")
        verdict.to_csv(VERDICT_OUT, index=False)
        print(f"{len(verdict)} verdicts -> {VERDICT_OUT}")
        mirror(verdict, VERDICT_OUT.name)

        print("\nCOUPLED HEADLINE, fs_F by anchor")
        print("  h=3 is the only horizon with a balanced window count.")
        for rule in RULES:
            print(f"\n  winsor_rule = {rule}")
            print(
                f"    {'h':>2s} {'len':>4s} "
                + "".join(f"{a:>12d}" for a in SCHEDULE_ANCHORS)
                + f"  {'above9':>7s} {'balanced':>9s}"
            )
            for horizon in HORIZONS:
                for length in SHARE_LENGTHS:
                    block = headline[
                        headline["horizon"].eq(horizon)
                        & headline["share_len"].eq(length)
                    ].set_index("sched_anchor")
                    values = "".join(
                        f"{block.loc[a, f'fs_F_{rule}']:>12.3f}"
                        for a in SCHEDULE_ANCHORS
                    )
                    above = int(block[f"above9_{rule}"].sum())
                    balanced = bool(block["n_windows_balanced"].iloc[0])
                    print(
                        f"    {horizon:>2d} {length:>4d} {values}  "
                        f"{above:>4d}/3 {str(balanced):>9s}"
                    )

        print("\nDECOMPOSITION, which leg moved the partial correlation")
        print(
            f"    {'h':>2s} {'len':>4s} {'rule':>6s} {'schedule':>10s} "
            f"{'exposure':>10s} {'sched share':>12s} {'dominant':>10s}"
        )
        for row in decomposition.itertuples(index=False):
            print(
                f"    {row.horizon:>2d} {row.share_len:>4d} {row.winsor_rule:>6s} "
                f"{row.schedule_leg_rho_range:>10.5f} "
                f"{row.exposure_leg_rho_range:>10.5f} "
                f"{row.schedule_share_of_coupled_rho:>12.3f} "
                f"{row.dominant_leg:>10s}"
            )

        print("\nVERDICT, against the rule frozen in the design JSON")
        print(
            f"    {'h':>2s} {'len':>4s} {'rule':>6s} {'anchor rho':>11s} "
            f"{'lowo rho':>10s} {'F>9 flip':>9s} {'sign flip':>10s} "
            f"{'moves':>6s}"
        )
        for row in verdict.itertuples(index=False):
            print(
                f"    {row.horizon:>2d} {row.share_len:>4d} {row.winsor_rule:>6s} "
                f"{row.anchor_rho_range:>11.5f} {row.lowo_rho_range:>10.5f} "
                f"{str(row.a_f9_verdict_flips):>9s} "
                f"{str(row.b_coef_sign_flips):>10s} "
                f"{str(row.anchor_moves_first_stage):>6s}"
            )
        moved = int(verdict["anchor_moves_first_stage"].sum())
        print(
            f"\n  the anchor moves the first stage in {moved}/{len(verdict)} "
            "families under the pre-committed rule"
        )
        for rule in RULES:
            block = verdict[verdict["winsor_rule"].eq(rule)]
            print(
                f"    winsor_rule {rule:<5s}: "
                f"{int(block['anchor_moves_first_stage'].sum())}/{len(block)} families, "
                f"F>9 in {int(block['cells_above9'].sum())}/{3 * len(block)} coupled cells"
            )
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
