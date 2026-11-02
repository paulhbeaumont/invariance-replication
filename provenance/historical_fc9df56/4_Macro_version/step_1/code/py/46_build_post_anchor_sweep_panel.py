#!/usr/bin/env python3
"""Build post-1998 county first-stage panels over a sweep of anchor years.

41 anchors the post period at 1998, and that one number does two jobs: it ends
the fixed exposure window at anchor-1, and it sets the phase of the
nonoverlapping start-year schedule range(anchor, 2018-h+1, h). 1998 was chosen
for a data reason - the CBP NAICS break - and not an identification one, and
nothing in the pipeline tests whether the first stage survives moving it. This
file sweeps the anchor over 1998, 1999 and 2000 so 47 can find out.

WHY THE PANEL CARRIES NINE INSTRUMENTS AND NOT THREE

A coupled anchor moves the exposure window and the schedule phase together, so
a coupled-only grid cannot say which of the two moved the first stage. It is
not an academic worry: the exposure share is Pearson-correlated about 0.99
across anchors, and a time-invariant s_c that is 0.99 correlated with itself
cannot move a first stage after county fixed effects, so the coupled effect is
likely to be the schedule wearing the exposure's name.

Each of the nine (schedule anchor, horizon) panels therefore carries all nine
exposure columns, three exposure anchors by three window lengths:

    s{a}_{L}  fixed allocation share over [a-L, a-1], national denominator
    z{a}_{L}  s{a}_{L} x loo_future_ct / Y_ct

The diagonal, exposure anchor == schedule anchor, is the coupled headline. The
off-diagonal is the decomposition: read down a column to move the schedule with
the exposure held, across a row to move the exposure with the schedule held.
Building all nine is nearly free because the LHS residualization and the
reghdfe machinery in 47 dominate and are already paid.

The county_income denominator only, per the request. loo_income is kept in the
panel anyway so that the national_income follow-up - README register item 2 -
is a 47 edit rather than a rebuild.

THE ANCHOR-1998 PIN

Every schedule-anchor-1998 frame is compared column-for-column against
41.build_period_horizon("post", h, ...), called directly and unmodified. A
re-implementation bug is caught here, in the builder, rather than three scripts
later. The pin only ever exercises share_last == 1997, so the out-of-range
asserts below cover what it cannot.

Outputs:
    _data/temp/post_anchor_sweep/panel.parquet
    _data/temp/post_anchor_sweep/post_s<anchor>_h<h>.dta
    batch_logs/macro_version/post_anchor_sweep_design.json
    batch_logs/macro_version/post_anchor_sweep_share_audit.csv
    batch_logs/macro_version/post_anchor_sweep_panel_audit.csv
    batch_logs/macro_version/46_build_post_anchor_sweep_panel.log
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


np.random.seed(0)

CODE = Path(__file__).resolve().parent
SYMMETRIC_BUILDER = CODE / "41_build_symmetric_nonoverlap_panel.py"
TEMP = dc.POST_ANCHOR_SWEEP_DIR
LOG = dc.LOG_DIR / "46_build_post_anchor_sweep_panel.log"
DESIGN_OUT = dc.LOG_DIR / "post_anchor_sweep_design.json"
SHARE_AUDIT_OUT = dc.LOG_DIR / "post_anchor_sweep_share_audit.csv"
PANEL_AUDIT_OUT = dc.LOG_DIR / "post_anchor_sweep_panel_audit.csv"
PAIRING_AUDIT_OUT = dc.LOG_DIR / "post_anchor_sweep_pairing_audit.csv"

# The schedule anchor sets the first start year; the post period always ends at
# 2018, so later anchors simply get fewer windows and n_windows is reported on
# every row rather than the schedules being truncated to match.
SCHEDULE_ANCHORS = (1998, 1999, 2000)
EXPOSURE_ANCHORS = (1998, 1999, 2000)
POST_LAST = 2018
SHARE_LENGTHS = (4, 5, 10)
HORIZONS = (2, 3, 4)
# 41's post anchor. The sweep must reproduce it exactly.
PINNED_ANCHOR = 1998
EXPECTED_ANALYSIS_UNITS = 3_080
EXPECTED_STATES = 48


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


def load_numbered(path: Path):
    spec = importlib.util.spec_from_file_location("symmetric_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def share_window(anchor: int, length: int):
    end = anchor - 1
    return end - length + 1, end


def anchor_starts(anchor: int, horizon: int):
    return list(range(anchor, POST_LAST - horizon + 1, horizon))


def build_shares(
    contracts: pd.DataFrame,
    universe: set[int],
    analysis_units: set[int],
):
    """41's build_shares, re-anchored.

    41's version cannot be reused: its audit rows read the module-global
    PERIODS[period] and would raise on an anchor-keyed period. The arithmetic
    below is identical, and 48's share check verifies it against raw sums.
    """
    levels = contracts[contracts["ccg"].isin(universe)].pivot(
        index="ccg", columns="year", values="dollars_real"
    )
    shares = {}
    audit_rows = []
    for anchor in EXPOSURE_ANCHORS:
        for length in SHARE_LENGTHS:
            label = f"{anchor}_{length}"
            first, last = share_window(anchor, length)
            numerator = levels.loc[:, first:last].sum(axis=1)
            denominator = float(numerator.sum())
            if denominator == 0:
                raise AssertionError(f"zero national contract total for {label}")
            values = numerator / denominator
            shares[label] = values
            analysis = values.loc[values.index.isin(analysis_units)]
            audit_rows.append(
                {
                    "exposure_anchor": anchor,
                    "share_len": length,
                    "share_label": label,
                    "share_first": first,
                    "share_last": last,
                    "share_years": last - first + 1,
                    "years_between_share_and_anchor": anchor - last - 1,
                    # 0, 1 or 2. At fixed4 and anchor 2000 half the exposure
                    # window sits inside the post-1998 era, so contract
                    # persistence could raise F for a mechanical reason. The
                    # exposure leg of the 3x3 is what rules that in or out.
                    "share_years_after_1997": int(
                        sum(1 for y in range(first, last + 1) if y > 1997)
                    ),
                    "national_real_contracts": denominator,
                    "sum_universe_shares": float(values.sum()),
                    "sum_analysis_shares": float(analysis.sum()),
                    "zero_share_count_universe": int((values == 0).sum()),
                    "zero_share_count_analysis": int((analysis == 0).sum()),
                    # Signed deobligations leave a handful of counties with a
                    # negative allocation share, so their instrument has the
                    # wrong sign by construction. 41 never reported this.
                    "negative_share_count_analysis": int((analysis < 0).sum()),
                    "min_share_analysis": float(analysis.min()),
                }
            )
    return shares, pd.DataFrame(audit_rows)


def pair_exposure_with_schedule():
    """One row per (schedule anchor, exposure anchor, window length).

    The first outcome year under schedule anchor s is s+1 whatever the horizon,
    so this table does not depend on h. The diagonal is clean by construction.
    Off the diagonal, an exposure anchor ahead of the schedule anchor can reach
    into the schedule's own outcome years, and that has to be visible on the
    cell rather than buried.
    """
    rows = []
    for sched in SCHEDULE_ANCHORS:
        first_outcome = sched + 1
        for exposure in EXPOSURE_ANCHORS:
            for length in SHARE_LENGTHS:
                first, last = share_window(exposure, length)
                rows.append(
                    {
                        "sched_anchor": sched,
                        "exposure_anchor": exposure,
                        "share_len": length,
                        "share_label": f"{exposure}_{length}",
                        "first_outcome_year": first_outcome,
                        "share_first": first,
                        "share_last": last,
                        "exposure_overlap_years": max(0, last - first_outcome + 1),
                        "is_diagonal": sched == exposure,
                    }
                )
    return pd.DataFrame(rows)


def build_anchor_horizon(
    anchor: int,
    horizon: int,
    levels: pd.DataFrame,
    income: pd.DataFrame,
    state_map: pd.Series,
    national: pd.Series,
    national_income: pd.Series,
    shares: dict[str, pd.Series],
):
    """41's build_period_horizon, re-anchored and carrying nine instruments.

    The float path is deliberately identical to 41's: the same slice-and-sum
    order, the same leave-one-out subtractions, the same int32/int8 casts and
    the same terminal sort. The anchor-1998 pin compares the result to 41's own
    output and fails on a last-ulp difference.
    """
    starts = anchor_starts(anchor, horizon)
    if not starts:
        raise AssertionError(f"anchor {anchor} h={horizon} has no start years")
    if max(starts) + horizon > POST_LAST:
        raise AssertionError(
            f"anchor {anchor} h={horizon} runs past FY{POST_LAST}"
        )
    pieces = []
    for start in starts:
        future_years = list(range(start + 1, start + horizon + 1))
        missing = [year for year in future_years if year not in levels.columns]
        if missing:
            raise AssertionError(
                f"anchor {anchor} h={horizon} needs contract years {missing}"
            )
        future = levels.loc[:, future_years].sum(axis=1)
        national_future = float(national.loc[future_years].sum())
        piece = pd.DataFrame(
            {
                "unit": future.index.astype(np.int32),
                "state": future.index.map(state_map).astype(np.int32),
                "year": np.int32(start),
                "future": future.to_numpy(),
                "income_real": income.loc[future.index, start].to_numpy(),
            }
        )
        piece["loo_future"] = national_future - piece["future"]
        piece["loo_income"] = float(national_income.loc[start]) - piece["income_real"]
        for label, values in shares.items():
            piece[f"s{label}"] = piece["unit"].map(values)
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    required = [
        "future",
        "income_real",
        "loo_future",
        "loo_income",
        *(f"s{label}" for label in shares),
    ]
    if not np.isfinite(frame[required].to_numpy()).all():
        raise AssertionError(f"anchor {anchor} h={horizon} contains nonfinite inputs")
    if not (frame["income_real"] > 0).all():
        raise AssertionError(f"anchor {anchor} h={horizon} contains nonpositive income")
    if not (frame["loo_income"] > 0).all():
        raise AssertionError(
            f"anchor {anchor} h={horizon} contains nonpositive "
            "leave-one-out national income"
        )
    frame["state_year"] = (
        frame["state"].astype(np.int32) * 10_000 + frame["year"].astype(np.int32)
    )
    frame["y_cum"] = frame["future"] / frame["income_real"]
    for label in shares:
        frame[f"z{label}"] = (
            frame[f"s{label}"] * frame["loo_future"] / frame["income_real"]
        )
    frame.insert(0, "sched_anchor", np.int32(anchor))
    frame.insert(1, "horizon", np.int8(horizon))
    frame = frame.sort_values(["unit", "year"]).reset_index(drop=True)
    expected = EXPECTED_ANALYSIS_UNITS * len(starts)
    if len(frame) != expected:
        raise AssertionError(
            f"anchor {anchor} h={horizon} has {len(frame)} rows, expected {expected}"
        )
    return frame, starts


def pin_to_symmetric(module, frame: pd.DataFrame, horizon: int, **kwargs):
    """Compare a schedule-anchor-1998 frame to 41's own post-period frame.

    "post" is a live key in 41.PERIODS, so build_period_horizon runs unmodified
    and this is a genuine cross-build comparison rather than a restatement.
    """
    reference, reference_starts = module.build_period_horizon("post", horizon, **kwargs)
    if reference_starts != anchor_starts(PINNED_ANCHOR, horizon):
        raise AssertionError(
            f"41 and 46 disagree on the h={horizon} schedule: "
            f"{reference_starts} vs {anchor_starts(PINNED_ANCHOR, horizon)}"
        )
    if len(reference) != len(frame):
        raise AssertionError(f"41 and 46 disagree on h={horizon} row count")
    shared = ["unit", "state", "year", "state_year"]
    for column in shared:
        if not reference[column].to_numpy().tolist() == frame[column].to_numpy().tolist():
            raise AssertionError(f"41 and 46 disagree on {column} at h={horizon}")
    exact = ["future", "income_real", "loo_future", "loo_income", "y_cum"]
    pairs = [(column, column) for column in exact]
    pairs.extend(
        (f"fixed{length}", f"s{PINNED_ANCHOR}_{length}") for length in SHARE_LENGTHS
    )
    pairs.extend(
        (f"zfixed{length}", f"z{PINNED_ANCHOR}_{length}") for length in SHARE_LENGTHS
    )
    worst = 0.0
    for reference_column, own_column in pairs:
        deviation = float(
            np.max(
                np.abs(
                    reference[reference_column].to_numpy()
                    - frame[own_column].to_numpy()
                )
            )
        )
        worst = max(worst, deviation)
        if deviation != 0.0:
            raise AssertionError(
                f"41 and 46 disagree on {reference_column} at h={horizon}: "
                f"maximum deviation {deviation:.3e}"
            )
    return len(pairs), worst


def design_contract(share_audit: pd.DataFrame, panel_audit: pd.DataFrame):
    windows = {
        row.share_label: [int(row.share_first), int(row.share_last)]
        for row in share_audit.itertuples(index=False)
    }
    schedules = {
        f"{int(row.sched_anchor)}_h{int(row.horizon)}": row.start_years
        for row in panel_audit.itertuples(index=False)
    }
    return {
        "contract": "post_anchor_sweep_v1",
        "question": (
            "does moving the post-period anchor year change the county first "
            "stage, and if so is it the exposure window or the start-year "
            "schedule that does it"
        ),
        "source": {
            "contracts": str(dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"),
            "county_panel": str(dc.PANEL_OUT_DIR / "county_panel.parquet"),
            "contract_flow": "signed award-year obligations",
            "dollar_treatment": "no nominal floor",
            "price_basis": "real 2006 dollars",
        },
        "sample": {
            "era": "post only",
            "schedule_anchors": list(SCHEDULE_ANCHORS),
            "exposure_anchors": list(EXPOSURE_ANCHORS),
            "period_last": POST_LAST,
            "horizons": list(HORIZONS),
            "share_lengths": list(SHARE_LENGTHS),
            "window_mode": "anchor-phased nonoverlapping start years",
            "window_count": "natural; later anchors get fewer windows and n_windows is reported",
            "positive_employment_years": [
                dc.POSITIVE_EMP_FIRST_YEAR,
                dc.POSITIVE_EMP_LAST_YEAR,
            ],
            "state_rule": "retain states with at least two assigned commuting zones",
            "expected_counties": EXPECTED_ANALYSIS_UNITS,
            "expected_states": EXPECTED_STATES,
        },
        "shares": windows,
        "schedules": schedules,
        "local_lhs": "sum of contracts from t+1 through t+h divided by real personal income in t",
        "instrument": (
            "fixed allocation share times leave-one-county-out national contracts "
            "from t+1 through t+h divided by the county's own real personal income in t"
        ),
        "denominators": {
            "county_income": {
                "column_prefix": "z",
                "scale_column": "income_real",
                "note": (
                    "the only denominator estimated here, per the request. "
                    "loo_income is kept in the panel so the national_income "
                    "follow-up is a 47 edit rather than a rebuild"
                ),
            }
        },
        "grid": {
            "posted_rows": len(SCHEDULE_ANCHORS)
            * len(HORIZONS)
            * len(EXPOSURE_ANCHORS)
            * len(SHARE_LENGTHS)
            * 2,
            "structural_families": len(SCHEDULE_ANCHORS)
            * len(HORIZONS)
            * len(EXPOSURE_ANCHORS)
            * len(SHARE_LENGTHS),
            "headline_families": "the diagonal, exposure_anchor == sched_anchor",
            "pinned_rows": len(HORIZONS) * len(SHARE_LENGTHS) * 2,
        },
        "winsorization": {
            "variable": "LHS only",
            "rule": "1st and 99th empirical percentiles",
            "group": "start year within schedule",
            "command": "winsor4, method(tail) level(1)",
            # Scoped as an audit and promoted to co-equal: the unwinsorized
            # cell is where the anchor changes the F > 9 verdict, so treating
            # it as a footnote would report the wrong finding.
            "rules_reported": ["none", "tail1"],
            "unwinsorized_is_co_equal": True,
        },
        "regression": {
            "fixed_effects": "county and state-by-year",
            "cluster": "county",
            "weights": "none",
            "estimator": "Frisch-Waugh-Lovell two-step, as in 42",
        },
        # Frozen BEFORE 47 runs. Without a pre-committed rule the sweep is
        # another max-F search over a 162-cell grid, which is exactly what
        # README register item 1 objects to.
        "decision_rule": {
            "statement": (
                "Within a winsorization rule, the anchor is declared to move "
                "the first stage if any of (a), (b) or (c) holds."
            ),
            "a_f9_verdict_flips": (
                "the fs_F > 9 verdict is not the same at all three schedule "
                "anchors within a (horizon, share_len) family"
            ),
            "b_coef_sign_flips": (
                "fs_coef does not have the same sign at all three schedule anchors"
            ),
            "c_rho_outside_lowo_band": (
                "the across-anchor range of fs_rho exceeds the anchor-1998 "
                "leave-one-window-out range of fs_rho for the same "
                "(horizon, share_len, winsor_rule)"
            ),
            "why_rho_and_not_F": (
                "n_obs differs across anchors, 10/9/9 windows at h=2 and 5/4/4 "
                "at h=4, and F scales with N at a fixed partial correlation. "
                "fs_rho is the N-free statistic; fs_F answers the separate "
                "question of whether the instrument clears the weak-instrument bar"
            ),
            "why_the_lowo_band": (
                "dropping one arbitrary start year moves F by a factor of two "
                "at constant exposure and constant N, so an across-anchor "
                "spread inside that band is not evidence that the anchor matters"
            ),
            "attribution": (
                "an effect is attributed to the schedule if it appears down a "
                "column of the 3x3 with the exposure anchor held fixed, and to "
                "the exposure if it appears across a row with the schedule held"
            ),
            "balanced_horizon": (
                "h=3 is the only horizon where all three anchors have the same "
                "window count, six, so it carries the cleanest comparison"
            ),
        },
    }


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        print("Post-1998 anchor-year sweep panel")
        module = load_numbered(SYMMETRIC_BUILDER)

        # 41's own post anchor and schedule must agree with this file's before
        # anything else runs. The pin below only exercises share_last == 1997;
        # these two catch a drift in the shared definition itself.
        for length in SHARE_LENGTHS:
            if share_window(PINNED_ANCHOR, length) != module.share_windows()["post"][
                f"fixed{length}"
            ]:
                raise AssertionError(f"41 and 46 disagree on the fixed{length} window")
        for horizon in HORIZONS:
            if anchor_starts(PINNED_ANCHOR, horizon) != module.period_starts(
                "post", horizon
            ):
                raise AssertionError(f"41 and 46 disagree on the h={horizon} schedule")
        print(f"41 agreement on anchor {PINNED_ANCHOR} windows and schedules: ok")

        contracts, panel = module.load_inputs()
        universe = module.positive_employment_universe(panel)
        county_cross = module.geography(universe)
        analysis_units = set(county_cross["ccg"].astype(int))
        shares, share_audit = build_shares(contracts, universe, analysis_units)

        analysis_contracts = contracts[
            contracts["ccg"].isin(analysis_units)
        ].pivot(index="ccg", columns="year", values="dollars_real")
        analysis_income = panel[panel["ccg"].isin(analysis_units)].pivot(
            index="ccg", columns="year", values="income_real"
        )
        state_map = county_cross.set_index("ccg")["state"]
        # Reindexed exactly as 41 reindexes it, and national_income deliberately
        # not reindexed, so both float paths match to the last bit.
        national = (
            contracts[contracts["ccg"].isin(universe)]
            .groupby("year", sort=False)["dollars_real"]
            .sum()
            .reindex(range(module.SOURCE_FIRST, module.SOURCE_LAST + 1))
        )
        national_income = (
            panel[panel["ccg"].isin(universe)]
            .groupby("year", sort=False)["income_real"]
            .sum()
        )
        missing_income = [
            year
            for year in range(min(SCHEDULE_ANCHORS), POST_LAST + 1)
            if year not in national_income.index
        ]
        if missing_income:
            raise AssertionError(
                f"national income series is missing years {missing_income}"
            )

        pairing_audit = pair_exposure_with_schedule()
        diagonal = pairing_audit[pairing_audit["is_diagonal"]]
        if not diagonal["exposure_overlap_years"].eq(0).all():
            raise AssertionError(
                "a coupled-anchor exposure window touches its own outcome years"
            )
        contaminated = pairing_audit[
            ~pairing_audit["is_diagonal"]
            & pairing_audit["exposure_overlap_years"].gt(0)
        ]
        # Inherent to the off-diagonal, not a bug: pairing exposure anchor a
        # with schedule anchor s < a - 1 measures the share partly from a year
        # that is an outcome year under s. It is confined to (1998, 2000) and
        # to a single year; the assert freezes that so a wider anchor set
        # cannot silently contaminate more cells.
        if set(map(tuple, contaminated[["sched_anchor", "exposure_anchor"]].to_numpy())) != {
            (1998, 2000)
        } or not contaminated["exposure_overlap_years"].eq(1).all():
            raise AssertionError(
                "unexpected exposure-outcome overlap in the off-diagonal: "
                f"{contaminated.to_dict('records')}"
            )
        pairing_audit.to_csv(PAIRING_AUDIT_OUT, index=False)
        print(
            f"exposure-schedule pairings: {len(diagonal)} diagonal clean, "
            f"{len(contaminated)} off-diagonal overlapping by one year "
            f"-> {PAIRING_AUDIT_OUT}"
        )

        pin_reference = {
            "levels": analysis_contracts,
            "income": analysis_income,
            "state_map": state_map,
            "national": national,
            "national_income": national_income,
            "shares": module.build_shares(
                contracts,
                universe,
                analysis_units,
                {"post": module.share_windows()["post"]},
            )[0]["post"],
        }

        frames = []
        audit_rows = []
        support = None
        for anchor in SCHEDULE_ANCHORS:
            for horizon in HORIZONS:
                frame, starts = build_anchor_horizon(
                    anchor,
                    horizon,
                    analysis_contracts,
                    analysis_income,
                    state_map,
                    national,
                    national_income,
                    shares,
                )
                # Common support licenses the whole comparison: an F that moved
                # because the county set moved would say nothing about anchors.
                units = tuple(sorted(frame["unit"].unique().tolist()))
                if support is None:
                    support = units
                elif units != support:
                    raise AssertionError(
                        f"anchor {anchor} h={horizon} has a different county set"
                    )
                if frame["unit"].nunique() != EXPECTED_ANALYSIS_UNITS:
                    raise AssertionError(f"anchor {anchor} h={horizon} lost counties")
                if frame["state"].nunique() != EXPECTED_STATES:
                    raise AssertionError(f"anchor {anchor} h={horizon} lost states")

                if anchor == PINNED_ANCHOR:
                    columns, worst = pin_to_symmetric(
                        module, frame, horizon, **pin_reference
                    )
                    print(
                        f"  pin h={horizon}: {columns} columns identical to 41 "
                        f"(maximum deviation {worst:.1e})"
                    )

                outcome_years = sorted(
                    {
                        year
                        for start in starts
                        for year in range(start + 1, start + horizon + 1)
                    }
                )
                unused_tail = [
                    year
                    for year in range(max(outcome_years) + 1, POST_LAST + 1)
                ]
                frames.append(frame)
                dta_path = TEMP / f"post_s{anchor}_h{horizon}.dta"
                frame.drop(columns=["sched_anchor"]).to_stata(
                    dta_path, write_index=False, version=118
                )
                audit_rows.append(
                    {
                        "sched_anchor": anchor,
                        "period_last": POST_LAST,
                        "horizon": horizon,
                        "start_years": " ".join(map(str, starts)),
                        "n_windows": len(starts),
                        "n_obs": len(frame),
                        "n_units": frame["unit"].nunique(),
                        "n_states": frame["state"].nunique(),
                        "first_outcome_year": min(outcome_years),
                        "last_outcome_year": max(outcome_years),
                        "n_outcome_years": len(outcome_years),
                        # The tail the schedule silently drops when the period
                        # length is not a multiple of h. Anchor 1999 at h=4
                        # never sees 2016-2018, so part of any anchor-to-anchor
                        # gap is "which years are in the sample".
                        "unused_tail_years": " ".join(map(str, unused_tail)),
                        "n_unused_tail_years": len(unused_tail),
                        # All nine exposure windows are pure DCADS, but the
                        # share of outcome years drawn from USAspending is not
                        # constant across anchors.
                        "frac_outcome_years_post_seam": float(
                            sum(1 for y in outcome_years if y >= dc.SEAM_YEAR)
                            / len(outcome_years)
                        ),
                        "first_start_national_income": float(
                            national_income.loc[min(starts)]
                        ),
                        "min_loo_income": float(frame["loo_income"].min()),
                        "input_path": str(dta_path),
                    }
                )
                print(
                    f"anchor {anchor} h={horizon}: {len(starts)} windows, "
                    f"{len(frame):,} rows, outcome {min(outcome_years)}-"
                    f"{max(outcome_years)}, starts={starts}"
                )

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(TEMP / "panel.parquet", index=False)
        share_audit.to_csv(SHARE_AUDIT_OUT, index=False)
        panel_audit = pd.DataFrame(audit_rows)
        panel_audit.to_csv(PANEL_AUDIT_OUT, index=False)
        DESIGN_OUT.write_text(
            json.dumps(design_contract(share_audit, panel_audit), indent=2) + "\n",
            encoding="utf-8",
        )

        # Reported, never asserted. At about 0.99 this is the single most
        # informative number in the sweep: it says ex ante that the exposure
        # leg should be close to a null, which is why the 3x3 exists.
        print("\ncross-anchor exposure share correlation, analysis units:")
        analysis_shares = {
            label: values.loc[values.index.isin(analysis_units)]
            for label, values in shares.items()
        }
        for length in SHARE_LENGTHS:
            for left, right in (
                (1998, 1999),
                (1998, 2000),
                (1999, 2000),
            ):
                pearson = float(
                    analysis_shares[f"{left}_{length}"].corr(
                        analysis_shares[f"{right}_{length}"]
                    )
                )
                spearman = float(
                    analysis_shares[f"{left}_{length}"].corr(
                        analysis_shares[f"{right}_{length}"], method="spearman"
                    )
                )
                print(
                    f"  fixed{length:<2d} {left} vs {right}: "
                    f"pearson {pearson:.6f}  spearman {spearman:.6f}"
                )

        print(f"\n{len(combined):,} rows -> {TEMP / 'panel.parquet'}")
        print(f"9 share checks -> {SHARE_AUDIT_OUT}")
        print(f"9 panel checks -> {PANEL_AUDIT_OUT}")
        print(f"27 pairing checks -> {PAIRING_AUDIT_OUT}")
        print(f"design contract -> {DESIGN_OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
