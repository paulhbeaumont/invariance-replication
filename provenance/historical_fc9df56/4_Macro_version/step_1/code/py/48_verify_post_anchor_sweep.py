#!/usr/bin/env python3
"""Verify the post-1998 anchor-year sweep.

The verifier re-declares every grid rather than importing 46's, so a drift in
the builder fails here instead of propagating. That is 43's pattern and the
reason 43 caught what it caught.

The load-bearing check is 16, the anchor-1998 pin. 46 re-implements 41's
builder for a moving anchor, and 47 re-implements 42's estimator; both are
copies of code that produces frozen numbers. Check 16 holds the eighteen
diagonal anchor-1998 cells to the git-tracked v1 reference to 1e-10, so a
re-implementation that drifted anywhere in the shared path fails loudly. It
pins n_obs and df_absorbed as integers first, because a mismatch there
localizes the failure to the schedule rather than to the arithmetic.

WHAT IS DELIBERATELY NOT ASSERTED

43's check 16 asserts fs_F > 9. Nothing here does. This sweep exists to find
out whether the first stage survives moving the anchor, and asserting that it
clears the bar would answer the question by fiat. The F counts are printed by
check 18 and asserted nowhere. The same applies to the cross-anchor share
correlations: they are the evidence that the exposure leg should be close to a
null, and they are reported rather than held to a threshold.

Outputs:
    batch_logs/macro_version/post_anchor_sweep_verification.csv
    batch_logs/macro_version/48_verify_post_anchor_sweep.log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


np.random.seed(0)

TEMP = dc.POST_ANCHOR_SWEEP_DIR
PANEL = TEMP / "panel.parquet"
DESIGN = dc.LOG_DIR / "post_anchor_sweep_design.json"
SHARE_AUDIT = dc.LOG_DIR / "post_anchor_sweep_share_audit.csv"
PANEL_AUDIT = dc.LOG_DIR / "post_anchor_sweep_panel_audit.csv"
PAIRING_AUDIT = dc.LOG_DIR / "post_anchor_sweep_pairing_audit.csv"
RESULTS = dc.LOG_DIR / "post_anchor_sweep_first_stage.csv"
LOWO = dc.LOG_DIR / "post_anchor_sweep_lowo.csv"
VERIFY_OUT = dc.LOG_DIR / "post_anchor_sweep_verification.csv"
LOG = dc.LOG_DIR / "48_verify_post_anchor_sweep.log"
CONTRACTS = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
COUNTY_PANEL = dc.PANEL_OUT_DIR / "county_panel.parquet"
HARMONIZED_REPORT = dc.HARMONIZED_REPORT
DCADS_PARSE_REPORT = dc.OUT_DIR / "parse_verification_report.csv"
COUNTY_FIPS_REPORT = dc.OUT_DIR / "county_fips_verification_report.csv"
USASPENDING_REPORT = dc.USASPENDING_VERIFICATION_REPORT
# The v1 grid, frozen before this sweep existed. Its eighteen post rows are
# what check 16 pins the coupled anchor-1998 diagonal to.
V1_BASELINE = Path(__file__).resolve().parent.parent / "reference" / (
    "symmetric_nonoverlap_first_stage_v1.csv"
)

SCHEDULE_ANCHORS = (1998, 1999, 2000)
EXPOSURE_ANCHORS = (1998, 1999, 2000)
POST_LAST = 2018
SHARE_LENGTHS = (4, 5, 10)
HORIZONS = (2, 3, 4)
PINNED_ANCHOR = 1998
EXPECTED_UNIVERSE = 3_094
EXPECTED_UNITS = 3_080
EXPECTED_STATES = 48
EXPECTED_CELLS = (
    len(SCHEDULE_ANCHORS)
    * len(HORIZONS)
    * len(EXPOSURE_ANCHORS)
    * len(SHARE_LENGTHS)
    * 2
)
EXPECTED_DIAGONAL = len(SCHEDULE_ANCHORS) * len(HORIZONS) * len(SHARE_LENGTHS) * 2
EXPECTED_PINNED = len(HORIZONS) * len(SHARE_LENGTHS) * 2
# 10 + 6 + 5 dropped start years, three lengths, two winsorization rules.
EXPECTED_LOWO = (10 + 6 + 5) * len(SHARE_LENGTHS) * 2
SAMPLE_PER_CELL = 150


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


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, number: int, label: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        print(f"[{number:02d}] {label:<56s} {status}  {detail}")
        self.rows.append(
            {"check": number, "label": label, "status": status, "detail": detail}
        )

    @property
    def failed(self):
        return [row for row in self.rows if row["status"] == "FAIL"]


def share_window(anchor: int, length: int):
    end = anchor - 1
    return end - length + 1, end


def starts(anchor: int, horizon: int):
    return list(range(anchor, POST_LAST - horizon + 1, horizon))


def positive_universe(panel: pd.DataFrame):
    profile = (
        panel[
            panel["year"].between(
                dc.POSITIVE_EMP_FIRST_YEAR, dc.POSITIVE_EMP_LAST_YEAR
            )
        ]
        .groupby("ccg", sort=False)["emp"]
        .min()
    )
    return set(profile[profile > 0].index.astype(int))


def winsor_tail_by_year(frame: pd.DataFrame):
    frame["y_python"] = frame["y_cum"]
    for _, group in frame.groupby("year", sort=False):
        lower, upper = np.quantile(
            group["y_cum"].to_numpy(), (0.01, 0.99), method="inverted_cdf"
        )
        frame.loc[group.index, "y_python"] = group["y_cum"].clip(lower, upper)


def residualize(frame: pd.DataFrame, column: str):
    """Residuals of `column` on county and state-by-year, 47's r_* variables."""
    model = pf.feols(
        f"{column} ~ 1 | unit + state_year",
        data=frame,
        demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000),
    )
    return model.resid()


def fit(frame: pd.DataFrame, outcome: str, instrument: str):
    model = pf.feols(
        f"{outcome} ~ {instrument} | unit + state_year",
        data=frame,
        vcov={"CRV1": "unit"},
        demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000),
    )
    coefficient = float(model.coef()[instrument])
    standard_error = float(model.se()[instrument])
    return coefficient, standard_error, (coefficient / standard_error) ** 2


def source_checks(checks: Checks, contracts: pd.DataFrame):
    parse = pd.read_csv(DCADS_PARSE_REPORT)
    checks.add(
        1,
        "DCADS parse verification is clean",
        bool(parse["status"].eq("ok").all() and parse["fails"].isna().all()),
        f"{len(parse)} source files",
    )

    fips = pd.read_csv(COUNTY_FIPS_REPORT)
    fips_bad = int(fips[["domain_violations", "unresolved", "wrong_state"]].sum().sum())
    checks.add(
        2,
        "DCADS county geography verification is clean",
        fips_bad == 0,
        f"{len(fips)} fiscal files, {fips_bad} flagged rows",
    )

    usa = pd.read_csv(USASPENDING_REPORT)
    checks.add(
        3,
        "USAspending verification is clean",
        len(usa) == 12 and usa["status"].eq("PASS").all(),
        f"{int(usa['status'].eq('PASS').sum())}/12 fiscal years",
    )

    report = pd.read_csv(HARMONIZED_REPORT)
    contracts["is_nonzero"] = contracts["dollars_nominal"].ne(0).astype(np.int8)
    contracts["is_negative"] = contracts["dollars_nominal"].lt(0).astype(np.int8)
    collapsed = contracts.groupby(["year", "source"], as_index=False, sort=False).agg(
        ccgs=("ccg", "size"),
        nonzero_ccgs=("is_nonzero", "sum"),
        actions=("n_actions", "sum"),
        negative_ccg_years=("is_negative", "sum"),
        dollars_nominal=("dollars_nominal", "sum"),
        dollars_real=("dollars_real", "sum"),
    )
    collapsed = collapsed.rename(columns={"year": "fiscal_year"})
    comparison = report.merge(
        collapsed,
        on=["fiscal_year", "source"],
        suffixes=("_report", "_cache"),
        validate="one_to_one",
    )
    count_columns = ["ccgs", "nonzero_ccgs", "actions", "negative_ccg_years"]
    count_ok = all(
        comparison[f"{column}_report"].eq(comparison[f"{column}_cache"]).all()
        for column in count_columns
    )
    dollar_deviation = max(
        float(np.max(np.abs(comparison[f"{c}_report"] - comparison[f"{c}_cache"])))
        for c in ("dollars_nominal", "dollars_real")
    )
    checks.add(
        4,
        "raw-derived county cache conserves annual totals",
        len(comparison) == 53 and count_ok and dollar_deviation < 0.01,
        f"53 years, maximum dollar deviation ${dollar_deviation:.6f}",
    )

    source_by_year = contracts.groupby("year", sort=False)["source"].first()
    source_ok = source_by_year.loc[1966:2006].eq("DCADS").all()
    source_ok &= source_by_year.loc[2007:2018].eq("USAspending").all()
    checks.add(
        5,
        "contract source splice is FY2007",
        bool(source_ok),
        "DCADS FY1966-2006; USAspending FY2007-2018",
    )


def design_checks(
    checks: Checks,
    contracts: pd.DataFrame,
    universe: set[int],
    panel: pd.DataFrame,
    share_audit: pd.DataFrame,
    panel_audit: pd.DataFrame,
    pairing_audit: pd.DataFrame,
):
    checks.add(
        6,
        "positive-employment universe is fixed",
        len(universe) == EXPECTED_UNIVERSE,
        f"{len(universe):,} counties",
    )

    # Common support is what licenses comparing anchors at all: an F that moved
    # because the county set moved would say nothing about anchor years.
    supports = {
        key: tuple(sorted(group["unit"].unique().tolist()))
        for key, group in panel.groupby(["sched_anchor", "horizon"], sort=False)
    }
    support_ok = (
        len(set(supports.values())) == 1
        and panel["unit"].nunique() == EXPECTED_UNITS
        and panel["state"].nunique() == EXPECTED_STATES
    )
    checks.add(
        7,
        "analysis support is common across every anchor and horizon",
        bool(support_ok),
        f"{len(supports)} panels sharing {panel['unit'].nunique():,} counties, "
        f"{panel['state'].nunique()} states",
    )

    expected_windows = {
        (anchor, length, *share_window(anchor, length), length)
        for anchor in EXPOSURE_ANCHORS
        for length in SHARE_LENGTHS
    }
    observed = {
        (
            int(row.exposure_anchor),
            int(row.share_len),
            int(row.share_first),
            int(row.share_last),
            int(row.share_years),
        )
        for row in share_audit.itertuples(index=False)
    }
    checks.add(
        8,
        "exposure windows end the year before their own anchor",
        observed == expected_windows
        and share_audit["years_between_share_and_anchor"].eq(0).all(),
        "4, 5 and 10 years ending at anchor-1, for anchors 1998, 1999, 2000",
    )

    levels = contracts[contracts["ccg"].isin(universe)].pivot(
        index="ccg", columns="year", values="dollars_real"
    )
    deviations = []
    for row in share_audit.itertuples(index=False):
        numerator = levels.loc[:, row.share_first : row.share_last].sum(axis=1)
        denominator = float(numerator.sum())
        deviations.extend(
            [
                abs(denominator - row.national_real_contracts),
                abs(float((numerator / denominator).sum()) - row.sum_universe_shares),
            ]
        )
    scale = max(float(share_audit["national_real_contracts"].abs().max()), 1.0)
    share_deviation = max(deviations, default=np.inf) / scale
    checks.add(
        9,
        "fixed shares reproduce raw county sums",
        share_deviation < 1e-12
        and np.allclose(share_audit["sum_universe_shares"], 1.0, atol=1e-12),
        f"9 windows, scaled maximum deviation {share_deviation:.2e}",
    )

    schedule_ok = True
    overlap_ok = True
    bounds_ok = True
    for anchor in SCHEDULE_ANCHORS:
        for horizon in HORIZONS:
            expected = starts(anchor, horizon)
            row = panel_audit[
                panel_audit["sched_anchor"].eq(anchor)
                & panel_audit["horizon"].eq(horizon)
            ]
            if len(row) != 1:
                schedule_ok = False
                continue
            observed_starts = list(map(int, row.iloc[0]["start_years"].split()))
            schedule_ok &= observed_starts == expected
            blocks = [
                set(range(year + 1, year + horizon + 1)) for year in observed_starts
            ]
            overlap_ok &= len(set().union(*blocks)) == sum(map(len, blocks))
            bounds_ok &= max(observed_starts) + horizon <= POST_LAST
            bounds_ok &= min(observed_starts) == anchor
    checks.add(
        10,
        "nonoverlapping schedules are anchor-phased and in range",
        schedule_ok and overlap_ok and bounds_ok,
        "9 schedules exact, future blocks disjoint, none running past "
        f"FY{POST_LAST}",
    )

    # The coupled diagonal must never measure exposure from one of its own
    # outcome years. Off the diagonal a one-year reach-in is inherent to the
    # decomposition, and is frozen to the single pairing where it occurs.
    diagonal = pairing_audit[pairing_audit["is_diagonal"]]
    off_diagonal = pairing_audit[~pairing_audit["is_diagonal"]]
    contaminated = off_diagonal[off_diagonal["exposure_overlap_years"].gt(0)]
    pairing_ok = (
        len(pairing_audit) == len(SCHEDULE_ANCHORS) * len(EXPOSURE_ANCHORS) * len(SHARE_LENGTHS)
        and diagonal["exposure_overlap_years"].eq(0).all()
        and set(
            map(tuple, contaminated[["sched_anchor", "exposure_anchor"]].to_numpy())
        )
        == {(1998, 2000)}
        and contaminated["exposure_overlap_years"].eq(1).all()
    )
    checks.add(
        11,
        "coupled diagonal is clear of its own outcome years",
        bool(pairing_ok),
        f"{len(diagonal)} diagonal pairings clean, {len(contaminated)} "
        "off-diagonal reaching in by one year, all (sched 1998, exposure 2000)",
    )

    expected_rows = sum(
        EXPECTED_UNITS * len(starts(anchor, horizon))
        for anchor in SCHEDULE_ANCHORS
        for horizon in HORIZONS
    )
    keys_ok = not panel.duplicated(["sched_anchor", "horizon", "unit", "year"]).any()
    checks.add(
        12,
        "analysis panel has exact keys and row count",
        len(panel) == expected_rows and keys_ok,
        f"{len(panel):,}/{expected_rows:,} rows",
    )


def cell_checks(
    checks: Checks,
    contracts: pd.DataFrame,
    county_panel: pd.DataFrame,
    universe: set[int],
    panel: pd.DataFrame,
):
    raw = contracts[contracts["ccg"].isin(universe)].set_index(["ccg", "year"])
    national = (
        contracts[contracts["ccg"].isin(universe)]
        .groupby("year", sort=False)["dollars_real"]
        .sum()
    )
    income = county_panel.set_index(["ccg", "year"])["income_real"]
    national_income = (
        county_panel[county_panel["ccg"].isin(universe)]
        .groupby("year", sort=False)["income_real"]
        .sum()
    )
    sample = panel.groupby(["sched_anchor", "horizon"], group_keys=False).sample(
        n=SAMPLE_PER_CELL, random_state=20260817
    )
    deviations = []
    for row in sample.itertuples(index=False):
        future_years = list(range(row.year + 1, row.year + row.horizon + 1))
        future = float(raw.loc[(row.unit, future_years), "dollars_real"].sum())
        national_future = float(national.loc[future_years].sum())
        loo_future = national_future - future
        income_value = float(income.loc[(row.unit, row.year)])
        loo_income = float(national_income.loc[row.year]) - income_value
        for anchor in EXPOSURE_ANCHORS:
            for length in SHARE_LENGTHS:
                first, last = share_window(anchor, length)
                local_share = float(
                    raw.loc[
                        (row.unit, list(range(first, last + 1))), "dollars_real"
                    ].sum()
                )
                national_share = float(national.loc[first:last].sum())
                share = local_share / national_share
                deviations.extend(
                    [
                        abs(getattr(row, f"s{anchor}_{length}") - share),
                        abs(
                            getattr(row, f"z{anchor}_{length}")
                            - share * loo_future / income_value
                        ),
                    ]
                )
        deviations.extend(
            [
                abs(row.future - future),
                abs(row.loo_future - loo_future),
                abs(row.income_real - income_value),
                abs(row.loo_income - loo_income),
                abs(row.y_cum - future / income_value),
            ]
        )
    scale = max(float(panel["future"].abs().max()), 1.0)
    cell_deviation = max(deviations, default=np.inf) / scale
    expected_sample = SAMPLE_PER_CELL * len(SCHEDULE_ANCHORS) * len(HORIZONS)
    checks.add(
        13,
        "sampled cells reproduce raw contracts and income",
        len(sample) == expected_sample and cell_deviation < 1e-12,
        f"{len(sample):,} cells x 9 instruments, scaled maximum deviation "
        f"{cell_deviation:.2e}",
    )

    identity = panel[["year", "income_real", "loo_income"]].copy()
    identity["total"] = identity["income_real"] + identity["loo_income"]
    identity["expected"] = identity["year"].map(national_income)
    identity_deviation = float(
        (identity["total"] - identity["expected"]).abs().max()
        / max(float(identity["expected"].max()), 1.0)
    )
    checks.add(
        14,
        "leave-one-out income identity holds and stays positive",
        identity_deviation < 1e-12 and bool((panel["loo_income"] > 0).all()),
        f"scaled maximum deviation {identity_deviation:.2e}, minimum "
        f"leave-one-out income {panel['loo_income'].min():,.0f}",
    )


def regression_checks(
    checks: Checks,
    panel: pd.DataFrame,
    results: pd.DataFrame,
    lowo: pd.DataFrame,
    share_audit: pd.DataFrame,
):
    structure_ok = (
        len(results) == EXPECTED_CELLS
        and results["status"].eq("ok").all()
        and np.isfinite(
            results[["fs_coef", "fs_se", "fs_F", "fs_rho"]].to_numpy()
        ).all()
        and int(results["is_diagonal"].sum()) == EXPECTED_DIAGONAL
        and results["n_obs"].eq(EXPECTED_UNITS * results["n_windows"]).all()
        and not results.duplicated(
            ["sched_anchor", "horizon", "exposure_anchor", "share_len", "winsor_rule"]
        ).any()
    )
    checks.add(
        15,
        "Stata regression grid is complete",
        bool(structure_ok),
        f"{len(results)}/{EXPECTED_CELLS} regressions, "
        f"{int(results['is_diagonal'].sum())} on the coupled diagonal",
    )

    # ---- check 16, the pin -------------------------------------------------
    baseline = pd.read_csv(V1_BASELINE)
    baseline = baseline[baseline["period"].eq("post")]
    pinned = results[
        results["sched_anchor"].eq(PINNED_ANCHOR)
        & results["exposure_anchor"].eq(PINNED_ANCHOR)
    ].copy()
    pinned["share_label"] = "fixed" + pinned["share_len"].astype(str)
    frozen = pinned.merge(
        baseline,
        on=["horizon", "share_label", "winsor_rule"],
        suffixes=("_sweep", "_v1"),
        validate="one_to_one",
    )
    # Integers first: a mismatch here means the schedule moved, which is a
    # different and much easier bug to find than a drift in the arithmetic.
    integer_ok = (
        frozen["n_obs_sweep"].eq(frozen["n_obs_v1"]).all()
        and frozen["df_absorbed_sweep"].eq(frozen["df_absorbed_v1"]).all()
        and frozen["n_units_sweep"].eq(frozen["n_units_v1"]).all()
        and frozen["n_states_sweep"].eq(frozen["n_states_v1"]).all()
    )
    frozen_deviation = max(
        float(np.max(np.abs(frozen[f"{column}_sweep"] - frozen[f"{column}_v1"])))
        for column in ("fs_coef", "fs_se", "fs_F")
    )
    checks.add(
        16,
        "coupled anchor 1998 reproduces the frozen v1 post grid",
        len(frozen) == EXPECTED_PINNED
        and bool(integer_ok)
        and frozen_deviation < 1e-10,
        f"{len(frozen)} cells, integer keys {'equal' if integer_ok else 'DIFFER'}, "
        f"maximum deviation {frozen_deviation:.2e}",
    )
    # ------------------------------------------------------------------------

    python_rows = []
    for anchor in SCHEDULE_ANCHORS:
        for horizon in HORIZONS:
            frame = panel[
                panel["sched_anchor"].eq(anchor) & panel["horizon"].eq(horizon)
            ].copy()
            winsor_tail_by_year(frame)
            # Residualize once per variable, then read every partial
            # correlation off the cached arrays. 47 computes fs_rho the same
            # way, from reghdfe residuals rather than from b and the standard
            # deviations, so this is a genuine second opinion on it.
            residual = {
                column: np.asarray(residualize(frame, column))
                for column in [
                    "y_cum",
                    "y_python",
                    *(
                        f"z{exposure}_{length}"
                        for exposure in EXPOSURE_ANCHORS
                        for length in SHARE_LENGTHS
                    ),
                ]
            }
            for rule, outcome in (("none", "y_cum"), ("tail1", "y_python")):
                for exposure in EXPOSURE_ANCHORS:
                    for length in SHARE_LENGTHS:
                        instrument = f"z{exposure}_{length}"
                        coefficient, standard_error, statistic = fit(
                            frame, outcome, instrument
                        )
                        python_rows.append(
                            {
                                "sched_anchor": anchor,
                                "horizon": horizon,
                                "exposure_anchor": exposure,
                                "share_len": length,
                                "winsor_rule": rule,
                                "py_coef": coefficient,
                                "py_se": standard_error,
                                "py_F": statistic,
                                "py_rho": float(
                                    np.corrcoef(
                                        residual[outcome], residual[instrument]
                                    )[0, 1]
                                ),
                            }
                        )
    python_results = pd.DataFrame(python_rows)
    compared = results.merge(
        python_results,
        on=[
            "sched_anchor",
            "horizon",
            "exposure_anchor",
            "share_len",
            "winsor_rule",
        ],
        validate="one_to_one",
    )
    estimate_deviation = max(
        float(np.max(np.abs(compared[stata] - compared[python])))
        for stata, python in (
            ("fs_coef", "py_coef"),
            ("fs_se", "py_se"),
            ("fs_F", "py_F"),
        )
    )
    rho_deviation = float(np.max(np.abs(compared["fs_rho"] - compared["py_rho"])))
    checks.add(
        17,
        "Stata estimates match independent pyfixest estimates",
        len(compared) == EXPECTED_CELLS
        and estimate_deviation < 1e-6
        and rho_deviation < 1e-6,
        f"{EXPECTED_CELLS} cells, estimate {estimate_deviation:.2e}, "
        f"rho {rho_deviation:.2e}",
    )

    expected_lowo_keys = {
        (horizon, dropped, length, rule)
        for horizon in HORIZONS
        for dropped in starts(PINNED_ANCHOR, horizon)
        for length in SHARE_LENGTHS
        for rule in ("none", "tail1")
    }
    observed_lowo_keys = {
        (int(r.horizon), int(r.dropped_start), int(r.share_len), r.winsor_rule)
        for r in lowo.itertuples(index=False)
    }
    lowo_ok = (
        len(lowo) == EXPECTED_LOWO
        and observed_lowo_keys == expected_lowo_keys
        and lowo["status"].eq("ok").all()
        and lowo["sched_anchor"].eq(PINNED_ANCHOR).all()
        and lowo["exposure_anchor"].eq(PINNED_ANCHOR).all()
        and lowo["n_obs"].eq(EXPECTED_UNITS * lowo["n_windows"]).all()
    )
    for horizon in HORIZONS:
        block = lowo[lowo["horizon"].eq(horizon)]
        lowo_ok &= block["n_windows"].eq(len(starts(PINNED_ANCHOR, horizon)) - 1).all()
    checks.add(
        18,
        "leave-one-window-out band is complete and one window short",
        bool(lowo_ok),
        f"{len(lowo)}/{EXPECTED_LOWO} refits over 21 dropped start years",
    )

    # A deterministic subsample rather than all 126: the LOWO refits run the
    # same prepare_cell path as the main grid, which check 17 already covers in
    # full, so this is here to catch a bug specific to the dropping itself.
    lowo_deviations = []
    frame_h4 = panel[
        panel["sched_anchor"].eq(PINNED_ANCHOR) & panel["horizon"].eq(4)
    ].copy()
    for dropped in starts(PINNED_ANCHOR, 4):
        subset = frame_h4[frame_h4["year"].ne(dropped)].copy()
        winsor_tail_by_year(subset)
        for rule, outcome in (("none", "y_cum"), ("tail1", "y_python")):
            for length in SHARE_LENGTHS:
                coefficient, standard_error, statistic = fit(
                    subset, outcome, f"z{PINNED_ANCHOR}_{length}"
                )
                row = lowo[
                    lowo["horizon"].eq(4)
                    & lowo["dropped_start"].eq(dropped)
                    & lowo["share_len"].eq(length)
                    & lowo["winsor_rule"].eq(rule)
                ]
                lowo_deviations.extend(
                    [
                        abs(float(row["fs_coef"].iloc[0]) - coefficient),
                        abs(float(row["fs_se"].iloc[0]) - standard_error),
                        abs(float(row["fs_F"].iloc[0]) - statistic),
                    ]
                )
    lowo_deviation = max(lowo_deviations, default=np.inf)
    checks.add(
        19,
        "leave-one-window-out refits match pyfixest on the h=4 block",
        len(lowo_deviations) == 5 * 2 * len(SHARE_LENGTHS) * 3
        and lowo_deviation < 1e-6,
        f"30 refits, maximum deviation {lowo_deviation:.2e}",
    )

    # ---- reported, never asserted ------------------------------------------
    print("\n     REPORTED, NOT ASSERTED")
    print("     The sweep exists to find out whether F survives the anchor, so")
    print("     nothing below is held to a threshold.")
    diagonal = results[results["is_diagonal"].eq(1)]
    print("\n     coupled-diagonal F by anchor and winsorization rule:")
    for rule in ("none", "tail1"):
        for anchor in SCHEDULE_ANCHORS:
            block = diagonal[
                diagonal["winsor_rule"].eq(rule) & diagonal["sched_anchor"].eq(anchor)
            ]
            print(
                f"       {rule:<5s} anchor {anchor}: "
                f"F {block['fs_F'].min():7.3f}-{block['fs_F'].max():7.3f}, "
                f"rho {block['fs_rho'].min():.4f}-{block['fs_rho'].max():.4f}, "
                f"{int((block['fs_F'] > 9).sum())}/{len(block)} cells above 9"
            )
    print("\n     leave-one-window-out F range at anchor 1998, exposure 1998:")
    for horizon in HORIZONS:
        for rule in ("none", "tail1"):
            block = lowo[lowo["horizon"].eq(horizon) & lowo["winsor_rule"].eq(rule)]
            print(
                f"       h={horizon} {rule:<5s}: F {block['fs_F'].min():7.3f}-"
                f"{block['fs_F'].max():7.3f}, rho {block['fs_rho'].min():.4f}-"
                f"{block['fs_rho'].max():.4f}"
            )
    print("\n     exposure share support on the analysis units:")
    for row in share_audit.itertuples(index=False):
        print(
            f"       {row.share_label:<9s} FY{row.share_first}-{row.share_last}: "
            f"{row.zero_share_count_analysis:4d} zero, "
            f"{row.negative_share_count_analysis:3d} negative, "
            f"{row.share_years_after_1997} year(s) after 1997"
        )


def main():
    required = [
        PANEL,
        DESIGN,
        SHARE_AUDIT,
        PANEL_AUDIT,
        PAIRING_AUDIT,
        RESULTS,
        LOWO,
        CONTRACTS,
        COUNTY_PANEL,
        HARMONIZED_REPORT,
        DCADS_PARSE_REPORT,
        COUNTY_FIPS_REPORT,
        USASPENDING_REPORT,
        V1_BASELINE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "required verification inputs missing: " + ", ".join(missing)
        )

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    checks = Checks()
    try:
        design = json.loads(DESIGN.read_text(encoding="utf-8"))
        contracts = pd.read_parquet(CONTRACTS)
        county_panel = pd.read_parquet(
            COUNTY_PANEL, columns=["ccg", "year", "emp", "income_real"]
        )
        panel = pd.read_parquet(PANEL)
        share_audit = pd.read_csv(SHARE_AUDIT)
        panel_audit = pd.read_csv(PANEL_AUDIT)
        pairing_audit = pd.read_csv(PAIRING_AUDIT)
        results = pd.read_csv(RESULTS)
        lowo = pd.read_csv(LOWO)

        rule = design.get("decision_rule", {})
        design_ok = (
            design.get("contract") == "post_anchor_sweep_v1"
            and design["source"]["contract_flow"] == "signed award-year obligations"
            and design["source"]["dollar_treatment"] == "no nominal floor"
            and design["sample"]["schedule_anchors"] == list(SCHEDULE_ANCHORS)
            and design["sample"]["exposure_anchors"] == list(EXPOSURE_ANCHORS)
            and design["sample"]["horizons"] == list(HORIZONS)
            and design["sample"]["share_lengths"] == list(SHARE_LENGTHS)
            and design["sample"]["period_last"] == POST_LAST
            and design["winsorization"]["variable"] == "LHS only"
            and design["winsorization"]["rules_reported"] == ["none", "tail1"]
            and design["regression"]["fixed_effects"] == "county and state-by-year"
            and set(design["denominators"]) == {"county_income"}
            and design["grid"]["posted_rows"] == EXPECTED_CELLS
            # The decision rule has to be present and complete BEFORE 47 runs,
            # or the sweep is a max-F search over 162 cells.
            and {
                "a_f9_verdict_flips",
                "b_coef_sign_flips",
                "c_rho_outside_lowo_band",
            }
            <= set(rule)
        )
        checks.add(
            0,
            "design contract and decision rule are pinned",
            bool(design_ok),
            design.get("contract", "missing"),
        )

        source_checks(checks, contracts)
        universe = positive_universe(county_panel)
        design_checks(
            checks,
            contracts,
            universe,
            panel,
            share_audit,
            panel_audit,
            pairing_audit,
        )
        cell_checks(checks, contracts, county_panel, universe, panel)
        regression_checks(checks, panel, results, lowo, share_audit)

        pd.DataFrame(checks.rows).to_csv(VERIFY_OUT, index=False)
        if checks.failed:
            print(f"\n{len(checks.failed)} verification checks failed")
            return 1
        print(f"\nall {len(checks.rows)} verification checks passed -> {VERIFY_OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
