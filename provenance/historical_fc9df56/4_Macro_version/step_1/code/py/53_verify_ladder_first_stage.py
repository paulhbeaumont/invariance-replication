#!/usr/bin/env python3
"""Verify the aggregation-ladder first stages.

Every constant is re-declared here rather than imported from 51, so that a
change in the builder fails in this file instead of being carried into it.

TWO KINDS OF CHECK, AND THE ORDER MATTERS

Checks 1-8 are the preflight: they assert that the design actually running is the
design that was asked for. They exist because the specification changed on
several axes at once - a new instrument, a new panel, forward dating, no floor -
and a script that quietly kept one of the old settings would still produce a
plausible table. Each of the eight names the setting and where it comes from.

Check 9 is the load-bearing one. The county rung of this ladder IS the candidate
first stage, so it must reproduce the certified 252-row result exactly. If it
does not, the collapse code is wrong somewhere upstream of the aggregation and
nothing at the commuting zone or the state can be read. It is asserted before any
aggregate identity, because a correct aggregation of a wrong county panel would
pass every check after it.

Checks 10-13 are the aggregation: the identities the plan's 3.5 asks for, the
frozen county support, and an independent re-estimation of all 72 cells.

Checks 14-15 are the absorption arms, added 2026-08-18 with the `uy` arm. 14 is
the plumbing check: at the state rung both arms reduce to unit + year, so they
must agree exactly, and a wrong column in 52's loop separates them. 15 is the
no-silent-revision check: the 36 cells that existed before the arm was added must
reproduce at 0.0 against a git-tracked copy of the frozen grid, so an addition
cannot become a revision.

Together with check 13 - which re-demeans both arms in pyfixest, independently of
52 - these are what make the new arm evidence rather than assertion. None of the
three can be satisfied by 52 and this file agreeing with each other.

The step_2 comparison is REPORTED, NOT ASSERTED, and printed at the end. It binds
the `uy` arm to an F produced by a pipeline sharing no estimator code with 52,
which is the strongest evidence available for it - but 52 rescales its standard
error by sqrt((N-1)/(N-df_a-2)) and step_2 may treat the degrees of freedom
differently, so a gap of that shape needs a decision rather than a tolerance and
must not fail a run.

WHAT IS DELIBERATELY NOT ASSERTED

Whether any F clears 9. Selecting a specification on first-stage strength is the
open register item, and this file reports the count without letting it gate
anything.

Output:
    batch_logs/macro_version/ladder_first_stage_verification.csv
    batch_logs/macro_version/53_verify_ladder_first_stage.log
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


CODE = Path(__file__).resolve().parent
PANEL = dc.MACRO_DATA_ROOT / "temp" / "ladder_first_stage" / "panel.parquet"
RESULTS = dc.LOG_DIR / "ladder_first_stage.csv"
COLLAPSE = dc.LOG_DIR / "ladder_first_stage_collapse.csv"
DESIGN = dc.LOG_DIR / "ladder_first_stage_design.json"
CONTRACTS = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
COUNTY_PANEL = dc.PANEL_OUT_DIR / "county_panel.parquet"
CANDIDATE_RESULTS = (
    dc.PROJECT_ROOT / "Candidate for first stage" / "output"
    / "candidate_first_stage_all.csv"
)
# The 36-cell grid as it stood before the uy arm was added, git-tracked next to
# the code so check 15 pins against a file this pipeline cannot overwrite. Same
# reasoning as 50's V1_BASELINE: the live CSV is rewritten on every run.
FROZEN_SXY = CODE.parent / "reference" / "ladder_first_stage_sxy_v1.csv"
# step_2's own county first stage on the uy arm, produced by a pipeline that
# shares no estimator code with 52. Reported against it at the end of the run.
STEP2_COEFF = (
    CODE.parents[2] / "step_2" / "result"
    / "coeff_candidate_iv_mpgr_pre_h4_fixed5_naics4_alloc.csv"
)
REPORT = dc.LOG_DIR / "ladder_first_stage_verification.csv"
LOG = dc.LOG_DIR / "53_verify_ladder_first_stage.log"

# Re-declared, not imported.
PERIODS = {"pre": (1979, 1997), "post": (1998, 2018)}
HORIZONS = (2, 3, 4)
HEADLINE_HORIZON = 4
SHARE = "fixed5"
SHARE_LENGTH = 5
WINSOR_LEVEL = 5.0
RUNGS = ("county", "cz", "state")
ARMS = ("none", "iq5")
# The two absorption arms 52 reports. `sxy` is the arm the candidate package
# uses and is therefore the only one check 9's pin can be fitted on; `uy` is the
# arm step_2 estimates on. Both are admissible under A.1.4.
FE_ARMS = ("sxy", "uy")
PIN_FE_ARM = "sxy"
STEP2_FE_ARM = "uy"
EXPECTED_UNITS = {"county": 3_080, "cz": 737, "state": 48}
EXPECTED_STATES = 48
EXPECTED_CELLS = (
    len(RUNGS) * len(PERIODS) * len(HORIZONS) * len(ARMS) * len(FE_ARMS)
)
EXPECTED_COLLAPSE = len(RUNGS) * len(PERIODS) * len(HORIZONS)
F_THRESHOLD = 9
SEED = 20260818


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

    def add(self, number: int, label: str, passed: bool | None, detail: str):
        # passed=None is SKIP, for a check whose target is legitimately absent.
        # It is not PASS: a check that did not run must not read as one that did.
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"[{number:02d}] {label:<58s} {status}  {detail}")
        self.rows.append(
            {"check": number, "label": label, "status": status, "detail": detail}
        )

    @property
    def failed(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    @property
    def skipped(self):
        return [row for row in self.rows if row["status"] == "SKIP"]


def starts(period: str, horizon: int):
    first, last = PERIODS[period]
    return list(range(first, last - horizon + 1, horizon))


def stata_pctile(xs, n, p):
    num = n * p
    if num % 100 == 0:
        j = num // 100
        return 0.5 * (xs[j - 1] + xs[j]) if j < n else xs[n - 1]
    return xs[-(-num // 100) - 1]


def winsor4_iq(v, level):
    xs = np.sort(v)
    n = len(xs)
    p25 = stata_pctile(xs, n, 25)
    p50 = stata_pctile(xs, n, 50)
    p75 = stata_pctile(xs, n, 75)
    iqr = p75 - p25
    return np.clip(v, p50 - level * iqr, p50 + level * iqr)


def fit(frame: pd.DataFrame, outcome: str, instrument: str, unit: str, cell: str):
    """One just-identified first stage, with the instrument standardized.

    The standardization is not cosmetic. The capped instrument is a share times
    a national ratio and lives around 1e-5 at the county, and the income-weighted
    aggregates are smaller still; pyfixest screens for collinearity on an
    absolute tolerance and rejects a lone regressor that small as degenerate.
    Stata's `regress` on the residuals has no such screen, which is why 52
    estimates these cells without complaint.

    Rescaling is exact rather than a workaround. With z' = k z the same fit gives
    b = k b' and se = k se', and F = (b/se)^2 is unchanged, so the comparison
    against 52 is on the original scale and no tolerance was relaxed to get it.
    """
    scaled = frame.copy()
    deviation = float(np.std(frame[instrument].to_numpy(np.float64)))
    if deviation <= 0.0:
        raise AssertionError(f"{instrument} has no variation")
    scale = 1.0 / deviation
    scaled[instrument] = frame[instrument].to_numpy(np.float64) * scale
    model = pf.feols(
        f"{outcome} ~ {instrument} | {unit} + {cell}",
        data=scaled,
        vcov={"CRV1": unit},
        demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000),
    )
    coefficient = float(model.coef()[instrument]) * scale
    standard_error = float(model.se()[instrument]) * scale
    return coefficient, standard_error, (coefficient / standard_error) ** 2


def relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))


# ------------------------------------------------------------------------------
# Checks 1-8, the preflight
# ------------------------------------------------------------------------------

def preflight(checks: Checks, design: dict, panel: pd.DataFrame):
    """Assert the design running is the design that was asked for."""
    contracts = pd.read_parquet(
        CONTRACTS, columns=["ccg", "year", "dollars_real"]
    )
    county = pd.read_parquet(
        COUNTY_PANEL, columns=["ccg", "year", "emp", "income_real"]
    )
    units = sorted(panel["unit"].unique().tolist())

    # 1. The instrument. Rebuilt from raw for one sampled cell and compared
    #    against the panel, with the two properties that define the candidate
    #    correction asserted directly rather than inferred from the value.
    rng = np.random.default_rng(SEED)
    period, horizon = "pre", HEADLINE_HORIZON
    block = panel[
        panel["period"].eq(period) & panel["horizon"].eq(horizon)
    ].reset_index(drop=True)
    first, last = PERIODS[period]
    window = (first - SHARE_LENGTH, first - 1)
    levels = contracts[contracts["ccg"].isin(units)].pivot(
        index="ccg", columns="year", values="dollars_real"
    )
    numerator = levels.loc[:, window[0]: window[1]].sum(axis=1)
    share = numerator / float(numerator.sum())
    income = county[county["ccg"].isin(units)].pivot(
        index="ccg", columns="year", values="income_real"
    )
    sample = rng.choice(len(block), size=min(400, len(block)), replace=False)
    worst_z = worst_shock = 0.0
    for i in sample:
        row = block.iloc[i]
        t, ccg = int(row["year"]), int(row["unit"])
        future_years = list(range(t + 1, t + horizon + 1))
        own_future = float(levels.loc[ccg, future_years].sum())
        national_future = float(levels.loc[:, future_years].to_numpy().sum())
        national_income = float(income.loc[:, t].sum())
        rebuilt_shock = own_future / float(income.loc[ccg, t])
        rebuilt_z = float(share.loc[ccg]) * national_future / national_income
        worst_shock = max(
            worst_shock, abs(rebuilt_shock - float(row["shock_raw"])) / max(abs(rebuilt_shock), 1.0)
        )
        worst_z = max(
            worst_z, abs(rebuilt_z - float(row[f"z_raw_{SHARE}"])) / max(abs(rebuilt_z), 1.0)
        )
    scaling = design["scaling"]
    checks.add(
        1,
        "instrument is the candidate build, no LOO, national denominator",
        worst_shock < 1e-12
        and worst_z < 1e-12
        and scaling["leave_one_out"] is False
        and scaling["instrument_divides_by_own_income"] is False,
        f"{len(sample)} sampled rows, shock {worst_shock:.2e}, z {worst_z:.2e}",
    )

    # 2. The dataset.
    span = (int(contracts["year"].min()), int(contracts["year"].max()))
    checks.add(
        2,
        "dataset is the harmonized FY1966-2018 panel on 3,080 counties",
        span == (1966, 2018)
        and len(units) == EXPECTED_UNITS["county"]
        and panel["state"].nunique() == EXPECTED_STATES
        and dc.SEAM_YEAR == 2007,
        f"contracts {span[0]}-{span[1]}, {len(units):,} counties, "
        f"{panel['state'].nunique()} states, seam {dc.SEAM_YEAR}",
    )

    # 3. Horizons, and which one is the headline.
    present = tuple(sorted(panel["horizon"].unique().tolist()))
    checks.add(
        3,
        "horizons 2, 3 and 4 present with 4 declared the headline",
        present == HORIZONS
        and design["sample"]["headline_horizon"] == HEADLINE_HORIZON,
        f"horizons {present}, headline h={design['sample']['headline_horizon']}",
    )

    # 4. The exposure window.
    windows = design["shares"]["windows"]
    expected = {
        p: [PERIODS[p][0] - SHARE_LENGTH, PERIODS[p][0] - 1] for p in PERIODS
    }
    got = {p: windows[p][SHARE] for p in PERIODS}
    share_sum = float(share.sum())
    checks.add(
        4,
        "fixed5 windows end the year before each era and shares sum to 1",
        got == expected
        and design["shares"]["headline"] == SHARE
        and abs(share_sum - 1.0) < 1e-12,
        f"pre {got['pre']}, post {got['post']}, sum {share_sum:.15f}",
    )

    # 5. Non-overlap. Spacing is necessary but not sufficient; assert the
    #    contract windows themselves are pairwise disjoint.
    overlaps = []
    for period in PERIODS:
        for horizon in HORIZONS:
            schedule = starts(period, horizon)
            observed = sorted(
                panel.loc[
                    panel["period"].eq(period) & panel["horizon"].eq(horizon), "year"
                ].unique().tolist()
            )
            if observed != schedule:
                overlaps.append(f"{period} h={horizon} schedule")
            covered = []
            for t in schedule:
                covered.extend(range(t + 1, t + horizon + 1))
            if len(covered) != len(set(covered)):
                overlaps.append(f"{period} h={horizon} contract years")
    checks.add(
        5,
        "start years are nonoverlapping and contract windows are disjoint",
        not overlaps,
        "6 blocks clean" if not overlaps else ", ".join(overlaps),
    )

    # 6. Winsorization: the rule, the grouping, both variables, and that the
    #    stored capped column is what the rule produces.
    worst_cap = 0.0
    n_capped = 0
    for period in PERIODS:
        for horizon in HORIZONS:
            block = panel[
                panel["period"].eq(period) & panel["horizon"].eq(horizon)
            ]
            for raw, capped in (
                ("shock_raw", "shock_iq5"),
                (f"z_raw_{SHARE}", f"z_iq5_{SHARE}"),
            ):
                for _, group in block.groupby("year", sort=False):
                    expected_values = winsor4_iq(
                        group[raw].to_numpy(np.float64), WINSOR_LEVEL
                    )
                    got_values = group[capped].to_numpy(np.float64)
                    worst_cap = max(
                        worst_cap, float(np.max(np.abs(expected_values - got_values)))
                    )
                    n_capped += int((expected_values != group[raw].to_numpy()).sum())
    winsor = design["winsorization"]
    checks.add(
        6,
        "IQR5 within start year, on shock and instrument, stored as built",
        worst_cap == 0.0
        and winsor["level"] == WINSOR_LEVEL
        and winsor["group"] == "start year"
        and winsor["computed_at"].startswith("county rung only")
        and list(winsor["rules_reported"]) == list(ARMS),
        f"max deviation {worst_cap:g}, {n_capped:,} values capped across 6 blocks",
    )

    # 7. No floor of any kind.
    checks.add(
        7,
        "no nominal dollar floor and no population or income floor",
        design["source"]["dollar_treatment"] == "no nominal floor"
        and design["sample"]["sample_floor"].startswith("none"),
        f"{design['source']['dollar_treatment']}; {design['sample']['sample_floor']}",
    )

    # 8. Both eras estimated.
    era_cells = {
        (p, h)
        for p in PERIODS
        for h in HORIZONS
        if not panel[panel["period"].eq(p) & panel["horizon"].eq(h)].empty
    }
    checks.add(
        8,
        "pre-1998 and post-1998 both built, at all three horizons",
        len(era_cells) == len(PERIODS) * len(HORIZONS)
        and design["sample"]["periods"]["pre"]["first"] == 1979
        and design["sample"]["periods"]["post"]["first"] == 1998,
        f"{len(era_cells)} era-horizon blocks, pre {PERIODS['pre']}, "
        f"post {PERIODS['post']}",
    )


# ------------------------------------------------------------------------------
# Check 9, the pin
# ------------------------------------------------------------------------------

def pin_county_rung(checks: Checks, results: pd.DataFrame):
    """The county rung IS the candidate first stage. It must reproduce it.

    Merged on the specification, not on row order, and the integer columns are
    compared as integers before the estimates are compared as floats, so a
    sample difference cannot hide inside a coefficient tolerance.

    Fitted on the `sxy` arm ONLY. The candidate package absorbs
    `unit state_year` (02_run_candidate_first_stage.do), so that is the arm this
    pin is defined against; the `uy` arm is a different absorption and reproducing
    the candidate under it would mean the arms were not distinct. Check 15 is the
    external pin for `uy`.
    """
    if not CANDIDATE_RESULTS.exists():
        checks.add(
            9, "county rung reproduces the certified candidate result", False,
            f"pin target missing: {CANDIDATE_RESULTS}",
        )
        return
    candidate = pd.read_csv(CANDIDATE_RESULTS)
    candidate = candidate[
        candidate["scaling"].eq("income")
        & candidate["share_label"].eq(SHARE)
        & (
            candidate["winsor_rule"].eq("none")
            | (candidate["winsor_rule"].eq("iq5") & candidate["winsor_group"].eq("year"))
        )
    ]
    mine = results[
        results["rung"].eq("county") & results["fe_arm"].eq(PIN_FE_ARM)
    ]
    merged = mine.merge(
        candidate,
        on=["period", "horizon", "winsor_rule"],
        suffixes=("", "_ref"),
        validate="one_to_one",
    )
    integers_ok = (
        merged["n_obs"].astype(int).equals(merged["n_obs_ref"].astype(int))
        and merged["n_units"].astype(int).equals(merged["n_units_ref"].astype(int))
        and merged["df_absorbed"].astype(int).equals(
            merged["df_absorbed_ref"].astype(int)
        )
    )
    deviation = max(
        relative(merged[column], merged[f"{column}_ref"])
        for column in ("fs_coef", "fs_se", "fs_F")
    )
    checks.add(
        9,
        "county rung reproduces the certified candidate result",
        len(merged) == len(PERIODS) * len(HORIZONS) * len(ARMS)
        and integers_ok
        and deviation < 1e-10,
        f"{len(merged)} cells, counts exact, estimates {deviation:.2e} relative",
    )


# ------------------------------------------------------------------------------
# Checks 10-13, the aggregation
# ------------------------------------------------------------------------------

def collapse(block: pd.DataFrame, key: str):
    """The income-weighted mean 52 computes, rebuilt independently in pandas."""
    out = block.groupby([key, "year"], sort=True).apply(
        lambda g: pd.Series(
            {
                "income_real": g["income_real"].sum(),
                "employment": g["employment"].sum(),
                "future": g["future"].sum(),
                "n_ccg": len(g),
                **{
                    column: float(
                        np.sum(g["income_real"].to_numpy(np.float64) * g[column].to_numpy(np.float64))
                        / g["income_real"].sum()
                    )
                    for column in (
                        "shock_raw",
                        "shock_iq5",
                        f"z_raw_{SHARE}",
                        f"z_iq5_{SHARE}",
                    )
                },
            }
        ),
        include_groups=False,
    )
    return out.reset_index()


def aggregation_checks(checks: Checks, panel: pd.DataFrame, collapse_audit: pd.DataFrame):
    identity_worst = 0.0
    support_bad = []
    omega_worst = 0.0
    for period in PERIODS:
        for horizon in HORIZONS:
            block = panel[
                panel["period"].eq(period) & panel["horizon"].eq(horizon)
            ]
            for key, rung in (("cz", "cz"), ("state", "state")):
                agg = collapse(block, key)
                for column in ("income_real", "employment", "future"):
                    per_year_county = block.groupby("year")[column].sum()
                    per_year_agg = agg.groupby("year")[column].sum()
                    identity_worst = max(
                        identity_worst,
                        relative(
                            per_year_agg.reindex(per_year_county.index).to_numpy(),
                            per_year_county.to_numpy(),
                        ),
                    )
                total = agg.groupby("year")["n_ccg"].sum()
                if total.nunique() != 1 or int(total.iloc[0]) != EXPECTED_UNITS["county"]:
                    support_bad.append(f"{period} h={horizon} {rung}")
                if agg[key].nunique() != EXPECTED_UNITS[rung]:
                    support_bad.append(f"{period} h={horizon} {rung} unit count")
                # omega sums to one inside every unit-year, asserted year by
                # year rather than pooled.
                omega = block.groupby([key, "year"], sort=True)["income_real"].transform(
                    "sum"
                )
                shares = (block["income_real"] / omega).groupby(
                    [block[key], block["year"]]
                ).sum()
                omega_worst = max(omega_worst, float(np.max(np.abs(shares - 1.0))))

    checks.add(
        10,
        "unit income, employment and contracts are exact county sums",
        identity_worst < 1e-12,
        f"12 rung-blocks, max relative deviation {identity_worst:.2e}",
    )
    checks.add(
        11,
        "the county support is frozen: 3,080 counties under every rung-year",
        not support_bad and omega_worst < 1e-12,
        f"omega sums to 1 within {omega_worst:.2e}"
        if not support_bad
        else ", ".join(sorted(set(support_bad))[:4]),
    )

    audit_ok = (
        len(collapse_audit) == EXPECTED_COLLAPSE
        and collapse_audit["total_ccg"].round().eq(EXPECTED_UNITS["county"]).all()
        and all(
            int(collapse_audit.loc[collapse_audit["rung"].eq(rung), "n_units"].iloc[0])
            == EXPECTED_UNITS[rung]
            for rung in RUNGS
        )
    )
    thinnest = collapse_audit.loc[collapse_audit["rung"].eq("cz"), "min_ccg"].min()
    widest = collapse_audit.loc[collapse_audit["rung"].eq("cz"), "max_ccg"].max()
    checks.add(
        12,
        "Stata's own collapse audit agrees on unit counts and coverage",
        audit_ok,
        f"{len(collapse_audit)} rows, CZ holds {int(thinnest)} to {int(widest)} counties",
    )


def reestimate(checks: Checks, panel: pd.DataFrame, results: pd.DataFrame):
    """All 72 cells, independently, in pyfixest.

    Relative rather than absolute: pyfixest's default alternating-projection
    demeaner stops on a tolerance rather than at a fixed point, so equality
    against these fits is only ever meaningful at a relative scale. README
    1.8.6 records where that was found out.

    Both absorption arms are re-fitted here. This is the check that makes the new
    arm evidence rather than assertion: the `uy` cells are demeaned on `year`
    independently of anything 52 did, in a different language and a different
    estimator, so a wrong column in 52's `fe_arm` loop cannot survive it.
    """
    rows = []
    for period in PERIODS:
        for horizon in HORIZONS:
            block = panel[
                panel["period"].eq(period) & panel["horizon"].eq(horizon)
            ].copy()
            for rung in RUNGS:
                if rung == "county":
                    frame = block.rename(
                        columns={"unit": "rung_unit", "state_year": "fe_sxy"}
                    )
                else:
                    key = "cz" if rung == "cz" else "state"
                    frame = collapse(block, key).rename(columns={key: "rung_unit"})
                    if rung == "cz":
                        bridge = block[["cz", "year", "cz_state_year"]].drop_duplicates()
                        frame = frame.merge(
                            bridge,
                            left_on=["rung_unit", "year"],
                            right_on=["cz", "year"],
                            validate="one_to_one",
                        ).rename(columns={"cz_state_year": "fe_sxy"})
                    else:
                        frame["fe_sxy"] = frame["year"]
                # The second effect is the whole difference between the arms.
                frame["fe_uy"] = frame["year"]
                for fe_arm in FE_ARMS:
                    for arm, outcome, instrument in (
                        ("none", "shock_raw", f"z_raw_{SHARE}"),
                        ("iq5", "shock_iq5", f"z_iq5_{SHARE}"),
                    ):
                        coefficient, standard_error, statistic = fit(
                            frame, outcome, instrument, "rung_unit", f"fe_{fe_arm}"
                        )
                        rows.append(
                            {
                                "rung": rung,
                                "period": period,
                                "horizon": horizon,
                                "fe_arm": fe_arm,
                                "winsor_rule": arm,
                                "py_coef": coefficient,
                                "py_se": standard_error,
                                "py_F": statistic,
                            }
                        )
    python = pd.DataFrame(rows)
    merged = results.merge(
        python,
        on=["rung", "period", "horizon", "fe_arm", "winsor_rule"],
        validate="one_to_one",
    )
    deviation = max(
        relative(merged[stata], merged[py])
        for stata, py in (("fs_coef", "py_coef"), ("fs_se", "py_se"), ("fs_F", "py_F"))
    )
    checks.add(
        13,
        "Stata estimates match independent pyfixest estimates",
        len(merged) == EXPECTED_CELLS and deviation < 1e-6,
        f"{len(merged)} cells, max {deviation:.2e} relative",
    )


# ------------------------------------------------------------------------------
# Checks 14-16, the absorption arms
# ------------------------------------------------------------------------------

def state_arms_coincide(checks: Checks, results: pd.DataFrame):
    """At the state rung the two arms are the same regression. Exactly.

    The unit IS the state, so `state x year` is not defined and 52 sets both arms
    to `unit + year`. 52 asserts this itself; it is re-asserted here from the
    written CSV, because 52's assertion runs on data still in memory and this one
    runs on the artifact anybody downstream will read.
    """
    block = results[results["rung"].eq("state")]
    wide = block.pivot_table(
        index=["period", "horizon", "winsor_rule"],
        columns="fe_arm",
        values=["fs_coef", "fs_se", "fs_F", "df_absorbed"],
    )
    gaps = {
        column: float(
            (wide[(column, "sxy")] - wide[(column, "uy")]).abs().max()
        )
        for column in ("fs_coef", "fs_se", "fs_F", "df_absorbed")
    }
    checks.add(
        14,
        "state rung: the two absorption arms agree exactly",
        len(wide) == len(PERIODS) * len(HORIZONS) * len(ARMS)
        and all(gap == 0.0 for gap in gaps.values()),
        f"{len(wide)} pairs, max |gap| "
        + ", ".join(f"{k}={v:.1e}" for k, v in gaps.items()),
    )


def pin_frozen_arm(checks: Checks, results: pd.DataFrame):
    """The 36 cells that existed before the `uy` arm must not have moved.

    Adding an arm is a pure addition or it is a silent revision of a frozen
    exhibit. FROZEN_SXY is the git-tracked copy of ladder_first_stage.csv as it
    stood at the h=4 freeze, so this compares against a file this run cannot
    write. Skipped with an explicit note rather than silently if it is absent.
    """
    if not FROZEN_SXY.exists():
        checks.add(
            15, "the 36 frozen sxy cells are unchanged", None,
            f"baseline absent, not evaluated: {FROZEN_SXY}",
        )
        return
    frozen = pd.read_csv(FROZEN_SXY)
    mine = results[results["fe_arm"].eq(PIN_FE_ARM)]
    keys = ["rung", "period", "horizon", "winsor_rule"]
    merged = mine.merge(frozen, on=keys, suffixes=("", "_frozen"), validate="one_to_one")
    deviation = max(
        relative(merged[column], merged[f"{column}_frozen"])
        for column in ("fs_coef", "fs_se", "fs_F")
    )
    counts_ok = (
        merged["n_obs"].astype(int).equals(merged["n_obs_frozen"].astype(int))
        and merged["df_absorbed"].astype(int).equals(
            merged["df_absorbed_frozen"].astype(int)
        )
    )
    checks.add(
        15,
        "the 36 frozen sxy cells are unchanged",
        len(merged) == 36 and counts_ok and deviation == 0.0,
        f"{len(merged)} cells, counts exact, estimates {deviation:.2e} relative",
    )


def pin_step2_arm(checks: Checks, results: pd.DataFrame):
    """The `uy` arm against step_2's own county first stage. The external pin.

    step_2's Tables 1-5 estimate the county first stage on this instrument, this
    cap and these rows, absorbing ccg + year - which is the `uy` arm. Its F is
    written to step_2/result/coeff_candidate_iv_mpgr_*.csv by a pipeline that
    shares no code with 52. This is the only check in this file that binds the new
    arm to a number produced outside it.

    REPORTED, NOT ASSERTED. 52 rescales the second-step standard error by
    sqrt((N-1)/(N-df_a-2)) and step_2's estimator may treat the degrees of freedom
    differently; a gap of that shape is a convention difference, not a defect, and
    resolving which it is takes a decision rather than a tolerance. So this
    records the comparison and never fails the run on it.
    """
    if not STEP2_COEFF.exists():
        print(f"  step_2 coefficient file absent, not compared: {STEP2_COEFF}")
        return
    step2 = pd.read_csv(STEP2_COEFF)
    county = step2[step2["spec"].eq("county")]
    if county.empty:
        print(f"  no county row in {STEP2_COEFF.name}, not compared")
        return
    reference = county.iloc[0]
    mine = results[
        results["rung"].eq("county")
        & results["fe_arm"].eq(STEP2_FE_ARM)
        & results["period"].eq(str(reference["period"]))
        & results["horizon"].eq(int(reference["horizon"]))
        & results["winsor_rule"].eq("iq5")
    ]
    if len(mine) != 1:
        print(f"  no unique matching cell for the step_2 pin ({len(mine)} found)")
        return
    row = mine.iloc[0]
    gap = abs(float(row["fs_F"]) - float(reference["first_stage_F"]))
    scale = abs(float(reference["first_stage_F"])) or 1.0
    print(
        f"  step_2 pin, county {reference['period']} h={int(reference['horizon'])} "
        f"iq5 on the {STEP2_FE_ARM} arm:"
    )
    print(
        f"    this run F = {float(row['fs_F']):.6f} on N = {int(row['n_obs']):,}; "
        f"step_2 F = {float(reference['first_stage_F']):.6f} on N = "
        f"{int(reference['n']):,}"
    )
    print(f"    absolute gap {gap:.3e}, relative {gap / scale:.3e}")
    if int(row["n_obs"]) != int(reference["n"]):
        print(
            "    ROWS DIFFER, so the F gap is not attributable to the degrees of "
            "freedom alone. This needs a decision, not a tolerance."
        )


def main():
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        for path in (PANEL, RESULTS, COLLAPSE, DESIGN, CONTRACTS, COUNTY_PANEL):
            if not path.exists():
                raise FileNotFoundError(f"required input is missing: {path}")
        panel = pd.read_parquet(PANEL)
        results = pd.read_csv(RESULTS)
        collapse_audit = pd.read_csv(COLLAPSE)
        design = json.loads(DESIGN.read_text(encoding="utf-8"))

        checks = Checks()
        print("PREFLIGHT: the design running is the design that was asked for")
        preflight(checks, design, panel)
        print()
        print("THE PIN: the county rung is the candidate first stage")
        pin_county_rung(checks, results)

        # A failed pin means the county panel or the estimator is wrong, and an
        # aggregate built correctly out of a wrong county panel would pass every
        # check below. Stop here rather than report numbers nobody should read.
        if any(row["check"] == 9 and row["status"] == "FAIL" for row in checks.rows):
            print()
            print("county-rung pin FAILED; the aggregate rungs are not evaluated")
        else:
            print()
            print("THE LADDER: aggregation identities and independent estimates")
            grid_ok = (
                len(results) == EXPECTED_CELLS
                and results["status"].eq("ok").all()
                and results["n_ccg"].eq(EXPECTED_UNITS["county"]).all()
            )
            checks.add(
                0,
                "the result grid is complete and every fit converged",
                grid_ok,
                f"{len(results)} of {EXPECTED_CELLS} cells, "
                f"{int(results['status'].ne('ok').sum())} failures",
            )
            aggregation_checks(checks, panel, collapse_audit)
            reestimate(checks, panel, results)

            print()
            print("THE ABSORPTION ARMS: sxy is frozen, uy is what step_2 estimates")
            state_arms_coincide(checks, results)
            pin_frozen_arm(checks, results)

        print()
        print("REPORTED, NOT ASSERTED")
        above = results[results["fs_F"] > F_THRESHOLD]
        print(
            f"  {len(above)} of {len(results)} cells have F > {F_THRESHOLD}. "
            "Nothing in this pipeline is selected on that."
        )
        for fe_arm in FE_ARMS:
            for arm in ARMS:
                block = results[
                    results["fe_arm"].eq(fe_arm) & results["winsor_rule"].eq(arm)
                ]
                if block.empty:
                    continue
                print(
                    f"    fe={fe_arm:<4s} winsor={arm:<5s} F from "
                    f"{block['fs_F'].min():.2f} to {block['fs_F'].max():.2f}, "
                    f"{int(block['fs_F'].gt(F_THRESHOLD).sum())}/{len(block)} above "
                    f"{F_THRESHOLD}"
                )
        print()
        pin_step2_arm(checks, results)

        report = pd.DataFrame(checks.rows).sort_values("check")
        report.to_csv(REPORT, index=False)
        print()
        print(f"{len(report)} checks -> {REPORT}")
        if checks.failed:
            for row in checks.failed:
                print(f"  FAILED [{row['check']:02d}] {row['label']}: {row['detail']}")
            return 1
        print("all checks passed")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
