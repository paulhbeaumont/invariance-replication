#!/usr/bin/env python3
"""Verify the symmetric nonoverlapping first-stage pipeline.

The verifier independently checks the raw-derived county collapse, fixed-share
windows, nonoverlapping schedules, cumulative local and national flows, income
and employment scaling, symmetric winsorization, and Stata estimates.

Outputs:
    batch_logs/macro_version/symmetric_nonoverlap_verification.csv
    batch_logs/macro_version/43_verify_symmetric_nonoverlap_pipeline.log
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

TEMP = dc.SYMMETRIC_FIRST_STAGE_DIR
PANEL = TEMP / "panel.parquet"
DESIGN = dc.LOG_DIR / "symmetric_nonoverlap_design.json"
SHARE_AUDIT = dc.LOG_DIR / "symmetric_nonoverlap_share_audit.csv"
PANEL_AUDIT = dc.LOG_DIR / "symmetric_nonoverlap_panel_audit.csv"
RESULTS = dc.LOG_DIR / "symmetric_nonoverlap_first_stage.csv"
VERIFY_OUT = dc.LOG_DIR / "symmetric_nonoverlap_verification.csv"
LOG = dc.LOG_DIR / "43_verify_symmetric_nonoverlap_pipeline.log"
CONTRACTS = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
COUNTY_PANEL = dc.PANEL_OUT_DIR / "county_panel.parquet"
HARMONIZED_REPORT = dc.HARMONIZED_REPORT
DCADS_PARSE_REPORT = dc.OUT_DIR / "parse_verification_report.csv"
COUNTY_FIPS_REPORT = dc.OUT_DIR / "county_fips_verification_report.csv"
USASPENDING_REPORT = dc.USASPENDING_VERIFICATION_REPORT

PERIODS = {
    "pre": {"first": 1979, "last": 1997},
    "post": {"first": 1998, "last": 2018},
}
HORIZONS = (2, 3, 4)
SHARE_LENGTHS = (4, 5, 10)
EXPECTED_UNIVERSE = 3_094
EXPECTED_UNITS = 3_080
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


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, number: int, label: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        print(f"[{number:02d}] {label:<54s} {status}  {detail}")
        self.rows.append(
            {
                "check": number,
                "label": label,
                "status": status,
                "detail": detail,
            }
        )

    @property
    def failed(self):
        return [row for row in self.rows if row["status"] == "FAIL"]


def starts(period: str, horizon: int):
    bounds = PERIODS[period]
    return list(range(bounds["first"], bounds["last"] - horizon + 1, horizon))


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


WINSOR_CELLS = (
    ("none", "none"),
    ("iq4", "pooled"),
    ("iq4", "year"),
    ("iq5", "pooled"),
    ("iq5", "year"),
    ("tail1", "pooled"),
    ("tail1", "year"),
)


def winsor_bounds(values: pd.Series, rule: str):
    if rule == "tail1":
        return tuple(
            np.quantile(
                values.to_numpy(), (0.01, 0.99), method="averaged_inverted_cdf"
            )
        )
    level = {"iq4": 4, "iq5": 5}[rule]
    q25, q50, q75 = np.quantile(
        values.to_numpy(),
        (0.25, 0.50, 0.75),
        method="averaged_inverted_cdf",
    )
    return q50 - level * (q75 - q25), q50 + level * (q75 - q25)


def winsor(frame: pd.DataFrame, source: str, rule: str, group: str):
    result = frame[source].copy()
    if rule == "none":
        return result
    groups = [(None, frame)] if group == "pooled" else frame.groupby("year", sort=False)
    for _, cell in groups:
        lower, upper = winsor_bounds(cell[source], rule)
        result.loc[cell.index] = cell[source].clip(lower, upper)
    return result


def fit(frame: pd.DataFrame, outcome: str, instrument: str):
    outcome_scale = 1.0 / max(float(frame[outcome].std()), 1e-30)
    instrument_scale = 1.0 / max(float(frame[instrument].std()), 1e-30)
    fit_frame = frame.assign(
        outcome_scaled=frame[outcome] * outcome_scale,
        instrument_scaled=frame[instrument] * instrument_scale,
    )
    model = pf.feols(
        "outcome_scaled ~ instrument_scaled | unit + state_year",
        data=fit_frame,
        vcov={"CRV1": "unit"},
        demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000),
    )
    rescale = instrument_scale / outcome_scale
    coefficient = float(model.coef()["instrument_scaled"]) * rescale
    standard_error = float(model.se()["instrument_scaled"]) * rescale
    return coefficient, standard_error, (coefficient / standard_error) ** 2


def source_checks(checks: Checks, contracts: pd.DataFrame):
    parse = pd.read_csv(DCADS_PARSE_REPORT)
    parse_ok = parse["status"].eq("ok").all() and parse["fails"].isna().all()
    checks.add(
        1,
        "DCADS parse verification is clean",
        bool(parse_ok),
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
        float(
            np.max(
                np.abs(
                    comparison[f"{column}_report"]
                    - comparison[f"{column}_cache"]
                )
            )
        )
        for column in ("dollars_nominal", "dollars_real")
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


def share_checks(
    checks: Checks,
    contracts: pd.DataFrame,
    universe: set[int],
    panel: pd.DataFrame,
    share_audit: pd.DataFrame,
):
    checks.add(
        6,
        "positive-employment universe is fixed",
        len(universe) == EXPECTED_UNIVERSE,
        f"{len(universe):,} counties",
    )
    checks.add(
        7,
        "analysis support is common across periods",
        panel["unit"].nunique() == EXPECTED_UNITS
        and panel["state"].nunique() == EXPECTED_STATES,
        f"{panel['unit'].nunique():,} counties, {panel['state'].nunique()} states",
    )

    expected_windows = []
    for period, bounds in PERIODS.items():
        end = bounds["first"] - 1
        for length in SHARE_LENGTHS:
            expected_windows.append(
                (period, f"fixed{length}", end - length + 1, end, length)
            )
    observed = {
        (
            row.period,
            row.share_label,
            int(row.share_first),
            int(row.share_last),
            int(row.share_years),
        )
        for row in share_audit.itertuples(index=False)
    }
    checks.add(
        8,
        "share windows are symmetric relative to each period",
        observed == set(expected_windows)
        and share_audit["years_between_share_and_period"].eq(0).all(),
        "4, 5, and 10 years ending immediately before each period",
    )

    analysis_units = set(panel["unit"].astype(int).unique())
    levels = contracts[contracts["ccg"].isin(analysis_units)].pivot(
        index="ccg", columns="year", values="dollars_real"
    )
    deviations = []
    for row in share_audit.itertuples(index=False):
        numerator = levels.loc[:, row.share_first : row.share_last].sum(axis=1)
        denominator = float(numerator.sum())
        deviations.extend(
            [
                abs(denominator - row.national_real_contracts),
                abs(float((numerator / denominator).sum()) - row.sum_analysis_shares),
            ]
        )
    scale = max(float(share_audit["national_real_contracts"].abs().max()), 1.0)
    share_deviation = max(deviations, default=np.inf) / scale
    checks.add(
        9,
        "fixed shares reproduce raw county sums",
        share_deviation < 1e-12
        and np.allclose(share_audit["sum_analysis_shares"], 1.0, atol=1e-12),
        f"scaled maximum deviation {share_deviation:.2e}",
    )


def panel_checks(
    checks: Checks,
    contracts: pd.DataFrame,
    county_panel: pd.DataFrame,
    panel: pd.DataFrame,
    panel_audit: pd.DataFrame,
):
    schedule_ok = True
    overlap_ok = True
    for period in PERIODS:
        for horizon in HORIZONS:
            expected = starts(period, horizon)
            row = panel_audit[
                panel_audit["period"].eq(period)
                & panel_audit["horizon"].eq(horizon)
            ]
            if len(row) != 1:
                schedule_ok = False
                continue
            observed = list(map(int, row.iloc[0]["start_years"].split()))
            schedule_ok &= observed == expected
            blocks = [set(range(year + 1, year + horizon + 1)) for year in observed]
            overlap_ok &= len(set().union(*blocks)) == sum(map(len, blocks))
    checks.add(
        10,
        "nonoverlapping schedules are period-anchored",
        schedule_ok and overlap_ok,
        "all six period-horizon schedules exact with disjoint future blocks",
    )

    expected_rows = sum(
        EXPECTED_UNITS * len(starts(period, horizon))
        for period in PERIODS
        for horizon in HORIZONS
    )
    keys_ok = not panel.duplicated(["period", "horizon", "unit", "year"]).any()
    checks.add(
        11,
        "analysis panel has exact keys and row count",
        len(panel) == expected_rows and keys_ok,
        f"{len(panel):,}/{expected_rows:,} rows",
    )

    analysis_units = set(panel["unit"].astype(int).unique())
    raw = contracts[contracts["ccg"].isin(analysis_units)].set_index(["ccg", "year"])
    national_contracts = (
        contracts[contracts["ccg"].isin(analysis_units)]
        .groupby("year", sort=False)["dollars_real"]
        .sum()
    )
    scaling = county_panel[county_panel["ccg"].isin(analysis_units)].set_index(
        ["ccg", "year"]
    )[["income_real", "emp"]]
    national_scaling = (
        county_panel[county_panel["ccg"].isin(analysis_units)]
        .groupby("year", sort=False)[["income_real", "emp"]]
        .sum()
    )
    sample = panel.groupby(["period", "horizon"], group_keys=False).sample(
        n=200, random_state=20260815
    )
    deviations = []

    def relative_deviation(observed, expected):
        return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1.0)

    for row in sample.itertuples(index=False):
        future_years = list(range(row.year + 1, row.year + row.horizon + 1))
        future = float(
            raw.loc[(row.unit, future_years), "dollars_real"].sum()
        )
        national_future = float(national_contracts.loc[future_years].sum())
        income_value = float(scaling.loc[(row.unit, row.year), "income_real"])
        employment_value = float(scaling.loc[(row.unit, row.year), "emp"])
        national_income = float(national_scaling.loc[row.year, "income_real"])
        national_employment = float(national_scaling.loc[row.year, "emp"])
        share_end = PERIODS[row.period]["first"] - 1
        for length in SHARE_LENGTHS:
            label = f"fixed{length}"
            share_first = share_end - length + 1
            local_share = float(
                raw.loc[
                    (row.unit, list(range(share_first, share_end + 1))),
                    "dollars_real",
                ].sum()
            )
            national_share = float(national_contracts.loc[share_first:share_end].sum())
            share = local_share / national_share
            deviations.extend(
                [
                    relative_deviation(getattr(row, label), share),
                    relative_deviation(
                        getattr(row, f"z_income_{label}"),
                        share * national_future / national_income,
                    ),
                    relative_deviation(
                        getattr(row, f"z_employment_{label}"),
                        share * national_future / national_employment,
                    ),
                ]
            )
        deviations.extend(
            [
                relative_deviation(row.future, future),
                relative_deviation(row.national_future, national_future),
                relative_deviation(row.income_real, income_value),
                relative_deviation(row.employment, employment_value),
                relative_deviation(row.national_income, national_income),
                relative_deviation(row.national_employment, national_employment),
                relative_deviation(row.y_income, future / income_value),
                relative_deviation(row.y_employment, future / employment_value),
            ]
        )
    cell_deviation = max(deviations, default=np.inf)
    checks.add(
        12,
        "sampled cells reproduce county and national scaling",
        len(sample) == 1_200 and cell_deviation < 1e-12,
        f"1,200 cells, maximum relative deviation {cell_deviation:.2e}",
    )


def regression_checks(checks: Checks, panel: pd.DataFrame, results: pd.DataFrame):
    key = [
        "period",
        "horizon",
        "scaling",
        "share_label",
        "winsor_rule",
        "winsor_group",
    ]
    observed_winsor = set(
        map(tuple, results[["winsor_rule", "winsor_group"]].drop_duplicates().to_numpy())
    )
    structure_ok = (
        len(results) == 252
        and not results.duplicated(key).any()
        and set(results["scaling"]) == {"income", "employment"}
        and observed_winsor == set(WINSOR_CELLS)
        and results["status"].eq("ok").all()
        and np.isfinite(results[["fs_coef", "fs_se", "fs_F"]].to_numpy()).all()
    )
    checks.add(
        13,
        "Stata regression grid is complete",
        structure_ok,
        f"{len(results)}/252 successful regressions",
    )

    python_rows = []
    changed_lhs = []
    changed_rhs = []
    for period in PERIODS:
        for horizon in HORIZONS:
            frame = panel[
                panel["period"].eq(period) & panel["horizon"].eq(horizon)
            ].copy()
            for scaling in ("income", "employment"):
                for rule, group in WINSOR_CELLS:
                    frame["y_python"] = winsor(
                        frame, f"y_{scaling}", rule, group
                    )
                    if rule != "none":
                        changed_lhs.append(
                            not np.array_equal(
                                frame["y_python"].to_numpy(),
                                frame[f"y_{scaling}"].to_numpy(),
                            )
                        )
                    for length in SHARE_LENGTHS:
                        label = f"fixed{length}"
                        source_z = f"z_{scaling}_{label}"
                        frame["z_python"] = winsor(frame, source_z, rule, group)
                        if rule != "none":
                            changed_rhs.append(
                                not np.array_equal(
                                    frame["z_python"].to_numpy(),
                                    frame[source_z].to_numpy(),
                                )
                            )
                        coefficient, standard_error, statistic = fit(
                            frame, "y_python", "z_python"
                        )
                        python_rows.append(
                            {
                                "period": period,
                                "horizon": horizon,
                                "scaling": scaling,
                                "share_label": label,
                                "winsor_rule": rule,
                                "winsor_group": group,
                                "py_coef": coefficient,
                                "py_se": standard_error,
                                "py_F": statistic,
                            }
                        )
    python_results = pd.DataFrame(python_rows)
    compared = results.merge(
        python_results,
        on=key,
        validate="one_to_one",
    )
    estimator_deviation = max(
        float(np.max(np.abs(compared[stata] - compared[python])))
        for stata, python in (
            ("fs_coef", "py_coef"),
            ("fs_se", "py_se"),
            ("fs_F", "py_F"),
        )
    )
    checks.add(
        14,
        "Stata estimates match independent pyfixest estimates",
        len(compared) == 252 and estimator_deviation < 1e-6,
        f"252 cells, maximum deviation {estimator_deviation:.2e}",
    )

    checks.add(
        15,
        "winsorization changes both sides of every capped cell",
        bool(changed_lhs)
        and bool(changed_rhs)
        and all(changed_lhs)
        and all(changed_rhs),
        f"LHS {sum(changed_lhs)}/{len(changed_lhs)}; "
        f"RHS {sum(changed_rhs)}/{len(changed_rhs)} cells changed",
    )

    f_identity = np.max(
        np.abs(results["fs_F"] - (results["fs_coef"] / results["fs_se"]) ** 2)
    )
    checks.add(
        16,
        "reported F statistics match coefficients and standard errors",
        (results["fs_se"] > 0).all() and f_identity < 1e-10,
        f"maximum identity deviation {f_identity:.2e}; "
        f"F range {results['fs_F'].min():.6f} to {results['fs_F'].max():.6f}",
    )


def main():
    required = [
        PANEL,
        DESIGN,
        SHARE_AUDIT,
        PANEL_AUDIT,
        RESULTS,
        CONTRACTS,
        COUNTY_PANEL,
        HARMONIZED_REPORT,
        DCADS_PARSE_REPORT,
        COUNTY_FIPS_REPORT,
        USASPENDING_REPORT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required verification inputs missing: " + ", ".join(missing))

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
        results = pd.read_csv(RESULTS)

        design_ok = (
            design.get("contract") == "symmetric_nonoverlap_first_stage_v2"
            and design["source"]["contract_flow"] == "signed award-year obligations"
            and design["source"]["dollar_treatment"] == "no nominal floor"
            and set(design["scalings"]) == {"income", "employment"}
            and design["winsorization"]["variables"]
            == "LHS and RHS separately with the same method and grouping"
            and design["regression"]["fixed_effects"]
            == "county and state-by-year"
        )
        checks.add(0, "design contract is pinned", design_ok, design.get("contract", "missing"))
        source_checks(checks, contracts)
        universe = positive_universe(county_panel)
        share_checks(checks, contracts, universe, panel, share_audit)
        panel_checks(checks, contracts, county_panel, panel, panel_audit)
        regression_checks(checks, panel, results)

        pd.DataFrame(checks.rows).to_csv(VERIFY_OUT, index=False)
        if checks.failed:
            print(f"{len(checks.failed)} verification checks failed")
            return 1
        print(f"all {len(checks.rows)} verification checks passed -> {VERIFY_OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
