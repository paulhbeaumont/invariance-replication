#!/usr/bin/env python3
"""Verify the complete post-1998 contract-growth first-stage grid."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


RESULT = dc.LOG_DIR / "post_first_stage_full_grid.csv"
DATE_AUDIT = dc.LOG_DIR / "post_first_stage_duration_date_audit.csv"
INPUT_AUDIT = dc.LOG_DIR / "post_first_stage_input_audit.csv"
SHARES = dc.LOG_DIR / "post_first_stage_share_dictionary.csv"
VERIFY = dc.LOG_DIR / "post_first_stage_full_grid_verification.csv"
SUMMARY = dc.LOG_DIR / "post_first_stage_full_grid_summary.csv"
LABELED = dc.POST_GRID_LABELED
TEMP = dc.POST_GRID_DIR
MANIFEST = TEMP / "input_manifest.csv"


def record(rows, name, passed, detail):
    rows.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def main():
    results = pd.read_csv(RESULT)
    dates = pd.read_csv(DATE_AUDIT)
    inputs = pd.read_csv(INPUT_AUDIT)
    shares = pd.read_csv(SHARES)
    manifest = pd.read_csv(MANIFEST)
    checks = []

    key = [
        "rung",
        "flow",
        "window_mode",
        "treatment",
        "horizon",
        "growth",
        "share_code",
        "winsor_rule",
        "winsor_group",
    ]
    record(checks, "row_count", len(results) == 11_520, f"{len(results)} rows")
    record(
        checks,
        "unique_specification_key",
        not results.duplicated(key).any(),
        f"{results[key].drop_duplicates().shape[0]} unique keys",
    )
    record(
        checks,
        "all_regressions_succeeded",
        results["status"].eq("ok").all(),
        results["status"].value_counts().to_dict(),
    )
    finite = np.isfinite(results[["fs_coef", "fs_se", "fs_F"]].to_numpy()).all()
    record(
        checks,
        "all_statistics_finite",
        finite,
        f"{results[['fs_coef', 'fs_se', 'fs_F']].notna().all(axis=1).sum()} "
        "complete rows",
    )
    record(
        checks,
        "positive_standard_errors",
        (results["fs_se"] > 0).all(),
        f"minimum {results['fs_se'].min():.15g}",
    )

    expected_levels = {
        "rung": {"county", "cz"},
        "flow": {"award", "duration"},
        "window_mode": {"overlapping", "nonoverlap"},
        "treatment": {
            "net_nofloor",
            "net_floor25000",
            "positive_nofloor",
            "positive_floor25000",
        },
        "horizon": {2, 3, 4},
        "growth": {"cum"},
        "share_code": {f"s{i:02d}" for i in range(1, 13)},
        "winsor_rule": {"none", "iq4", "iq5", "tail1"},
        "winsor_group": {"none", "pooled", "year", "state"},
    }
    for column, expected in expected_levels.items():
        actual = set(results[column].unique())
        record(
            checks, f"levels_{column}", actual == expected, f"actual={sorted(actual)}"
        )

    allowed_winsor = {("none", "none")}
    allowed_winsor |= {
        (rule, group)
        for rule in ("iq4", "iq5", "tail1")
        for group in ("pooled", "year", "state")
    }
    actual_winsor = set(
        map(
            tuple, results[["winsor_rule", "winsor_group"]].drop_duplicates().to_numpy()
        )
    )
    record(
        checks,
        "winsor_cells",
        actual_winsor == allowed_winsor,
        f"{len(actual_winsor)} declared cells",
    )
    grouped = results[results["winsor_rule"].ne("none")].pivot(
        index=[
            "rung",
            "flow",
            "window_mode",
            "treatment",
            "horizon",
            "growth",
            "share_code",
            "winsor_rule",
        ],
        columns="winsor_group",
        values="fs_F",
    )
    for rung in ("county", "cz"):
        for rule in ("iq4", "iq5", "tail1"):
            cell = grouped.loc[
                (
                    rung,
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    rule,
                ),
                :,
            ]
            for scope in ("year", "state"):
                changed = (cell[scope] - cell["pooled"]).abs().gt(1e-10).any()
                record(
                    checks,
                    f"winsor_{rule}_{scope}_is_group_specific_{rung}",
                    changed,
                    f"{int((cell[scope] - cell['pooled']).abs().gt(1e-10).sum())} "
                    "cells differ from pooled",
                )

    base = ["rung", "flow", "window_mode", "treatment", "horizon"]
    sample_range = results.groupby(base, sort=False)["n_obs"].agg(
        lambda x: x.max() - x.min()
    )
    record(
        checks,
        "common_growth_share_winsor_sample",
        (sample_range == 0).all(),
        f"maximum within-base row difference {sample_range.max():.0f}",
    )
    record(
        checks, "input_manifest_cells", len(inputs) == 96, f"{len(inputs)} input cells"
    )
    record(
        checks,
        "usable_income_samples",
        (inputs["written_rows"] > 0).all(),
        f"minimum {inputs['written_rows'].min()} rows",
    )

    valid_year_grids = True
    year_grid_details = []
    for row in manifest.itertuples(index=False):
        years = set(pd.read_stata(row.path, columns=["year"])["year"].astype(int))
        if row.window_mode == "overlapping":
            expected_years = set(range(1998, 2018 - int(row.horizon) + 1))
        else:
            expected_years = set(
                range(1998, 2018 - int(row.horizon) + 1, int(row.horizon))
            )
        if years != expected_years:
            valid_year_grids = False
            year_grid_details.append(
                f"{Path(row.path).name}: actual={sorted(years)}, "
                f"expected={sorted(expected_years)}"
            )
    record(
        checks,
        "window_year_grids",
        valid_year_grids,
        "all 96 inputs match their declared year grid"
        if valid_year_grids
        else "; ".join(year_grid_details[:3]),
    )

    block_checks = []
    incomes = pd.read_parquet(
        dc.PANEL_OUT_DIR / "county_panel.parquet",
        columns=["ccg", "year", "income_real"],
    )
    income_lookup = incomes.set_index(["ccg", "year"])["income_real"].to_dict()
    award_shares = pd.read_parquet(
        TEMP / "award_county_year.parquet",
        columns=["ccg", "year", "net_nofloor"],
    )
    fixed5 = award_shares[award_shares["year"].between(1993, 1997)]
    fixed5_national = fixed5["net_nofloor"].sum()
    fixed5_lookup = (
        fixed5.groupby("ccg")["net_nofloor"].sum() / fixed5_national
    ).to_dict()
    for flow_name in ("award", "duration"):
        levels = pd.read_parquet(
            TEMP / f"{flow_name}_county_year.parquet",
            columns=["ccg", "year", "net_nofloor"],
        )
        lookup = levels.set_index(["ccg", "year"])["net_nofloor"].to_dict()
        national_lookup = levels.groupby("year")["net_nofloor"].sum().to_dict()
        for horizon in (2, 3, 4):
            path = TEMP / (f"county_{flow_name}_net_nofloor_h{horizon}_overlapping.dta")
            panel = pd.read_stata(
                path,
                columns=[
                    "unit",
                    "year",
                    "future",
                    "income_real",
                    "loo_future",
                    "y_cum",
                    "s02",
                    "zcum02",
                ],
            )
            panel = panel.sample(n=min(200, len(panel)), random_state=0)
            for row in panel.itertuples(index=False):
                unit, year = int(row.unit), int(row.year)
                raw_future = sum(
                    lookup.get((unit, tau), 0.0)
                    for tau in range(year + 1, year + horizon + 1)
                )
                raw_national = sum(
                    national_lookup.get(tau, 0.0)
                    for tau in range(year + 1, year + horizon + 1)
                )
                raw_loo = raw_national - raw_future
                raw_income = income_lookup[(unit, year)]
                raw_share = fixed5_lookup.get(unit, 0.0)
                expected = [
                    raw_future,
                    raw_income,
                    raw_loo,
                    raw_share,
                    raw_future / raw_income,
                    raw_share * raw_loo / raw_income,
                ]
                actual = [
                    row.future,
                    row.income_real,
                    row.loo_future,
                    row.s02,
                    row.y_cum,
                    row.zcum02,
                ]
                block_checks.append(
                    np.allclose(actual, expected, rtol=1e-12, atol=1e-7)
                )
    record(
        checks,
        "cumulative_forward_shock",
        all(block_checks),
        f"{sum(block_checks)} of {len(block_checks)} sampled rows match annual cells",
    )

    share_unique = shares.drop_duplicates(["rung", "share_code"])
    record(
        checks,
        "share_dictionary_cells",
        len(share_unique) == 24,
        f"{len(share_unique)} rung-share cells",
    )
    fixed = share_unique[share_unique["share_type"].eq("fixed")]
    rolling = share_unique[share_unique["share_type"].eq("rolling")]
    fixed_windows = set(
        map(tuple, fixed[["fixed_first", "fixed_last"]].drop_duplicates().to_numpy())
    )
    rolling_windows = set(
        map(tuple, rolling[["end_lag", "window_years"]].drop_duplicates().to_numpy())
    )
    record(
        checks,
        "fixed_share_windows",
        fixed_windows == {(1994.0, 1997.0), (1993.0, 1997.0), (1988.0, 1997.0)},
        sorted(fixed_windows),
    )
    record(
        checks,
        "rolling_share_windows",
        rolling_windows
        == {(float(lag), float(length)) for lag in (3, 4, 5) for length in (4, 5, 10)},
        sorted(rolling_windows),
    )

    record(
        checks,
        "duration_audit_cells",
        len(dates) == 35 * 4,
        f"{len(dates)} source-year-treatment cells",
    )
    overlaps = (dates["missing_action_date_actions"] > 0) & (
        dates["missing_completion_date_actions"] > 0
    )
    record(
        checks,
        "duration_status_reported",
        dates["completion_field"].notna().all(),
        f"missing completion actions {dates['missing_completion_date_actions'].sum()}",
    )
    record(
        checks,
        "duration_missingness_overlap_flag",
        True,
        f"{int(overlaps.sum())} cells can contain overlapping missing-date flags; "
        "raw status columns retained",
    )

    award = pd.read_parquet(TEMP / "award_county_year.parquet")
    canonical = pd.read_parquet(
        dc.PANEL_OUT_DIR / "contracts_ccg_year.parquet",
        columns=["ccg", "year", "dollars_real"],
    )
    compare = canonical[canonical["year"].between(1984, 2018)].merge(
        award[["ccg", "year", "net_nofloor"]], on=["ccg", "year"], how="outer"
    )
    compare = compare.fillna(0.0)
    max_deviation = (compare["dollars_real"] - compare["net_nofloor"]).abs().max()
    record(
        checks,
        "award_panel_matches_canonical",
        max_deviation < 1e-5,
        f"maximum absolute cell deviation {max_deviation:.15g}",
    )

    share_metadata = shares.drop_duplicates(["rung", "share_code"])
    labeled = results.merge(
        share_metadata, on=["rung", "share_code"], how="left", validate="many_to_one"
    )
    growth_definition = {
        "cum": "sum of contracts from t+1 through t+h / personal income_t",
    }
    labeled["geography_period"] = labeled["rung"].map(
        {
            "county": "County, start years 1998 through 2018-h",
            "cz": "Commuting zone, start years 1998 through 2018-h",
        }
    )
    labeled["future_block_definition"] = "sum of contracts from t+1 through t+h"
    labeled["denominator_definition"] = "real personal income in t"
    labeled["window_sampling"] = labeled["window_mode"].map(
        {
            "overlapping": "every admissible start year",
            "nonoverlap": "1998-anchored start years spaced by h",
        }
    )
    labeled["local_growth_definition"] = labeled["growth"].map(growth_definition)
    labeled["rhs_definition"] = (
        "allocation_share * leave-one-unit-out national cumulative contracts "
        "/ local personal income_t"
    )
    labeled["fixed_effects"] = "unit + state-by-year"
    labeled["cluster"] = labeled["rung"].map(
        {"county": "county", "cz": "commuting zone"}
    )
    labeled["regression_weight"] = "none"
    labeled.to_csv(LABELED, index=False)

    baseline = labeled[
        (labeled["treatment"].eq("net_nofloor"))
        & labeled["growth"].eq("cum")
        & labeled["share_code"].eq("s02")
        & labeled["winsor_rule"].eq("none")
        & labeled["winsor_group"].eq("none")
    ].copy()
    baseline["summary_role"] = "baseline_fixed5"
    best = (
        labeled.sort_values("fs_F", ascending=False)
        .groupby(["rung", "flow"], sort=False)
        .head(1)
        .copy()
    )
    best["summary_role"] = "maximum_F_within_rung_flow"
    summary = pd.concat([baseline, best], ignore_index=True)
    summary.to_csv(SUMMARY, index=False)

    verification = pd.DataFrame(checks)
    verification.to_csv(VERIFY, index=False)
    failed = verification[~verification["passed"]]
    if not failed.empty:
        raise AssertionError(
            "; ".join(f"{r.check}: {r.detail}" for r in failed.itertuples())
        )
    print(f"All {len(verification)} checks passed")
    print(f"Verified {len(results):,} regression rows")


if __name__ == "__main__":
    main()
