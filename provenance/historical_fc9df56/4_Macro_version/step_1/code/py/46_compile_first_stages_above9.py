#!/usr/bin/env python3
"""Compile every first-stage specification with an F-statistic above nine.

The output combines the endpoint-growth benchmark grid with the symmetric
county-scaled grid. Design labels preserve differences in national aggregation,
winsorization scope, sample period, and transformation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402


ENDPOINT = dc.LOG_DIR / "endpoint_growth_first_stage_grid.csv"
SCALED = dc.LOG_DIR / "symmetric_nonoverlap_first_stage.csv"
SHARES = dc.LOG_DIR / "post_first_stage_share_dictionary.csv"
OUT = dc.LOG_DIR / "all_first_stage_specs_f_gt9.csv"
FAMILIES = dc.LOG_DIR / "all_first_stage_f_gt9_families.csv"


def endpoint_rows() -> pd.DataFrame:
    d = pd.read_csv(ENDPOINT)
    shares = pd.read_csv(SHARES)
    d = d.merge(
        shares,
        on=["rung", "share_code"],
        how="left",
        validate="many_to_one",
    )
    d = d[d["fs_F"] > 9].copy()
    d["design"] = "endpoint_growth_benchmark"
    d["period"] = "post"
    d["scaling"] = "none"
    d["window_sampling"] = "overlapping_endpoint"
    d["winsor_scope"] = "lhs_only"
    d["national_scope"] = "leave_one_unit_out"
    d["lhs_definition"] = d["growth"].map(
        {
            "std": "(D_ct-D_c,t-h)/D_c,t-h",
            "log": "ln(D_ct)-ln(D_c,t-h)",
            "mpgr": "2(D_ct-D_c,t-h)/(D_ct+D_c,t-h)",
        }
    )
    d["rhs_definition"] = (
        "share_c times leave-one-unit-out national growth of the same type"
    )
    return d


def scaled_rows() -> pd.DataFrame:
    d = pd.read_csv(SCALED)
    d = d[d["fs_F"] > 9].copy()
    d["design"] = "symmetric_county_scaled"
    d["rung"] = "county"
    d["flow"] = "award"
    d["treatment"] = "net_nofloor"
    d["growth"] = "future_level_over_denominator"
    d["share_code"] = d["share_label"].map(
        {"fixed4": "s01", "fixed5": "s02", "fixed10": "s03"}
    )
    d["share_type"] = "fixed"
    d["window_years"] = pd.to_numeric(
        d["share_label"].str.replace("fixed", "", regex=False)
    )
    d["end_lag"] = pd.NA
    period_first = d["period"].map({"pre": 1979, "post": 1998})
    d["fixed_first"] = period_first - d["window_years"]
    d["fixed_last"] = period_first - 1
    d["window_sampling"] = "period_anchored_nonoverlap"
    d["winsor_scope"] = "lhs_and_rhs_separately"
    d["national_scope"] = "sum_over_3080_counties"
    d["lhs_definition"] = d["scaling"].map(
        {
            "income": "sum(D_c,t+1:t+h)/county_income_ct",
            "employment": "sum(D_c,t+1:t+h)/county_employment_ct",
        }
    )
    d["rhs_definition"] = d["scaling"].map(
        {
            "income": "share_c times national sum(D_t+1:t+h)/national_income_t",
            "employment": "share_c times national sum(D_t+1:t+h)/national_employment_t",
        }
    )
    return d


def main() -> None:
    missing = [str(path) for path in (ENDPOINT, SCALED, SHARES) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs missing: {missing}")

    d = pd.concat([endpoint_rows(), scaled_rows()], ignore_index=True, sort=False)
    d["positive_coef"] = d["fs_coef"] > 0
    columns = [
        "design",
        "period",
        "rung",
        "flow",
        "treatment",
        "horizon",
        "growth",
        "scaling",
        "share_code",
        "share_type",
        "share_label",
        "window_years",
        "end_lag",
        "fixed_first",
        "fixed_last",
        "window_sampling",
        "winsor_rule",
        "winsor_group",
        "winsor_scope",
        "national_scope",
        "lhs_definition",
        "rhs_definition",
        "n_obs",
        "n_units",
        "n_states",
        "df_absorbed",
        "fs_coef",
        "fs_se",
        "fs_F",
        "positive_coef",
        "status",
    ]
    d = d[columns].sort_values(
        ["fs_F", "design", "period", "scaling"], ascending=[False, True, True, True]
    )
    d.to_csv(OUT, index=False)

    family_keys = ["design", "period", "rung", "flow", "growth", "scaling", "positive_coef"]
    families = (
        d.groupby(family_keys, dropna=False)
        .agg(
            n_specs=("fs_F", "size"),
            min_F=("fs_F", "min"),
            max_F=("fs_F", "max"),
            min_coef=("fs_coef", "min"),
            max_coef=("fs_coef", "max"),
        )
        .reset_index()
        .sort_values(["design", "period", "rung", "flow", "growth", "scaling", "positive_coef"])
    )
    families.to_csv(FAMILIES, index=False)

    assert len(d) == 941, len(d)
    assert int(d["positive_coef"].sum()) == 440
    print(f"{len(d):,} specifications -> {OUT}")
    print(f"{len(families):,} families -> {FAMILIES}")


if __name__ == "__main__":
    main()
