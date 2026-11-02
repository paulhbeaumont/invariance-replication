#!/usr/bin/env python3
"""Estimate pre-1998 county first stages on nonoverlapping award-year windows.

The regression mirrors the post-1998 award-year specification. At horizon h,
the local shock sums signed real obligations from t+1 through t+h and scales
them by real personal income in t. The instrument multiplies a fixed national
allocation share by the matching leave-one-county-out national sum and uses the
same income denominator. Regressions absorb county and state-by-year fixed
effects, cluster by county, and report the raw outcome together with annual
1st-to-99th-percentile winsorization.

Output:
    batch_logs/macro_version/pre_first_stage_nonoverlap.csv
    batch_logs/macro_version/40_pre_first_stage_nonoverlap.log
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


CODE = Path(__file__).resolve().parent
POST_BUILDER = CODE / "35_build_post_first_stage_full_grid.py"
CONTRACT_CACHE = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
POST_INPUT = (
    dc.POST_GRID_DIR
    / "county_award_net_nofloor_h3_nonoverlap.dta"
)
POST_RESULTS = dc.POST_GRID_LABELED
OUT = dc.LOG_DIR / "pre_first_stage_nonoverlap.csv"
LOG = dc.LOG_DIR / "40_pre_first_stage_nonoverlap.log"

ANALYSIS_FIRST = 1979
ANALYSIS_LAST = 1997
HORIZONS = (2, 3, 4)
FIXED_SHARES = (
    ("fixed4", 1971, 1974),
    ("fixed5", 1970, 1974),
    ("fixed10", 1966, 1975),
)
EXPECTED_WINDOWS = {2: 9, 3: 6, 4: 4}


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
    spec = importlib.util.spec_from_file_location("post_grid_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def winsor_tail_by_year(frame: pd.DataFrame, source: str, target: str):
    """Apply Stata winsor2's empirical 1st and 99th percentiles by year."""
    frame[target] = frame[source]
    for _, group in frame.groupby("year", sort=False):
        lower, upper = np.quantile(
            group[source].to_numpy(), (0.01, 0.99), method="inverted_cdf"
        )
        frame.loc[group.index, target] = group[source].clip(lower, upper)


def fit(frame: pd.DataFrame, outcome: str, instrument: str):
    model = pf.feols(
        f"{outcome} ~ {instrument} | unit + state_year",
        data=frame,
        vcov={"CRV1": "unit"},
        demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000),
    )
    coefficient = float(model.coef()[instrument])
    standard_error = float(model.se()[instrument])
    return {
        "n_obs": int(model._N),
        "n_units": int(frame["unit"].nunique()),
        "n_states": int(frame["state"].nunique()),
        "fs_coef": coefficient,
        "fs_se": standard_error,
        "fs_F": (coefficient / standard_error) ** 2,
    }


def validate_estimator():
    """Reproduce the corresponding post-period Stata result exactly."""
    if not POST_INPUT.exists() or not POST_RESULTS.exists():
        raise FileNotFoundError("post-period validation inputs are missing")
    frame = pd.read_stata(POST_INPUT)
    winsor_tail_by_year(frame, "y_cum", "y_tail1_year")
    result = fit(frame, "y_tail1_year", "zcum03")
    post = pd.read_csv(POST_RESULTS)
    match = post[
        post["rung"].eq("county")
        & post["flow"].eq("award")
        & post["window_mode"].eq("nonoverlap")
        & post["treatment"].eq("net_nofloor")
        & post["horizon"].eq(3)
        & post["share_label"].eq("fixed10")
        & post["winsor_rule"].eq("tail1")
        & post["winsor_group"].eq("year")
    ]
    if len(match) != 1:
        raise AssertionError(f"post-period validation matched {len(match)} rows")
    expected = match.iloc[0]
    deviations = {
        key: abs(result[key] - float(expected[key]))
        for key in ("fs_coef", "fs_se", "fs_F")
    }
    if max(deviations.values()) >= 1e-6:
        raise AssertionError(f"post-period estimator validation failed: {deviations}")
    print(
        "Estimator validation: post h=3 fixed10 tail1-year "
        f"F={result['fs_F']:.9f}, max deviation={max(deviations.values()):.2e}"
    )


def build_base(builder):
    if not CONTRACT_CACHE.exists():
        raise FileNotFoundError(f"certified contract cache missing: {CONTRACT_CACHE}")
    panel = pd.read_parquet(
        dc.PANEL_OUT_DIR / "county_panel.parquet",
        columns=["ccg", "year", "emp", "income_real"],
    )
    positive = (
        panel[
            panel["year"].between(
                dc.POSITIVE_EMP_FIRST_YEAR, dc.POSITIVE_EMP_LAST_YEAR
            )
        ]
        .groupby("ccg", sort=False)["emp"]
        .min()
    )
    universe = set(positive[positive > 0].index.astype(int))
    if len(universe) != 3_094:
        raise AssertionError(f"positive-employment universe is {len(universe)}, expected 3094")

    contracts = pd.read_parquet(
        CONTRACT_CACHE, columns=["ccg", "year", "dollars_real"]
    )
    contracts = contracts[contracts["ccg"].isin(universe)].copy()
    contracts = contracts.rename(columns={"dollars_real": "level"})
    if contracts.duplicated(["ccg", "year"]).any():
        raise AssertionError("contract cache is not unique by county group and year")

    cross, eligible_states = builder.geography(universe)
    county_cross = cross[cross["actual_state"].isin(eligible_states)][
        ["ccg", "actual_state"]
    ].drop_duplicates()
    years = pd.DataFrame({"year": np.arange(1966, ANALYSIS_LAST + 1)})
    levels = (
        county_cross.merge(years, how="cross")
        .merge(contracts, on=["ccg", "year"], how="left", validate="one_to_one")
        .rename(
            columns={
                "ccg": "unit",
                "actual_state": "state",
                "level": "contract_level",
            }
        )
        .sort_values(["unit", "year"])
        .reset_index(drop=True)
    )
    levels["contract_level"] = levels["contract_level"].fillna(0.0)
    income = builder.make_income_spine(
        panel[["ccg", "year", "income_real"]],
        "county",
        cross,
        eligible_states,
    )
    national = (
        contracts.groupby("year", sort=False)["level"]
        .sum()
        .reindex(range(1966, ANALYSIS_LAST + 1), fill_value=0.0)
    )
    print(
        f"Analysis support: {levels['unit'].nunique():,} counties in "
        f"{levels['state'].nunique()} states"
    )
    return levels, income, national


def allocation_shares(levels: pd.DataFrame, national: pd.Series):
    pivot = levels.pivot(index="unit", columns="year", values="contract_level").fillna(0.0)
    shares = {}
    for label, first, last in FIXED_SHARES:
        denominator = float(national.loc[first:last].sum())
        if denominator == 0:
            raise AssertionError(f"zero national denominator for {label}")
        shares[label] = pivot.loc[:, first:last].sum(axis=1) / denominator
    return shares


def horizon_frame(
    levels: pd.DataFrame,
    income: pd.DataFrame,
    national: pd.Series,
    shares: dict[str, pd.Series],
    horizon: int,
):
    frame = levels.copy()
    frame["future"] = (
        frame.groupby("unit", sort=False)["contract_level"]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .reset_index(level=0, drop=True)
    )
    frame["future"] = frame.groupby("unit", sort=False)["future"].shift(-horizon)
    frame = frame[frame["year"].between(ANALYSIS_FIRST, ANALYSIS_LAST - horizon)].copy()
    frame = frame[(frame["year"] - ANALYSIS_FIRST) % horizon == 0].copy()
    starts = sorted(frame["year"].unique().tolist())
    if len(starts) != EXPECTED_WINDOWS[horizon]:
        raise AssertionError(f"h={horizon} has {len(starts)} starts, expected {EXPECTED_WINDOWS[horizon]}")

    frame = frame.merge(
        income[["unit", "year", "income_real"]],
        on=["unit", "year"],
        how="left",
        validate="one_to_one",
    )
    national_future = national.rolling(horizon, min_periods=horizon).sum()
    national_future.index = national_future.index - horizon
    frame["loo_future"] = frame["year"].map(national_future) - frame["future"]
    if frame[["future", "income_real", "loo_future"]].isna().any().any():
        raise AssertionError(f"h={horizon} has missing shock components")
    if not (frame["income_real"] > 0).all():
        raise AssertionError(f"h={horizon} has nonpositive personal income")

    frame["state_year"] = frame["state"] * 10_000 + frame["year"]
    frame["y_cum"] = frame["future"] / frame["income_real"]
    for label, share in shares.items():
        frame[label] = frame["unit"].map(share)
        frame[f"z_{label}"] = (
            frame[label] * frame["loo_future"] / frame["income_real"]
        )
    winsor_tail_by_year(frame, "y_cum", "y_tail1_year")
    expected_rows = frame["unit"].nunique() * EXPECTED_WINDOWS[horizon]
    if len(frame) != expected_rows:
        raise AssertionError(f"h={horizon} has {len(frame)} rows, expected {expected_rows}")
    print(f"h={horizon}: starts={starts}, rows={len(frame):,}")
    return frame, starts


def verify_cells(
    frame: pd.DataFrame,
    levels: pd.DataFrame,
    national: pd.Series,
    shares: dict[str, pd.Series],
    horizon: int,
):
    level_lookup = levels.set_index(["unit", "year"])["contract_level"]
    sample = frame.sample(n=min(200, len(frame)), random_state=10_000 + horizon)
    deviations = []
    for row in sample.itertuples(index=False):
        years = range(row.year + 1, row.year + horizon + 1)
        future = sum(float(level_lookup.loc[(row.unit, year)]) for year in years)
        national_future = sum(float(national.loc[year]) for year in years)
        loo_future = national_future - future
        deviations.extend(
            [
                abs(row.future - future),
                abs(row.loo_future - loo_future),
                abs(row.y_cum - future / row.income_real),
            ]
        )
        for label, share in shares.items():
            expected = float(share.loc[row.unit]) * loo_future / row.income_real
            deviations.append(abs(getattr(row, f"z_{label}") - expected))
    maximum = max(deviations, default=np.inf)
    scale = max(float(frame["future"].abs().max()), 1.0)
    if maximum / scale >= 1e-12:
        raise AssertionError(
            f"h={horizon} cell verification failed: scaled deviation {maximum / scale:.2e}"
        )
    print(
        f"h={horizon}: verified {len(sample)} raw cells, "
        f"scaled max deviation={maximum / scale:.2e}"
    )


def main():
    np.random.seed(0)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        validate_estimator()
        builder = load_numbered(POST_BUILDER)
        levels, income, national = build_base(builder)
        shares = allocation_shares(levels, national)
        rows = []
        for horizon in HORIZONS:
            frame, starts = horizon_frame(levels, income, national, shares, horizon)
            verify_cells(frame, levels, national, shares, horizon)
            for outcome, rule, group in (
                ("y_cum", "none", "none"),
                ("y_tail1_year", "tail1", "year"),
            ):
                for label, first, last in FIXED_SHARES:
                    result = fit(frame, outcome, f"z_{label}")
                    rows.append(
                        {
                            "rung": "county",
                            "flow": "award",
                            "window_mode": "nonoverlap",
                            "treatment": "net_nofloor",
                            "horizon": horizon,
                            "share_label": label,
                            "share_first": first,
                            "share_last": last,
                            "winsor_rule": rule,
                            "winsor_group": group,
                            "start_years": " ".join(map(str, starts)),
                            **result,
                        }
                    )
        results = pd.DataFrame(rows)
        if len(results) != 18 or not np.isfinite(
            results[["fs_coef", "fs_se", "fs_F"]].to_numpy()
        ).all():
            raise AssertionError("pre-period result grid is incomplete")
        results.to_csv(OUT, index=False)
        print(f"18 regressions written to {OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
