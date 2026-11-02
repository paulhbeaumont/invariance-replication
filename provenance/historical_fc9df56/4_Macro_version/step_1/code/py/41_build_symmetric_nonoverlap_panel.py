#!/usr/bin/env python3
"""Build symmetric pre- and post-1998 county first-stage panels.

The input is the raw-derived FY1966-2018 county contract panel produced by 33.
Both periods use signed award-year obligations without a nominal dollar floor.
Fixed allocation shares cover the 4, 5, or 10 fiscal years immediately before
the first start year in each period. At horizon h, the local variables divide
county contracts from t+1 through t+h by county income or employment in t. The
instruments apply the county share to national contracts and divide by national
income or employment, with all national quantities summed over the analysis
counties.

Outputs:
    _data/temp/symmetric_nonoverlap_first_stage/panel.parquet
    _data/temp/symmetric_nonoverlap_first_stage/<period>_h<h>.dta
    batch_logs/macro_version/symmetric_nonoverlap_design.json
    batch_logs/macro_version/symmetric_nonoverlap_share_audit.csv
    batch_logs/macro_version/symmetric_nonoverlap_panel_audit.csv
    batch_logs/macro_version/41_build_symmetric_nonoverlap_panel.log
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
POST_BUILDER = CODE / "35_build_post_first_stage_full_grid.py"
CONTRACTS = dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet"
COUNTY_PANEL = dc.PANEL_OUT_DIR / "county_panel.parquet"
TEMP = dc.SYMMETRIC_FIRST_STAGE_DIR
LOG = dc.LOG_DIR / "41_build_symmetric_nonoverlap_panel.log"
DESIGN_OUT = dc.LOG_DIR / "symmetric_nonoverlap_design.json"
SHARE_AUDIT_OUT = dc.LOG_DIR / "symmetric_nonoverlap_share_audit.csv"
PANEL_AUDIT_OUT = dc.LOG_DIR / "symmetric_nonoverlap_panel_audit.csv"

PERIODS = {
    "pre": {"first": 1979, "last": 1997},
    "post": {"first": 1998, "last": 2018},
}
SHARE_LENGTHS = (4, 5, 10)
HORIZONS = (2, 3, 4)
SOURCE_FIRST = 1966
SOURCE_LAST = 2018
EXPECTED_SOURCE_UNITS = 3_101
EXPECTED_EMPLOYMENT_UNIVERSE = 3_094
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
    spec = importlib.util.spec_from_file_location("post_grid_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive_employment_universe(panel: pd.DataFrame):
    profile = (
        panel[
            panel["year"].between(
                dc.POSITIVE_EMP_FIRST_YEAR, dc.POSITIVE_EMP_LAST_YEAR
            )
        ]
        .groupby("ccg", sort=False)["emp"]
        .min()
    )
    universe = set(profile[profile > 0].index.astype(int))
    if len(universe) != EXPECTED_EMPLOYMENT_UNIVERSE:
        raise AssertionError(
            f"positive-employment universe is {len(universe)}, "
            f"expected {EXPECTED_EMPLOYMENT_UNIVERSE}"
        )
    return universe


def load_inputs():
    for path in (CONTRACTS, COUNTY_PANEL):
        if not path.exists():
            raise FileNotFoundError(f"required raw-derived input is missing: {path}")
    contracts = pd.read_parquet(
        CONTRACTS,
        columns=[
            "ccg",
            "year",
            "source",
            "dollars_nominal",
            "dollars_real",
            "n_actions",
        ],
    )
    panel = pd.read_parquet(
        COUNTY_PANEL, columns=["ccg", "year", "emp", "income_real"]
    )
    expected_rows = EXPECTED_SOURCE_UNITS * (SOURCE_LAST - SOURCE_FIRST + 1)
    if len(contracts) != expected_rows:
        raise AssertionError(
            f"contract cache has {len(contracts)} rows, expected {expected_rows}"
        )
    if contracts.duplicated(["ccg", "year"]).any():
        raise AssertionError("contract cache is not unique by county group and year")
    if sorted(contracts["year"].unique()) != list(
        range(SOURCE_FIRST, SOURCE_LAST + 1)
    ):
        raise AssertionError("contract cache does not cover FY1966-FY2018")
    return contracts, panel


def geography(universe: set[int]):
    builder = load_numbered(POST_BUILDER)
    cross, eligible_states = builder.geography(universe)
    county_cross = (
        cross[cross["actual_state"].isin(eligible_states)][
            ["ccg", "actual_state"]
        ]
        .drop_duplicates()
        .rename(columns={"actual_state": "state"})
    )
    if len(county_cross) != EXPECTED_ANALYSIS_UNITS:
        raise AssertionError(
            f"analysis geography has {len(county_cross)} counties, "
            f"expected {EXPECTED_ANALYSIS_UNITS}"
        )
    if county_cross["state"].nunique() != EXPECTED_STATES:
        raise AssertionError(
            f"analysis geography has {county_cross['state'].nunique()} states, "
            f"expected {EXPECTED_STATES}"
        )
    return county_cross


def share_windows():
    windows = {}
    for period, bounds in PERIODS.items():
        end = bounds["first"] - 1
        windows[period] = {
            f"fixed{length}": (end - length + 1, end)
            for length in SHARE_LENGTHS
        }
    return windows


def build_shares(
    contracts: pd.DataFrame,
    analysis_units: set[int],
    windows: dict,
):
    levels = contracts[contracts["ccg"].isin(analysis_units)].pivot(
        index="ccg", columns="year", values="dollars_real"
    )
    shares = {}
    audit_rows = []
    for period, period_windows in windows.items():
        shares[period] = {}
        for label, (first, last) in period_windows.items():
            numerator = levels.loc[:, first:last].sum(axis=1)
            denominator = float(numerator.sum())
            if denominator == 0:
                raise AssertionError(f"zero national contract total for {period} {label}")
            values = numerator / denominator
            shares[period][label] = values
            audit_rows.append(
                {
                    "period": period,
                    "share_label": label,
                    "share_first": first,
                    "share_last": last,
                    "share_years": last - first + 1,
                    "period_first": PERIODS[period]["first"],
                    "years_between_share_and_period": PERIODS[period]["first"]
                    - last
                    - 1,
                    "national_real_contracts": denominator,
                    "sum_analysis_shares": float(values.sum()),
                    "zero_share_count_analysis": int((values == 0).sum()),
                }
            )
    return shares, pd.DataFrame(audit_rows)


def period_starts(period: str, horizon: int):
    bounds = PERIODS[period]
    return list(range(bounds["first"], bounds["last"] - horizon + 1, horizon))


def build_period_horizon(
    period: str,
    horizon: int,
    levels: pd.DataFrame,
    income: pd.DataFrame,
    employment: pd.DataFrame,
    state_map: pd.Series,
    national_contracts: pd.Series,
    national_income: pd.Series,
    national_employment: pd.Series,
    shares: dict[str, pd.Series],
):
    starts = period_starts(period, horizon)
    pieces = []
    for start in starts:
        future_years = list(range(start + 1, start + horizon + 1))
        future = levels.loc[:, future_years].sum(axis=1)
        national_future = float(national_contracts.loc[future_years].sum())
        piece = pd.DataFrame(
            {
                "unit": future.index.astype(np.int32),
                "state": future.index.map(state_map).astype(np.int32),
                "year": np.int32(start),
                "future": future.to_numpy(),
                "income_real": income.loc[future.index, start].to_numpy(),
                "employment": employment.loc[future.index, start].to_numpy(),
                "national_future": national_future,
                "national_income": float(national_income.loc[start]),
                "national_employment": float(national_employment.loc[start]),
            }
        )
        for label, values in shares.items():
            piece[label] = piece["unit"].map(values)
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    required = [
        "future",
        "income_real",
        "employment",
        "national_future",
        "national_income",
        "national_employment",
        *shares,
    ]
    if not np.isfinite(frame[required].to_numpy()).all():
        raise AssertionError(f"{period} h={horizon} contains nonfinite inputs")
    positive = [
        "income_real",
        "employment",
        "national_income",
        "national_employment",
    ]
    if not (frame[positive] > 0).all().all():
        raise AssertionError(f"{period} h={horizon} contains nonpositive scaling inputs")
    frame["state_year"] = (
        frame["state"].astype(np.int32) * 10_000 + frame["year"].astype(np.int32)
    )
    frame["y_income"] = frame["future"] / frame["income_real"]
    frame["y_employment"] = frame["future"] / frame["employment"]
    for label in shares:
        frame[f"z_income_{label}"] = (
            frame[label] * frame["national_future"] / frame["national_income"]
        )
        frame[f"z_employment_{label}"] = (
            frame[label]
            * frame["national_future"]
            / frame["national_employment"]
        )
    frame.insert(0, "period", period)
    frame.insert(1, "horizon", np.int8(horizon))
    frame = frame.sort_values(["unit", "year"]).reset_index(drop=True)
    expected = EXPECTED_ANALYSIS_UNITS * len(starts)
    if len(frame) != expected:
        raise AssertionError(
            f"{period} h={horizon} has {len(frame)} rows, expected {expected}"
        )
    return frame, starts


def design_contract(windows: dict):
    return {
        "contract": "symmetric_nonoverlap_first_stage_v2",
        "source": {
            "contracts": str(CONTRACTS),
            "county_panel": str(COUNTY_PANEL),
            "contract_flow": "signed award-year obligations",
            "dollar_treatment": "no nominal floor",
            "price_basis": "real 2006 dollars",
        },
        "sample": {
            "periods": PERIODS,
            "horizons": list(HORIZONS),
            "window_mode": "period-anchored nonoverlapping start years",
            "positive_employment_years": [
                dc.POSITIVE_EMP_FIRST_YEAR,
                dc.POSITIVE_EMP_LAST_YEAR,
            ],
            "state_rule": "retain states with at least two assigned commuting zones",
            "expected_counties": EXPECTED_ANALYSIS_UNITS,
            "expected_states": EXPECTED_STATES,
        },
        "shares": windows,
        "national_universe": "the 3,080 analysis counties for contracts, denominators, and fixed-share normalization",
        "scalings": {
            "income": {
                "local_lhs": "county contracts from t+1 through t+h divided by county real personal income in t",
                "instrument": "county fixed share times national contracts from t+1 through t+h divided by national real personal income in t",
            },
            "employment": {
                "local_lhs": "county contracts from t+1 through t+h divided by county employment in t",
                "instrument": "county fixed share times national contracts from t+1 through t+h divided by national employment in t",
            },
        },
        "winsorization": {
            "variables": "LHS and RHS separately with the same method and grouping",
            "rules": ["none", "IQR-4", "IQR-5", "1st and 99th empirical percentiles"],
            "groups": ["pooled within period-horizon-scaling", "start year within period"],
            "command": "winsor4",
        },
        "regression": {
            "fixed_effects": "county and state-by-year",
            "cluster": "county",
            "weights": "none",
        },
    }


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        print("Symmetric nonoverlapping first-stage panel")
        contracts, panel = load_inputs()
        universe = positive_employment_universe(panel)
        county_cross = geography(universe)
        analysis_units = set(county_cross["ccg"].astype(int))
        windows = share_windows()
        shares, share_audit = build_shares(contracts, analysis_units, windows)

        analysis_contracts = contracts[
            contracts["ccg"].isin(analysis_units)
        ].pivot(index="ccg", columns="year", values="dollars_real")
        analysis_income = panel[panel["ccg"].isin(analysis_units)].pivot(
            index="ccg", columns="year", values="income_real"
        )
        analysis_employment = panel[panel["ccg"].isin(analysis_units)].pivot(
            index="ccg", columns="year", values="emp"
        )
        state_map = county_cross.set_index("ccg")["state"]
        national_contracts = analysis_contracts.sum(axis=0)
        national_income = analysis_income.sum(axis=0)
        national_employment = analysis_employment.sum(axis=0)

        frames = []
        audit_rows = []
        for period in PERIODS:
            for horizon in HORIZONS:
                frame, starts = build_period_horizon(
                    period,
                    horizon,
                    analysis_contracts,
                    analysis_income,
                    analysis_employment,
                    state_map,
                    national_contracts,
                    national_income,
                    national_employment,
                    shares[period],
                )
                frames.append(frame)
                dta_path = TEMP / f"{period}_h{horizon}.dta"
                frame.drop(columns=["period"]).to_stata(
                    dta_path, write_index=False, version=118
                )
                audit_rows.append(
                    {
                        "period": period,
                        "period_first": PERIODS[period]["first"],
                        "period_last": PERIODS[period]["last"],
                        "horizon": horizon,
                        "start_years": " ".join(map(str, starts)),
                        "n_windows": len(starts),
                        "n_obs": len(frame),
                        "n_units": frame["unit"].nunique(),
                        "n_states": frame["state"].nunique(),
                        "first_future_year": min(starts) + 1,
                        "last_future_year": max(starts) + horizon,
                        "input_path": str(dta_path),
                    }
                )
                print(
                    f"{period:<4s} h={horizon}: {len(starts)} windows, "
                    f"{len(frame):,} rows, starts={starts}"
                )

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(TEMP / "panel.parquet", index=False)
        share_audit.to_csv(SHARE_AUDIT_OUT, index=False)
        pd.DataFrame(audit_rows).to_csv(PANEL_AUDIT_OUT, index=False)
        DESIGN_OUT.write_text(
            json.dumps(design_contract(windows), indent=2) + "\n", encoding="utf-8"
        )
        print(f"{len(combined):,} rows -> {TEMP / 'panel.parquet'}")
        print(f"6 share checks -> {SHARE_AUDIT_OUT}")
        print(f"6 panel checks -> {PANEL_AUDIT_OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
