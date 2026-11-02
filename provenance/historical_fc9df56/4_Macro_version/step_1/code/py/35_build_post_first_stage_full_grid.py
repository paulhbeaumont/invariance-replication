#!/usr/bin/env python3
"""Build grid-ready county and commuting-zone contract-growth panels.

The producer reads harmonized DCADS and USAspending transaction parquets,
constructs award-year and duration-spread real contract flows, applies the four
declared obligation treatments at the transaction level, and writes grid inputs
for overlapping and 1998-anchored non-overlapping cumulative shocks. At horizon
h, the local shock is contract dollars in t+1 through t+h divided by real
personal income in t. Allocation shares are based on award-year flows. National
shifters use the same cumulative operator and leave out the local unit.

Outputs are temporary Stata inputs under _data/temp/post_first_stage_full_grid.
Auditable date coverage and share dictionaries are written to batch_logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


np.random.seed(0)

CODE = Path(__file__).resolve().parent
TEMP = dc.POST_GRID_DIR
LOG = dc.LOG_DIR / "35_build_post_first_stage_full_grid.log"
DATE_AUDIT = dc.LOG_DIR / "post_first_stage_duration_date_audit.csv"
INPUT_AUDIT = dc.LOG_DIR / "post_first_stage_input_audit.csv"
SHARE_DICTIONARY = dc.LOG_DIR / "post_first_stage_share_dictionary.csv"
MANIFEST = TEMP / "input_manifest.csv"

RAW_FIRST = 1984
RAW_LAST = 2018
ANALYSIS_FIRST = 1998
ANALYSIS_LAST = 2018
HORIZONS = (2, 3, 4)
FLOWS = ("award", "duration")
WINDOW_MODES = ("overlapping", "nonoverlap")
TREATMENTS = (
    ("net_nofloor", False, 0),
    ("net_floor25000", False, 25_000),
    ("positive_nofloor", True, 0),
    ("positive_floor25000", True, 25_000),
)
FIXED_SHARES = (("fixed4", 1994, 1997), ("fixed5", 1993, 1997), ("fixed10", 1988, 1997))
ROLL_LAGS = (3, 4, 5)
ROLL_LENGTHS = (4, 5, 10)


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def parse_contract_date(series: pd.Series, source_year: int) -> pd.Series:
    """Parse ISO, YYYYMMDD, and YYMMDD contract dates."""
    text = (
        series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    six = text.str.fullmatch(r"\d{6}")
    text.loc[six] = "19" + text.loc[six]
    eight = text.str.fullmatch(r"\d{8}") & text.str[:4].between("1950", "2050")
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}") & text.str[:4].between(
        "1950", "2050"
    )
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    parsed.loc[eight] = pd.to_datetime(
        text.loc[eight], format="%Y%m%d", errors="coerce"
    )
    parsed.loc[iso] = pd.to_datetime(text.loc[iso], format="%Y-%m-%d", errors="coerce")
    unreasonable = parsed.dt.year.notna() & ~parsed.dt.year.between(1950, 2050)
    parsed.loc[unreasonable] = pd.NaT
    return parsed


def fiscal_year(date: pd.Series) -> pd.Series:
    return date.dt.year + (date.dt.month >= 10).astype("Int64")


def load_actions(year: int, fips_to_ccg: dict[str, int], universe: set[int]):
    """Read one annual source and return mapped transactions in nominal dollars."""
    if year < dc.SEAM_YEAR:
        path = dc.OUT_DIR / f"fy{year}.parquet"
        schema = set(pq.read_schema(path).names)
        columns = [
            "dollars",
            "dollars_unit",
            "action_date",
            "geo_class",
            "county_fips5",
        ]
        if "completion_date" in schema:
            columns.append("completion_date")
        d = pd.read_parquet(path, columns=columns)
        d = d[d["geo_class"].eq("county")].copy()
        d["ccg"] = d["county_fips5"].map(fips_to_ccg)
        nominal = d["dollars"].to_numpy(dtype=np.float64)
        nominal = np.where(
            d["dollars_unit"].eq("thousands").to_numpy(), nominal * 1000.0, nominal
        )
        completion = d.get("completion_date", pd.Series("", index=d.index))
        source = "DCADS"
        completion_field = (
            "completion_date" if "completion_date" in schema else "absent"
        )
    else:
        path = dc.USASPENDING_OUT_DIR / f"fy{year}.parquet"
        columns = [
            "federal_action_obligation",
            "action_date",
            "period_of_performance_current_end_date",
            "geo_class",
            "ccg",
        ]
        d = pd.read_parquet(path, columns=columns)
        d = d[d["geo_class"].eq("county")].copy()
        nominal = (
            pd.to_numeric(d["federal_action_obligation"], errors="coerce")
            .fillna(0)
            .to_numpy()
        )
        completion = d["period_of_performance_current_end_date"]
        source = "USAspending"
        completion_field = "period_of_performance_current_end_date"
    d["ccg"] = pd.to_numeric(d["ccg"], errors="coerce")
    keep = d["ccg"].isin(universe).to_numpy() & np.isfinite(nominal)
    d = d.loc[keep].reset_index(drop=True)
    nominal = nominal[keep]
    completion = completion.loc[keep].reset_index(drop=True)
    return d, nominal, completion, source, completion_field


def aggregate_actions(universe: set[int]):
    """Build action-year and duration-spread county-year cells in one raw pass."""
    fips_to_ccg, _, _ = dc.load_ccg_map()
    award_parts = []
    duration_parts = []
    audit = []
    for year in range(RAW_FIRST, RAW_LAST + 1):
        d, nominal, completion_raw, source, completion_field = load_actions(
            year, fips_to_ccg, universe
        )
        real = nominal * dc.deflate(year)
        values = {}
        for name, positive, floor in TREATMENTS:
            keep = nominal > 0 if positive else np.ones(len(nominal), dtype=bool)
            if floor:
                keep &= nominal >= floor if positive else np.abs(nominal) >= floor
            values[name] = np.where(keep, real, 0.0)
        value_frame = pd.DataFrame(values)
        value_frame["ccg"] = d["ccg"].astype(np.int32).to_numpy()
        award = value_frame.groupby("ccg", sort=False)[list(values)].sum().reset_index()
        award["year"] = year
        award_parts.append(award)

        start = parse_contract_date(d["action_date"], year)
        end = parse_contract_date(completion_raw, year)
        missing_start = start.isna()
        missing_end = end.isna()
        invalid_order = start.notna() & end.notna() & (end < start)
        valid = ~(missing_start | missing_end | invalid_order)
        start_fy = fiscal_year(start)
        source_fy_mismatch = valid & start_fy.notna() & (start_fy != year)
        duration_days = (end - start).dt.days + 1

        for name, _, _ in TREATMENTS:
            active = values[name] != 0
            audit.append(
                {
                    "source": source,
                    "source_year": year,
                    "completion_field": completion_field,
                    "treatment": name,
                    "actions_retained": int(active.sum()),
                    "real_dollars_retained": float(values[name].sum()),
                    "missing_action_date_actions": int((active & missing_start).sum()),
                    "missing_completion_date_actions": int(
                        (active & missing_end).sum()
                    ),
                    "invalid_completion_before_action_actions": int(
                        (active & invalid_order).sum()
                    ),
                    "action_date_source_fy_mismatch_actions": int(
                        (active & source_fy_mismatch).sum()
                    ),
                    "valid_duration_actions": int((active & valid).sum()),
                    "valid_duration_real_dollars": float(
                        values[name][active & valid].sum()
                    ),
                }
            )

        if valid.any():
            v = pd.DataFrame(
                {
                    "ccg": d.loc[valid, "ccg"].astype(np.int32).to_numpy(),
                    "start": start.loc[valid].to_numpy(),
                    "end": end.loc[valid].to_numpy(),
                    "days": duration_days.loc[valid].to_numpy(dtype=np.float64),
                }
            )
            for name in values:
                v[name] = values[name][valid]
            max_fy = min(ANALYSIS_LAST, int(fiscal_year(end.loc[valid]).max()))
            min_fy = max(1988, int(fiscal_year(start.loc[valid]).min()))
            for target in range(min_fy, max_fy + 1):
                fy_start = np.datetime64(f"{target - 1}-10-01")
                fy_end = np.datetime64(f"{target}-09-30")
                overlap_start = np.maximum(
                    v["start"].to_numpy(dtype="datetime64[ns]"), fy_start
                )
                overlap_end = np.minimum(
                    v["end"].to_numpy(dtype="datetime64[ns]"), fy_end
                )
                overlap = (overlap_end - overlap_start).astype("timedelta64[D]").astype(
                    np.int64
                ) + 1
                use = overlap > 0
                if not use.any():
                    continue
                part = pd.DataFrame(
                    {"ccg": v.loc[use, "ccg"].to_numpy(), "year": target}
                )
                fraction = overlap[use] / v.loc[use, "days"].to_numpy()
                for name in values:
                    part[name] = v.loc[use, name].to_numpy() * fraction
                part = (
                    part.groupby(["ccg", "year"], sort=False)[list(values)]
                    .sum()
                    .reset_index()
                )
                duration_parts.append(part)
        print(f"  {year}: {source:<11s} {len(d):>9,d} mapped actions")

    award = (
        pd.concat(award_parts, ignore_index=True)
        .groupby(["ccg", "year"], sort=False)[[x[0] for x in TREATMENTS]]
        .sum()
        .reset_index()
    )
    duration = (
        pd.concat(duration_parts, ignore_index=True)
        .groupby(["ccg", "year"], sort=False)[[x[0] for x in TREATMENTS]]
        .sum()
        .reset_index()
    )
    return award, duration, pd.DataFrame(audit)


def geography(universe: set[int]):
    """Return county-to-CZ mappings and eligible state assignments."""
    raw = pd.read_csv(dc.CCG_CROSSWALK, dtype={"ccg_id": str, "ccz_id": str})
    raw = raw[(raw["ccg_id"] != "unplaceable") & raw["ccz_id"].notna()].copy()
    raw["ccg"] = raw["ccg_id"].astype(np.int32)
    raw = raw[raw["ccg"].isin(universe)]
    raw["cz"] = raw["ccz_id"].astype(np.int32)
    raw["actual_state"] = raw["ccg"] // 1000
    cross = raw[["ccg", "cz", "actual_state"]].drop_duplicates()
    if cross["ccg"].duplicated().any():
        raise AssertionError("A retained county group maps to multiple CZs")
    assignment = pd.read_csv(
        dc.REFERENCE_DIR / "cz_state_1969_population.csv",
        dtype={"cz": str, "assigned_state_fips2": str},
    )
    assignment["cz"] = assignment["cz"].astype(np.int32)
    assignment["assigned_state"] = assignment["assigned_state_fips2"].astype(np.int32)
    cross = cross.merge(
        assignment[["cz", "assigned_state"]],
        on="cz",
        how="left",
        validate="many_to_one",
    )
    counts = (
        cross[["cz", "assigned_state"]]
        .drop_duplicates()
        .groupby("assigned_state", sort=False)["cz"]
        .nunique()
    )
    eligible = set(counts[counts >= 2].index.astype(int))
    return cross, eligible


def make_spine(
    levels: pd.DataFrame,
    treatment: str,
    rung: str,
    cross: pd.DataFrame,
    eligible_states: set[int],
):
    years = pd.DataFrame({"year": np.arange(RAW_FIRST, RAW_LAST + 1, dtype=np.int32)})
    units = pd.DataFrame({"ccg": sorted(cross["ccg"].unique())})
    spine = units.merge(years, how="cross").merge(
        levels[["ccg", "year", treatment]], on=["ccg", "year"], how="left"
    )
    spine[treatment] = spine[treatment].fillna(0.0)
    spine = spine.merge(cross, on="ccg", how="left", validate="many_to_one")
    if rung == "county":
        spine = spine[spine["actual_state"].isin(eligible_states)].copy()
        spine["unit"] = spine["ccg"]
        spine["state"] = spine["actual_state"]
        out = spine[["unit", "state", "year", treatment]].rename(
            columns={treatment: "level"}
        )
    else:
        out = (
            spine.groupby(["cz", "assigned_state", "year"], sort=False)[treatment]
            .sum()
            .reset_index()
        )
        out = out[out["assigned_state"].isin(eligible_states)].copy()
        out = out.rename(
            columns={"cz": "unit", "assigned_state": "state", treatment: "level"}
        )
    out["unit"] = out["unit"].astype(np.int32)
    out["state"] = out["state"].astype(np.int32)
    out["year"] = out["year"].astype(np.int32)
    return out.sort_values(["unit", "year"]).reset_index(drop=True)


def make_income_spine(
    income_levels: pd.DataFrame,
    rung: str,
    cross: pd.DataFrame,
    eligible_states: set[int],
):
    """Map real county personal income to the requested regression unit."""
    spine = income_levels.merge(cross, on="ccg", how="inner", validate="many_to_one")
    if rung == "county":
        spine = spine[spine["actual_state"].isin(eligible_states)].copy()
        spine["unit"] = spine["ccg"]
        spine["state"] = spine["actual_state"]
        out = spine[["unit", "state", "year", "income_real"]].copy()
    else:
        out = (
            spine.groupby(["cz", "assigned_state", "year"], sort=False)["income_real"]
            .sum()
            .reset_index()
        )
        out = out[out["assigned_state"].isin(eligible_states)].copy()
        out = out.rename(columns={"cz": "unit", "assigned_state": "state"})
    out["unit"] = out["unit"].astype(np.int32)
    out["state"] = out["state"].astype(np.int32)
    out["year"] = out["year"].astype(np.int32)
    return out.sort_values(["unit", "year"]).reset_index(drop=True)


def shares(award: pd.DataFrame, national_award: pd.Series):
    """Return twelve allocation-share arrays keyed by their short Stata names."""
    pivot = award.pivot(index="unit", columns="year", values="level").fillna(0.0)
    rows = []
    dictionary = []
    for idx, (label, first, last) in enumerate(FIXED_SHARES, start=1):
        numerator = pivot.loc[:, first:last].sum(axis=1)
        denominator = float(national_award.loc[first:last].sum())
        value = numerator / denominator
        short = f"s{idx:02d}"
        rows.append(
            pd.DataFrame(
                {
                    "unit": pivot.index,
                    "year": ANALYSIS_FIRST,
                    "share": value.to_numpy(),
                    "share_code": short,
                }
            )
        )
        dictionary.append(
            {
                "share_code": short,
                "share_type": "fixed",
                "share_label": label,
                "window_years": last - first + 1,
                "end_lag": np.nan,
                "fixed_first": first,
                "fixed_last": last,
            }
        )
    idx = len(FIXED_SHARES)
    for lag in ROLL_LAGS:
        for length in ROLL_LENGTHS:
            idx += 1
            short = f"s{idx:02d}"
            pieces = []
            for year in range(ANALYSIS_FIRST, ANALYSIS_LAST + 1):
                last = year - lag
                first = last - length + 1
                numerator = pivot.loc[:, first:last].sum(axis=1)
                denominator = float(national_award.loc[first:last].sum())
                pieces.append(
                    pd.DataFrame(
                        {
                            "unit": pivot.index,
                            "year": year,
                            "share": (numerator / denominator).to_numpy(),
                            "share_code": short,
                        }
                    )
                )
            rows.append(pd.concat(pieces, ignore_index=True))
            dictionary.append(
                {
                    "share_code": short,
                    "share_type": "rolling",
                    "share_label": f"rolling_lag{lag}_window{length}",
                    "window_years": length,
                    "end_lag": lag,
                    "fixed_first": np.nan,
                    "fixed_last": np.nan,
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(dictionary)


def build_input(
    flow_levels: pd.DataFrame,
    award_levels: pd.DataFrame,
    income_levels: pd.DataFrame,
    rung: str,
    treatment: str,
    horizon: int,
    window_mode: str,
    cross,
    eligible_states,
):
    flow = make_spine(flow_levels, treatment, rung, cross, eligible_states)
    award = make_spine(award_levels, treatment, rung, cross, eligible_states)
    income = make_income_spine(income_levels, rung, cross, eligible_states)
    national_award = (
        award_levels.groupby("year", sort=False)[treatment]
        .sum()
        .reindex(range(RAW_FIRST, RAW_LAST + 1), fill_value=0.0)
    )
    share_long, share_dict = shares(award, national_award)
    fixed_codes = {"s01", "s02", "s03"}
    fixed = share_long[share_long["share_code"].isin(fixed_codes)].pivot(
        index="unit", columns="share_code", values="share"
    )
    rolling = share_long[~share_long["share_code"].isin(fixed_codes)].pivot(
        index=["unit", "year"], columns="share_code", values="share"
    )

    d = flow.copy()
    d["future"] = (
        d.groupby("unit", sort=False)["level"]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .reset_index(level=0, drop=True)
    )
    d["future"] = d.groupby("unit", sort=False)["future"].shift(-horizon)
    d = d[d["year"].between(ANALYSIS_FIRST, ANALYSIS_LAST - horizon)].copy()
    if window_mode == "nonoverlap":
        d = d[(d["year"] - ANALYSIS_FIRST) % horizon == 0].copy()
    elif window_mode != "overlapping":
        raise ValueError(window_mode)
    full_analysis_rows = len(d)
    d = d.merge(
        income[["unit", "year", "income_real"]],
        on=["unit", "year"],
        how="left",
        validate="one_to_one",
    )
    national_level = (
        flow_levels.groupby("year", sort=False)[treatment]
        .sum()
        .reindex(range(RAW_FIRST, RAW_LAST + 1), fill_value=0.0)
    )
    national_future = national_level.rolling(horizon, min_periods=horizon).sum()
    national_future.index = national_future.index - horizon
    d["loo_future"] = d["year"].map(national_future) - d["future"]
    required_cells = ["future", "income_real", "loo_future"]
    common = np.isfinite(d[required_cells]).all(axis=1)
    common &= d["income_real"] > 0
    d = d[common].copy()
    d = d.merge(fixed.reset_index(), on="unit", how="left", validate="many_to_one")
    d = d.merge(
        rolling.reset_index(), on=["unit", "year"], how="left", validate="one_to_one"
    )
    d["state_year"] = (d["state"] * 10_000 + d["year"]).astype(np.int32)

    d["y_cum"] = d["future"] / d["income_real"]
    for share_code in share_dict["share_code"]:
        d[f"zcum{share_code[1:]}"] = (
            d[share_code].to_numpy() * d["loo_future"] / d["income_real"]
        )

    keep = [
        "unit",
        "year",
        "state",
        "state_year",
        "future",
        "income_real",
        "loo_future",
        "y_cum",
    ]
    keep += [f"s{i:02d}" for i in range(1, 13)]
    keep += [f"zcum{i:02d}" for i in range(1, 13)]
    out = d[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out, share_dict, full_analysis_rows, len(d)


def load_flow_cache():
    """Load the certified transaction aggregation when only panel logic changes."""
    award_path = TEMP / "award_county_year.parquet"
    duration_path = TEMP / "duration_county_year.parquet"
    required = {"ccg", "year", *(name for name, _, _ in TREATMENTS)}
    for path in (award_path, duration_path, DATE_AUDIT):
        if not path.exists():
            raise FileNotFoundError(f"required certified cache is missing: {path}")
    award = pd.read_parquet(award_path)
    duration = pd.read_parquet(duration_path)
    date_audit = pd.read_csv(DATE_AUDIT)
    for label, frame in (("award", award), ("duration", duration)):
        missing = required.difference(frame.columns)
        if missing or frame.empty:
            raise AssertionError(f"invalid {label} cache; missing={sorted(missing)}")
    if date_audit.empty:
        raise AssertionError("duration date-audit cache is empty")
    print("Certified flow cache loaded; transaction aggregation skipped")
    return award, duration, date_audit


def main(reuse_flow_cache: bool):
    TEMP.mkdir(parents=True, exist_ok=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    failures = []
    try:
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
            failures.append(
                f"positive-employment universe is {len(universe)}, expected 3094"
            )
        print(f"Fixed positive-employment universe: {len(universe):,} county groups")
        if reuse_flow_cache:
            award, duration, date_audit = load_flow_cache()
        else:
            award, duration, date_audit = aggregate_actions(universe)
            award.to_parquet(TEMP / "award_county_year.parquet", index=False)
            duration.to_parquet(TEMP / "duration_county_year.parquet", index=False)
            date_audit.to_csv(DATE_AUDIT, index=False)

        cross, eligible_states = geography(universe)
        manifest_rows, input_audit, dictionaries = [], [], []
        for rung in ("county", "cz"):
            for flow_name, flow_levels in (("award", award), ("duration", duration)):
                for treatment, _, _ in TREATMENTS:
                    for h in HORIZONS:
                        for window_mode in WINDOW_MODES:
                            out, dictionary, full_rows, common_rows = build_input(
                                flow_levels,
                                award,
                                panel[["ccg", "year", "income_real"]],
                                rung,
                                treatment,
                                h,
                                window_mode,
                                cross,
                                eligible_states,
                            )
                            fname = (
                                f"{rung}_{flow_name}_{treatment}_h{h}_{window_mode}.dta"
                            )
                            path = TEMP / fname
                            out.to_stata(path, write_index=False, version=118)
                            manifest_rows.append(
                                {
                                    "rung": rung,
                                    "flow": flow_name,
                                    "treatment": treatment,
                                    "horizon": h,
                                    "window_mode": window_mode,
                                    "path": str(path),
                                }
                            )
                            input_audit.append(
                                {
                                    "rung": rung,
                                    "flow": flow_name,
                                    "treatment": treatment,
                                    "horizon": h,
                                    "window_mode": window_mode,
                                    "full_analysis_rows": full_rows,
                                    "usable_income_rows": common_rows,
                                    "written_rows": len(out),
                                    "n_units": out["unit"].nunique(),
                                    "n_states": out["state"].nunique(),
                                }
                            )
                            dictionary["rung"] = rung
                            dictionaries.append(dictionary)
                            print(f"  {fname:<64s} {len(out):>7,d} rows")
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(MANIFEST, index=False)
        pd.DataFrame(input_audit).to_csv(INPUT_AUDIT, index=False)
        pd.concat(dictionaries, ignore_index=True).drop_duplicates(
            ["rung", "share_code"]
        ).to_csv(SHARE_DICTIONARY, index=False)

        expected = 2 * 2 * 4 * 3 * 2
        if len(manifest) != expected:
            failures.append(f"manifest has {len(manifest)} rows, expected {expected}")
        if manifest["path"].map(lambda x: Path(x).exists()).sum() != expected:
            failures.append("at least one manifest input is missing")
        if date_audit.empty:
            failures.append("duration date audit is empty")
        summary = {
            "inputs": len(manifest),
            "universe_ccgs": len(universe),
            "eligible_states": len(eligible_states),
            "failures": failures,
        }
        (TEMP / "build_summary.json").write_text(json.dumps(summary, indent=2))
        if failures:
            raise AssertionError("; ".join(failures))
        print(f"All checks passed: {len(manifest)} Stata inputs")
    finally:
        tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-flow-cache",
        action="store_true",
        help="reuse certified award and duration county-year aggregation",
    )
    main(parser.parse_args().reuse_flow_cache)
