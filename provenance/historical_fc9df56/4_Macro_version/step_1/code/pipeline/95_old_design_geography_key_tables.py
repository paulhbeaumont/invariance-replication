"""Measure geography sensitivity for five tables under the locked instrument.

Both arms use the composition instrument declared in producer 57: top-800
harmonized FSC categories, positive non-construction procurement, a five-year
rolling exposure window lagged five years, clipped full-sample category shifts,
and the base-period population denominator. The only moving object is contract
geography.

``reported_source_geography`` reverses every guarded DCADS city-county route in
the two production override tables. ``verified_production_geography`` uses the
production contract and category caches. Both the treatment and instrument are
constructed from the contract geography assigned to their arm. Employment,
population, fixed effects, weights, clusters, samples, and table definitions are
held fixed.

The producer reuses the estimators and panel builders owned by producers 40, 57,
68, 34, and 35. It writes only to the dedicated comparison directory.
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/old_design_geography_key_tables/
# mpgr-new-estimates: displayed cells in five registered tables, two geographies
# mpgr-batch-approved: explicit
# mpgr-variation-scope: old-composition-design|contract-geography|five-key-tables

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
STEP1_PY = PIPELINE.parent / "py"
STEP2_PY = ROOT / "step_2" / "code" / "py"

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402


OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization" / \
    "old_design_geography_key_tables"
LONG_OUT = OUTDIR / "estimates_long.csv"
COMPARISON_OUT = OUTDIR / "comparison_wide.csv"
SUMMARY_OUT = OUTDIR / "table_summary.csv"
LOG_OUT = dc.LOG_DIR / "95_old_design_geography_key_tables.log"
MEMO_OUT = (
    dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
    / "Cleaning_spendingData" / "OLD_DESIGN_GEOGRAPHY_KEY_TABLES.md"
)

BASE_OVERRIDE = PIPELINE.parent / "reference" / "dcads_city_county_overrides.csv"
VERIFIED_OVERRIDE = (
    PIPELINE.parent / "reference" / "dcads_verified_city_county_overrides.csv"
)
GEOGRAPHIES = ("reported_source_geography", "verified_production_geography")
HORIZONS = (1, 2, 3, 4)
TABLE_H = 4


class Tee:
    """Mirror stdout to the registered batch log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, value: str) -> None:
        self.file.write(value)
        self.stdout.write(value)

    def flush(self) -> None:
        self.file.flush()
        self.stdout.flush()

    def close(self) -> None:
        sys.stdout = self.stdout
        self.file.close()


def load_module(name: str, path: Path):
    """Load one registered producer without invoking its command-line entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_guarded_routes() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return every DCADS city-year and exact-contract geography route."""
    dtype = {
        "fiscal_year": "string", "state_code": "string",
        "city_name": "string", "reported_county_fips": "string",
        "corrected_county_fips": "string", "contract_id": "string",
    }
    local = pd.read_csv(BASE_OVERRIDE, dtype=dtype).fillna("")
    local["contract_id"] = ""
    systematic = pd.read_csv(VERIFIED_OVERRIDE, dtype=dtype).fillna("")
    rules = pd.concat([local, systematic], ignore_index=True)
    rules["fiscal_year"] = pd.to_numeric(
        rules["fiscal_year"], errors="raise").astype(int)
    rules["state_code"] = rules["state_code"].str.zfill(2)
    rules["reported_county_fips"] = rules["reported_county_fips"].str.zfill(5)
    rules["corrected_county_fips"] = rules["corrected_county_fips"].str.zfill(5)

    route_key = [
        "fiscal_year", "state_code", "city_name",
        "reported_county_fips", "contract_id",
    ]
    if rules.duplicated(route_key).any():
        raise SystemExit("FAIL: guarded DCADS route keys are not unique")
    expected_rows = int(pd.to_numeric(rules["expected_rows"], errors="raise").sum())
    common = rules.rename(columns={
        "fiscal_year": "year", "state_code": "state", "city_name": "city",
        "reported_county_fips": "suspect_county",
        "corrected_county_fips": "alternative_county",
    })
    city = common.loc[common["contract_id"].eq(""), [
        "year", "state", "city", "suspect_county", "alternative_county",
    ]].drop_duplicates()
    contract = common.loc[common["contract_id"].ne(""), [
        "year", "state", "city", "suspect_county", "contract_id",
        "alternative_county",
    ]].drop_duplicates()
    if len(city) != 154 or len(contract) != 4 or expected_rows != 19_189:
        raise SystemExit(
            "FAIL: guarded route universe differs from 154 city-year routes, "
            "four exact-contract routes, and 19,189 expected source rows")
    return city, contract, expected_rows


def usaspending_guarded_rows(p93) -> pd.DataFrame:
    """Return post-seam Sealy actions routed by the production parser."""
    paths = [str(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet")
             for year in range(dc.SEAM_YEAR, 2019)]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"FAIL: missing USAspending inputs: {missing}")
    connection = duckdb.connect()
    moved = connection.execute("""
        SELECT
            CAST(source_fiscal_year AS INTEGER) AS year,
            CAST(federal_action_obligation AS DOUBLE) AS dollars_nominal,
            CAST(product_or_service_code AS VARCHAR) AS fsc,
            '48089' AS suspect_county,
            '48015' AS alternative_county
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(award_or_idv_flag AS VARCHAR) = 'AWARD'
          AND TRIM(REGEXP_REPLACE(
                UPPER(TRIM(CAST(primary_place_of_performance_city_name AS VARCHAR))),
                '[^A-Z0-9]+', ' ', 'g')) = 'SEALY'
          AND LPAD(TRIM(CAST(
                prime_award_transaction_place_of_performance_county_fips_code
                AS VARCHAR)), 5, '0') = '48089'
          AND LPAD(TRIM(CAST(county_fips AS VARCHAR)), 5, '0') = '48015'
          AND CAST(county_fips_source AS VARCHAR) = 'city_county_override'
    """, [paths]).df()
    connection.close()
    if len(moved) != 5_716:
        raise SystemExit(
            f"FAIL: Sealy route selects {len(moved):,} USAspending rows, "
            "expected 5,716")
    crosswalk = pd.read_csv(
        p93.CCG_IN, dtype={"county_fips5": "string", "ccg_id": "string"})
    mapping = crosswalk.set_index("county_fips5")["ccg_id"]
    moved["suspect_ccg"] = moved["suspect_county"].map(mapping)
    moved["alternative_ccg"] = moved["alternative_county"].map(mapping)
    if moved[["suspect_ccg", "alternative_ccg"]].isna().any().any():
        raise SystemExit("FAIL: Sealy route does not map to the fixed CCG geography")
    moved["suspect_ccg"] = moved["suspect_ccg"].astype(np.int64)
    moved["alternative_ccg"] = moved["alternative_ccg"].astype(np.int64)
    moved["dollars_real"] = (
        moved["dollars_nominal"] * moved["year"].map(dc.deflate))
    return moved


def geography_panels(p57, p93) -> tuple[dict[str, dict], dict[str, int]]:
    """Build panel dictionaries for reported and verified contract geography."""
    production = p57.panels()
    city, contract, expected_rows = all_guarded_routes()
    moved = p93.attach_ccgs(p93.extract_moved_rows(city, contract))
    if len(moved) != expected_rows:
        raise SystemExit(
            f"FAIL: guarded routes select {len(moved):,} source rows, expected "
            f"{expected_rows:,}")

    key = ("positive", 0.0)
    sealy = usaspending_guarded_rows(p93)
    con_reported = p93.transfer_contract_matrix(
        production["CON"][key], moved, "alternative_ccg", "suspect_ccg")
    con_reported = p93.transfer_contract_matrix(
        con_reported, sealy, "alternative_ccg", "suspect_ccg")
    arr_reported = p93.transfer_category_tensor(
        production["ARR"][key], production["units"], production["cats"],
        production["cy"], moved, "alternative_ccg", "suspect_ccg")
    arr_reported = p93.transfer_category_tensor(
        arr_reported, production["units"], production["cats"],
        production["cy"], sealy, "alternative_ccg", "suspect_ccg")

    reported = production.copy()
    reported["CON"] = production["CON"].copy()
    reported["ARR"] = production["ARR"].copy()
    reported["CON"][key] = con_reported
    reported["ARR"][key] = arr_reported

    return {
        "reported_source_geography": reported,
        "verified_production_geography": production,
    }, {"dcads": len(moved), "usaspending": len(sealy)}


def activate(p57, panels: dict) -> None:
    """Select one in-memory contract geography for producer 57."""
    p57._PANELS.clear()
    p57._PANELS.update(panels)


def add_row(rows: list[dict], geography: str, table: str, *, horizon: int,
            level: str, weight: str, statistic: str, value: float,
            se: float = np.nan, n: float = np.nan,
            n_units: float = np.nan, outcome: str = "y_mpgr") -> None:
    """Append one displayed table cell in a common schema."""
    se_value = float(se) if np.isfinite(se) else np.nan
    rows.append({
        "geography": geography, "table": table, "horizon": int(horizon),
        "level": level, "weight": weight, "statistic": statistic,
        "outcome": outcome, "value": float(value), "se": se_value,
        "t": float(value / se_value) if np.isfinite(se_value) and se_value else np.nan,
        "n": float(n) if np.isfinite(n) else np.nan,
        "n_units": float(n_units) if np.isfinite(n_units) else np.nan,
    })


def weighting_table(rows: list[dict], geography: str, p57, p40) -> None:
    """Estimate every displayed cell of the weighting ladder."""
    for h in HORIZONS:
        d, dropped = p57.build_county(
            "composition", h=h, span=5, rule="positive", floor=0.0,
            denom="population")
        if dropped:
            raise SystemExit(f"FAIL: weighting frame drops {dropped} counties at h={h}")
        d["popw"] = d["P0"]
        d["divyear"] = d["division"].astype(str) + "_" + d["year"].astype(str)
        for rung in p40.RUNGS:
            g = d if rung == "county" else d[d[rung].notna()]
            g = g.copy()
            om, p0j = p40.weight_vectors(g, rung)
            for weight, vector in (("P_j", om * p0j), ("EW", om)):
                beta, se, n = p40.fit(g, vector)
                add_row(
                    rows, geography, "weighting_ladder", horizon=h,
                    level=rung, weight=weight, statistic="beta", value=beta,
                    se=se, n=n,
                    n_units=(g["ccg"].nunique() if rung == "county"
                             else g[rung].nunique()))


def aggregation_tables(rows: list[dict], geography: str, p57, p68,
                       t28, w22, scratch: Path) -> None:
    """Estimate the main rung ladder and the naive-regressor table."""
    baseline_path = scratch / geography / "aggregation_baseline.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    p57.BASE_OUT = baseline_path
    p57.baseline(None, t28, w22, time.time())
    base = pd.read_csv(baseline_path)

    p68.p57, p68.t28, p68.w22 = p57, t28, w22
    battery_rows: list[dict] = []
    definition = p57.BASELINE
    d, _ = p57.build_county(
        definition["instrument"], h=TABLE_H, span=definition["span"],
        rule=definition["rule"], floor=definition["floor"],
        denom=definition["denom"])
    for level in p68.RUNGS:
        g, agg, _, _ = p68.rung_frame(d, level, definition["fe"])
        for weight, _ in p68.WEIGHTS:
            wc_naive, wc_match, wa = p68.weights_for(g, agg, weight)
            aligned = p68.t2_naive_rhs(
                agg, level, wa, battery_rows, TABLE_H, weight, "y_mpgr")
            _, beta_b, delta = p68.t3_implicit(
                g, agg, level, wc_naive, wc_match, battery_rows,
                TABLE_H, weight, "y_mpgr")
            anchor = base.loc[
                base["horizon"].eq(TABLE_H)
                & base["level"].eq(level)
                & base["weight"].eq(weight)
            ].iloc[0]
            for observed, expected, label in (
                    (aligned * p57.SCALE, anchor["beta_a_aligned"], "aligned"),
                    (beta_b * p57.SCALE, anchor["beta_b"], "beta_b"),
                    (delta * p57.SCALE, anchor["delta"], "delta")):
                if abs(observed - expected) > 1e-6:
                    raise SystemExit(
                        f"FAIL: aggregation tie-out {label}, {level}, {weight}: "
                        f"{observed:.10f} vs {expected:.10f}")

    battery = pd.DataFrame(battery_rows)
    for record in base.loc[base["horizon"].eq(TABLE_H)].itertuples(index=False):
        selectors = (
            ("beta_b", "beta_b_matched", record.beta_b),
            ("delta", "delta_matched", record.delta),
        )
        for statistic, source_stat, value in selectors:
            source = battery.loc[
                battery["test"].eq("T3")
                & battery["level"].eq(record.level)
                & battery["weight"].eq(record.weight)
                & battery["outcome"].eq("y_mpgr")
                & battery["stat"].eq(source_stat)
            ].iloc[0]
            add_row(
                rows, geography, "up_the_rungs", horizon=TABLE_H,
                level=record.level, weight=record.weight,
                statistic=statistic, value=value, se=source.se,
                n=source.n, n_units=record.n_units)
        add_row(
            rows, geography, "up_the_rungs", horizon=TABLE_H,
            level=record.level, weight=record.weight,
            statistic="beta_a_own", value=record.beta_a_own,
            se=record.beta_a_own_se, n=record.n_agg,
            n_units=record.n_units)

    for record in battery.loc[
            battery["test"].eq("T2") & battery["outcome"].eq("y_mpgr")
            ].itertuples(index=False):
        add_row(
            rows, geography, "naive_aggregation_regressor", horizon=TABLE_H,
            level=record.level, weight=record.weight,
            statistic=record.stat, value=record.value, se=record.se,
            n=record.n, n_units=record.n_units)


def disaggregation_tables(rows: list[dict], geography: str, p57, p34, p35,
                          w22) -> None:
    """Estimate the displayed cells of Table 1 and Table 5 at h=4."""
    p34.md._P57 = p57
    p34.md.w22.DEMEANER = w22.DEMEANER
    panels = {
        rung: p34.md.Panel(
            TABLE_H, rung, frame_src="baseline", source="efsy", quiet=False)
        for rung in p34.md.RUNGS
    }

    cell_fits = {}
    for rung, panel in panels.items():
        y, weight, mask = p34.cell_terms(panel, "midpoint_mpgr")
        cell_fits[rung] = panel.fit_cell_base(
            y, weight, mask, indyear=False)
    county_panel = panels["naics4_alloc"]
    county_y, county_mask = p34.county_outcome(county_panel, "midpoint_mpgr")
    county_fit = county_panel.fit_county_base(county_y, county_mask)
    table1 = (
        ("county", county_fit),
        ("naics2", cell_fits["naics2"]),
        ("naics4", cell_fits["naics4_alloc"]),
    )
    for level, fit in table1:
        add_row(
            rows, geography, "table1_midpoint_invariance", horizon=TABLE_H,
            level=level, weight="midpoint_share", statistic="beta",
            value=fit["beta"] * 1e5, se=fit["se"] * 1e5,
            n=fit["n"],
            n_units=county_panel.frame["ccg"].nunique())
    if max(abs(fit["beta"] - county_fit["beta"])
           for _, fit in table1[1:]) > p34.md.t28.TOL_INVAR:
        raise SystemExit("FAIL: midpoint invariance does not hold")

    p35.md._P57 = p57
    original_factory = p35.md.Panel
    p35.md.Panel = lambda h, rung, *args, **kwargs: panels[rung]
    try:
        decomposition = [
            p35.one_cell(
                TABLE_H, rung, "fixed5", 5.0,
                frame_src="baseline", source="efsy")
            for rung in p35.md.RUNGS
        ]
    finally:
        p35.md.Panel = original_factory

    displayed = (
        "county", "cell_naive", "gap_naive", "entry", "exit", "jensen",
        "renorm", "cell_equal", "gap_equal", "n_cy", "n_log_blind",
    )
    for record in decomposition:
        level = record["rung"].replace("_alloc", "")
        for statistic in displayed:
            add_row(
                rows, geography, "table5_log_gap", horizon=TABLE_H,
                level=level, weight="naive_or_equal_as_labeled",
                statistic=statistic, value=record[statistic], outcome="y_log")


def comparison_frame(long: pd.DataFrame) -> pd.DataFrame:
    """Put both geography arms and their differences on one row per cell."""
    keys = ["table", "horizon", "level", "weight", "statistic", "outcome"]
    measures = ["value", "se", "t", "n", "n_units"]
    pieces = []
    for measure in measures:
        wide = long.pivot(index=keys, columns="geography", values=measure)
        wide.columns = [f"{measure}_{column}" for column in wide.columns]
        pieces.append(wide)
    out = pd.concat(pieces, axis=1).reset_index()
    for measure in measures:
        reported = f"{measure}_reported_source_geography"
        verified = f"{measure}_verified_production_geography"
        out[f"{measure}_difference"] = out[verified] - out[reported]
        if measure in ("value", "se", "t"):
            denominator = out[reported].abs()
            out[f"{measure}_percent_difference"] = np.where(
                denominator > 1e-12,
                100.0 * out[f"{measure}_difference"] / denominator,
                np.nan)
    return out.sort_values(keys).reset_index(drop=True)


def table_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient movements and identify each table's largest one."""
    coefficient = comparison.loc[
        ~comparison["statistic"].isin(("n_cy", "n_log_blind"))
    ].copy()
    records = []
    for table, frame in coefficient.groupby("table", sort=False):
        frame = frame.assign(abs_difference=frame["value_difference"].abs())
        largest = frame.loc[frame["abs_difference"].idxmax()]
        records.append({
            "table": table,
            "displayed_coefficient_cells": len(frame),
            "mean_absolute_difference": frame["abs_difference"].mean(),
            "max_absolute_difference": largest["abs_difference"],
            "largest_difference_horizon": int(largest["horizon"]),
            "largest_difference_level": largest["level"],
            "largest_difference_weight": largest["weight"],
            "largest_difference_statistic": largest["statistic"],
            "reported_value_at_largest_difference":
                largest["value_reported_source_geography"],
            "verified_value_at_largest_difference":
                largest["value_verified_production_geography"],
        })
    return pd.DataFrame(records)


def verify_reported_artifacts(long: pd.DataFrame) -> None:
    """Tie the reconstructed source-geography arm to registered table inputs."""
    reference = long.loc[long["geography"].eq("reported_source_geography")]

    weighting_saved = pd.read_csv(
        dc.MACRO_RESULTS_ROOT / "weighting_ladder" / "weighting_ladder.csv")
    weighting_saved = weighting_saved.rename(columns={"rung": "level"})
    weighting = reference.loc[
        reference["table"].eq("weighting_ladder")
    ].merge(
        weighting_saved,
        on=["horizon", "level", "weight"],
        how="outer", validate="one_to_one", indicator=True,
        suffixes=("_comparison", "_saved"),
    )
    if not weighting["_merge"].eq("both").all():
        raise SystemExit("FAIL: weighting reference keys differ from saved inputs")
    for left, right in (("value", "beta"), ("se_comparison", "se_saved")):
        gap = float(np.max(np.abs(weighting[left] - weighting[right])))
        if gap > 1e-6:
            raise SystemExit(
                f"FAIL: weighting reference {left} differs from saved input by {gap:.3e}")

    table1_saved = pd.read_csv(
        ROOT / "step_2" / "result" / "coeff_tables_payoff_baseline_h4.csv")
    table1_saved = table1_saved.loc[table1_saved["spec"].eq("midpoint_mpgr")].iloc[0]
    table1 = reference.loc[
        reference["table"].eq("table1_midpoint_invariance")
    ].set_index("level")
    for level, column in (("county", "county"), ("naics2", "n2"),
                          ("naics4", "n4")):
        gap = abs(float(table1.loc[level, "value"]) - float(table1_saved[column]))
        if gap > 1e-6:
            raise SystemExit(
                f"FAIL: Table 1 reference {level} differs from saved input by {gap:.3e}")

    table5_saved = pd.read_csv(
        ROOT / "step_2" / "result" / "coeff_table_decomposition_baseline.csv")
    table5_saved = table5_saved.loc[table5_saved["h"].eq(TABLE_H)].copy()
    table5_saved["level"] = table5_saved["rung"].str.replace(
        "_alloc", "", regex=False)
    table5 = reference.loc[reference["table"].eq("table5_log_gap")]
    for record in table5.itertuples(index=False):
        saved = table5_saved.loc[table5_saved["level"].eq(record.level)].iloc[0]
        gap = abs(float(record.value) - float(saved[record.statistic]))
        if gap > 1e-6:
            raise SystemExit(
                f"FAIL: Table 5 reference {record.level}/{record.statistic} "
                f"differs from saved input by {gap:.3e}")
    print("reported-source reference reproduces weighting, Table 1, and Table 5 inputs")


def fmt(value: float, digits: int = 4) -> str:
    """Format a finite result for the internal Markdown tables."""
    return "" if not np.isfinite(value) else f"{value:.{digits}f}"


def memo(comparison: pd.DataFrame, summary: pd.DataFrame,
         source_rows: dict[str, int]) -> None:
    """Write a table-backed internal record of the geography comparison."""
    weight = comparison.loc[
        comparison["table"].eq("weighting_ladder")
        & comparison["level"].eq("county")
        & comparison["horizon"].eq(2)
    ].set_index("weight")
    h1_cz = comparison.loc[
        comparison["table"].eq("weighting_ladder")
        & comparison["level"].eq("czone")
        & comparison["horizon"].eq(1)
        & comparison["weight"].eq("EW")
    ].iloc[0]

    def stars(t_value: float) -> str:
        absolute = abs(t_value)
        return "***" if absolute >= 2.576 else "**" if absolute >= 1.96 \
            else "*" if absolute >= 1.645 else ""

    inferential = comparison.loc[comparison["t_difference"].notna()].copy()
    inferential["reported_stars"] = inferential[
        "t_reported_source_geography"].map(stars)
    inferential["verified_stars"] = inferential[
        "t_verified_production_geography"].map(stars)
    star_changes = int((inferential["reported_stars"]
                        != inferential["verified_stars"]).sum())

    lines = [
        "# Old instrument design: sensitivity to contract geography",
        "",
        "## Design",
        "",
        ("Both arms use the locked composition instrument and the registered "
         "specification of each table. The reported-source arm assigns DCADS "
         "records to their source county. The verified-production arm applies "
         "154 guarded city-year routes and four guarded exact-contract routes, "
         f"covering {source_rows['dcads']:,} DCADS source rows, plus the guarded "
         f"Sealy route covering {source_rows['usaspending']:,} USAspending source "
         "rows. Employment, population, "
         "fixed effects, weights, clusters, and table definitions are identical "
         "across the two arms."),
        "",
        "## Headline",
        "",
        (f"At h=2, the equal-county baseline coefficient is "
         f"{weight.loc['EW', 'value_reported_source_geography']:.6f} under "
         f"reported-source geography and "
         f"{weight.loc['EW', 'value_verified_production_geography']:.6f} under "
         f"verified production geography. Its t-statistic is "
         f"{weight.loc['EW', 't_reported_source_geography']:.6f} and "
         f"{weight.loc['EW', 't_verified_production_geography']:.6f}, "
         "respectively."),
        "",
        (f"At h=2, the population-weighted baseline coefficient is "
         f"{weight.loc['P_j', 'value_reported_source_geography']:.6f} and "
         f"{weight.loc['P_j', 'value_verified_production_geography']:.6f}; "
         f"the corresponding t-statistics are "
         f"{weight.loc['P_j', 't_reported_source_geography']:.6f} and "
         f"{weight.loc['P_j', 't_verified_production_geography']:.6f}."),
        "",
        (f"The largest movement is the h=1 equal-weight commuting-zone cell: "
         f"{h1_cz['value_reported_source_geography']:.6f} to "
         f"{h1_cz['value_verified_production_geography']:.6f}, with t-statistics "
         f"{h1_cz['t_reported_source_geography']:.6f} and "
         f"{h1_cz['t_verified_production_geography']:.6f}. Across all displayed "
         f"cells with standard errors, {star_changes} cells change conventional "
         "significance category."),
        "",
        "## Summary",
        "",
        "| Table | Cells | Mean absolute change | Maximum absolute change | Largest-change cell | Reported source | Verified production |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for record in summary.itertuples(index=False):
        cell = (f"h={record.largest_difference_horizon}, "
                f"{record.largest_difference_level}, "
                f"{record.largest_difference_weight}, "
                f"{record.largest_difference_statistic}")
        lines.append(
            f"| {record.table} | {record.displayed_coefficient_cells} | "
            f"{record.mean_absolute_difference:.6f} | "
            f"{record.max_absolute_difference:.6f} | {cell} | "
            f"{record.reported_value_at_largest_difference:.6f} | "
            f"{record.verified_value_at_largest_difference:.6f} |")

    lines += ["", "## Displayed cells", ""]
    table_labels = {
        "weighting_ladder": "What the weight does to the baseline",
        "up_the_rungs": "Up the rungs",
        "naive_aggregation_regressor": "Naive aggregation of the regressor bites",
        "table1_midpoint_invariance": "Table 1: midpoint growth is invariant to the split",
        "table5_log_gap": "Table 5: decomposing the log gap",
    }
    for table, label in table_labels.items():
        frame = comparison.loc[comparison["table"].eq(table)].copy()
        lines += [f"### {label}", "",
                  "| h | Level | Weight | Statistic | Reported source | Verified production | Difference |",
                  "|---:|---|---|---|---:|---:|---:|"]
        for record in frame.itertuples(index=False):
            lines.append(
                f"| {record.horizon} | {record.level} | {record.weight} | "
                f"{record.statistic} | "
                f"{fmt(record.value_reported_source_geography)} | "
                f"{fmt(record.value_verified_production_geography)} | "
                f"{fmt(record.value_difference)} |")
        lines.append("")

    lines += [
        "## Reproducible files",
        "",
        "- `Results/instrument_normalization/old_design_geography_key_tables/estimates_long.csv`",
        "- `Results/instrument_normalization/old_design_geography_key_tables/comparison_wide.csv`",
        "- `Results/instrument_normalization/old_design_geography_key_tables/table_summary.csv`",
        "- Producer: `step_1/code/pipeline/95_old_design_geography_key_tables.py`",
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the five-table geography comparison."""
    start = time.time()
    tee = Tee(LOG_OUT)
    try:
        print("== Old composition design: five-table geography comparison")
        p57 = load_module("p57_geo_key_tables", STEP1_PY / "57_aggregation_invariance.py")
        p40 = load_module("p40_geo_key_tables", PIPELINE / "40_weighting_ladder.py")
        p68 = load_module("p68_geo_key_tables", STEP1_PY / "68_aggregation_battery.py")
        p93 = load_module("p93_geo_key_tables", PIPELINE / "93_h2_geography_cleaning_ft_comparison.py")
        t28 = load_module("t28_geo_key_tables", STEP2_PY / "28_table_d1.py")
        w22 = t28.w22
        p34 = load_module("p34_geo_key_tables", STEP2_PY / "34_tables_payoff.py")
        p35 = load_module("p35_geo_key_tables", STEP2_PY / "35_table_decomposition.py")

        import pyfixest as pf
        demeaner = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
        w22.DEMEANER = demeaner
        t28.w22.DEMEANER = demeaner
        p34.md.w22.DEMEANER = demeaner
        p35.md.w22.DEMEANER = demeaner

        variants, source_rows = geography_panels(p57, p93)
        rows: list[dict] = []
        scratch = ROOT / "data" / "temp" / "old_design_geography_key_tables"
        scratch.mkdir(parents=True, exist_ok=True)

        for geography in GEOGRAPHIES:
            print(f"\n{'=' * 92}\n  {geography}\n{'=' * 92}")
            activate(p57, variants[geography])
            weighting_table(rows, geography, p57, p40)
            aggregation_tables(rows, geography, p57, p68, t28, w22, scratch)
            disaggregation_tables(rows, geography, p57, p34, p35, w22)

        long = pd.DataFrame(rows)
        verify_reported_artifacts(long)
        comparison = comparison_frame(long)
        summary = table_summary(comparison)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        long.to_csv(LONG_OUT, index=False, float_format="%.15g")
        comparison.to_csv(COMPARISON_OUT, index=False, float_format="%.15g")
        summary.to_csv(SUMMARY_OUT, index=False, float_format="%.15g")
        memo(comparison, summary, source_rows)

        print("\nSummary of displayed coefficient cells")
        print(summary.to_string(index=False))
        print(f"\nwrote {LONG_OUT}")
        print(f"wrote {COMPARISON_OUT}")
        print(f"wrote {SUMMARY_OUT}")
        print(f"wrote {MEMO_OUT}")
        print(f"done in {time.time() - start:.1f}s")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
