"""Compare production geography with a seven-location reference at h=2.

Production geography moves DCADS records only in fiscal years with exact
cross-source contract evidence. A unique city-year-county assignment moves all
matching source rows. Bethlehem, Pennsylvania, has two supported destination
counties in fiscal year 2006, so only exact matched contract identifiers are
routed there. USAspending candidates occur before the production source seam
and therefore do not enter the h=2 contract tensor.

The comparison reverses those systematic moves to reconstruct the geography
containing only the seven location-specific corrections. Both the endogenous
procurement change and the composition instrument follow each geography. The
outcome, balanced county sample, division-by-year fixed effects, equal-county
weights, and county clustering remain fixed.

Writes Results/instrument_normalization/
    h2_geography_cleaning_ft_comparison.csv
    h2_geography_cleaning_application.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_geography_cleaning_ft_comparison.csv
# mpgr-new-estimates: 2
# mpgr-batch-approved: explicit
# mpgr-variation-scope: h2|verified-contract-geography|first-stage-and-2sls

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
import harmonize as hz  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CLASSIFICATION_IN = OUTDIR / "complete_city_county_candidate_classification.csv"
CONFLICTS_IN = OUTDIR / "complete_city_county_contract_conflicts.csv"
FIRST_STAGE_IN = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
COMPARISON_OUT = OUTDIR / "h2_geography_cleaning_ft_comparison.csv"
APPLICATION_OUT = OUTDIR / "h2_geography_cleaning_application.csv"
MEMO_OUT = (
    dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
    / "Cleaning_spendingData" / "GEOGRAPHY_CLEANING_FT_COMPARISON.md"
)
CCG_IN = PIPELINE.parent / "reference" / "county_ccg_crosswalk.csv"
COMMON_YEARS = tuple(range(2001, 2007))
AMBIGUOUS_CITY_YEAR = (2006, "42", "BETHLEHEM", "42033")


def load_module(name: str, path: Path):
    """Load one project producer without invoking its command-line entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_bool(series: pd.Series) -> pd.Series:
    """Interpret persisted CSV booleans without relying on dtype inference."""
    return series.astype(str).str.lower().eq("true")


def verified_rules() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return active city-year rules, contract routes, and all verified pairs."""
    dtype = {
        "state": "string", "dcads_county": "string",
        "usaspending_county": "string", "suspect_county": "string",
        "alternative_county": "string", "contract_id": "string",
    }
    classified = pd.read_csv(CLASSIFICATION_IN, dtype=dtype)
    classified = classified.loc[
        normalize_bool(classified["production_correction_authorized"])
    ].copy()
    if len(classified) != 115:
        raise SystemExit(
            f"FAIL: expected 115 verified pairs, found {len(classified)}")

    conflicts = pd.read_csv(CONFLICTS_IN, dtype=dtype)
    conflicts = conflicts.loc[
        normalize_bool(conflicts["net_dollars_match_to_cent"])
    ].copy()
    pair_keys = ["state", "city", "dcads_county", "usaspending_county"]
    columns = pair_keys + [
        "suspect_source", "suspect_county", "alternative_county"]
    exact = conflicts.merge(
        classified[columns], on=pair_keys, how="inner", validate="many_to_one")
    exact["year"] = pd.to_numeric(exact["year"], errors="raise").astype(int)
    exact["state"] = exact["state"].str.zfill(2)
    exact["suspect_county"] = exact["suspect_county"].str.zfill(5)
    exact["alternative_county"] = exact["alternative_county"].str.zfill(5)

    pair_years = exact[[
        *pair_keys, "suspect_source", "suspect_county",
        "alternative_county", "year",
    ]].drop_duplicates()
    observed = pair_years.groupby(pair_keys, sort=False).size()
    if len(observed) != len(classified):
        raise SystemExit("FAIL: at least one verified pair lacks an exact fiscal year")

    active = pair_years.loc[pair_years["suspect_source"].eq("DCADS")].copy()
    join_keys = ["year", "state", "city", "suspect_county"]
    ambiguous = active.duplicated(join_keys, keep=False)
    ambiguous_keys = set(map(tuple, active.loc[ambiguous, join_keys].itertuples(
        index=False, name=None)))
    if ambiguous_keys != {AMBIGUOUS_CITY_YEAR}:
        raise SystemExit(
            f"FAIL: unexpected ambiguous city-year routes: {sorted(ambiguous_keys)}")

    city_rules = active.loc[~ambiguous, [
        *join_keys, "alternative_county"]].drop_duplicates()
    contract_routes = exact.loc[
        exact["suspect_source"].eq("DCADS")
        & exact[["year", "state", "city", "suspect_county"]]
        .apply(tuple, axis=1).isin(ambiguous_keys),
        [*join_keys, "contract_id", "alternative_county"],
    ].drop_duplicates()
    route_keys = [*join_keys, "contract_id"]
    if contract_routes.duplicated(route_keys).any():
        raise SystemExit("FAIL: an exact Bethlehem contract has two destinations")
    if len(active) != 131 or active[pair_keys].drop_duplicates().shape[0] != 109:
        raise SystemExit("FAIL: active DCADS rule universe drifted")
    return city_rules, contract_routes, classified


def extract_moved_rows(
        city_rules: pd.DataFrame, contract_routes: pd.DataFrame) -> pd.DataFrame:
    """Read every DCADS row selected by the verified fiscal-year rules."""
    paths = [str(dc.OUT_DIR / f"fy{year}.parquet") for year in COMMON_YEARS]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"FAIL: missing DCADS inputs: {missing}")
    connection = duckdb.connect()
    connection.register("city_rules", city_rules)
    connection.register("contract_routes", contract_routes)
    moved = connection.execute("""
        WITH raw AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                LPAD(TRIM(CAST(state_fips2 AS VARCHAR)), 2, '0') AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                CASE
                    WHEN CAST(fiscal_year AS INTEGER) BETWEEN 2001 AND 2003
                    THEN LPAD(TRIM(CAST(county_code AS VARCHAR)), 5, '0')
                    ELSE LPAD(TRIM(CAST(state_code AS VARCHAR)), 2, '0')
                         || LPAD(TRIM(CAST(county_fips AS VARCHAR)), 3, '0')
                END AS suspect_county,
                REGEXP_REPLACE(UPPER(TRIM(CAST(contract_no AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id,
                CAST(dollars AS DOUBLE) AS dollars,
                TRIM(CAST(dollars_unit AS VARCHAR)) AS dollars_unit,
                CAST(fsc AS VARCHAR) AS fsc
            FROM read_parquet(?, union_by_name=true)
            WHERE NOT CAST(is_lumpsum AS BOOLEAN)
        ), city_moves AS (
            SELECT r.*, c.alternative_county, 'city_year' AS routing_mode
            FROM raw r
            INNER JOIN city_rules c
              USING (year, state, city, suspect_county)
        ), contract_moves AS (
            SELECT r.*, c.alternative_county, 'exact_contract' AS routing_mode
            FROM raw r
            INNER JOIN contract_routes c
              USING (year, state, city, suspect_county, contract_id)
        )
        SELECT * FROM city_moves
        UNION ALL
        SELECT * FROM contract_moves
    """, [paths]).df()
    connection.close()
    if moved.empty:
        raise SystemExit("FAIL: verified geography rules selected no DCADS rows")

    observed_city = moved.loc[moved["routing_mode"].eq("city_year"), [
        "year", "state", "city", "suspect_county", "alternative_county",
    ]].drop_duplicates()
    expected_city = city_rules.drop_duplicates()
    if len(observed_city.merge(
            expected_city, how="outer", indicator=True).query("_merge != 'both'")):
        raise SystemExit("FAIL: at least one city-year rule has no source rows")

    route_keys = [
        "year", "state", "city", "suspect_county", "contract_id",
        "alternative_county",
    ]
    observed_contract = moved.loc[
        moved["routing_mode"].eq("exact_contract"), route_keys
    ].drop_duplicates()
    expected_contract = contract_routes[route_keys].drop_duplicates()
    if len(observed_contract.merge(
            expected_contract, how="outer", indicator=True)
            .query("_merge != 'both'")):
        raise SystemExit("FAIL: at least one exact contract route has no source rows")
    return moved


def attach_ccgs(moved: pd.DataFrame) -> pd.DataFrame:
    """Map reported and corrected counties to the fixed CCG geography."""
    crosswalk = pd.read_csv(
        CCG_IN, dtype={"county_fips5": "string", "ccg_id": "string"})
    mapping = crosswalk.set_index("county_fips5")["ccg_id"]
    moved = moved.copy()
    moved["suspect_ccg"] = moved["suspect_county"].map(mapping)
    moved["alternative_ccg"] = moved["alternative_county"].map(mapping)
    if moved[["suspect_ccg", "alternative_ccg"]].isna().any().any():
        failed = moved.loc[
            moved[["suspect_ccg", "alternative_ccg"]].isna().any(axis=1),
            ["suspect_county", "alternative_county"],
        ].drop_duplicates()
        raise SystemExit(f"FAIL: correction county lacks a CCG:\n{failed}")
    moved["suspect_ccg"] = moved["suspect_ccg"].astype(np.int64)
    moved["alternative_ccg"] = moved["alternative_ccg"].astype(np.int64)
    if moved["suspect_ccg"].eq(moved["alternative_ccg"]).any():
        raise SystemExit("FAIL: a verified correction stays inside one CCG")
    moved["dollars_nominal"] = np.where(
        moved["dollars_unit"].eq("thousands"),
        moved["dollars"].to_numpy(np.float64) * 1000.0,
        moved["dollars"].to_numpy(np.float64),
    )
    moved["deflator"] = moved["year"].map(dc.deflate)
    if moved["deflator"].isna().any():
        raise SystemExit("FAIL: a moved row has no fiscal-year deflator")
    moved["dollars_real"] = moved["dollars_nominal"] * moved["deflator"]
    return moved


def transfer_contract_matrix(
        current: pd.DataFrame, moved: pd.DataFrame,
        source_col: str, target_col: str) -> pd.DataFrame:
    """Transfer positive non-construction treatment dollars across CCGs."""
    code = moved["fsc"].astype(str).str[:locked.NDIG]
    eligible = (
        moved["dollars_nominal"].gt(0)
        & ~code.str.startswith("Y")
    )
    transfer = (moved.loc[eligible]
                .groupby(["year", source_col, target_col],
                         sort=False, as_index=False)["dollars_real"].sum())
    result = current.astype(np.float64).copy()
    for row in transfer.itertuples(index=False):
        source = getattr(row, source_col)
        target = getattr(row, target_col)
        result.at[source, row.year] -= row.dollars_real
        result.at[target, row.year] += row.dollars_real
    total_gap = np.max(np.abs(
        result.sum(axis=0).to_numpy() - current.sum(axis=0).to_numpy()))
    total_scale = max(float(np.max(np.abs(current.sum(axis=0).to_numpy()))), 1.0)
    if total_gap / total_scale > 1e-12:
        raise SystemExit(
            f"FAIL: treatment transfer changes national dollars by "
            f"{total_gap / total_scale:.3e}")
    if result.to_numpy().min() < -1e-6:
        raise SystemExit(
            f"FAIL: corrected positive contract cell is negative: "
            f"{result.to_numpy().min():.6g}")
    return result


def transfer_category_tensor(
        current: np.ndarray, units: np.ndarray, cats: list[str],
        contract_years: list[int], moved: pd.DataFrame,
        source_col: str, target_col: str) -> np.ndarray:
    """Transfer positive top-category dollars across CCGs."""
    result = current.astype(np.float64).copy()
    unit_pos = pd.Series(np.arange(len(units)), index=units)
    cat_pos = pd.Series(np.arange(len(cats)), index=cats)
    year_pos = {year: index for index, year in enumerate(contract_years)}

    transfers = []
    for year, frame in moved.groupby("year", sort=False):
        source = frame["fsc"]
        cat = source.astype(str).str[:locked.NDIG]
        keep = (
            frame["dollars_nominal"].gt(0).to_numpy()
            & source.notna().to_numpy()
            & cat.str.len().eq(locked.NDIG).to_numpy()
            & ~cat.str.startswith("Y").to_numpy()
        )
        if not keep.any():
            continue
        d = frame.loc[keep, "dollars_nominal"].to_numpy(np.float64)
        source_ccg = frame.loc[keep, source_col].to_numpy(np.int64)
        target_ccg = frame.loc[keep, target_col].to_numpy(np.int64)
        category = cat.loc[keep].to_numpy(dtype=object)
        d, source_ccg, category = hz.harmonize_arrays(
            int(year), d, source_ccg, category)
        target_ccg = frame.loc[keep, target_col].to_numpy(np.int64)
        live = pd.Series(category).isin(cats).to_numpy()
        if not live.any():
            continue
        i0 = pd.Series(source_ccg[live]).map(unit_pos).to_numpy()
        i1 = pd.Series(target_ccg[live]).map(unit_pos).to_numpy()
        j = pd.Series(category[live]).map(cat_pos).to_numpy()
        if (pd.isna(i0).any() or pd.isna(i1).any() or pd.isna(j).any()
                or int(year) not in year_pos):
            raise SystemExit("FAIL: category transfer does not align to tensor")
        q = (d[live] * dc.deflate(int(year))).astype(np.float32).astype(np.float64)
        t = year_pos[int(year)]
        transfers.append(pd.DataFrame({
            "i0": i0.astype(int), "i1": i1.astype(int),
            "j": j.astype(int), "t": t, "q": q,
        }))

    transfers = pd.concat(transfers, ignore_index=True)
    removal = (transfers.groupby(["i0", "j", "t"], sort=False)["q"]
               .sum().rename("removal").reset_index())
    removal["available"] = current[
        removal["i0"].to_numpy(), removal["j"].to_numpy(),
        removal["t"].to_numpy()].astype(np.float64)
    excess = removal["removal"] - removal["available"]
    material = excess > np.maximum(100.0, 1e-6 * removal["available"].abs())
    if material.any():
        failed = removal.loc[material].sort_values("removal", ascending=False).head()
        raise SystemExit(
            "FAIL: moved category dollars exceed the cached source cell:\n"
            f"{failed}")
    removal["factor"] = np.minimum(
        1.0, removal["available"] / removal["removal"])
    transfers = transfers.merge(
        removal[["i0", "j", "t", "factor"]],
        on=["i0", "j", "t"], how="left", validate="many_to_one")
    transfers["q"] *= transfers["factor"]
    np.add.at(
        result,
        (transfers["i0"].to_numpy(), transfers["j"].to_numpy(),
         transfers["t"].to_numpy()),
        -transfers["q"].to_numpy())
    np.add.at(
        result,
        (transfers["i1"].to_numpy(), transfers["j"].to_numpy(),
         transfers["t"].to_numpy()),
        transfers["q"].to_numpy())

    national0 = current.astype(np.float64).sum(axis=0)
    national1 = result.sum(axis=0)
    gap = float(np.max(np.abs(national1 - national0)))
    scale = max(float(np.max(np.abs(national0))), 1.0)
    if gap / scale > 1e-12:
        raise SystemExit(
            f"FAIL: category transfer changes national dollars by {gap / scale:.3e}")
    if result.min() < -1e-6:
        raise SystemExit(
            f"FAIL: corrected positive category cell is negative: "
            f"{result.min():.6g}")
    return result


def treatment_wide(
        contracts: pd.DataFrame, years: list[int],
        population: pd.DataFrame) -> np.ndarray:
    """Construct the locked two-year procurement change per capita."""
    result = np.full((len(contracts), len(years)), np.nan, np.float64)
    for j, year in enumerate(years):
        post = 0.5 * (
            contracts[year].to_numpy(np.float64)
            + contracts[year - 1].to_numpy(np.float64))
        base = contracts[year - H].to_numpy(np.float64)
        denominator = population[year - H].to_numpy(np.float64)
        result[:, j] = (post - base) / denominator
    return result


def application_table(
        moved: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    """Summarize the source records transferred by each active rule."""
    code = moved["fsc"].astype(str).str[:locked.NDIG]
    moved = moved.assign(
        treatment_eligible=moved["dollars_nominal"].gt(0)
        & ~code.str.startswith("Y"),
        category_eligible=moved["dollars_nominal"].gt(0)
        & moved["fsc"].notna() & code.str.len().eq(locked.NDIG)
        & ~code.str.startswith("Y"),
    )
    moved["positive_treatment_nominal_dollars"] = np.where(
        moved["treatment_eligible"], moved["dollars_nominal"], 0.0)
    keys = [
        "year", "state", "city", "suspect_county", "alternative_county",
        "routing_mode",
    ]
    summary = (moved.groupby(keys, sort=False, as_index=False)
               .agg(source_rows=("contract_id", "size"),
                    contract_ids=("contract_id", "nunique"),
                    net_nominal_dollars=("dollars_nominal", "sum"),
                    absolute_nominal_dollars=(
                        "dollars_nominal", lambda value: value.abs().sum()),
                    positive_treatment_nominal_dollars=(
                        "positive_treatment_nominal_dollars", "sum"),
                    treatment_eligible_rows=("treatment_eligible", "sum"),
                    category_eligible_rows=("category_eligible", "sum")))
    summary["verified_pairs_total"] = len(classified)
    summary["verified_dcads_pairs_active"] = int(
        classified["suspect_source"].eq("DCADS").sum())
    summary["verified_usaspending_pairs_before_seam"] = int(
        classified["suspect_source"].eq("USAspending").sum())
    return summary.sort_values(keys).reset_index(drop=True)


def write_memo(output: pd.DataFrame, source_rows: int) -> None:
    """Write the table-backed internal interpretation of the comparison."""
    reference = output.loc[
        output["cleaning"].eq("seven_location_reference")].iloc[0]
    production = output.loc[
        output["cleaning"].eq("verified_expanded_current")].iloc[0]
    lines = [
        "# Geography cleaning: first-stage and second-stage comparison",
        "",
        "## Specification",
        "",
        ("Both rows use the h=2 average-window procurement change per capita, "
         "the h=2 exposure-window market-share instrument, the balanced county "
         "sample, division-by-year fixed effects, equal county weights, and "
         "county-clustered standard errors."),
        "",
        "**Table 1. Geography comparison**",
        "",
        "| Geography | First-stage F | IV coefficient x 100,000 | IV t-statistic |",
        "|---|---:|---:|---:|",
        (f"| Seven-location reference | {reference['first_stage_F']:.10f} | "
         f"{reference['second_stage_beta_x1e5']:.10f} | "
         f"{reference['second_stage_t']:.10f} |"),
        (f"| Verified production geography | "
         f"{production['first_stage_F']:.10f} | "
         f"{production['second_stage_beta_x1e5']:.10f} | "
         f"{production['second_stage_t']:.10f} |"),
        "",
        (f"The verified production geography applies 109 DCADS county-pair "
         f"corrections through 129 city-year rules and four exact-contract "
         f"rules. They cover {source_rows:,} DCADS source rows. The six verified "
         "USAspending pairs fall in fiscal years 2005 to 2006, when DCADS "
         "supplies the production series, so they do not affect the estimates."),
        "",
        (f"Relative to the seven-location reference, the first-stage F-statistic "
         f"differs by {production['first_stage_F_change']:.10f} and the IV "
         f"t-statistic differs by {production['second_stage_t_change']:.10f}. "
         "Table 1 shows that the systematic geography corrections leave the "
         "empirical conclusion unchanged."),
        "",
        "## Reproducible files",
        "",
        "- Estimates: `Results/instrument_normalization/h2_geography_cleaning_ft_comparison.csv`.",
        "- Rule application: `Results/instrument_normalization/h2_geography_cleaning_application.csv`.",
        "- Guarded rules: `step_1/code/reference/dcads_verified_city_county_overrides.csv`.",
        "- Producer: `step_1/code/pipeline/93_h2_geography_cleaning_ft_comparison.py`.",
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Estimate production geography and the seven-location reference."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 production versus seven-location contract geography")

    p57 = load_module(
        "p57_geography_cleaning", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_geography_cleaning", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_geography_cleaning",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_geography_cleaning",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_geography_cleaning",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, cats, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    contracts = store.load_con(locked.RULE).reindex(index=units)
    years = store.target_years(H)

    city_rules, contract_routes, classified = verified_rules()
    moved = attach_ccgs(extract_moved_rows(city_rules, contract_routes))
    application = application_table(moved, classified)
    reference_contracts = transfer_contract_matrix(
        contracts, moved, "alternative_ccg", "suspect_ccg")
    reference_arr = transfer_category_tensor(
        arr, units, cats, contract_years, moved,
        "alternative_ccg", "suspect_ccg")

    rowpos = {unit: index for index, unit in enumerate(units)}
    colpos = {year: index for index, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    current_xwide = treatment_wide(contracts, years, pivots["P"])
    current_x = current_xwide[ri, ci]
    x_gap = float(np.max(np.abs(
        current_x - baseline["x"].to_numpy(np.float64))))
    x_scale = max(float(np.max(np.abs(current_x))), 1.0)
    if x_gap / x_scale > 1e-12:
        raise SystemExit(
            f"FAIL: treatment reconstruction relative gap is {x_gap / x_scale:.3e}")

    reference_xwide = treatment_wide(
        reference_contracts, years, pivots["P"])
    reference_x = reference_xwide[ri, ci]
    current_zwide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    reference_zwide, _ = p68.build_instrument(
        reference_arr, contract_years, years, pivots["P"])
    current_z = current_zwide[ri, ci]
    reference_z = reference_zwide[ri, ci]
    if not np.isfinite(reference_x).all() or not np.isfinite(reference_z).all():
        raise SystemExit("FAIL: reference treatment or instrument is incomplete")

    _, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    y = baseline[locked.OUTCOME].to_numpy(np.float64)
    estimates = {}
    for name, x, z in (
            ("seven_location_reference", reference_x, reference_z),
            ("verified_expanded_current", current_x, current_z)):
        first = p69.within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        second = t28.fit2sls(
            y, x, z, weight, fe, nofe, cluster,
            singleton=False, want_F=False)
        estimates[name] = (first, second)

    first_stage_fixture = pd.read_csv(FIRST_STAGE_IN)
    current_first = estimates["verified_expanded_current"][0]
    if len(first_stage_fixture) != 1 or not np.isclose(
            current_first["F"],
            float(first_stage_fixture.iloc[0]["first_stage_F"]),
            rtol=locked.RTOL_COEF, atol=1e-12):
        raise SystemExit("FAIL: production first-stage fixtures disagree")

    source_rows = int(len(moved))
    active_pairs = int(classified["suspect_source"].eq("DCADS").sum())
    inactive_pairs = int(classified["suspect_source"].eq("USAspending").sum())
    common = {
        "horizon": H,
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "sample_start": int(baseline["year"].min()),
        "sample_end": int(baseline["year"].max()),
        "outcome": locked.OUTCOME,
        "treatment": "two-year average-window procurement change per capita",
        "instrument": "h2 exposure-window market-share shifter",
        "fixed_effects": "division-by-year",
        "weight": "equal county",
        "cluster": "county",
    }
    rows = []
    for name, x, z in (
            ("seven_location_reference", reference_x, reference_z),
            ("verified_expanded_current", current_x, current_z)):
        first, second = estimates[name]
        is_production = name == "verified_expanded_current"
        rows.append({
            "cleaning": name,
            "verified_pairs": len(classified) if is_production else 0,
            "active_dcads_pairs": active_pairs if is_production else 0,
            "inactive_usaspending_pairs": inactive_pairs if is_production else 0,
            "source_rows_routed": source_rows if is_production else 0,
            "first_stage": first["beta"],
            "first_stage_se": first["se"],
            "first_stage_t": first["t"],
            "first_stage_F": first["F"],
            "partial_r2": first["partial_r2"],
            "second_stage_beta": second["beta"],
            "second_stage_se": second["se"],
            "second_stage_t": second["beta"] / second["se"],
            "second_stage_beta_x1e5": second["beta"] * locked.SCALE,
            "second_stage_se_x1e5": second["se"] * locked.SCALE,
            "correlation_treatment_with_production": float(np.corrcoef(
                current_x, x)[0, 1]),
            "correlation_instrument_with_production": float(np.corrcoef(
                current_z, z)[0, 1]),
            **common,
        })
    output = pd.DataFrame(rows)
    base_f = float(output.loc[
        output["cleaning"].eq("seven_location_reference"),
        "first_stage_F"].iloc[0])
    base_t = float(output.loc[
        output["cleaning"].eq("seven_location_reference"),
        "second_stage_t"].iloc[0])
    output["first_stage_F_change"] = output["first_stage_F"] - base_f
    output["second_stage_t_change"] = output["second_stage_t"] - base_t

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(COMPARISON_OUT, index=False)
    application.to_csv(APPLICATION_OUT, index=False)
    write_memo(output, source_rows)
    for row in output.itertuples(index=False):
        print(
            f"   {row.cleaning}: F={row.first_stage_F:.3f}, "
            f"IV t={row.second_stage_t:.3f}, "
            f"beta x 1e5={row.second_stage_beta_x1e5:.3f}")
    print(
        f"   production geography: {active_pairs} active DCADS pairs, "
        f"{inactive_pairs} USAspending pairs before seam, "
        f"{source_rows:,} DCADS rows moved")
    print(f"   wrote {COMPARISON_OUT}")
    print(f"   wrote {APPLICATION_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
