"""Audit source-specific city-county assignments at overlap and seam windows.

DCADS and USAspending are compared using positive DoD contract actions and raw
reported places of performance. One comparison uses their common 2001-2006
coverage; another compares DCADS 2004-2006 with USAspending 2007-2009 across
the production seam. The city audit tests state-city assignments; the
city-category audit repeats the comparison within four-character FSC/PSC
categories. Candidate conflicts require at least 80 percent dominant-county
shares in both sources and at least $1 million of real procurement in each
source.

Writes Results/instrument_normalization/
    usaspending_city_county_source_audit.csv
    usaspending_city_category_county_source_audit.csv
    usaspending_city_county_override_candidates.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_year/fy2001.parquet
# mpgr-output-cache: Results/instrument_normalization/usaspending_city_county_source_audit.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|2001-2006-overlap|city-county-geography

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


COMPARISONS = {
    "overlap_2001_2006": (tuple(range(2001, 2007)), tuple(range(2001, 2007))),
    "seam_2004_2006_vs_2007_2009": (
        tuple(range(2004, 2007)), tuple(range(2007, 2010))),
}
DOMINANCE_MIN = 0.80
DOLLARS_MIN = 1_000_000.0
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CITY_OUT = OUTDIR / "usaspending_city_county_source_audit.csv"
CATEGORY_OUT = (
    OUTDIR / "usaspending_city_category_county_source_audit.csv")
CANDIDATE_OUT = OUTDIR / "usaspending_city_county_override_candidates.csv"


def normalize_place(series):
    """Normalize state, city, category, and county tokens."""
    return (series.fillna("").astype(str).str.upper()
            .str.replace(r"[^A-Z0-9]+", " ", regex=True).str.strip())


def normalize_county(series):
    """Normalize raw county FIPS to five digits."""
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    valid = text.str.fullmatch(r"\d{1,5}")
    return text.str.zfill(5).where(valid, "")


def source_cells(dcads_years, usa_years):
    """Read selected city-county-category cells from both sources."""
    import duckdb

    dcads_paths = [
        str(dc.OUT_DIR / f"fy{year}.parquet") for year in dcads_years]
    usa_paths = [
        str(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet") for year in usa_years]
    missing = [
        path for path in dcads_paths + usa_paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} source-audit files are missing")
    connection = duckdb.connect()
    dcads = connection.execute("""
        SELECT
            CAST(fiscal_year AS INTEGER) AS year,
            CAST(state_fips2 AS VARCHAR) AS state,
            CAST(city_name AS VARCHAR) AS city,
            CAST(county_fips5 AS VARCHAR) AS county,
            SUBSTR(UPPER(TRIM(CAST(fsc AS VARCHAR))), 1, 4) AS category,
            SUM(CASE WHEN dollars_unit = 'thousands'
                     THEN dollars * 1000.0 ELSE dollars END) AS dollars_nominal,
            COUNT(*) AS actions
        FROM read_parquet(?, union_by_name=true)
        WHERE geo_class = 'county'
          AND CASE WHEN dollars_unit = 'thousands'
                   THEN dollars * 1000.0 ELSE dollars END > 0
        GROUP BY ALL
    """, [dcads_paths]).df()
    usa = connection.execute("""
        SELECT
            CAST(source_fiscal_year AS INTEGER) AS year,
            CAST(prime_award_transaction_place_of_performance_state_fips_code
                 AS VARCHAR) AS state,
            CAST(primary_place_of_performance_city_name AS VARCHAR) AS city,
            CAST(prime_award_transaction_place_of_performance_county_fips_code
                 AS VARCHAR) AS county,
            SUBSTR(UPPER(TRIM(CAST(product_or_service_code AS VARCHAR))),
                   1, 4) AS category,
            SUM(federal_action_obligation) AS dollars_nominal,
            COUNT(*) AS actions
        FROM read_parquet(?, union_by_name=true)
        WHERE award_or_idv_flag = 'AWARD'
          AND geo_class = 'county'
          AND federal_action_obligation > 0
        GROUP BY ALL
    """, [usa_paths]).df()
    connection.close()
    parts = []
    for source, frame in (("DCADS", dcads), ("USAspending", usa)):
        frame["state"] = normalize_place(frame["state"])
        frame["city"] = normalize_place(frame["city"])
        frame["county"] = normalize_county(frame["county"])
        frame["category"] = normalize_place(frame["category"]).str[:4]
        frame["dollars_real"] = (
            frame["dollars_nominal"]
            * frame["year"].map(dc.deflate).astype(float))
        frame["source"] = source
        frame = frame.loc[
            frame["state"].ne("")
            & frame["city"].ne("")
            & frame["county"].ne("")]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def dominant_assignments(cells, category_level):
    """Compare dominant counties for common state-city keys."""
    keys = ["state", "city"] + (["category"] if category_level else [])
    grouped = (cells.groupby(keys + ["source", "county"], sort=False)
               .agg(dollars_real=("dollars_real", "sum"),
                    actions=("actions", "sum"))
               .reset_index())
    totals = (grouped.groupby(keys + ["source"], sort=False)
              .agg(total_dollars_real=("dollars_real", "sum"),
                   total_actions=("actions", "sum"),
                   counties=("county", "nunique"))
              .reset_index())
    ordered = grouped.sort_values(
        keys + ["source", "dollars_real", "county"],
        ascending=[True] * (len(keys) + 1) + [False, True])
    dominant = ordered.drop_duplicates(keys + ["source"])
    dominant = dominant.merge(totals, on=keys + ["source"], validate="one_to_one")
    dominant["dominant_share"] = (
        dominant["dollars_real"] / dominant["total_dollars_real"])
    values = [
        "county", "dollars_real", "dominant_share", "total_dollars_real",
        "total_actions", "counties",
    ]
    wide = dominant.pivot(index=keys, columns="source", values=values)
    wide.columns = [
        f"{source.lower()}_{value}" for value, source in wide.columns]
    wide = wide.reset_index()
    required = [
        "dcads_county", "usaspending_county",
        "dcads_dominant_share", "usaspending_dominant_share",
        "dcads_total_dollars_real", "usaspending_total_dollars_real",
    ]
    wide = wide.dropna(subset=required)
    wide["county_switch"] = wide["dcads_county"].ne(
        wide["usaspending_county"])
    wide["minimum_source_dollars_real"] = wide[[
        "dcads_total_dollars_real", "usaspending_total_dollars_real"]].min(
            axis=1)
    wide["strong_candidate"] = (
        wide["county_switch"]
        & wide["dcads_dominant_share"].ge(DOMINANCE_MIN)
        & wide["usaspending_dominant_share"].ge(DOMINANCE_MIN)
        & wide["minimum_source_dollars_real"].ge(DOLLARS_MIN)
    )
    return wide.sort_values(
        ["strong_candidate", "minimum_source_dollars_real"],
        ascending=[False, False]).reset_index(drop=True)


def main():
    """Run the city-county source audit."""
    start = time.time()
    print("== USAspending city-county source audit")
    city_parts, category_parts = [], []
    for comparison, (dcads_years, usa_years) in COMPARISONS.items():
        cells = source_cells(dcads_years, usa_years)
        city_part = dominant_assignments(cells, category_level=False)
        category_part = dominant_assignments(cells, category_level=True)
        city_part.insert(0, "comparison", comparison)
        category_part.insert(0, "comparison", comparison)
        city_parts.append(city_part)
        category_parts.append(category_part)
    city = pd.concat(city_parts, ignore_index=True)
    category = pd.concat(category_parts, ignore_index=True)
    candidates = city.loc[city["strong_candidate"]].copy()
    candidates.insert(
        1, "candidate_rank",
        candidates.groupby("comparison", sort=False).cumcount() + 1)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    city.to_csv(CITY_OUT, index=False)
    category.to_csv(CATEGORY_OUT, index=False)
    candidates.to_csv(CANDIDATE_OUT, index=False)

    seam_label = "seam_2004_2006_vs_2007_2009"
    sealy = city.loc[
        city["comparison"].eq(seam_label)
        & city["state"].eq("48")
        & city["city"].eq("SEALY")]
    if len(sealy) != 1:
        raise SystemExit(f"FAIL: city audit has {len(sealy)} Sealy rows")
    row = sealy.iloc[0]
    print(
        f"   state-city comparisons={len(city):,}; "
        f"county switches={int(city['county_switch'].sum()):,}; "
        f"strong candidates={len(candidates):,}")
    print(
        f"   Sealy: DCADS county={row['dcads_county']}, "
        f"share={100 * row['dcads_dominant_share']:.2f}%; "
        f"USAspending county={row['usaspending_county']}, "
        f"share={100 * row['usaspending_dominant_share']:.2f}%; "
        f"rank={int(candidates.loc[
            candidates['comparison'].eq(seam_label)
            & candidates['state'].eq('48')
            & candidates['city'].eq('SEALY'), 'candidate_rank'].iloc[0])}")
    print(f"   wrote {CITY_OUT}")
    print(f"   wrote {CATEGORY_OUT}")
    print(f"   wrote {CANDIDATE_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
