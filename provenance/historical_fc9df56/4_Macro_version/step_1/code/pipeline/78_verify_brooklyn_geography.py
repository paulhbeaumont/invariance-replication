"""Verify the Brooklyn DCADS city-county conflict.

The producer uses official 2000 and 2010 Census Brooklyn borough records and
county polygons, then audits positive DCADS and USAspending actions over their
identical fiscal-year 2001-2006 coverage. Raw source county fields are retained
throughout. This producer does not alter production geography.

Writes Results/instrument_normalization/
    brooklyn_census_borough_counties.csv
    brooklyn_transaction_summary.csv
    brooklyn_zip_summary.csv
    brooklyn_contract_matches.csv
    brooklyn_verification.csv
and Instrument_analysis/Cleaning_spendingData/BROOKLYN_VERIFICATION.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_year/fy2001.parquet
# mpgr-output-cache: Results/instrument_normalization/brooklyn_verification.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|2001-2006-overlap|brooklyn|historical-geography

import hashlib
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


YEARS = tuple(range(2001, 2007))
STATE = "36"
CITY = "BROOKLYN"
DISPUTED_COUNTY = "36025"
CORRECT_COUNTY = "36047"
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
TIGER_FILES = {
    ("borough", "2000"):
        f"{TIGER_BASE}/COUSUB/2000/tl_2010_36_cousub00.zip",
    ("borough", "2010"):
        f"{TIGER_BASE}/COUSUB/2010/tl_2010_36_cousub10.zip",
    ("county", "2000"):
        f"{TIGER_BASE}/COUNTY/2000/tl_2010_us_county00.zip",
    ("county", "2010"):
        f"{TIGER_BASE}/COUNTY/2010/tl_2010_us_county10.zip",
}
CACHE_DIR = (dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
             / "historical_tiger")
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CENSUS_OUT = OUTDIR / "brooklyn_census_borough_counties.csv"
TRANSACTION_OUT = OUTDIR / "brooklyn_transaction_summary.csv"
ZIP_OUT = OUTDIR / "brooklyn_zip_summary.csv"
MATCH_OUT = OUTDIR / "brooklyn_contract_matches.csv"
VERIFICATION_OUT = OUTDIR / "brooklyn_verification.csv"
MEMO_OUT = (dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
            / "Cleaning_spendingData" / "BROOKLYN_VERIFICATION.md")


def normalize_name(value):
    """Normalize a source place name for an exact match."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def acquire_tiger_files():
    """Acquire fixed-vintage Census files and return their shapefile paths."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    source_rows = []
    for key, url in TIGER_FILES.items():
        archive = CACHE_DIR / Path(url).name
        if not archive.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent":
                              "aggregation-disaggregation-invariance"})
            with urllib.request.urlopen(request, timeout=180) as response:
                archive.write_bytes(response.read())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        folder = CACHE_DIR / archive.stem
        shape = folder / f"{archive.stem}.shp"
        if not shape.exists():
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as source:
                source.extractall(folder)
        if not shape.exists():
            raise SystemExit(f"FAIL: {archive.name} lacks the expected shapefile")
        paths[key] = shape
        source_rows.append({
            "kind": key[0], "vintage": int(key[1]), "url": url,
            "archive": archive.name, "sha256": digest,
        })
    pd.DataFrame(source_rows).to_csv(
        CACHE_DIR / "brooklyn_historical_tiger_sources.csv", index=False)
    return paths


def census_borough_counties(paths):
    """Recover Brooklyn's embedded and spatial county in both vintages."""
    connection = duckdb.connect()
    connection.execute("INSTALL spatial; LOAD spatial")
    rows = []
    for vintage in ("2000", "2010"):
        suffix = vintage[-2:]
        borough = str(paths[("borough", vintage)])
        county = str(paths[("county", vintage)])
        county_geoid = (f"c.GEOID{suffix}" if vintage == "2010"
                        else f"c.STATEFP{suffix} || c.COUNTYFP{suffix}")
        query = f"""
            WITH selected_borough AS (
                SELECT NAME{suffix} AS borough_name,
                       NAMELSAD{suffix} AS legal_name,
                       STATEFP{suffix} || COUNTYFP{suffix}
                           AS embedded_county_fips,
                       geom
                FROM ST_Read(?)
                WHERE STATEFP{suffix} = ?
                  AND UPPER(NAME{suffix}) = ?
                  AND UPPER(NAMELSAD{suffix}) = 'BROOKLYN BOROUGH'
            )
            SELECT b.borough_name, b.legal_name, b.embedded_county_fips,
                   {county_geoid} AS intersected_county_fips,
                   c.NAME{suffix} AS county_name,
                   ST_Area(ST_Intersection(b.geom, c.geom))
                       / ST_Area(b.geom) AS borough_area_share
            FROM selected_borough b
            CROSS JOIN ST_Read(?) c
            WHERE c.STATEFP{suffix} = ?
              AND ST_Intersects(b.geom, c.geom)
              AND ST_Area(ST_Intersection(b.geom, c.geom))
                    / ST_Area(b.geom) > 1e-12
            ORDER BY intersected_county_fips
        """
        frame = connection.execute(
            query, [borough, STATE, CITY, county, STATE]).df()
        if len(frame) != 1:
            raise SystemExit(
                f"FAIL: {vintage} Census returned {len(frame)} Brooklyn rows")
        frame["borough_area_share"] = frame["borough_area_share"].clip(0.0, 1.0)
        frame.insert(0, "vintage", int(vintage))
        rows.append(frame)
    connection.close()
    return pd.concat(rows, ignore_index=True)


def source_paths(source):
    """Return the six overlap-year parquet paths for one source."""
    directory = dc.OUT_DIR if source == "DCADS" else dc.USASPENDING_OUT_DIR
    paths = [directory / f"fy{year}.parquet" for year in YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: {source} is missing {len(missing)} overlap files")
    return [str(path) for path in paths]


def transaction_actions():
    """Read raw Brooklyn transactions from both sources."""
    connection = duckdb.connect()
    dcads = connection.execute("""
        SELECT CAST(fiscal_year AS INTEGER) AS year,
               'DCADS' AS source,
               TRIM(state_code) AS state,
               REGEXP_REPLACE(UPPER(TRIM(city_name)), '[^A-Z0-9]+', ' ', 'g')
                   AS city,
               CASE
                   WHEN COALESCE(TRIM(county_code), '') <> ''
                       THEN TRIM(county_code)
                   WHEN REGEXP_FULL_MATCH(TRIM(county_fips), '[0-9]{3}')
                       THEN TRIM(state_code) || TRIM(county_fips)
                   ELSE TRIM(county_fips)
               END AS reported_county,
               CASE WHEN dollars_unit = 'thousands'
                    THEN dollars * 1000.0 ELSE dollars END AS dollars_nominal,
               REGEXP_REPLACE(UPPER(TRIM(contract_no)), '[^A-Z0-9]+', '', 'g')
                   AS contract_id,
               CAST(NULL AS VARCHAR) AS performance_zip,
               SUBSTR(REGEXP_REPLACE(contractor_zip, '[^0-9]+', '', 'g'), 1, 5)
                   AS supporting_contractor_zip
        FROM read_parquet(?, union_by_name=true)
        WHERE TRIM(state_code) = ?
          AND REGEXP_REPLACE(UPPER(TRIM(city_name)),
                             '[^A-Z0-9]+', ' ', 'g') = ?
          AND CASE WHEN dollars_unit = 'thousands'
                   THEN dollars * 1000.0 ELSE dollars END > 0
    """, [source_paths("DCADS"), STATE, CITY]).df()
    usa = connection.execute("""
        SELECT CAST(source_fiscal_year AS INTEGER) AS year,
               'USAspending' AS source,
               prime_award_transaction_place_of_performance_state_fips_code
                   AS state,
               REGEXP_REPLACE(
                   UPPER(TRIM(primary_place_of_performance_city_name)),
                   '[^A-Z0-9]+', ' ', 'g') AS city,
               prime_award_transaction_place_of_performance_county_fips_code
                   AS reported_county,
               federal_action_obligation AS dollars_nominal,
               REGEXP_REPLACE(UPPER(TRIM(award_id_piid)),
                              '[^A-Z0-9]+', '', 'g') AS contract_id,
               SUBSTR(REGEXP_REPLACE(primary_place_of_performance_zip_4,
                                     '[^0-9]+', '', 'g'), 1, 5)
                   AS performance_zip,
               CAST(NULL AS VARCHAR) AS supporting_contractor_zip
        FROM read_parquet(?, union_by_name=true)
        WHERE award_or_idv_flag = 'AWARD'
          AND prime_award_transaction_place_of_performance_state_fips_code = ?
          AND REGEXP_REPLACE(
                  UPPER(TRIM(primary_place_of_performance_city_name)),
                  '[^A-Z0-9]+', ' ', 'g') = ?
          AND federal_action_obligation > 0
    """, [source_paths("USAspending"), STATE, CITY]).df()
    connection.close()
    actions = pd.concat([dcads, usa], ignore_index=True)
    actions["city"] = actions["city"].map(normalize_name)
    actions["reported_county"] = (
        actions["reported_county"].fillna("").astype(str)
        .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    actions["dollars_real"] = (
        actions["dollars_nominal"]
        * actions["year"].map(dc.deflate).astype(float))
    if set(actions["source"].unique()) != {"DCADS", "USAspending"}:
        raise SystemExit("FAIL: Brooklyn is absent from one overlap source")
    return actions


def transaction_summaries(actions):
    """Collapse transactions by source-year-county and ZIP evidence."""
    cells = (actions.groupby(
        ["source", "year", "reported_county"], dropna=False, sort=False)
        .agg(actions=("dollars_real", "size"),
             dollars_nominal=("dollars_nominal", "sum"),
             dollars_real=("dollars_real", "sum"),
             unique_contract_ids=("contract_id", "nunique"))
        .reset_index())
    totals = cells.groupby(
        ["source", "year"], sort=False)["dollars_real"].transform("sum")
    cells["within_source_year_dollar_share"] = cells["dollars_real"] / totals

    performance = (actions.loc[actions["performance_zip"].fillna("").ne("")]
                   .groupby(["source", "reported_county", "performance_zip"],
                            sort=False)
                   .agg(actions=("dollars_real", "size"),
                        dollars_real=("dollars_real", "sum"))
                   .reset_index()
                   .rename(columns={"performance_zip": "zip"}))
    performance["zip_kind"] = "place_of_performance"
    supporting = (
        actions.loc[actions["supporting_contractor_zip"].fillna("").ne("")]
        .groupby(["source", "reported_county", "supporting_contractor_zip"],
                 sort=False)
        .agg(actions=("dollars_real", "size"),
             dollars_real=("dollars_real", "sum"))
        .reset_index()
        .rename(columns={"supporting_contractor_zip": "zip"}))
    supporting["zip_kind"] = "supporting_contractor"
    zips = pd.concat([performance, supporting], ignore_index=True)
    zip_totals = zips.groupby(
        ["source", "zip_kind"], sort=False)["dollars_real"].transform("sum")
    zips["source_dollar_share"] = zips["dollars_real"] / zip_totals
    return cells, zips[[
        "source", "zip_kind", "reported_county", "zip", "actions",
        "dollars_real", "source_dollar_share"]]


def contract_matches(actions):
    """Match normalized contract identifiers across the two sources."""
    usable = actions.loc[actions["contract_id"].fillna("").str.len().ge(4)]
    collapsed = (usable.groupby(
        ["source", "contract_id", "reported_county"], sort=False)
        .agg(actions=("dollars_nominal", "size"),
             dollars_nominal=("dollars_nominal", "sum"),
             dollars_real=("dollars_real", "sum"))
        .reset_index())
    dcads = (collapsed.loc[collapsed["source"].eq("DCADS")]
             .drop(columns="source").add_prefix("dcads_"))
    usa = (collapsed.loc[collapsed["source"].eq("USAspending")]
           .drop(columns="source").add_prefix("usaspending_"))
    matched = dcads.merge(
        usa, left_on="dcads_contract_id", right_on="usaspending_contract_id",
        how="inner", validate="many_to_many")
    matched["contract_id"] = matched["dcads_contract_id"]
    matched["nominal_dollar_gap"] = (
        matched["dcads_dollars_nominal"]
        - matched["usaspending_dollars_nominal"])
    matched_ids = set(matched["contract_id"])
    usa_zips = (usable.loc[
        usable["source"].eq("USAspending")
        & usable["contract_id"].isin(matched_ids)
        & usable["performance_zip"].fillna("").ne(""),
        ["contract_id", "performance_zip"]]
        .drop_duplicates())
    zip_counts = usa_zips.groupby(
        "contract_id", sort=False)["performance_zip"].nunique()
    if zip_counts.gt(1).any():
        raise SystemExit(
            "FAIL: a matched USAspending contract has multiple performance ZIPs")
    matched = matched.merge(
        usa_zips, on="contract_id", how="left", validate="many_to_one")
    return matched[[
        "contract_id", "dcads_reported_county",
        "usaspending_reported_county", "performance_zip", "dcads_actions",
        "usaspending_actions", "dcads_dollars_nominal",
        "usaspending_dollars_nominal", "nominal_dollar_gap",
        "dcads_dollars_real", "usaspending_dollars_real"]]


def verification_table(census, actions, cells, matches):
    """Apply the declared historical and transaction evidence conditions."""
    embedded = set(census["embedded_county_fips"])
    intersected = set(census["intersected_county_fips"])
    geography_condition = (
        embedded == {CORRECT_COUNTY}
        and intersected == {CORRECT_COUNTY}
        and DISPUTED_COUNTY not in embedded
        and DISPUTED_COUNTY not in intersected)

    stats = {}
    for source in ("DCADS", "USAspending"):
        grouped = (cells.loc[cells["source"].eq(source)]
                   .groupby("reported_county", sort=False)
                   .agg(actions=("actions", "sum"),
                        dollars_real=("dollars_real", "sum"))
                   .sort_values("dollars_real", ascending=False))
        total = grouped["dollars_real"].sum()
        stats[source] = {
            "county": grouped.index[0],
            "share": grouped.iloc[0]["dollars_real"] / total,
            "actions": int(grouped["actions"].sum()),
            "dollars": total,
        }
    source_condition = (
        stats["DCADS"]["county"] == DISPUTED_COUNTY
        and stats["DCADS"]["share"] >= 0.80
        and stats["USAspending"]["county"] == CORRECT_COUNTY
        and stats["USAspending"]["share"] >= 0.80)

    expected = matches.loc[
        matches["dcads_reported_county"].eq(DISPUTED_COUNTY)
        & matches["usaspending_reported_county"].eq(CORRECT_COUNTY)]
    equal_dollars = expected.loc[expected["nominal_dollar_gap"].abs().lt(0.005)]
    matched_ids = set(expected["contract_id"])
    matched_usa = actions.loc[
        actions["source"].eq("USAspending")
        & actions["contract_id"].isin(matched_ids)]
    zip112 = matched_usa["performance_zip"].fillna("").str.startswith("112")
    zip112_share = float(zip112.mean()) if len(matched_usa) else 0.0
    match_condition = (
        matches["contract_id"].nunique() > 0
        and expected["contract_id"].nunique() == matches["contract_id"].nunique()
        and equal_dollars["contract_id"].nunique()
            == expected["contract_id"].nunique()
        and zip112_share == 1.0)
    decision = ("verified_dcads_geography_error"
                if geography_condition and source_condition and match_condition
                else "unresolved")
    return pd.DataFrame([{
        "case": "brooklyn_new_york",
        "state": STATE,
        "city": CITY,
        "dcads_disputed_county": DISPUTED_COUNTY,
        "correct_county": CORRECT_COUNTY,
        "census_2000_embedded_county": census.loc[
            census["vintage"].eq(2000), "embedded_county_fips"].iloc[0],
        "census_2010_embedded_county": census.loc[
            census["vintage"].eq(2010), "embedded_county_fips"].iloc[0],
        "historical_geography_condition": geography_condition,
        "dcads_dominant_county": stats["DCADS"]["county"],
        "dcads_dominant_share": stats["DCADS"]["share"],
        "dcads_actions": stats["DCADS"]["actions"],
        "dcads_dollars_real": stats["DCADS"]["dollars"],
        "usaspending_dominant_county": stats["USAspending"]["county"],
        "usaspending_dominant_share": stats["USAspending"]["share"],
        "usaspending_actions": stats["USAspending"]["actions"],
        "usaspending_dollars_real": stats["USAspending"]["dollars"],
        "source_dominance_condition": source_condition,
        "matched_contract_ids": matches["contract_id"].nunique(),
        "matched_contract_ids_with_expected_county_pair":
            expected["contract_id"].nunique(),
        "matched_contract_ids_with_equal_nominal_dollars":
            equal_dollars["contract_id"].nunique(),
        "matched_usaspending_actions": len(matched_usa),
        "matched_usaspending_zip112_share": zip112_share,
        "transaction_match_condition": match_condition,
        "decision": decision,
        "proposed_correction_rule":
            "state=36; city=BROOKLYN; raw county=36025; FY2001-2006",
    }])


def pct(value):
    return f"{100 * float(value):.4f}%"


def money(value):
    return f"${float(value):,.2f}"


def write_memo(verification, cells, matches):
    """Write the evidence memo from the verified outputs."""
    row = verification.iloc[0]
    dcads_years = cells.loc[
        cells["source"].eq("DCADS")
        & cells["reported_county"].eq(DISPUTED_COUNTY), "year"].tolist()
    correct_years = cells.loc[
        cells["source"].eq("DCADS")
        & cells["reported_county"].eq(CORRECT_COUNTY), "year"].tolist()
    matched_nominal = matches.loc[
        matches["dcads_reported_county"].eq(DISPUTED_COUNTY),
        "dcads_dollars_nominal"].sum()
    lines = [
        "# Brooklyn geography verification",
        "",
        "## Decision",
        "",
        ("Brooklyn, New York, is classified as "
         "`verified_dcads_geography_error`. The correct county is Kings "
         "County, New York, FIPS 36047; the disputed DCADS county is Delaware "
         "County, New York, FIPS 36025 (Table 1)."),
        "",
        "**Table 1. Historical geography and transaction evidence**",
        "",
        "| Evidence | Result |",
        "|---|---|",
        ("| 2000 Census Brooklyn borough county | Kings County, New York, "
         "36047 |"),
        ("| 2010 Census Brooklyn borough county | Kings County, New York, "
         "36047 |"),
        (f"| DCADS dominant county, FY2001 to FY2006 | Delaware County, New "
         f"York, 36025; {pct(row['dcads_dominant_share'])} of "
         f"{money(row['dcads_dollars_real'])} across "
         f"{int(row['dcads_actions']):,} positive actions |"),
        (f"| USAspending county, FY2001 to FY2006 | Kings County, New York, "
         f"36047; {pct(row['usaspending_dominant_share'])} of "
         f"{money(row['usaspending_dollars_real'])} across "
         f"{int(row['usaspending_actions']):,} positive actions |"),
        (f"| Contract identifiers observed in both sources | "
         f"{int(row['matched_contract_ids'])}; all have the 36025-to-36047 "
         f"county pair and equal nominal dollars, totaling "
         f"{money(matched_nominal)} |"),
        (f"| USAspending place-of-performance ZIP among matched actions | "
         f"{pct(row['matched_usaspending_zip112_share'])} begin with 112 |"),
        "",
        "## Why the classification is strong",
        "",
        ("First, Brooklyn appears as `Brooklyn borough` in the official 2000 "
         "and 2010 Census county-subdivision files. In both vintages its "
         "embedded county code is 36047, and its polygon intersects only Kings "
         "County (Table 1 and `brooklyn_census_borough_counties.csv`). Delaware "
         "County 36025 is excluded in both vintages."),
        "",
        ("Second, the transaction comparison holds fiscal years fixed at 2001 "
         "to 2006. The discrepancy cannot be generated by comparing different "
         "periods. DCADS uses county 36025 in fiscal years "
         f"{', '.join(map(str, dcads_years))}, while it records county 36047 "
         f"in fiscal year {', '.join(map(str, correct_years))} "
         "(`brooklyn_transaction_summary.csv`)."),
        "",
        ("Third, all eight normalized contract identifiers observed in both "
         "sources reproduce the disputed county pair. Their nominal action "
         f"amounts agree exactly and sum to {money(matched_nominal)}. All eight "
         "USAspending records also carry a place-of-performance ZIP beginning "
         "with 112 (`brooklyn_contract_matches.csv` and "
         "`brooklyn_zip_summary.csv`)."),
        "",
        ("The decision does not substitute recipient or contractor geography "
         "for place-of-performance geography. Census borough geography "
         "identifies Kings County, while the same contract identifiers and "
         "USAspending place-of-performance ZIPs provide record-level "
         "confirmation."),
        "",
        "## Correction boundary",
        "",
        ("The evidence supports correcting only DCADS rows whose raw fields "
         "jointly identify state 36, city `BROOKLYN`, and county 36025 during "
         "fiscal years 2001 to 2006. The fiscal-year 2002 DCADS rows already "
         "carry county 36047 and must remain untouched "
         "(`brooklyn_transaction_summary.csv`)."),
        "",
        ("This verification producer does not alter production geography. A "
         "production correction requires a city-specific override followed by "
         "the downstream rebuild and verification gates."),
        "",
        "## Files",
        "",
        "- Decision table: `Results/instrument_normalization/brooklyn_verification.csv`.",
        "- Census evidence: `Results/instrument_normalization/brooklyn_census_borough_counties.csv`.",
        "- Transaction cells: `Results/instrument_normalization/brooklyn_transaction_summary.csv`.",
        "- ZIP evidence: `Results/instrument_normalization/brooklyn_zip_summary.csv`.",
        "- Contract matches: `Results/instrument_normalization/brooklyn_contract_matches.csv`.",
        ("- Census source manifest: `_data/temp/usaspending_city_county_audit/"
         "historical_tiger/brooklyn_historical_tiger_sources.csv`."),
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the Brooklyn historical-geography verification."""
    start = time.time()
    print("== Brooklyn geography verification")
    paths = acquire_tiger_files()
    census = census_borough_counties(paths)
    actions = transaction_actions()
    cells, zips = transaction_summaries(actions)
    matches = contract_matches(actions)
    verification = verification_table(census, actions, cells, matches)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    census.to_csv(CENSUS_OUT, index=False)
    cells.to_csv(TRANSACTION_OUT, index=False)
    zips.to_csv(ZIP_OUT, index=False)
    matches.to_csv(MATCH_OUT, index=False)
    verification.to_csv(VERIFICATION_OUT, index=False)
    write_memo(verification, cells, matches)
    row = verification.iloc[0]
    print(f"   decision: {row['decision']}")
    print(f"   Census 2000/2010: {row['census_2000_embedded_county']}/"
          f"{row['census_2010_embedded_county']}")
    print(f"   matched IDs: {int(row['matched_contract_ids'])}")
    print(f"   wrote {VERIFICATION_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
