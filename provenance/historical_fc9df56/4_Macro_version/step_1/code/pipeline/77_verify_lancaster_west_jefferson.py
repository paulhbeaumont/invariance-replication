"""Verify the Lancaster and West Jefferson city-county source conflicts.

The producer intersects official 2000 and 2010 Census place polygons with the
matching county polygons, then audits positive DCADS and USAspending contract
actions over their identical fiscal-year 2001-2006 coverage. Raw source fields
are used throughout. This producer does not alter production geography.

Writes Results/instrument_normalization/
    lancaster_west_jefferson_census_counties.csv
    lancaster_west_jefferson_transaction_summary.csv
    lancaster_west_jefferson_zip_summary.csv
    lancaster_west_jefferson_contract_matches.csv
    lancaster_west_jefferson_verification.csv
and Instrument_analysis/Cleaning_spendingData/
    LANCASTER_WEST_JEFFERSON_VERIFICATION.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_year/fy2001.parquet
# mpgr-output-cache: Results/instrument_normalization/lancaster_west_jefferson_verification.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|2001-2006-overlap|lancaster-west-jefferson|historical-geography

import hashlib
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


YEARS = tuple(range(2001, 2007))
CASES = (
    {"case": "lancaster_california", "state": "06", "city": "LANCASTER",
     "dcads_county": "06005", "usaspending_county": "06037"},
    {"case": "west_jefferson_ohio", "state": "39",
     "city": "WEST JEFFERSON", "dcads_county": "39171",
     "usaspending_county": "39097"},
)
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
TIGER_FILES = {
    ("place", "2000", "06"):
        f"{TIGER_BASE}/PLACE/2000/tl_2010_06_place00.zip",
    ("place", "2000", "39"):
        f"{TIGER_BASE}/PLACE/2000/tl_2010_39_place00.zip",
    ("place", "2010", "06"):
        f"{TIGER_BASE}/PLACE/2010/tl_2010_06_place10.zip",
    ("place", "2010", "39"):
        f"{TIGER_BASE}/PLACE/2010/tl_2010_39_place10.zip",
    ("county", "2000", "us"):
        f"{TIGER_BASE}/COUNTY/2000/tl_2010_us_county00.zip",
    ("county", "2010", "us"):
        f"{TIGER_BASE}/COUNTY/2010/tl_2010_us_county10.zip",
}
CACHE_DIR = (dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
             / "historical_tiger")
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CENSUS_OUT = OUTDIR / "lancaster_west_jefferson_census_counties.csv"
TRANSACTION_OUT = OUTDIR / "lancaster_west_jefferson_transaction_summary.csv"
ZIP_OUT = OUTDIR / "lancaster_west_jefferson_zip_summary.csv"
MATCH_OUT = OUTDIR / "lancaster_west_jefferson_contract_matches.csv"
VERIFICATION_OUT = OUTDIR / "lancaster_west_jefferson_verification.csv"
MEMO_OUT = (dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
            / "Cleaning_spendingData"
            / "LANCASTER_WEST_JEFFERSON_VERIFICATION.md")


def normalize_name(value):
    """Normalize a source place name for an exact case match."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def download_and_extract():
    """Acquire the fixed-vintage Census shapefiles and return their paths."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
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
        shp = folder / f"{archive.stem}.shp"
        if not shp.exists():
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as source:
                source.extractall(folder)
        if not shp.exists():
            raise SystemExit(f"FAIL: {archive.name} contains no expected shapefile")
        result[key] = shp
        source_rows.append({
            "kind": key[0], "vintage": key[1], "state": key[2],
            "url": url, "archive": archive.name, "sha256": digest,
        })
    pd.DataFrame(source_rows).to_csv(
        CACHE_DIR / "historical_tiger_sources.csv", index=False)
    return result


def census_intersections(paths):
    """Intersect each named place with counties in 2000 and 2010 geography."""
    connection = duckdb.connect()
    connection.execute("INSTALL spatial; LOAD spatial")
    rows = []
    for case in CASES:
        for vintage in ("2000", "2010"):
            suffix = vintage[-2:]
            county_geoid = (f"c.GEOID{suffix}" if vintage == "2010"
                            else f"c.STATEFP{suffix} || c.COUNTYFP{suffix}")
            place = str(paths[("place", vintage, case["state"])])
            county = str(paths[("county", vintage, "us")])
            query = f"""
                WITH selected_place AS (
                    SELECT NAME{suffix} AS place_name, geom
                    FROM ST_Read(?)
                    WHERE STATEFP{suffix} = ?
                      AND REGEXP_REPLACE(UPPER(NAME{suffix}),
                          ' (CITY|VILLAGE|TOWN|CDP)$', '') = ?
                ), intersections AS (
                    SELECT p.place_name,
                           {county_geoid} AS county_fips,
                           c.NAME{suffix} AS county_name,
                           ST_Area(ST_Intersection(p.geom, c.geom))
                               / ST_Area(p.geom) AS place_area_share
                    FROM selected_place p
                    CROSS JOIN ST_Read(?) c
                    WHERE c.STATEFP{suffix} = ?
                      AND ST_Intersects(p.geom, c.geom)
                )
                SELECT * FROM intersections
                WHERE place_area_share > 1e-12
                ORDER BY county_fips
            """
            frame = connection.execute(
                query, [place, case["state"], case["city"], county,
                        case["state"]]).df()
            if len(frame) == 0:
                raise SystemExit(
                    f"FAIL: no {vintage} Census place match for {case['case']}")
            frame.insert(0, "vintage", int(vintage))
            frame.insert(0, "case", case["case"])
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


def transaction_cells():
    """Read raw place, ZIP, and contract identifiers over the common years."""
    connection = duckdb.connect()
    dcads = connection.execute("""
        SELECT CAST(fiscal_year AS INTEGER) AS year,
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
               contractor_name,
               contractor_city,
               contractor_state,
               SUBSTR(REGEXP_REPLACE(contractor_zip, '[^0-9]+', '', 'g'), 1, 5)
                   AS support_zip
        FROM read_parquet(?, union_by_name=true)
        WHERE geo_class = 'county'
          AND CASE WHEN dollars_unit = 'thousands'
                   THEN dollars * 1000.0 ELSE dollars END > 0
    """, [source_paths("DCADS")]).df()
    usa = connection.execute("""
        SELECT CAST(source_fiscal_year AS INTEGER) AS year,
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
               CAST(NULL AS VARCHAR) AS contractor_name,
               CAST(NULL AS VARCHAR) AS contractor_city,
               CAST(NULL AS VARCHAR) AS contractor_state,
               SUBSTR(REGEXP_REPLACE(primary_place_of_performance_zip_4,
                                     '[^0-9]+', '', 'g'), 1, 5) AS support_zip
        FROM read_parquet(?, union_by_name=true)
        WHERE award_or_idv_flag = 'AWARD'
          AND geo_class = 'county'
          AND federal_action_obligation > 0
    """, [source_paths("USAspending")]).df()
    connection.close()
    parts = []
    for source, frame in (("DCADS", dcads), ("USAspending", usa)):
        frame["source"] = source
        frame["state"] = frame["state"].fillna("").astype(str).str.zfill(2)
        frame["city"] = frame["city"].map(normalize_name)
        frame["reported_county"] = (
            frame["reported_county"].fillna("").astype(str)
            .str.replace(r"\.0$", "", regex=True).str.zfill(5))
        frame["dollars_real"] = (
            frame["dollars_nominal"]
            * frame["year"].map(dc.deflate).astype(float))
        parts.append(frame)
    all_rows = pd.concat(parts, ignore_index=True)
    case_keys = pd.DataFrame(CASES)[["case", "state", "city"]]
    selected = all_rows.merge(case_keys, on=["state", "city"], how="inner",
                              validate="many_to_one")
    if set(selected["case"].unique()) != {case["case"] for case in CASES}:
        raise SystemExit("FAIL: transaction audit is missing a requested case")
    return selected


def transaction_summaries(actions):
    """Collapse raw transactions by source-year-county and supporting ZIP."""
    cells = (actions.groupby(
        ["case", "source", "year", "reported_county"], dropna=False)
        .agg(actions=("dollars_real", "size"),
             dollars_real=("dollars_real", "sum"),
             unique_contract_ids=("contract_id", "nunique"))
        .reset_index())
    totals = cells.groupby(["case", "source", "year"])[
        "dollars_real"].transform("sum")
    cells["dollar_share"] = cells["dollars_real"] / totals

    zips = (actions.assign(support_zip=actions["support_zip"].fillna(""))
            .groupby(["case", "source", "reported_county", "support_zip"],
                     dropna=False)
            .agg(actions=("dollars_real", "size"),
                 dollars_real=("dollars_real", "sum"))
            .reset_index())
    total_zip = zips.groupby(["case", "source"])["dollars_real"].transform("sum")
    zips["source_dollar_share"] = zips["dollars_real"] / total_zip
    return cells, zips.sort_values(
        ["case", "source", "dollars_real"], ascending=[True, True, False])


def contract_matches(actions):
    """Match normalized contract identifiers across the two overlap sources."""
    usable = actions.loc[actions["contract_id"].fillna("").str.len().ge(4)]
    collapsed = (usable.groupby(
        ["case", "source", "contract_id", "reported_county"], dropna=False)
        .agg(actions=("dollars_real", "size"),
             dollars_real=("dollars_real", "sum"))
        .reset_index())
    dcads = collapsed.loc[collapsed["source"].eq("DCADS")].drop(
        columns="source").add_prefix("dcads_")
    usa = collapsed.loc[collapsed["source"].eq("USAspending")].drop(
        columns="source").add_prefix("usaspending_")
    matched = dcads.merge(
        usa, left_on=["dcads_case", "dcads_contract_id"],
        right_on=["usaspending_case", "usaspending_contract_id"],
        how="inner", validate="many_to_many")
    matched["case"] = matched["dcads_case"]
    matched["contract_id"] = matched["dcads_contract_id"]
    return matched[[
        "case", "contract_id", "dcads_reported_county",
        "usaspending_reported_county", "dcads_actions", "usaspending_actions",
        "dcads_dollars_real", "usaspending_dollars_real"]]


def verification_table(census, cells, matches):
    """Apply the predeclared two-source verification conditions."""
    rows = []
    for case in CASES:
        subc = census.loc[census["case"].eq(case["case"])]
        county_by_vintage = {
            vintage: set(subc.loc[subc["vintage"].eq(vintage), "county_fips"])
            for vintage in (2000, 2010)
        }
        sub = cells.loc[cells["case"].eq(case["case"])]
        source_stats = {}
        for source in ("DCADS", "USAspending"):
            source_cells = (sub.loc[sub["source"].eq(source)]
                            .groupby("reported_county")
                            .agg(actions=("actions", "sum"),
                                 dollars_real=("dollars_real", "sum"))
                            .sort_values("dollars_real", ascending=False))
            total = source_cells["dollars_real"].sum()
            dominant = source_cells.iloc[0]
            source_stats[source] = {
                "county": source_cells.index[0],
                "share": dominant["dollars_real"] / total,
                "actions": int(source_cells["actions"].sum()),
                "dollars": total,
            }
        subm = matches.loc[matches["case"].eq(case["case"])]
        expected_matches = subm.loc[
            subm["dcads_reported_county"].eq(case["dcads_county"])
            & subm["usaspending_reported_county"].eq(
                case["usaspending_county"])]
        geography_condition = all(
            case["usaspending_county"] in county_by_vintage[vintage]
            and case["dcads_county"] not in county_by_vintage[vintage]
            for vintage in (2000, 2010))
        source_condition = (
            source_stats["DCADS"]["county"] == case["dcads_county"]
            and source_stats["USAspending"]["county"]
                == case["usaspending_county"]
            and source_stats["DCADS"]["share"] >= 0.80
            and source_stats["USAspending"]["share"] >= 0.80)
        decision = ("verified_dcads_geography_error"
                    if geography_condition and source_condition
                    else "unresolved")
        rows.append({
            **case,
            "census_2000_counties": ";".join(sorted(county_by_vintage[2000])),
            "census_2010_counties": ";".join(sorted(county_by_vintage[2010])),
            "historical_geography_condition": geography_condition,
            "dcads_dominant_county": source_stats["DCADS"]["county"],
            "dcads_dominant_share": source_stats["DCADS"]["share"],
            "dcads_actions": source_stats["DCADS"]["actions"],
            "dcads_dollars_real": source_stats["DCADS"]["dollars"],
            "usaspending_dominant_county":
                source_stats["USAspending"]["county"],
            "usaspending_dominant_share": source_stats["USAspending"]["share"],
            "usaspending_actions": source_stats["USAspending"]["actions"],
            "usaspending_dollars_real": source_stats["USAspending"]["dollars"],
            "source_dominance_condition": source_condition,
            "matched_contract_ids": int(subm["contract_id"].nunique()),
            "matched_contract_ids_with_expected_county_pair":
                int(expected_matches["contract_id"].nunique()),
            "decision": decision,
            "proposed_corrected_county": (
                case["usaspending_county"] if decision.startswith("verified")
                else ""),
        })
    return pd.DataFrame(rows)


def pct(value):
    return f"{100 * float(value):.4f}%"


def money(value):
    return f"${float(value):,.2f}"


def noun(count, singular, plural=None):
    """Return a count-sensitive noun."""
    return singular if int(count) == 1 else (plural or f"{singular}s")


def write_memo(verification):
    """Write a concise evidence memo from the verified output table."""
    lines = [
        "# Lancaster and West Jefferson geography verification",
        "",
        "## Method",
        "",
        ("The verification combines official 2000 and 2010 Census place-county "
         "intersections with raw DCADS and USAspending contract actions over "
         "their identical fiscal-year 2001 to 2006 coverage. Table 1 reports "
         "the decision inputs. Recipient geography is not used to replace "
         "place-of-performance geography."),
        "",
        "**Table 1. Historical geography and source evidence**",
        "",
        ("| Location | Census counties, 2000 | Census counties, 2010 | DCADS "
         "county and share | USAspending county and share | Matched contract "
         "IDs | Decision |"),
        "|---|---|---|---|---|---:|---|",
    ]
    for row in verification.itertuples(index=False):
        location = row.case.replace("_", " ").title()
        lines.append(
            f"| {location} | {row.census_2000_counties} | "
            f"{row.census_2010_counties} | {row.dcads_dominant_county}, "
            f"{pct(row.dcads_dominant_share)} | "
            f"{row.usaspending_dominant_county}, "
            f"{pct(row.usaspending_dominant_share)} | "
            f"{row.matched_contract_ids} | `{row.decision}` |")
    lines += ["", "## Case evidence", ""]
    for row in verification.itertuples(index=False):
        location = row.case.replace("_", " ").title()
        lines += [
            f"### {location}",
            "",
            (f"The 2000 Census place intersects {row.census_2000_counties}; "
             f"the 2010 Census place intersects "
             f"{row.census_2010_counties} (Table 1). DCADS assigns "
             f"{pct(row.dcads_dominant_share)} of the case's positive real "
             f"procurement dollars to {row.dcads_dominant_county} across "
             f"{row.dcads_actions:,} "
             f"{noun(row.dcads_actions, 'action')} totaling "
             f"{money(row.dcads_dollars_real)}. USAspending assigns "
             f"{pct(row.usaspending_dominant_share)} to "
             f"{row.usaspending_dominant_county} across "
             f"{row.usaspending_actions:,} "
             f"{noun(row.usaspending_actions, 'action')} totaling "
             f"{money(row.usaspending_dollars_real)} (Table 1 and the "
             "transaction summary)."),
            "",
            (f"The overlap contains {row.matched_contract_ids} normalized "
             f"contract {noun(row.matched_contract_ids, 'identifier')} "
             "present in both sources, of which "
             f"{row.matched_contract_ids_with_expected_county_pair} "
             f"{'carries' if row.matched_contract_ids_with_expected_county_pair == 1 else 'carry'} the "
             "disputed DCADS county and the proposed USAspending county "
             "(Table 1 and the contract-match output)."),
            "",
        ]
    lines += [
        "## Decision boundary",
        "",
        ("A `verified_dcads_geography_error` decision requires the "
         "USAspending county to intersect the named place in both Census "
         "vintages, the DCADS county to intersect it in neither vintage, and "
         "each source to assign at least 80% of its case dollars to the "
         "reported dominant county. The transaction and contract-match files "
         "remain available for record-level inspection."),
        "",
        "This verification producer does not alter production geography.",
        "",
        "## Files",
        "",
        ("- Decision table: `Results/instrument_normalization/"
         "lancaster_west_jefferson_verification.csv`."),
        ("- Census intersections: `Results/instrument_normalization/"
         "lancaster_west_jefferson_census_counties.csv`."),
        ("- Transaction cells: `Results/instrument_normalization/"
         "lancaster_west_jefferson_transaction_summary.csv`."),
        ("- Supporting ZIP cells: `Results/instrument_normalization/"
         "lancaster_west_jefferson_zip_summary.csv`."),
        ("- Contract matches: `Results/instrument_normalization/"
         "lancaster_west_jefferson_contract_matches.csv`."),
        ("- Census archive manifest with SHA-256 hashes: "
         "`_data/temp/usaspending_city_county_audit/historical_tiger/"
         "historical_tiger_sources.csv`."),
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the two-case historical-geography verification."""
    start = time.time()
    print("== Lancaster and West Jefferson geography verification")
    paths = download_and_extract()
    census = census_intersections(paths)
    actions = transaction_cells()
    cells, zips = transaction_summaries(actions)
    matches = contract_matches(actions)
    verification = verification_table(census, cells, matches)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    census.to_csv(CENSUS_OUT, index=False)
    cells.to_csv(TRANSACTION_OUT, index=False)
    zips.to_csv(ZIP_OUT, index=False)
    matches.to_csv(MATCH_OUT, index=False)
    verification.to_csv(VERIFICATION_OUT, index=False)
    write_memo(verification)
    for row in verification.itertuples(index=False):
        print(f"   {row.case}: {row.decision}; Census 2000="
              f"{row.census_2000_counties}; Census 2010="
              f"{row.census_2010_counties}; matched IDs="
              f"{row.matched_contract_ids}")
    print(f"   wrote {VERIFICATION_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
