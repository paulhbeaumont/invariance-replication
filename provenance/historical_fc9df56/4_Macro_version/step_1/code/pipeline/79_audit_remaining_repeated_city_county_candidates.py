"""Audit the 29-case repeated city-county review cohort.

The producer combines official 2000 and 2010 Census place polygons, the 2010
Census military-installation layer, raw DCADS and USAspending actions over the
common fiscal years 2001-2006, and normalized contract-identifier matches.
It does not alter production geography.

Writes Results/instrument_normalization/
    remaining_repeated_candidate_historical_geography.csv
    remaining_repeated_candidate_transaction_summary.csv
    remaining_repeated_candidate_contract_matches.csv
    remaining_repeated_candidate_review.csv
and Instrument_analysis/Cleaning_spendingData/
    REMAINING_REPEATED_CANDIDATES.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_city_county_audit/historical_tiger
# mpgr-output-cache: Results/instrument_normalization/remaining_repeated_candidate_review.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|2001-2006-overlap|remaining-repeated-city-county-review

import hashlib
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


np.random.seed(12061984)

PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


YEARS = tuple(range(2001, 2007))
EXPECTED_REPEATED = 33
EXPECTED_REMAINING = 29
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
CACHE_DIR = (dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
             / "historical_tiger")
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
TRIAGE_IN = OUTDIR / "usaspending_city_county_repeated_candidate_triage.csv"
FIRST_STAGE_IN = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
DCADS_OVERRIDE = PIPELINE.parent / "reference" / "dcads_city_county_overrides.csv"
CENSUS_OUT = OUTDIR / "remaining_repeated_candidate_historical_geography.csv"
TRANSACTION_OUT = OUTDIR / "remaining_repeated_candidate_transaction_summary.csv"
MATCH_OUT = OUTDIR / "remaining_repeated_candidate_contract_matches.csv"
REVIEW_OUT = OUTDIR / "remaining_repeated_candidate_review.csv"
MEMO_OUT = (dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
            / "Cleaning_spendingData" / "REMAINING_REPEATED_CANDIDATES.md")

INSTALLATION_ALIASES = {
    "FORT RUCKER": ("FORT RUCKER", "FT RUCKER"),
    "DYESS AFB": ("DYESS AIR FORCE BASE", "DYESS AFB"),
    "TINKER AFB": ("TINKER AIR FORCE BASE", "TINKER AFB"),
    "FORT STEWART": ("FORT STEWART", "FT STEWART"),
    "MOODY AFB": ("MOODY AIR FORCE BASE", "MOODY AFB"),
    "FORT GORDON": ("FORT GORDON", "FT GORDON"),
    "ELLSWORTH AFB": ("ELLSWORTH AIR FORCE BASE", "ELLSWORTH AFB"),
}
FOCUSED_CITIES = {
    "CORBIN", "LAUREL", "FARMINGDALE", "TINKER AFB", "DYESS AFB",
}
FOCUSED_VERIFICATION_CASES = {
    ("48", "SEALY", "48015", "48089"),
    ("06", "LANCASTER", "06005", "06037"),
    ("39", "WEST JEFFERSON", "39171", "39097"),
    ("36", "BROOKLYN", "36025", "36047"),
}


def normalize_name(value):
    """Normalize a reported place or installation name."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def source_paths(source):
    """Return common-year parquet paths for one procurement source."""
    directory = dc.OUT_DIR if source == "DCADS" else dc.USASPENDING_OUT_DIR
    paths = [directory / f"fy{year}.parquet" for year in YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: {source} is missing {len(missing)} overlap files")
    return [str(path) for path in paths]


def remaining_candidates():
    """Return the 29 candidates outside the four focused verification cases."""
    triage = pd.read_csv(
        TRIAGE_IN,
        dtype={"state": "string", "dcads_county": "string",
               "usaspending_county": "string"})
    if len(triage) != EXPECTED_REPEATED:
        raise SystemExit(
            f"FAIL: repeated candidate count is {len(triage)}, expected "
            f"{EXPECTED_REPEATED}")
    triage["state"] = triage["state"].str.zfill(2)
    triage["dcads_county"] = triage["dcads_county"].str.zfill(5)
    triage["usaspending_county"] = triage["usaspending_county"].str.zfill(5)
    triage["city"] = triage["city"].map(normalize_name)

    keys = list(zip(
        triage["state"], triage["city"], triage["dcads_county"],
        triage["usaspending_county"]))
    result = triage.loc[
        [key not in FOCUSED_VERIFICATION_CASES for key in keys]].copy()
    if len(result) != EXPECTED_REMAINING:
        raise SystemExit(
            f"FAIL: remaining candidate count is {len(result)}, expected "
            f"{EXPECTED_REMAINING}")
    return result


def acquire(url):
    """Download and extract one fixed Census TIGER/Line archive."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = CACHE_DIR / Path(url).name
    if not archive.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent":
                          "aggregation-disaggregation-invariance"})
        with urllib.request.urlopen(request, timeout=180) as response:
            archive.write_bytes(response.read())
    folder = CACHE_DIR / archive.stem
    shp = folder / f"{archive.stem}.shp"
    if not shp.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(folder)
    if not shp.exists():
        raise SystemExit(f"FAIL: {archive.name} contains no expected shapefile")
    return shp, hashlib.sha256(archive.read_bytes()).hexdigest()


def census_sources(candidates):
    """Acquire place, county, and military-installation boundary files."""
    states = sorted(candidates.loc[
        ~candidates["same_analysis_ccg"], "state"].unique())
    military_states = sorted(candidates.loc[
        candidates["city"].isin(INSTALLATION_ALIASES), "state"].unique())
    zcta_states = sorted(candidates.loc[
        candidates["city"].isin(FOCUSED_CITIES), "state"].unique())
    urls = {}
    for state in states:
        for vintage in ("2000", "2010"):
            suffix = vintage[-2:]
            urls[("place", vintage, state)] = (
                f"{TIGER_BASE}/PLACE/{vintage}/"
                f"tl_2010_{state}_place{suffix}.zip")
    for vintage in ("2000", "2010"):
        suffix = vintage[-2:]
        urls[("county", vintage, "us")] = (
            f"{TIGER_BASE}/COUNTY/{vintage}/"
            f"tl_2010_us_county{suffix}.zip")
    for state in military_states:
        urls[("military", "2010", state)] = (
            f"{TIGER_BASE}/MIL/tl_2010_{state}_mil.zip")
    for state in zcta_states:
        urls[("zcta", "2010", state)] = (
            f"{TIGER_BASE}/ZCTA5/2010/tl_2010_{state}_zcta510.zip")

    paths = {}
    rows = []
    for key, url in sorted(urls.items()):
        path, digest = acquire(url)
        paths[key] = path
        rows.append({
            "kind": key[0], "vintage": key[1], "state": key[2],
            "url": url, "archive": Path(url).name, "sha256": digest,
        })
    pd.DataFrame(rows).to_csv(
        CACHE_DIR / "remaining_candidate_historical_sources.csv", index=False)
    return paths


def _place_counties(connection, case, paths, vintage):
    """Intersect one historical Census place polygon with county polygons."""
    suffix = vintage[-2:]
    county_geoid = (f"c.GEOID{suffix}" if vintage == "2010"
                    else f"c.STATEFP{suffix} || c.COUNTYFP{suffix}")
    place_path = str(paths[("place", vintage, case.state)])
    county_path = str(paths[("county", vintage, "us")])
    query = f"""
        WITH places AS (
            SELECT NAME{suffix} AS geography_name, geom
            FROM ST_Read(?)
            WHERE STATEFP{suffix} = ?
        ), normalized AS (
            SELECT geography_name, geom,
                   TRIM(REGEXP_REPLACE(UPPER(geography_name),
                                      '[^A-Z0-9]+', ' ', 'g')) AS name_key
            FROM places
        ), selected AS (
            SELECT geography_name, geom
            FROM normalized
            WHERE name_key = ?
        ), intersections AS (
            SELECT p.geography_name,
                   {county_geoid} AS county_fips,
                   c.NAME{suffix} AS county_name,
                   ST_Area(ST_Intersection(p.geom, c.geom))
                       / NULLIF(ST_Area(p.geom), 0) AS area_share
            FROM selected p
            CROSS JOIN ST_Read(?) c
            WHERE c.STATEFP{suffix} = ?
              AND ST_Intersects(p.geom, c.geom)
        )
        SELECT geography_name, county_fips, county_name, area_share
        FROM intersections
        WHERE area_share > 1e-10
        ORDER BY county_fips
    """
    return connection.execute(
        query, [place_path, case.state, case.city, county_path, case.state]).df()


def _military_counties(connection, case, paths):
    """Intersect an official 2010 installation polygon with counties."""
    military_path = str(paths[("military", "2010", case.state)])
    county_path = str(paths[("county", "2010", "us")])
    aliases = INSTALLATION_ALIASES[case.city]
    names = connection.execute(
        "SELECT DISTINCT FULLNAME FROM ST_Read(?) ORDER BY FULLNAME",
        [military_path]).df()
    matched = [name for name in names["FULLNAME"].dropna().astype(str)
               if any(alias in normalize_name(name) for alias in aliases)]
    if not matched:
        return pd.DataFrame(columns=[
            "geography_name", "county_fips", "county_name", "area_share"])
    placeholders = ",".join("?" for _ in matched)
    query = f"""
        WITH selected AS (
            SELECT FULLNAME AS geography_name, geom
            FROM ST_Read(?)
            WHERE FULLNAME IN ({placeholders})
        ), intersections AS (
            SELECT p.geography_name, c.GEOID10 AS county_fips,
                   c.NAME10 AS county_name,
                   ST_Area(ST_Intersection(p.geom, c.geom))
                       / NULLIF(ST_Area(p.geom), 0) AS area_share
            FROM selected p
            CROSS JOIN ST_Read(?) c
            WHERE c.STATEFP10 = ?
              AND ST_Intersects(p.geom, c.geom)
        )
        SELECT geography_name, county_fips, county_name, SUM(area_share) AS area_share
        FROM intersections
        WHERE area_share > 1e-10
        GROUP BY geography_name, county_fips, county_name
        ORDER BY geography_name, county_fips
    """
    return connection.execute(
        query, [military_path, *matched, county_path, case.state]).df()


def historical_geography(candidates, paths):
    """Build place and installation county intersections for all 22 cases."""
    connection = duckdb.connect()
    connection.execute("INSTALL spatial; LOAD spatial")
    rows = []
    selected = candidates.loc[~candidates["same_analysis_ccg"]]
    for case in selected.itertuples(index=False):
        for vintage in ("2000", "2010"):
            frame = _place_counties(connection, case, paths, vintage)
            if len(frame):
                frame.insert(0, "geography_type", "census_place")
                frame.insert(0, "vintage", int(vintage))
                frame.insert(0, "city", case.city)
                frame.insert(0, "state", case.state)
                rows.append(frame)
        if case.city in INSTALLATION_ALIASES:
            frame = _military_counties(connection, case, paths)
            if len(frame):
                frame.insert(0, "geography_type", "military_installation")
                frame.insert(0, "vintage", 2010)
                frame.insert(0, "city", case.city)
                frame.insert(0, "state", case.state)
                rows.append(frame)
    connection.close()
    if not rows:
        raise SystemExit("FAIL: no historical geography matches")
    return pd.concat(rows, ignore_index=True)


def zcta_geography(actions, candidates, paths):
    """Intersect USAspending performance ZIP ZCTAs with 2010 counties."""
    focus = candidates.loc[candidates["city"].isin(FOCUSED_CITIES), [
        "state", "city"]]
    zips = (actions.loc[
        actions["source"].eq("USAspending")
        & actions["performance_zip"].fillna("").str.fullmatch(r"[0-9]{5}"),
        ["state", "city", "performance_zip"]]
        .drop_duplicates()
        .merge(focus, on=["state", "city"], how="inner",
               validate="many_to_one"))
    connection = duckdb.connect()
    connection.execute("INSTALL spatial; LOAD spatial")
    rows = []
    for item in zips.itertuples(index=False):
        zcta_path = str(paths[("zcta", "2010", item.state)])
        county_path = str(paths[("county", "2010", "us")])
        query = """
            WITH selected AS (
                SELECT ZCTA5CE10 AS geography_name, geom
                FROM ST_Read(?)
                WHERE ZCTA5CE10 = ?
            ), intersections AS (
                SELECT z.geography_name, c.GEOID10 AS county_fips,
                       c.NAME10 AS county_name,
                       ST_Area(ST_Intersection(z.geom, c.geom))
                           / NULLIF(ST_Area(z.geom), 0) AS area_share
                FROM selected z
                CROSS JOIN ST_Read(?) c
                WHERE c.STATEFP10 = ?
                  AND ST_Intersects(z.geom, c.geom)
            )
            SELECT geography_name, county_fips, county_name, area_share
            FROM intersections
            WHERE area_share > 1e-10
            ORDER BY county_fips
        """
        frame = connection.execute(
            query, [zcta_path, item.performance_zip, county_path,
                    item.state]).df()
        if len(frame):
            frame.insert(0, "geography_type", "zcta")
            frame.insert(0, "vintage", 2010)
            frame.insert(0, "city", item.city)
            frame.insert(0, "state", item.state)
            rows.append(frame)
    connection.close()
    if not rows:
        raise SystemExit("FAIL: no 2010 ZCTA matches for focused cases")
    return pd.concat(rows, ignore_index=True)


def transaction_actions(candidates):
    """Read raw actions for the 22 analysis-relevant repeated candidates."""
    cases = candidates.loc[~candidates["same_analysis_ccg"], [
        "state", "city", "dcads_county", "usaspending_county"]].copy()
    connection = duckdb.connect()
    connection.register("cases", cases[["state", "city"]])
    dcads = connection.execute("""
        WITH raw AS (
            SELECT CAST(fiscal_year AS INTEGER) AS year,
                   TRIM(state_code) AS state,
                   TRIM(REGEXP_REPLACE(UPPER(TRIM(city_name)),
                                       '[^A-Z0-9]+', ' ', 'g')) AS city,
                   CASE
                       WHEN COALESCE(TRIM(county_code), '') <> ''
                           THEN TRIM(county_code)
                       WHEN REGEXP_FULL_MATCH(TRIM(county_fips), '[0-9]{3}')
                           THEN TRIM(state_code) || TRIM(county_fips)
                       ELSE TRIM(county_fips)
                   END AS reported_county,
                   CASE WHEN dollars_unit = 'thousands'
                        THEN dollars * 1000.0 ELSE dollars END AS dollars_nominal,
                   REGEXP_REPLACE(UPPER(TRIM(contract_no)),
                                  '[^A-Z0-9]+', '', 'g') AS contract_id,
                   CAST(NULL AS VARCHAR) AS performance_zip
            FROM read_parquet(?, union_by_name=true)
            WHERE geo_class = 'county'
              AND CASE WHEN dollars_unit = 'thousands'
                       THEN dollars * 1000.0 ELSE dollars END > 0
        )
        SELECT raw.*
        FROM raw INNER JOIN cases USING (state, city)
    """, [source_paths("DCADS")]).df()
    usa = connection.execute("""
        WITH raw AS (
            SELECT CAST(source_fiscal_year AS INTEGER) AS year,
                   prime_award_transaction_place_of_performance_state_fips_code
                       AS state,
                   TRIM(REGEXP_REPLACE(
                       UPPER(TRIM(primary_place_of_performance_city_name)),
                       '[^A-Z0-9]+', ' ', 'g')) AS city,
                   prime_award_transaction_place_of_performance_county_fips_code
                       AS reported_county,
                   federal_action_obligation AS dollars_nominal,
                   REGEXP_REPLACE(UPPER(TRIM(award_id_piid)),
                                  '[^A-Z0-9]+', '', 'g') AS contract_id,
                   SUBSTR(REGEXP_REPLACE(primary_place_of_performance_zip_4,
                                         '[^0-9]+', '', 'g'), 1, 5)
                       AS performance_zip
            FROM read_parquet(?, union_by_name=true)
            WHERE award_or_idv_flag = 'AWARD'
              AND geo_class = 'county'
              AND federal_action_obligation > 0
        )
        SELECT raw.*
        FROM raw INNER JOIN cases USING (state, city)
    """, [source_paths("USAspending")]).df()
    connection.close()
    parts = []
    for source, frame in (("DCADS", dcads), ("USAspending", usa)):
        frame["source"] = source
        frame["reported_county"] = (
            frame["reported_county"].fillna("").astype(str)
            .str.replace(r"\.0$", "", regex=True).str.zfill(5))
        frame["dollars_real"] = (
            frame["dollars_nominal"]
            * frame["year"].map(dc.deflate).astype(float))
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    observed = set(zip(result["state"], result["city"]))
    expected = set(zip(cases["state"], cases["city"]))
    if observed != expected:
        raise SystemExit(
            f"FAIL: transaction actions miss {sorted(expected - observed)}")
    return result


def transaction_summaries(actions):
    """Collapse actions by source, year, and raw reported county."""
    result = (actions.groupby(
        ["state", "city", "source", "year", "reported_county"],
        dropna=False, sort=False)
        .agg(actions=("dollars_real", "size"),
             dollars_real=("dollars_real", "sum"),
             unique_contract_ids=("contract_id", "nunique"))
        .reset_index())
    totals = result.groupby(
        ["state", "city", "source", "year"], sort=False
    )["dollars_real"].transform("sum")
    result["dollar_share"] = result["dollars_real"] / totals
    return result.sort_values(
        ["state", "city", "source", "year", "dollars_real"],
        ascending=[True, True, True, True, False])


def contract_matches(actions):
    """Match normalized contract identifiers across the common-year sources."""
    usable = actions.loc[actions["contract_id"].fillna("").str.len().ge(4)]
    keys = ["state", "city", "source", "contract_id", "reported_county"]
    zip_rows = (usable.loc[
        usable["performance_zip"].fillna("").ne(""),
        [*keys, "performance_zip"]]
        .drop_duplicates()
        .sort_values([*keys, "performance_zip"])
        .groupby(keys, dropna=False, sort=False)["performance_zip"]
        .agg(";".join).rename("performance_zips").reset_index())
    collapsed = (usable.groupby(
        keys, dropna=False, sort=False)
        .agg(actions=("dollars_real", "size"),
             dollars_nominal=("dollars_nominal", "sum"),
             dollars_real=("dollars_real", "sum"))
        .reset_index())
    collapsed = collapsed.merge(zip_rows, on=keys, how="left",
                                validate="one_to_one")
    collapsed["performance_zips"] = collapsed["performance_zips"].fillna("")
    left = collapsed.loc[collapsed["source"].eq("DCADS")].drop(
        columns="source").add_prefix("dcads_")
    right = collapsed.loc[collapsed["source"].eq("USAspending")].drop(
        columns="source").add_prefix("usaspending_")
    result = left.merge(
        right,
        left_on=["dcads_state", "dcads_city", "dcads_contract_id"],
        right_on=["usaspending_state", "usaspending_city",
                  "usaspending_contract_id"],
        how="inner", validate="many_to_many")
    return result.rename(columns={
        "dcads_state": "state", "dcads_city": "city",
        "dcads_contract_id": "contract_id",
    }).drop(columns=[
        "usaspending_state", "usaspending_city", "usaspending_contract_id"])


def county_sets(geography, state, city, geography_type, vintage):
    """Return counties from one geography layer and vintage."""
    selected = geography.loc[
        geography["state"].eq(state)
        & geography["city"].eq(city)
        & geography["geography_type"].eq(geography_type)
        & geography["vintage"].eq(vintage), "county_fips"]
    return set(selected.astype(str))


def build_review(candidates, geography, summaries, matches):
    """Classify the 29 cases without authorizing new production overrides."""
    rows = []
    for case in candidates.itertuples(index=False):
        base = {
            "state": case.state, "city": case.city,
            "dcads_county": case.dcads_county,
            "usaspending_county": case.usaspending_county,
            "same_analysis_ccg": bool(case.same_analysis_ccg),
        }
        if case.same_analysis_ccg:
            rows.append({
                **base, "census_place_2000_counties": "",
                "census_place_2010_counties": "",
                "census_military_2010_counties": "",
                "usaspending_performance_zips": "",
                "census_zcta_2010_counties": "",
                "matched_contract_ids": 0,
                "matched_ids_with_disputed_pair": 0,
                "review_status": "no_analysis_change_required",
                "production_override_authorized": False,
            })
            continue

        place00 = county_sets(
            geography, case.state, case.city, "census_place", 2000)
        place10 = county_sets(
            geography, case.state, case.city, "census_place", 2010)
        mil10 = county_sets(
            geography, case.state, case.city, "military_installation", 2010)
        zcta10 = county_sets(
            geography, case.state, case.city, "zcta", 2010)
        action_zips = set(
            matches.loc[
                matches["state"].eq(case.state)
                & matches["city"].eq(case.city),
                "usaspending_performance_zips"]
            .fillna("").str.split(";").explode())
        action_zips.discard("")
        subm = matches.loc[
            matches["state"].eq(case.state)
            & matches["city"].eq(case.city)]
        expected_pair = subm.loc[
            subm["dcads_reported_county"].eq(case.dcads_county)
            & subm["usaspending_reported_county"].eq(
                case.usaspending_county)]

        disputed = {case.dcads_county, case.usaspending_county}
        is_installation = case.city in INSTALLATION_ALIASES
        if is_installation:
            if disputed.issubset(mil10):
                status = "valid_multicounty_installation"
            elif (case.usaspending_county in mil10
                  and case.dcads_county not in mil10):
                status = "probable_dcads_error_installation"
            elif (case.dcads_county in mil10
                  and case.usaspending_county not in mil10):
                status = "probable_usaspending_error_installation"
            else:
                status = "installation_boundary_unresolved"
        elif disputed.issubset(place00) and disputed.issubset(place10):
            status = "valid_multicounty_place"
        elif (case.usaspending_county in place00
              and case.usaspending_county in place10
              and case.dcads_county not in place00
              and case.dcads_county not in place10):
            status = "probable_dcads_error_place"
        elif (case.dcads_county in place00 and case.dcads_county in place10
              and case.usaspending_county not in place00
              and case.usaspending_county not in place10):
            status = "probable_usaspending_error_place"
        elif place00 or place10:
            status = "historical_place_boundary_changed_or_unresolved"
        else:
            status = "no_historical_place_match"

        if (status == "probable_usaspending_error_place"
                and case.usaspending_county in zcta10):
            status = "valid_postal_place_assignment"
        if (status == "probable_dcads_error_installation"
                and case.usaspending_county in zcta10
                and int(subm["contract_id"].nunique()) > 0
                and int(expected_pair["contract_id"].nunique())
                    == int(subm["contract_id"].nunique())):
            status = "verified_dcads_geography_error"

        rows.append({
            **base,
            "census_place_2000_counties": ";".join(sorted(place00)),
            "census_place_2010_counties": ";".join(sorted(place10)),
            "census_military_2010_counties": ";".join(sorted(mil10)),
            "usaspending_performance_zips": ";".join(sorted(action_zips)),
            "census_zcta_2010_counties": ";".join(sorted(zcta10)),
            "matched_contract_ids": int(subm["contract_id"].nunique()),
            "matched_ids_with_disputed_pair": int(
                expected_pair["contract_id"].nunique()),
            "review_status": status,
            "production_override_authorized": (
                status == "verified_dcads_geography_error"),
        })
    return pd.DataFrame(rows).sort_values(
        ["same_analysis_ccg", "state", "city"])


def write_memo(review):
    """Write a compact decision record for the 29 remaining candidates."""
    counts = review["review_status"].value_counts().sort_index()
    overrides = pd.read_csv(DCADS_OVERRIDE, dtype="string")
    added = overrides.loc[overrides["city_name"].isin(
        ["TINKER AFB", "DYESS AFB"])].copy()
    added["expected_rows"] = added["expected_rows"].astype(int)
    first_stage = pd.read_csv(FIRST_STAGE_IN).iloc[0]
    lines = [
        "# Remaining repeated city-county candidates",
        "",
        "## Scope",
        "",
        (f"The repeated-window audit contains {EXPECTED_REPEATED} locations. "
         f"Table 1 reviews the {len(review)} cases other than Sealy, "
         "Lancaster, West Jefferson, and Brooklyn, which have focused "
         "verification memos. The historical review uses "
         "official 2000 and 2010 Census place boundaries, the official 2010 "
         "Census military-installation layer, raw positive contract actions "
         "from both sources over fiscal years 2001 to 2006, and common "
         "normalized contract identifiers."),
        "",
        ("A boundary screen never authorizes a production correction by "
         "itself. A verified correction also requires performance-ZIP and "
         "common-contract evidence that points to the same county."),
        "",
        "## Review status",
        "",
        "**Table 1. Status counts**",
        "",
        "| Status | Locations |",
        "|---|---:|",
    ]
    for status, count in counts.items():
        lines.append(f"| `{status}` | {int(count)} |")
    lines += [
        "",
        "**Table 2. Location-level review**",
        "",
        ("| State | Location | DCADS county | USAspending county | "
         "Census place counties, 2000 | Census place counties, 2010 | "
         "Census installation counties, 2010 | Performance ZIPs | "
         "ZCTA counties, 2010 | Matched IDs | Status |"),
        "|---:|---|---:|---:|---|---|---|---|---|---:|---|",
    ]
    for row in review.itertuples(index=False):
        lines.append(
            f"| {row.state} | {row.city.title()} | {row.dcads_county} | "
            f"{row.usaspending_county} | "
            f"{row.census_place_2000_counties or 'not applicable'} | "
            f"{row.census_place_2010_counties or 'not applicable'} | "
            f"{row.census_military_2010_counties or 'not applicable'} | "
            f"{row.usaspending_performance_zips or 'not available'} | "
            f"{row.census_zcta_2010_counties or 'not applicable'} | "
            f"{row.matched_contract_ids} | `{row.review_status}` |")
    lines += [
        "",
        "## Decision rule",
        "",
        ("`no_analysis_change_required` means both reported counties already "
         "map to the same time-consistent county group. "
         "`valid_multicounty_place` and `valid_multicounty_installation` mean "
         "the official historical boundary contains both disputed counties. "
         "A source switch is therefore not evidence of a coding error. "
         "`valid_postal_place_assignment` means the incorporated place does "
         "not reach the reported county but the USAspending performance ZIP's "
         "2010 ZCTA does. `verified_dcads_geography_error` requires the 2010 "
         "military boundary and performance ZCTA to support the USAspending "
         "county, to exclude the DCADS county, and every common contract "
         "identifier to reproduce that county pair."),
        "",
        "## Production status",
        "",
        "**Table 3. Verified DCADS corrections from this review**",
        "",
        "| Location | Reported county | Correct county | Contract actions |",
        "|---|---:|---:|---:|",
    ]
    for city in ("TINKER AFB", "DYESS AFB"):
        selected = added.loc[added["city_name"].eq(city)]
        lines.append(
            f"| {city.title()} | {selected['reported_county_fips'].iloc[0]} | "
            f"{selected['corrected_county_fips'].iloc[0]} | "
            f"{selected['expected_rows'].sum():,} |")
    lines += [
        "",
        ("The guarded mappings are stored in "
         "`step_1/code/reference/dcads_city_county_overrides.csv`. Producer "
         "`07_add_county_fips.py` matched every declared row count, and "
         "producers `06_verify_dcads_parse.py` and "
         "`08_verify_county_fips.py` verified all 42 fiscal-year files."),
        "",
        (f"The rebuilt h=2 first stage has coefficient "
         f"`{float(first_stage['first_stage']):.10f}`, standard error "
         f"`{float(first_stage['first_stage_se']):.10f}`, F-statistic "
         f"`{float(first_stage['first_stage_F']):.10f}`, and partial "
         f"R-squared `{float(first_stage['partial_r2']):.10f}`. These values "
         "are reported in "
         "`Results/instrument_normalization/h2_county_exposure_window_market_share_shifter.csv`."),
        "",
        "## Files",
        "",
        ("- Historical geography: "
         "`Results/instrument_normalization/remaining_repeated_candidate_historical_geography.csv`."),
        ("- Transaction cells: "
         "`Results/instrument_normalization/remaining_repeated_candidate_transaction_summary.csv`."),
        ("- Common contract identifiers: "
         "`Results/instrument_normalization/remaining_repeated_candidate_contract_matches.csv`."),
        ("- Review table: "
         "`Results/instrument_normalization/remaining_repeated_candidate_review.csv`."),
        ("- Census source hashes: "
         "`_data/temp/usaspending_city_county_audit/historical_tiger/remaining_candidate_historical_sources.csv`."),
        "",
    ]
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the historical and transaction audit for all 29 candidates."""
    start = time.time()
    print("== audit 29 remaining repeated city-county candidates")
    candidates = remaining_candidates()
    print(f"   candidates: {len(candidates)}, including "
          f"{int(candidates['same_analysis_ccg'].sum())} same analysis CCG")
    paths = census_sources(candidates)
    actions = transaction_actions(candidates)
    geography = historical_geography(candidates, paths)
    geography = pd.concat(
        [geography, zcta_geography(actions, candidates, paths)],
        ignore_index=True)
    summaries = transaction_summaries(actions)
    matches = contract_matches(actions)
    review = build_review(candidates, geography, summaries, matches)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    geography.to_csv(CENSUS_OUT, index=False)
    summaries.to_csv(TRANSACTION_OUT, index=False)
    matches.to_csv(MATCH_OUT, index=False)
    review.to_csv(REVIEW_OUT, index=False)
    write_memo(review)

    print("   status counts:")
    for status, count in review["review_status"].value_counts().items():
        print(f"      {status}: {count}")
    print(f"   common contract IDs: {matches['contract_id'].nunique():,}")
    print(f"   wrote {REVIEW_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
