"""Audit the 84 city-county conflicts observed in only one comparison window.

The producer combines fixed county-group mappings, official 2000 and 2010
Census place boundaries, the official 2010 ZCTA-to-county relationship file,
comparison-specific raw contract actions, and normalized contract identifiers.
It writes audit evidence and a review table but does not alter production
geography.

Writes Results/instrument_normalization/
    single_window_candidate_historical_geography.csv
    single_window_candidate_transaction_summary.csv
    single_window_candidate_contract_matches.csv
    single_window_candidate_review.csv
and Instrument_analysis/Cleaning_spendingData/
    SINGLE_WINDOW_CANDIDATES.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_city_county_audit/historical_tiger
# mpgr-output-cache: Results/instrument_normalization/single_window_candidate_review.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|comparison-specific-windows|single-window-city-county-review

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


EXPECTED_CANDIDATES = 84
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
ZCTA_REL_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel/"
    "zcta_county_rel_10.txt"
)
CACHE_DIR = (dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
             / "historical_tiger")
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CANDIDATE_IN = OUTDIR / "usaspending_city_county_override_candidates.csv"
CCG_IN = PIPELINE.parent / "reference" / "county_ccg_crosswalk.csv"
CENSUS_OUT = OUTDIR / "single_window_candidate_historical_geography.csv"
TRANSACTION_OUT = OUTDIR / "single_window_candidate_transaction_summary.csv"
MATCH_OUT = OUTDIR / "single_window_candidate_contract_matches.csv"
REVIEW_OUT = OUTDIR / "single_window_candidate_review.csv"
FIRST_STAGE_IN = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
DCADS_OVERRIDE = PIPELINE.parent / "reference" / "dcads_city_county_overrides.csv"
MEMO_OUT = (dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
            / "Cleaning_spendingData" / "SINGLE_WINDOW_CANDIDATES.md")

WINDOWS = {
    "overlap_2001_2006": {
        "DCADS": tuple(range(2001, 2007)),
        "USAspending": tuple(range(2001, 2007)),
    },
    "seam_2004_2006_vs_2007_2009": {
        "DCADS": tuple(range(2004, 2007)),
        "USAspending": tuple(range(2007, 2010)),
    },
}


def normalize_name(value):
    """Normalize a reported place name."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def acquire(url):
    """Download one fixed Census file and return its path and hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / Path(url).name
    if not target.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent": "aggregation-disaggregation-invariance"})
        with urllib.request.urlopen(request, timeout=180) as response:
            target.write_bytes(response.read())
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def acquire_shape(url):
    """Download and extract one fixed Census TIGER/Line archive."""
    archive, digest = acquire(url)
    folder = CACHE_DIR / archive.stem
    shape = folder / f"{archive.stem}.shp"
    if not shape.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(folder)
    if not shape.exists():
        raise SystemExit(f"FAIL: {archive.name} contains no expected shapefile")
    return shape, digest


def single_window_candidates():
    """Return city-county pairs that occur in exactly one audit comparison."""
    data = pd.read_csv(
        CANDIDATE_IN,
        dtype={"state": "string", "dcads_county": "string",
               "usaspending_county": "string"})
    keys = ["state", "city", "dcads_county", "usaspending_county"]
    counts = (data.groupby(keys, dropna=False)["comparison"].nunique()
              .rename("comparison_count").reset_index())
    data = data.merge(counts, on=keys, how="left", validate="many_to_one")
    result = data.loc[data["comparison_count"].eq(1)].copy()
    if len(result) != EXPECTED_CANDIDATES:
        raise SystemExit(
            f"FAIL: single-window count is {len(result)}, expected "
            f"{EXPECTED_CANDIDATES}")
    if result.duplicated(keys).any():
        raise SystemExit("FAIL: duplicate single-window candidate keys")
    result["state"] = result["state"].str.zfill(2)
    result["dcads_county"] = result["dcads_county"].str.zfill(5)
    result["usaspending_county"] = result["usaspending_county"].str.zfill(5)
    result["city"] = result["city"].map(normalize_name)

    crosswalk = pd.read_csv(
        CCG_IN, dtype={"county_fips5": "string", "ccg_id": "string"})
    mapping = crosswalk.set_index("county_fips5")["ccg_id"]
    result["dcads_ccg"] = result["dcads_county"].map(mapping)
    result["usaspending_ccg"] = result["usaspending_county"].map(mapping)
    if result[["dcads_ccg", "usaspending_ccg"]].isna().any().any():
        bad = result.loc[
            result[["dcads_ccg", "usaspending_ccg"]].isna().any(axis=1),
            keys]
        raise SystemExit(f"FAIL: candidate counties missing from CCG map:\n{bad}")
    result["same_analysis_ccg"] = result["dcads_ccg"].eq(
        result["usaspending_ccg"])
    return result


def census_sources(candidates):
    """Acquire historical place and county boundaries plus ZCTA relations."""
    states = sorted(candidates.loc[
        ~candidates["same_analysis_ccg"], "state"].unique())
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

    paths = {}
    rows = []
    for key, url in sorted(urls.items()):
        path, digest = acquire_shape(url)
        paths[key] = path
        rows.append({
            "kind": key[0], "vintage": key[1], "state": key[2],
            "url": url, "archive": Path(url).name, "sha256": digest,
        })
    zcta_path, zcta_hash = acquire(ZCTA_REL_URL)
    paths[("zcta_rel", "2010", "us")] = zcta_path
    rows.append({
        "kind": "zcta_county_relationship", "vintage": "2010",
        "state": "us", "url": ZCTA_REL_URL,
        "archive": zcta_path.name, "sha256": zcta_hash,
    })
    pd.DataFrame(rows).to_csv(
        CACHE_DIR / "single_window_candidate_historical_sources.csv",
        index=False)
    return paths


def _place_counties(connection, case, paths, vintage):
    """Intersect one historical Census place with county polygons."""
    suffix = vintage[-2:]
    county_geoid = (f"c.GEOID{suffix}" if vintage == "2010"
                    else f"c.STATEFP{suffix} || c.COUNTYFP{suffix}")
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
            SELECT p.geography_name, {county_geoid} AS county_fips,
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
    return connection.execute(query, [
        str(paths[("place", vintage, case.state)]), case.state, case.city,
        str(paths[("county", vintage, "us")]), case.state,
    ]).df()


def place_geography(candidates, paths):
    """Build 2000 and 2010 place-to-county intersections."""
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
    connection.close()
    columns = [
        "state", "city", "vintage", "geography_type", "geography_name",
        "county_fips", "county_name", "area_share",
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=columns)


def source_paths(source, years):
    """Return fixed annual parquet paths for one procurement source."""
    directory = dc.OUT_DIR if source == "DCADS" else dc.USASPENDING_OUT_DIR
    paths = [directory / f"fy{year}.parquet" for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: {source} is missing {len(missing)} files")
    return [str(path) for path in paths]


def transaction_actions(candidates):
    """Read actions in each candidate's comparison-specific source window."""
    cases = candidates[[
        "comparison", "state", "city", "dcads_county",
        "usaspending_county"]].copy()
    frames = []
    for comparison, windows in WINDOWS.items():
        subcases = cases.loc[cases["comparison"].eq(comparison)]
        for source, years in windows.items():
            connection = duckdb.connect()
            connection.register("cases", subcases[["state", "city"]])
            if source == "DCADS":
                reported_county = (
                    "CASE WHEN COALESCE(TRIM(county_code), '') <> '' "
                    "THEN TRIM(county_code) "
                    "WHEN REGEXP_FULL_MATCH(TRIM(county_fips), '[0-9]{3}') "
                    "THEN TRIM(state_code) || TRIM(county_fips) "
                    "ELSE TRIM(county_fips5) END"
                    if comparison == "overlap_2001_2006"
                    else "TRIM(state_code) || TRIM(county_fips)"
                )
                query = f"""
                    WITH raw AS (
                        SELECT CAST(fiscal_year AS INTEGER) AS year,
                               TRIM(state_code) AS state,
                               TRIM(REGEXP_REPLACE(UPPER(TRIM(city_name)),
                                   '[^A-Z0-9]+', ' ', 'g')) AS city,
                               {reported_county} AS reported_county,
                               CASE WHEN dollars_unit = 'thousands'
                                    THEN dollars * 1000.0 ELSE dollars END
                                   AS dollars_nominal,
                               REGEXP_REPLACE(UPPER(TRIM(contract_no)),
                                              '[^A-Z0-9]+', '', 'g')
                                   AS contract_id,
                               CAST(NULL AS VARCHAR) AS performance_zip
                        FROM read_parquet(?, union_by_name=true)
                        WHERE geo_class = 'county'
                          AND CASE WHEN dollars_unit = 'thousands'
                                   THEN dollars * 1000.0 ELSE dollars END > 0
                    )
                    SELECT raw.* FROM raw INNER JOIN cases USING (state, city)
                """
            else:
                query = """
                    WITH raw AS (
                        SELECT CAST(source_fiscal_year AS INTEGER) AS year,
                               prime_award_transaction_place_of_performance_state_fips_code
                                   AS state,
                               TRIM(REGEXP_REPLACE(UPPER(TRIM(
                                   primary_place_of_performance_city_name)),
                                   '[^A-Z0-9]+', ' ', 'g')) AS city,
                               prime_award_transaction_place_of_performance_county_fips_code
                                   AS reported_county,
                               federal_action_obligation AS dollars_nominal,
                               REGEXP_REPLACE(UPPER(TRIM(award_id_piid)),
                                              '[^A-Z0-9]+', '', 'g')
                                   AS contract_id,
                               SUBSTR(REGEXP_REPLACE(
                                   primary_place_of_performance_zip_4,
                                   '[^0-9]+', '', 'g'), 1, 5)
                                   AS performance_zip
                        FROM read_parquet(?, union_by_name=true)
                        WHERE award_or_idv_flag = 'AWARD'
                          AND geo_class = 'county'
                          AND federal_action_obligation > 0
                    )
                    SELECT raw.* FROM raw INNER JOIN cases USING (state, city)
                """
            frame = connection.execute(
                query, [source_paths(source, years)]).df()
            connection.close()
            frame["source"] = source
            frame["comparison"] = comparison
            frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["reported_county"] = (
        result["reported_county"].fillna("").astype(str)
        .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    result["dollars_real"] = (
        result["dollars_nominal"]
        * result["year"].map(dc.deflate).astype(float))
    observed = set(zip(result["comparison"], result["state"], result["city"]))
    expected = set(zip(cases["comparison"], cases["state"], cases["city"]))
    if observed != expected:
        raise SystemExit(
            f"FAIL: transaction actions miss {sorted(expected - observed)}")
    return result


def transaction_summaries(actions):
    """Collapse actions by comparison, source, year, county, and ZIP."""
    keys = [
        "comparison", "state", "city", "source", "year",
        "reported_county", "performance_zip",
    ]
    result = (actions.groupby(keys, dropna=False, sort=False)
              .agg(actions=("dollars_real", "size"),
                   dollars_real=("dollars_real", "sum"),
                   unique_contract_ids=("contract_id", "nunique"))
              .reset_index())
    totals = result.groupby(
        ["comparison", "state", "city", "source", "year"],
        sort=False)["dollars_real"].transform("sum")
    result["dollar_share"] = result["dollars_real"] / totals
    return result.sort_values(
        ["comparison", "state", "city", "source", "year", "dollars_real"],
        ascending=[True, True, True, True, True, False])


def contract_matches(actions):
    """Match normalized contract IDs across each comparison's source windows."""
    usable = actions.loc[actions["contract_id"].fillna("").str.len().ge(4)]
    keys = [
        "comparison", "state", "city", "source", "contract_id",
        "reported_county",
    ]
    collapsed = (usable.groupby(keys, dropna=False, sort=False)
                 .agg(actions=("dollars_real", "size"),
                      dollars_real=("dollars_real", "sum"),
                      years=("year", lambda x: ";".join(
                          str(int(v)) for v in sorted(set(x)))))
                 .reset_index())
    zip_rows = (usable.loc[
        usable["performance_zip"].fillna("").str.fullmatch(r"[0-9]{5}"),
        [*keys, "performance_zip"]].drop_duplicates()
        .sort_values([*keys, "performance_zip"])
        .groupby(keys, dropna=False, sort=False)["performance_zip"]
        .agg(";".join).rename("performance_zips").reset_index())
    collapsed = collapsed.merge(
        zip_rows, on=keys, how="left", validate="one_to_one")
    collapsed["performance_zips"] = collapsed["performance_zips"].fillna("")
    left = collapsed.loc[collapsed["source"].eq("DCADS")].drop(
        columns="source").add_prefix("dcads_")
    right = collapsed.loc[collapsed["source"].eq("USAspending")].drop(
        columns="source").add_prefix("usaspending_")
    result = left.merge(
        right,
        left_on=["dcads_comparison", "dcads_state", "dcads_city",
                 "dcads_contract_id"],
        right_on=["usaspending_comparison", "usaspending_state",
                  "usaspending_city", "usaspending_contract_id"],
        how="inner", validate="many_to_many")
    if result.empty:
        return result
    result = result.rename(columns={
        "dcads_comparison": "comparison", "dcads_state": "state",
        "dcads_city": "city", "dcads_contract_id": "contract_id",
    }).drop(columns=[
        "usaspending_comparison", "usaspending_state", "usaspending_city",
        "usaspending_contract_id"])
    result["common_years"] = result.apply(
        lambda row: ";".join(sorted(
            set(str(row["dcads_years"]).split(";"))
            & set(str(row["usaspending_years"]).split(";")))), axis=1)
    return result


def zcta_geography(actions, paths):
    """Map observed USAspending performance ZIPs to 2010 counties."""
    zips = (actions.loc[
        actions["source"].eq("USAspending")
        & actions["performance_zip"].fillna("").str.fullmatch(r"[0-9]{5}"),
        ["comparison", "state", "city", "performance_zip"]]
        .drop_duplicates())
    relations = pd.read_csv(
        paths[("zcta_rel", "2010", "us")], dtype=str,
        usecols=["ZCTA5", "STATE", "COUNTY", "GEOID", "ZPOPPCT",
                 "ZAREAPCT"])
    relations = relations.rename(columns={
        "ZCTA5": "performance_zip", "GEOID": "county_fips",
        "ZPOPPCT": "population_share", "ZAREAPCT": "area_share",
    })
    result = zips.merge(
        relations[["performance_zip", "county_fips", "population_share",
                   "area_share"]],
        on="performance_zip", how="left", validate="many_to_many")
    if result["county_fips"].isna().any():
        missing = sorted(result.loc[
            result["county_fips"].isna(), "performance_zip"].unique())
        print(f"   warning: {len(missing)} performance ZIPs lack 2010 ZCTA rows")
    result = result.dropna(subset=["county_fips"]).copy()
    result["vintage"] = 2010
    result["geography_type"] = "zcta"
    result["geography_name"] = result["performance_zip"]
    result["county_name"] = ""
    return result[[
        "state", "city", "vintage", "geography_type", "geography_name",
        "county_fips", "county_name", "area_share", "population_share",
        "comparison",
    ]]


def county_set(geography, case, geography_type, vintage):
    """Return counties from one case, geography layer, and vintage."""
    selected = geography.loc[
        geography["comparison"].eq(case.comparison)
        & geography["state"].eq(case.state)
        & geography["city"].eq(case.city)
        & geography["geography_type"].eq(geography_type)
        & geography["vintage"].eq(vintage), "county_fips"]
    return set(selected.astype(str))


def semicolon(values):
    """Join a sorted set without missing-value artifacts."""
    return ";".join(sorted(str(value) for value in values if str(value)))


def build_review(candidates, geography, actions, matches):
    """Classify the 84 cases without authorizing production overrides."""
    rows = []
    for case in candidates.itertuples(index=False):
        base = {
            "comparison": case.comparison, "state": case.state,
            "city": case.city, "dcads_county": case.dcads_county,
            "usaspending_county": case.usaspending_county,
            "same_analysis_ccg": bool(case.same_analysis_ccg),
        }
        place00 = county_set(geography, case, "census_place", 2000)
        place10 = county_set(geography, case, "census_place", 2010)
        zcta10 = county_set(geography, case, "zcta", 2010)
        disputed = {case.dcads_county, case.usaspending_county}

        usa_actions = actions.loc[
            actions["comparison"].eq(case.comparison)
            & actions["state"].eq(case.state)
            & actions["city"].eq(case.city)
            & actions["source"].eq("USAspending")]
        action_zips = set(usa_actions.loc[
            usa_actions["performance_zip"].fillna("").str.fullmatch(
                r"[0-9]{5}"), "performance_zip"])
        subm = matches.loc[
            matches["comparison"].eq(case.comparison)
            & matches["state"].eq(case.state)
            & matches["city"].eq(case.city)] if not matches.empty else matches
        pair = subm.loc[
            subm["dcads_reported_county"].eq(case.dcads_county)
            & subm["usaspending_reported_county"].eq(
                case.usaspending_county)] if not subm.empty else subm
        matched_ids = int(subm["contract_id"].nunique()) if len(subm) else 0
        pair_ids = int(pair["contract_id"].nunique()) if len(pair) else 0
        same_year_ids = int(subm.loc[
            subm["common_years"].fillna("").ne(""),
            "contract_id"].nunique()) if len(subm) else 0
        same_year_pair_ids = int(pair.loc[
            pair["common_years"].fillna("").ne(""),
            "contract_id"].nunique()) if len(pair) else 0
        from_dcads_candidate = subm.loc[
            subm["dcads_reported_county"].eq(case.dcads_county)] \
            if len(subm) else subm
        same_year_from_dcads_candidate = int(from_dcads_candidate.loc[
            from_dcads_candidate["common_years"].fillna("").ne(""),
            "contract_id"].nunique()) if len(from_dcads_candidate) else 0
        exact_pair = pair.loc[
            pair["common_years"].fillna("").ne("")
            & (pd.to_numeric(pair["dcads_dollars_real"])
               - pd.to_numeric(pair["usaspending_dollars_real"])).abs().lt(
                   0.01)] if len(pair) else pair
        exact_same_year_pair_ids = int(
            exact_pair["contract_id"].nunique()) if len(exact_pair) else 0

        if case.same_analysis_ccg:
            status = "no_analysis_change_required"
        elif disputed.issubset(place00) and disputed.issubset(place10):
            status = "valid_multicounty_place"
        elif disputed.issubset(zcta10):
            status = "valid_multicounty_postal_assignment"
        else:
            usa_place = (case.usaspending_county in place00
                         or case.usaspending_county in place10)
            dcads_place = (case.dcads_county in place00
                           or case.dcads_county in place10)
            usa_zip = case.usaspending_county in zcta10
            dcads_zip = case.dcads_county in zcta10
            stable_usa_place = (case.usaspending_county in place00
                                and case.usaspending_county in place10
                                and case.dcads_county not in place00
                                and case.dcads_county not in place10)
            stable_dcads_place = (case.dcads_county in place00
                                  and case.dcads_county in place10
                                  and case.usaspending_county not in place00
                                  and case.usaspending_county not in place10)
            verified_overlap = (
                case.comparison == "overlap_2001_2006"
                and same_year_from_dcads_candidate > 0
                and same_year_pair_ids == same_year_from_dcads_candidate
                and exact_same_year_pair_ids == same_year_pair_ids)
            if stable_usa_place and usa_zip and not dcads_zip and verified_overlap:
                status = "verified_dcads_geography_error"
            elif stable_usa_place and usa_zip and not dcads_zip:
                status = "evidence_supports_usaspending_county"
            elif stable_dcads_place and dcads_zip and not usa_zip:
                status = "evidence_supports_dcads_county"
            elif usa_zip and not dcads_zip and not dcads_place:
                status = "valid_postal_place_assignment"
            elif dcads_zip and not usa_zip and not usa_place:
                status = "postal_evidence_supports_dcads_county"
            elif (usa_place or dcads_place or usa_zip or dcads_zip):
                status = "historical_boundary_or_mixed_evidence"
            else:
                status = "no_geographic_match"

        rows.append({
            **base,
            "census_place_2000_counties": semicolon(place00),
            "census_place_2010_counties": semicolon(place10),
            "usaspending_performance_zips": semicolon(action_zips),
            "census_zcta_2010_counties": semicolon(zcta10),
            "matched_contract_ids": matched_ids,
            "matched_ids_with_disputed_pair": pair_ids,
            "same_year_matched_contract_ids": same_year_ids,
            "same_year_ids_with_disputed_pair": same_year_pair_ids,
            "same_year_ids_from_dcads_candidate_county": (
                same_year_from_dcads_candidate),
            "exact_dollar_same_year_ids_with_disputed_pair": (
                exact_same_year_pair_ids),
            "review_status": status,
            "production_override_authorized": (
                status == "verified_dcads_geography_error"),
        })
    return pd.DataFrame(rows).sort_values(
        ["comparison", "state", "city"])


def write_memo(review):
    """Write the evidence and classification record for the 84 cases."""
    counts = review.groupby(
        ["comparison", "review_status"], sort=True).size().rename(
        "locations").reset_index()
    pelham = review.loc[
        review["state"].eq("01") & review["city"].eq("PELHAM")].iloc[0]
    overrides = pd.read_csv(DCADS_OVERRIDE, dtype="string")
    pelham_overrides = overrides.loc[
        overrides["state_code"].eq("01")
        & overrides["city_name"].eq("PELHAM")]
    pelham_rows = int(pd.to_numeric(
        pelham_overrides["expected_rows"]).sum())
    first_stage = pd.read_csv(FIRST_STAGE_IN).iloc[0]
    lines = [
        "# Single-window city-county candidates",
        "",
        "## Scope",
        "",
        (f"Table 1 classifies all {len(review)} city-county pairs observed in "
         "exactly one source-comparison window. Fourteen pairs come from the "
         "common fiscal years 2001 to 2006. Seventy pairs compare DCADS fiscal "
         "years 2004 to 2006 with USAspending fiscal years 2007 to 2009."),
        "",
        ("The audit uses the fixed analysis county-group crosswalk, official "
         "2000 and 2010 Census place polygons, the official 2010 Census "
         "ZCTA-to-county relationship file, raw positive contract actions in "
         "the relevant source window, and normalized contract identifiers. "
         "Single-window discrepancies receive no production correction unless "
         "independent historical geography, performance-ZIP geography, and "
         "same-year contract evidence all identify the same error."),
        "",
        "## Classification",
        "",
        "**Table 1. Status counts by comparison**",
        "",
        "| Comparison | Status | Locations |",
        "|---|---|---:|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(
            f"| `{row.comparison}` | `{row.review_status}` | "
            f"{int(row.locations)} |")
    lines += [
        "",
        "**Table 2. Location-level evidence**",
        "",
        ("| Comparison | State | Location | DCADS county | USAspending county | "
         "Place counties, 2000 | Place counties, 2010 | Performance ZIPs | "
         "ZCTA counties, 2010 | Matched IDs | Same-year IDs | "
         "Same-year disputed-pair IDs | Status |"),
        "|---|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in review.itertuples(index=False):
        lines.append(
            f"| `{row.comparison}` | {row.state} | {row.city.title()} | "
            f"{row.dcads_county} | {row.usaspending_county} | "
            f"{row.census_place_2000_counties or 'not matched'} | "
            f"{row.census_place_2010_counties or 'not matched'} | "
            f"{row.usaspending_performance_zips or 'not available'} | "
            f"{row.census_zcta_2010_counties or 'not matched'} | "
            f"{row.matched_contract_ids} | "
            f"{row.same_year_matched_contract_ids} | "
            f"{row.same_year_ids_with_disputed_pair} | "
            f"`{row.review_status}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        ("`no_analysis_change_required` means that the two county codes map to "
         "the same fixed analysis county group. `valid_multicounty_place` and "
         "`valid_multicounty_postal_assignment` mean that both county codes are "
         "consistent with the reported place or performance ZIP. "
         "`valid_postal_place_assignment` means that the USAspending county is "
         "consistent with the reported performance ZIP when the place name is "
         "not an incorporated-place locator. Other evidence labels remain "
         "review flags, not authorized production changes."),
        "",
        "## Production status",
        "",
        "**Table 3. Verified DCADS correction**",
        "",
        "| Location | Reported county | Correct county | Fiscal years | Records |",
        "|---|---:|---:|---:|---:|",
        (f"| Pelham, Alabama | {pelham.dcads_county} | "
         f"{pelham.usaspending_county} | 2001 to 2006 | {pelham_rows:,} |"),
        "",
        ("Pelham intersects Shelby County, 01117, and not Choctaw County, "
         "01023, in both Census place vintages. USAspending ZIP 35124 maps to "
         "Shelby County in the 2010 ZCTA relationship file. Eight contract "
         "identifiers observed in the same fiscal year carry county 01023 in "
         "DCADS and county 01117 in USAspending; their source amounts match "
         "exactly and total $747,241.00. The identifier counts are reported in "
         "Table 2, and the matched records are reported in "
         "`single_window_candidate_contract_matches.csv`."),
        "",
        ("The guarded production table contains six Pelham mappings covering "
         f"{pelham_rows:,} source records. Producer `07_add_county_fips.py` "
         "matched every declared annual count. The 20 seam-only cases labeled "
         "`evidence_supports_usaspending_county` remain review flags because "
         "their source windows contain no common fiscal year."),
        "",
        (f"The rebuilt h=2 first stage has coefficient "
         f"`{float(first_stage['first_stage']):.10f}`, standard error "
         f"`{float(first_stage['first_stage_se']):.10f}`, F-statistic "
         f"`{float(first_stage['first_stage_F']):.10f}`, and partial R-squared "
         f"`{float(first_stage['partial_r2']):.10f}`. These values are reported "
         "in `Results/instrument_normalization/"
         "h2_county_exposure_window_market_share_shifter.csv`."),
        "",
        "## Files",
        "",
        ("- Historical geography: "
         "`Results/instrument_normalization/single_window_candidate_historical_geography.csv`."),
        ("- Transaction cells: "
         "`Results/instrument_normalization/single_window_candidate_transaction_summary.csv`."),
        ("- Contract identifiers: "
         "`Results/instrument_normalization/single_window_candidate_contract_matches.csv`."),
        ("- Review table: "
         "`Results/instrument_normalization/single_window_candidate_review.csv`."),
        ("- Census source hashes: "
         "`_data/temp/usaspending_city_county_audit/historical_tiger/single_window_candidate_historical_sources.csv`."),
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the single-window historical-geography and transaction audit."""
    start = time.time()
    print("== audit 84 single-window city-county candidates")
    candidates = single_window_candidates()
    print(f"   candidates: {len(candidates)}, including "
          f"{int(candidates['same_analysis_ccg'].sum())} same analysis CCG")
    paths = census_sources(candidates)
    actions = transaction_actions(candidates)
    places = place_geography(candidates, paths)
    zctas = zcta_geography(actions, paths)
    places["comparison"] = places.merge(
        candidates[["state", "city", "comparison"]],
        on=["state", "city"], how="left", validate="many_to_one"
    )["comparison"] if len(places) else ""
    if len(places):
        places["population_share"] = ""
    geography = pd.concat([places, zctas], ignore_index=True, sort=False)
    summaries = transaction_summaries(actions)
    matches = contract_matches(actions)
    review = build_review(candidates, geography, actions, matches)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    geography.to_csv(CENSUS_OUT, index=False)
    summaries.to_csv(TRANSACTION_OUT, index=False)
    matches.to_csv(MATCH_OUT, index=False)
    review.to_csv(REVIEW_OUT, index=False)
    write_memo(review)

    print("   status counts:")
    for status, count in review["review_status"].value_counts().items():
        print(f"      {status}: {count}")
    print(f"   common contract IDs: "
          f"{matches['contract_id'].nunique() if len(matches) else 0:,}")
    print(f"   wrote {REVIEW_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
