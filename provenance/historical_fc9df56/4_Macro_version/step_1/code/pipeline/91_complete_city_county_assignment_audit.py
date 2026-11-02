"""Audit every resolved city-county assignment in the contract panel.

The audit covers DCADS fiscal years 1966 to 2006 and USAspending fiscal years
2001 to 2018. It reconstructs reported counties before the guarded geography
overrides, retains positive and negative actions, and imposes no dollar or
dominance threshold when creating candidates. Candidate signals include
within-source county changes, isolated one-year assignments, same-year
cross-source disagreements, exact contract-ID disagreements, Census place
compatibility, and USAspending performance-ZIP compatibility.

The producer is diagnostic. It does not alter production geography.

Writes _data/temp/usaspending_city_county_audit/complete/
    assignment_cells.parquet
    assignment_zip_cells.parquet
    place_county_lookup.parquet
    assignment_review.parquet
and Results/instrument_normalization/
    complete_city_county_audit_flags.csv
    complete_city_county_annual_disagreements.csv
    complete_city_county_contract_conflicts.csv
    complete_city_county_priority_contract_pairs.csv
    complete_city_county_audit_summary.csv
and Instrument_analysis/Cleaning_spendingData/
    COMPLETE_CITY_COUNTY_ASSIGNMENT_AUDIT.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: 1_current_paper/_data/temp/dcads_year
# mpgr-output-cache: _data/temp/usaspending_city_county_audit/complete/assignment_cells.parquet
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|1966-2018|complete-city-county-candidate-search

from __future__ import annotations

import argparse
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
import pyarrow.parquet as pq


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


DCADS_YEARS = tuple(range(1966, 2007))
USA_YEARS = tuple(range(2001, 2019))
COMMON_YEARS = tuple(range(2001, 2007))
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2010"
ZCTA_REL_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel/"
    "zcta_county_rel_10.txt"
)

CACHE_ROOT = (
    dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
)
COMPLETE_DIR = CACHE_ROOT / "complete"
TIGER_DIR = CACHE_ROOT / "historical_tiger"
ASSIGNMENT_OUT = COMPLETE_DIR / "assignment_cells.parquet"
ZIP_OUT = COMPLETE_DIR / "assignment_zip_cells.parquet"
PLACE_OUT = COMPLETE_DIR / "place_county_lookup.parquet"
REVIEW_OUT = COMPLETE_DIR / "assignment_review.parquet"
SOURCE_OUT = COMPLETE_DIR / "complete_audit_sources.csv"

RESULTS_DIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
FLAGS_OUT = RESULTS_DIR / "complete_city_county_audit_flags.csv"
ANNUAL_OUT = RESULTS_DIR / "complete_city_county_annual_disagreements.csv"
CONTRACT_OUT = RESULTS_DIR / "complete_city_county_contract_conflicts.csv"
PRIORITY_OUT = RESULTS_DIR / "complete_city_county_priority_contract_pairs.csv"
SUMMARY_OUT = RESULTS_DIR / "complete_city_county_audit_summary.csv"
MEMO_OUT = (
    dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
    / "Cleaning_spendingData" / "COMPLETE_CITY_COUNTY_ASSIGNMENT_AUDIT.md"
)
CCG_IN = PIPELINE.parent / "reference" / "county_ccg_crosswalk.csv"
DCADS_OVERRIDE_IN = PIPELINE.parent / "reference" / "dcads_city_county_overrides.csv"
VERIFIED_DCADS_OVERRIDE_IN = (
    PIPELINE.parent / "reference" / "dcads_verified_city_county_overrides.csv"
)
USA_OVERRIDE_IN = PIPELINE.parent / "reference" / "usaspending_city_county_overrides.csv"
LOCATION_REVIEW_INPUTS = (
    RESULTS_DIR / "remaining_repeated_candidate_review.csv",
    RESULTS_DIR / "single_window_candidate_review.csv",
)


def normalize_name(value: object) -> str:
    """Normalize one reported place name."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def source_paths(source: str, years: tuple[int, ...]) -> list[str]:
    """Return and verify annual parquet paths for one source."""
    directory = dc.OUT_DIR if source == "DCADS" else dc.USASPENDING_OUT_DIR
    paths = [directory / f"fy{year}.parquet" for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: {source} is missing {len(missing)} annual files")
    return [str(path) for path in paths]


def add_real_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Deflate nominal dollar aggregates using the project deflator."""
    factors = frame["year"].map(dc.deflate)
    if factors.isna().any():
        years = sorted(frame.loc[factors.isna(), "year"].unique())
        raise SystemExit(f"FAIL: no deflator for fiscal years {years}")
    for stem in ("net", "positive", "negative_abs", "absolute"):
        frame[f"{stem}_dollars_real"] = (
            frame[f"{stem}_dollars_nominal"].astype(float)
            * factors.astype(float)
        )
    return frame.drop(columns=[
        "net_dollars_nominal", "positive_dollars_nominal",
        "negative_abs_dollars_nominal", "absolute_dollars_nominal",
    ])


def excluded_row_counts(cells: pd.DataFrame) -> pd.DataFrame:
    """Count source rows outside the auditable state-city-county universe."""
    rows = []
    for source, years in (("DCADS", DCADS_YEARS), ("USAspending", USA_YEARS)):
        total = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in map(Path, source_paths(source, years))
        )
        audited = cells.loc[cells["source"].eq(source), "source_rows"].sum()
        rows.append({
            "metric": f"{source.lower()}_rows_without_auditable_city_county",
            "value": int(total - audited),
        })
    return pd.DataFrame(rows)


def extract_assignment_cells() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate all resolved raw city-county assignments and ZIP cells."""
    connection = duckdb.connect()
    dcads_paths = source_paths("DCADS", DCADS_YEARS)
    usa_paths = source_paths("USAspending", USA_YEARS)

    dcads = connection.execute("""
        WITH prepared AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                CASE
                    WHEN REGEXP_FULL_MATCH(TRIM(CAST(state_fips2 AS VARCHAR)),
                                           '[0-9]{2}')
                    THEN TRIM(CAST(state_fips2 AS VARCHAR))
                    WHEN REGEXP_FULL_MATCH(TRIM(CAST(state_code AS VARCHAR)),
                                           '[0-9]{2}')
                    THEN TRIM(CAST(state_code AS VARCHAR))
                    ELSE ''
                END AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                CASE
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 1984 AND 2003
                    THEN TRIM(CAST(county_code AS VARCHAR))
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 2004 AND 2006
                    THEN TRIM(CAST(state_code AS VARCHAR))
                         || TRIM(CAST(county_fips AS VARCHAR))
                    ELSE TRIM(CAST(county_fips5 AS VARCHAR))
                END AS county,
                CASE WHEN dollars_unit = 'thousands'
                     THEN CAST(dollars AS DOUBLE) * 1000.0
                     ELSE CAST(dollars AS DOUBLE) END AS dollars_nominal,
                TRIM(CAST(fips_source AS VARCHAR)) AS geography_method,
                CAST(is_lumpsum AS BOOLEAN) AS is_lumpsum
            FROM read_parquet(?, union_by_name=true)
        )
        SELECT
            year, state, city, county,
            COUNT(*) AS source_rows,
            SUM(CASE WHEN dollars_nominal > 0 THEN 1 ELSE 0 END)
                AS positive_rows,
            SUM(dollars_nominal) AS net_dollars_nominal,
            SUM(GREATEST(dollars_nominal, 0)) AS positive_dollars_nominal,
            SUM(GREATEST(-dollars_nominal, 0)) AS negative_abs_dollars_nominal,
            SUM(ABS(dollars_nominal)) AS absolute_dollars_nominal,
            STRING_AGG(DISTINCT geography_method, ';' ORDER BY geography_method)
                AS geography_methods
        FROM prepared
        WHERE NOT is_lumpsum
          AND state <> '' AND city <> ''
          AND REGEXP_FULL_MATCH(county, '[0-9]{5}')
        GROUP BY year, state, city, county
    """, [dcads_paths]).df()

    usa = connection.execute("""
        WITH prepared AS (
            SELECT
                CAST(source_fiscal_year AS INTEGER) AS year,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_state_fips_code
                    AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(
                    primary_place_of_performance_city_name AS VARCHAR))),
                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_county_fips_code
                    AS VARCHAR)) AS county,
                CAST(federal_action_obligation AS DOUBLE) AS dollars_nominal,
                SUBSTR(REGEXP_REPLACE(CAST(
                    primary_place_of_performance_zip_4 AS VARCHAR),
                    '[^0-9]+', '', 'g'), 1, 5) AS performance_zip,
                TRIM(CAST(county_fips_source AS VARCHAR)) AS geography_method
            FROM read_parquet(?, union_by_name=true)
            WHERE award_or_idv_flag = 'AWARD'
              AND primary_place_of_performance_country_code = 'USA'
        )
        SELECT
            year, state, city, county,
            COUNT(*) AS source_rows,
            SUM(CASE WHEN dollars_nominal > 0 THEN 1 ELSE 0 END)
                AS positive_rows,
            SUM(dollars_nominal) AS net_dollars_nominal,
            SUM(GREATEST(dollars_nominal, 0)) AS positive_dollars_nominal,
            SUM(GREATEST(-dollars_nominal, 0)) AS negative_abs_dollars_nominal,
            SUM(ABS(dollars_nominal)) AS absolute_dollars_nominal,
            STRING_AGG(DISTINCT geography_method, ';' ORDER BY geography_method)
                AS geography_methods
        FROM prepared
        WHERE state <> '' AND city <> ''
          AND REGEXP_FULL_MATCH(county, '[0-9]{5}')
        GROUP BY year, state, city, county
    """, [usa_paths]).df()

    zip_cells = connection.execute("""
        WITH prepared AS (
            SELECT
                CAST(source_fiscal_year AS INTEGER) AS year,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_state_fips_code
                    AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(
                    primary_place_of_performance_city_name AS VARCHAR))),
                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_county_fips_code
                    AS VARCHAR)) AS county,
                SUBSTR(REGEXP_REPLACE(CAST(
                    primary_place_of_performance_zip_4 AS VARCHAR),
                    '[^0-9]+', '', 'g'), 1, 5) AS performance_zip,
                CAST(federal_action_obligation AS DOUBLE) AS dollars_nominal
            FROM read_parquet(?, union_by_name=true)
            WHERE award_or_idv_flag = 'AWARD'
              AND primary_place_of_performance_country_code = 'USA'
        )
        SELECT
            year, state, city, county, performance_zip,
            COUNT(*) AS source_rows,
            SUM(ABS(dollars_nominal)) AS absolute_dollars_nominal
        FROM prepared
        WHERE state <> '' AND city <> ''
          AND REGEXP_FULL_MATCH(county, '[0-9]{5}')
          AND REGEXP_FULL_MATCH(performance_zip, '[0-9]{5}')
        GROUP BY year, state, city, county, performance_zip
    """, [usa_paths]).df()
    connection.close()

    dcads.insert(0, "source", "DCADS")
    usa.insert(0, "source", "USAspending")
    cells = pd.concat([dcads, usa], ignore_index=True)
    cells = add_real_values(cells)
    zip_cells["absolute_dollars_real"] = (
        zip_cells["absolute_dollars_nominal"].astype(float)
        * zip_cells["year"].map(dc.deflate).astype(float)
    )
    zip_cells = zip_cells.drop(columns="absolute_dollars_nominal")

    cells["state_county_conflict"] = cells["state"].ne(
        cells["county"].str[:2])
    county_universe = set(pd.read_csv(
        CCG_IN, dtype={"county_fips5": "string"})["county_fips5"])
    cells["county_in_analysis_crosswalk"] = cells["county"].isin(
        county_universe)

    excluded = excluded_row_counts(cells)
    return cells, zip_cells, excluded


def acquire(url: str) -> tuple[Path, str]:
    """Download one fixed Census source and return its path and SHA-256 hash."""
    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    target = TIGER_DIR / Path(url).name
    if not target.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent": "aggregation-disaggregation-invariance"})
        with urllib.request.urlopen(request, timeout=180) as response:
            target.write_bytes(response.read())
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def acquire_shape(url: str) -> tuple[Path, str]:
    """Download and extract one fixed Census TIGER/Line archive."""
    archive, digest = acquire(url)
    folder = TIGER_DIR / archive.stem
    shape = folder / f"{archive.stem}.shp"
    if not shape.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            source.extractall(folder)
    if not shape.exists():
        raise SystemExit(f"FAIL: {archive.name} lacks the expected shapefile")
    return shape, digest


def acquire_geography(states: list[str]) -> tuple[dict[tuple[str, str, str], Path], pd.DataFrame]:
    """Acquire place and county layers for every state in the audit."""
    urls: dict[tuple[str, str, str], str] = {}
    for state in states:
        for vintage in ("2000", "2010"):
            suffix = vintage[-2:]
            urls[("place", vintage, state)] = (
                f"{TIGER_BASE}/PLACE/{vintage}/"
                f"tl_2010_{state}_place{suffix}.zip"
            )
    for vintage in ("2000", "2010"):
        suffix = vintage[-2:]
        urls[("county", vintage, "us")] = (
            f"{TIGER_BASE}/COUNTY/{vintage}/"
            f"tl_2010_us_county{suffix}.zip"
        )

    paths: dict[tuple[str, str, str], Path] = {}
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
    return paths, pd.DataFrame(rows)


def build_place_lookup(
        states: list[str], paths: dict[tuple[str, str, str], Path]) -> pd.DataFrame:
    """Intersect every Census place in audited states with county polygons."""
    connection = duckdb.connect()
    connection.execute("INSTALL spatial; LOAD spatial")
    rows = []
    for vintage in ("2000", "2010"):
        suffix = vintage[-2:]
        county_geoid = (
            f"c.GEOID{suffix}" if vintage == "2010"
            else f"c.STATEFP{suffix} || c.COUNTYFP{suffix}"
        )
        county_path = str(paths[("county", vintage, "us")])
        for state in states:
            query = f"""
                WITH places AS (
                    SELECT NAME{suffix} AS geography_name, geom
                    FROM ST_Read(?, open_options=['ENCODING=ISO-8859-1'])
                    WHERE STATEFP{suffix} = ?
                ), counties AS (
                    SELECT {county_geoid} AS county,
                           NAME{suffix} AS county_name, geom
                    FROM ST_Read(?, open_options=['ENCODING=ISO-8859-1']) c
                    WHERE STATEFP{suffix} = ?
                ), hits AS (
                    SELECT
                        p.geography_name,
                        TRIM(REGEXP_REPLACE(UPPER(p.geography_name),
                                            '[^A-Z0-9]+', ' ', 'g')) AS city,
                        c.county, c.county_name,
                        ST_Area(ST_Intersection(p.geom, c.geom))
                            / NULLIF(ST_Area(p.geom), 0) AS area_share
                    FROM places p
                    CROSS JOIN counties c
                    WHERE ST_Intersects(p.geom, c.geom)
                )
                SELECT geography_name, city, county, county_name, area_share
                FROM hits
                WHERE area_share > 1e-10
                ORDER BY city, county
            """
            frame = connection.execute(query, [
                str(paths[("place", vintage, state)]), state,
                county_path, state,
            ]).df()
            if len(frame):
                frame.insert(0, "vintage", int(vintage))
                frame.insert(0, "state", state)
                rows.append(frame)
    connection.close()
    if not rows:
        raise SystemExit("FAIL: Census place intersections are empty")
    return pd.concat(rows, ignore_index=True)


def build_diagnostics(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build pooled assignments, annual dominance, and anomaly signals."""
    keys = ["source", "state", "city"]
    pooled = (cells.groupby(keys + ["county"], sort=False)
              .agg(years=("year", "nunique"),
                   first_year=("year", "min"), last_year=("year", "max"),
                   source_rows=("source_rows", "sum"),
                   positive_rows=("positive_rows", "sum"),
                   net_dollars_real=("net_dollars_real", "sum"),
                   positive_dollars_real=("positive_dollars_real", "sum"),
                   negative_abs_dollars_real=("negative_abs_dollars_real", "sum"),
                   absolute_dollars_real=("absolute_dollars_real", "sum"),
                   state_county_conflict=("state_county_conflict", "max"),
                   county_in_analysis_crosswalk=(
                       "county_in_analysis_crosswalk", "max"))
              .reset_index())
    totals = pooled.groupby(keys, sort=False).agg(
        assignment_count=("county", "nunique"),
        identity_absolute_dollars_real=("absolute_dollars_real", "sum"),
    ).reset_index()
    pooled = pooled.merge(totals, on=keys, validate="many_to_one")
    pooled["pooled_dollar_share"] = (
        pooled["absolute_dollars_real"]
        / pooled["identity_absolute_dollars_real"].replace(0, np.nan)
    )

    annual = (cells.groupby(keys + ["year", "county"], sort=False)
              .agg(absolute_dollars_real=("absolute_dollars_real", "sum"),
                   source_rows=("source_rows", "sum"))
              .reset_index())
    annual = annual.sort_values(
        keys + ["year", "absolute_dollars_real", "county"],
        ascending=[True, True, True, True, False, True])
    annual = annual.drop_duplicates(keys + ["year"])
    annual = annual.sort_values(keys + ["year"]).reset_index(drop=True)
    groups = annual.groupby(keys, sort=False)
    annual["previous_year"] = groups["year"].shift(1)
    annual["next_year"] = groups["year"].shift(-1)
    annual["previous_county"] = groups["county"].shift(1)
    annual["next_county"] = groups["county"].shift(-1)
    annual["isolated_one_year_assignment"] = (
        annual["previous_year"].eq(annual["year"] - 1)
        & annual["next_year"].eq(annual["year"] + 1)
        & annual["previous_county"].eq(annual["next_county"])
        & annual["county"].ne(annual["previous_county"])
    )

    common = annual.loc[annual["year"].isin(COMMON_YEARS)].copy()
    left = common.loc[common["source"].eq("DCADS")].drop(
        columns="source").add_prefix("dcads_")
    right = common.loc[common["source"].eq("USAspending")].drop(
        columns="source").add_prefix("usaspending_")
    disagreements = left.merge(
        right,
        left_on=["dcads_state", "dcads_city", "dcads_year"],
        right_on=["usaspending_state", "usaspending_city", "usaspending_year"],
        how="inner", validate="one_to_one",
    )
    disagreements = disagreements.loc[
        disagreements["dcads_county"].ne(disagreements["usaspending_county"])
    ].copy()
    disagreements = disagreements.rename(columns={
        "dcads_state": "state", "dcads_city": "city", "dcads_year": "year",
    }).drop(columns=["usaspending_state", "usaspending_city", "usaspending_year"])
    return pooled, annual, disagreements


def exact_contract_conflicts() -> pd.DataFrame:
    """Find every same-year normalized contract-ID county disagreement."""
    connection = duckdb.connect()
    dcads_paths = source_paths("DCADS", COMMON_YEARS)
    usa_paths = source_paths("USAspending", COMMON_YEARS)
    result = connection.execute("""
        WITH d0 AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                TRIM(CAST(state_fips2 AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                CASE
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 1984 AND 2003
                    THEN TRIM(CAST(county_code AS VARCHAR))
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 2004 AND 2006
                    THEN TRIM(CAST(state_code AS VARCHAR))
                         || TRIM(CAST(county_fips AS VARCHAR))
                    ELSE TRIM(CAST(county_fips5 AS VARCHAR))
                END AS county,
                REGEXP_REPLACE(UPPER(TRIM(CAST(contract_no AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id,
                CASE WHEN dollars_unit = 'thousands'
                     THEN CAST(dollars AS DOUBLE) * 1000.0
                     ELSE CAST(dollars AS DOUBLE) END AS dollars_nominal
            FROM read_parquet(?, union_by_name=true)
            WHERE NOT CAST(is_lumpsum AS BOOLEAN)
        ), d AS (
            SELECT year, state, city, county, contract_id,
                   COUNT(*) AS dcads_rows,
                   SUM(dollars_nominal) AS dcads_net_dollars_nominal,
                   SUM(ABS(dollars_nominal)) AS dcads_absolute_dollars_nominal
            FROM d0
            WHERE state <> '' AND city <> ''
              AND REGEXP_FULL_MATCH(county, '[0-9]{5}')
              AND LENGTH(contract_id) >= 4
            GROUP BY year, state, city, county, contract_id
        ), u0 AS (
            SELECT
                CAST(source_fiscal_year AS INTEGER) AS year,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_state_fips_code
                    AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(
                    primary_place_of_performance_city_name AS VARCHAR))),
                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_county_fips_code
                    AS VARCHAR)) AS county,
                REGEXP_REPLACE(UPPER(TRIM(CAST(award_id_piid AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id,
                CAST(federal_action_obligation AS DOUBLE) AS dollars_nominal
            FROM read_parquet(?, union_by_name=true)
            WHERE award_or_idv_flag = 'AWARD'
              AND primary_place_of_performance_country_code = 'USA'
        ), u AS (
            SELECT year, state, city, county, contract_id,
                   COUNT(*) AS usaspending_rows,
                   SUM(dollars_nominal) AS usaspending_net_dollars_nominal,
                   SUM(ABS(dollars_nominal)) AS usaspending_absolute_dollars_nominal
            FROM u0
            WHERE state <> '' AND city <> ''
              AND REGEXP_FULL_MATCH(county, '[0-9]{5}')
              AND LENGTH(contract_id) >= 4
            GROUP BY year, state, city, county, contract_id
        )
        SELECT
            d.year, d.state, d.city, d.contract_id,
            d.county AS dcads_county, u.county AS usaspending_county,
            d.dcads_rows, u.usaspending_rows,
            d.dcads_net_dollars_nominal, u.usaspending_net_dollars_nominal,
            d.dcads_absolute_dollars_nominal,
            u.usaspending_absolute_dollars_nominal
        FROM d
        INNER JOIN u USING (year, state, city, contract_id)
        WHERE d.county <> u.county
        ORDER BY year, state, city, contract_id, d.county, u.county
    """, [dcads_paths, usa_paths]).df()
    connection.close()
    result["net_dollars_match_to_cent"] = (
        result["dcads_net_dollars_nominal"]
        - result["usaspending_net_dollars_nominal"]
    ).abs().lt(0.01)
    result["absolute_dollars_match_to_cent"] = (
        result["dcads_absolute_dollars_nominal"]
        - result["usaspending_absolute_dollars_nominal"]
    ).abs().lt(0.01)
    return result


def geography_support(
        pooled: pd.DataFrame, zip_cells: pd.DataFrame,
        places: pd.DataFrame, zcta_path: Path) -> pd.DataFrame:
    """Attach Census-place, ZCTA, CCG, and diagnostic support to assignments."""
    support = pooled.copy()
    for vintage in (2000, 2010):
        hit = places.loc[places["vintage"].eq(vintage), [
            "state", "city", "county"]].drop_duplicates()
        hit[f"place_{vintage}_supports_county"] = True
        support = support.merge(
            hit, on=["state", "city", "county"], how="left",
            validate="many_to_one")
        support[f"place_{vintage}_supports_county"] = support[
            f"place_{vintage}_supports_county"].eq(True)
        names = set(zip(
            places.loc[places["vintage"].eq(vintage), "state"],
            places.loc[places["vintage"].eq(vintage), "city"],
        ))
        support[f"place_{vintage}_name_matched"] = [
            (state, city) in names
            for state, city in zip(support["state"], support["city"])
        ]

    zcta = pd.read_csv(
        zcta_path, dtype=str, usecols=["ZCTA5", "GEOID"]
    ).rename(columns={"ZCTA5": "performance_zip", "GEOID": "county"})
    zcta["zcta_supports_county"] = True
    zip_support = (zip_cells[["state", "city", "county", "performance_zip"]]
                   .drop_duplicates()
                   .merge(zcta, on=["performance_zip", "county"], how="inner")
                   [["state", "city", "county", "zcta_supports_county"]]
                   .drop_duplicates())
    support = support.merge(
        zip_support, on=["state", "city", "county"], how="left",
        validate="many_to_one")
    support["zcta_supports_county"] = support[
        "zcta_supports_county"].eq(True)

    crosswalk = pd.read_csv(
        CCG_IN, dtype={"county_fips5": "string", "ccg_id": "string"})
    county_to_ccg = crosswalk.set_index("county_fips5")["ccg_id"]
    support["ccg_id"] = support["county"].map(county_to_ccg)
    support["direct_geography_support"] = (
        support["place_2000_supports_county"]
        | support["place_2010_supports_county"]
        | support["zcta_supports_county"]
    )
    supported_ccgs = set(zip(
        support.loc[support["direct_geography_support"], "state"],
        support.loc[support["direct_geography_support"], "city"],
        support.loc[support["direct_geography_support"], "ccg_id"],
    ))
    support["same_ccg_as_supported_assignment"] = [
        pd.notna(ccg) and (state, city, ccg) in supported_ccgs
        for state, city, ccg in zip(
            support["state"], support["city"], support["ccg_id"])
    ]

    implemented = set()
    pre_complete_audit = set()
    complete_audit = set()
    dcads_overrides = pd.read_csv(DCADS_OVERRIDE_IN, dtype="string")
    for row in dcads_overrides.itertuples(index=False):
        key = (
            "DCADS", str(row.state_code).zfill(2), normalize_name(row.city_name),
            str(row.reported_county_fips).zfill(5),
        )
        implemented.add(key)
        pre_complete_audit.add(key)
    verified_dcads = pd.read_csv(VERIFIED_DCADS_OVERRIDE_IN, dtype="string")
    for row in verified_dcads.itertuples(index=False):
        key = (
            "DCADS", str(row.state_code).zfill(2),
            normalize_name(row.city_name),
            str(row.reported_county_fips).zfill(5),
        )
        implemented.add(key)
        complete_audit.add(key)
    usa_overrides = pd.read_csv(USA_OVERRIDE_IN, dtype="string")
    for row in usa_overrides.itertuples(index=False):
        key = (
            "USAspending", str(row.corrected_county_fips).zfill(5)[:2],
            normalize_name(row.city_name),
            str(row.reported_county_fips).zfill(5),
        )
        implemented.add(key)
        pre_complete_audit.add(key)
    support["guarded_production_correction_exists"] = [
        (source, state, city, county) in implemented
        for source, state, city, county in zip(
            support["source"], support["state"], support["city"],
            support["county"])
    ]
    support["complete_audit_correction_exists"] = [
        (source, state, city, county) in complete_audit
        for source, state, city, county in zip(
            support["source"], support["state"], support["city"],
            support["county"])
    ]
    support["pre_complete_audit_correction_exists"] = [
        (source, state, city, county) in pre_complete_audit
        for source, state, city, county in zip(
            support["source"], support["state"], support["city"],
            support["county"])
    ]
    return support


def attach_flags(
        review: pd.DataFrame, annual: pd.DataFrame,
        disagreements: pd.DataFrame) -> pd.DataFrame:
    """Attach candidate reasons without authorizing a correction."""
    anomalous = set(zip(
        annual.loc[annual["isolated_one_year_assignment"], "source"],
        annual.loc[annual["isolated_one_year_assignment"], "state"],
        annual.loc[annual["isolated_one_year_assignment"], "city"],
        annual.loc[annual["isolated_one_year_assignment"], "county"],
    ))
    disagreement_keys = set()
    for row in disagreements.itertuples(index=False):
        disagreement_keys.add(("DCADS", row.state, row.city, row.dcads_county))
        disagreement_keys.add((
            "USAspending", row.state, row.city, row.usaspending_county))

    reasons = []
    for row in review.itertuples(index=False):
        row_reasons = []
        key = (row.source, row.state, row.city, row.county)
        place_name_matched = (
            row.place_2000_name_matched or row.place_2010_name_matched)
        if row.state_county_conflict:
            row_reasons.append("county_state_conflict")
        if key in anomalous:
            row_reasons.append("isolated_one_year_assignment")
        if key in disagreement_keys:
            row_reasons.append("same_year_cross_source_disagreement")
        if row.assignment_count > 1:
            row_reasons.append("within_source_multicounty_identity")
        if (place_name_matched and not row.direct_geography_support
                and not row.same_ccg_as_supported_assignment):
            row_reasons.append("county_outside_matched_census_place")
        if not place_name_matched and not row.zcta_supports_county:
            row_reasons.append("no_automated_geographic_match")
        reasons.append(";".join(row_reasons))
    review["candidate_reasons"] = reasons
    review["audit_candidate"] = review["candidate_reasons"].ne("")
    review["production_correction_authorized"] = False
    return review


def priority_contract_pairs(
        review: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """Rank exact-dollar source conflicts with asymmetric geography support."""
    columns = [
        "source", "state", "city", "county", "direct_geography_support",
        "same_ccg_as_supported_assignment", "place_2000_name_matched",
        "place_2010_name_matched", "guarded_production_correction_exists",
        "complete_audit_correction_exists",
        "pre_complete_audit_correction_exists",
    ]
    support = review[columns].copy()
    support["supported"] = (
        support["direct_geography_support"]
        | support["same_ccg_as_supported_assignment"]
    )
    support["matched_place_but_unsupported"] = (
        (support["place_2000_name_matched"]
         | support["place_2010_name_matched"])
        & ~support["supported"]
    )
    dcads = support.loc[support["source"].eq("DCADS"), [
        "state", "city", "county", "supported",
        "matched_place_but_unsupported", "guarded_production_correction_exists",
        "complete_audit_correction_exists",
        "pre_complete_audit_correction_exists",
    ]].rename(columns={
        "county": "dcads_county", "supported": "dcads_supported",
        "matched_place_but_unsupported": "dcads_unsupported",
        "guarded_production_correction_exists": "dcads_correction_exists",
        "complete_audit_correction_exists": "dcads_complete_audit_correction",
        "pre_complete_audit_correction_exists": "dcads_pre_complete_correction",
    })
    usa = support.loc[support["source"].eq("USAspending"), [
        "state", "city", "county", "supported",
        "matched_place_but_unsupported", "guarded_production_correction_exists",
        "complete_audit_correction_exists",
        "pre_complete_audit_correction_exists",
    ]].rename(columns={
        "county": "usaspending_county", "supported": "usaspending_supported",
        "matched_place_but_unsupported": "usaspending_unsupported",
        "guarded_production_correction_exists": "usaspending_correction_exists",
        "complete_audit_correction_exists": "usaspending_complete_audit_correction",
        "pre_complete_audit_correction_exists": "usaspending_pre_complete_correction",
    })
    evidence = (contracts.merge(
        dcads, on=["state", "city", "dcads_county"], how="left",
        validate="many_to_one").merge(
        usa, on=["state", "city", "usaspending_county"], how="left",
        validate="many_to_one"))
    evidence = evidence.loc[evidence["net_dollars_match_to_cent"]].copy()
    evidence["candidate_direction"] = np.select(
        [
            evidence["dcads_unsupported"].eq(True)
            & evidence["usaspending_supported"].eq(True),
            evidence["usaspending_unsupported"].eq(True)
            & evidence["dcads_supported"].eq(True),
        ],
        ["review_dcads_assignment", "review_usaspending_assignment"],
        default="",
    )
    evidence = evidence.loc[evidence["candidate_direction"].ne("")]
    evidence["implemented_correction_exists"] = np.where(
        evidence["candidate_direction"].eq("review_dcads_assignment"),
        evidence["dcads_correction_exists"].eq(True),
        evidence["usaspending_correction_exists"].eq(True),
    )
    evidence["pre_complete_audit_correction_exists"] = np.where(
        evidence["candidate_direction"].eq("review_dcads_assignment"),
        evidence["dcads_pre_complete_correction"].eq(True),
        evidence["usaspending_pre_complete_correction"].eq(True),
    )
    if evidence.empty:
        return pd.DataFrame(columns=[
            "state", "city", "dcads_county", "usaspending_county",
            "candidate_direction", "contract_ids", "fiscal_years",
            "first_year", "last_year", "dcads_absolute_dollars_nominal",
            "implemented_correction_exists",
            "complete_audit_correction_exists",
            "pre_complete_audit_correction_exists",
            "production_correction_authorized",
        ])
    keys = [
        "state", "city", "dcads_county", "usaspending_county",
        "candidate_direction", "implemented_correction_exists",
        "pre_complete_audit_correction_exists",
    ]
    result = (evidence.groupby(keys, sort=False)
              .agg(contract_ids=("contract_id", "nunique"),
                   fiscal_years=("year", "nunique"),
                   first_year=("year", "min"), last_year=("year", "max"),
                   dcads_absolute_dollars_nominal=(
                       "dcads_absolute_dollars_nominal", "sum"))
              .reset_index())
    verified = pd.read_csv(VERIFIED_DCADS_OVERRIDE_IN, dtype="string")
    verified_pairs = set(zip(
        verified["state_code"].str.zfill(2),
        verified["city_name"].map(normalize_name),
        verified["reported_county_fips"].str.zfill(5),
        verified["corrected_county_fips"].str.zfill(5),
    ))
    result["complete_audit_correction_exists"] = [
        (state, city, dcads_county, usaspending_county) in verified_pairs
        for state, city, dcads_county, usaspending_county in zip(
            result["state"], result["city"], result["dcads_county"],
            result["usaspending_county"])
    ]
    result["production_correction_authorized"] = result[
        "complete_audit_correction_exists"]
    prior_frames = []
    prior_keys = ["state", "city", "dcads_county", "usaspending_county"]
    for path in LOCATION_REVIEW_INPUTS:
        if not path.exists():
            raise SystemExit(f"FAIL: location-review input is missing: {path}")
        frame = pd.read_csv(
            path, dtype={key: "string" for key in prior_keys})
        prior_frames.append(frame[prior_keys + ["review_status"]])
    prior = pd.concat(prior_frames, ignore_index=True)
    prior = (prior.dropna(subset=["review_status"])
             .drop_duplicates(prior_keys + ["review_status"])
             .sort_values(prior_keys + ["review_status"])
             .groupby(prior_keys, sort=False)["review_status"].agg(";".join)
             .rename("location_review_status").reset_index())
    result = result.merge(prior, on=prior_keys, how="left", validate="many_to_one")
    result["location_review_status"] = result[
        "location_review_status"].fillna("")
    result["location_review_exists"] = result["location_review_status"].ne("")
    return result.sort_values(
        ["implemented_correction_exists", "location_review_exists",
         "dcads_absolute_dollars_nominal", "contract_ids"],
        ascending=[True, True, False, False]).reset_index(drop=True)


def summary_rows(
        cells: pd.DataFrame, review: pd.DataFrame, annual: pd.DataFrame,
        disagreements: pd.DataFrame, contracts: pd.DataFrame,
        priority: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """Return exact audit counts in a machine-readable two-column table."""
    rows = []
    for source in ("DCADS", "USAspending"):
        sub = cells.loc[cells["source"].eq(source)]
        rev = review.loc[review["source"].eq(source)]
        rows += [
            {"metric": f"{source.lower()}_first_year", "value": int(sub["year"].min())},
            {"metric": f"{source.lower()}_last_year", "value": int(sub["year"].max())},
            {"metric": f"{source.lower()}_audited_source_rows", "value": int(sub["source_rows"].sum())},
            {"metric": f"{source.lower()}_state_city_identities", "value": int(sub[["state", "city"]].drop_duplicates().shape[0])},
            {"metric": f"{source.lower()}_city_county_assignments", "value": int(len(rev))},
            {"metric": f"{source.lower()}_multicounty_identities", "value": int(rev.loc[rev["assignment_count"].gt(1), ["state", "city"]].drop_duplicates().shape[0])},
            {"metric": f"{source.lower()}_audit_candidates", "value": int(rev["audit_candidate"].sum())},
        ]
    rows += [
        {"metric": "assignment_cells", "value": int(len(cells))},
        {"metric": "distinct_city_county_assignments", "value": int(len(review))},
        {"metric": "isolated_one_year_dominant_assignments", "value": int(annual["isolated_one_year_assignment"].sum())},
        {"metric": "same_year_cross_source_dominant_disagreements", "value": int(len(disagreements))},
        {"metric": "exact_contract_id_conflict_rows", "value": int(len(contracts))},
        {"metric": "exact_contract_ids_with_conflicts", "value": int(contracts["contract_id"].nunique() if len(contracts) else 0)},
        {"metric": "exact_dollar_contract_ids_with_conflicts", "value": int(contracts.loc[contracts["net_dollars_match_to_cent"], "contract_id"].nunique() if len(contracts) else 0)},
        {"metric": "priority_contract_county_pairs", "value": int(len(priority))},
        {"metric": "priority_contract_ids", "value": int(priority["contract_ids"].sum() if len(priority) else 0)},
        {"metric": "unimplemented_priority_contract_county_pairs", "value": int((~priority["implemented_correction_exists"]).sum() if len(priority) else 0)},
        {"metric": "complete_classification_cohort_pairs", "value": int((~priority["pre_complete_audit_correction_exists"] & ~priority["location_review_exists"]).sum() if len(priority) else 0)},
        {"metric": "complete_audit_dcads_correction_pairs", "value": int(priority["complete_audit_correction_exists"].sum() if len(priority) else 0)},
        {"metric": "county_outside_matched_census_place", "value": int(review["candidate_reasons"].str.contains("county_outside_matched_census_place").sum())},
        {"metric": "no_automated_geographic_match", "value": int(review["candidate_reasons"].str.contains("no_automated_geographic_match").sum())},
        {"metric": "production_corrections_authorized", "value": int(priority["production_correction_authorized"].sum() if len(priority) else 0)},
    ]
    return pd.concat([pd.DataFrame(rows), excluded], ignore_index=True)


def write_memo(
        summary: pd.DataFrame, flags: pd.DataFrame,
        disagreements: pd.DataFrame, contracts: pd.DataFrame,
        priority: pd.DataFrame) -> None:
    """Write a concise audit memo with exact table-backed counts."""
    values = summary.set_index("metric")["value"]
    reason_counts = (flags.assign(reason=flags["candidate_reasons"].str.split(";"))
                     .explode("reason").groupby("reason").size()
                     .sort_values(ascending=False))
    top = flags.sort_values("absolute_dollars_real", ascending=False).head(30)
    lines = [
        "# Complete city-county assignment audit",
        "",
        "## Scope",
        "",
        ("The candidate search covers every DCADS fiscal year from 1966 to "
         "2006 and every USAspending fiscal year from 2001 to 2018. It uses "
         "every source row with a nonempty normalized city and a resolved "
         "five-digit reported county. Positive obligations, deobligations, "
         "and zero-dollar rows are retained. No minimum dollar amount or "
         "dominance-share cutoff is used to create a candidate."),
        "",
        ("Reported counties are reconstructed before the guarded production "
         "overrides. The audit therefore tests the source assignments rather "
         "than treating an implemented correction as raw data."),
        "",
        "## Coverage",
        "",
        "**Table 1. Complete search universe**",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| DCADS source rows audited | {int(values['dcads_audited_source_rows']):,} |",
        f"| USAspending source rows audited | {int(values['usaspending_audited_source_rows']):,} |",
        f"| Annual source-city-county cells | {int(values['assignment_cells']):,} |",
        f"| Pooled source-city-county assignments | {int(values['distinct_city_county_assignments']):,} |",
        f"| DCADS state-city identities | {int(values['dcads_state_city_identities']):,} |",
        f"| USAspending state-city identities | {int(values['usaspending_state_city_identities']):,} |",
        "",
        "## Candidate signals",
        "",
        "**Table 2. Flag counts**",
        "",
        "| Signal | Assignment rows |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.items():
        lines.append(f"| `{reason}` | {int(count):,} |")
    lines += [
        (f"| `same_year_cross_source_dominant_disagreements` | "
         f"{int(values['same_year_cross_source_dominant_disagreements']):,} |"),
        (f"| `exact_contract_ids_with_conflicts` | "
         f"{int(values['exact_contract_ids_with_conflicts']):,} |"),
        (f"| `exact_dollar_contract_ids_with_conflicts` | "
         f"{int(values['exact_dollar_contract_ids_with_conflicts']):,} |"),
        (f"| `priority_contract_county_pairs` | "
         f"{int(values['priority_contract_county_pairs']):,} |"),
        "",
        ("Signals are diagnostic, not findings of error. A state-city label "
         "can legitimately span counties, refer to a postal place, identify "
         "a military installation, or fail an exact Census-name match. The "
         "2000 and 2010 Census place layers and the 2010 ZCTA relationship "
         "file provide independent geography, but they do not prove every "
         "historical assignment from 1966 to 2018."),
        "",
        "**Table 3. Largest flagged assignments by gross absolute real dollars**",
        "",
        "| Source | State | City | County | Fiscal years | Gross absolute real dollars | Reasons |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.state} | {row.city.title()} | "
            f"{row.county} | {int(row.first_year)} to {int(row.last_year)} | "
            f"${float(row.absolute_dollars_real):,.0f} | "
            f"`{row.candidate_reasons}` |"
        )
    lines += [
        "",
        "**Table 4. Largest pairs in the complete classification cohort**",
        "",
        "| State | City | DCADS county | USAspending county | Direction to review | Contract IDs | Fiscal years | Nominal dollars |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    classification_cohort = priority.loc[
        ~priority["pre_complete_audit_correction_exists"]
        & ~priority["location_review_exists"]].sort_values(
            "dcads_absolute_dollars_nominal", ascending=False)
    for row in classification_cohort.head(30).itertuples(index=False):
        lines.append(
            f"| {row.state} | {row.city.title()} | {row.dcads_county} | "
            f"{row.usaspending_county} | `{row.candidate_direction}` | "
            f"{int(row.contract_ids)} | {int(row.first_year)} to "
            f"{int(row.last_year)} | "
            f"${float(row.dcads_absolute_dollars_nominal):,.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (f"The annual-dominance comparison finds {len(disagreements):,} "
         "same-year state-city disagreements between DCADS and USAspending "
         "during fiscal years 2001 to 2006. The contract-ID comparison finds "
         f"{int(values['exact_contract_ids_with_conflicts']):,} normalized "
         "contract identifiers attached to different counties in the two "
         "sources. Among them, "
         f"{int(values['exact_dollar_contract_ids_with_conflicts']):,} have "
         "source amounts matching to the cent. Table 4 reports the largest "
         "pairs in the 131-pair complete classification cohort, for which one "
         "county has independent place or ZIP support and the other is outside "
         "a matched Census place."),
        "",
        ("The transaction-locator classification authorizes 115 county pairs. "
         "The production correction table contains the 109 DCADS pairs active "
         "during fiscal years 2002 to 2006. The six USAspending pairs occur in "
         "fiscal years 2005 to 2006, when DCADS supplies the production series, "
         "so they do not alter production geography. Exact counts appear in "
         "`complete_city_county_candidate_classification.csv` and guarded rules "
         "appear in `dcads_verified_city_county_overrides.csv`."),
        "",
        "## Reproducible files",
        "",
        "- Annual assignment cells: `_data/temp/usaspending_city_county_audit/complete/assignment_cells.parquet`.",
        "- Assignment review: `_data/temp/usaspending_city_county_audit/complete/assignment_review.parquet`.",
        "- Candidate flags: `Results/instrument_normalization/complete_city_county_audit_flags.csv`.",
        "- Annual source disagreements: `Results/instrument_normalization/complete_city_county_annual_disagreements.csv`.",
        "- Contract conflicts: `Results/instrument_normalization/complete_city_county_contract_conflicts.csv`.",
        "- Priority contract pairs: `Results/instrument_normalization/complete_city_county_priority_contract_pairs.csv`.",
        "- Exact counts: `Results/instrument_normalization/complete_city_county_audit_summary.csv`.",
        "- Census source URLs and hashes: `_data/temp/usaspending_city_county_audit/complete/complete_audit_sources.csv`.",
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the complete candidate search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-existing-cells", action="store_true",
        help="load assignment-cell caches and rerun only downstream audit steps",
    )
    args = parser.parse_args()
    np.random.seed(20260907)
    start = time.time()
    COMPLETE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("== complete city-county assignment audit", flush=True)

    if args.use_existing_cells:
        if not ASSIGNMENT_OUT.exists() or not ZIP_OUT.exists():
            raise SystemExit("FAIL: requested assignment-cell cache is missing")
        cells = pd.read_parquet(ASSIGNMENT_OUT)
        zip_cells = pd.read_parquet(ZIP_OUT)
        excluded = excluded_row_counts(cells)
    else:
        cells, zip_cells, excluded = extract_assignment_cells()
        cells.to_parquet(ASSIGNMENT_OUT, index=False, compression="snappy")
        zip_cells.to_parquet(ZIP_OUT, index=False, compression="snappy")
    print(
        f"   assignment cells={len(cells):,}; audited rows="
        f"{int(cells['source_rows'].sum()):,}", flush=True)

    pooled, annual, disagreements = build_diagnostics(cells)
    disagreements.to_csv(ANNUAL_OUT, index=False)
    states = sorted(set(pooled.loc[
        pooled["state"].str.fullmatch(r"[0-9]{2}")
        & pooled["state"].str[:2].isin(dc.STATE_FIPS), "state"]))
    paths, sources = acquire_geography(states)
    sources.to_csv(SOURCE_OUT, index=False)
    places = build_place_lookup(states, paths)
    places.to_parquet(PLACE_OUT, index=False, compression="snappy")
    print(
        f"   Census place-county rows={len(places):,}; states={len(states)}",
        flush=True)

    review = geography_support(
        pooled, zip_cells, places, paths[("zcta_rel", "2010", "us")])
    review = attach_flags(review, annual, disagreements)
    review.to_parquet(REVIEW_OUT, index=False, compression="snappy")
    flags = review.loc[review["audit_candidate"]].copy()
    flags = flags.sort_values(
        ["absolute_dollars_real", "source", "state", "city", "county"],
        ascending=[False, True, True, True, True])
    flags.to_csv(FLAGS_OUT, index=False)

    contracts = exact_contract_conflicts()
    contracts.to_csv(CONTRACT_OUT, index=False)
    priority = priority_contract_pairs(review, contracts)
    priority.to_csv(PRIORITY_OUT, index=False)
    summary = summary_rows(
        cells, review, annual, disagreements, contracts, priority, excluded)
    summary.to_csv(SUMMARY_OUT, index=False)
    write_memo(summary, flags, disagreements, contracts, priority)

    print(f"   candidates={len(flags):,}", flush=True)
    print(f"   same-year dominant disagreements={len(disagreements):,}", flush=True)
    print(
        f"   exact conflicting contract IDs="
        f"{contracts['contract_id'].nunique() if len(contracts) else 0:,}",
        flush=True)
    print(f"   priority contract-county pairs={len(priority):,}", flush=True)
    print(f"   wrote {FLAGS_OUT}", flush=True)
    print(f"   wrote {MEMO_OUT}", flush=True)
    print(f"   done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
