"""Classify the complete city-county candidate cohort with transaction locators.

The input universe is the exact-dollar priority list produced by
91_complete_city_county_assignment_audit.py. Each county pair is evaluated
against the 2000 and 2010 Census place intersections and the 2010
ZCTA-to-county relationship for the performance ZIPs attached to the matched
USAspending contract identifiers. The rules separate verified assignment
errors, postal assignments, and unresolved historical geography. This producer
does not modify production geography.

Writes Results/instrument_normalization/
    complete_city_county_candidate_contract_zip_evidence.csv
    complete_city_county_candidate_classification.csv
and Instrument_analysis/Cleaning_spendingData/
    COMPLETE_CITY_COUNTY_CANDIDATE_CLASSIFICATION.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/usaspending_city_county_audit/complete/assignment_review.parquet
# mpgr-output-cache: Results/instrument_normalization/complete_city_county_candidate_classification.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|2001-2006|transaction-locator-candidate-classification

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


COMMON_YEARS = tuple(range(2001, 2007))
CACHE_DIR = (
    dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit" / "complete"
)
RESULTS_DIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
PRIORITY_IN = RESULTS_DIR / "complete_city_county_priority_contract_pairs.csv"
CONTRACT_IN = RESULTS_DIR / "complete_city_county_contract_conflicts.csv"
PLACE_IN = CACHE_DIR / "place_county_lookup.parquet"
ZCTA_IN = (
    dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
    / "historical_tiger" / "zcta_county_rel_10.txt"
)
ZIP_EVIDENCE_OUT = (
    RESULTS_DIR / "complete_city_county_candidate_contract_zip_evidence.csv"
)
CLASSIFICATION_OUT = (
    RESULTS_DIR / "complete_city_county_candidate_classification.csv"
)
MEMO_OUT = (
    dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
    / "Cleaning_spendingData"
    / "COMPLETE_CITY_COUNTY_CANDIDATE_CLASSIFICATION.md"
)


def source_paths(source: str) -> list[str]:
    """Return and verify common-year source parquets."""
    directory = dc.OUT_DIR if source == "DCADS" else dc.USASPENDING_OUT_DIR
    paths = [directory / f"fy{year}.parquet" for year in COMMON_YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"FAIL: {source} is missing {len(missing)} files")
    return [str(path) for path in paths]


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the stable 131-pair cohort and exact-dollar matched contracts."""
    dtype = {
        "state": "string", "dcads_county": "string",
        "usaspending_county": "string", "contract_id": "string",
    }
    priority = pd.read_csv(PRIORITY_IN, dtype=dtype)
    pre_complete_audit = priority[
        "pre_complete_audit_correction_exists"].astype(
        str).str.lower().eq("true")
    location_review = priority["location_review_exists"].astype(
        str).str.lower().eq("true")
    priority = priority.loc[
        ~pre_complete_audit & ~location_review
    ].copy()
    if len(priority) != 131:
        raise SystemExit(
            f"FAIL: classification cohort is {len(priority)}, expected 131")

    contracts = pd.read_csv(CONTRACT_IN, dtype=dtype)
    contracts = contracts.loc[contracts["net_dollars_match_to_cent"]].copy()
    keys = ["state", "city", "dcads_county", "usaspending_county"]
    exact = contracts.merge(
        priority[keys], on=keys, how="inner", validate="many_to_one")
    observed = set(map(tuple, exact[keys].drop_duplicates().itertuples(
        index=False, name=None)))
    expected = set(map(tuple, priority[keys].itertuples(index=False, name=None)))
    if observed != expected:
        raise SystemExit(
            f"FAIL: exact contract evidence misses {sorted(expected - observed)}")
    return priority, exact


def extract_transaction_locators(exact: pd.DataFrame) -> pd.DataFrame:
    """Extract USA ZIPs and source county names for exact contract matches."""
    keys = exact[[
        "year", "state", "city", "contract_id", "dcads_county",
        "usaspending_county",
    ]].drop_duplicates().copy()
    connection = duckdb.connect()
    connection.register("candidate_contracts", keys)
    usa = connection.execute("""
        WITH raw AS (
            SELECT
                CAST(source_fiscal_year AS INTEGER) AS year,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_state_fips_code
                    AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(
                    primary_place_of_performance_city_name AS VARCHAR))),
                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                REGEXP_REPLACE(UPPER(TRIM(CAST(award_id_piid AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id,
                TRIM(CAST(
                    prime_award_transaction_place_of_performance_county_fips_code
                    AS VARCHAR)) AS county,
                TRIM(CAST(primary_place_of_performance_county_name AS VARCHAR))
                    AS county_name,
                SUBSTR(REGEXP_REPLACE(CAST(
                    primary_place_of_performance_zip_4 AS VARCHAR),
                    '[^0-9]+', '', 'g'), 1, 5) AS performance_zip,
                CAST(federal_action_obligation AS DOUBLE) AS dollars_nominal
            FROM read_parquet(?, union_by_name=true)
            WHERE award_or_idv_flag = 'AWARD'
              AND primary_place_of_performance_country_code = 'USA'
        )
        SELECT
            k.year, k.state, k.city, k.contract_id,
            k.dcads_county, k.usaspending_county,
            r.county AS source_county, r.county_name,
            r.performance_zip, COUNT(*) AS source_rows,
            SUM(r.dollars_nominal) AS net_dollars_nominal,
            SUM(ABS(r.dollars_nominal)) AS absolute_dollars_nominal
        FROM raw r
        INNER JOIN candidate_contracts k
          ON r.year = k.year AND r.state = k.state AND r.city = k.city
         AND r.contract_id = k.contract_id
         AND r.county = k.usaspending_county
        GROUP BY ALL
    """, [source_paths("USAspending")]).df()

    dcads = connection.execute("""
        WITH raw AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                TRIM(CAST(state_fips2 AS VARCHAR)) AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                REGEXP_REPLACE(UPPER(TRIM(CAST(contract_no AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id,
                CASE
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 2001 AND 2003
                    THEN TRIM(CAST(county_code AS VARCHAR))
                    WHEN fips_source = 'manual_override'
                         AND CAST(fiscal_year AS INTEGER) BETWEEN 2004 AND 2006
                    THEN TRIM(CAST(state_code AS VARCHAR))
                         || TRIM(CAST(county_fips AS VARCHAR))
                    ELSE TRIM(CAST(county_fips5 AS VARCHAR))
                END AS county,
                TRIM(CAST(county_name AS VARCHAR)) AS county_name,
                CASE WHEN dollars_unit = 'thousands'
                     THEN CAST(dollars AS DOUBLE) * 1000.0
                     ELSE CAST(dollars AS DOUBLE) END AS dollars_nominal
            FROM read_parquet(?, union_by_name=true)
            WHERE NOT CAST(is_lumpsum AS BOOLEAN)
        )
        SELECT
            k.year, k.state, k.city, k.contract_id,
            k.dcads_county, k.usaspending_county,
            r.county AS source_county, r.county_name,
            COUNT(*) AS source_rows,
            SUM(r.dollars_nominal) AS net_dollars_nominal,
            SUM(ABS(r.dollars_nominal)) AS absolute_dollars_nominal
        FROM raw r
        INNER JOIN candidate_contracts k
          ON r.year = k.year AND r.state = k.state AND r.city = k.city
         AND r.contract_id = k.contract_id
         AND r.county = k.dcads_county
        GROUP BY ALL
    """, [source_paths("DCADS")]).df()
    connection.close()

    usa["source"] = "USAspending"
    dcads["source"] = "DCADS"
    dcads["performance_zip"] = ""
    columns = [
        "source", "year", "state", "city", "contract_id",
        "dcads_county", "usaspending_county", "source_county",
        "county_name", "performance_zip", "source_rows",
        "net_dollars_nominal", "absolute_dollars_nominal",
    ]
    return pd.concat([dcads[columns], usa[columns]], ignore_index=True)


def zcta_contract_evidence(
        actions: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    """Map each exact matched contract's performance ZIPs to counties."""
    usa = actions.loc[actions["source"].eq("USAspending")].copy()
    relations = pd.read_csv(
        ZCTA_IN, dtype=str,
        usecols=["ZCTA5", "GEOID", "ZPOPPCT", "ZAREAPCT"],
    ).rename(columns={
        "ZCTA5": "performance_zip", "GEOID": "zcta_county",
        "ZPOPPCT": "zcta_population_share", "ZAREAPCT": "zcta_area_share",
    })
    evidence = usa.merge(
        relations, on="performance_zip", how="left", validate="many_to_many")
    evidence["zip_maps_dcads_county"] = evidence["zcta_county"].eq(
        evidence["dcads_county"])
    evidence["zip_maps_usaspending_county"] = evidence["zcta_county"].eq(
        evidence["usaspending_county"])

    expected_ids = exact[[
        "year", "state", "city", "contract_id", "dcads_county",
        "usaspending_county",
    ]].drop_duplicates()
    observed = evidence[[
        "year", "state", "city", "contract_id", "dcads_county",
        "usaspending_county",
    ]].drop_duplicates()
    if len(observed.merge(expected_ids, how="outer", indicator=True).query(
            "_merge != 'both'")):
        raise SystemExit("FAIL: USA locator extraction misses exact contract IDs")
    return evidence.sort_values([
        "state", "city", "contract_id", "performance_zip", "zcta_county",
    ])


def place_sets(places: pd.DataFrame, state: str, city: str, vintage: int) -> set[str]:
    """Return official counties for one exact place name and vintage."""
    selected = places.loc[
        places["state"].eq(state) & places["city"].eq(city)
        & places["vintage"].eq(vintage), "county"]
    return set(selected.astype(str))


def semicolon(values: set[str]) -> str:
    """Join a set in stable order."""
    return ";".join(sorted(value for value in values if value))


def classify(
        priority: pd.DataFrame, exact: pd.DataFrame,
        zip_evidence: pd.DataFrame) -> pd.DataFrame:
    """Apply the stated valid, verified-error, and unresolved rules."""
    places = pd.read_parquet(PLACE_IN)
    rows = []
    pair_keys = ["state", "city", "dcads_county", "usaspending_county"]
    for case in priority.itertuples(index=False):
        key_mask = (
            exact["state"].eq(case.state) & exact["city"].eq(case.city)
            & exact["dcads_county"].eq(case.dcads_county)
            & exact["usaspending_county"].eq(case.usaspending_county)
        )
        contracts = exact.loc[key_mask]
        zmask = (
            zip_evidence["state"].eq(case.state)
            & zip_evidence["city"].eq(case.city)
            & zip_evidence["dcads_county"].eq(case.dcads_county)
            & zip_evidence["usaspending_county"].eq(case.usaspending_county)
        )
        zips = zip_evidence.loc[zmask]
        p00 = place_sets(places, case.state, case.city, 2000)
        p10 = place_sets(places, case.state, case.city, 2010)

        if case.candidate_direction == "review_dcads_assignment":
            suspect_source = "DCADS"
            suspect_county = case.dcads_county
            alternative_county = case.usaspending_county
            target_zip_column = "zip_maps_usaspending_county"
            suspect_zip_column = "zip_maps_dcads_county"
        else:
            suspect_source = "USAspending"
            suspect_county = case.usaspending_county
            alternative_county = case.dcads_county
            target_zip_column = "zip_maps_dcads_county"
            suspect_zip_column = "zip_maps_usaspending_county"

        zips = zips.copy()
        zips["mapped_zip_row"] = zips["zcta_county"].notna()
        by_id = (zips.groupby("contract_id", sort=False)
                 .agg(mapped_zip=("mapped_zip_row", "max"),
                      target_zip=(target_zip_column, "max"),
                      suspect_zip=(suspect_zip_column, "max"))
                 .reset_index())
        mapped = by_id.loc[by_id["mapped_zip"]]
        target_only = int((mapped["target_zip"] & ~mapped["suspect_zip"]).sum())
        suspect_only = int((mapped["suspect_zip"] & ~mapped["target_zip"]).sum())
        both = int((mapped["target_zip"] & mapped["suspect_zip"]).sum())
        neither = int((~mapped["target_zip"] & ~mapped["suspect_zip"]).sum())

        alternative_stable_place = (
            alternative_county in p00 and alternative_county in p10)
        suspect_any_place = suspect_county in p00 or suspect_county in p10
        both_legitimate_place = (
            suspect_county in p00 and alternative_county in p00
            and suspect_county in p10 and alternative_county in p10)

        if both_legitimate_place:
            status = "valid_multicounty_place"
        elif suspect_only > 0 and target_only == 0:
            status = "valid_postal_assignment"
        elif (alternative_stable_place and not suspect_any_place
              and target_only > 0 and suspect_only == 0 and both == 0):
            status = "verified_assignment_error"
        elif (alternative_stable_place and not suspect_any_place
              and target_only == 0 and suspect_only == 0 and both == 0):
            status = "probable_error_missing_transaction_locator"
        else:
            status = "historical_or_postal_geography_unresolved"

        rows.append({
            **{key: getattr(case, key) for key in pair_keys},
            "suspect_source": suspect_source,
            "suspect_county": suspect_county,
            "alternative_county": alternative_county,
            "exact_contract_ids": int(contracts["contract_id"].nunique()),
            "exact_contract_years": int(contracts["year"].nunique()),
            "first_exact_year": int(contracts["year"].min()),
            "last_exact_year": int(contracts["year"].max()),
            "exact_nominal_dollars": float(
                contracts["dcads_absolute_dollars_nominal"].sum()),
            "census_place_2000_counties": semicolon(p00),
            "census_place_2010_counties": semicolon(p10),
            "contract_ids_with_mapped_zip": int(len(mapped)),
            "contract_ids_zip_supports_alternative_only": target_only,
            "contract_ids_zip_supports_suspect_only": suspect_only,
            "contract_ids_zip_supports_both": both,
            "contract_ids_zip_supports_neither": neither,
            "review_status": status,
            "production_correction_authorized": (
                status == "verified_assignment_error"),
        })
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["production_correction_authorized", "exact_nominal_dollars"],
        ascending=[False, False]).reset_index(drop=True)


def write_memo(classification: pd.DataFrame) -> None:
    """Write exact classification counts and location-level results."""
    counts = (classification.groupby("review_status", sort=True)
              .agg(county_pairs=("city", "size"))
              .reset_index())
    location_counts = (classification[["state", "city", "review_status"]]
                       .drop_duplicates().groupby("review_status").size())
    counts["locations"] = counts["review_status"].map(location_counts)
    verified = classification.loc[
        classification["review_status"].eq("verified_assignment_error")]
    lines = [
        "# Complete city-county candidate classification",
        "",
        "## Decision rules",
        "",
        ("A county pair is a verified assignment error when the same contract "
         "appears in both sources in the same fiscal year with the same amount, "
         "the alternative county contains the exact Census place in both 2000 "
         "and 2010, the suspect county contains it in neither vintage, and at "
         "least one mapped transaction ZIP supports only the alternative county. "
         "A ZIP supporting only the suspect county yields a valid postal "
         "assignment. Missing or mixed transaction locators remain unresolved."),
        "",
        "## Results",
        "",
        "**Table 1. Classification counts**",
        "",
        "| Status | County pairs | Locations |",
        "|---|---:|---:|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(
            f"| `{row.review_status}` | {int(row.county_pairs)} | "
            f"{int(row.locations)} |")
    lines += [
        "",
        "**Table 2. Verified assignment errors**",
        "",
        "| Suspect source | State | City | Suspect county | Alternative county | Exact contract IDs | Fiscal years | Exact nominal dollars | ZIP IDs supporting alternative only |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in verified.itertuples(index=False):
        lines.append(
            f"| {row.suspect_source} | {row.state} | {row.city.title()} | "
            f"{row.suspect_county} | {row.alternative_county} | "
            f"{int(row.exact_contract_ids)} | {int(row.first_exact_year)} to "
            f"{int(row.last_exact_year)} | "
            f"${float(row.exact_nominal_dollars):,.0f} | "
            f"{int(row.contract_ids_zip_supports_alternative_only)} |")
    lines += [
        "",
        "## Production status",
        "",
        ("The classification authorizes 115 county pairs. The guarded DCADS "
         "table contains the 109 pairs active during fiscal years 2002 to 2006, "
         "represented by 129 city-year rules and four exact-contract rules. "
         "Those rules cover 12,865 source rows. The six USAspending pairs occur "
         "in fiscal years 2005 to 2006, when DCADS supplies the production "
         "series, so they do not alter production geography."),
        "",
        "## Files",
        "",
        "- Classification: `Results/instrument_normalization/complete_city_county_candidate_classification.csv`.",
        "- Contract ZIP evidence: `Results/instrument_normalization/complete_city_county_candidate_contract_zip_evidence.csv`.",
        "- Guarded DCADS rules: `step_1/code/reference/dcads_verified_city_county_overrides.csv`.",
        "- First-stage and second-stage comparison: `Results/instrument_normalization/h2_geography_cleaning_ft_comparison.csv`.",
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the candidate classification."""
    np.random.seed(20260907)
    start = time.time()
    print("== classify complete city-county candidates", flush=True)
    priority, exact = load_candidates()
    actions = extract_transaction_locators(exact)
    zip_evidence = zcta_contract_evidence(actions, exact)
    classification = classify(priority, exact, zip_evidence)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_evidence.to_csv(ZIP_EVIDENCE_OUT, index=False)
    classification.to_csv(CLASSIFICATION_OUT, index=False)
    write_memo(classification)

    print("   status counts:", flush=True)
    for status, count in classification["review_status"].value_counts().items():
        print(f"      {status}: {count}", flush=True)
    print(f"   wrote {CLASSIFICATION_OUT}", flush=True)
    print(f"   wrote {MEMO_OUT}", flush=True)
    print(f"   done in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
