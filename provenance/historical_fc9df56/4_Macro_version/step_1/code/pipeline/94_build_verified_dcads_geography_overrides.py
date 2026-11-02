"""Build guarded DCADS geography overrides from the complete audit.

The correction universe contains verified DCADS county pairs with exact
same-contract, same-year, same-dollar evidence and independent Census-place and
performance-ZIP support. Unique city-year-county assignments apply to every
matching DCADS row. Bethlehem, Pennsylvania, has two supported destination
counties in fiscal year 2006, so its rules additionally match the exact contract
identifier.

Writes step_1/code/reference/
    dcads_verified_city_county_overrides.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: Results/instrument_normalization/complete_city_county_candidate_classification.csv
# mpgr-output-cache: step_1/code/reference/dcads_verified_city_county_overrides.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|verified-dcads-geography|guarded-overrides

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
STEP1_PY = PIPELINE.parent / "py"
sys.path.insert(0, str(STEP1_PY))
import dcads_common as dc  # noqa: E402


RESULTS = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CLASSIFICATION_IN = RESULTS / "complete_city_county_candidate_classification.csv"
CONFLICTS_IN = RESULTS / "complete_city_county_contract_conflicts.csv"
OUT = PIPELINE.parent / "reference" / "dcads_verified_city_county_overrides.csv"
YEARS = tuple(range(2001, 2007))
AMBIGUOUS_KEY = (2006, "42", "BETHLEHEM", "42033")
OUTPUT_COLUMNS = [
    "fiscal_year", "state_code", "city_name", "reported_county_fips",
    "contract_id", "corrected_county_fips", "expected_rows", "evidence",
]


def truth(series: pd.Series) -> pd.Series:
    """Read persisted CSV booleans without relying on dtype inference."""
    return series.astype(str).str.lower().eq("true")


def normalize_contract(value: object) -> str:
    """Normalize one DCADS contract number for an exact override key."""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper()).strip()


def rule_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build city-year and contract-specific correction keys."""
    dtype = {
        "state": "string", "dcads_county": "string",
        "usaspending_county": "string", "suspect_county": "string",
        "alternative_county": "string", "contract_id": "string",
    }
    classified = pd.read_csv(CLASSIFICATION_IN, dtype=dtype)
    classified = classified.loc[
        truth(classified["production_correction_authorized"])
        & classified["suspect_source"].eq("DCADS")
    ].copy()
    if len(classified) != 109:
        raise SystemExit(
            f"FAIL: expected 109 verified DCADS pairs, found {len(classified)}")

    conflicts = pd.read_csv(CONFLICTS_IN, dtype=dtype)
    conflicts = conflicts.loc[truth(conflicts["net_dollars_match_to_cent"])]
    pair = ["state", "city", "dcads_county", "usaspending_county"]
    exact = conflicts.merge(
        classified[pair + ["suspect_county", "alternative_county"]],
        on=pair, how="inner", validate="many_to_one")
    exact["year"] = pd.to_numeric(exact["year"], errors="raise").astype(int)
    exact["state"] = exact["state"].str.zfill(2)
    exact["suspect_county"] = exact["suspect_county"].str.zfill(5)
    exact["alternative_county"] = exact["alternative_county"].str.zfill(5)
    exact["contract_id"] = exact["contract_id"].map(normalize_contract)

    annual = exact[[
        *pair, "year", "suspect_county", "alternative_county",
    ]].drop_duplicates()
    if len(annual) != 131 or annual[pair].drop_duplicates().shape[0] != 109:
        raise SystemExit("FAIL: verified DCADS pair-year universe drifted")
    key = ["year", "state", "city", "suspect_county"]
    ambiguous = annual.duplicated(key, keep=False)
    ambiguous_keys = set(map(tuple, annual.loc[ambiguous, key].itertuples(
        index=False, name=None)))
    if ambiguous_keys != {AMBIGUOUS_KEY}:
        raise SystemExit(
            f"FAIL: unexpected ambiguous correction keys {sorted(ambiguous_keys)}")

    city = annual.loc[
        ~ambiguous, key + ["alternative_county"]].drop_duplicates()
    contract = exact.loc[
        exact[key].apply(tuple, axis=1).isin(ambiguous_keys),
        key + ["contract_id", "alternative_county"],
    ].drop_duplicates()
    if contract.duplicated(key + ["contract_id"]).any():
        raise SystemExit("FAIL: a Bethlehem contract has two corrected counties")
    if len(city) != 129 or len(contract) != 4:
        raise SystemExit(
            f"FAIL: expected 129 city rules and 4 contract rules, found "
            f"{len(city)} and {len(contract)}")
    return city, contract


def source_counts(
        city: pd.DataFrame, contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach exact non-lump-sum source-row counts to every rule."""
    paths = [str(dc.OUT_DIR / f"fy{year}.parquet") for year in YEARS]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"FAIL: missing DCADS source parquets: {missing}")
    connection = duckdb.connect()
    connection.register("city_rules", city)
    connection.register("contract_rules", contract)
    city_hits = connection.execute("""
        WITH raw AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                LPAD(TRIM(CAST(state_code AS VARCHAR)), 2, '0') AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                CASE
                    WHEN CAST(fiscal_year AS INTEGER) BETWEEN 2001 AND 2003
                    THEN LPAD(TRIM(CAST(county_code AS VARCHAR)), 5, '0')
                    ELSE LPAD(TRIM(CAST(state_code AS VARCHAR)), 2, '0')
                         || LPAD(TRIM(CAST(county_fips AS VARCHAR)), 3, '0')
                END AS suspect_county
            FROM read_parquet(?, union_by_name=true)
            WHERE NOT CAST(is_lumpsum AS BOOLEAN)
        )
        SELECT c.*, COUNT(*) AS expected_rows
        FROM raw r
        INNER JOIN city_rules c
          USING (year, state, city, suspect_county)
        GROUP BY ALL
    """, [paths]).df()
    contract_hits = connection.execute("""
        WITH raw AS (
            SELECT
                CAST(fiscal_year AS INTEGER) AS year,
                LPAD(TRIM(CAST(state_code AS VARCHAR)), 2, '0') AS state,
                TRIM(REGEXP_REPLACE(UPPER(TRIM(CAST(city_name AS VARCHAR))),
                                    '[^A-Z0-9]+', ' ', 'g')) AS city,
                CASE
                    WHEN CAST(fiscal_year AS INTEGER) BETWEEN 2001 AND 2003
                    THEN LPAD(TRIM(CAST(county_code AS VARCHAR)), 5, '0')
                    ELSE LPAD(TRIM(CAST(state_code AS VARCHAR)), 2, '0')
                         || LPAD(TRIM(CAST(county_fips AS VARCHAR)), 3, '0')
                END AS suspect_county,
                REGEXP_REPLACE(UPPER(TRIM(CAST(contract_no AS VARCHAR))),
                               '[^A-Z0-9]+', '', 'g') AS contract_id
            FROM read_parquet(?, union_by_name=true)
            WHERE NOT CAST(is_lumpsum AS BOOLEAN)
        )
        SELECT c.*, COUNT(*) AS expected_rows
        FROM raw r
        INNER JOIN contract_rules c
          USING (year, state, city, suspect_county, contract_id)
        GROUP BY ALL
    """, [paths]).df()
    connection.close()
    if len(city_hits) != len(city) or len(contract_hits) != len(contract):
        raise SystemExit("FAIL: at least one verified override has no source row")
    if int(city_hits["expected_rows"].sum()
           + contract_hits["expected_rows"].sum()) != 12_865:
        raise SystemExit("FAIL: verified overrides do not cover 12,865 source rows")
    return city_hits, contract_hits


def output_rows(city: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    """Format guarded correction rows for the production geography parser."""
    evidence = (
        "Exact common-year contract amount, Census place in 2000 and 2010, "
        "and USAspending performance ZIP; see "
        "COMPLETE_CITY_COUNTY_CANDIDATE_CLASSIFICATION.md"
    )
    city = city.rename(columns={
        "year": "fiscal_year", "state": "state_code",
        "city": "city_name",
        "suspect_county": "reported_county_fips",
        "alternative_county": "corrected_county_fips",
    })
    city["contract_id"] = ""
    contract = contract.rename(columns={
        "year": "fiscal_year", "state": "state_code",
        "city": "city_name",
        "suspect_county": "reported_county_fips",
        "alternative_county": "corrected_county_fips",
    })
    result = pd.concat([city, contract], ignore_index=True)
    result["evidence"] = evidence
    result = result[OUTPUT_COLUMNS].sort_values([
        "fiscal_year", "state_code", "city_name", "reported_county_fips",
        "contract_id", "corrected_county_fips",
    ]).reset_index(drop=True)
    keys = [
        "fiscal_year", "state_code", "city_name", "reported_county_fips",
        "contract_id",
    ]
    if result.duplicated(keys).any() or len(result) != 133:
        raise SystemExit("FAIL: production override keys are not unique")
    return result


def main() -> None:
    """Write the guarded systematic correction table."""
    city, contract = rule_universe()
    city, contract = source_counts(city, contract)
    result = output_rows(city, contract)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(
        f"wrote {OUT}: {len(result)} guarded rules, "
        f"{int(result['expected_rows'].sum()):,} source rows")


if __name__ == "__main__":
    main()
