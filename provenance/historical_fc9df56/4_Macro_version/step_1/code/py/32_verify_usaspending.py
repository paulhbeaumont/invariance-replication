#!/usr/bin/env python3
"""Verify annual USAspending transaction parquets before county aggregation.

The checks cover transaction identity, fiscal timing, award family, agency,
geography mapping, state-code consistency, and the independent Pallante
state-year benchmark. Every excluded geography class is reported in actions,
signed dollars, and absolute dollars.

Output:
    dc.USASPENDING_OUT_DIR/usaspending_verification_report.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 2007
LAST_YEAR = 2018
AWARD_TYPES = {"A", "B", "C", "D"}
PALLANTE_RATIO_BAND = (0.95, 1.05)
UNRESOLVED_ABS_SHARE_MAX = 0.01
MISSING_COUNTRY_ABS_SHARE_MAX = 0.001
GEO_CLASSES = (
    "county",
    "domestic_conflict",
    "domestic_territory",
    "domestic_unmapped",
    "domestic_missing_county",
    "domestic_invalid_county",
    "foreign",
    "missing_country",
)
BASE_FIELDS = (
    "fiscal_year",
    "rows",
    "unique_transactions",
    "duplicate_transactions",
    "award_rows",
    "idv_rows",
    "raw_dollars",
    "award_dollars",
    "county_dollars",
    "pallante_dollars",
    "county_to_pallante_ratio",
    "muratori_dollars",
    "county_to_muratori_ratio",
    "unresolved_abs_share",
    "missing_country_abs_share",
    "state_fips_mismatches",
    "counties",
    "ccgs",
    "status",
)
GEO_FIELDS = tuple(
    f"{label}_{suffix}"
    for label in GEO_CLASSES
    for suffix in ("actions", "dollars", "absolute_dollars")
)
REPORT_FIELDS = BASE_FIELDS + GEO_FIELDS
READ_COLUMNS = (
    "contract_transaction_unique_key",
    "federal_action_obligation",
    "action_date",
    "action_date_fiscal_year",
    "awarding_agency_code",
    "primary_place_of_performance_country_code",
    "prime_award_transaction_place_of_performance_county_fips_code",
    "prime_award_transaction_place_of_performance_state_fips_code",
    "award_or_idv_flag",
    "award_type_code",
    "county_fips",
    "ccg",
    "geo_class",
    "county_fips_source",
)


def parse_years(text):
    if not text:
        return list(range(FIRST_YEAR, LAST_YEAR + 1))
    years = set()
    for item in text.split(","):
        item = item.strip()
        if "-" in item:
            first, last = map(int, item.split("-", 1))
            years.update(range(first, last + 1))
        else:
            years.add(int(item))
    return sorted(years)


def pallante_totals():
    rows = dc.read_pallante_csv("data_replication/aggregates/all_dod_statename_noadj.csv")
    if rows is None:
        raise FileNotFoundError("Pallante state-year benchmark is not available")
    totals = {}
    for row in rows:
        year = int(row["fy_year"])
        totals[year] = totals.get(year, 0.0) + float(row["tot_spending"] or 0)
    return totals


def muratori_totals():
    path = dc.PANEL_OUT_DIR / "contracts_muratori_ccg_year.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["year", "dollars_nominal"])
    return frame.groupby("year")["dollars_nominal"].sum().to_dict()


def add_failure(failures, year, label, detail):
    failures.append((year, label, detail))
    print(f"    FAIL {label}: {detail}")


def verify_year(year, pallante, muratori, failures):
    path = dc.USASPENDING_OUT_DIR / f"fy{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=list(READ_COLUMNS))
    dollars = frame["federal_action_obligation"]
    award = frame["award_or_idv_flag"].eq("AWARD")
    idv = frame["award_or_idv_flag"].eq("IDV")
    county_award = award & frame["geo_class"].eq("county")

    row = {field: 0 for field in REPORT_FIELDS}
    duplicates = int(frame["contract_transaction_unique_key"].duplicated().sum())
    county_dollars = float(dollars[county_award].sum())
    pallante_dollars = float(pallante.get(year, float("nan")))
    muratori_dollars = float(muratori.get(year, float("nan")))
    row.update(
        {
            "fiscal_year": year,
            "rows": len(frame),
            "unique_transactions": frame["contract_transaction_unique_key"].nunique(),
            "duplicate_transactions": duplicates,
            "award_rows": int(award.sum()),
            "idv_rows": int(idv.sum()),
            "raw_dollars": float(dollars.sum()),
            "award_dollars": float(dollars[award].sum()),
            "county_dollars": county_dollars,
            "pallante_dollars": pallante_dollars,
            "county_to_pallante_ratio": county_dollars / pallante_dollars,
            "muratori_dollars": muratori_dollars,
            "county_to_muratori_ratio": county_dollars / muratori_dollars,
            "counties": frame.loc[county_award, "county_fips"].nunique(),
            "ccgs": frame.loc[county_award, "ccg"].nunique(),
        }
    )
    for label in GEO_CLASSES:
        mask = award & frame["geo_class"].eq(label)
        row[f"{label}_actions"] = int(mask.sum())
        row[f"{label}_dollars"] = float(dollars[mask].sum())
        row[f"{label}_absolute_dollars"] = float(dollars[mask].abs().sum())

    award_absolute = float(dollars[award].abs().sum())
    unresolved_labels = {
        "domestic_conflict", "domestic_unmapped", "domestic_missing_county",
        "domestic_invalid_county",
    }
    unresolved_abs = sum(row[f"{label}_absolute_dollars"] for label in unresolved_labels)
    row["unresolved_abs_share"] = unresolved_abs / award_absolute
    row["missing_country_abs_share"] = (
        row["missing_country_absolute_dollars"] / award_absolute
    )

    state = frame["prime_award_transaction_place_of_performance_state_fips_code"].str.strip()
    state_comparable = (
        county_award
        & frame["county_fips_source"].eq("reported_fips")
        & state.str.fullmatch(r"\d{2}", na=False)
    )
    state_mismatch = state_comparable & frame["county_fips"].str[:2].ne(state)
    row["state_fips_mismatches"] = int(state_mismatch.sum())

    before = len(failures)
    if duplicates:
        add_failure(failures, year, "transaction uniqueness", f"{duplicates:,} duplicates")
    if frame["contract_transaction_unique_key"].eq("").any():
        add_failure(failures, year, "transaction identity", "blank transaction keys")
    if not frame["action_date_fiscal_year"].eq(year).all():
        add_failure(failures, year, "fiscal timing", "action fiscal year differs from archive")
    if frame["action_date"].eq("").any():
        add_failure(failures, year, "action date", "blank action dates")
    agencies = sorted(set(frame["awarding_agency_code"]) - {"097"})
    if agencies:
        add_failure(failures, year, "awarding agency", f"unexpected codes {agencies[:10]}")
    other_flags = sorted(set(frame["award_or_idv_flag"]) - {"AWARD", "IDV"})
    if other_flags:
        add_failure(failures, year, "award flag", f"unexpected values {other_flags}")
    bad_types = sorted(set(frame.loc[award, "award_type_code"]) - AWARD_TYPES)
    if bad_types:
        add_failure(failures, year, "award type", f"unexpected award types {bad_types}")
    mapping_error = frame["geo_class"].eq("county").ne(frame["ccg"].notna())
    if mapping_error.any():
        add_failure(
            failures, year, "county mapping", f"{int(mapping_error.sum()):,} inconsistent rows"
        )
    bad_sources = sorted(
        set(frame.loc[frame["geo_class"].eq("county"), "county_fips_source"])
        - {"reported_fips", "state_county_name", "city_county_override"}
    )
    if bad_sources:
        add_failure(failures, year, "county mapping source", f"unexpected values {bad_sources}")
    if row["state_fips_mismatches"]:
        add_failure(
            failures,
            year,
            "state-county identity",
            f"{row['state_fips_mismatches']:,} county prefixes disagree with state FIPS",
        )
    if row["unresolved_abs_share"] > UNRESOLVED_ABS_SHARE_MAX:
        add_failure(
            failures,
            year,
            "unresolved domestic geography",
            f"{100 * row['unresolved_abs_share']:.3f}% of absolute award dollars",
        )
    if row["missing_country_abs_share"] > MISSING_COUNTRY_ABS_SHARE_MAX:
        add_failure(
            failures,
            year,
            "missing country",
            f"{100 * row['missing_country_abs_share']:.3f}% of absolute award dollars",
        )
    lo, hi = PALLANTE_RATIO_BAND
    if not lo <= row["county_to_pallante_ratio"] <= hi:
        add_failure(
            failures,
            year,
            "Pallante benchmark",
            f"ratio {row['county_to_pallante_ratio']:.4f} outside [{lo}, {hi}]",
        )
    row["status"] = "PASS" if len(failures) == before else "FAIL"
    print(
        f"    {len(frame):,} rows, {duplicates:,} duplicate keys, "
        f"county ${county_dollars / 1e9:,.3f}bn, "
        f"Pallante ratio {row['county_to_pallante_ratio']:.4f}, {row['status']}"
    )
    return row


def write_report(rows, report_name):
    path = dc.USASPENDING_OUT_DIR / report_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", help="comma-separated years or ranges; default FY2007-2018")
    parser.add_argument(
        "--report-name",
        default="usaspending_verification_report.csv",
        help="CSV basename under the shared USAspending output directory",
    )
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name or not args.report_name.endswith(".csv"):
        raise ValueError("--report-name must be a CSV basename")
    years = parse_years(args.years)
    pallante = pallante_totals()
    muratori = muratori_totals()
    failures = []
    rows = []

    print(f"USAspending verification: FY{years[0]}-{years[-1]}")
    for year in years:
        print(f"  FY{year}")
        rows.append(verify_year(year, pallante, muratori, failures))
    report = write_report(rows, args.report_name)
    print(f"  {len(rows)} rows -> {report}")
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed", file=sys.stderr)
        for year, label, detail in failures:
            print(f"  FY{year} {label}: {detail}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
