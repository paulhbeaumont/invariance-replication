#!/usr/bin/env python3
"""Parse USAspending DoD prime-contract archives to annual transaction parquets.

The annual source zips contain split CSV files with the same schema. This
producer streams selected accounting, identifier, date, and place-of-performance
fields, retains signed obligations and deobligations, and assigns every domestic
county code to the project's constant-county geography when possible.

Outputs:
    dc.USASPENDING_OUT_DIR/fy<year>.parquet
    dc.LOG_DIR/usaspending_parse_report.csv
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CHUNK_ROWS = 150_000
SELECTED = (
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    "award_id_piid",
    "modification_number",
    "transaction_number",
    "parent_award_agency_id",
    "parent_award_id_piid",
    "parent_award_modification_number",
    "federal_action_obligation",
    "total_dollars_obligated",
    "action_date",
    "action_date_fiscal_year",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "period_of_performance_potential_end_date",
    "ordering_period_end_date",
    "awarding_agency_code",
    "awarding_agency_name",
    "awarding_sub_agency_code",
    "awarding_sub_agency_name",
    "funding_agency_code",
    "funding_agency_name",
    "funding_sub_agency_code",
    "funding_sub_agency_name",
    "foreign_funding",
    "foreign_funding_description",
    "recipient_country_code",
    "prime_award_transaction_recipient_county_fips_code",
    "recipient_state_code",
    "primary_place_of_performance_country_code",
    "primary_place_of_performance_country_name",
    "primary_place_of_performance_city_name",
    "prime_award_transaction_place_of_performance_county_fips_code",
    "primary_place_of_performance_county_name",
    "prime_award_transaction_place_of_performance_state_fips_code",
    "primary_place_of_performance_state_code",
    "primary_place_of_performance_state_name",
    "primary_place_of_performance_zip_4",
    "award_or_idv_flag",
    "award_type_code",
    "award_type",
    "action_type_code",
    "action_type",
    "number_of_actions",
    "product_or_service_code",
    "dod_claimant_program_code",
    "naics_code",
    "domestic_or_foreign_entity_code",
    "country_of_product_or_service_origin_code",
    "place_of_manufacture_code",
    "transaction_description",
    "initial_report_date",
    "last_modified_date",
)
REPORT_FIELDS = (
    "fiscal_year",
    "source_archive",
    "zip_members",
    "rows",
    "dollars_nominal",
    "dollars_absolute",
    "negative_actions",
    "county_actions",
    "county_dollars",
    "name_resolved_actions",
    "name_resolved_dollars",
    "domestic_conflict_actions",
    "domestic_conflict_dollars",
    "domestic_territory_actions",
    "domestic_territory_dollars",
    "domestic_unmapped_actions",
    "domestic_unmapped_dollars",
    "domestic_missing_county_actions",
    "domestic_missing_county_dollars",
    "domestic_invalid_county_actions",
    "domestic_invalid_county_dollars",
    "foreign_actions",
    "foreign_dollars",
    "missing_country_actions",
    "missing_country_dollars",
)
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
CITY_COUNTY_OVERRIDE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reference" / "usaspending_city_county_overrides.csv"
)

STATE_NAME_TO_FIPS = {
    "ALABAMA": "01", "ALASKA": "02", "ARIZONA": "04", "ARKANSAS": "05",
    "CALIFORNIA": "06", "COLORADO": "08", "CONNECTICUT": "09",
    "DELAWARE": "10", "DISTRICT OF COLUMBIA": "11", "FLORIDA": "12",
    "GEORGIA": "13", "HAWAII": "15", "IDAHO": "16", "ILLINOIS": "17",
    "INDIANA": "18", "IOWA": "19", "KANSAS": "20", "KENTUCKY": "21",
    "LOUISIANA": "22", "MAINE": "23", "MARYLAND": "24",
    "MASSACHUSETTS": "25", "MICHIGAN": "26", "MINNESOTA": "27",
    "MISSISSIPPI": "28", "MISSOURI": "29", "MONTANA": "30",
    "NEBRASKA": "31", "NEVADA": "32", "NEW HAMPSHIRE": "33",
    "NEW JERSEY": "34", "NEW MEXICO": "35", "NEW YORK": "36",
    "NORTH CAROLINA": "37", "NORTH DAKOTA": "38", "OHIO": "39",
    "OKLAHOMA": "40", "OREGON": "41", "PENNSYLVANIA": "42",
    "RHODE ISLAND": "44", "SOUTH CAROLINA": "45", "SOUTH DAKOTA": "46",
    "TENNESSEE": "47", "TEXAS": "48", "UTAH": "49", "VERMONT": "50",
    "VIRGINIA": "51", "WASHINGTON": "53", "WEST VIRGINIA": "54",
    "WISCONSIN": "55", "WYOMING": "56",
}


def parse_years(text):
    if not text:
        return None
    years = set()
    for item in text.split(","):
        item = item.strip()
        if "-" in item:
            first, last = map(int, item.split("-", 1))
            years.update(range(first, last + 1))
        else:
            years.add(int(item))
    return years


def source_archives(selected, include_overlap):
    manifest = dc.USASPENDING_SOURCE_DIR / "manifest.csv"
    receipt = dc.USASPENDING_SOURCE_DIR / "download_receipt.csv"
    if not manifest.exists() or not receipt.exists():
        raise FileNotFoundError("run scripts 29 and 30 before parsing USAspending")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with receipt.open(newline="", encoding="utf-8") as handle:
        received = {row["file_name"]: row for row in csv.DictReader(handle)}
    out = []
    if include_overlap:
        overlap_dir = dc.USASPENDING_SOURCE_DIR / "overlap"
        overlap_receipt = overlap_dir / "download_receipt.json"
        if not overlap_receipt.exists():
            raise FileNotFoundError(
                f"{overlap_receipt} is missing; run 34_download_usaspending_overlap.py"
            )
        for row in json.loads(overlap_receipt.read_text(encoding="utf-8")):
            year = int(row["fiscal_year"])
            if selected is not None and year not in selected:
                continue
            path = overlap_dir / f"fy{year}" / row["file_name"]
            if not path.exists():
                raise FileNotFoundError(path)
            out.append((year, path))
    for row in rows:
        year = int(row["fiscal_year"])
        if selected is not None and year not in selected:
            continue
        if row["file_name"] not in received:
            raise AssertionError(f"{row['file_name']} has no verified download receipt")
        path = dc.USASPENDING_SOURCE_DIR / row["file_name"]
        if not path.exists():
            raise FileNotFoundError(path)
        out.append((year, path))
    if not out:
        raise AssertionError("no USAspending archives selected")
    return out


def csv_members(archive):
    members = [
        item for item in archive.infolist() if not item.is_dir() and item.filename.endswith(".csv")
    ]
    if not members:
        raise AssertionError("USAspending zip contains no CSV members")
    return members


def check_header(archive, member):
    with archive.open(member) as handle:
        header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
    columns = next(csv.reader([header]))
    missing = sorted(set(SELECTED) - set(columns))
    if missing:
        raise AssertionError(f"{member.filename}: required columns absent: {missing}")
    if len(columns) != len(set(columns)):
        raise AssertionError(f"{member.filename}: duplicate CSV headers")
    return columns


def normalize_place_name(series):
    return (
        series.str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(
            r" (COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY)$",
            "",
            regex=True,
        )
        .str.strip()
    )


def county_name_lookup(fips_to_ccg):
    xwalk = pd.read_csv(dc.CCG_CROSSWALK, dtype="string", keep_default_na=False)
    xwalk = xwalk[xwalk["county_fips5"].isin(fips_to_ccg)].copy()
    xwalk["state_fips"] = xwalk["county_fips5"].str[:2]
    xwalk["county_name"] = normalize_place_name(xwalk["ccg_name"])
    xwalk["name_count"] = xwalk.groupby(
        ["state_fips", "county_name"]
    )["county_fips5"].transform("nunique")
    unique = xwalk[xwalk["name_count"].eq(1)].drop_duplicates(
        ["state_fips", "county_name"]
    )
    return {
        (row.state_fips, row.county_name): row.county_fips5
        for row in unique.itertuples()
    }


def load_city_county_overrides(fips_to_ccg):
    """Load verified state-city corrections for reported county codes."""
    required = {
        "first_year", "last_year", "state_code", "city_name",
        "reported_county_fips", "corrected_county_fips", "evidence",
    }
    frame = pd.read_csv(
        CITY_COUNTY_OVERRIDE_PATH, dtype="string", keep_default_na=False)
    if set(frame.columns) != required:
        raise AssertionError(
            f"city-county override columns are {list(frame.columns)}")
    frame["first_year"] = pd.to_numeric(frame["first_year"], errors="raise")
    frame["last_year"] = pd.to_numeric(frame["last_year"], errors="raise")
    frame["state_code"] = frame["state_code"].str.strip().str.upper()
    frame["city_name"] = normalize_place_name(frame["city_name"])
    for column in ("reported_county_fips", "corrected_county_fips"):
        frame[column] = frame[column].str.strip()
        if not frame[column].str.fullmatch(r"\d{5}").all():
            raise AssertionError(f"invalid county code in {column}")
    if (frame["first_year"] > frame["last_year"]).any():
        raise AssertionError("city-county override has reversed year bounds")
    missing = sorted(set(frame["corrected_county_fips"]) - set(fips_to_ccg))
    if missing:
        raise AssertionError(f"override corrections are outside CCG map: {missing}")
    keys = ["first_year", "last_year", "state_code", "city_name",
            "reported_county_fips"]
    if frame.duplicated(keys).any():
        raise AssertionError("duplicate city-county override keys")
    return frame


def classify_geography(
        frame, year, fips_to_ccg, name_to_fips, city_county_overrides):
    country = frame["primary_place_of_performance_country_code"].str.strip().str.upper()
    raw_county = frame[
        "prime_award_transaction_place_of_performance_county_fips_code"
    ].str.strip()
    raw_state = frame[
        "prime_award_transaction_place_of_performance_state_fips_code"
    ].str.strip()
    state_name = normalize_place_name(frame["primary_place_of_performance_state_name"])
    named_state = state_name.map(STATE_NAME_TO_FIPS).fillna("")
    analysis_state = named_state.where(named_state.ne(""), raw_state)
    county_name = normalize_place_name(frame["primary_place_of_performance_county_name"])
    named_county = pd.Series(
        [name_to_fips.get(key, "") for key in zip(analysis_state, county_name)],
        index=frame.index,
        dtype="string",
    )

    valid = raw_county.str.fullmatch(r"\d{5}", na=False)
    domestic = country.eq("USA")
    missing_country = country.eq("")
    territory = domestic & raw_state.isin(dc.TERRITORY_FIPS)
    direct_ccg = raw_county.map(fips_to_ccg).astype("Int64")
    state_conflict = (
        domestic
        & direct_ccg.notna()
        & named_state.isin(dc.STATE_FIPS)
        & raw_county.str[:2].ne(analysis_state)
    )
    county_name_conflict = (
        domestic
        & direct_ccg.notna()
        & named_county.ne("")
        & raw_county.ne(named_county)
    )
    use_name = named_county.ne("") & (
        direct_ccg.isna() | state_conflict | county_name_conflict
    )
    county = raw_county.where(~use_name, named_county)
    ccg = county.map(fips_to_ccg).astype("Int64")
    unresolved_conflict = state_conflict & ~use_name
    county = county.mask(unresolved_conflict, "")
    ccg = ccg.mask(unresolved_conflict)

    geo = np.full(len(frame), "foreign", dtype=object)
    geo[missing_country.to_numpy()] = "missing_country"
    geo[territory.to_numpy()] = "domestic_territory"
    geo[(domestic & ~territory & raw_county.eq("") & ~use_name).to_numpy()] = (
        "domestic_missing_county"
    )
    geo[(domestic & ~territory & raw_county.ne("") & ~valid & ~use_name).to_numpy()] = (
        "domestic_invalid_county"
    )
    geo[(domestic & ~territory & ccg.isna() & ~unresolved_conflict).to_numpy()] = (
        "domestic_unmapped"
    )
    geo[unresolved_conflict.to_numpy()] = "domestic_conflict"
    geo[(domestic & ~territory & ccg.notna()).to_numpy()] = "county"
    source = pd.Series("none", index=frame.index, dtype="string")
    source[domestic & ~territory & ccg.notna() & ~use_name] = "reported_fips"
    source[domestic & ~territory & ccg.notna() & use_name] = "state_county_name"

    city_name = normalize_place_name(
        frame["primary_place_of_performance_city_name"])
    state_code = frame["primary_place_of_performance_state_code"].str.strip().str.upper()
    active = city_county_overrides.loc[
        city_county_overrides["first_year"].le(year)
        & city_county_overrides["last_year"].ge(year)]
    override_hits = np.zeros(len(frame), dtype=bool)
    for row in active.itertuples(index=False):
        hit = (
            domestic
            & ~territory
            & state_code.eq(row.state_code)
            & city_name.eq(row.city_name)
            & raw_county.eq(row.reported_county_fips)
        ).to_numpy()
        if np.any(override_hits & hit):
            raise AssertionError(
                f"FY{year}: overlapping city-county overrides")
        if hit.any():
            corrected_ccg = fips_to_ccg[row.corrected_county_fips]
            county.iloc[np.flatnonzero(hit)] = row.corrected_county_fips
            ccg.iloc[np.flatnonzero(hit)] = corrected_ccg
            geo[np.flatnonzero(hit)] = "county"
            source.iloc[np.flatnonzero(hit)] = "city_county_override"
            override_hits |= hit
    county = county.where(ccg.notna(), "")
    return county, ccg, pd.Series(geo, index=frame.index, dtype="string"), source


def canonical_chunk(
        raw, year, member_name, row_start, fips_to_ccg, name_to_fips,
        city_county_overrides):
    obligation_text = raw["federal_action_obligation"].str.strip()
    obligation = pd.to_numeric(obligation_text, errors="coerce")
    invalid_obligation = obligation.isna() & obligation_text.ne("")
    if invalid_obligation.any():
        sample = obligation_text[invalid_obligation].head().tolist()
        raise AssertionError(f"FY{year}: invalid federal_action_obligation values {sample}")
    if obligation.isna().any():
        raise AssertionError(f"FY{year}: {int(obligation.isna().sum())} blank obligations")

    action_fy = pd.to_numeric(raw["action_date_fiscal_year"], errors="coerce")
    if action_fy.isna().any() or not action_fy.eq(year).all():
        counts = action_fy.value_counts(dropna=False).to_dict()
        raise AssertionError(f"FY{year}: action-date fiscal years are {counts}")

    county, ccg, geo_class, county_fips_source = classify_geography(
        raw, year, fips_to_ccg, name_to_fips, city_county_overrides
    )
    frame = raw.copy()
    frame.insert(0, "source_fiscal_year", np.int16(year))
    frame.insert(1, "source_member", member_name)
    frame.insert(2, "source_row_number", np.arange(row_start, row_start + len(frame)))
    frame["federal_action_obligation"] = obligation.astype("float64")
    frame["action_date_fiscal_year"] = action_fy.astype("int16")
    frame["county_fips"] = county.astype("string")
    frame["ccg"] = ccg
    frame["geo_class"] = geo_class
    frame["county_fips_source"] = county_fips_source
    return frame


def empty_report(year, source, members):
    report = {field: 0 for field in REPORT_FIELDS}
    report.update(
        {"fiscal_year": year, "source_archive": source.name, "zip_members": len(members)}
    )
    return report


def update_report(report, frame):
    dollars = frame["federal_action_obligation"]
    report["rows"] += len(frame)
    report["dollars_nominal"] += float(dollars.sum())
    report["dollars_absolute"] += float(dollars.abs().sum())
    report["negative_actions"] += int((dollars < 0).sum())
    for label in GEO_CLASSES:
        mask = frame["geo_class"].eq(label)
        report[f"{label}_actions"] += int(mask.sum())
        report[f"{label}_dollars"] += float(dollars[mask].sum())
    name_resolved = frame["county_fips_source"].eq("state_county_name")
    report["name_resolved_actions"] += int(name_resolved.sum())
    report["name_resolved_dollars"] += float(dollars[name_resolved].sum())


def parse_archive(
        year, source, fips_to_ccg, name_to_fips, city_county_overrides):
    destination = dc.USASPENDING_OUT_DIR / f"fy{year}.parquet"
    temporary = destination.with_suffix(".parquet.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    writer = None

    with zipfile.ZipFile(source) as archive:
        members = csv_members(archive)
        headers = [check_header(archive, member) for member in members]
        if any(header != headers[0] for header in headers[1:]):
            raise AssertionError(f"FY{year}: split CSV members have different headers")
        report = empty_report(year, source, members)
        try:
            for member in members:
                row_start = 2
                with archive.open(member) as handle:
                    chunks = pd.read_csv(
                        handle,
                        usecols=list(SELECTED),
                        dtype="string",
                        keep_default_na=False,
                        chunksize=CHUNK_ROWS,
                        low_memory=False,
                    )
                    for raw in chunks:
                        frame = canonical_chunk(
                            raw, year, member.filename, row_start, fips_to_ccg,
                            name_to_fips, city_county_overrides,
                        )
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                temporary, table.schema, compression="zstd", use_dictionary=True
                            )
                        elif table.schema != writer.schema:
                            table = table.cast(writer.schema)
                        writer.write_table(table)
                        update_report(report, frame)
                        row_start += len(frame)
                        print(
                            f"      {member.filename}: {row_start - 2:,} rows",
                            end="\r",
                            flush=True,
                        )
        finally:
            if writer is not None:
                writer.close()
        print()

    if writer is None or report["rows"] == 0:
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"FY{year}: no transactions parsed")
    temporary.replace(destination)
    return destination, report


def write_report(rows):
    path = dc.LOG_DIR / "usaspending_parse_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = {int(row["fiscal_year"]): row for row in rows}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                combined.setdefault(int(row["fiscal_year"]), row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(combined[year] for year in sorted(combined))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", help="comma-separated years or ranges; default is all")
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="include custom-download FY2001-2006 archives",
    )
    args = parser.parse_args()
    selected = parse_years(args.years)
    archives = source_archives(selected, args.include_overlap)
    fips_to_ccg, _, _ = dc.load_ccg_map()
    name_to_fips = county_name_lookup(fips_to_ccg)
    city_county_overrides = load_city_county_overrides(fips_to_ccg)

    reports = []
    print(f"USAspending transaction parse: {len(archives)} archive(s)")
    for year, source in archives:
        print(f"  FY{year}: {source.name}")
        destination, report = parse_archive(
            year, source, fips_to_ccg, name_to_fips, city_county_overrides)
        reports.append(report)
        print(
            f"      {report['rows']:,} rows, "
            f"${report['dollars_nominal'] / 1e9:,.3f}bn -> {destination}"
        )
    report_path = write_report(reports)
    print(f"  {len(reports)} rows -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
