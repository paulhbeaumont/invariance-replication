"""Stage 09: recipient identity for every USAspending contract transaction.

WS3 step 0. The parsed transaction panel that this pipeline consumes carries no
recipient identifier at all - producer 31 selects 54 accounting, date, and
place-of-performance columns out of the archive's 297, and every name, DUNS,
UEI, and address is left behind. WS3 step 1 (the era-tiered contract-to-NETS
match memo) cannot start without them. This producer reads the SAME archives,
in the SAME row order, and writes the identity columns to a sidecar keyed to
producer 31's rows. It creates no outcome, no shock, and no estimand.

    FY2001-06  overlap/fy<year>/All_PrimeTransactions_*.zip   (custom download)
    FY2007-18  FY<year>_097_Contracts_Full_*.zip              (pinned, DoD)
              |
              |  usecols = 4 join keys + 22 identity columns
              |  dtype=string, keep_default_na=False
              v
    data/temp/usaspending_recipient/fy<year>.parquet
              |
              |  verified row-for-row against the store 31 wrote
              v
    _data/temp/usaspending_year/fy<year>.parquet   (NOT rewritten)

A SIDECAR, NOT A WIDER PARQUET. `usaspending_year/` sits upstream of stage 1 ->
stage 5's baseline tie-out and of stage 23's industry axis. Rewriting it to add
columns would put every locked number in the blast radius of an ingest change
that no estimand depends on. Writing beside it instead means the locked store is
byte-identical by construction and there is nothing to re-tie-out. The join key
is (source_fiscal_year, source_member, source_row_number), computed here by the
same rule producer 31 uses - row_start at 2, the CSV line number - and
`contract_transaction_unique_key` is carried as a semantic key so the join can
be checked independently of row order. Every run asserts that the sidecar and
the existing parquet agree row for row; that assertion is the only thing
standing between a joinable sidecar and a silently misaligned one.

THE OUTPUT IS LOCAL. These parquets are an intermediate that only WS3 step 1
reads. They stay in the git checkout, gitignored, under a directory that mirrors
the Dropbox path exactly, so promoting them later is a copy and not a path
rewrite. Same arrangement as stage 23.

THE COVERAGE TABLE IS THE DELIVERABLE, NOT THE PARQUETS. WS3 step 0 exists to
answer one question: which years can actually be joined on DUNS. DUNS was
retired for UEI in 2022 and these are 2026-vintage re-publications of pre-2022
data, so both fields may be present, back-filled, or empty, and the plan is
explicit that this is to be checked rather than assumed. Nothing here asserts a
fill rate. A thin DUNS is a finding that sets the era tiers in step 1, not a
failure. Degenerate-but-present values (all-zero, all-nine, "UNKNOWN") are
counted separately, because a column that is 100% non-blank and 100% sentinel
would otherwise read as full coverage.

Outputs:
    data/temp/usaspending_recipient/fy<year>.parquet    LOCAL, gitignored
        source_fiscal_year, source_member, source_row_number,
        contract_transaction_unique_key, + the 22 identity columns
    batch_logs/macro_version/usaspending_recipient_fill_report.csv
        fiscal_year x column: rows, nonblank, share, sentinel, distinct,
        dollars_abs, dollars_abs_nonblank, dollar_share
    batch_logs/macro_version/usaspending_identifier_coverage.csv
        the five join-candidate identifiers only, same fields, one row per
        identifier x year - the coverage profile
    batch_logs/macro_version/usaspending_identifier_coverage_summary.csv
        per identifier: first and last covered year, whether contiguous

  --years 2007 | 2001-2018 | 2001,2007   default is every archive
  --include-overlap   also parse the custom FY2001-06 downloads
  --dry-run           resolve archives and check headers; write nothing
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
import dcads_common as dc  # noqa: E402

CHUNK_ROWS = 150_000
OUT_DIR = PIPELINE.parents[2] / "data" / "temp" / "usaspending_recipient"

# The transaction key producer 31 also writes, carried so the join can be
# validated without trusting row order.
TRANSACTION_KEY = "contract_transaction_unique_key"

# Read for the report only; NOT written to the sidecar. It already lives in
# usaspending_year/fy<year>.parquet and duplicating it would invite the two
# copies to disagree.
OBLIGATION = "federal_action_obligation"

RECIPIENT_IDENTITY = (
    "recipient_duns",
    "recipient_uei",
    "cage_code",
    "recipient_name",
    "recipient_name_raw",
    "recipient_doing_business_as_name",
)
RECIPIENT_PARENT = (
    "recipient_parent_duns",
    "recipient_parent_uei",
    "recipient_parent_name",
    "recipient_parent_name_raw",
)
RECIPIENT_ADDRESS = (
    "recipient_address_line_1",
    "recipient_address_line_2",
    "recipient_city_name",
    "recipient_county_name",
    "prime_award_transaction_recipient_state_fips_code",
    "recipient_state_name",
    "recipient_zip_4_code",
    "recipient_country_name",
    "prime_award_transaction_recipient_cd_original",
    "prime_award_transaction_recipient_cd_current",
)
# The only two place-of-performance fields producer 31 does not already read.
PLACE_OF_PERFORMANCE = (
    "prime_award_transaction_place_of_performance_cd_original",
    "prime_award_transaction_place_of_performance_cd_current",
)
IDENTITY = RECIPIENT_IDENTITY + RECIPIENT_PARENT + RECIPIENT_ADDRESS + PLACE_OF_PERFORMANCE

# Everything read out of the CSV. OBLIGATION is dropped before writing.
SELECTED = (TRANSACTION_KEY,) + IDENTITY + (OBLIGATION,)
# Everything written, in order.
OUTPUT_COLUMNS = (
    "source_fiscal_year",
    "source_member",
    "source_row_number",
    TRANSACTION_KEY,
) + IDENTITY

# The join candidates for WS3 step 1. Distinct-value counting is restricted to
# these plus name and street, because a set per column per year is the only
# expensive thing in this producer.
JOIN_IDENTIFIERS = (
    "recipient_duns",
    "recipient_uei",
    "recipient_parent_duns",
    "recipient_parent_uei",
    "cage_code",
)
DISTINCT_COLUMNS = JOIN_IDENTIFIERS + ("recipient_name", "recipient_address_line_1")

# Present, but carrying no identity. Counted apart from true blanks so that a
# column which is 100% non-blank and 100% placeholder cannot read as coverage.
SENTINEL_LITERALS = {
    "UNKNOWN", "NULL", "NONE", "NA", "N/A", "NOT AVAILABLE", "NOT APPLICABLE",
    "MISSING", "-", "--", ".",
}
SENTINEL_RUN = re.compile(r"^(0+|9+|X+|x+)$")

REPORT_FIELDS = (
    "fiscal_year",
    "column",
    "rows",
    "nonblank",
    "nonblank_share",
    "sentinel",
    "sentinel_share",
    "distinct",
    "dollars_abs",
    "dollars_abs_nonblank",
    "dollar_share",
)
SUMMARY_FIELDS = (
    "identifier",
    "first_year_nonblank",
    "last_year_nonblank",
    "years_covered",
    "years_seen",
    "contiguous",
    "min_row_share",
    "max_row_share",
)


def parse_years(text):
    """Producer 31's --years grammar: comma list, with a-b ranges."""
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
    """Producer 31's archive resolution, so both read exactly the same files.

    Copied rather than imported: the module name starts with a digit, and
    PIPELINE.md's rule for this band is to consume persisted artefacts rather
    than reach back into the old folder's import chain.
    """
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


def archive_headers(year, source):
    """Open, enumerate members, and assert one shared header across the split."""
    with zipfile.ZipFile(source) as archive:
        members = csv_members(archive)
        headers = [check_header(archive, member) for member in members]
        if any(header != headers[0] for header in headers[1:]):
            raise AssertionError(f"FY{year}: split CSV members have different headers")
    return [member.filename for member in members], headers[0]


def empty_tally(year):
    """Per-column counters for one fiscal year."""
    return {
        "fiscal_year": year,
        "rows": 0,
        "dollars_abs": 0.0,
        "nonblank": {column: 0 for column in IDENTITY},
        "sentinel": {column: 0 for column in IDENTITY},
        "dollars_nonblank": {column: 0.0 for column in IDENTITY},
        "distinct": {column: set() for column in DISTINCT_COLUMNS},
    }


def is_sentinel(values):
    """Non-blank but carrying no identity."""
    upper = values.str.upper()
    return upper.isin(SENTINEL_LITERALS) | upper.str.fullmatch(SENTINEL_RUN, na=False)


def update_tally(tally, frame, dollars_abs):
    tally["rows"] += len(frame)
    tally["dollars_abs"] += float(dollars_abs.sum())
    for column in IDENTITY:
        values = frame[column].fillna("").str.strip()
        filled = values.ne("")
        tally["nonblank"][column] += int(filled.sum())
        tally["dollars_nonblank"][column] += float(dollars_abs[filled].sum())
        if filled.any():
            present = values[filled]
            tally["sentinel"][column] += int(is_sentinel(present).sum())
            if column in DISTINCT_COLUMNS:
                tally["distinct"][column].update(present.unique().tolist())


def tally_rows(tally):
    """One report row per column, ready for the CSV."""
    rows = []
    total = tally["rows"]
    dollars = tally["dollars_abs"]
    for column in IDENTITY:
        nonblank = tally["nonblank"][column]
        sentinel = tally["sentinel"][column]
        covered = tally["dollars_nonblank"][column]
        rows.append({
            "fiscal_year": tally["fiscal_year"],
            "column": column,
            "rows": total,
            "nonblank": nonblank,
            "nonblank_share": round(nonblank / total, 6) if total else 0.0,
            "sentinel": sentinel,
            "sentinel_share": round(sentinel / total, 6) if total else 0.0,
            "distinct": len(tally["distinct"][column]) if column in DISTINCT_COLUMNS else "",
            "dollars_abs": round(dollars, 2),
            "dollars_abs_nonblank": round(covered, 2),
            "dollar_share": round(covered / dollars, 6) if dollars else 0.0,
        })
    return rows


def canonical_chunk(raw, year, member_name, row_start):
    """The join keys, by producer 31's rule, in front of the identity columns."""
    frame = raw.drop(columns=[OBLIGATION])
    frame.insert(0, "source_fiscal_year", np.int16(year))
    frame.insert(1, "source_member", member_name)
    frame.insert(2, "source_row_number", np.arange(row_start, row_start + len(frame)))
    return frame[list(OUTPUT_COLUMNS)]


def verify_against_parsed(year, destination):
    """The sidecar is worthless unless it lines up with the store 31 wrote."""
    parsed_path = dc.USASPENDING_OUT_DIR / f"fy{year}.parquet"
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"{parsed_path} is missing; run 31_parse_usaspending.py for FY{year} first"
        )
    keys = ["source_member", "source_row_number", TRANSACTION_KEY]
    parsed = pd.read_parquet(parsed_path, columns=keys)
    sidecar = pd.read_parquet(destination, columns=keys)
    if len(parsed) != len(sidecar):
        raise AssertionError(
            f"FY{year}: sidecar has {len(sidecar):,} rows, "
            f"usaspending_year has {len(parsed):,}"
        )
    for key in keys:
        left = parsed[key].astype("string").fillna("")
        right = sidecar[key].astype("string").fillna("")
        mismatch = left.ne(right)
        if mismatch.any():
            first = int(np.flatnonzero(mismatch.to_numpy())[0])
            raise AssertionError(
                f"FY{year}: {key} disagrees with usaspending_year at row {first} "
                f"({left.iloc[first]!r} vs {right.iloc[first]!r}); "
                f"{int(mismatch.sum()):,} rows differ"
            )
    return len(parsed)


def parse_archive(year, source):
    destination = OUT_DIR / f"fy{year}.parquet"
    temporary = destination.with_suffix(".parquet.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    writer = None
    tally = empty_tally(year)

    with zipfile.ZipFile(source) as archive:
        members = csv_members(archive)
        headers = [check_header(archive, member) for member in members]
        if any(header != headers[0] for header in headers[1:]):
            raise AssertionError(f"FY{year}: split CSV members have different headers")
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
                        dollars_abs = (
                            pd.to_numeric(raw[OBLIGATION].str.strip(), errors="coerce")
                            .abs()
                            .fillna(0.0)
                        )
                        frame = canonical_chunk(raw, year, member.filename, row_start)
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                temporary, table.schema, compression="zstd", use_dictionary=True
                            )
                        elif table.schema != writer.schema:
                            table = table.cast(writer.schema)
                        writer.write_table(table)
                        update_tally(tally, frame, dollars_abs)
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

    if writer is None or tally["rows"] == 0:
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"FY{year}: no transactions parsed")
    temporary.replace(destination)
    verify_against_parsed(year, destination)
    return destination, tally_rows(tally)


def merge_report(path, fieldnames, rows, keys):
    """Rewrite the CSV, keeping rows for years this run did not touch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = {tuple(str(row[key]) for key in keys): row for row in rows}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                combined.setdefault(tuple(str(row[key]) for key in keys), row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(combined, key=lambda item: (item[0], item[1])):
            writer.writerow(combined[key])
    return path


def coverage_summary(coverage_rows):
    """First and last covered year per identifier, and whether the run has gaps.

    A gap year is exactly the thing that silently breaks an era-tiered join, so
    it is named here rather than left for someone to notice in the matrix.
    """
    frame = pd.DataFrame(coverage_rows)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame["nonblank_share"] = frame["nonblank_share"].astype(float)
    summary = []
    for identifier in JOIN_IDENTIFIERS:
        block = frame[frame["column"].eq(identifier)].sort_values("fiscal_year")
        if block.empty:
            continue
        seen = block["fiscal_year"].tolist()
        covered = block[block["nonblank"].astype(int) > 0]["fiscal_year"].tolist()
        # Contiguity is judged against the years actually PARSED, not the
        # calendar span: a year absent from the report was never looked at, and
        # calling that a gap would flag every partial run.
        inside = [year for year in seen if covered and covered[0] <= year <= covered[-1]]
        summary.append({
            "identifier": identifier,
            "first_year_nonblank": covered[0] if covered else "",
            "last_year_nonblank": covered[-1] if covered else "",
            "years_covered": len(covered),
            "years_seen": len(block),
            "contiguous": bool(covered) and covered == inside,
            "min_row_share": round(block["nonblank_share"].min(), 6),
            "max_row_share": round(block["nonblank_share"].max(), 6),
        })
    return summary


def print_matrix(coverage_rows, field, title):
    frame = pd.DataFrame(coverage_rows)
    frame["fiscal_year"] = frame["fiscal_year"].astype(int)
    frame[field] = frame[field].astype(float)
    pivot = frame.pivot(index="column", columns="fiscal_year", values=field)
    pivot = pivot.reindex([c for c in JOIN_IDENTIFIERS if c in pivot.index])
    print(f"\n  {title}")
    header = "    " + f"{'identifier':<26}" + "".join(f"{y:>8}" for y in pivot.columns)
    print(header)
    for identifier, row in pivot.iterrows():
        cells = "".join(
            "       -" if pd.isna(value) else f"{value:>8.3f}" for value in row
        )
        print(f"    {identifier:<26}{cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", help="comma-separated years or ranges; default is all")
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="include custom-download FY2001-2006 archives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve archives and check headers; write nothing",
    )
    args = parser.parse_args()
    selected = parse_years(args.years)
    archives = source_archives(selected, args.include_overlap)

    print(f"USAspending recipient identity: {len(archives)} archive(s)")
    print(f"  {len(IDENTITY)} identity columns + 4 join keys -> {OUT_DIR}")

    if args.dry_run:
        for year, source in archives:
            members, header = archive_headers(year, source)
            print(f"  FY{year}: {source.name}")
            print(f"      {len(header)} columns in header, {len(members)} CSV member(s)")
            for name in members:
                print(f"      would read {name}")
        print("\n  dry run: nothing written")
        return 0

    report_rows = []
    for year, source in archives:
        print(f"  FY{year}: {source.name}")
        destination, rows = parse_archive(year, source)
        report_rows.extend(rows)
        duns = next(r for r in rows if r["column"] == "recipient_duns")
        uei = next(r for r in rows if r["column"] == "recipient_uei")
        print(
            f"      {duns['rows']:,} rows, "
            f"DUNS {duns['nonblank_share']:.3f} of rows / {duns['dollar_share']:.3f} of $, "
            f"UEI {uei['nonblank_share']:.3f} / {uei['dollar_share']:.3f} -> {destination}"
        )

    fill_path = merge_report(
        dc.LOG_DIR / "usaspending_recipient_fill_report.csv",
        REPORT_FIELDS,
        report_rows,
        ("fiscal_year", "column"),
    )
    coverage_path = merge_report(
        dc.LOG_DIR / "usaspending_identifier_coverage.csv",
        REPORT_FIELDS,
        [row for row in report_rows if row["column"] in JOIN_IDENTIFIERS],
        ("fiscal_year", "column"),
    )
    with coverage_path.open(newline="", encoding="utf-8") as handle:
        coverage_rows = list(csv.DictReader(handle))
    summary_path = dc.LOG_DIR / "usaspending_identifier_coverage_summary.csv"
    summary = coverage_summary(coverage_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    print_matrix(coverage_rows, "nonblank_share", "share of ROWS with a non-blank value")
    print_matrix(coverage_rows, "dollar_share", "share of |DOLLARS| with a non-blank value")
    print("\n  time coverage")
    for row in summary:
        gap = "" if row["contiguous"] else "  WITH GAPS"
        span = (
            f"FY{row['first_year_nonblank']}-FY{row['last_year_nonblank']}"
            if row["first_year_nonblank"] != "" else "never populated"
        )
        print(
            f"    {row['identifier']:<26}{span}  "
            f"({row['years_covered']}/{row['years_seen']} years, "
            f"row share {row['min_row_share']:.3f}-{row['max_row_share']:.3f}){gap}"
        )

    print(f"\n  {len(report_rows)} rows -> {fill_path}")
    print(f"  -> {coverage_path}")
    print(f"  -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
