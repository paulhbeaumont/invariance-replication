#!/usr/bin/env python3
"""Build the pinned manifest for DoD prime-contract archives, FY2007-2018.

USAspending publishes one full annual zip per agency and award family. The
archive-list endpoint supplies the current file name and URL. This producer
requires exactly one Department of Defense full contracts archive for each
fiscal year and records the HTTP metadata needed by the downloader.

Outputs:
    dc.USASPENDING_SOURCE_DIR/manifest.csv
    dc.LOG_DIR/usaspending_manifest.csv
"""

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

API = "https://api.usaspending.gov/api/v2/bulk_download/list_monthly_files/"
AGENCY_ID = 126
AGENCY_CODE = "097"
FIRST_YEAR = 2007
LAST_YEAR = 2018
TIMEOUT = 120
FIELDS = (
    "fiscal_year",
    "agency_name",
    "agency_acronym",
    "agency_code",
    "archive_type",
    "updated_date",
    "file_name",
    "url",
    "content_length",
    "last_modified",
    "etag",
)


def json_post(url, payload, attempts=5):
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for the USAspending API")
    cmd = [
        curl,
        "--ipv4",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--max-time",
        str(TIMEOUT),
        "--request",
        "POST",
        url,
        "--header",
        "Content-Type: application/json",
        "--data",
        json.dumps(payload),
    ]
    for attempt in range(1, attempts + 1):
        try:
            return json.loads(subprocess.check_output(cmd))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def select_full_archive(year, records):
    candidates = [
        row
        for row in records
        if row.get("fiscal_year") == year
        and row.get("type") == "contracts"
        and f"FY{year}_{AGENCY_CODE}_Contracts_Full_" in row.get("file_name", "")
    ]
    if len(candidates) != 1:
        raise AssertionError(
            f"FY{year}: expected one full DoD contracts archive, found {len(candidates)}"
        )
    return candidates[0]


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    print(f"USAspending DoD contracts manifest, FY{FIRST_YEAR}-{LAST_YEAR}")
    seed_payload = {"agency": AGENCY_ID, "fiscal_year": FIRST_YEAR, "type": "contracts"}
    seed_listing = json_post(API, seed_payload).get("monthly_files", [])
    seed = select_full_archive(FIRST_YEAR, seed_listing)
    stamp = seed["file_name"].removesuffix(".zip").rsplit("_", 1)[-1]
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        file_name = f"FY{year}_{AGENCY_CODE}_Contracts_Full_{stamp}.zip"
        url = f"https://files.usaspending.gov/award_data_archive/{file_name}"
        row = {
            "fiscal_year": year,
            "agency_name": seed["agency_name"],
            "agency_acronym": seed["agency_acronym"],
            "agency_code": AGENCY_CODE,
            "archive_type": "contracts_full",
            "updated_date": seed["updated_date"],
            "file_name": file_name,
            "url": url,
            "content_length": "",
            "last_modified": "",
            "etag": "",
        }
        rows.append(row)
        print(f"  FY{year}: {file_name}", flush=True)

    years = [int(row["fiscal_year"]) for row in rows]
    expected = list(range(FIRST_YEAR, LAST_YEAR + 1))
    if years != expected:
        raise AssertionError(f"manifest years {years} do not equal {expected}")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise AssertionError("manifest archive names are not unique")

    source_manifest = dc.USASPENDING_SOURCE_DIR / "manifest.csv"
    log_manifest = dc.LOG_DIR / "usaspending_manifest.csv"
    write_manifest(source_manifest, rows)
    write_manifest(log_manifest, rows)
    print(f"  {len(rows)} rows -> {source_manifest}")
    print(f"  audit copy -> {log_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
