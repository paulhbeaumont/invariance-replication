#!/usr/bin/env python3
"""Acquire the FY2001-2006 USAspending overlap with the DCADS archive.

The annual archive center begins at FY2007. USAspending's custom award-download
endpoint supplies the earlier transaction records. Six annual requests cover
FY2001-2006 and are pinned to Department of Defense prime contract awards and
award types A-D.

Outputs:
    dc.USASPENDING_SOURCE_DIR/overlap/fy<year>/overlap_request.json
    dc.USASPENDING_SOURCE_DIR/overlap/fy<year>/<generated zip>
    dc.USASPENDING_SOURCE_DIR/overlap/download_receipt.json
    dc.LOG_DIR/usaspending_overlap_download_receipt.json
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

API = "https://api.usaspending.gov/api/v2/bulk_download/awards/"
POLL_SECONDS = 30
POLL_LIMIT = 240
RANGE_BYTES = 64 << 20
FIRST_YEAR = 2001
LAST_YEAR = 2006


def payload_for_year(year):
    return {
        "filters": {
            "prime_award_types": ["A", "B", "C", "D"],
            "date_type": "action_date",
            "date_range": {
                "start_date": f"{year - 1}-10-01",
                "end_date": f"{year}-09-30",
            },
            "agencies": [
                {
                    "type": "awarding",
                    "tier": "toptier",
                    "name": "Department of Defense",
                }
            ],
        },
        "file_format": "csv",
    }


def curl_base():
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for USAspending acquisition")
    return [
        curl,
        "--ipv4",
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "--speed-limit",
        "10240",
        "--speed-time",
        "60",
    ]


def post_json(url, payload):
    cmd = curl_base() + [
        "--silent",
        "--show-error",
        "--request",
        "POST",
        url,
        "--header",
        "Content-Type: application/json",
        "--data",
        json.dumps(payload),
    ]
    return json.loads(subprocess.check_output(cmd))


def get_json(url):
    cmd = curl_base() + ["--silent", "--show-error", url]
    return json.loads(subprocess.check_output(cmd))


def remote_size(url):
    cmd = curl_base() + [
        "--silent",
        "--show-error",
        "--range",
        "0-0",
        "--dump-header",
        "-",
        "--output",
        "/dev/null",
        url,
    ]
    headers = subprocess.check_output(cmd, text=True)
    matches = re.findall(r"(?im)^content-range:\s*bytes\s+\d+-\d+/(\d+)\s*$", headers)
    if not matches:
        raise AssertionError("generated download has no Content-Range total")
    return int(matches[-1])


def range_download(url, destination):
    total = remote_size(url)
    part = destination.with_suffix(destination.suffix + ".part")
    chunk = destination.with_suffix(destination.suffix + ".chunk")
    start = part.stat().st_size if part.exists() else 0
    if start > total:
        raise AssertionError(f"partial file has {start:,} bytes, remote file has {total:,}")
    while start < total:
        end = min(start + RANGE_BYTES, total) - 1
        chunk.unlink(missing_ok=True)
        cmd = curl_base() + [
            "--silent",
            "--show-error",
            "--range",
            f"{start}-{end}",
            "--output",
            str(chunk),
            url,
        ]
        subprocess.run(cmd, check=True)
        expected = end - start + 1
        if chunk.stat().st_size != expected:
            raise AssertionError(
                f"range {start}-{end} returned {chunk.stat().st_size:,} bytes, "
                f"expected {expected:,}"
            )
        with part.open("ab") as target, chunk.open("rb") as source:
            shutil.copyfileobj(source, target, length=8 << 20)
        chunk.unlink()
        start = part.stat().st_size
        print(f"    {start:,} / {total:,} bytes", flush=True)
    part.replace(destination)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"corrupt zip member {bad}")
        members = len([item for item in archive.infolist() if not item.is_dir()])
    if members == 0:
        raise AssertionError("generated zip contains no files")
    return members


def request_download(request_path, payload):
    if request_path.exists():
        return json.loads(request_path.read_text(encoding="utf-8"))
    response = post_json(API, payload)
    required = {"status_url", "file_name", "file_url"}
    missing = required - set(response)
    if missing:
        raise AssertionError(f"download response lacks {sorted(missing)}: {response}")
    record = {"payload": payload, "response": response}
    request_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def wait_for_file(response):
    for attempt in range(1, POLL_LIMIT + 1):
        status = get_json(response["status_url"])
        state = status.get("status")
        print(
            f"  generation {state}, poll {attempt}, rows {status.get('total_rows')}",
            flush=True,
        )
        if state == "finished":
            return status
        if state == "failed":
            raise RuntimeError(f"USAspending generation failed: {status.get('message')}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"USAspending generation did not finish in {POLL_LIMIT * POLL_SECONDS}s")


def write_receipt(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    source_dir = dc.USASPENDING_SOURCE_DIR / "overlap"
    source_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    print("Submitting annual overlap requests")
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        year_dir = source_dir / f"fy{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        payload = payload_for_year(year)
        request_path = year_dir / "overlap_request.json"
        record = request_download(request_path, payload)
        jobs.append((year, year_dir, payload, record["response"]))
        print(f"  FY{year}: {record['response']['file_name']}")

    receipts = []
    for year, year_dir, payload, response in jobs:
        print(f"FY{year}")
        destination = year_dir / response["file_name"]

        if destination.exists():
            status = {"status": "local_file_present"}
        else:
            status = wait_for_file(response)
            file_url = status.get("url") or response["file_url"]
            if file_url.startswith("/"):
                file_url = "https://files.usaspending.gov" + file_url
            print(f"  downloading {file_url}")
            range_download(file_url, destination)

        members = verify_zip(destination)
        digest = sha256(destination)
        receipts.append(
            {
                "fiscal_year": year,
                "file_name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "zip_members": members,
                "request": payload,
                "generation_status": status,
            }
        )
        print(
            f"  {destination.stat().st_size:,} bytes, {members} members, "
            f"sha256 {digest[:16]}..."
        )

    source_receipt = source_dir / "download_receipt.json"
    log_receipt = dc.LOG_DIR / "usaspending_overlap_download_receipt.json"
    log_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(source_receipt, receipts)
    write_receipt(log_receipt, receipts)
    print(f"  receipt -> {source_receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
