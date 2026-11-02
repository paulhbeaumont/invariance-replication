#!/usr/bin/env python3
"""Download and verify the USAspending archives declared by script 29.

Downloads are resumable. A completed archive must match the manifest's byte
count when one is available, pass the zip central-directory check, and receive
a SHA-256 digest before it is entered in the receipt.

Outputs:
    dc.USASPENDING_SOURCE_DIR/FY*.zip
    dc.USASPENDING_SOURCE_DIR/download_receipt.csv
    dc.LOG_DIR/usaspending_download_receipt.csv
"""

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

RECEIPT_FIELDS = (
    "fiscal_year",
    "file_name",
    "url",
    "bytes",
    "sha256",
    "zip_members",
    "status",
)
RANGE_BYTES = 64 << 20


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_zip(path, expected_size):
    size = path.stat().st_size
    if expected_size and size != expected_size:
        raise AssertionError(f"{path.name}: {size:,} bytes, expected {expected_size:,}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"{path.name}: corrupt member {bad}")
        members = len([item for item in archive.infolist() if not item.is_dir()])
    if members == 0:
        raise AssertionError(f"{path.name}: no files in archive")
    return size, members


def curl_command(url):
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable multi-gigabyte downloads")
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


def remote_size(url):
    cmd = curl_command(url) + [
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
        raise AssertionError(f"{url}: byte-range response has no Content-Range total")
    return int(matches[-1])


def download(url, destination):
    total = remote_size(url)
    part = destination.with_suffix(destination.suffix + ".part")
    chunk = destination.with_suffix(destination.suffix + ".chunk")
    start = part.stat().st_size if part.exists() else 0
    if start > total:
        raise AssertionError(f"{part}: {start:,} bytes exceeds remote size {total:,}")
    while start < total:
        end = min(start + RANGE_BYTES, total) - 1
        chunk.unlink(missing_ok=True)
        cmd = curl_command(url) + [
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
        actual = chunk.stat().st_size
        if actual != expected:
            raise AssertionError(
                f"{destination.name}: range {start}-{end} returned {actual:,} bytes, "
                f"expected {expected:,}"
            )
        with part.open("ab") as target, chunk.open("rb") as source:
            shutil.copyfileobj(source, target, length=8 << 20)
        chunk.unlink()
        start = part.stat().st_size
        print(f"      {start:,} / {total:,} bytes", flush=True)
    part.replace(destination)


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


def write_receipt(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECEIPT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", help="comma-separated years or ranges; default is all")
    args = parser.parse_args()
    selected = parse_years(args.years)

    manifest = dc.USASPENDING_SOURCE_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run 29_build_usaspending_manifest.py")
    with manifest.open(newline="", encoding="utf-8") as handle:
        declared = list(csv.DictReader(handle))
    if selected is not None:
        declared = [row for row in declared if int(row["fiscal_year"]) in selected]
    if not declared:
        raise AssertionError("no manifest rows selected")

    dc.USASPENDING_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    receipts = []
    print(f"USAspending downloads: {len(declared)} archive(s)")
    for row in declared:
        year = int(row["fiscal_year"])
        destination = dc.USASPENDING_SOURCE_DIR / row["file_name"]
        expected = int(row["content_length"] or 0)
        status = "verified_existing"
        try:
            size, members = verify_zip(destination, expected)
        except (FileNotFoundError, AssertionError, zipfile.BadZipFile):
            status = "downloaded"
            print(f"  FY{year}: downloading {row['url']}", flush=True)
            download(row["url"], destination)
            size, members = verify_zip(destination, expected)
        digest = sha256(destination)
        print(
            f"  FY{year}: {size:,} bytes, {members:,} members, sha256 {digest[:16]}...",
            flush=True,
        )
        receipts.append(
            {
                "fiscal_year": year,
                "file_name": destination.name,
                "url": row["url"],
                "bytes": size,
                "sha256": digest,
                "zip_members": members,
                "status": status,
            }
        )

    source_receipt = dc.USASPENDING_SOURCE_DIR / "download_receipt.csv"
    log_receipt = dc.LOG_DIR / "usaspending_download_receipt.csv"
    write_receipt(source_receipt, receipts)
    write_receipt(log_receipt, receipts)
    print(f"  {len(receipts)} rows -> {source_receipt}")
    print(f"  audit copy -> {log_receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
