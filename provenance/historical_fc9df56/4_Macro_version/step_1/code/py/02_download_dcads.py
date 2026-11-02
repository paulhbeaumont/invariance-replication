#!/usr/bin/env python3
"""Download every DCADS and MPCF object listed in the manifest.

Reads $source/DCADS/manifest.csv and writes each object to
$source/DCADS/<series_label>/<fiscal_year>/<filename>.

Objects already present at their manifest byte size are skipped, so the script
is safe to rerun after an interruption. Every completed file is checksummed and
recorded in download_receipt.csv. Any object whose downloaded size differs from
the manifest is reported as a failure and left on disk with a .partial suffix.
"""

import csv
import hashlib
import subprocess
import sys
import time

import dcads_common as dc

SOURCE_DIR = dc.SOURCE_DIR
MANIFEST = SOURCE_DIR / "manifest.csv"
RECEIPT = SOURCE_DIR / "download_receipt.csv"

RETRIES = 4
BACKOFF = 5
CHUNK = 1 << 20

# NARA serves large files over connections that can go silent without closing,
# so the transfer aborts whenever throughput stays under STALL_BYTES per second
# for STALL_SECONDS and resumes from the byte already on disk.
STALL_BYTES = 10_000
STALL_SECONDS = 60

# NARA throttles sustained downloading, so the queue is ordered to make partial
# progress usable. Years inside PRIORITY_WINDOW come first, and within every
# year the documentation is fetched before the data, so an interrupted run
# leaves whole parseable fiscal years rather than data files with no layout.
PRIORITY_WINDOW = (1977, 2006)


def target_path(row):
    year = row["fiscal_year"] or "series"
    return SOURCE_DIR / row["series_label"] / year / row["filename"]


def priority_key(row):
    """Order the queue so interrupted runs leave complete, parseable years."""
    try:
        year = int(row["fiscal_year"])
    except ValueError:
        year = 0
    lo, hi = PRIORITY_WINDOW
    in_window = 0 if lo <= year <= hi else 1
    doc_first = 0 if row["kind"] == "doc" else 1
    return (in_window, year, doc_first, row["filename"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    """Fetch one object, resuming a partial transfer and aborting on a stall."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    cmd = [
        "curl", "-sSL", "--fail",
        "-A", "Mozilla/5.0",
        "-C", "-",
        "--speed-limit", str(STALL_BYTES),
        "--speed-time", str(STALL_SECONDS),
        "--connect-timeout", "60",
        "-o", str(tmp),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr.strip()[:200]}")
    tmp.rename(dest)
    return dest.stat().st_size


def main():
    if not MANIFEST.exists():
        print(f"FAIL: manifest missing at {MANIFEST}", file=sys.stderr)
        return 1

    with open(MANIFEST, encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=priority_key)

    total_bytes = sum(int(r["bytes"]) for r in rows)
    lo, hi = PRIORITY_WINDOW
    print(f"manifest: {len(rows)} objects, {total_bytes / 1e9:.2f} GB", flush=True)
    print(f"queue order: FY{lo}-FY{hi} first, documentation before data within "
          f"each year, then all remaining years", flush=True)

    receipts, failures, skipped, fetched = [], [], 0, 0
    done_bytes = 0
    start = time.time()

    for i, row in enumerate(rows, 1):
        expected = int(row["bytes"])
        dest = target_path(row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tag = f"[{i:>3}/{len(rows)}] {row['series_label']} {row['fiscal_year']:>4} {row['filename']}"

        if dest.exists() and dest.stat().st_size == expected:
            skipped += 1
            done_bytes += expected
            receipts.append({**row, "path": str(dest.relative_to(SOURCE_DIR)),
                             "actual_bytes": expected, "sha256": sha256(dest),
                             "status": "present"})
            print(f"{tag}  skip (present)", flush=True)
            continue

        actual = None
        for attempt in range(1, RETRIES + 1):
            try:
                actual = download(row["url"], dest)
                break
            except (RuntimeError, OSError) as exc:
                if attempt == RETRIES:
                    failures.append({**row, "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"{tag}  FAIL after {RETRIES} attempts: {exc}", flush=True)
                else:
                    time.sleep(BACKOFF * attempt)

        if actual is None:
            continue

        if expected and actual != expected:
            failures.append({**row, "reason": f"size mismatch: expected {expected}, got {actual}"})
            print(f"{tag}  FAIL size {actual} != {expected}", flush=True)
            continue

        fetched += 1
        done_bytes += actual
        receipts.append({**row, "path": str(dest.relative_to(SOURCE_DIR)),
                         "actual_bytes": actual, "sha256": sha256(dest),
                         "status": "downloaded"})
        pct = 100 * done_bytes / total_bytes if total_bytes else 0
        print(f"{tag}  ok {actual / 1e6:.1f} MB  ({pct:.1f}%)", flush=True)

    if receipts:
        with open(RECEIPT, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(receipts[0].keys()))
            writer.writeheader()
            writer.writerows(receipts)

    elapsed = time.time() - start
    print(f"\ndownloaded {fetched}, already present {skipped}, failed {len(failures)}")
    print(f"bytes on disk {done_bytes / 1e9:.2f} GB of {total_bytes / 1e9:.2f} GB")
    print(f"elapsed {elapsed / 60:.1f} min")
    print(f"receipt: {RECEIPT}")

    if failures:
        print(f"\nFAIL: {len(failures)} objects did not complete", file=sys.stderr)
        for f in failures:
            print(f"  {f['series_label']} {f['fiscal_year']} {f['filename']}: {f['reason']}",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
