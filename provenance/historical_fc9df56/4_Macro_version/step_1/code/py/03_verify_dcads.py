#!/usr/bin/env python3
"""Verify the downloaded DCADS and MPCF objects by content, not by byte count.

The NARA catalog's recorded file sizes disagree with what the server actually
serves for two families of object: every zip arrives 160 bytes larger than its
catalog figure, and DCADS_TSS.pdf arrives 2,171 bytes smaller. Both are valid
files, so byte equality against the catalog produces false failures. This script
checks what the file actually is instead.

Checks applied by type:
  .zip           central directory parses and every member passes CRC
  .pdf           starts with %PDF and contains an EOF marker
  .lay .doc .txt non-empty and decodable as text
  data files     non-empty, and record length divides the file size evenly

Exit status is non-zero if any object fails or is missing.
"""

import csv
import sys
import zipfile

import dcads_common as dc

SOURCE_DIR = dc.SOURCE_DIR
MANIFEST = SOURCE_DIR / "manifest.csv"
REPORT = SOURCE_DIR / "verification_report.csv"

# The NARA catalog's byte counts are close but not exact: verified-clean files
# sit within a couple of hundred bytes either side. A data file that differs by
# more than this is truncated, or has had a resumed transfer appended to it.
SIZE_TOLERANCE = 4096

# Record lengths confirmed against the NARA-prepared layouts. FY1976 through
# FY1981 divide evenly by 272 and DCADS.FY03.LAY declares 500. Lengths for the
# remaining eras are not yet read off their layouts, so record structure is
# reported as a diagnostic and never used to fail a file.
RECORD_LENGTHS = {
    "RG330.DEFCON.FY76": 272,
    "RG330.DEFCON.FY77": 272,
    "RG330.DEFCON.FY78": 272,
    "RG330.DEFCON.FY79": 272,
    "RG330.DEFCON.FY80": 272,
    "RG330.DEFCON.FY81": 272,
}


def target_path(row):
    year = row["fiscal_year"] or "series"
    return SOURCE_DIR / row["series_label"] / year / row["filename"]


def check_zip(path):
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"CRC failure in member {bad}"
            return True, f"{len(zf.namelist())} members"
    except zipfile.BadZipFile as exc:
        return False, f"bad zip: {exc}"


def check_pdf(path):
    with open(path, "rb") as fh:
        head = fh.read(5)
        fh.seek(max(0, path.stat().st_size - 2048))
        tail = fh.read()
    if head != b"%PDF-":
        return False, "missing %PDF header"
    if b"%%EOF" not in tail:
        return False, "missing EOF marker"
    return True, "pdf ok"


def check_text(path):
    if path.stat().st_size == 0:
        return False, "empty"
    try:
        path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        try:
            path.read_text(encoding="latin-1")
        except Exception as exc:
            return False, f"undecodable: {exc}"
    return True, "text ok"


def check_data(path, catalog_bytes):
    size = path.stat().st_size
    if size == 0:
        return False, "empty"
    if catalog_bytes and abs(size - catalog_bytes) > SIZE_TOLERANCE:
        delta = size - catalog_bytes
        cause = "appended resume" if delta > 0 else "truncated transfer"
        return False, f"size {size:,} differs from catalog {catalog_bytes:,} by {delta:+,}, {cause}"
    declared = RECORD_LENGTHS.get(path.name)
    if declared:
        if size % declared:
            return False, f"size {size:,} is not a multiple of record length {declared}"
        return True, f"{size // declared:,} records of {declared} bytes"
    inferred = [n for n in range(80, 1400) if size % n == 0]
    hint = f"record length undeclared, divides by {inferred[:6]}" if inferred else "record length undeclared"
    return True, f"{size:,} bytes, {hint}"


def verify(path, catalog_bytes):
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return check_zip(path)
    if suffix == ".pdf":
        return check_pdf(path)
    if suffix in (".lay", ".doc", ".txt"):
        return check_text(path)
    return check_data(path, catalog_bytes)


def main():
    if not MANIFEST.exists():
        print(f"FAIL: manifest missing at {MANIFEST}", file=sys.stderr)
        return 1

    with open(MANIFEST, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    results, failed, missing = [], [], []
    for row in rows:
        path = target_path(row)
        if not path.exists():
            missing.append(row)
            results.append({**row, "status": "missing", "detail": "", "actual_bytes": 0})
            continue
        ok, detail = verify(path, int(row["bytes"]) if row["bytes"] else 0)
        actual = path.stat().st_size
        delta = actual - int(row["bytes"]) if row["bytes"] else 0
        results.append({**row, "status": "ok" if ok else "FAIL",
                        "detail": detail, "actual_bytes": actual,
                        "bytes_vs_catalog": delta})
        if not ok:
            failed.append({**row, "detail": detail})

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"verified {ok_count} of {len(rows)} objects")
    print(f"  failed  {len(failed)}")
    print(f"  missing {len(missing)}")
    print(f"report: {REPORT}")

    deltas = {r["bytes_vs_catalog"] for r in results
              if r["status"] == "ok" and r.get("bytes_vs_catalog")}
    if deltas:
        print(f"\nobjects whose size differs from the NARA catalog but verify clean: "
              f"offsets {sorted(deltas)}")

    if missing:
        print(f"\nMISSING: {len(missing)} objects not on disk", file=sys.stderr)
        for r in missing[:20]:
            print(f"  {r['series_label']} {r['fiscal_year']} {r['filename']}", file=sys.stderr)
    if failed:
        print(f"\nFAIL: {len(failed)} objects did not verify", file=sys.stderr)
        for r in failed:
            print(f"  {r['series_label']} {r['fiscal_year']} {r['filename']}: {r['detail']}",
                  file=sys.stderr)

    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
