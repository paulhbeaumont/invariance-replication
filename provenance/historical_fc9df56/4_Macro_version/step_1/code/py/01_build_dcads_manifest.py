#!/usr/bin/env python3
"""Build the download manifest for the NARA DCADS and MPCF record series.

Queries the National Archives catalog for every file unit under the two prime
contract series and writes one manifest row per digital object.

Series 606901: Military Prime Contract File (MPCF), FY1966-FY1975.
Series 578589: Defense Contract Action Data System (DCADS), FY1976-FY2006.

Output: $source/DCADS/manifest.csv
"""

import csv
import json
import re
import sys
import urllib.request

import dcads_common as dc

SOURCE_DIR = dc.SOURCE_DIR
MANIFEST = SOURCE_DIR / "manifest.csv"

API = "https://catalog.archives.gov/proxy/records/parentNaId/{naid}?limit=500"
SERIES = {
    "606901": "MPCF",
    "578589": "DCADS",
}

DOC_SUFFIXES = (".pdf", ".doc", ".lay", ".txt")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def fiscal_year(title):
    """Pull the fiscal year label out of a file unit title."""
    m = re.search(r"Fiscal Year (\d{4})", title)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2}/\d{1,2}/(\d{4}))", title)
    if m:
        return m.group(2)
    return ""


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for naid, label in SERIES.items():
        payload = fetch(API.format(naid=naid))
        hits = payload["body"]["hits"]["hits"]
        print(f"{label} (series {naid}): {len(hits)} file units", flush=True)
        for hit in hits:
            record = hit["_source"]["record"]
            title = record.get("title", "")
            restriction = (record.get("accessRestriction") or {}).get("status", "")
            for obj in record.get("digitalObjects") or []:
                name = obj.get("objectFilename") or ""
                kind = "doc" if name.lower().endswith(DOC_SUFFIXES) else "data"
                rows.append(
                    {
                        "series": naid,
                        "series_label": label,
                        "na_id": hit["_id"],
                        "fiscal_year": fiscal_year(title),
                        "title": title,
                        "restriction": restriction,
                        "kind": kind,
                        "filename": name,
                        "bytes": obj.get("objectFileSize") or 0,
                        "url": obj.get("objectUrl") or "",
                    }
                )

    rows.sort(key=lambda r: (r["series_label"], r["fiscal_year"], r["kind"], r["filename"]))
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    data = [r for r in rows if r["kind"] == "data"]
    docs = [r for r in rows if r["kind"] == "doc"]
    total = sum(r["bytes"] for r in rows)
    print(f"\nmanifest: {MANIFEST}")
    print(f"  objects      {len(rows):>4}  {total / 1e9:.2f} GB")
    print(f"  data files   {len(data):>4}  {sum(r['bytes'] for r in data) / 1e9:.2f} GB")
    print(f"  documentation{len(docs):>4}  {sum(r['bytes'] for r in docs) / 1e9:.2f} GB")

    missing = [r for r in rows if not r["url"]]
    if missing:
        print(f"\nFAIL: {len(missing)} objects have no URL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
