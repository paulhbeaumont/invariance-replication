#!/usr/bin/env python3
"""The 2004 BEA Economic Areas, parsed onto the CCG universe.

WHY THIS FILE EXISTS

The aggregation ladder jumps from 410 MSA units straight to 51 states, a factor
of eight, with nothing in between. BEA Economic Areas fill it at 179 units, and
their Component Economic Areas fill it again at 344. Both are built from whole
counties and both cover the ENTIRE country, rural counties included, which is
what neither the CBSA nor the MSA rung manages: 1,325 of our counties sit in no
CBSA at all and have to be pooled into an administrative rest-of-state residual.
An economic area is a real economic geography, a metropolitan or micropolitan
node plus the surrounding counties economically tied to it.

WHICH VINTAGE, AND WHY BEING RETIRED IS AN ADVANTAGE

The November 2004 redefinition, which is the last one BEA made. BEA stopped
publishing economic-area statistics with the release of 21 November 2013, citing
sequestration, and the delineation has not been revised since. For a panel that
needs ONE fixed geography across 1975-2018 that is a feature rather than a
problem: a live delineation would be revised under us, where a retired one cannot
be. It is the same argument 21 makes for freezing the 2003 CBSA vintage, and here
it costs nothing at all.

WHERE THE FILE COMES FROM, WHICH NEEDS SAYING PLAINLY

BEA no longer serves it. Both documentation pages return a "page moved" stub, the
current regional-definitions page carries no economic-area link, and the 2004
Survey of Current Business article has the 179 codes and names but not the county
composition. The file is recovered from the Internet Archive's copy of BEA's own
page as it stood in 2006, when economic areas were still current. The payload is
BEA's own workbook, authored and last saved by BEA in November 2004, and its
sha256 is pinned below so a substituted file fails the run rather than passing
quietly.

WHAT THE WORKBOOK CONTAINS

One sheet, 3,142 rows, one per county, with four nested geographies:

    OMB CBSA code and type      the 2003 delineation 21 already parses
    OMB CSA code                combined statistical areas
    BEA CEA code                344 component economic areas
    BEA EA code                 179 economic areas

Verified against our own universe: every one of the 3,101 consistent county
groups receives an economic area, 179 distinct, median 13 counties each, and NO
group is split across two economic areas, so counties nest exactly and the
aggregate outcome is a true partition of the baseline. The component areas split
2 groups of 3,101, which is reported and assigned by plurality of population.

Output:
    dc.BEA_AREA_SOURCE_DIR/eastructure.xls    the workbook, as served
    dc.BEA_AREA_SOURCE_DIR/download_receipt.csv
    code/reference/county_bea_area_2004.csv   county_fips5, cea_code, cea_name,
                                              ea_code, ea_name
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

# The Internet Archive's copy of BEA's own file as it stood on 22 September 2006.
# The "id_" before the path is load-bearing: without it the archive answers with
# a ten-kilobyte Wayback interstitial under a 200, which is a wrong payload
# rather than an error and would pass any check that only looked at the status.
URL = ("http://web.archive.org/web/20060922191726id_/"
       "http://www.bea.gov/bea/regional/docs/eastructure.xls")
RAW_NAME = "eastructure.xls"
# BEA's own workbook: 578,560 bytes, created 29 September 2004 and last saved
# 4 November 2004. A different payload at this address fails the run.
EXPECT_BYTES = 578_560
EXPECT_SHA = "26c339cfd577c1e57b3d5785699dd8d2e1b5fa9524ca8b480c4b0bf9569886ac"

SOURCE_DIR = dc.CURRENT_PAPER_DATA_ROOT / "source" / "BEA_AREAS"
OUT = CODE.parent / "reference" / "county_bea_area_2004.csv"

N_EA, N_CEA, N_COUNTIES = 179, 344, 3_142


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest):
    """urllib first, curl as the fallback, and record which one worked.

    The fallback is not defensive padding: 21 documents that urllib raises
    SSLCertVerificationError against census.gov on this machine because the venv
    does not use the system trust store, and curl succeeds. The same applies
    here, and which path was taken goes into the receipt.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    transport = None
    try:
        with urllib.request.urlopen(URL, timeout=180) as r, open(tmp, "wb") as f:
            f.write(r.read())
        transport = "urllib"
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"    urllib: {exc.__class__.__name__}")
    # A transport that RETURNS is not a transport that succeeded. The archive
    # answers urllib with a ten-kilobyte interstitial rather than the workbook,
    # and a 200 carrying the wrong payload is the failure this guard exists for.
    if transport is None or tmp.stat().st_size != EXPECT_BYTES:
        got = tmp.stat().st_size if tmp.exists() else 0
        print(f"    urllib returned {got:,} bytes, not {EXPECT_BYTES:,}; "
              f"retrying with curl")
        subprocess.run(["curl", "-sSL", "--fail", "--max-time", "180",
                        "-o", str(tmp), URL], check=True)
        transport = "curl"
    if tmp.stat().st_size != EXPECT_BYTES:
        raise SystemExit(f"  FAIL: got {tmp.stat().st_size:,} bytes from both "
                         f"transports, not {EXPECT_BYTES:,}. The archive rate "
                         f"limits; wait and rerun.")
    tmp.replace(dest)
    return transport


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", action="store_true",
                    help="print the sha256 of the downloaded file and exit, for "
                         "recording in EXPECT_SHA")
    args = ap.parse_args()

    print("=" * 78)
    print("  BEA Economic Areas, 2004 redefinition")
    print("=" * 78)

    raw = SOURCE_DIR / RAW_NAME
    if raw.exists():
        print(f"  already downloaded: {raw}")
        transport = "cached"
    else:
        print(f"  downloading {URL}")
        transport = download(raw)
    size, digest = raw.stat().st_size, sha256(raw)
    print(f"  {size:,} bytes, sha256 {digest[:16]}..., via {transport}")
    if args.pin:
        print(f"\n  EXPECT_SHA = \"{digest}\"")
        return 0
    if size != EXPECT_BYTES:
        raise SystemExit(f"  FAIL: {size:,} bytes against the expected "
                         f"{EXPECT_BYTES:,}; the address is serving something else")
    if EXPECT_SHA and digest != EXPECT_SHA:
        raise SystemExit("  FAIL: sha256 does not match the pinned value")

    d = pd.read_excel(raw, header=0, dtype=str)
    d.columns = [str(c).strip() for c in d.columns]
    # The EA column's header carries doubled spaces in BEA's own file, so it is
    # found by shape rather than by an exact string.
    eacol = [c for c in d.columns
             if c.startswith("BEA") and "CEA" not in c and "Code" in c]
    if len(eacol) != 1:
        raise SystemExit(f"  FAIL: expected one BEA EA column, found {eacol}")
    d = d.rename(columns={"County FIPS Code": "county_fips5",
                          "BEA CEA Code": "cea_code", eacol[0]: "ea_code",
                          "BEA CEA Name": "cea_name", "BEA EA Name": "ea_name"})
    keep = ["county_fips5", "cea_code", "cea_name", "ea_code", "ea_name"]
    d = d[[c for c in keep if c in d.columns]].copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    d["county_fips5"] = d["county_fips5"].str.zfill(5)

    checks = [
        ("counties in the workbook", len(d), N_COUNTIES),
        ("distinct economic areas", d["ea_code"].nunique(), N_EA),
        ("distinct component areas", d["cea_code"].nunique(), N_CEA),
    ]
    print()
    ok = True
    for lab, got, want in checks:
        good = got == want
        ok &= good
        print(f"  {lab:<32}{got:>7,} against {want:>7,}   "
              f"{'PASS' if good else 'FAIL'}")

    # Nesting, which is what the aggregation test actually requires: a consistent
    # county group must not straddle two areas, or the aggregate is not a
    # partition of the baseline.
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"].assign(ccg=lambda t:
                                                  t["ccg_id"].astype(int))
    m = xw.merge(d, on="county_fips5", how="left")
    for k in ("ea_code", "cea_code"):
        sub = m.dropna(subset=[k])
        split = int((sub.groupby("ccg")[k].nunique() > 1).sum())
        covered = sub["ccg"].nunique()
        print(f"  {k:<32}{covered:>7,} of {m['ccg'].nunique():,} groups covered, "
              f"{split} split across two")
        ok &= covered == m["ccg"].nunique()
        if k == "ea_code":
            ok &= split == 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    rec = SOURCE_DIR / "download_receipt.csv"
    pd.DataFrame([dict(url=URL, bytes=size, sha256=digest, transport=transport,
                       parsed_rows=len(d), out=str(OUT))]).to_csv(rec, index=False)
    print(f"\n  wrote {OUT}")
    print(f"  wrote {rec}")
    if not ok:
        raise SystemExit("  FAIL: a check above did not pass")
    print("  all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
