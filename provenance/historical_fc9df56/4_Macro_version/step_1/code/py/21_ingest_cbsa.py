#!/usr/bin/env python3
"""The 2003 OMB CBSA delineation, parsed onto the CCG universe. Part 1.5.

WHY THIS FILE EXISTS

23_sample_screen_grid.py needs to know which counties are "purely rural", and
nothing in this project has ever carried an urban/rural classification. The
definition taken is Auerbach, Gorodnichenko and Murphy's (step_1/s5-auerbach.pdf,
footnote 4): a core-based statistical area is "one or more counties (or
equivalents) with an urban center of at least 10,000 people", plus the outlying
counties commuting into it. Their unit of analysis IS the CBSA, so a county in no
CBSA is a county their design never sees, and that is the screen.

WHICH VINTAGE, AND WHY IT IS FROZEN

June 6, 2003 - the first CBSA delineation, and the one footnote 4 describes.
It sits in the middle of the 1975-2018 panel. A moving delineation would make the
screen a different county set in each era, and then a pre/post difference in F
would mix a change in the instrument with a change in who is in the sample. One
fixed set is the whole point; the anachronism at either end of the panel is the
price and is smaller than the alternative.

THE FILE IS FIXED-WIDTH, AND ONLY LOOKS DELIMITED

    columns 0:6    CBSA code, on both the header line and each county line
    columns 7:16   division code, present only for the eleven divided CBSAs
    columns 16:22  county FIPS, present ONLY on county lines
    columns 22:    the title - the CBSA's own on a header line, the county's
                   name on a county line

A header line is what says Metropolitan or Micropolitan; the county lines that
follow it inherit that type. Parsing this with a whitespace split puts the
division code in the county column for the eleven divided areas - New York,
Chicago, Los Angeles among them - so the columns are sliced, not split.

Output:
    dc.CBSA_SOURCE_DIR/03mfips.txt             the raw delineation, as served
    dc.CBSA_SOURCE_DIR/download_receipt.csv    url, bytes, sha256, transport
    dc.CBSA_CROSSWALK                          county_fips5, cbsa_code,
                                               cbsa_title, cbsa_type
    dc.LOG_DIR/21_ingest_cbsa.log
"""

import argparse
import csv
import hashlib
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

RAW_NAME = "03mfips.txt"
ENCODING = "latin-1"        # the file carries Spanish place names in Puerto Rico

# The column slices above, named once so the parser and its docstring cannot
# drift apart.
COL_CBSA = slice(0, 6)
COL_DIVISION = slice(7, 16)
COL_COUNTY = slice(16, 22)
COL_TITLE = slice(22, None)

METRO = "Metropolitan"
MICRO = "Micropolitan"

# What the June 2003 delineation contains, asserted rather than trusted. If
# census.gov ever reposts a different vintage at this URL these are what fail.
EXPECT_COUNTIES = 1838
EXPECT_METRO = 1159         # counties, not areas
EXPECT_AREAS = {METRO: 370, MICRO: 565}   # 935 CBSAs, plus 29 metropolitan
                                          # divisions inside eleven of them

# Counties in the delineation that our crosswalk does not contain. Declared by
# name so that a drop is a decision and not an accident: every one is Puerto
# Rico, which has no DCADS county records, no BEA CAINC1 area and no EFSY
# employment, and so has never been in the CCG universe. An unmatched FIPS from
# any other state is a failure.
UNMATCHED_STATES = {"72": "Puerto Rico - outside the CCG universe entirely"}


def check(failures, ok, num, label, detail=""):
    """One numbered PASS/FAIL line, in 18_ingest_muratori.py's format."""
    print(f"    [{num}] {label:<50s} {detail:<40s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label} {detail}"))
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(refresh=False):
    """Download 03mfips.txt if it is not cached. Returns (path, transport).

    urllib is tried first, for consistency with 12_build_county_income.py, and
    curl is the fallback. The fallback is not defensive padding: on this machine
    urllib raises SSLCertVerificationError against census.gov because the venv
    carries no certifi bundle and Python does not read the macOS keychain, while
    curl - which does - succeeds. Which path was used is written into the
    receipt, because a file fetched with a verified chain and one fetched without
    are not equally trustworthy and the receipt is where that is recorded.
    """
    dc.CBSA_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    dest = dc.CBSA_SOURCE_DIR / RAW_NAME
    transport = "cached"

    if refresh or not dest.exists():
        tmp = dest.with_suffix(".txt.tmp")
        print(f"    downloading {dc.CBSA_URL}")
        try:
            with urllib.request.urlopen(dc.CBSA_URL, timeout=120) as r, \
                    open(tmp, "wb") as out:
                out.write(r.read())
            transport = "urllib"
        except urllib.error.URLError as exc:
            # urllib wraps the SSL error in a URLError, so the reason is what
            # has to be inspected. Only a certificate failure falls back; a 404
            # or a timeout is a real problem and must not be retried into a
            # different transport that would report the same thing.
            if not isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            print(f"    urllib: {exc.reason.__class__.__name__}; "
                  f"falling back to curl")
            subprocess.run(["curl", "-sSL", "--fail", "-o", str(tmp), dc.CBSA_URL],
                           check=True)
            transport = "curl"
        tmp.replace(dest)
    else:
        print(f"    using the cached copy at {dest}")

    digest = sha256(dest)
    with open(dc.CBSA_SOURCE_DIR / "download_receipt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "url", "bytes", "sha256", "transport"])
        w.writerow([RAW_NAME, dc.CBSA_URL, dest.stat().st_size, digest, transport])
    print(f"    {dest.stat().st_size:,} bytes   sha256 {digest[:16]}...   "
          f"transport {transport}")
    return dest, transport


def parse(path):
    """(frame, n_cbsas) - one row per county FIPS in a 2003 CBSA.

    A line is a county line if its FIPS slice holds five digits, and a header
    line if it does not but its title slice is non-empty. The type carried
    forward is the last header's, which is why the file is read in order and not
    filtered first.
    """
    rows, areas, cbsa_type, cbsa_title = [], {}, None, None
    with open(path, encoding=ENCODING) as fh:
        for line in fh:
            cbsa = line[COL_CBSA].strip()
            if not (len(cbsa) == 5 and cbsa.isdigit()):
                continue
            division = line[COL_DIVISION].strip()
            county = line[COL_COUNTY].strip()
            title = line[COL_TITLE].strip()
            if len(county) == 5 and county.isdigit():
                rows.append((county, cbsa, cbsa_title, cbsa_type))
                continue
            if not title:
                continue
            cbsa_title = title
            cbsa_type = (METRO if METRO in title else
                         MICRO if MICRO in title else None)
            # A metropolitan division header carries its parent's CBSA code and
            # its own division code, and its title says "Metropolitan Division".
            # It is not a CBSA and must not be counted as one; its counties keep
            # the parent code, which is the level the screen operates at.
            if not division:
                areas[cbsa] = cbsa_type
    out = pd.DataFrame(rows, columns=["county_fips5", "cbsa_code",
                                      "cbsa_title", "cbsa_type"])
    # Counties inside a divided CBSA appear under the division line, and a
    # county on a state line can appear under two CBSAs; the screen is about
    # membership, so a county appears once.
    out = out.drop_duplicates("county_fips5").sort_values("county_fips5")
    return out.reset_index(drop=True), areas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download 03mfips.txt even if it is already cached")
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(dc.LOG_DIR / "21_ingest_cbsa.log", "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log.write(s)

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = Tee()
    try:
        failures = []
        print("=" * 78)
        print("The 2003 OMB CBSA delineation -> county_cbsa_2003.csv")
        print("=" * 78)

        print("\n[0] the source")
        path, transport = fetch(refresh=args.refresh)

        print("\n[1] the parse")
        cbsa, areas = parse(path)
        n_metro = int((cbsa["cbsa_type"] == METRO).sum())
        n_micro = int((cbsa["cbsa_type"] == MICRO).sum())
        got_areas = {t: sum(1 for v in areas.values() if v == t)
                     for t in (METRO, MICRO)}
        print(f"    {len(cbsa):,} county FIPS in {len(areas)} CBSAs "
              f"({got_areas[METRO]} metro, {got_areas[MICRO]} micro)   "
              f"{n_metro:,} metropolitan counties   {n_micro:,} micropolitan")

        check(failures, len(cbsa) == EXPECT_COUNTIES, 1,
              "county count is the June 2003 delineation's",
              f"{len(cbsa):,} of {EXPECT_COUNTIES:,}")
        check(failures, n_metro == EXPECT_METRO and got_areas == EXPECT_AREAS, 2,
              "metro counties and the area split are that vintage's",
              f"{n_metro:,} counties, {got_areas[METRO]}+{got_areas[MICRO]} areas")
        check(failures, cbsa["cbsa_type"].notna().all(), 3,
              "every county inherits a Metro/Micro type",
              f"{int(cbsa['cbsa_type'].isna().sum())} untyped")

        print("\n[2] onto the CCG universe")
        fips_to_ccg, _, universe = dc.load_ccg_map()
        cbsa["ccg"] = cbsa["county_fips5"].map(fips_to_ccg)
        unmatched = cbsa[cbsa["ccg"].isna()]
        by_state = unmatched["county_fips5"].str[:2].value_counts().to_dict()
        for st, n in sorted(by_state.items()):
            why = UNMATCHED_STATES.get(st, "UNDECLARED")
            print(f"    state {st}: {n:,} county FIPS not in the crosswalk   [{why}]")
        check(failures, set(by_state) <= set(UNMATCHED_STATES), 4,
              "every unmatched FIPS is a declared state",
              f"{len(unmatched)} unmatched in {sorted(by_state)}")

        in_any = set(cbsa.loc[cbsa["ccg"].notna(), "ccg"].astype(int))
        in_metro = set(cbsa.loc[cbsa["ccg"].notna() & (cbsa["cbsa_type"] == METRO),
                                "ccg"].astype(int))
        uni = set(int(c) for c in universe)
        print(f"    {len(in_any):,} of {len(uni):,} CCGs touch a CBSA "
              f"-> {len(uni) - len(in_any):,} purely rural")
        print(f"    {len(in_metro):,} of {len(uni):,} CCGs touch a METRO CBSA "
              f"-> {len(uni) - len(in_metro):,} dropped by the stricter screen")

        # A CCG is a GROUP of counties, so "in a CBSA" has to mean "any member
        # is". The alternative - all members - would drop a metropolitan county
        # merely for having been merged with a rural neighbour by the 1990
        # harmonisation, which is a property of our crosswalk and not of the
        # delineation.
        check(failures, in_metro <= in_any and in_any <= uni, 5,
              "metro CCGs subset CBSA CCGs subset the universe",
              f"{len(in_metro):,} <= {len(in_any):,} <= {len(uni):,}")
        check(failures, 0 < len(in_metro) < len(in_any) < len(uni), 6,
              "both screens bite and neither empties the sample",
              f"drops {len(uni) - len(in_any):,} and {len(uni) - len(in_metro):,}")

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            return 1

        out = cbsa[["county_fips5", "cbsa_code", "cbsa_title", "cbsa_type"]]
        out.to_csv(dc.CBSA_CROSSWALK, index=False)
        print(f"\nall checks passed   {len(out):,} rows -> {dc.CBSA_CROSSWALK}")
        return 0
    finally:
        sys.stdout = sys.__stdout__
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
