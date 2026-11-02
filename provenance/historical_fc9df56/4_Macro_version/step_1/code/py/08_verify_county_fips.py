#!/usr/bin/env python3
"""Verify the county FIPS columns 07_add_county_fips.py wrote.

Scoped deliberately narrowly. Row counts, dollar decoding, the record's own
fiscal year and the county-sums-to-state invariant against the raw source bytes
are 06_verify_dcads_parse.py's job and are not repeated here; 06 is left
untouched and rerun afterwards, which is what confirms the rewrite disturbed
nothing. This script asks one question 06 cannot: is the geography right.

Four criteria, which can fail, then two reports, which cannot:

  1  domain      the four columns exist, hold only declared values, and agree
                 with each other
  2  completeness every well-formed county code in the 50 states and DC resolved
  3  state       no county was mapped into a state other than its own
  4  names       ZALF's county name agrees with the name the FY1984-96 files
                 carry for the FIPS code it maps to
  5  seam        REPORTED: how the FY1983 and FY1984-88 sets of counties differ
  6  coverage    REPORTED: which counties each era resolves to, and which ZARK
                 codes never appear in a FIPS era

Check 4 is the load-bearing one and the reason this script can say anything at
all. Every other check compares the new columns against the parse they were
derived from, so a systematic error would agree with itself. Check 4 does not:
ZALF's county names are ZARK's own naming, and the FY1984-96 names come from the
FIPS-era files, which are a different naming system entirely. A crosswalk that
sent Wisconsin's counties to the wrong FIPS would still satisfy checks 1 to 3 and
would fail here on almost every code.

It fails only on a mismatch nothing accounts for. Four codes disagree because the
place was renamed or absorbed between the eras while the FIPS stayed right; they
are listed individually in KNOWN_RENAMES, in the style of check 9 of 06, so that a
fifth stops the pipeline.

5 and 6 carry no criterion, on purpose. Both compare the crosswalk against data
derived from the same parse, so neither can confirm it, and any threshold on them
would be a number invented in order to be passed - which reads as verification
while testing nothing. They are here because they are worth reading: 6 names the
counties that stopped existing before FY1984, which is what Part 1.4 has to map
forward, and 5 is where the Caddo Parish reassignment shows up.

Output: $temp/dcads_year/county_fips_verification_report.csv
"""

import collections
import csv
import difflib
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

REPORT = dc.OUT_DIR / "county_fips_verification_report.csv"

ZARK_LAYOUT = "dcads_272"
DELIM_LAYOUT = "dcads_delim_2004"

# Check 6 is a magnitude check, not an exactness one, and says so. The FIPS eras
# need no crosswalk, so their county counts are ground truth for how many counties
# receive defence contracts in a decade. The ZARK era, which does need one, should
# land in the same neighbourhood: a crosswalk that invented or lost counties
# wholesale would be out by hundreds. The band is wide because the eras genuinely
# differ - the reporting threshold moves twice and the panel is sparse.
# The ten Alaska census divisions and three other entities that stopped existing
# before FY1984 and so cannot appear in a FIPS-era file. Named because they are
# the codes Part 1.4 has to map forward; the rest of the ZARK-only list is
# ordinary sparsity.
#
# Moved to dcads_common so that 10_build_county_universe.py reads the same copy
# and turns it into a guarded crosswalk. Here it is still only reported, which is
# 6's whole point: this script asserts nothing it cannot check independently.
SUPERSEDED = dc.SUPERSEDED

# The four places where ZALF's name and the FY1984-96 name disagree while the FIPS
# code is right, because the place was renamed or absorbed between the two eras.
# Accounted for individually rather than tolerated, following check 9 of
# 06_verify_dcads_parse.py: a fifth would be a misread and has to fail.
KNOWN_RENAMES = {
    "51650": "Fort Monroe is an installation inside Hampton city and ZALF gives it "
             "Hampton's FIPS, as it also does for ZARK 650 = HAMPTON.",
    "56029": "The Wyoming side of Yellowstone National Park lies in Park County and "
             "ZALF gives it Park County's FIPS.",
    "02070": "The 1970-census Bristol Bay Division was renamed the Dillingham Census "
             "Area, keeping 02070. Not the Bristol Bay Borough, which is 02060.",
    "32510": "Ormsby County became Carson City in 1969, keeping FIPS 32510. ZARK 510 "
             "carries the new name on the same code.",
}

# Words that say what kind of place something is rather than which place, and so
# must not decide whether two names match. "Saint" is folded because the eras
# spell it three ways.
NOISE = {"COUNTY", "CITY", "BOROUGH", "PARISH", "CENSUS", "AREA", "DIVISION",
         "MUNICIPIO", "DISTRICT", "THE", "OF", "AND"}


def normalise(name):
    """A county name reduced to the tokens that identify the place."""
    text = (name or "").upper().replace("&", " AND ")
    text = re.sub(r"\bSTE?\b", "SAINT", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    tokens = [t for t in text.split() if t not in NOISE]
    return " ".join(tokens)


def names_agree(a, b):
    """True when two county names plausibly denote the same place.

    Generous on purpose. The eras abbreviate differently - "Fairbanks" against
    "Fairbanks North Star", "Wrangell-Petersbur" against "Wrangell-Petersburg" -
    and the ZARK files carry outright misspellings such as SNOHOKISH and CAPE
    GIREADEAU. What has to be caught is a county mapped to a different county,
    which shares no tokens at all.
    """
    x, y = normalise(a), normalise(b)
    if not x or not y:
        return True                       # nothing to compare against
    if x == y or x.startswith(y) or y.startswith(x):
        return True
    if set(x.split()) & set(y.split()):
        return True
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.80


def read_year(path, columns):
    return pq.read_table(path, columns=columns).to_pydict()


def main():
    files = sorted(dc.OUT_DIR.glob("fy*.parquet"),
                   key=lambda p: (dc.zark_year(p.stem[2:]), p.stem))
    if not files:
        print(f"FAIL: no parquet in {dc.OUT_DIR}", file=sys.stderr)
        return 1
    print(f"input {dc.OUT_DIR}\n{len(files)} fiscal years\n", flush=True)

    # ZALF's own state table, so check 3 can ask whether a county landed in the
    # state its ZARK code belongs to.
    zalf_states, zalf_counties, _ = dc.read_zalf()

    rows_out, failures = [], []
    # Gathered across years for checks 4, 5 and 6.
    zark_names = collections.defaultdict(collections.Counter)   # fips5 -> ZARK names
    zark_dollars = collections.Counter()                        # fips5 -> thousands
    zark_sources = collections.defaultdict(set)                 # fips5 -> fips_source
    ref_names = collections.defaultdict(collections.Counter)    # fips5 -> FY1984-96
    fips_by_era = collections.defaultdict(set)
    seam = {}

    print(f"{'fy':<9}{'rows':>10}{'county':>10}{'domain':>8}{'unres':>7}"
          f"{'wrongst':>9}{'counties':>10}")
    for path in files:
        label = path.stem[2:]
        year = dc.zark_year(label)
        cols = ["state_code", "county_name", "is_lumpsum", "dollars", "layout_id",
                "county_fips5", "state_fips2", "geo_class", "fips_source"]
        head = pq.ParquetFile(path).schema_arrow.names
        absent = [c for c in ("county_fips5", "state_fips2", "geo_class",
                              "fips_source") if c not in head]
        if absent:
            failures.append((label, f"missing columns {absent}; run 07 first"))
            print(f"FAIL fy{label}: missing {absent}", file=sys.stderr, flush=True)
            continue
        layout = pq.read_table(path, columns=["layout_id"]).column(0)[0].as_py()
        era = ("zark" if layout == ZARK_LAYOUT else
               "concat" if layout == DELIM_LAYOUT else "native")
        cols.append("county_fips" if era == "concat" else "county_code")
        d = read_year(path, cols)
        county_raw = d["county_fips" if era == "concat" else "county_code"]
        n = len(d["geo_class"])

        overrides = dc.overrides_for(label, era)
        domain, unresolved, wrong_state = 0, 0, 0
        counties = set()
        fails = []
        for i in range(n):
            g, c5, s2, src = (d["geo_class"][i], d["county_fips5"][i],
                              d["state_fips2"][i], d["fips_source"][i])
            # 1 domain
            if g not in dc.GEO_CLASSES or not src:
                domain += 1
            elif c5 and (len(c5) != 5 or not c5.isdigit()):
                domain += 1
            elif c5 and s2 != c5[:2]:
                domain += 1
            elif s2 and not (s2 in dc.STATE_FIPS or s2 in dc.TERRITORY_FIPS
                             or s2 in dc.ASSOCIATED_FIPS):
                domain += 1
            elif (g == "county") != bool(c5 and s2 in dc.STATE_FIPS):
                domain += 1
            # 2 completeness, recomputed from the parquet rather than trusted.
            # A hand-authored override deliberately drops a well-formed county
            # code, which is otherwise exactly the shape this check exists to
            # catch, so it is excluded here and held to account by 07's own
            # guard instead: 07 declares how many rows each override should hit
            # and fails if it hits a different number.
            if g == "unplaceable" and not c5:
                cr = (county_raw[i] or "").strip()
                width = 5 if era == "native" else 3
                if (len(cr) == width and cr.isdigit()
                        and cr[-3:] != dc.UNDISTRIBUTED_COUNTY
                        and not dc.is_blank_code(cr)
                        and ((d["state_code"][i] or "").strip(), cr) not in overrides):
                    unresolved += 1
            if not c5:
                continue
            counties.add(c5)
            # 3 no county in the wrong state, checked against ZALF's own state table
            if era == "zark" and d["fips_source"][i] == "zalf":
                want = zalf_states.get((d["state_code"][i] or "").strip(), "")
                if want.isdigit() and want != c5[:2]:
                    wrong_state += 1
            if era == "zark":
                zark_names[c5][d["county_name"][i].strip()] += 1
                zark_dollars[c5] += d["dollars"][i] or 0
                zark_sources[c5].add(src)
            elif 1984 <= year <= 1996:
                ref_names[c5][d["county_name"][i].strip()] += 1
            fips_by_era["FY1984-96" if 1984 <= year <= 1996 else
                        "FY1997-2003" if 1997 <= year <= 2003 else
                        "zark" if era == "zark" else "FY2004-06"].add(c5)

        if label == "1983":
            seam["1983"] = set(counties)
        if 1984 <= year <= 1988:
            seam.setdefault("1984-88", set()).update(counties)

        if domain:
            fails.append(f"{domain:,} rows violate the column domain")
        if unresolved:
            fails.append(f"{unresolved:,} well-formed county codes did not resolve")
        if wrong_state:
            fails.append(f"{wrong_state:,} counties sit in the wrong state")
        if fails:
            failures.append((label, "; ".join(fails)))
        klass = collections.Counter(d["geo_class"])
        rows_out.append({"fiscal_year": label, "era": era, "rows": n,
                         "county_rows": klass["county"], "domain_violations": domain,
                         "unresolved": unresolved, "wrong_state": wrong_state,
                         "distinct_counties": len(counties),
                         **{f"rows_{k}": klass[k] for k in dc.GEO_CLASSES}})
        print(f"{label:<9}{n:>10,}{klass['county']:>10,}{domain:>8,}{unresolved:>7,}"
              f"{wrong_state:>9,}{len(counties):>10,}", flush=True)

    # ---- 4 the independent name check -------------------------------------
    print(f"\n{'=' * 78}\n  check 4, the county names against the FY1984-96 names for "
          f"the same FIPS")
    print("  The only check whose two sides come from different naming systems, so the")
    print("  only one a systematically wrong crosswalk cannot satisfy by agreeing with")
    print("  itself. A crosswalk that sent a state's counties to the wrong FIPS would")
    print("  pass checks 1 to 3 and fail here on nearly every code.")
    print("\n  4a is the pass criterion and compares ZALF's own county name against the")
    print("  FY1984-96 name, one comparison per code. 4b compares the name the contract")
    print("  records carry, which is the same naming system as ZALF's and so cannot")
    print("  confirm it, but does surface both the supplement's deliberate judgements and")
    print("  corruption in individual records. 4b is reported, never a criterion.")

    compared = 0
    unintended = []
    for (zs, zc), (zalf_name, fips5) in sorted(zalf_counties.items()):
        ref = ref_names.get(fips5)
        if not ref or fips5 not in zark_names:
            continue
        compared += 1
        ref_name = ref.most_common(1)[0][0]
        if not names_agree(zalf_name, ref_name):
            unintended.append((fips5, f"{zs}/{zc} {zalf_name}", ref_name,
                               zark_dollars[fips5]))
    known = [e for e in unintended if e[0] in KNOWN_RENAMES]
    unknown = [e for e in unintended if e[0] not in KNOWN_RENAMES]
    print(f"\n  4a  {compared:,} ZALF county records reach a FIPS code that FY1984-96 "
          f"also names; {len(unintended)} disagree, {len(unknown)} of them unaccounted for")
    if known:
        print("\n      renamed or absorbed between the eras, FIPS correct, each one "
              "accounted for:")
        for fips5, a, b, dollars in sorted(known, key=lambda e: -e[3]):
            print(f"      {fips5}  {a[:28]:<29}-> {b}")
            print(f"               {KNOWN_RENAMES[fips5]}")
    if unknown:
        print(f"\n      NOT accounted for:")
        print(f"      {'fips':<8}{'ZALF name':<30}{'FY1984-96 name':<26}{'k$ 1966-83':>13}")
        for fips5, a, b, dollars in sorted(unknown, key=lambda e: -e[3]):
            print(f"      {fips5:<8}{a[:29]:<30}{b[:25]:<26}{dollars:>13,}")
        failures.append(("names", f"{len(unknown)} ZALF county records carry a name "
                                  f"that disagrees with the FY1984-96 name for the FIPS "
                                  f"they map to"))

    record_diff = []
    for fips5, names in sorted(zark_names.items()):
        ref = ref_names.get(fips5)
        if not ref:
            continue
        zark_name = names.most_common(1)[0][0]
        if names_agree(zark_name, ref.most_common(1)[0][0]):
            continue
        record_diff.append((fips5, zark_name, ref.most_common(1)[0][0],
                            sum(names.values()), zark_dollars[fips5],
                            "supplement" if "supplement" in zark_sources[fips5] else "zalf"))
    print(f"\n  4b  {len(zark_names):,} FIPS codes appear in the ZARK era; "
          f"{len(record_diff)} carry a record name that disagrees with FY1984-96")
    if record_diff:
        print(f"      {'fips':<8}{'ZARK record name':<22}{'FY1984-96 name':<22}"
              f"{'rows':>7}{'k$':>12}  source")
        for fips5, a, b, rows, dollars, src in sorted(record_diff, key=lambda e: -e[4]):
            print(f"      {fips5:<8}{a[:21]:<22}{b[:21]:<22}{rows:>7,}{dollars:>12,}  {src}")

    # ---- 5 the FY1983/84 seam ---------------------------------------------
    print(f"\n{'=' * 78}\n  check 5, the FY1983 to FY1984 seam")
    print("  FY1983 is the last ZARK year and FY1984 the first FIPS year. The two sets of")
    print("  counties should differ only where a county genuinely stopped or started")
    print("  receiving contracts. Hundreds of differences would mean the crosswalk is")
    print("  systematically wrong; tens mean the panel is sparse, which it is.")
    a, b = seam.get("1983", set()), seam.get("1984-88", set())
    only83, only84 = sorted(a - b), sorted(b - a)
    print(f"\n    FY1983 {len(a):,} counties, FY1984-88 {len(b):,}, "
          f"{len(a & b):,} in both ({100 * len(a & b) / max(len(a), 1):.1f}% of FY1983)")
    print(f"    only FY1983: {len(only83)}   only FY1984-88: {len(only84)}")
    if only83:
        print("    FY1983 codes absent from FY1984-88, with their ZARK names:")
        for fips5 in only83[:40]:
            nm = zark_names[fips5].most_common(1)[0][0] if zark_names.get(fips5) else ""
            print(f"      {fips5}  {nm:<24}{zark_dollars[fips5]:>12,} k$")

    # ---- 6 counts ----------------------------------------------------------
    print(f"\n{'=' * 78}\n  6, which counties each era resolves to - REPORTED, NOT A "
          f"CRITERION")
    print("  There is no honest pass criterion here. Every count or set test available")
    print("  compares the crosswalk against data derived from the same parse, so it could")
    print("  only ever agree with itself, and any threshold on the counts would be a")
    print("  number invented to be passed. Check 4 is the criterion that does this work,")
    print("  because its two sides come from different naming systems. What follows is")
    print("  for reading, and the superseded block is input to Part 1.4.")
    print()
    for era in ("zark", "FY1984-96", "FY1997-2003", "FY2004-06"):
        print(f"    {era:<14}{len(fips_by_era[era]):>7,} counties")

    fips_era = set().union(*(fips_by_era[e] for e in
                             ("FY1984-96", "FY1997-2003", "FY2004-06")))
    only_zark = sorted(fips_by_era["zark"] - fips_era)
    gone = [f for f in only_zark if f in SUPERSEDED]
    sparse = [f for f in only_zark if f not in SUPERSEDED]
    print(f"\n    {len(only_zark)} codes appear in the ZARK era and in no FIPS-era year.")
    print(f"    {len(gone)} stopped existing before FY1984 and so cannot appear again -")
    print(f"    these are the ones a time-consistent county key has to map forward:")
    for fips5 in sorted(gone, key=lambda f: -zark_dollars[f]):
        nm = zark_names[fips5].most_common(1)[0][0]
        print(f"      {fips5}  {nm:<22}{zark_dollars[fips5]:>12,} k$   {SUPERSEDED[fips5]}")
    print(f"\n    the other {len(sparse)} are counties that still exist and simply drew no")
    print(f"    defence contract after FY1983, ordered by ZARK-era dollars:")
    for fips5 in sorted(sparse, key=lambda f: -zark_dollars[f]):
        nm = zark_names[fips5].most_common(1)[0][0]
        rows = sum(zark_names[fips5].values())
        print(f"      {fips5}  {nm:<22}{zark_dollars[fips5]:>12,} k${rows:>7} rows")

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n{'=' * 78}")
    print(f"verified {len(rows_out) - len({f[0] for f in failures} & {r['fiscal_year'] for r in rows_out})}"
          f" of {len(rows_out)} fiscal years")
    print(f"report: {REPORT}")
    if failures:
        print(f"\nFAIL: {len(failures)} checks did not pass", file=sys.stderr)
        for what, why in failures:
            print(f"  {what}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
