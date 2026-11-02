#!/usr/bin/env python3
"""BEA county total employment, 1969-2000, on the CCG universe. Part 1.5, section 1.8.10.

WHY THIS EXISTS

26_winsor_screen_grid.py flags counties on the distribution of their EMPLOYMENT
GROWTH, measured over the same window as the era's fixed exposure - 1969-74 on the
pre era, 1993-97 on the post era. EFSY, the employment panel every outcome in this
project is built on, begins in 1975 (README 1.6.10: "FY1966-74 has no employment"),
so it cannot supply a 1969-74 growth rate at all. An EFSY-based flag would have to
fall back to 1975->1976, two years against the exposure's six, and the pre and post
panels would then be the same rule on differently-sized windows - the family of
defect 19_first_stage_table.py exists to have fixed.

BEA's Regional Economic Accounts reach back to 1969, so the flag can honour the
window rule exactly on both eras. That is the whole reason this file exists.

WHICH BEA TABLE, AND HOW THAT WAS ESTABLISHED                         2026-08-13

Not by memory. Four tables were downloaded and read before this one was chosen:

    CAINC4     Personal Income and Employment by Major Component - despite the
               name, 20 line codes, ALL of them income. No employment at all.
    CAINC30    Economic Profile - 23 line codes, no total employment either.
    CAEMP25N   Total employment by industry, NAICS basis - 2001-2022 only, so it
               covers NEITHER flag window.
    CAEMP25S   Total employment by industry, SIC basis - 1969-2000. This one.

CAEMP25S LineCode 10 is "Total employment (number of jobs)", 3,192 areas, and its
1969-2000 span CONTAINS BOTH FLAG WINDOWS with room to spare. So the SIC-to-NAICS
break at 2001 never enters the flag: there is no splice here, and there is no
version of this file that needs CAEMP25N.

That also fixes this file's span at 1969-2000. It is not a general-purpose
employment panel and must not be used as one - EFSY is that, and it is the series
every OUTCOME is built on. See the warning below.

THE UNIT IS NOT EFSY's, AND THAT MATTERS TO WHOEVER READS TABLE XIII

    BEA total employment   jobs by place of work: wage and salary jobs PLUS
                           proprietors, farm and government included, full- and
                           part-time counted alike, one job one count.
    EFSY / CBP             paid employees of private establishments in covered
                           industries, at a March 12 reference date.

So Table XIII RANKS counties on BEA employment growth and ESTIMATES on EFSY
employment growth. For a flag that is arguably a feature rather than a defect - it
makes the flag manifestly not a lagged copy of the dependent variable, which is
the worry that put the measurement window outside the estimation era in the first
place. It is stated in the table note either way, and it is why this file writes
`bea_emp` and not `emp`: a column named `emp` would eventually be joined to
something that meant EFSY.

THE GEOGRAPHY IS 12_build_county_income.py's, REUSED AND NOT RESTATED

BEA's county set is not ours - 3,192 areas here against 3,101 groups - and it
disagrees in exactly the four ways 12 already declares: the Virginia combinations,
the Alaska time-split at 1979, the two straddles, and areas outside the window.
Every one of those declarations, and the resolver and straddle rules that apply
them, are imported from 12. Restating any of them would be how this file and the
income panel end up disagreeing about which county Menominee is.

The check that catches any of it going wrong is the same one too: BEA's areas
partition each state, so a dropped combination, a missed Alaska division or a
double-counted straddle shows up immediately as a state that no longer adds up.

Output:
    dc.PANEL_OUT_DIR/bea_employment_ccg_year.parquet
    dc.LOG_DIR/25_ingest_bea_employment.log
"""

import argparse
import csv
import io
import importlib.util
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 12 owns the BEA geography: the reader, the area resolver, the straddle rules and
# the four declarations behind them. Imported rather than restated. Its writers
# live behind its main(), which exec_module does not run, so this import writes
# nothing and cannot touch income_ccg_year.parquet.
inc = _load("build_county_income", "12_build_county_income.py")

TABLE = "CAEMP25S"
URL = f"https://apps.bea.gov/regional/zip/{TABLE}.zip"
MEMBER = f"{TABLE}__ALL_AREAS_1969_2000.csv"

FIRST_YEAR = 1969          # CAEMP25S begins here, and so does the pre flag window
LAST_YEAR = 2000           # and ends here. NOT 2018 - see the docstring

# The line code is RESOLVED BY ITS DESCRIPTION, not hardcoded.
#
# 12 hardcodes LineCode "1" for income because that number has been stable since
# 1969 and its own checks would catch a change. Here the number is worth less
# trust: this table was picked out of four candidates on 2026-08-13 and the wrong
# line code would silently deliver wage-and-salary employment (20) or proprietors
# (40) under the name "total". So the code is found by matching the Description
# and the run fails if the match is not unique.
LINECODE_DESC = "Total employment"
LINECODE_UNIT = "Number of jobs"

# Kalawao HI, for the reason 12 declares: BEA has never separated it from Maui, so
# it has no income AND no employment in any year. Restated here so this file
# declares its own universe rather than inheriting one silently.
EMP_GAP_CCGS = (15005,)

VALUE_COL = "bea_emp"

failures = []


def check(ok, num, label, detail=""):
    print(f"    [{num}] {label:<52s} {detail:<40s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


def fetch(refresh=False):
    """Download the zip if it is not here, and verify it opens.

    12's rule, and its reason: a served size disagreeing with a published one is
    not evidence of a damaged file. What matters is that the central directory
    reads and the member is present, so that is what is asserted.
    """
    dest_dir = dc.BEA_SOURCE_DIR / TABLE
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{TABLE}.zip"
    if refresh or not dest.exists():
        print(f"    downloading {URL}")
        tmp = dest.with_suffix(".zip.tmp")
        with urllib.request.urlopen(URL, timeout=180) as r, open(tmp, "wb") as out:
            out.write(r.read())
        tmp.replace(dest)
    else:
        print(f"    already here: {dest}")
    with zipfile.ZipFile(dest) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise AssertionError(f"{TABLE}.zip: member {bad} fails CRC")
        if MEMBER not in zf.namelist():
            raise AssertionError(
                f"{TABLE}.zip does not contain {MEMBER}; BEA changed the layout. "
                f"Members: {zf.namelist()[:4]}")
    print(f"    {dest}   {dest.stat().st_size:,} bytes   "
          f"sha256 {inc.sha256_of(dest)[:16]}...")
    return dest


def resolve_linecode(zip_path):
    """The LineCode whose Description is total employment, found and not assumed.

    Returns the code as a string. Fails the run unless exactly one code matches
    and its Unit is the expected one - a table that started reporting thousands
    of jobs under the same description would otherwise pass straight through.
    """
    seen = {}
    with zipfile.ZipFile(zip_path) as zf, zf.open(MEMBER) as fh:
        for row in csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1",
                                                   newline="")):
            code = (row.get("LineCode") or "").strip()
            if code and code not in seen:
                seen[code] = ((row.get("Description") or "").strip(),
                              (row.get("Unit") or "").strip())
    hits = [c for c, (d, _) in seen.items() if d.startswith(LINECODE_DESC)]
    ok = len(hits) == 1 and seen[hits[0]][1] == LINECODE_UNIT
    detail = (f"LineCode {hits[0]} = {seen[hits[0]][0]!r}" if len(hits) == 1
              else f"{len(hits)} codes match {LINECODE_DESC!r}: {hits}")
    check(ok, 1, "the total-employment line code is unique", detail)
    if not hits:
        raise AssertionError(f"no LineCode described as {LINECODE_DESC!r}")
    return hits[0]


def to_ccg(raw, fips_to_ccg, universe, years):
    """BEA areas -> CCG-years, through 12's resolver and 12's straddle rules."""
    mapping, straddles, unmapped = inc.resolve_areas(
        raw["geofips"], fips_to_ccg, failures)

    # An unmapped area is only acceptable if it is outside the window - the 2020s
    # Connecticut planning regions and the two new Alaska areas. Anything else is
    # a county this file would silently drop.
    live = set(raw.loc[raw["year"].between(*years), "geofips"])
    stray = sorted(f for f in unmapped if f in live)
    check(not stray, 2, "every in-window BEA area maps to a CCG",
          "none unmapped" if not stray else f"{len(stray)}: {stray[:6]}")

    body = raw[raw["geofips"].isin(mapping)].copy()
    body["ccg"] = body["geofips"].map(mapping).astype(np.int64)
    body = body.groupby(["ccg", "year"], as_index=False)[VALUE_COL].sum()

    extra = inc.apply_straddle_rules(raw, straddles, fips_to_ccg, failures,
                                     verbose=True, value_col=VALUE_COL)
    if extra is not None and len(extra):
        body = pd.concat([body, extra[["ccg", "year", VALUE_COL]]],
                         ignore_index=True)
        body = body.groupby(["ccg", "year"], as_index=False)[VALUE_COL].sum()

    body = body[body["ccg"].isin(universe)]
    return body.sort_values(["ccg", "year"]).reset_index(drop=True)


def check_state_rollup(raw, emp, linecode, universe):
    """Check 4. Every state's CCG total equals BEA's own state row.

    THE LOAD-BEARING CHECK, and the same one that validated the income panel at
    max relative deviation 0.00e+00. BEA's areas partition each state, so a
    Virginia combination dropped, an Alaska division missed or a straddle
    double-counted shows up here immediately and cannot show up anywhere else -
    a county quietly assigned another county's employment leaves the county
    totals looking perfectly reasonable and the state total wrong.

    Alaska is EXCLUDED from the comparison and counted separately. Its 1969-78
    divisions and its 1979+ boroughs are different area sets that BEA does not
    reconcile, so a state total there is not the sum of one consistent set of
    areas in every year. 12 makes the same exclusion for the same reason.
    """
    states = raw[raw["geofips"].str.endswith("000")].copy()
    states["state"] = states["geofips"].str[:2].astype(int)
    states = states[states["state"].between(1, 56) & (states["state"] != 2)]
    want = states.groupby(["state", "year"])[VALUE_COL].sum()

    got = emp.copy()
    got["state"] = (got["ccg"] // 1000).astype(int)
    got = got[got["state"] != 2].groupby(["state", "year"])[VALUE_COL].sum()

    j = pd.concat([want.rename("bea"), got.rename("ours")], axis=1).dropna()
    rel = ((j["ours"] - j["bea"]).abs() / j["bea"].abs().clip(lower=1.0))
    worst = float(rel.max())
    where = ""
    if len(rel):
        i = rel.idxmax()
        where = f" worst state {i[0]} in {i[1]}"
    return check(worst < 1e-9 and len(j) > 0, 4,
                 "every state's CCG total equals BEA's own state row",
                 f"{len(j):,} state-years, max rel dev {worst:.2e}{where}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--last-year", type=int, default=LAST_YEAR)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if the zip is already here")
    args = ap.parse_args()

    tee = None
    try:
        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        sym = _load("first_stage_table", "19_first_stage_table.py")
        tee = sym.Tee(dc.LOG_DIR / "25_ingest_bea_employment.log")

        years = (args.first_year, args.last_year)
        span = list(range(args.first_year, args.last_year + 1))
        print("=" * 78)
        print(f"BEA county total employment - {TABLE}, {args.first_year}-"
              f"{args.last_year}, onto the CCG universe")
        print("=" * 78)
        print(f"  table      {TABLE}, {MEMBER}")
        print(f"  unit       jobs by place of work, NOT EFSY's covered employees")
        print(f"  geography  12_build_county_income.py's resolver and straddle "
              f"rules, imported")

        print("\n[0] the download")
        zip_path = fetch(args.refresh)

        print("\n[1] the line code")
        linecode = resolve_linecode(zip_path)

        print("\n[2] reading, and onto the CCG spine")
        raw = inc.read_cainc1(zip_path, span, linecode=linecode,
                              value_col=VALUE_COL, scale=1.0, member=MEMBER)
        print(f"    {len(raw):,} area-years, "
              f"{raw['geofips'].nunique():,} BEA areas")

        fips_to_ccg, _, universe = dc.load_ccg_map()
        emp = to_ccg(raw, fips_to_ccg, set(universe), years)
        print(f"    {len(emp):,} CCG-years, {emp['ccg'].nunique():,} CCGs")

        print("\n[3] the checks")
        # Check 3: the support is the estimation support, less the declared gap.
        have = set(emp["ccg"].unique())
        want_ccgs = set(int(c) for c in universe) - set(EMP_GAP_CCGS)
        missing = sorted(want_ccgs - have)
        surplus = sorted(have - set(int(c) for c in universe))
        check(not missing and not surplus, 3,
              "the support is the CCG universe less the declared gap",
              f"{len(have):,} CCGs, {len(EMP_GAP_CCGS)} declared absent"
              if not missing and not surplus
              else f"{len(missing)} missing {missing[:4]}, {len(surplus)} surplus")

        check_state_rollup(raw, emp, linecode, universe)

        # Check 5: rectangular on the years, so a flag window can never be short
        # for one county and full for another - which would make the flag's
        # growth rate mean different things in different rows.
        n_years = emp.groupby("ccg")["year"].nunique()
        rect = bool((n_years == len(span)).all())
        check(rect, 5, "every CCG has every year, so no window is short",
              f"{len(span)} years x {len(have):,} CCGs"
              if rect else f"{int((n_years != len(span)).sum())} CCGs short")

        # Check 6: positive and finite. A zero would make a growth rate infinite
        # and a negative one is not a job count.
        bad = emp[~np.isfinite(emp[VALUE_COL]) | (emp[VALUE_COL] <= 0)]
        check(bad.empty, 6, "employment is finite and strictly positive",
              "all rows" if bad.empty else f"{len(bad):,} bad rows")

        # Check 7: the two flag windows are present for every county, which is
        # the only thing 26 actually needs out of this file.
        win_ok, win_detail = True, []
        for w0, w1 in ((1969, 1974), (1993, 1997)):
            n = emp[emp["year"].isin((w0, w1))].groupby("ccg").size()
            ok = bool((n == 2).all()) and len(n) == len(have)
            win_ok &= ok
            win_detail.append(f"{w0}->{w1} {'ok' if ok else 'SHORT'}")
        check(win_ok, 7, "both flag windows are complete for every CCG",
              ", ".join(win_detail))

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            return 1

        print("\n[4] writing")
        dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.PANEL_OUT_DIR / "bea_employment_ccg_year.parquet"
        emp.to_parquet(out, index=False)
        print(f"    {len(emp):,} rows -> {out}")

        for w0, w1 in ((1969, 1974), (1993, 1997)):
            a = emp[emp["year"] == w0].set_index("ccg")[VALUE_COL]
            b = emp[emp["year"] == w1].set_index("ccg")[VALUE_COL]
            g = (b - a) / a
            print(f"    growth {w0}->{w1}   mean {g.mean():+.4f}   "
                  f"sd {g.std():.4f}   p50 {g.median():+.4f}   "
                  f"IQR {g.quantile(.75) - g.quantile(.25):.4f}")

        print("\nall checks passed")
        return 0
    finally:
        if tee is not None:
            tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
