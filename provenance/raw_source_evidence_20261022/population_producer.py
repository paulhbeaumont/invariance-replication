#!/usr/bin/env python3
"""County personal income, BEA CAINC1, mapped onto the CCG universe. Part 1.4.

The shock is contract growth over county income, so the panel needs a denominator
and nothing in this project has ever had one. The only county income anywhere on
disk is a 2003-2010 extract inside the Dupor and McCrory package, which is used
here as an overlap check and cannot be the source.

CAINC1 ships as a static zip - no API key - carrying one row per (area, LineCode)
and one column per year 1969 to 2024. LineCode 1 is personal income in thousands
of current dollars. It is deflated to CPI_BASE dollars with the same CPI table
13_build_county_panel.py deflates contracts with, because the shock is a ratio of
the two and two price indices divided by each other is not a real quantity.

THE HARD PART IS NOT THE DOWNLOAD, IT IS THAT BEA'S COUNTY SET IS NOT OURS.
BEA reports 3,149 areas, we have 3,101 consistent county groups, and the two
disagree in four different ways. Every one of them is declared below and each
declaration fails the run if it stops being true:

  1  BEA combines Virginia counties with their independent cities, which is the
     same convention county_ccg_crosswalk.csv already uses (step_1/README 1.6.3),
     so va_bea_combinations.csv resolves those codes. Our groups are sometimes
     WIDER than BEA's - BEA keeps Roanoke city 51770 out of its 51944 and reports
     it separately, and both rows land on the same CCG and are summed. That is
     correct rather than a double count, because BEA's areas partition the
     country.
  2  BEA's Alaska is time-split. The 029xx historical divisions carry 1969-1978
     and the modern boroughs carry 1979 onward, with no overlap. Both halves are
     needed and BEA_ALASKA_DIVISIONS maps the first onto the 1990 vintage.
  3  Two BEA areas straddle two of our CCGs, and they get opposite treatments
     because the data allows one and not the other. See BEA_AREA_SPLITS.
  4  Some BEA areas are outside the window entirely - the 2024 Connecticut
     planning regions, Chugach and Copper River from 2020 - and are skipped by
     the window filter rather than by name.

LINECODE 2 IS READ TOO, AND GOES TO ITS OWN FILE
CAINC1 carries population on LineCode 2 beside income on LineCode 1, on the same
areas. 23_sample_screen_grid.py's rule (b) needs a county population to rank on
and this project has never had one, so it is built here, through the same four
declarations, rather than in a second reader that would restate them. It is
written to population_ccg_year.parquet and NOT added to income_ccg_year.parquet:
that file is read by 13, 14, 19 and 20 and transitively pins step_2's beta_0, so
it is not rewritten to gain a column none of them reads. See population_pass.

Output:
    dc.PANEL_OUT_DIR/income_ccg_year.parquet   ccg, year, income_nominal, income_real
    dc.PANEL_OUT_DIR/population_ccg_year.parquet   ccg, year, population
    dc.LOG_DIR/county_income_report.csv        per-year coverage and the state check
"""

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 1969          # CAINC1 begins here, and so does any income denominator
LAST_YEAR = 2018           # was 2006 until 2026-08-11; the contract panel now
                           # reaches FY2018 through the post-seam block that
                           # 18_ingest_muratori.py builds, and EFSY employment
                           # ends there. CAINC1 itself runs to 2024.

CAINC1_MEMBER = "CAINC1__ALL_AREAS_1969_2024.csv"
INCOME_LINECODE = "1"      # personal income, thousands of current dollars
POP_LINECODE = "2"         # population, persons - read for the sample screens
                           # 23_sample_screen_grid.py imposes, and for nothing
                           # else. It goes to its OWN parquet: income_ccg_year
                           # is read by 13, 14, 19, 20 and transitively pins
                           # step_2's beta_0, and a column added to it would
                           # rewrite a file four frozen tables depend on to gain
                           # nothing they use.
POP_SCALE = 1.0            # persons, already
INCOME_SCALE = 1000.0      # thousands of dollars -> dollars

# BEA's own missing-value tokens. A county-year carrying one of these has no
# income, which is a different thing from an income of zero.
MISSING = {"", "(NA)", "(D)", "(NM)", "(L)"}

# ---------------------------------------------------------------------------
# Declaration 2: BEA's historical Alaska divisions, 1969-1978.
#
# BEA switches Alaska geography at 1979: the 029xx census divisions carry
# 1969-1978 and the modern boroughs carry 1979 onward, and no area appears on
# both sides. Every code here maps to the 1970-census division FIPS that
# county_ccg_crosswalk.csv already knows, which is the same vintage
# zark_fips_supplement.csv resolves the pre-1984 contract records onto.
#
# The map is asserted, not trusted: an entry whose target is not in the
# crosswalk, or which stops appearing in CAINC1, fails the run.
# ---------------------------------------------------------------------------
BEA_ALASKA_DIVISIONS = {
    "02901": "02010",   # Aleutian Islands
    "02903": "02030",   # Angoon
    "02904": "02040",   # Barrow / North Slope
    "02905": "02050",   # Bethel
    "02907": "02070",   # Bristol Bay Division - renamed Dillingham, keeps 02070
    "02908": "02080",   # Cordova-McCarthy
    "02910": "02100",   # Haines
    "02912": "02120",   # Kenai-Cook Inlet
    "02916": "02160",   # Kuskokwim
    "02919": "02190",   # Outer Ketchikan
    "02920": "02200",   # Prince of Wales
    "02921": "02210",   # Seward
    "02922": "02220",   # Sitka
    "02923": "02230",   # Skagway-Yakutat
    "02924": "02240",   # Southeast Fairbanks
    "02925": "02250",   # Upper Yukon
    "02926": "02260",   # Valdez-Chitina-Whittier
    "02928": "02280",   # Wrangell-Petersburg
    "02929": "02290",   # Yukon-Koyukuk
}

# Combination areas that va_bea_combinations.csv does not key. 51941's two
# members were folded into the wider 51918 group by transitive closure through
# Prince George and Petersburg, so it is recorded there only in a note; both
# members sit in one CCG. The other two are the straddles, and their membership
# has to be declared HERE rather than at the point of the rule, so that
# resolve_areas can see they span two CCGs and demand a rule instead of quietly
# reporting them as unmapped.
BEA_EXTRA_MEMBERS = {
    "51941": ("51149", "51670"),   # Prince George + Hopewell -> one CCG
    "15901": ("15009", "15005"),   # Maui + Kalawao           -> two CCGs
    "55901": ("55115", "55078"),   # Shawano incl. Menominee  -> two CCGs
}

# ---------------------------------------------------------------------------
# Declaration 3: the two BEA areas that straddle two CCGs.
#
# Everything else in CAINC1 sits inside exactly one consistent county group. Two
# do not, and the data allows a split for one of them and not the other. Both are
# stated here rather than resolved silently, because a straddle handled quietly
# is a county receiving another county's income.
# ---------------------------------------------------------------------------
#
# 55901 "Shawano (includes Menominee)" carries 1969-1988; from 1989 BEA reports
# Shawano 55115 and Menominee 55078 separately. The split is therefore taken from
# BEA's own first year of separation and applied backwards - a predetermined
# share computed from the source rather than an invented one. Menominee is a real
# unit here: 740 workers on average and $2.7m of contracts, so assigning it
# nothing for twenty of the panel's years would be worse than splitting.
BEA_BACKCAST_SPLITS = {
    "55901": {
        "members": ("55115", "55078"),
        "share_from_year": 1989,
        "why": "Shawano and Menominee are one BEA area through 1988 and two from "
               "1989. The 1989 shares are applied to 1969-1988.",
    },
}
#
# 15901 "Maui + Kalawao" is one area in every year BEA has ever published, so
# there is no year to take a share from. Kalawao is a settlement of about ninety
# people; over 1975-2006 it averages 0.8 covered workers and receives $0 of
# defense contracts in every single year, both verified in code below. The whole
# area is therefore assigned to Maui and Kalawao is declared to have no income
# denominator at all, which drops it from the estimation sample and from nothing
# else. It is named here so that the shock's county universe is 3,100 by
# declaration rather than 3,100 by accident.
BEA_WHOLE_TO = {
    "15901": "15009",   # Maui + Kalawao -> Maui
}
INCOME_GAPS = {
    15005: "Kalawao HI. BEA has never separated it from Maui, so it has no "
           "income in any year. It draws $0 of defense contracts in every year "
           "of the panel and averages 0.8 covered workers, both asserted below.",
}


# ---------------------------------------------------------------------------
# the download
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_cainc1(refresh=False):
    """Download CAINC1.zip if it is not already here, and verify it opens.

    Verified by content, not by byte count, for the reason 03_verify_dcads.py
    gives: a served size that disagrees with a published one is common and is not
    evidence of a damaged file. What matters is that the zip's central directory
    parses and every member passes CRC.
    """
    dest_dir = dc.BEA_SOURCE_DIR / "CAINC1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "CAINC1.zip"

    if refresh or not dest.exists():
        print(f"    downloading {dc.CAINC1_URL}")
        tmp = dest.with_suffix(".zip.tmp")
        with urllib.request.urlopen(dc.CAINC1_URL, timeout=120) as r, open(tmp, "wb") as out:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
        tmp.replace(dest)
    else:
        print(f"    using the cached copy at {dest}")

    with zipfile.ZipFile(dest) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise AssertionError(f"CAINC1.zip: member {bad} fails CRC")
        if CAINC1_MEMBER not in zf.namelist():
            raise AssertionError(
                f"CAINC1.zip does not contain {CAINC1_MEMBER}; BEA changed the "
                f"packaging and the reader has to be revisited")

    digest = sha256_of(dest)
    receipt = dest_dir / "download_receipt.csv"
    with open(receipt, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "url", "bytes", "sha256"])
        w.writerow([dest.name, dc.CAINC1_URL, dest.stat().st_size, digest])
    print(f"    {dest.stat().st_size:,} bytes   sha256 {digest[:16]}...   "
          f"every member passes CRC")
    return dest


def read_cainc1(zip_path, years, linecode=INCOME_LINECODE,
                value_col="income_nominal", scale=INCOME_SCALE,
                member=CAINC1_MEMBER):
    """Rows of one LineCode, long format: geofips, geoname, year, <value_col>.

    Read out of the zip in place. LineCode 1's values are thousands of current
    dollars; the thousands are multiplied out here so that nothing downstream
    has to remember the unit, which is the mistake FY1989 of the contract panel
    exists to warn about. LineCode 2 is persons and scales by one - the argument
    exists so the unit is stated at the call site rather than assumed.

    `member` is the CSV inside the zip, defaulting to CAINC1's. It exists for
    25_ingest_bea_employment.py, which reads the SAME row layout out of a
    different BEA table (CAEMP25S) and would otherwise have to carry a second
    copy of this loop - and a second copy is how the two would end up disagreeing
    about MISSING, about the quoted GeoFIPS, or about the unit.

    The defaults are LineCode 1's, so the income path calls this exactly as it
    always did and cannot move.
    """
    wanted = [str(y) for y in years]
    recs = []
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        rd = csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1", newline=""))
        for row in rd:
            if (row.get("LineCode") or "").strip() != linecode:
                continue
            geo = (row.get("GeoFIPS") or "").strip().strip('"').strip()
            if len(geo) != 5 or not geo.isdigit():
                continue
            name = (row.get("GeoName") or "").strip()
            for y in wanted:
                raw = (row.get(y) or "").strip()
                if raw in MISSING:
                    continue
                recs.append((geo, name, int(y), float(raw.replace(",", "")) * scale))
    return pd.DataFrame(recs, columns=["geofips", "geoname", "year", value_col])


# ---------------------------------------------------------------------------
# BEA area -> CCG
# ---------------------------------------------------------------------------

def resolve_areas(geofips, fips_to_ccg, failures):
    """geofips -> ccg_id for every BEA area that is not a straddle or a state.

    Returns (mapping, straddles, unmapped). A BEA area resolves through, in
    order: its own FIPS, the Alaska division map, the Virginia combination
    members, and BEA_EXTRA_MEMBERS. An area whose members land in more than one
    CCG is a straddle and is returned rather than mapped, so the caller has to
    have declared a rule for it.
    """
    va = pd.read_csv(dc.VA_COMBINATIONS, dtype=str)
    va_members = va.groupby("bea_code")["county_fips5"].apply(tuple).to_dict()

    mapping, straddles, unmapped = {}, {}, []
    for f in sorted(set(geofips)):
        if f.endswith("000"):
            continue                                  # a state or the nation
        if f in BEA_ALASKA_DIVISIONS:
            members = (BEA_ALASKA_DIVISIONS[f],)
        elif f in va_members:
            members = va_members[f]
        elif f in BEA_EXTRA_MEMBERS:
            members = BEA_EXTRA_MEMBERS[f]
        else:
            members = (f,)
        ccgs = {fips_to_ccg[m] for m in members if m in fips_to_ccg}
        if not ccgs:
            unmapped.append(f)
        elif len(ccgs) > 1:
            straddles[f] = sorted(ccgs)
        else:
            mapping[f] = ccgs.pop()

    # Every declared Alaska division must still be in the file, and its target
    # must still be in the crosswalk. A declaration that stops matching is worse
    # than no declaration, because nothing fails.
    present = set(geofips)
    for code, target in BEA_ALASKA_DIVISIONS.items():
        if code not in present:
            failures.append(("alaska declaration",
                             f"{code} is declared but no longer in CAINC1"))
        if target not in fips_to_ccg:
            failures.append(("alaska declaration",
                             f"{code} -> {target}, which is not in the crosswalk"))
    return mapping, straddles, unmapped


def apply_straddle_rules(inc, straddles, fips_to_ccg, failures, verbose=True,
                         value_col="income_nominal"):
    """Turn the declared straddles into CCG rows, and fail on an undeclared one.

    Two rules, and they are different because the data is:

      BEA_BACKCAST_SPLITS  the area splits in a later year, so the later year's
                           shares are applied to the earlier ones
      BEA_WHOLE_TO         the area never splits, so it goes whole to one member
                           and the other is declared an income gap

    `value_col` is what makes this reusable for LineCode 2. Note that the
    backcast shares are then POPULATION shares of the split year rather than
    income shares - which is the right choice for a population series and is
    also why the shares are recomputed here rather than passed in from the
    income pass.
    """
    declared = set(BEA_BACKCAST_SPLITS) | set(BEA_WHOLE_TO)
    undeclared = sorted(set(straddles) - declared)
    if undeclared:
        for f in undeclared:
            failures.append(("undeclared straddle",
                             f"BEA area {f} spans CCGs {straddles[f]}"))
        return pd.DataFrame(columns=["ccg", "year", value_col])

    out = []
    for f, spec in BEA_BACKCAST_SPLITS.items():
        if f not in straddles:
            failures.append(("split declaration", f"{f} is declared but does not straddle"))
            continue
        block = inc[inc["geofips"] == f]
        a, b = spec["members"]
        base_year = spec["share_from_year"]
        base = inc[inc["geofips"].isin((a, b)) & (inc["year"] == base_year)]
        if set(base["geofips"]) != {a, b}:
            failures.append(("split declaration",
                             f"{f}: BEA has no separate {a}/{b} rows in {base_year}"))
            continue
        tot = base[value_col].sum()
        share = {g: float(v) / tot for g, v in
                 zip(base["geofips"], base[value_col])}
        if verbose:
            print(f"    {f} split on its {base_year} shares: "
                  + ", ".join(f"{g} {share[g]:.4%}" for g in (a, b))
                  + f"   applied to {block['year'].min()}-{block['year'].max()}")
        for g in (a, b):
            out.append(pd.DataFrame({
                "ccg": fips_to_ccg[g],
                "year": block["year"].to_numpy(),
                value_col: block[value_col].to_numpy() * share[g],
            }))

    for f, target in BEA_WHOLE_TO.items():
        if f not in straddles:
            failures.append(("whole-to declaration", f"{f} is declared but does not straddle"))
            continue
        block = inc[inc["geofips"] == f]
        other = [c for c in straddles[f] if c != fips_to_ccg[target]]
        if verbose:
            print(f"    {f} assigned whole to {target} (CCG {fips_to_ccg[target]}); "
                  f"CCG {other} left with no income, as declared")
        out.append(pd.DataFrame({
            "ccg": fips_to_ccg[target],
            "year": block["year"].to_numpy(),
            value_col: block[value_col].to_numpy(),
        }))

    return pd.concat(out, ignore_index=True) if out else \
        pd.DataFrame(columns=["ccg", "year", value_col])


# ---------------------------------------------------------------------------
# LineCode 2, population. Part 1.5's sample screens, and nothing else.
# ---------------------------------------------------------------------------

def population_pass(zip_path, years, fips_to_ccg, universe, failures):
    """County population on the CCG spine, through the SAME reconciliations.

    23_sample_screen_grid.py's rule (b) ranks counties by population in a
    pre-sample year and drops the smallest 500 and 1,000. That needs a county
    population, and this project has never had one.

    It is built here rather than in its own script for one reason: BEA's area
    set is not ours, and 12 is where the four ways they disagree are already
    declared - the Virginia combinations, the Alaska time-split, the two
    straddles, the out-of-window areas. A second reader would either restate
    those declarations, and drift, or skip them, and be wrong in Virginia and
    Alaska. So the population pass calls resolve_areas and apply_straddle_rules
    verbatim and differs from the income pass only in the LineCode.

    It writes its own parquet. income_ccg_year.parquet is read by 13, 14, 19 and
    20 and transitively pins step_2's beta_0; adding a column to it would
    rewrite a frozen input to gain nothing any of them reads.
    """
    print("\n[8] LineCode 2, population, through the same reconciliations")
    pop = read_cainc1(zip_path, years, linecode=POP_LINECODE,
                      value_col="population", scale=POP_SCALE)
    print(f"    {len(pop):,} area-years on LineCode {POP_LINECODE}, "
          f"{pop['geofips'].nunique():,} distinct areas")

    mapping, straddles, unmapped = resolve_areas(pop["geofips"], fips_to_ccg,
                                                 failures)
    split_rows = apply_straddle_rules(pop, straddles, fips_to_ccg, failures,
                                      verbose=False, value_col="population")
    direct = pop[pop["geofips"].isin(mapping)].copy()
    direct["ccg"] = direct["geofips"].map(mapping)
    panel = pd.concat([direct[["ccg", "year", "population"]], split_rows],
                      ignore_index=True)
    panel = panel.groupby(["ccg", "year"], as_index=False)["population"].sum()

    # The same coverage rule as income, and it has to give the same answer: BEA
    # reports Kalawao inside Maui for population exactly as it does for income,
    # so the declared gap set is identical. A population gap somewhere income
    # has none would mean the two LineCodes disagree about BEA's own geography,
    # which is worth failing on rather than filling.
    have = panel.groupby("ccg")["year"].agg(set)
    want = set(years)
    gaps = {int(c): sorted(want - have.get(c, set())) for c in universe
            if want - have.get(c, set())}
    same = set(gaps) == set(INCOME_GAPS)
    print(f"    {len(universe) - len(gaps):,} of {len(universe):,} CCGs carry "
          f"population in every year; gaps {sorted(gaps)}")
    if not same:
        failures.append(("population gap",
                         f"population gaps {sorted(gaps)} differ from the "
                         f"declared income gaps {sorted(INCOME_GAPS)}"))

    # BEA's own state population rows, the same rollup check the income pass
    # runs. Population is an integer count, so ours must equal theirs exactly
    # rather than to a rounding tolerance - except where a backcast split has
    # divided an area on non-integer shares, which is why the comparison is
    # relative and the tolerance is 1e-9 as it is for income.
    state_of = {}
    for f, c in mapping.items():
        state_of.setdefault(c, f[:2])
    for spec in BEA_BACKCAST_SPLITS.values():
        for g in spec["members"]:
            state_of.setdefault(fips_to_ccg[g], g[:2])
    for target in BEA_WHOLE_TO.values():
        state_of.setdefault(fips_to_ccg[target], target[:2])
    ours = (panel.assign(state=panel["ccg"].map(state_of))
            .groupby(["state", "year"])["population"].sum())
    theirs_raw = pop[pop["geofips"].str.endswith("000")].copy()
    theirs_raw["state"] = theirs_raw["geofips"].str[:2]
    theirs = theirs_raw[theirs_raw["state"] != "00"].set_index(
        ["state", "year"])["population"]
    j = pd.concat([ours.rename("ours"), theirs.rename("theirs")], axis=1).dropna()
    rel = ((j["ours"] - j["theirs"]) / j["theirs"]).abs()
    print(f"    {len(j):,} state-years compared against BEA's own totals   "
          f"max |relative gap| {rel.max():.2e}")
    if float(rel.max()) > 1e-9:
        bad = rel.groupby(level=0).max().sort_values(ascending=False).head(3)
        failures.append(("population state total",
                         f"worst {dict(bad.round(6))}"))

    panel = panel.sort_values(["ccg", "year"]).reset_index(drop=True)
    panel["ccg"] = panel["ccg"].astype(np.int32)
    panel["year"] = panel["year"].astype(np.int16)
    out = dc.PANEL_OUT_DIR / "population_ccg_year.parquet"
    panel.to_parquet(out, index=False)
    first = panel[panel["year"] == years[0]]
    print(f"    {len(panel):,} CCG-years -> {out}")
    print(f"    {years[0]}: {len(first):,} CCGs   national "
          f"{first['population'].sum() / 1e6:.1f}m   median "
          f"{first['population'].median():,.0f}   smallest "
          f"{first['population'].min():,.0f}")
    return panel


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download CAINC1.zip even if it is already cached")
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--last-year", type=int, default=LAST_YEAR)
    args = ap.parse_args()

    years = list(range(args.first_year, args.last_year + 1))
    failures = []
    print(f"\ncounty personal income, BEA CAINC1   {years[0]}-{years[-1]}")

    print("\n[0] the source")
    zip_path = fetch_cainc1(refresh=args.refresh)
    inc = read_cainc1(zip_path, years)
    print(f"    {len(inc):,} area-years on LineCode {INCOME_LINECODE}, "
          f"{inc['geofips'].nunique():,} distinct areas")

    fips_to_ccg, _, universe = dc.load_ccg_map()

    print("\n[1] BEA areas onto the CCG universe")
    mapping, straddles, unmapped = resolve_areas(inc["geofips"], fips_to_ccg, failures)

    # An unmapped area is only acceptable if it carries no data inside the
    # window - the 2024 Connecticut planning regions, Chugach and Copper River
    # from 2020. Anything else is a hole.
    in_window = inc.groupby("geofips")["year"].size()
    live_unmapped = [f for f in unmapped if in_window.get(f, 0) > 0]
    dead_unmapped = [f for f in unmapped if in_window.get(f, 0) == 0]
    print(f"    {len(mapping):,} areas map to one CCG   "
          f"{len(straddles)} straddle two   "
          f"{len(dead_unmapped)} outside the window")
    if live_unmapped:
        names = inc[inc["geofips"].isin(live_unmapped)].groupby("geofips")["geoname"].first()
        for f in live_unmapped:
            failures.append(("unmapped BEA area",
                             f"{f} {names.get(f, '')} carries "
                             f"{in_window[f]} year(s) in the window"))
    for f, ccgs in straddles.items():
        print(f"    straddle: {f} "
              f"{inc[inc['geofips'] == f]['geoname'].iloc[0]} -> CCGs {ccgs}")

    print("\n[2] the declared straddle rules")
    split_rows = apply_straddle_rules(inc, straddles, fips_to_ccg, failures)

    direct = inc[inc["geofips"].isin(mapping)].copy()
    direct["ccg"] = direct["geofips"].map(mapping)
    panel = pd.concat(
        [direct[["ccg", "year", "income_nominal"]], split_rows], ignore_index=True)
    panel = panel.groupby(["ccg", "year"], as_index=False)["income_nominal"].sum()

    print("\n[3] coverage: every CCG, every year")
    gaps = []
    have = panel.groupby("ccg")["year"].agg(set)
    want = set(years)
    for c in universe:
        got = have.get(c, set())
        missing = sorted(want - got)
        if missing:
            gaps.append((int(c), missing))
    declared_gap_ccgs = set(INCOME_GAPS)
    hard = [(c, m) for c, m in gaps if c not in declared_gap_ccgs or len(m) != len(years)]
    for c, m in gaps:
        tag = "declared" if c in declared_gap_ccgs else "UNDECLARED"
        print(f"    CCG {c}: no income in {len(m)} of {len(years)} years   [{tag}]")
    for c, m in hard:
        failures.append(("income gap",
                         f"CCG {c} is missing {len(m)} year(s) and is not a "
                         f"declared all-years gap"))
    for c in declared_gap_ccgs:
        if c not in dict(gaps):
            failures.append(("income gap declaration",
                             f"CCG {c} is declared to have no income and now has some"))
    covered = len(universe) - len(declared_gap_ccgs)
    print(f"    {covered:,} of {len(universe):,} CCGs carry income in every year; "
          f"{len(declared_gap_ccgs)} declared gap(s)")
    for c, why in INCOME_GAPS.items():
        print(f"      {c}: {why}")

    print("\n[4] the declared gap costs nothing")
    # A gap is only acceptable if the unit it removes never receives a shock.
    # The window follows the panel rather than a literal 2007: with the contract
    # panel extended to FY2018 a hardcoded end year would stop testing the years
    # the gap now has to survive, and a guard that stops guarding is worse than
    # no guard. 1975 is where the contract panel starts, not where CAINC1 does.
    contracts = dc.build_contract_panel(universe,
                                        list(range(1975, args.last_year + 1)),
                                        floor=0, verbose=False)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    for c in INCOME_GAPS:
        d = contracts.loc[contracts["ccg"] == c, "dollars_real"]
        e = emp.loc[emp["ccg"] == c, "emp"]
        print(f"    CCG {c}: ${d.abs().sum():,.0f} of contracts over "
              f"{int((d != 0).sum())} nonzero year(s), mean employment "
              f"{e.mean():.1f}")
        if float(d.abs().sum()) != 0.0:
            failures.append(("income gap",
                             f"CCG {c} has no income denominator but "
                             f"${d.abs().sum():,.0f} of contracts"))

    print("\n[5] against BEA's own state totals")
    # The rollup has to reproduce BEA's SS000 rows. This is the check that a
    # straddle rule, a Virginia combination or an Alaska division was dropped or
    # counted twice: BEA's areas partition each state, so anything mishandled
    # shows up as a state that no longer adds up.
    state_of = {}
    for f, c in mapping.items():
        state_of.setdefault(c, f[:2])
    for spec in BEA_BACKCAST_SPLITS.values():
        for g in spec["members"]:
            state_of.setdefault(fips_to_ccg[g], g[:2])
    for target in BEA_WHOLE_TO.values():
        state_of.setdefault(fips_to_ccg[target], target[:2])
    panel_state = panel.assign(state=panel["ccg"].map(state_of))
    ours = panel_state.groupby(["state", "year"])["income_nominal"].sum()

    theirs_raw = inc[inc["geofips"].str.endswith("000")].copy()
    theirs_raw["state"] = theirs_raw["geofips"].str[:2]
    theirs = theirs_raw[theirs_raw["state"] != "00"].set_index(
        ["state", "year"])["income_nominal"]

    joined = pd.concat([ours.rename("ours"), theirs.rename("theirs")], axis=1).dropna()
    # No state is exempt. Both straddle rules conserve the area's total - the
    # split shares sum to one and the whole-to rule moves the whole area - so
    # every state must reproduce BEA's own figure to its $1,000-per-area
    # rounding. Kalawao in particular is NOT missing from Hawaii: its income is
    # inside the Maui + Kalawao area and travels with it. What Kalawao lacks is a
    # denominator of its own, which is a different thing and is check 3's.
    joined["gap"] = joined["ours"] - joined["theirs"]
    joined["rel"] = joined["gap"] / joined["theirs"]
    worst = joined["rel"].abs().groupby(level=0).max().sort_values(ascending=False)
    tol = 1e-9
    off = worst[worst > tol]
    print(f"    {len(joined):,} state-years compared   "
          f"max |relative gap| {joined['rel'].abs().max():.2e}   "
          f"{len(off)} state(s) off by more than {tol:g}")
    for st, v in off.items():
        n = int((joined.loc[st, "rel"].abs() > tol).sum())
        print(f"    state {st}: worst {v:.3e} in {n} of {len(years)} years")
        failures.append(("state total",
                         f"state {st} is off by {v:.3e} in {n} year(s)"))

    print("\n[6] the Dupor and McCrory overlap, 2003-2006 (reported, not asserted)")
    # Their extract is a different BEA vintage on a different county set, so this
    # is evidence rather than a criterion: a median relative gap near zero says
    # the LineCode, the unit and the county mapping are all right. A tail is
    # expected, because their file is a modern-vintage county list and ours is
    # 1990 groups.
    dm_path = (
        dc.DROPBOX_PROJECT_ROOT
        / "2_illustration_aggregation"
        / "DuporMcCrory"
        / "data"
        / "source"
        / "author_package"
        / "data"
        / "county_personal_income_larger.parquet"
    )
    if not dm_path.exists():
        print("    not present in Dropbox; skipped")
    else:
        dm = pd.read_parquet(dm_path)
        dm["ccg"] = dm["fips"].astype(str).str.zfill(5).map(fips_to_ccg)
        dm = dm.dropna(subset=["ccg"])
        for y in (2003, 2004, 2005, 2006):
            col = f"pi_{y}"
            if col not in dm.columns:
                continue
            v = pd.to_numeric(dm[col], errors="coerce")
            theirs_y = dm.assign(v=v).dropna(subset=["v"]).groupby("ccg")["v"].sum() * 1000.0
            ours_y = panel[panel["year"] == y].set_index("ccg")["income_nominal"]
            j = pd.concat([ours_y.rename("a"), theirs_y.rename("b")], axis=1).dropna()
            rel = ((j["a"] - j["b"]) / j["b"]).abs()
            print(f"    {y}: {len(j):,} CCGs   median |rel| {rel.median():.2e}   "
                  f"p99 {rel.quantile(0.99):.2e}   national gap "
                  f"{(j['a'].sum() / j['b'].sum() - 1):+.4%}")

    print("\n[7] writing")
    panel = panel.sort_values(["ccg", "year"]).reset_index(drop=True)
    panel["income_real"] = panel["income_nominal"] * panel["year"].map(dc.deflate)
    panel["ccg"] = panel["ccg"].astype(np.int32)
    panel["year"] = panel["year"].astype(np.int16)
    dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = dc.PANEL_OUT_DIR / "income_ccg_year.parquet"
    panel.to_parquet(out, index=False)
    print(f"    {len(panel):,} CCG-years -> {out}")
    print(f"    national income {panel[panel['year'] == args.last_year]['income_real'].sum() / 1e12:.3f}tn "
          f"({dc.CPI_BASE}$) in {args.last_year}")

    rep = panel.groupby("year").agg(
        n_ccg=("ccg", "size"),
        income_real_bn=("income_real", lambda s: s.sum() / 1e9)).reset_index()
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(dc.LOG_DIR / "county_income_report.csv", index=False)

    population_pass(zip_path, years, fips_to_ccg, universe, failures)

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
