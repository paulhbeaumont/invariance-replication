#!/usr/bin/env python3
"""Build the time-consistent county universe Part 1.4 needs, and verify it.

county_fips5 is CONTEMPORANEOUS: each fiscal year carries whatever county code
the Pentagon used that year. Correct as a parse, unusable as a panel key. This
script produces the one thing that makes a county long panel possible - a
many-to-one, TIME-INVARIANT map from every county code the data ever uses onto a
fixed set of units, the "consistent county groups" (CCG).

Why time-invariant rather than date-triggered. The source keeps using retired
codes long after the successor exists, often in the same year: $1.67bn is booked
to Dade 12025 after Miami-Dade 12086 appears, $288m to Aleutian Islands 02010
after the 1987 split, and Yellowstone National Park 30113 carries rows in FY2003,
six years after it was abolished. A map keyed on the year of the change would
split those places across two units. One map applied to all 42 years cannot.

Why merges rather than weighted splits. Every change is expressed as a merge:
predecessor and successor go into the same group, and so do parent and daughter
where a county was created inside the panel. That is additive and needs no
weights - which matters because the source cannot tell us what Cibola County's
dollars were in 1979, since Cibola did not exist. The cost is losing about thirty
distinctions; the alternative is inventing dollars.

The 1990 county vintage is the labelling. The commuting-zone bridge already in
the repo is a 1990-vintage file, 3,141 counties to 741 zones, so labelling the
CCG set on that vintage makes the CZ rung fall out rather than needing a new
reference file.

Inputs, all under code/reference/:
  county_changes_census.csv   transcribed from the Census "Substantial Changes to
                              Counties and County Equivalent Entities" pages for
                              the 1970s, 1980s, 1990s and 2000s
  va_bea_combinations.csv     the BEA Virginia county-city combination areas,
                              each row flagged for whether a Census annexation
                              edge also backs it
  dcads_common.SUPERSEDED     the thirteen codes that stop at the FY1983/84 seam
  PROJECT_EDGES below         everything hand-authored, each with a declared
                              expected action count that fails the run if wrong

Output: code/reference/county_ccg_crosswalk.csv, and a report next to the parquet.

The edge rules, stated once because everything downstream depends on them:

  new / deleted / code_change  ALWAYS form a group, unless the source explicitly
                               says the population moved was zero. A unit that
                               appears or vanishes is not survivable in a fixed
                               universe. A boundary that moves no people moves no
                               employment and essentially no establishments, so a
                               zero-population parent edge is not a merge.
  boundary                     NEVER forms a group by itself. These are reported
                               with their populations so the reader can judge, and
                               check 7 sweeps the data for level breaks so an
                               unlisted one still surfaces. The exception is
                               Virginia, where the BEA convention already merges
                               every annexing city with its county - and the
                               Census record turns out to be entirely absorbed by
                               those pairs, which is the argument for adopting it.

Eleven checks. 1 to 6 and 9 to 11 can fail; 7 and 8 are reports, because any
threshold on them would be a number invented in order to be passed.
"""

import argparse
import collections
import csv
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

REPORT = dc.OUT_DIR / "county_universe_report.csv"

# The panel the employment data can actually support. EFSY's NAICS panel starts
# in 1975, so FY1966-74 has no employment to be joined to. The contracts stop at
# the end of FY2006, so PANEL_LAST_DATE is where the CONTRACT rung ends and it is
# what in_window means in county_changes_census.csv.
PANEL_FIRST_YEAR = 1975
PANEL_LAST_DATE = "2006-09-30"

# The employment panel outruns the contract panel by twelve years, so "the window"
# is now two windows and county_changes_census.csv carries a flag for each. Group
# formation uses the LONGER one: one CCG set has to serve every rung of both
# ladders, because the tests require an identical set of baseline units at every
# level (INVARIANCE_TEST_PROTOCOL.md A.1.6). The cost is losing a few distinctions
# in the contract years - Bedford city from Bedford County, and the Alaska
# predecessors from their successors - which is the same trade this file already
# makes everywhere else: merges rather than weighted splits, because the
# alternative is inventing dollars.
#
# The York / Newport News exchange of July 2007 is safe from this. It is a
# `boundary` row, and boundary rows are reported and never merged, so widening the
# window cannot fold Newport News - at $50.9bn the largest defence county in the
# country - into York.
EMP_LAST_DATE = "2018-12-31"
EMP_LAST_YEAR = 2018

# Codes whose dollars must never enter a county rung even though they parse as
# well-formed. Kept as a group of their own so the crosswalk stays total.
UNPLACEABLE = "unplaceable"

# ---------------------------------------------------------------------------
# Hand-authored edges. Everything here is a judgement, so everything here
# declares how many contract actions it expects to touch and fails the run on any
# other number. That is the guard SUPERSEDED never had: an override that quietly
# matched nothing, or a hundred rows after a parse change, is worse than none.
#
# (from, to, kind, expected_actions, source, note)
PROJECT_EDGES = [
    ("08014", "08013", "plurality_parent", 16, "project",
     "Broomfield County CO was created 2001-11-15 from parts of Adams 08001 "
     "(15,870 people), Boulder 08013 (21,512), Jefferson 08059 (1,726) and Weld "
     "08123 (69). Merging all five would destroy four major Front Range counties "
     "carrying $7.1bn of contracts in order to absorb $0.14m. Broomfield is "
     "assigned to Boulder alone, its plurality parent at 55% of the transferred "
     "population. THIS IS THE ONLY GROUP IN THE FILE THAT IS NOT A UNION OF WHOLE "
     "1990-vintage counties, and the only place the crosswalk knowingly puts "
     "money in a unit that did not contain it. Run with --broomfield-merge-all "
     "for the four-parent alternative."),
    ("02030", "02231", "merge", 0, "project",
     "Angoon Division AK. The Census 1970s list records it as deleted in the "
     "Alaska reorganisation and names no successor; Angoon became part of the "
     "Skagway-Yakutat-Angoon Census Area 02231. It never appears in the contract "
     "data - the expected count of 0 is deliberate, and says so - but it is in "
     "EFSY from 1975, so check 6 needs it."),
    ("48210", "48201", "typo", 11, "project",
     "48210 is not a Texas county FIPS; Harris County is 48201 and these rows "
     "carry the county name Harris. A source digit error that survives "
     "07_add_county_fips.py's well-formedness test, which accepts any 5-digit "
     "numeric county whose last three digits are not 999."),
    ("36173", UNPLACEABLE, "unplaceable", 1, "project",
     "36173 is not a New York county FIPS; New York runs 36001 to 36123. The one "
     "row carries no county name, so nothing identifies which county was meant "
     "and no crosswalk can recover it. $99,000 in FY1991."),
]

# ---------------------------------------------------------------------------
# The DC statewide-residual recode moved to dcads_common when
# 11_build_employment_cells.py needed the same declaration at the cell level.
# One copy, two consumers; see dcads_common.EFSY_RESIDUAL_AS_COUNTY for the
# reasoning and the discovery rule.
EFSY_RESIDUAL_AS_COUNTY = dc.EFSY_RESIDUAL_AS_COUNTY

# CCG-years that carry contract dollars and have no employment in EFSY for that
# year. A county absent from CBP in a year had no covered establishment in it,
# which is an economic zero rather than a defect - but it is a zero the panel has
# to know it is carrying, so each is declared with its dollars and check 6 fails
# on any that is not. After the DC recode these are the only two in 1975-2006.
EMPLOYMENT_GAPS = {
    ("02188", 1987): "Northwest Arctic Borough AK, $0.68m of contracts. EFSY "
                     "carries 02188 in 30 of the 32 panel years and not 1987.",
    ("48269", 2006): "King County TX, $19,899 of contracts. Population about "
                     "350; EFSY carries 48269 in 28 of the 32 panel years.",
}

# Census edges that are deliberately not honoured, each with the reason. Listed
# by edge_id so that a change to the transcription cannot silently drop the
# suppression.
EDGE_SUPPRESS = {
    "2000_new_08014b": "Broomfield's Adams County parent edge. See PROJECT_EDGES.",
    "2000_new_08014c": "Broomfield's Jefferson County parent edge. See PROJECT_EDGES.",
    "2000_new_08014d": "Broomfield's Weld County parent edge. See PROJECT_EDGES.",
    "2000_new_08014": "Broomfield's Boulder parent edge, replaced by the "
                      "explicit plurality_parent edge in PROJECT_EDGES so that "
                      "the judgement carries an expected count.",
}


class UnionFind:
    """Smallest structure that can express 'these codes are one place'."""

    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self):
        out = collections.defaultdict(set)
        for x in self.parent:
            out[self.find(x)].add(x)
        return out


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_domestic(fips5):
    return len(fips5) == 5 and fips5[:2] in dc.STATE_FIPS


def load_cz_base():
    """The 1990-vintage county set and its commuting zones, 3,141 to 741."""
    if dc.CZ_CROSSWALK is None:
        raise SystemExit("no commuting-zone crosswalk found; check 5 cannot run")
    if dc.CZ_CROSSWALK.suffix == ".parquet":
        tab = pq.read_table(dc.CZ_CROSSWALK).to_pydict()
        cty, cz = tab["cty_fips"], tab["czone"]
    else:
        import pandas as pd
        frame = pd.read_stata(dc.CZ_CROSSWALK)
        cty = [int(v) for v in frame["cty_fips"]]
        cz = [int(v) for v in frame["czone"]]
    return {str(c).zfill(5): int(z) for c, z in zip(cty, cz)}


def load_contract_panel():
    """One streaming pass over all 42 files.

    Returns the county-year cells, the per-year action-size histogram the
    threshold report needs, and the modal county name per code.
    """
    cells = collections.defaultdict(lambda: [0.0, 0])       # (fips, fy) -> [$, n]
    names = collections.defaultdict(collections.Counter)
    thresh = collections.defaultdict(lambda: collections.Counter())
    files = sorted(dc.OUT_DIR.glob("fy*.parquet"),
                   key=lambda p: (dc.zark_year(p.stem[2:]), p.stem))
    if not files:
        raise SystemExit(f"no parsed parquet under {dc.OUT_DIR}")
    for path in files:
        pf = pq.ParquetFile(path)
        cols = ["fiscal_year", "geo_class", "county_fips5", "county_name",
                "dollars", "dollars_unit"]
        for batch in pf.iter_batches(batch_size=200_000, columns=cols):
            d = batch.to_pydict()
            for i, klass in enumerate(d["geo_class"]):
                if klass != "county":
                    continue
                fips = d["county_fips5"][i]
                fy = d["fiscal_year"][i]
                raw = d["dollars"][i]
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = 0.0
                if d["dollars_unit"][i] == "thousands":
                    val *= 1000.0
                cell = cells[(fips, fy)]
                cell[0] += val
                cell[1] += 1
                nm = d["county_name"][i]
                if nm:
                    names[fips][nm] += 1
                size = abs(val)
                bucket = ("lt10k" if size < 10_000 else
                          "10to25k" if size < 25_000 else "ge25k")
                thresh[fy][bucket] += 1
                thresh[fy]["dol_" + bucket] += val
                if size >= 25_000:
                    thresh[fy][("cty25k", fips)] = 1
                thresh[fy][("cty", fips)] = 1
    return cells, names, thresh


def load_employment():
    """EFSY county employment over the panel window, residual recoded.

    Returns (emp, dropped, recoded, problems):
      emp       {(county_fips5, year): employment}
      dropped   employee-years left in the statewide residual and not attributed
      recoded   one row per declared state: (state, code, years, employee_years)
      problems  guard violations, which check 6 turns into failures
    """
    d = pq.read_table(dc.EFSY_PANEL,
                      columns=["fipstate", "fipscty", "year", "emp"]).to_pydict()
    emp = collections.defaultdict(float)            # (code, year) -> employment
    residual = collections.defaultdict(float)       # (state, year) -> employment
    counties = collections.defaultdict(set)         # (state, year) -> {cty}
    for st, cty, yr, e in zip(d["fipstate"], d["fipscty"], d["year"], d["emp"]):
        if not (PANEL_FIRST_YEAR <= yr <= EMP_LAST_YEAR):
            continue
        state = f"{int(st):02d}"
        if int(cty) == 999:
            residual[(state, yr)] += e or 0.0
            continue
        emp[(f"{state}{int(cty):03d}", yr)] += e or 0.0
        counties[(state, yr)].add(int(cty))

    # The sweep: which state-years have their entire employment in the residual.
    found = collections.defaultdict(set)
    for state, yr in residual:
        if not counties.get((state, yr)):
            found[state].add(yr)

    problems, plan = [], []
    for state in sorted(set(found) | set(EFSY_RESIDUAL_AS_COUNTY)):
        got = found.get(state, set())
        if state not in EFSY_RESIDUAL_AS_COUNTY:
            problems.append(
                f"state {state} files its whole employment under fipscty=999 in "
                f"{len(got)} years ({min(got)}-{max(got)}) and is not declared "
                f"in EFSY_RESIDUAL_AS_COUNTY")
            continue
        code, lo, hi, _why = EFSY_RESIDUAL_AS_COUNTY[state]
        want = set(range(lo, hi + 1))
        if got != want:
            problems.append(
                f"state {state} declared as a whole-state residual for {lo}-{hi} "
                f"({len(want)} years) but the sweep finds {len(got)}: "
                f"missing {sorted(want - got)}, extra {sorted(got - want)}")
        plan.append((state, code, sorted(got & want)))

    # Attribute only after every guard has been evaluated, so a run that is going
    # to fail does not also move employment on the strength of the declaration
    # that failed.
    recoded = []
    if not problems:
        for state, code, years in plan:
            moved = 0.0
            for yr in years:
                v = residual.pop((state, yr))
                emp[(code, yr)] += v
                moved += v
            recoded.append((state, code, years, moved))
    return dict(emp), sum(residual.values()), recoded, problems


def build_groups(changes, va_rows, merge_all_broomfield, census_only_va):
    """Apply the edge rules and return the union-find plus an edge log."""
    uf = UnionFind()
    log = []

    def edge(a, b, kind, source, note):
        if b == UNPLACEABLE or a == UNPLACEABLE:
            log.append((a, b, kind, source, note))
            return
        if not (is_domestic(a) and is_domestic(b)):
            return
        uf.union(a, b)
        log.append((a, b, kind, source, note))

    # 1. The Census change list.
    for row in changes:
        eid = row["edge_id"]
        if eid in EDGE_SUPPRESS and not merge_all_broomfield:
            continue
        if row["in_emp_window"] != "1":
            continue                                   # effective after 2018
        kind, a, b = row["kind"], row["fips_from"], row["fips_to"]
        if kind == "boundary":
            continue                                   # reported, never merged
        if not a or not b:
            continue                                   # Census names no counterpart
        if row["pop_moved"] == "0":
            continue                                   # moved no people
        edge(a, b, kind, "census:" + eid, row["description"][:200])

    # 2. The thirteen codes that stop at the FY1983/84 seam.
    for code, prose in dc.SUPERSEDED.items():
        edge(code, dc.SUPERSEDED_TARGET[code], "merge", "superseded", prose)

    # 3. Virginia, by the BEA combination convention. Every annexing city in the
    #    Census record 1970-1995 turns out to be a BEA combination-area city, so
    #    the convention absorbs the whole documented Virginia annexation history;
    #    that is the argument for it. The BEA-only rows go beyond what Census
    #    documents and --census-only-va drops them.
    by_group = collections.defaultdict(list)
    for row in va_rows:
        if census_only_va and row["census_backed"] != "1":
            continue
        by_group[row["bea_code"]].append(row)
    for code, members in by_group.items():
        fips = [m["county_fips5"] for m in members]
        name = members[0]["bea_name"]
        for other in fips[1:]:
            edge(fips[0], other, "merge", "bea:" + code, name)

    # 4. The hand-authored edges.
    for a, b, kind, _expected, source, note in PROJECT_EDGES:
        edge(a, b, kind, source, note)

    return uf, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broomfield-merge-all", action="store_true",
                    help="honour all four Broomfield parent edges instead of "
                         "assigning it to Boulder; the robustness alternative")
    ap.add_argument("--skip-employment", action="store_true",
                    help="skip check 6, which reads the 227 MB EFSY panel")
    ap.add_argument("--census-only-va", action="store_true",
                    help="use only the Virginia pairs a Census annexation backs, "
                         "dropping the BEA-only rows; the robustness alternative "
                         "to the BEA convention")
    args = ap.parse_args()

    print("=" * 78)
    print("10_build_county_universe.py - the time-consistent county universe")
    print("=" * 78)

    changes = read_csv(dc.COUNTY_CHANGES)
    va_rows = read_csv(dc.VA_COMBINATIONS)
    base_cz = load_cz_base()
    # The two window flags are transcribed, not derived, so they get the same
    # treatment as every other declaration in this file: assert they say what the
    # effective date says, and fail rather than quietly grouping the wrong era.
    for row in changes:
        eff = row["effective_date"]
        for col, last in (("in_window", PANEL_LAST_DATE), ("in_emp_window", EMP_LAST_DATE)):
            expect = "1" if (not eff or eff <= last) else "0"
            if row[col] != expect:
                raise SystemExit(
                    f"FAIL: {row['edge_id']} has {col}={row[col]} but "
                    f"effective_date={eff!r} against {last}; expected {expect}")

    print(f"\nCensus change edges transcribed : {len(changes)}")
    print(f"  in the contract window         : {sum(1 for r in changes if r['in_window'] == '1')}")
    print(f"  in the employment window       : {sum(1 for r in changes if r['in_emp_window'] == '1')}"
          f"  <- group formation uses these")
    print(f"  after 2018, reported only      : {sum(1 for r in changes if r['in_emp_window'] != '1')}")
    print(f"Virginia BEA combination rows    : {len(va_rows)} "
          f"in {len({r['bea_code'] for r in va_rows})} groups")
    print(f"  backed by a Census annexation  : {sum(1 for r in va_rows if r['census_backed'] == '1')}")
    print(f"1990-vintage base counties       : {len(base_cz)} "
          f"in {len(set(base_cz.values()))} commuting zones")

    print("\nreading the 42 parsed fiscal years ...")
    cells, names, thresh = load_contract_panel()
    contract_codes = {f for f, _ in cells}
    total_actions = sum(c[1] for c in cells.values())
    print(f"  {len(cells):,} county-year cells, {len(contract_codes):,} distinct "
          f"county codes, {total_actions:,} actions")

    uf, log = build_groups(changes, va_rows, args.broomfield_merge_all,
                           args.census_only_va)

    # Codes that are not places. They still need a ccg_id so the crosswalk is
    # total, but they are not members of anything: counting 48210 as a second
    # member of Harris County would report a source typo as a boundary change.
    aliases = {a: kind for a, _b, kind, _e, _s, _n in PROJECT_EDGES
               if kind in ("typo", "unplaceable")}

    # Every code any consumer might present has to resolve, so the universe is
    # the union of the three worlds, not just the contracts.
    universe = set(base_cz) | contract_codes | set(uf.parent)
    universe = {f for f in universe if is_domestic(f)}
    for f in universe:
        uf.add(f)
    roots = {r: m & universe for r, m in uf.groups().items()}

    # The representative is the lowest-numbered member that exists in the 1990
    # base, so that every ccg_id is a real 1990 county and the CZ join works.
    # Groups are then re-keyed on the representative, not on the union-find root,
    # which is an implementation detail nothing downstream should see.
    ccg_of, rep_name = {}, {}
    groups = {}
    no_base = []
    for members in roots.values():
        if not members:
            continue
        real = {m for m in members if m not in aliases}
        # A code that is not a place and has dragged no county in with it is not
        # a unit. 36173 is the case: it goes to UNPLACEABLE below, and the
        # crosswalk says so - counting it here as well made the unit total one
        # higher than the crosswalk it writes.
        if not real and all(aliases.get(m) == "unplaceable" for m in members):
            continue
        in_base = sorted(m for m in real if m in base_cz)
        rep = in_base[0] if in_base else sorted(real or members)[0]
        if not in_base:
            no_base.append(sorted(members))
        for m in members:
            ccg_of[m] = rep
        groups[rep] = real or set(members)
        label = ""
        for m in [rep] + sorted(real):
            if names[m]:
                label = names[m].most_common(1)[0][0]
                break
        n_real = len(groups[rep])
        if n_real > 1:
            label = f"{label} +{n_real - 1}" if label else f"{rep} +{n_real - 1}"
        rep_name[rep] = label

    # Codes that are not counties at all go to a bucket of their own, which is
    # what keeps their dollars out of every rung. Assigned after the groups so
    # the union-find cannot drag a real county in with them.
    for code, kind in aliases.items():
        if kind == "unplaceable":
            ccg_of[code] = UNPLACEABLE
    ccg_of[UNPLACEABLE] = UNPLACEABLE

    multi = {r: m for r, m in groups.items() if len(m) > 1}
    print(f"\nCCG units                        : {len(groups):,}")
    print(f"  singletons                     : {len(groups) - len(multi):,}")
    print(f"  multi-county groups            : {len(multi)}")
    print(f"  codes inside them              : {sum(len(m) for m in multi.values())}")
    print(f"  aliases (not places)           : {len(aliases)} {sorted(aliases)}")

    # Contract codes the 1990 vintage does not contain. These are the codes the
    # panel would silently lose without a crosswalk, so they are listed rather
    # than counted.
    dollars_of_code = collections.defaultdict(float)
    actions_of_code = collections.Counter()
    for (fips, _fy), (dollars, n) in cells.items():
        dollars_of_code[fips] += dollars
        actions_of_code[fips] += n
    outside = sorted((f for f in contract_codes if f not in base_cz),
                     key=lambda f: -dollars_of_code[f])
    print(f"\ncontract codes outside the 1990 base, mapped forward not dropped "
          f"({len(outside)}):")
    for f in outside:
        tgt = ccg_of.get(f, "?")
        print(f"    {f} {str(names[f].most_common(1)[0][0] if names[f] else ''):<24} "
              f"${dollars_of_code[f]/1e6:9.1f}m {actions_of_code[f]:6,} actions -> {tgt}")

    # Commuting zones, made consistent with the groups rather than assumed to be.
    czuf = UnionFind()
    for z in set(base_cz.values()):
        czuf.add(z)
    straddle = []
    for rep, members in groups.items():
        zones = {base_cz[m] for m in members if m in base_cz}
        if len(zones) > 1:
            straddle.append((rep, sorted(zones), sorted(members)))
            zs = sorted(zones)
            for z in zs[1:]:
                czuf.union(zs[0], z)
    ccz_of_zone = {z: czuf.find(z) for z in set(base_cz.values())}
    n_ccz = len(set(ccz_of_zone.values()))

    # ---------------------------------------------------------------- checks
    failures = []

    # 1  partition
    unmapped = sorted(f for f in contract_codes if f not in ccg_of)
    if unmapped:
        failures.append(("partition", f"{len(unmapped)} contract codes unmapped"))
    seen = collections.Counter()
    for members in groups.values():
        seen.update(members)
    dupes = [f for f, k in seen.items() if k > 1]
    if dupes:
        failures.append(("partition", f"{len(dupes)} codes in more than one group"))
    orphan = [g for g in no_base if not (set(g) & set(aliases))]
    if orphan:
        failures.append(("partition", f"{len(orphan)} groups contain no 1990-base county"))
    print(f"\n[1] partition            unmapped contract codes: {len(unmapped)}; "
          f"codes in >1 group: {len(dupes)}; "
          f"groups with no 1990-base county: {len(orphan)} {orphan[:5]}")

    # 2  dollar conservation
    by_year_cty = collections.defaultdict(float)
    by_year_ccg = collections.defaultdict(float)
    lost = collections.defaultdict(float)
    for (fips, fy), (dollars, _n) in cells.items():
        by_year_cty[fy] += dollars
        rep = ccg_of.get(fips)
        if rep is None or rep == UNPLACEABLE:
            lost[fy] += dollars
        else:
            by_year_ccg[fy] += dollars
    bad_years = []
    for fy in sorted(by_year_cty):
        diff = by_year_cty[fy] - by_year_ccg[fy] - lost[fy]
        if abs(diff) > 0.5:
            bad_years.append((fy, diff))
    if bad_years:
        failures.append(("dollars", f"{len(bad_years)} years do not reconcile"))
    print(f"[2] dollar conservation  {len(by_year_cty)} fiscal years, "
          f"{'all exact' if not bad_years else str(bad_years[:3])}")
    print(f"    routed to {UNPLACEABLE} and excluded from every rung: "
          f"${sum(lost.values()):,.0f}")

    # 3  time-invariance. The map is keyed on the county code alone, so a group's
    #    membership cannot vary by year by construction. What can still be
    #    checked is the property that makes that safe: every group's dollars are
    #    invariant to which member code the source happened to write. Relabel the
    #    whole panel onto representatives and confirm the per-year group totals
    #    are unchanged when each member's rows are attributed to any other member
    #    of its own group.
    per_group_year = collections.defaultdict(float)
    shuffled = collections.defaultdict(float)
    for (fips, fy), (dollars, _n) in cells.items():
        rep = ccg_of.get(fips)
        if rep is None or rep == UNPLACEABLE:
            continue
        per_group_year[(rep, fy)] += dollars
        alt = sorted(groups.get(rep, {rep}))[-1]        # a different member
        shuffled[(ccg_of.get(alt, rep), fy)] += dollars
    drift = [k for k in per_group_year
             if abs(per_group_year[k] - shuffled.get(k, 0.0)) > 0.5]
    if drift:
        failures.append(("time-invariance", f"{len(drift)} group-years move on relabel"))
    print(f"[3] time-invariance      {len(per_group_year):,} group-years invariant to "
          f"which member code the source wrote: {'yes' if not drift else 'NO'}")

    # 4  stale-code leakage: predecessor and successor must land together
    STALE_PAIRS = [
        ("12025", "12086", "Dade -> Miami-Dade 1997, both codes used FY2004-06"),
        ("02010", "02013", "Aleutian Islands -> Aleutians East 1987"),
        ("02010", "02016", "Aleutian Islands -> Aleutians West 1987"),
        ("02290", "02068", "Yukon-Koyukuk -> Denali 1990"),
        ("02070", "02164", "Dillingham -> Lake and Peninsula 1989"),
        ("02140", "02188", "Kobuk -> Northwest Arctic 1986"),
        ("02231", "02282", "Skagway-Yakutat-Angoon -> Yakutat 1992"),
        ("02231", "02232", "Skagway-Yakutat-Angoon -> Skagway-Hoonah-Angoon 1992"),
        ("51780", "51083", "South Boston reverted to a town 1995"),
        ("51560", "51005", "Clifton Forge reverted to a town 2001"),
        ("30113", "30067", "Yellowstone National Park abolished 1997"),
        ("51695", "51800", "Nansemond city -> Suffolk"),
        ("51123", "51800", "Nansemond County -> Suffolk 1972"),
        ("13510", "13215", "Columbus city GA -> Muscogee 1971"),
        ("29193", "29186", "Ste. Genevieve MO code change"),
    ]
    split = [(a, b, why) for a, b, why in STALE_PAIRS
             if ccg_of.get(a) != ccg_of.get(b)]
    if split:
        failures.append(("stale", f"{len(split)} predecessor/successor pairs split"))
    print(f"[4] stale codes          {len(STALE_PAIRS)} pairs checked, "
          f"{len(split)} split")
    for a, b, why in split:
        print(f"    FAIL {a} -> {ccg_of.get(a)}  vs  {b} -> {ccg_of.get(b)}   {why}")

    # 5  commuting-zone containment
    print(f"[5] CZ containment       {len(set(base_cz.values()))} commuting zones -> "
          f"{n_ccz} consistent zones ({len(straddle)} groups straddled a boundary)")
    for rep, zones, members in straddle:
        print(f"    merged CZ {zones} for group {rep}: {members}")
    bad_cz = []
    for rep, members in groups.items():
        zs = {ccz_of_zone[base_cz[m]] for m in members if m in base_cz}
        if len(zs) > 1:
            bad_cz.append(rep)
    if bad_cz:
        failures.append(("czone", f"{len(bad_cz)} groups still straddle a CZ"))

    # 6  the contract universe must sit inside the employment universe, and it
    #    must do so YEAR BY YEAR. The pooled set comparison this check used to
    #    make is too weak: a unit present in the employment file in any one year
    #    passed it while being absent from every other, which is exactly what DC
    #    was doing for 28 years. See EFSY_RESIDUAL_AS_COUNTY.
    efsy_only, contracts_only = [], []
    efsy_emp = {}
    if args.skip_employment or not dc.EFSY_PANEL.exists():
        print("[6] employment           SKIPPED")
    else:
        efsy_emp, dropped_residual, recoded, problems = load_employment()
        for why in problems:
            failures.append(("employment", why))
        efsy_codes = {c for c, _ in efsy_emp}
        efsy_codes = {c for c in efsy_codes if is_domestic(c)}
        efsy_ccg = {ccg_of[c] for c in efsy_codes if c in ccg_of}
        missing = sorted(c for c in efsy_codes if c not in ccg_of)
        if missing:
            failures.append(("employment", f"{len(missing)} EFSY codes unmapped"))

        # CCG-years on each side. A contract fiscal year maps to its own label;
        # the transition quarter folds into FY1976, which it sits inside.
        emp_years = collections.defaultdict(set)
        for code, yr in efsy_emp:
            if code in ccg_of:
                emp_years[ccg_of[code]].add(yr)
        con_years = collections.defaultdict(set)
        con_dollars = collections.defaultdict(float)
        for (fips, fy), (dollars, _n) in cells.items():
            yr = int(fy.replace("tq", ""))
            if not (PANEL_FIRST_YEAR <= yr <= 2006) or fips not in ccg_of:
                continue
            rep = ccg_of[fips]
            con_years[rep].add(yr)
            con_dollars[(rep, yr)] += dollars

        contract_ccg = set(con_years) - {UNPLACEABLE}
        efsy_only = sorted(efsy_ccg - contract_ccg)
        contracts_only = sorted(contract_ccg - efsy_ccg)
        if contracts_only:
            failures.append(("employment",
                             f"{len(contracts_only)} CCGs have contracts and no "
                             f"employment in any year: {contracts_only[:10]}"))

        gaps = sorted((rep, yr) for rep in contract_ccg
                      for yr in con_years[rep] - emp_years.get(rep, set()))
        undeclared = [g for g in gaps if g not in EMPLOYMENT_GAPS]
        stale = sorted(g for g in EMPLOYMENT_GAPS if g not in gaps)
        if undeclared:
            failures.append(("employment",
                             f"{len(undeclared)} CCG-years carry contracts with no "
                             f"employment and are not declared in "
                             f"EMPLOYMENT_GAPS: {undeclared[:10]}"))
        if stale:
            failures.append(("employment",
                             f"{len(stale)} EMPLOYMENT_GAPS entries no longer "
                             f"occur and should be deleted: {stale}"))

        print(f"[6] employment           EFSY {PANEL_FIRST_YEAR}-{EMP_LAST_YEAR}: "
              f"{len(efsy_codes):,} codes -> {len(efsy_ccg):,} CCGs; "
              f"contracts -> {len(contract_ccg):,} CCGs")
        for state, code, years, moved in recoded:
            print(f"    residual recoded       state {state} fipscty=999 -> {code} "
                  f"in {len(years)} years {min(years)}-{max(years)}, "
                  f"{moved:,.0f} employee-years")
        print(f"    residual left unplaced : {dropped_residual:,.0f} employee-years, "
              f"none of them a whole state")
        print(f"    unmapped EFSY codes    : {len(missing)} {missing[:10]}")
        print(f"    CCGs with employment and no contract ever: {len(efsy_only)}")
        print(f"    CCGs with contracts and no employment    : {len(contracts_only)} "
              f"{contracts_only[:10]}")
        print(f"    CCG-YEARS with contracts and no employment: {len(gaps)}, "
              f"{len(undeclared)} undeclared")
        for rep, yr in gaps:
            tag = "declared" if (rep, yr) in EMPLOYMENT_GAPS else "UNDECLARED"
            print(f"      {tag:<10} {rep} {yr}  "
                  f"${con_dollars[(rep, yr)]/1e6:,.3f}m  "
                  f"{EMPLOYMENT_GAPS.get((rep, yr), '')[:70]}")

    # 7  level-break sweep (REPORTED)
    change_years = collections.defaultdict(set)
    for row in changes:
        if not row["effective_date"]:
            continue
        yr = int(row["effective_date"][:4])
        for f in (row["fips_from"], row["fips_to"]):
            if f:
                change_years[f].update({yr, yr + 1})
    series = collections.defaultdict(dict)
    for (fips, fy), (dollars, _n) in cells.items():
        if fy.endswith("tq"):
            continue
        series[fips][int(fy)] = dollars
    def window_mean(obs, lo, hi):
        vals = [obs.get(y, 0.0) for y in range(lo, hi + 1)]
        return sum(vals) / len(vals)

    def sweep(obs_by_unit, floor, ratio, first, last):
        """Three-year means either side of every year, deduplicated per unit.

        A one-year spike is what a single large contract looks like; a boundary
        change does not come back down, which is why the comparison is on
        three-year means rather than on adjacent years. Consecutive flagged years
        are one event and are collapsed to the largest.
        """
        listed, unlisted = [], []
        for unit, obs in obs_by_unit.items():
            hits = []
            for yr in range(first + 3, last - 1):
                before = window_mean(obs, yr - 3, yr - 1)
                after = window_mean(obs, yr, yr + 2)
                if max(before, after) < floor:
                    continue
                lo, hi = sorted((before, after))
                if lo <= 0 or hi / lo < ratio:
                    continue
                hits.append((yr, before, after))
            run = []
            for hit in hits + [None]:
                if hit is not None and (not run or hit[0] == run[-1][0] + 1):
                    run.append(hit)
                    continue
                if run:
                    yr, before, after = max(run, key=lambda h: abs(h[2] - h[1]))
                    item = (unit, yr, before, after)
                    (listed if set(range(run[0][0], run[-1][0] + 1))
                     & change_years.get(unit, set()) else unlisted).append(item)
                run = [hit] if hit is not None else []
        return listed, unlisted

    c_listed, c_unlisted = sweep(series, 1_000_000, 10, 1966, 2006)
    print(f"[7] level breaks         REPORT")
    print(f"    contract dollars, 10x on 3-year means above $1m: "
          f"{len(c_listed)} at a documented change date, {len(c_unlisted)} not.")
    print(f"    Contract dollars are too lumpy for this to identify a boundary "
          f"change - a single large award moves a county tenfold - so the "
          f"employment sweep below is the one that discriminates.")
    for fips, yr, before, after in sorted(c_listed, key=lambda t: -abs(t[3] - t[2]))[:8]:
        print(f"      listed   {fips} FY{yr}: ${before/1e6:,.1f}m -> ${after/1e6:,.1f}m")

    if efsy_emp:
        emp_series = collections.defaultdict(dict)
        for (code, yr), emp in efsy_emp.items():
            emp_series[code][yr] = emp
        e_listed, e_unlisted = sweep(emp_series, 1_000, 2.0,
                                     PANEL_FIRST_YEAR, 2006)
        print(f"    county employment, 2x on 3-year means above 1,000 employees: "
              f"{len(e_listed)} at a documented change date, {len(e_unlisted)} not")
        for code, yr, before, after in sorted(e_listed,
                                              key=lambda t: -abs(t[3] - t[2]))[:10]:
            print(f"      listed   {code} {yr}: {before:,.0f} -> {after:,.0f} emp "
                  f"-> {ccg_of.get(code)}")
        for code, yr, before, after in sorted(e_unlisted,
                                              key=lambda t: -abs(t[3] - t[2]))[:15]:
            grp = ccg_of.get(code, "?")
            tag = "already in a group" if len(groups.get(grp, ())) > 1 else "SINGLETON"
            print(f"      unlisted {code} {yr}: {before:,.0f} -> {after:,.0f} emp "
                  f"({tag})")
    else:
        print("    county employment sweep SKIPPED - no EFSY panel read")

    # 8  threshold accounting (REPORTED)
    print("[8] threshold            REPORT: no rows are filtered by this script")
    print("      fy    actions   <$10k  $10-25k   counties  cty>=25k  only<25k    $bn")
    for fy in sorted(thresh, key=lambda s: (dc.zark_year(s), s)):
        t = thresh[fy]
        n = t["lt10k"] + t["10to25k"] + t["ge25k"]
        ncty = sum(1 for k in t if isinstance(k, tuple) and k[0] == "cty")
        ncty25 = sum(1 for k in t if isinstance(k, tuple) and k[0] == "cty25k")
        dollars = t["dol_lt10k"] + t["dol_10to25k"] + t["dol_ge25k"]
        print(f"    {fy:>6} {n:9,} {t['lt10k']:7,} {t['10to25k']:8,} "
              f"{ncty:10,} {ncty25:9,} {ncty - ncty25:9,} {dollars/1e9:7.2f}")

    # 9  coverage against the known tables
    missing_sup = [c for c in dc.SUPERSEDED
                   if len(groups.get(ccg_of.get(c, c), ())) < 2]
    wrong_sup = [c for c in dc.SUPERSEDED
                 if ccg_of.get(c) != ccg_of.get(dc.SUPERSEDED_TARGET[c])]
    if missing_sup or wrong_sup:
        failures.append(("coverage",
                         f"{len(missing_sup)} SUPERSEDED singletons, "
                         f"{len(wrong_sup)} not with their successor"))
    print(f"[9] coverage             all {len(dc.SUPERSEDED)} SUPERSEDED codes sit "
          f"with their successor in a multi-county group: "
          f"{'yes' if not (missing_sup or wrong_sup) else (missing_sup, wrong_sup)}")
    va_bea_only = sorted({r["county_fips5"] for r in va_rows
                          if r["census_backed"] != "1"})
    print(f"    Virginia codes merged on the BEA convention alone, with no Census "
          f"annexation behind them ({len(va_bea_only)}):")
    for f in va_bea_only:
        print(f"      {f} {str(names[f].most_common(1)[0][0] if names[f] else ''):<22} "
              f"${dollars_of_code.get(f, 0)/1e9:7.3f}bn -> {ccg_of.get(f)}   "
              f"(--census-only-va keeps it separate)")

    # 10  guarded judgements
    actions_of = collections.Counter()
    for (fips, _fy), (_d, n) in cells.items():
        actions_of[fips] += n
    print("[10] guards              declared vs actual contract actions")
    for a, b, kind, expected, _src, _note in PROJECT_EDGES:
        got = actions_of.get(a, 0)
        ok = got == expected
        if not ok:
            failures.append(("guard", f"{a}: expected {expected} actions, got {got}"))
        print(f"    {'ok  ' if ok else 'FAIL'} {a} -> {b:<12} {kind:<17} "
              f"expected {expected:>5}, got {got:>5}")

    # 11  write the crosswalk
    #
    # Two kinds of label, kept apart on purpose. `kind` says what the ROW is to a
    # consumer joining on county_fips5; `change_kind` and `source` say why the
    # group exists. A representative and the code merged into it share the second
    # pair and differ on the first.
    note_of, source_of, change_of, expected_of = {}, {}, {}, {}
    for a, b, kind, source, note in log:
        for side in (a, b):
            note_of.setdefault(side, note)
            source_of.setdefault(side, source)
            change_of.setdefault(side, kind)
    for a, _b, kind, expected, source, note in PROJECT_EDGES:
        change_of[a], source_of[a], note_of[a], expected_of[a] = (
            kind, source, note, expected)
    group_sources = {}
    for rep, members in groups.items():
        srcs = sorted({source_of[m].split(":")[0] for m in members if m in source_of})
        group_sources[rep] = "+".join(srcs)

    out = dc.CCG_CROSSWALK
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["county_fips5", "ccg_id", "ccg_name", "czone", "ccz_id",
                    "n_members", "kind", "change_kind", "source",
                    "expected_actions", "in_1990_base", "in_contracts", "note"])
        for fips in sorted(universe):
            rep = ccg_of[fips]
            members = groups.get(rep, {fips})
            zone = base_cz.get(fips, "")
            if fips in aliases:
                kind = aliases[fips]
            elif len(members) == 1:
                kind = "identity"
            elif fips == rep:
                kind = "group_head"
            else:
                kind = "group_member"
            w.writerow([
                fips, rep, rep_name.get(rep, ""), zone,
                ccz_of_zone.get(zone, "") if zone != "" else "",
                len(members), kind,
                change_of.get(fips, ""),
                source_of.get(fips) or group_sources.get(rep) or "base",
                expected_of.get(fips, ""),
                1 if fips in base_cz else 0,
                1 if fips in contract_codes else 0,
                note_of.get(fips, ""),
            ])
    print(f"[11] crosswalk           wrote {len(universe):,} rows to {out}")

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ccg_id", "ccg_name", "n_members", "members", "czone_count",
                    "contract_dollars", "contract_actions"])
        dollars_of = collections.defaultdict(float)
        acts_of = collections.Counter()
        for (fips, _fy), (dollars, n) in cells.items():
            rep = ccg_of.get(fips, UNPLACEABLE)
            dollars_of[rep] += dollars
            acts_of[rep] += n
        for rep, members in sorted(multi.items(), key=lambda kv: -dollars_of[kv[0]]):
            w.writerow([rep, rep_name.get(rep, ""), len(members),
                        " ".join(sorted(members)),
                        len({base_cz[m] for m in members if m in base_cz}),
                        f"{dollars_of[rep]:.0f}", acts_of[rep]])
        print(f"                         wrote {len(multi)} multi-county groups "
              f"to {REPORT}")

        print("\nthe multi-county groups, by contract dollars:")
        for rep, members in sorted(multi.items(), key=lambda kv: -dollars_of[kv[0]])[:40]:
            print(f"    {rep} {rep_name.get(rep,''):<28} "
                  f"${dollars_of[rep]/1e9:8.3f}bn  {' '.join(sorted(members))}")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED: {len(failures)} criteria")
        for what, why in failures:
            print(f"  {what}: {why}")
        return 1
    print("All criteria passed. Exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
