#!/usr/bin/env python3
"""Build the county-by-industry employment cells Part 2 needs, and report on them.

EFSY as shipped is unusable as a disaggregation outcome for two reasons, and this
script fixes both.

NOT EVERYTHING REACHES FOUR DIGITS. EFSY assigns employment to the most detailed
NAICS code it can and parks the rest on a shorter root: "237///" is the part of
sector 237 that could not be pushed further down, not the sector total. Its own
readme says so. Roots come in two forms and EFSY marks them differently - the
three-digit ones are slash-padded and DOCUMENTED, the two-digit ones are
dash-padded and are not - so anything that looks for a slash counts every sector
root as a fully resolved six-digit code. Together they are 63.7m employee-years,
all of it before 1998.

Each root is split across the genuine four-digit children THAT SAME COUNTY-YEAR
already reports, in proportion to their employment:

    emp[237///] = 1,000 ; observed children 2371 = 600, 2372 = 200, 2373 absent
      -> 2371 += 750, 2372 += 250, 2373 stays ZERO

The county's own mix is used rather than a national one because a national split
puts employment into industries the county genuinely has none of, and the zero
pattern is the object of study here, not noise to be smoothed. 95% of residual
employment has a same-county-year sibling to be split across; the fallback ladder
covers the rest, and only its last two rungs can put employment where the county
never had any, which is why they are measured and capped. Sectors 95 (Auxiliaries)
and 99 (Unclassified) have no four-digit codes anywhere in NAICS, so their roots
cannot be pushed down by any rule and stay as cells of their own. Both variants
are written, allocated and raw, because post-1998 they are identical and pre-1998
the difference is the whole question.

The two-digit rung is unaffected by all of this: every code, root or not, has a
two-digit sector, so that rung is exact and needs no imputation.

INDUSTRY CELLS ARE RECTANGULAR. An industry absent within a reported county-year
is materialised as zero so entry and exit, defined on a cell being zero at t-h or
t, can be computed. The raw file also contains nonpositive rows and entire missing
county-years. Those cannot be interpreted as literal zero county employment. The
county-total output records them, and dcads_common.load_analysis_ccg_map excludes
every CCG that is not strictly positive in every year from 1975 through 2006
before any estimation sample or aggregation rung is constructed.

Because the support is the same in every year, the row order is identical in every
year too, so entry and exit are a comparison of two aligned arrays rather than a
join, and every year's file has exactly the same number of rows.

NEGATIVE CELLS ARE REMOVED AT FOUR DIGITS. EFSY's desuppression leaves a handful of
tiny negative values behind, and since 2026-08-17 they are deleted at the allocated
four-digit cell - see --negative-cells and section [4]. The deletion happens on the
allocated frame BEFORE the support is built, so rectangularisation puts the cell
back as a zero: no row leaves any file, the support and row order are untouched, and
the cell becomes indistinguishable from a county-industry EFSY never reported. The
two-digit rung and the county rung are then summed from the cleaned four-digit
cells, so all three rungs stay additive and none of them can carry a negative.

WHAT THIS IS NOT. It is not a re-run of check 6. That the CCG set is complete, and
that the contract universe sits inside the employment universe year by year, is
10_build_county_universe.py's job and stays there. It is also not an additivity
check of cells against county totals - the county rung here is aggregated FROM the
cells, so that identity holds by construction and asserting it would verify
nothing. What is checked instead is conservation against RAW EFSY: the ALLOCATION
is asserted to move nothing, and the deletion is then accounted for separately as
the one deliberate departure. Neither is circular.

Output: dc.EMP_OUT_DIR/{naics4_alloc,naics4_raw,naics2}/y<year>.parquet
        dc.EMP_OUT_DIR/efsy_ccg_year.parquet
        dc.EMP_OUT_DIR/employment_cells_report.csv
        dc.LOG_DIR/efsy_granularity_by_year.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 1975
LAST_YEAR = 2018
BATCH = 2_000_000

# EFSY's own vintage breaks, both of which change the number of cells for reasons
# that have nothing to do with defence spending. 1998 is CBP going native NAICS;
# before it EFSY imputes NAICS from SIC and spreads employment over more codes.
# 2017 is sparser still - the row count roughly halves while employment falls 11%
# - so the last two years are not comparable with what precedes them and entry and
# exit across 2016->2017 would measure an artifact. Reported, never smoothed.
VINTAGE_BREAKS = (1998, 2017)

# A four-digit cell is encoded as the integer 1000-9999 it already is. A residual
# root is encoded above it so the two never collide and the root is recoverable by
# subtraction. No lookup table, no strings until write time.
#
# THERE ARE TWO KINDS OF ROOT AND EFSY MARKS THEM DIFFERENTLY. A three-digit root
# is slash-padded, "237///", which is the form the EFSY readme documents. A
# two-digit root is DASH-padded, "92----", which it does not. Anything that looks
# for a slash to find a root therefore silently counts every two-digit root as a
# fully resolved six-digit code - 738,693 rows and 13,045,404 employee-years of
# them, all before 1998. Resolved digits are counted as leading digits here, which
# catches both.
RESIDUAL3 = 100_000     # 100_011 .. 100_999, a three-digit root
RESIDUAL2 = 200_000     # 200_011 .. 200_099, a two-digit sector root

ALLOC_RULES = {
    0: "same CCG, same year",
    1: "same CCG, pooled over all years",
    2: "same state, same year",
    3: "national, same year",
    4: "no genuine child anywhere - kept as a cell",
}
# Rules 2 and 3 are the only ones that can put employment into an industry the
# county never reports. Measured at 0.024% of national employment; anything above
# this fails the run rather than being noticed later.
OUTSIDE_SHARE_LIMIT_PCT = 0.05


def code_to_label(code):
    """4321 -> '4321'; a three-digit root -> '432R'; a sector root -> '43R'; the
    two-digit rung -> '43'."""
    if code >= RESIDUAL2:
        return f"{code - RESIDUAL2:02d}R"
    if code >= RESIDUAL3:
        return f"{code - RESIDUAL3:03d}R"
    return f"{code:04d}" if code >= 1000 else f"{code:02d}"


def root_len(codes):
    """How many digits an encoded code resolves to: 2, 3 or 4."""
    codes = np.asarray(codes)
    return np.where(codes >= RESIDUAL2, 2, np.where(codes >= RESIDUAL3, 3, 4)).astype(np.int8)


def prefix_of(codes, n):
    """The n-digit prefix of any encoded code, genuine or root. A code that does
    not reach n digits returns -1 and can never be a child at that level."""
    codes = np.asarray(codes)
    if n == 2:
        return np.where(codes >= RESIDUAL2, codes - RESIDUAL2,
                        np.where(codes >= RESIDUAL3, (codes - RESIDUAL3) // 10,
                                 codes // 100)).astype(np.int32)
    if n == 3:
        return np.where(codes >= RESIDUAL2, -1,
                        np.where(codes >= RESIDUAL3, codes - RESIDUAL3,
                                 codes // 10)).astype(np.int32)
    raise ValueError(n)


def sector_of(codes):
    """Two-digit sector. Every kind of code has one, which is why the two-digit
    rung needs no imputation even though the four-digit rung does."""
    return prefix_of(codes, 2)


def encode(naics):
    """naics12 -> (code, resolved digits).

    Resolved digits are the LEADING DIGITS, not the characters before a slash:
    three-digit roots are slash-padded and two-digit roots are dash-padded, and
    only counting digits catches both.
    """
    digits = naics.str.replace(r"\D.*$", "", regex=True)
    resolved = digits.str.len().to_numpy().astype(np.int8)
    if (resolved < 2).any():
        raise SystemExit("FAIL: a NAICS code resolving to fewer than two digits "
                         "appeared; the encoding in this script has no room for it")
    code = np.empty(len(naics), dtype=np.int64)
    for n, base in ((4, 0), (3, RESIDUAL3), (2, RESIDUAL2)):
        m = (resolved >= 4) if n == 4 else (resolved == n)
        if not m.any():
            continue
        head = np.asarray(pd.to_numeric(naics.str.slice(0, n)[m]), dtype=np.int64)
        code[m] = base + head
    return code.astype(np.int32), resolved


def load_crosswalk():
    """county_fips5 -> ccg_id as ints, plus the CCG universe the panel spans."""
    ccg_of, universe = {}, set()
    with dc.CCG_CROSSWALK.open() as fh:
        for row in csv.DictReader(fh):
            if row["ccg_id"] == "unplaceable":
                continue
            ccg_of[int(row["county_fips5"])] = int(row["ccg_id"])
            universe.add(int(row["ccg_id"]))
    return ccg_of, sorted(universe)


def load_cells(ccg_of):
    """Stream EFSY into (ccg, year, code) employment with the geography applied."""
    parts, residual_parts, gran_parts = [], [], []
    unmapped = set()
    neg_rows, neg_total = [0], [0.0]

    pf = pq.ParquetFile(dc.EFSY_PANEL)
    for batch in pf.iter_batches(batch_size=BATCH,
                                 columns=["fipstate", "fipscty", "naics12", "emp", "year"]):
        d = batch.to_pandas()
        d = d[(d["year"] >= FIRST_YEAR) & (d["year"] <= LAST_YEAR)]
        if d.empty:
            continue

        state = d["fipstate"].to_numpy(dtype=np.int32)
        cty = d["fipscty"].to_numpy(dtype=np.int32)
        year = d["year"].to_numpy(dtype=np.int16)
        emp = d["emp"].to_numpy(dtype=np.float64)
        # EFSY carries a handful of tiny negative values - 4,661 rows summing to
        # about -1,276 employee-years out of 4.35 billion, none below -1.0, all
        # before 1998 - which are rounding artifacts of its desuppression.
        #
        # They are kept and counted HERE, through the whole allocation, and are
        # deleted only at the allocated four-digit cell in section [4], under
        # --negative-cells. Nothing in this function or in the ladder floors them,
        # for two reasons: the ladder's conservation against raw EFSY is asserted
        # on the pre-deletion frame and would break if the input were quietly
        # changed, and a negative root still has to be distributed rather than
        # discarded or the allocated total stops matching the root total. What a
        # negative must never become is an allocation WEIGHT, and _ladder clips
        # for that.
        #
        # So the two rules read together: negatives are data on the way in, and a
        # declared, measured deletion on the way out.
        neg = emp < 0
        neg_rows[0] += int(neg.sum())
        neg_total[0] += float(emp[neg].sum())

        code, resolved = encode(d["naics12"])

        # The granularity table, measured on every row before any filtering.
        gran_parts.append(pd.DataFrame({"year": year, "resolved": resolved, "emp": emp})
                          .groupby(["year", "resolved"], observed=True, as_index=False)
                          .agg(emp=("emp", "sum"), rows=("emp", "size")))

        is_res = cty == 999
        residual_parts.append(pd.DataFrame({
            "state": state[is_res], "year": year[is_res],
            "code": code[is_res], "emp": emp[is_res],
        }).groupby(["state", "year", "code"], observed=True, as_index=False)["emp"].sum())

        # Which state-years have any non-999 row at all, for the discovery sweep.
        residual_parts[-1].attrs["_"] = None
        seen = pd.DataFrame({"state": state[~is_res], "year": year[~is_res]}).drop_duplicates()

        fips = state[~is_res] * 1000 + cty[~is_res]
        mapped = pd.Series(fips).map(ccg_of)
        bad = mapped.isna().to_numpy()
        if bad.any():
            unmapped.update(np.unique(fips[bad]).tolist())

        part = pd.DataFrame({
            "ccg": mapped.to_numpy(),
            "year": year[~is_res],
            "code": code[~is_res],
            "emp": emp[~is_res],
        })
        part = part[part["ccg"].notna()]
        part["ccg"] = part["ccg"].astype(np.int32)
        parts.append(part.groupby(["ccg", "year", "code"], observed=True,
                                  as_index=False)["emp"].sum())
        parts[-1].attrs["seen"] = seen

    seen = pd.concat([p.attrs["seen"] for p in parts], ignore_index=True).drop_duplicates()
    for p in parts:
        p.attrs.pop("seen", None)

    gran = (pd.concat(gran_parts, ignore_index=True)
            .groupby(["year", "resolved"], observed=True, as_index=False)
            .agg(emp=("emp", "sum"), rows=("rows", "sum")))
    residual = (pd.concat(residual_parts, ignore_index=True)
                .groupby(["state", "year", "code"], observed=True, as_index=False)["emp"].sum())
    cells = pd.concat(parts, ignore_index=True)
    del parts, gran_parts, residual_parts
    cells = cells.groupby(["ccg", "year", "code"], observed=True,
                          as_index=False)["emp"].sum()
    cells = cells[cells["emp"] != 0].reset_index(drop=True)

    # The discovery sweep, as 10_build_county_universe.py runs it but at cell
    # level: a residual becomes the county only where the state has no non-999 row
    # at all that year, so the whole of the state's employment is in the residual
    # and no double count is possible.
    res_years = (residual[residual["emp"] > 0]
                 .groupby(["state", "year"], observed=True, as_index=False)["emp"].sum())
    res_years = res_years.merge(seen.assign(_seen=1), on=["state", "year"], how="left")
    wholly = res_years[res_years["_seen"].isna()]
    found = {f"{int(s):02d}": set(g["year"].astype(int))
             for s, g in wholly.groupby("state", observed=True)}

    problems = []
    for st, years in sorted(found.items()):
        if st not in dc.EFSY_RESIDUAL_AS_COUNTY:
            problems.append(f"state {st} has its whole employment in the statewide "
                            f"residual in {sorted(years)} and is not declared in "
                            f"dcads_common.EFSY_RESIDUAL_AS_COUNTY")
            continue
        _c, first, last, _w = dc.EFSY_RESIDUAL_AS_COUNTY[st]
        if years != set(range(first, last + 1)):
            problems.append(f"state {st} is declared for {first}-{last} but the "
                            f"sweep finds {min(years)}-{max(years)} "
                            f"({len(years)} years)")
    for st in dc.EFSY_RESIDUAL_AS_COUNTY:
        if st not in found:
            problems.append(f"state {st} is declared in EFSY_RESIDUAL_AS_COUNTY and "
                            f"the sweep no longer finds it")
    if unmapped:
        problems.append(f"{len(unmapped)} county codes are not in the CCG crosswalk: "
                        f"{sorted(unmapped)[:10]}")

    # Attribute the declared residuals only once every guard has passed, so a run
    # that is going to fail does not also move employment on a failed declaration.
    recoded = {}
    if not problems:
        for st, (code5, first, last, _w) in dc.EFSY_RESIDUAL_AS_COUNTY.items():
            take = residual[(residual["state"] == int(st))
                            & (residual["year"] >= first)
                            & (residual["year"] <= last)].copy()
            take["ccg"] = ccg_of[int(code5)]
            recoded[st] = float(take["emp"].sum())
            cells = pd.concat([cells, take[["ccg", "year", "code", "emp"]]],
                              ignore_index=True)
        cells = cells.groupby(["ccg", "year", "code"], observed=True,
                              as_index=False)["emp"].sum()
        cells["ccg"] = cells["ccg"].astype(np.int32)

    dropped = residual.groupby("year", observed=True)["emp"].sum()
    for st, (code5, first, last, _w) in dc.EFSY_RESIDUAL_AS_COUNTY.items():
        if st in recoded:
            moved = (residual[(residual["state"] == int(st))
                              & (residual["year"] >= first)
                              & (residual["year"] <= last)]
                     .groupby("year", observed=True)["emp"].sum())
            dropped = dropped.subtract(moved, fill_value=0.0)

    return cells, dict(gran=gran, dropped=dropped, recoded=recoded, problems=problems,
                       neg_rows=neg_rows[0], neg_total=neg_total[0])


def _ladder(gen, outstanding, plen):
    """Split `outstanding` across the genuine children sharing its `plen`-digit
    prefix. Returns (additions, provenance, leftover).

    gen and outstanding both carry ccg, year, state, emp and a `pfx` column that
    is already the prefix at this level.
    """
    additions, provenance = [], {}

    # Weights are clipped at zero. EFSY's few negative cells are kept in `emp` for
    # conservation but must never become a negative share, and a group whose whole
    # weight clips to zero has nothing to distribute across, so it falls through to
    # the next rule instead of dividing by zero.
    gen = gen.copy()
    gen["wgt"] = gen["emp"].clip(lower=0.0)

    # Rule 0 expands nothing: the weights already sit on the genuine rows, so map
    # each root's residual onto them and scale.
    tot0 = gen.groupby(["ccg", "year", "pfx"], observed=True)["wgt"].transform("sum").to_numpy()
    r0 = (outstanding.groupby(["ccg", "year", "pfx"], observed=True, as_index=False)["emp"]
          .sum().rename(columns={"emp": "res_amt"}))
    merged = gen[["ccg", "year", "pfx"]].merge(r0, on=["ccg", "year", "pfx"], how="left")
    usable = tot0 > 0
    add0 = np.zeros(len(gen))
    add0[usable] = (merged["res_amt"].fillna(0.0).to_numpy()[usable]
                    * gen["wgt"].to_numpy()[usable] / tot0[usable])
    additions.append(pd.DataFrame({"ccg": gen["ccg"].to_numpy(), "year": gen["year"].to_numpy(),
                                   "code": gen["code"].to_numpy(), "emp": add0,
                                   "rule": np.int8(0)}))
    provenance[0] = float(add0.sum())
    outstanding = _drop_served(outstanding,
                               gen.loc[usable, ["ccg", "year", "pfx"]].drop_duplicates(),
                               ["ccg", "year", "pfx"])

    for rule, keys in ((1, ["ccg", "pfx"]),
                       (2, ["state", "year", "pfx"]),
                       (3, ["year", "pfx"])):
        if outstanding.empty:
            provenance[rule] = 0.0
            continue
        w = gen.groupby(keys + ["code"], observed=True, as_index=False)["wgt"].sum()
        denom = w.groupby(keys, observed=True)["wgt"].transform("sum")
        w = w[denom.to_numpy() > 0].copy()
        w["share"] = w["wgt"] / w.groupby(keys, observed=True)["wgt"].transform("sum")
        bad = float(w.groupby(keys, observed=True)["share"].sum().sub(1.0).abs().max())
        if bad > 1e-10:
            raise SystemExit(f"FAIL: rule {rule} shares do not sum to one at "
                             f"{plen} digits (max deviation {bad:.3e})")
        left = outstanding.groupby(["ccg", "year", "state", "pfx"], observed=True,
                                   as_index=False)["emp"].sum()
        hit = left.merge(w[keys + ["code", "share"]], on=keys, how="inner")
        if hit.empty:
            provenance[rule] = 0.0
            continue
        additions.append(pd.DataFrame({
            "ccg": hit["ccg"].to_numpy(np.int32), "year": hit["year"].to_numpy(np.int16),
            "code": hit["code"].to_numpy(np.int32),
            "emp": hit["emp"].to_numpy() * hit["share"].to_numpy(),
            "rule": np.int8(rule)}))
        provenance[rule] = float((hit["emp"] * hit["share"]).sum())
        outstanding = _drop_served(outstanding,
                                   hit[["ccg", "year", "pfx"]].drop_duplicates(),
                                   ["ccg", "year", "pfx"])

    # Rule 4: a root with no genuine child anywhere in the panel. Sectors 95
    # (Auxiliaries) and 99 (Unclassified) have no four-digit codes at all, so this
    # is where they land. The cell stays rather than being dropped or spread over
    # industries nobody has ever observed.
    provenance[4] = float(outstanding["emp"].sum()) if not outstanding.empty else 0.0
    if not outstanding.empty:
        additions.append(pd.DataFrame({
            "ccg": outstanding["ccg"].to_numpy(np.int32),
            "year": outstanding["year"].to_numpy(np.int16),
            "code": outstanding["code"].to_numpy(np.int32),
            "emp": outstanding["emp"].to_numpy(), "rule": np.int8(4)}))
    return additions, provenance, outstanding


def allocate(cells):
    """Push every residual root down to four digits, three-digit roots first.

    The cascade matters. A sector root like "92----" is split across the sector's
    four-digit codes, and those codes should already carry whatever came to them
    from the three-digit roots inside the same sector, or the coarser residual
    would be spread using weights that are still missing part of their own
    employment. So: three-digit roots first, fold the result into the weights,
    then the sector roots.

    Returns (alloc, provenance) where provenance is keyed by (root digits, rule).
    alloc carries emp, emp_observed, emp_alloc and alloc_rule, where alloc_rule is
    the COARSEST rule that contributed to the cell (-1 if none did). A cell can
    receive from more than one rule, so a single per-row rule would be a fiction;
    the per-rule totals are what to read.
    """
    cells = cells.copy()
    rl = root_len(cells["code"].to_numpy())
    cells["state"] = (cells["ccg"].to_numpy() // 1000).astype(np.int16)

    gen = cells[rl == 4].reset_index(drop=True)
    all_additions, provenance = [], {}

    for plen in (3, 2):
        outstanding = cells[rl == plen].reset_index(drop=True)
        if outstanding.empty:
            continue
        gen = gen.copy()
        gen["pfx"] = prefix_of(gen["code"].to_numpy(), plen)
        outstanding["pfx"] = prefix_of(outstanding["code"].to_numpy(), plen)
        adds, prov, _left = _ladder(gen, outstanding, plen)
        for rule, amount in prov.items():
            provenance[(plen, rule)] = amount
        all_additions += adds

        # Fold this stage's additions into the weights the next stage will use.
        if plen == 3:
            got = (pd.concat(adds, ignore_index=True)
                   .groupby(["ccg", "year", "code"], observed=True, as_index=False)["emp"].sum())
            got = got[got["code"] < RESIDUAL3]
            gen = (pd.concat([gen[["ccg", "year", "code", "emp", "state"]],
                              got.assign(state=(got["ccg"].to_numpy() // 1000).astype(np.int16))],
                             ignore_index=True)
                   .groupby(["ccg", "year", "code", "state"], observed=True,
                            as_index=False)["emp"].sum())

    add = pd.concat(all_additions, ignore_index=True)
    # != 0, not > 0. A residual cell can itself be one of EFSY's tiny negatives,
    # and dropping the negative slice of the allocation instead of distributing it
    # loses exactly that much employment against raw EFSY.
    add = add[add["emp"] != 0]
    add = add.groupby(["ccg", "year", "code"], observed=True, as_index=False).agg(
        emp_alloc=("emp", "sum"), alloc_rule=("rule", "max"))

    observed = cells[rl == 4][["ccg", "year", "code", "emp"]].rename(
        columns={"emp": "emp_observed"})
    out = observed.merge(add, on=["ccg", "year", "code"], how="outer")
    out["emp_observed"] = out["emp_observed"].fillna(0.0)
    out["emp_alloc"] = out["emp_alloc"].fillna(0.0)
    out["alloc_rule"] = out["alloc_rule"].fillna(-1).astype(np.int8)
    out["emp"] = out["emp_observed"] + out["emp_alloc"]
    for col, typ in (("ccg", np.int32), ("year", np.int16), ("code", np.int32)):
        out[col] = out[col].astype(typ)
    return out, provenance


def _drop_served(outstanding, served, keys):
    """Rows of `outstanding` whose key is not in `served`. A merge, not a set of
    twenty million tuples."""
    marked = outstanding.merge(served.assign(_hit=1), on=keys, how="left")
    return outstanding[marked["_hit"].isna().to_numpy()].reset_index(drop=True)


def build_support(cells, universe, mode):
    """The (ccg, code) pairs every year carries a row for, sorted so that the row
    order is identical in every year."""
    if mode == "national":
        codes = np.sort(cells["code"].unique())
        support = pd.DataFrame({"ccg": np.repeat(universe, len(codes)),
                                "code": np.tile(codes, len(universe))})
    else:
        support = cells[["ccg", "code"]].drop_duplicates()
    support = support.sort_values(["ccg", "code"]).reset_index(drop=True)
    support["ccg"] = support["ccg"].astype(np.int32)
    support["code"] = support["code"].astype(np.int32)
    missing = sorted(set(universe) - set(support["ccg"].tolist()))
    return support, missing


def write_year_panels(cells, support, out_dir, years):
    """One parquet per year on the fixed support, zeros materialised."""
    out_dir.mkdir(parents=True, exist_ok=True)
    value_cols = [c for c in ("emp", "emp_observed", "emp_alloc") if c in cells.columns]
    stats, prev = [], None
    for yr in years:
        block = support.copy()
        year_cells = cells[cells["year"] == yr].drop(columns=["year"])
        block = block.merge(year_cells, on=["ccg", "code"], how="left")
        for col in value_cols:
            block[col] = block[col].fillna(0.0)
        if "alloc_rule" in block.columns:
            block["alloc_rule"] = block["alloc_rule"].fillna(-1).astype(np.int8)
        block.insert(2, "year", np.int16(yr))
        block["naics"] = [code_to_label(c) for c in block["code"].to_numpy()]
        block["in_efsy"] = block.get("emp_observed", block["emp"]).to_numpy() > 0
        block["filled_zero"] = block["emp"].to_numpy() == 0

        emp = block["emp"].to_numpy()
        entry = exit_ = cont = 0
        if prev is not None:
            entry = int(((prev == 0) & (emp > 0)).sum())
            exit_ = int(((prev > 0) & (emp == 0)).sum())
            cont = int(((prev > 0) & (emp > 0)).sum())
        prev = emp

        pq.write_table(pa.Table.from_pandas(block, preserve_index=False),
                       out_dir / f"y{yr}.parquet", compression="zstd")
        stats.append(dict(year=yr, rows=len(block), positive=int((emp > 0).sum()),
                          zero=int((emp == 0).sum()), emp=float(emp.sum()),
                          entry=entry, exit=exit_, cont=cont))
    return pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # DEFAULT CHANGED county-own -> national, 2026-08-14, to satisfy
    # PLAN_macro_fiscal_multiplier.md section 2.1: "Cells with no employment have to
    # be present as zeros rather than absent as missing, because their absence is
    # the object of study."
    #
    # Under county-own a county carried rows only for the industries it had EVER
    # reported, so a county that never had a refinery had no refinery row at all
    # rather than a row of zeros. At four digits that left the panel 194,460 pairs a
    # year short of the 3,101 x 303 cross product - 20.7% - and the median county
    # carrying 246 of 303 industries.
    #
    # What this does and does not move. The added cells are zero in EVERY year by
    # construction, so they are empty in both periods at any horizon: zero midpoint
    # denominator, zero weight, no contribution to the reconstruction, not entry
    # (their current employment is zero too), and outside the winsorization cutoff
    # distribution, which takes cells with lagged employment > 0. Every admissible
    # specification is therefore indifferent to them, and county totals are
    # untouched, so 13's check 4 still passes.
    #
    # ONE THING IT DOES MOVE, and it is why this is a flag rather than a rewrite:
    # 14_build_instrument.py weights its cell-rung first stage by n_cells, which
    # under national is uniform (26 and 303) instead of varying by county. The
    # `county x NAICS2` and `county x NAICS4` rows of instrument_first_stage.csv
    # would change if 14 were rerun. The county rung does not use cell counts, so
    # nothing step 2 anchors on moves, and 19/20/23/24/26 read no cell files at all.
    ap.add_argument("--industry-support", choices=("county-own", "national"),
                    default="national",
                    help="which industries each county carries a row for: every "
                         "industry in the panel (default, required by plan section "
                         "2.1) or only the ones it has ever reported")
    # DEFAULT CHANGED keep -> drop, 2026-08-17. EFSY's desuppression artifacts are
    # deleted at the allocated four-digit cell, and the two-digit and county rungs
    # are summed from the cleaned cells so all three stay additive.
    #
    # WHAT IT DOES NOT MOVE. The deletion happens on the allocated frame before
    # build_support, so rectangularisation restores every deleted cell as a zero:
    # 939,603 rows a year at four digits and 303 codes, unchanged, which is what
    # 22_winsor_construction_error.py and 28_table_d1.py rely on when they align a
    # t-h lag as an array shift rather than a join. All 191 codes that carry a
    # negative somewhere are positive in some other county-year, so no code leaves
    # the support - asserted below rather than assumed. 14_build_instrument.py's
    # n_cells is a row count, so it does not move either.
    #
    # WHAT IT DOES MOVE. 1,674 four-digit cells, -134.44 employee-years, 1981-1997.
    # County employment rises in 826 of 136,444 county-years, median 4.9e-6 of the
    # county total and at most 1.03e-2, so 13's county panel and every first stage
    # downstream of 14's emp_growth shift in trailing digits. No county-year crosses
    # zero, so the emp_lag > 0 estimation sample is unchanged.
    #
    # WHAT IT COSTS. Conservation against raw EFSY stops being exact: the panel runs
    # +134.44 employee-years above raw EFSY, 3.1e-8 of it. That is the whole reason
    # this is a flag - --negative-cells keep restores the conserving build - and the
    # reason section [4] asserts the ALLOCATION at zero deviation and accounts for
    # the deletion on its own line instead of folding the two together.
    ap.add_argument("--negative-cells", choices=("drop", "keep"), default="drop",
                    help="EFSY's desuppression artifacts at the allocated four-digit "
                         "cell: delete them (default; they return as zeros when the "
                         "panel is rectangularised) or keep them for exact "
                         "conservation against raw EFSY")
    args = ap.parse_args()
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    failures = []

    print("=" * 78)
    print("11_build_employment_cells.py - the county-by-industry employment cells")
    print("=" * 78)

    ccg_of, universe = load_crosswalk()
    print(f"\nCCG universe                     : {len(universe):,} units, "
          f"{len(ccg_of):,} county codes mapped onto them")
    print(f"Panel window                     : {FIRST_YEAR}-{LAST_YEAR}, {len(years)} years")
    print(f"Industry support                 : {args.industry_support}")

    print("\nreading EFSY ...", flush=True)
    cells, rep = load_cells(ccg_of)
    failures += [("geography", p) for p in rep["problems"]]
    print(f"  observed CCG-year-cell rows    : {len(cells):,}")

    gran = rep["gran"]
    raw_by_year = gran.groupby("year", observed=True)["emp"].sum()
    raw_total = float(raw_by_year.sum())
    kept_total = float(cells["emp"].sum())
    dropped_total = float(rep["dropped"].sum())

    # --- [1] granularity ------------------------------------------------------
    tab = []
    for yr in years:
        g = gran[gran["year"] == yr]
        tot = float(g["emp"].sum())
        lt4 = float(g[g["resolved"] < 4]["emp"].sum())
        six = float(g[g["resolved"] == 6]["emp"].sum())
        drop = float(rep["dropped"].get(yr, 0.0))
        tab.append(dict(year=yr, rows=int(g["rows"].sum()), emp_total=tot, emp_lt4=lt4,
                        share_lt4=100 * lt4 / tot if tot else 0.0,
                        share_6dig=100 * six / tot if tot else 0.0,
                        residual_dropped=drop,
                        share_residual_dropped=100 * drop / tot if tot else 0.0))
    tab = pd.DataFrame(tab)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(dc.LOG_DIR / "efsy_granularity_by_year.csv", index=False)

    print("\n[1] granularity          employment not resolving to four digits, and "
          "the statewide residual")
    print(f"    {'year':>6}{'rows':>10}{'emp (m)':>11}{'<4 digits':>11}{'6 digits':>10}"
          f"{'999 dropped':>13}")
    for r in tab.itertuples():
        if r.year in (1975, 1980, 1990, 1997, 1998, 2006, 2010, 2016, 2017, 2018):
            print(f"    {r.year:>6}{r.rows:>10,}{r.emp_total / 1e6:>11.1f}"
                  f"{r.share_lt4:>10.2f}%{r.share_6dig:>9.1f}%"
                  f"{r.share_residual_dropped:>12.2f}%")
    print(f"    written to {dc.LOG_DIR / 'efsy_granularity_by_year.csv'}")

    # --- [2] geography --------------------------------------------------------
    print("\n[2] geography            the statewide residual and the DC recode")
    for st, moved in sorted(rep["recoded"].items()):
        code5, first, last, _w = dc.EFSY_RESIDUAL_AS_COUNTY[st]
        print(f"    state {st} fipscty=999 -> {code5} in {last - first + 1} years "
              f"{first}-{last}, {moved:,.0f} employee-years")
    print(f"    residual left unplaced         : {dropped_total:,.0f} employee-years "
          f"({100 * dropped_total / raw_total:.2f}% of EFSY)")
    print(f"    and it GROWS: "
          f"{tab.iloc[0].share_residual_dropped:.2f}% in {FIRST_YEAR}, "
          f"{tab[tab.year == 2006].iloc[0].share_residual_dropped:.2f}% in 2006, "
          f"{tab.iloc[-1].share_residual_dropped:.2f}% in {LAST_YEAR}.")
    print(f"    CBP places less and less employment in a county over time, so the "
          f"later the year the")
    print(f"    more of it this panel cannot see. Not a defect here; a property of "
          f"the source.")

    # --- [3] allocation -------------------------------------------------------
    print("\n[3] allocation           pushing the roots down to four digits")
    alloc, prov = allocate(cells)
    rl = root_len(cells["code"].to_numpy())
    residual_total = float(cells.loc[rl < 4, "emp"].sum())
    for plen in (3, 2):
        sub = float(cells.loc[rl == plen, "emp"].sum())
        if not sub:
            continue
        print(f"    {plen}-digit roots{'':<20}{sub:>15,.0f} employee-years")
        for rule in sorted(r for p, r in prov if p == plen):
            amount = prov[(plen, rule)]
            pct = 100 * amount / sub if sub else 0.0
            print(f"      rule {rule}  {ALLOC_RULES[rule]:<41}{amount:>15,.0f}{pct:>8.2f}%")
    print(f"    {'all roots':<33}{residual_total:>15,.0f}")
    outside = sum(v for (p, r), v in prov.items() if r in (2, 3))
    outside_pct = 100 * outside / raw_total if raw_total else 0.0
    print(f"    rules 2+3 as a share of ALL employment{outside_pct:>29.4f}%")
    if outside_pct > OUTSIDE_SHARE_LIMIT_PCT:
        failures.append(("allocation", f"rules 2+3 moved {outside_pct:.4f}% of national "
                                       f"employment, over the "
                                       f"{OUTSIDE_SHARE_LIMIT_PCT}% limit"))
    if abs(sum(prov.values()) - residual_total) > max(1e-3, residual_total * 1e-12):
        failures.append(("allocation", f"allocated {sum(prov.values()):,.2f} against a "
                                       f"residual of {residual_total:,.2f}"))

    # --- [4] conservation against raw EFSY ------------------------------------
    # The ALLOCATION is asserted here, on the pre-deletion frame, and the deletion
    # is accounted for below it. Folding the two together would let a real ladder
    # bug hide inside the deletion's tolerance, which is the whole point of keeping
    # them on separate lines.
    print("\n[4] conservation         allocated cells against RAW EFSY, by year")
    dev = (alloc.groupby("year", observed=True)["emp"].sum()
           - cells.groupby("year", observed=True)["emp"].sum()).abs()
    worst = float(dev.max())
    print(f"    max absolute deviation in any year{worst:>33.6e}   <- the ladder, "
          f"before any deletion")
    print(f"    employment kept                   {kept_total:>25,.0f} "
          f"({100 * kept_total / raw_total:.2f}% of raw EFSY)")
    if worst > 1e-3:
        failures.append(("conservation", f"allocation moved {worst:.6f} employee-years"))

    neg_cells = alloc[alloc["emp"].to_numpy() < 0]
    neg_emp = float(neg_cells["emp"].sum())
    print(f"    negative rows in raw EFSY         {rep['neg_rows']:>25,} "
          f"({rep['neg_total']:,.2f} employee-years)")
    print(f"    negative cells after aggregation  {len(neg_cells):>25,} "
          f"({neg_emp:,.2f} employee-years)")

    # A negative observation offset by a positive allocation leaves a cell whose
    # employment is non-negative. Those are not negative cells and they stay; the
    # number is printed so it is on the record rather than discovered later.
    obs_neg = int(((alloc["emp_observed"].to_numpy() < 0)
                   & (alloc["emp"].to_numpy() >= 0)).sum())
    obs_neg_emp = float(alloc.loc[(alloc["emp_observed"].to_numpy() < 0)
                                  & (alloc["emp"].to_numpy() >= 0),
                                  "emp_observed"].sum())
    print(f"    of which offset to >= 0 by alloc  {obs_neg:>25,} "
          f"({obs_neg_emp:,.2f} observed employee-years) - not negative cells, kept")

    if args.negative_cells == "drop":
        codes_before = set(alloc["code"].unique())
        cty_before = alloc.groupby(["ccg", "year"], observed=True)["emp"].sum()
        alloc = alloc[alloc["emp"].to_numpy() >= 0].reset_index(drop=True)
        codes_after = set(alloc["code"].unique())

        # The load-bearing check. The deletion is safe ONLY because every affected
        # code survives somewhere else, so build_support still returns the same
        # codes and the per-year support and row order do not move. If that ever
        # stops being true the panel silently misaligns downstream, where a t-h lag
        # is an array shift, so it fails here instead.
        if codes_after != codes_before:
            lost = sorted(code_to_label(c) for c in codes_before - codes_after)
            failures.append(("negative cells",
                             f"deleting negatives removed {len(lost)} industries from "
                             f"the panel support: {lost[:10]}"))

        cty_after = alloc.groupby(["ccg", "year"], observed=True)["emp"].sum()
        moved = (cty_after - cty_before.reindex(cty_after.index)).dropna()
        moved = moved[moved.abs() > 0]
        rel = (moved / cty_after.reindex(moved.index)).abs()
        crossed = int(((cty_before.reindex(cty_after.index) <= 0)
                       & (cty_after > 0)).sum())
        print(f"\n    negative cells DELETED (--negative-cells drop). They return as "
              f"zeros when the")
        print(f"    panel is rectangularised, so no row leaves any file and the "
              f"support does not move.")
        print(f"      codes on the support           : {len(codes_after):,} of "
              f"{len(codes_before):,} before - must be equal")
        print(f"      employee-years restored        : {-neg_emp:,.2f} "
              f"({100 * -neg_emp / raw_total:.2e}% of raw EFSY)")
        print(f"      county-years affected          : {len(moved):,} of "
              f"{len(cty_after):,}, {moved.index.get_level_values('ccg').nunique():,} CCGs, "
              f"{int(moved.index.get_level_values('year').min())}-"
              f"{int(moved.index.get_level_values('year').max())}")
        print(f"      relative change in county emp  : median {rel.median():.2e}, "
              f"max {rel.max():.2e}")
        print(f"      county-years crossing zero     : {crossed} - must be 0, or the "
              f"emp_lag > 0 estimation sample changes")
        print(f"    THE COST: conservation against raw EFSY is no longer exact. The "
              f"panel now runs")
        print(f"    {-neg_emp:,.2f} employee-years ABOVE raw EFSY, "
              f"{100 * -neg_emp / raw_total:.2e}% of it, all of it pre-1998. "
              f"Use --negative-cells keep")
        print(f"    for the conserving build. A.3.1 holds in the letter - the cell "
              f"stays in the panel as a")
        print(f"    zero and no baseline unit enters or leaves - but the VALUE was "
              f"changed by hand, and that")
        print(f"    is a declared departure rather than a neutral cleanup.")
        if crossed:
            failures.append(("negative cells",
                             f"{crossed} county-years cross zero, so the emp_lag > 0 "
                             f"estimation sample is not the same one"))
    else:
        print(f"    kept, not dropped (--negative-cells keep): they are data, they are "
              f"3e-7 of the panel,")
        print(f"    and dropping them would put a hole in conservation. Allocation "
              f"weights are clipped at")
        print(f"    zero so no share can go negative.")

    # --- [5] the new-cell audit ----------------------------------------------
    # Only a GENUINE four-digit industry can be a fabricated presence. A rule-4
    # cell is the residual carried under its own label, which invents nothing and
    # would flatter this number badly if it were counted here.
    #
    # Measured on the frame AS WRITTEN, so under `drop` the deleted cells are out of
    # both the numerator and the denominator. That is the right frame for a
    # description of the panel and the wrong one for a description of the ladder; it
    # understates the imputation's footprint by whatever share of the 1,674 deleted
    # cells was purely allocated, which is small but not zero.
    is_genuine = alloc["code"].to_numpy() < RESIDUAL3
    created = alloc[is_genuine
                    & (alloc["emp_observed"].to_numpy() == 0)
                    & (alloc["emp_alloc"].to_numpy() > 0)]
    kept = alloc[~is_genuine]
    print("\n[5] new cells            industries positive ONLY because of allocation")
    print(f"    this is the extensive margin the imputation touched, and the reason "
          f"rules 2 and 3 are capped")
    print(f"    cells                          : {len(created):,} of "
          f"{int(is_genuine.sum()):,} genuine cells "
          f"({100 * len(created) / int(is_genuine.sum()):.3f}%)")
    print(f"    employment in them             : {created['emp'].sum():,.0f} "
          f"({100 * created['emp'].sum() / raw_total:.4f}% of national)")
    for rule, row in created.groupby("alloc_rule")["emp"].agg(["size", "sum"]).iterrows():
        print(f"      via rule {rule}  {ALLOC_RULES[int(rule)]:<41}"
              f"{int(row['size']):>10,} cells {row['sum']:>13,.0f}")
    print(f"    residual cells retained (rule 4){'':<2}{len(kept):>10,} cells "
          f"{kept['emp'].sum():>13,.0f}   not new industries, the root under its own label")

    # --- [6] rectangularise and write ----------------------------------------
    print("\n[6] rectangularise       the county spine, zeros materialised")
    support, missing = build_support(alloc, universe, args.industry_support)
    if missing:
        failures.append(("rectangularise", f"{len(missing)} CCGs have no employment in "
                                           f"any year: {missing[:10]}"))
    print(f"    CCGs on the spine              : {support['ccg'].nunique():,} of "
          f"{len(universe):,} in the universe")
    print(f"    (ccg, industry) pairs          : {len(support):,}, mean "
          f"{len(support) / support['ccg'].nunique():.1f} industries per county")
    print(f"    rows per year                  : {len(support):,}")
    print(f"    panel rows                     : {len(support) * len(years):,}")

    print("    writing naics4_alloc ...", flush=True)
    stats4 = write_year_panels(alloc, support, dc.EMP_OUT_DIR / "naics4_alloc", years)

    # naics4_raw is built from `cells`, NOT from `alloc`, so --negative-cells does
    # not touch it and must not. It is the raw record the ladder is priced against,
    # it is not a rung, and it is estimated on nowhere; cleaning it would destroy
    # the only thing it is for.
    print("    writing naics4_raw ...", flush=True)
    raw_cells = cells[["ccg", "year", "code", "emp"]].copy()
    raw_cells["emp_observed"] = raw_cells["emp"]
    raw_cells["emp_alloc"] = 0.0
    raw_support, _ = build_support(raw_cells, universe, args.industry_support)
    write_year_panels(raw_cells, raw_support, dc.EMP_OUT_DIR / "naics4_raw", years)

    # The two-digit rung. No imputation is involved at two digits because no root
    # is shorter than three, so a residual root carries a sector like any code.
    print("    writing naics2 ...", flush=True)
    two = alloc.copy()
    two["code"] = sector_of(two["code"].to_numpy()).astype(np.int32)
    two = two.groupby(["ccg", "year", "code"], observed=True, as_index=False)[
        ["emp", "emp_observed", "emp_alloc"]].sum()
    two_support, _ = build_support(two, universe, args.industry_support)
    stats2 = write_year_panels(two, two_support, dc.EMP_OUT_DIR / "naics2", years)

    # The county rung, which is S_i.
    county = alloc.groupby(["ccg", "year"], observed=True, as_index=False)["emp"].sum()

    # Both derived rungs are sums over the same `alloc`, so under `drop` neither can
    # carry a negative. Asserted rather than argued: it is the one-line proof that
    # the deletion reached the two-digit rung and the county rung and not just the
    # four-digit cells, which is what the request was.
    if args.negative_cells == "drop":
        for name, frame in (("naics2", two), ("county", county)):
            bad = int((frame["emp"].to_numpy() < 0).sum())
            if bad:
                failures.append(("negative cells",
                                 f"{bad} {name} rows are still negative after the "
                                 f"four-digit deletion"))

    spine = pd.DataFrame({"ccg": np.repeat(universe, len(years)).astype(np.int32),
                          "year": np.tile(np.array(years, np.int16), len(universe))})
    county = spine.merge(county, on=["ccg", "year"], how="left")
    county["emp"] = county["emp"].fillna(0.0)
    county["filled_zero"] = county["emp"].to_numpy() == 0
    dc.EMP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(county, preserve_index=False),
                   dc.EMP_OUT_DIR / "efsy_ccg_year.parquet", compression="zstd")
    print(f"    county rung                    : {len(county):,} CCG-years, "
          f"{int(county['filled_zero'].sum()):,} of them zero")

    # --- [7] zero share -------------------------------------------------------
    print("\n[7] zero share           the extensive margin the disaggregation test turns on")
    print(f"    {'year':>6}{'4-digit rows':>14}{'zero':>9}{'2-digit rows':>14}{'zero':>9}")
    for yr in (1975, 1990, 1997, 1998, 2006, 2016, 2017, 2018):
        a = stats4[stats4.year == yr].iloc[0]
        b = stats2[stats2.year == yr].iloc[0]
        print(f"    {yr:>6}{int(a.rows):>14,}{100 * a.zero / a.rows:>8.1f}%"
              f"{int(b.rows):>14,}{100 * b.zero / b.rows:>8.1f}%")

    # --- [8] entry and exit ---------------------------------------------------
    print(f"\n[8] entry and exit       four-digit cells, adjacent years, "
          f"vintage breaks at {VINTAGE_BREAKS}")
    if args.negative_cells == "drop":
        # entry, exit and continuing all test > 0 or == 0, so a negative cell used to
        # fall into NONE of the three. Once it is a zero it can be an exit, or make
        # its successor an entry. These counts therefore move against the keep build
        # over 1981-1997 by construction, and that is not drift.
        print(f"    (1981-1997 counts differ from --negative-cells keep by "
              f"construction: a negative cell")
        print(f"    was in none of the three classes, and a zero one can be an exit.)")
    print(f"    {'pair':>13}{'entry':>10}{'exit':>10}{'continuing':>12}   note")
    for r in stats4.itertuples():
        if r.year == FIRST_YEAR:
            continue
        note = "VINTAGE BREAK - not economic" if r.year in VINTAGE_BREAKS else ""
        if r.year in (1976, 1990, 1998, 2006, 2016, 2017, 2018) or note:
            print(f"    {r.year - 1}-{r.year:<8}{r.entry:>10,}{r.exit:>10,}"
                  f"{r.cont:>12,}   {note}")

    # --- [9] the spot check ---------------------------------------------------
    print("\n[9] spot check           one county in 1990, to be read by eye")
    cand = cells[(cells["year"] == 1990)
                 & (root_len(cells["code"].to_numpy()) == 3)]
    if not cand.empty:
        tot90 = cells[cells["year"] == 1990].groupby("ccg", observed=True)["emp"].sum()
        pool = tot90[tot90.index.isin(cand["ccg"].unique())]
        target = int((pool - pool.median()).abs().idxmin())
        mine = cand[cand["ccg"] == target]
        root = int(mine.loc[mine["emp"].idxmax(), "code"]) - RESIDUAL3
        print(f"    CCG {target}, root {root}, 1990. County employment "
              f"{tot90.loc[target]:,.0f}")
        rr = cells[(cells["ccg"] == target) & (cells["year"] == 1990)
                   & (prefix_of(cells["code"].to_numpy(), 3) == root)]
        aa = alloc[(alloc["ccg"] == target) & (alloc["year"] == 1990)
                   & (prefix_of(alloc["code"].to_numpy(), 3) == root)]
        print(f"      {'cell':>8}{'raw':>13}{'allocated':>13}{'of which added':>16}")
        for code in sorted(set(rr["code"]) | set(aa["code"])):
            a = aa[aa["code"] == code]
            print(f"      {code_to_label(int(code)):>8}"
                  f"{rr[rr['code'] == code]['emp'].sum():>13,.1f}"
                  f"{a['emp'].sum():>13,.1f}{a['emp_alloc'].sum():>16,.1f}")
        print(f"      {'TOTAL':>8}{rr['emp'].sum():>13,.1f}{aa['emp'].sum():>13,.1f}"
              f"{aa['emp_alloc'].sum():>16,.1f}   <- the two totals must match"
              + (", net of any deleted negative"
                 if args.negative_cells == "drop" else ""))

    stats4["rung"], stats2["rung"] = "naics4", "naics2"
    pd.concat([stats4, stats2]).to_csv(dc.EMP_OUT_DIR / "employment_cells_report.csv",
                                       index=False)
    print(f"\nwrote {dc.EMP_OUT_DIR}")
    print(f"  naics4_alloc/ naics4_raw/ naics2/  y{FIRST_YEAR}.parquet .. y{LAST_YEAR}.parquet")
    print("  efsy_ccg_year.parquet, employment_cells_report.csv")

    sys.stdout.flush()
    if failures:
        print(f"\nFAIL: {len(failures)} checks failed", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
