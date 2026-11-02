"""WS1 step 2, part A: restate the EFSY four-digit cells onto the vintage-stable axis.

WHY THIS EXISTS. Producer 11 writes `naics4_alloc` on a 303-code UNION support: it
rectangularises on every code the panel carries in any year, so `4521, 4529` (NAICS
2012) and `4522, 4523` (NAICS 2017) are all present as separate cells in every year,
and likewise `5171, 5172` against `5173`. EFSY carries NAICS 2012 for 1975-2016 and
NAICS 2017 for 2017-2018, so within any single year only one side of each pair is
live and the other is a structural zero. Employment then moves wholesale across the
2016->2017 boundary:

    year   4521      4529        4522     4523       5171     5172     5173
    2016   969,811   1,956,960   0        0          749,611  242,696  0
    2017   0         0           437,033  1,778,652  0        0        918,402

A cell growth rate spanning that boundary is +/-2 by classification rather than by
economics. Producer 07's own per-year check does NOT catch this, and correctly so: it
tests each year's LIVE codes against THAT year's vintage, and every year passes. It is
the union support, not any single year, that mixes vintages.

WHAT THIS DOES. Sums `emp` (and the two provenance columns) within (ccg, year, axis4)
using the axis 07 derived, writing a new rung `naics4_axis` alongside `naics4_alloc`.
The existing rung is NOT touched, so nothing that reads it moves.

    4521 + 4529 (2012) == 4522 + 4523 (2017)  ->  452X
    5171 + 5172 (2012) == 5173        (2017)  ->  517X

Both groups are closed in the Census 2012->2017 concordance - no mapping leaks in or
out - so the collapse is lossless, and it is a MERGE of two cells into one, never a
deletion of either. INVARIANCE_TEST_PROTOCOL.md A.3.1 is therefore not in play: no
cell is dropped, and county totals are conserved exactly (asserted below at 1e-12).

EFSY's six residual roots (111R, 112R, 114R, 922R, 95R, 99R) are NOT on the 310-cell
axis, because they are EFSY's own unclassifiable buckets rather than NAICS industries.
They pass through unchanged as their own cells and are written into the declared
support file, so the support is stated rather than inferred. Contract dollars never
land on them, which is correct: a county has no procurement exposure to "could not be
classified".

Run with no arguments to build; --dry-run prints the report and writes nothing.
Output: <EMP_OUT_DIR>/naics4_axis/y<year>.parquet, one per year, plus
        Results/pipeline_diagnostics/efsy_axis_restatement.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402

REFERENCE = PIPELINE.parent / "reference"
AXIS_CSV = REFERENCE / "efsy_naics4_axis.csv"
SUPPORT_CSV = REFERENCE / "efsy_axis_support.csv"
DIAG = PIPELINE.parents[2] / "Results" / "pipeline_diagnostics"

SRC_RUNG = "naics4_alloc"
OUT_RUNG = "naics4_axis"

FIRST_YEAR, LAST_YEAR = 1975, 2018
VINTAGE_BREAK = 2017          # first year EFSY reports on NAICS 2017

# The two groups the 2012->2017 revision rearranges, as 07 derives them. Asserted
# against the axis file rather than trusted, so a rebuild of 07 that changed the
# merge set fails here instead of silently restating employment a new way.
EXPECTED_MERGES = {
    "452X": {"2012": {"4521", "4529"}, "2017": {"4522", "4523"}},
    "517X": {"2012": {"5171", "5172"}, "2017": {"5173"}},
}

# EFSY's own unclassifiable roots. Not NAICS industries and not on the axis; they
# pass through as themselves.
EFSY_ROOTS = ("111R", "112R", "114R", "922R", "95R", "99R")

RTOL_TOTALS = 1e-12


def log(msg):
    print(msg, flush=True)


def load_axis():
    """(naics4 -> axis4) per vintage, from 07's committed reference."""
    ax = pd.read_csv(AXIS_CSV, comment="#", dtype=str)
    assert list(ax.columns) == ["vintage", "naics4", "axis4"], ax.columns.tolist()
    merges = {}
    for axis4, grp in ax[ax["naics4"] != ax["axis4"]].groupby("axis4"):
        merges[axis4] = {v: set(g["naics4"]) for v, g in grp.groupby("vintage")}
    assert merges == EXPECTED_MERGES, (
        "the axis file's merge set is not the one this producer was written for; "
        f"found {merges}, expected {EXPECTED_MERGES}")
    # One map over both vintages. A source code that appears at both vintages must
    # reach the same axis cell either way, or the collapse would not be well defined.
    m = {}
    for _, r in ax.iterrows():
        prev = m.setdefault(r["naics4"], r["axis4"])
        assert prev == r["axis4"], (r["naics4"], prev, r["axis4"])
    return m, merges


def axis_code(label, src_codes):
    """A stable int32 `code` for an axis cell.

    The source `code` column is producer 11's integer encoding, and a merged cell
    has no single source code. Merged cells take the SMALLEST member code (452X ->
    4521, 517X -> 5171), which keeps the column unique, sorted and int32 while the
    `naics` label carries the real identity. Unmerged cells keep their own code.
    """
    return int(min(src_codes))


def build_support(src_years, axis_map):
    """The declared axis support: one row per axis cell, with its members."""
    seen = {}
    for lab, code in src_years:
        target = axis_map.get(lab, lab)
        seen.setdefault(target, set()).add((lab, code))
    rows = []
    for target in sorted(seen):
        members = sorted(seen[target])
        rows.append(dict(
            axis4=target,
            code=axis_code(target, [c for _, c in members]),
            n_members=len(members),
            members=" ".join(lab for lab, _ in members),
            is_merged=len(members) > 1,
            is_efsy_root=target in EFSY_ROOTS,
            on_axis=target not in EFSY_ROOTS,
        ))
    sup = pd.DataFrame(rows)
    assert sup["code"].is_unique, "axis code collision"
    return sup.sort_values("code").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report and assert, write nothing")
    args = ap.parse_args()

    t0 = time.time()
    src_dir = dc.EMP_OUT_DIR / SRC_RUNG
    out_dir = dc.EMP_OUT_DIR / OUT_RUNG
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    missing = [y for y in years if not (src_dir / f"y{y}.parquet").exists()]
    assert not missing, f"{SRC_RUNG} is missing years: {missing}"

    axis_map, merges = load_axis()
    log(f"axis: {len(set(axis_map.values()))} cells, "
        f"{len(merges)} merged groups {sorted(merges)}")

    # ---- pass 1: the source support, and the checks that do not need the values
    first = pd.read_parquet(src_dir / f"y{FIRST_YEAR}.parquet",
                            columns=["ccg", "code", "naics"])
    src_pairs = sorted(set(zip(first["naics"], first["code"].astype(int))))
    src_units = np.array(sorted(first["ccg"].unique()))
    log(f"source rung: {len(src_pairs)} codes x {len(src_units)} counties")

    off_axis = sorted({lab for lab, _ in src_pairs
                       if lab not in axis_map and lab not in EFSY_ROOTS})
    assert not off_axis, (
        "EFSY reports codes that are neither on the axis nor declared EFSY roots: "
        f"{off_axis}")

    support = build_support(src_pairs, axis_map)
    n_merged = int(support["is_merged"].sum())
    log(f"axis rung support: {len(support)} cells "
        f"({n_merged} merged, {int(support['is_efsy_root'].sum())} EFSY roots)")

    # ---- pass 2: restate year by year, asserting as we go
    diag, pair_rows = [], []
    value_cols = ["emp", "emp_observed", "emp_alloc"]
    keep = support[["axis4", "code"]].rename(columns={"code": "axis_code"})

    for yr in years:
        d = pd.read_parquet(src_dir / f"y{yr}.parquet")
        src_total_by_ccg = d.groupby("ccg")["emp"].sum()
        d["axis4"] = d["naics"].map(lambda s: axis_map.get(s, s))

        # alloc_rule is per-source-cell provenance. It is well defined on the 291
        # cells that do not merge and ambiguous on the two that do, so a merged
        # cell reports -1 rather than an arbitrary member's value.
        agg = {c: "sum" for c in value_cols if c in d.columns}
        g = d.groupby(["ccg", "axis4"], as_index=False).agg(
            **{c: (c, "sum") for c in agg},
            n_src=("naics", "size"),
        )
        if "alloc_rule" in d.columns:
            ar = d.groupby(["ccg", "axis4"])["alloc_rule"].agg(
                lambda s: s.iloc[0] if s.nunique() == 1 else -1).reset_index()
            g = g.merge(ar, on=["ccg", "axis4"], how="left")

        g = g.merge(keep, on="axis4", how="left")
        assert g["axis_code"].notna().all(), "an axis cell has no declared code"
        g = g.rename(columns={"axis_code": "code"})
        g["code"] = g["code"].astype(np.int32)

        # --- assertion 1: county totals are conserved to the last digit
        new_total_by_ccg = g.groupby("ccg")["emp"].sum()
        aligned = src_total_by_ccg.reindex(new_total_by_ccg.index)
        assert set(new_total_by_ccg.index) == set(src_total_by_ccg.index), (
            f"{yr}: county set changed under restatement")
        num = np.abs(new_total_by_ccg.to_numpy() - aligned.to_numpy())
        den = np.maximum(np.abs(aligned.to_numpy()), 1.0)
        worst = float((num / den).max())
        assert worst <= RTOL_TOTALS, f"{yr}: county totals moved, rel {worst:.3e}"

        # --- assertion 2: the support is the full cross product, every year
        assert len(g) == len(src_units) * len(support), (
            f"{yr}: {len(g)} rows against the "
            f"{len(src_units)}x{len(support)} cross product")

        # --- assertion 3: no negative cells (the --negative-cells drop rebuild)
        neg = int((g["emp"] < 0).sum())
        assert neg == 0, f"{yr}: {neg} negative cells survived into the axis rung"

        g["year"] = np.int16(yr)
        g["naics"] = g["axis4"]
        g["in_efsy"] = g.get("emp_observed", g["emp"]).to_numpy() > 0
        g["filled_zero"] = g["emp"].to_numpy() == 0
        cols = (["ccg", "code", "year"] + [c for c in value_cols if c in g.columns]
                + (["alloc_rule"] if "alloc_rule" in g.columns else [])
                + ["naics", "in_efsy", "filled_zero"])
        block = g.sort_values(["ccg", "code"])[cols].reset_index(drop=True)

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(block, preserve_index=False),
                           out_dir / f"y{yr}.parquet", compression="zstd")

        emp = block["emp"].to_numpy()
        diag.append(dict(year=yr, rows=len(block), cells=len(support),
                         positive=int((emp > 0).sum()), zero=int((emp == 0).sum()),
                         emp=float(emp.sum()), max_rel_dev=worst))

        # the merged groups' national totals, before and after, for the continuity check
        nat_src = d.groupby("naics")["emp"].sum()
        nat_new = g.groupby("axis4")["emp"].sum()
        row = dict(year=yr)
        for axis4, byv in merges.items():
            for v, members in byv.items():
                row[f"{axis4}_{v}"] = float(sum(nat_src.get(m, 0.0) for m in members))
            row[axis4] = float(nat_new.get(axis4, 0.0))
        pair_rows.append(row)
        if yr % 10 == 0 or yr in (VINTAGE_BREAK - 1, VINTAGE_BREAK, LAST_YEAR):
            log(f"  {yr}: {len(block):,} rows, emp {emp.sum():,.0f}, "
                f"rel dev {worst:.2e}")

    pairs = pd.DataFrame(pair_rows).set_index("year")

    # --- assertion 4: the continuity check that motivates the whole producer.
    #
    # THE TEST IS A PERCENTILE, NOT A THRESHOLD, AND THE REASON MATTERS. EFSY has a
    # LARGE PANEL-WIDE BREAK AT 2017 that has nothing to do with the NAICS revision:
    # national employment falls 11.5% (121.5M -> 107.5M) and the MEDIAN four-digit
    # cell falls 22%, with the 5th percentile at -79%. An absolute continuity
    # threshold would therefore be testing EFSY's 2017 coverage break, which the
    # collapse cannot fix and should not be blamed for.
    #
    # What the collapse IS responsible for is the CLASSIFICATION break, and that has
    # a sharp signature: before the collapse, 4521, 4529, 5171 and 5172 are the ONLY
    # FOUR CELLS IN THE WHOLE 289-CELL PANEL whose 2016->2017 change is exactly
    # -100%, because their employment does not fall, it moves to a differently
    # numbered cell. So the test is that after the collapse the merged cells are
    # UNREMARKABLE within the cross-sectional distribution of every cell's jump.
    # They are: 452X lands at the 47th percentile and 517X at the 71st.
    log("")
    log(f"continuity across the {VINTAGE_BREAK-1}->{VINTAGE_BREAK} vintage break")
    log("  (judged against the cross-section of all cells, because EFSY has a "
        "panel-wide break here)")

    nat_prev = pd.read_parquet(src_dir / f"y{VINTAGE_BREAK-1}.parquet",
                               columns=["naics", "emp"]).groupby("naics")["emp"].sum()
    nat_brk = pd.read_parquet(src_dir / f"y{VINTAGE_BREAK}.parquet",
                              columns=["naics", "emp"]).groupby("naics")["emp"].sum()
    src_jump = (nat_brk / nat_prev - 1.0)[nat_prev > 0].dropna()
    axis_prev = pairs.loc[VINTAGE_BREAK - 1]
    axis_brk = pairs.loc[VINTAGE_BREAK]
    # the collapsed panel's own cross-section of jumps
    all_prev, all_brk = {}, {}
    for lab, _ in src_pairs:
        tgt = axis_map.get(lab, lab)
        all_prev[tgt] = all_prev.get(tgt, 0.0) + float(nat_prev.get(lab, 0.0))
        all_brk[tgt] = all_brk.get(tgt, 0.0) + float(nat_brk.get(lab, 0.0))
    ap_ = pd.Series(all_prev)
    ab_ = pd.Series(all_brk)
    axis_jump = (ab_ / ap_ - 1.0)[ap_ > 0].dropna()

    total_break = float(nat_brk.sum() / nat_prev.sum() - 1.0)
    log(f"  panel-wide: national employment {nat_prev.sum():,.0f} -> "
        f"{nat_brk.sum():,.0f} ({total_break:+.4f}); "
        f"median cell {src_jump.median():+.4f}")

    exact = sorted(src_jump.index[src_jump <= -0.99999])
    members_all = sorted({m for byv in merges.values()
                          for ms in byv.values() for m in ms
                          if m in src_jump.index})
    log(f"  cells at exactly -100% BEFORE the collapse: {exact}")
    assert set(exact) <= set(members_all), (
        "a cell outside the merged groups vanishes entirely across the vintage "
        f"break: {sorted(set(exact) - set(members_all))}. That is a second "
        "classification break this producer does not handle.")

    breaks = {}
    for axis4 in merges:
        for v, members in merges[axis4].items():
            a = pairs[f"{axis4}_{v}"].loc[VINTAGE_BREAK - 1]
            b = pairs[f"{axis4}_{v}"].loc[VINTAGE_BREAK]
            log(f"  {axis4} members @{v} ({' '.join(sorted(members))}): "
                f"{a:,.0f} -> {b:,.0f}")
        jump = float(axis_jump[axis4])
        pct = 100.0 * float((axis_jump < jump).mean())
        breaks[axis4] = dict(jump=jump, pct=pct)
        log(f"  {axis4} collapsed: {axis_prev[axis4]:,.0f} -> "
            f"{axis_brk[axis4]:,.0f}  jump {jump:+.4f}, "
            f"percentile {pct:.1f}% of the {len(axis_jump)}-cell cross-section")
        # Unremarkable means inside the bulk. A merged cell still sitting in the
        # extreme lower tail would mean the collapse did not actually reunite the
        # employment, which is the failure this producer exists to prevent.
        assert pct >= 5.0, (
            f"{axis4} still sits at the {pct:.1f}th percentile of the 2017 jump "
            "distribution after the collapse; the classification break survived")
        assert jump > -0.99, (
            f"{axis4} still loses ~all employment across the break ({jump:+.4f})")

    # --- assertion 5: the collapsed cell is exactly the sum of ALL its members,
    # every year, both vintages at once.
    #
    # It is NOT asserted that only one vintage's members are live in any given year.
    # EFSY leaks a little across the pair: 5173 (a 2017 code) carries employment in
    # 1975-1997, peaking at 38.4 employee-years, 6.1e-5 of the cell, and 4522/4523
    # carry a rounding trace in nine early years. That is EFSY's own allocation
    # noise, it is three orders of magnitude below anything that matters, and the
    # collapse absorbs it correctly by summing everything. The leak is reported
    # rather than asserted away.
    leaks = {}
    for axis4, byv in merges.items():
        allm = pairs[[f"{axis4}_{v}" for v in byv]].sum(axis=1)
        assert np.allclose(allm, pairs[axis4], rtol=RTOL_TOTALS), (
            f"{axis4}: members do not sum to the collapsed cell")
        off = []
        for yr in pairs.index:
            v_live = "2012" if yr < VINTAGE_BREAK else "2017"
            other = sum(pairs.loc[yr, f"{axis4}_{v}"] for v in byv if v != v_live)
            tot_ = pairs.loc[yr, axis4]
            off.append(other / tot_ if tot_ > 0 else 0.0)
        leaks[axis4] = float(np.max(off))
    log("  merged-member sums reproduce the collapsed cell exactly, every year")
    log("  cross-vintage leak (EFSY allocation noise, absorbed by the collapse): "
        + ", ".join(f"{k} max {v:.2e}" for k, v in leaks.items()))

    # ---- outputs
    st = pd.DataFrame(diag)
    out_rows = [dict(metric="src_rung", key1=SRC_RUNG, key2="", value=len(src_pairs)),
                dict(metric="axis_rung", key1=OUT_RUNG, key2="", value=len(support)),
                dict(metric="counties", key1="", key2="", value=len(src_units)),
                dict(metric="merged_cells", key1="", key2="", value=n_merged),
                dict(metric="efsy_roots", key1="", key2="",
                     value=int(support["is_efsy_root"].sum())),
                dict(metric="max_rel_dev_any_year", key1="", key2="",
                     value=float(st["max_rel_dev"].max()))]
    for axis4, b in breaks.items():
        out_rows += [dict(metric="break_jump", key1=axis4, key2="", value=b["jump"]),
                     dict(metric="break_jump_percentile", key1=axis4, key2="",
                          value=b["pct"])]
    # The panel-wide 2017 break, recorded as its own finding. It is NOT the NAICS
    # revision and this producer does not address it; see the caveat in the README.
    out_rows += [
        dict(metric="efsy_2017_break_national", key1="", key2="", value=total_break),
        dict(metric="efsy_2017_break_median_cell", key1="", key2="",
             value=float(src_jump.median())),
        dict(metric="efsy_2017_break_p05_cell", key1="", key2="",
             value=float(src_jump.quantile(0.05))),
        dict(metric="efsy_2017_break_p95_cell", key1="", key2="",
             value=float(src_jump.quantile(0.95))),
        dict(metric="efsy_2017_cells_at_minus100_precollapse", key1="", key2="",
             value=" ".join(exact)),
    ]
    for axis4, v in leaks.items():
        out_rows.append(dict(metric="cross_vintage_leak_max", key1=axis4, key2="",
                             value=v))
    for _, r in st.iterrows():
        out_rows.append(dict(metric="emp_by_year", key1=int(r["year"]), key2="",
                             value=r["emp"]))

    if not args.dry_run:
        DIAG.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(DIAG / "efsy_axis_restatement.csv", index=False)
        pairs.reset_index().to_csv(DIAG / "efsy_axis_merged_pairs.csv", index=False)
        with open(SUPPORT_CSV, "w") as fh:
            fh.write("# The declared support of the naics4_axis employment rung, "
                     "built by pipeline/24_efsy_axis_rung.py.\n")
            fh.write("# axis4 cells come from reference/efsy_naics4_axis.csv; the "
                     "*R rows are EFSY's own unclassifiable roots, which are not\n")
            fh.write("# NAICS industries and are not on that axis. `code` is the "
                     "int32 key; a merged cell takes its smallest member's code.\n")
            support.to_csv(fh, index=False)
        log(f"\nwrote {OUT_RUNG}/y{FIRST_YEAR}..y{LAST_YEAR}.parquet, "
            f"{SUPPORT_CSV.name}, and 2 diagnostics")
    else:
        log("\n--dry-run: nothing written")

    log(f"\nsupport: {len(src_pairs)} source codes -> {len(support)} axis cells "
        f"({len(support) - int(support['is_efsy_root'].sum())} on the axis + "
        f"{int(support['is_efsy_root'].sum())} EFSY roots)")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
