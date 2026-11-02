"""Stage 8 (construction): the FSC concordance restated on the EFSY axis.

Producer 06 estimates the FSC x NAICS6 concordance on DCADS 2001-06 records, so
it carries that era's classification. Producer 07 builds the vintage-stable
4-digit axis the EFSY employment cells sit on. This producer puts the first onto
the second, writing

    reference/fsc_efsy_axis_concordance.csv    fsc x axis4 dollar shares

which is what producer 23 applies to county dollars.

THE ESTIMATION WINDOW IS TWO VINTAGES, NOT ONE, AND THAT IS WHY THIS PRODUCER
GOES BACK TO THE YEAR-LEVEL CELLS. Coverage against each vintage's six-digit
universe, computed here at run time and written to the diagnostics, says
FY2001-02 records are NAICS 1997 (99.98% covered by it, 88% by NAICS 2002) and
FY2003-06 are NAICS 2002 (100.00%). FY2001-02 is a quarter of the pooled
dollars. The committed concordance pools all six years into one number per
(fsc, naics6) with no year dimension, so a code's dollars in that pool are a
mixture of two classifications.

Resolving the mixture by a precedence rule on the code alone - "valid in 2002,
so call it 2002" - is not good enough, and this is not a hypothetical: 1.61% of
concordance dollars sit on codes that reach a DIFFERENT axis cell depending on
which vintage they are entered at. `213112` is unchanged from 1997 to 2002 but
splits differently downstream; `511110` reaches 5111 one way and 5191 the
other. So each cell's dollars are placed at the vintage of the YEAR THEY WERE
OBSERVED IN, by re-deriving the year-level cells through producer 06's own
`cell_table` - imported, not reimplemented, so the two cannot drift - and the
pooled result is asserted equal to the committed concordance before anything is
converted.

WHY THE CONCORDANCE IS RESTATED AND NOT THE CELLS. The vintage conversion is
linear in dollars, so restating the concordance and restating the ~46M county x
industry cells give the same answer; the first is some 500x cheaper. That is an
argument, not a proof, so the equivalence is asserted on three sample years
against the county panel rather than assumed - see check_cells().

Codes that no vintage recognises are NOT dropped. They go to an explicit
`INVALID` axis cell, so dollars are conserved exactly through the restatement
and the residual is a number someone can look at rather than a silent leak.
Seven codes land there - 023331, 023832, 033421, 033994, 056211, 056291 and
081211, none of which is NAICS in any vintage and all of which look like SIC
leakage into the DCADS `naics` field - carrying 7.4e-7 of the concordance's
dollars between them.

Modes, following 06 and 07:
  --verify (default)  recompute and require byte-identical to the committed
                      reference CSV; writes nothing.
  --write             regenerate the reference CSV and the diagnostics.
"""
import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402

REF_DIR = PIPELINE.parent / "reference"
RESULTS_DIR = PIPELINE.parents[2] / "Results" / "pipeline_diagnostics"
PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"

SOURCE = REF_DIR / "fsc_naics_concordance.csv"
CHAIN = REF_DIR / "naics_to_efsy_axis.csv"
OUT = REF_DIR / "fsc_efsy_axis_concordance.csv"

INVALID = "INVALID"
LOW_SUPPORT_DOLLARS = 1_000_000.0      # the same pin producer 06 uses
CHECK_YEARS = (1970, 1990, 2005)       # one per contract era, for check_cells
# The floor on the cell-equivalence tolerance is the tie-out gap against the
# committed concordance (~3e-10, RTOL_POOL below), because path B starts from
# the committed share numbers and path A from the re-derived ones. It is not
# slack for a real disagreement to hide in.
RTOL_CELLS = 1e-8
RTOL_POOL = 1e-9                       # the tie-out against the committed table

META = [
    "FSC x vintage-stable 4-digit industry axis, dollar shares.",
    "reference/fsc_naics_concordance.csv (FSC x NAICS6, estimated on DCADS "
    "2001-06) restated onto reference/efsy_naics4_axis.csv through "
    "reference/naics_to_efsy_axis.csv.",
    "The estimation window spans two vintages - FY2001-02 is NAICS 1997, "
    "FY2003-06 is NAICS 2002 - so each cell's dollars are converted at the "
    "vintage of the year they were observed in, not at one vintage guessed "
    "for the pooled cell.",
    "Codes no vintage recognises are kept in an explicit INVALID cell so that "
    "dollars are conserved exactly.",
    "n_years counts the years in 2001-2006 contributing positive dollars to "
    "the cell.",
    "Per-FSC shares sum to 1; asserted.",
    "Built by pipeline/08_fsc_concordance_on_efsy_axis.py.",
]


def load_p06():
    """Producer 06's own cell builder, imported rather than reimplemented.

    Its file name starts with a digit, so it cannot be imported by name; the
    point of going through importlib anyway is that the year-level cells here
    are produced by exactly the code that produced the committed concordance,
    and cannot drift from it.
    """
    spec = importlib.util.spec_from_file_location(
        "p06", PIPELINE / "06_fsc_industry_concordance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_chain():
    ch = pd.read_csv(CHAIN, comment="#",
                     dtype={"vintage": int, "naics6": str, "axis4": str})
    universe = {v: set(g["naics6"]) for v, g in ch.groupby("vintage")}
    return ch, universe


def year_vintages(cells, universe, diag):
    """Each estimation year's NAICS vintage, read off the data.

    The vintage a year's records are written in is not recorded anywhere, so it
    is inferred the only way available: whichever vintage's six-digit universe
    covers the most of that year's dollars. The whole coverage matrix goes to
    the diagnostics, not just the winner, so a marginal call is visible.
    """
    out = {}
    for y, g in cells.groupby("year"):
        tot = g["d"].sum()
        cov = {v: g.loc[g["industry"].isin(u), "d"].sum() / tot
               for v, u in universe.items()}
        best = max(cov, key=cov.get)
        out[int(y)] = best
        for v, c in sorted(cov.items()):
            diag.append(("year_vintage_coverage", int(y), v, c))
        diag.append(("year_vintage", int(y), "", best))
        print(f"    FY{y}: NAICS {best} ({cov[best]:.4f} of dollars covered)",
              flush=True)
    return out


def place_rows(cells, yv, universe, diag):
    """Attach a source vintage to every (fsc, industry, year) cell.

    The year's own vintage first. A code the year's vintage does not have is a
    straggler on a long-running record, so the remaining vintages are tried
    nearest-first; anything no vintage has is INVALID.
    """
    vints = sorted(universe)
    v = np.full(len(cells), -1, dtype=np.int64)
    ind = cells["industry"].to_numpy()
    yr = cells["year"].to_numpy()
    for year, home in yv.items():
        order = [home] + sorted((x for x in vints if x != home),
                                key=lambda x: (abs(x - home), x))
        rows = yr == year
        for cand in order:
            todo = rows & (v < 0)
            if not todo.any():
                break
            hit = todo & pd.Series(ind).isin(universe[cand]).to_numpy()
            v[hit] = cand
    cells = cells.assign(vintage=v)
    tot = cells["d"].sum()
    home_hit = cells["vintage"].to_numpy() == cells["year"].map(yv).to_numpy()
    diag.append(("placed_at_year_vintage_dollar_share", "", "",
                 cells.loc[home_hit, "d"].sum() / tot))
    diag.append(("placed_at_fallback_dollar_share", "", "",
                 cells.loc[~home_hit & (cells["vintage"] > 0), "d"].sum() / tot))
    diag.append(("placed_invalid_dollar_share", "", "",
                 cells.loc[cells["vintage"] < 0, "d"].sum() / tot))
    diag.append(("invalid_codes", "", "",
                 " ".join(sorted(cells.loc[cells["vintage"] < 0,
                                           "industry"].unique()))))
    print(f"    placed at the year's own vintage: "
          f"{cells.loc[home_hit, 'd'].sum() / tot:.4%}; fallback "
          f"{cells.loc[~home_hit & (cells['vintage'] > 0), 'd'].sum() / tot:.4%}; "
          f"invalid {cells.loc[cells['vintage'] < 0, 'd'].sum() / tot:.4%}",
          flush=True)
    return cells


def tie_to_committed(cells, src, diag):
    """The re-derived cells must pool to the committed concordance exactly."""
    pooled = (cells.groupby(["fsc", "industry"], as_index=False)["d"].sum()
              .rename(columns={"d": "dollars_new"}))
    pooled = pooled[pooled["dollars_new"] > 0]
    m = src[["fsc", "industry", "dollars"]].merge(
        pooled, on=["fsc", "industry"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), (
        "the re-derived cells and the committed concordance do not have the "
        f"same (fsc, industry) pairs: {m.loc[m['_merge'] != 'both'].head(5)}")
    rel = ((m["dollars_new"] - m["dollars"]).abs()
           / m["dollars"].abs().clip(lower=1.0)).max()
    assert rel < RTOL_POOL, f"pooled dollars differ from the committed table ({rel:.2e})"
    diag.append(("committed_tieout_relgap", "", "", rel))
    print(f"  tie-out to the committed concordance: {len(m):,} pairs, worst "
          f"relative gap {rel:.2e}", flush=True)


def to_axis(cells, ch, diag):
    """(fsc, industry, year, vintage) dollars -> (fsc, axis4) dollars."""
    ok = cells[cells["vintage"] > 0]
    bad = cells[cells["vintage"] < 0]
    m = ok.merge(ch, left_on=["vintage", "industry"],
                 right_on=["vintage", "naics6"], how="left")
    assert m["axis4"].notna().all(), "a placed code failed to reach the axis"
    m["dv"] = m["d"] * m["weight"]
    # Two different drift measures, both dollar-weighted, because they answer
    # different questions. SPLIT is the exposure of the equal-split weighting
    # the unweighted Census tables force on us - dollars whose source code
    # reaches more than one axis cell, and whose division between those cells
    # is therefore an assumption. CHANGED is the exposure of the vintage
    # exercise itself - dollars that end up on a different 4-digit industry
    # than the one their own code names, i.e. what a vintage-blind truncation
    # would have got wrong. Neither is a subset of the other.
    m["split"] = m.groupby(["fsc", "industry", "year"])["axis4"].transform("size") > 1
    m["changed"] = m["axis4"] != m["industry"].str[:4]
    total = cells["d"].sum()
    diag.append(("split_dollar_share", "", "", m.loc[m["split"], "dv"].sum() / total))
    diag.append(("changed_dollar_share", "", "", m.loc[m["changed"], "dv"].sum() / total))
    for v, g in m.groupby("vintage"):
        diag.append(("vintage_pool_dollar_share", v, "", g["dv"].sum() / total))
        diag.append(("vintage_split_dollar_share", v, "",
                     g.loc[g["split"], "dv"].sum() / g["dv"].sum()))
        diag.append(("vintage_changed_dollar_share", v, "",
                     g.loc[g["changed"], "dv"].sum() / g["dv"].sum()))
    print(f"  drift: {m.loc[m['split'], 'dv'].sum() / total:.4%} of dollars split "
          f"across cells, {m.loc[m['changed'], 'dv'].sum() / total:.4%} land on a "
          f"different 4-digit industry than their own code names", flush=True)
    keep = ["fsc", "industry", "axis4", "year", "dv"]
    rows = pd.concat([m[keep],
                      bad.assign(axis4=INVALID, dv=bad["d"])[keep]],
                     ignore_index=True)
    assert np.isclose(rows["dv"].sum(), cells["d"].sum(), rtol=1e-12), \
        "the restatement must conserve concordance dollars"
    return rows


def build_table(rows, diag):
    cy = rows.groupby(["fsc", "axis4", "year"], as_index=False)["dv"].sum()
    tab = (cy.groupby(["fsc", "axis4"], as_index=False)
           .agg(dollars=("dv", "sum"), n_years=("dv", lambda x: int((x > 0).sum()))))
    tab = tab[tab["dollars"] > 0].reset_index(drop=True)
    tot = tab.groupby("fsc")["dollars"].transform("sum")
    tab["share"] = tab["dollars"] / tot
    assert (tab["share"] >= 0).all() and (tab["share"] <= 1).all()
    assert np.allclose(tab.groupby("fsc")["share"].sum().to_numpy(), 1.0, atol=1e-9), \
        "per-category shares must sum to 1"
    top = (tab.sort_values(["share", "axis4"], ascending=[False, True])
           .drop_duplicates("fsc")[["fsc", "axis4", "share"]]
           .rename(columns={"axis4": "top1_axis4", "share": "top1_share"}))
    tab = tab.merge(top, on="fsc", how="left")
    tab["is_construction"] = tab["fsc"].str.startswith("Y")
    tab["low_support"] = ((tab["dollars"] < LOW_SUPPORT_DOLLARS)
                          | (tab["n_years"] <= 1))
    tab = tab.sort_values(["fsc", "axis4"]).reset_index(drop=True)
    diag.append(("axis_cells_used", "", "", tab["axis4"].nunique()))
    diag.append(("fsc_keys", "", "", tab["fsc"].nunique()))
    diag.append(("pairs", "", "", len(tab)))
    print(f"  restated: {tab['fsc'].nunique():,} FSC keys, {len(tab):,} pairs, "
          f"{tab['axis4'].nunique():,} axis cells", flush=True)
    return tab


def report_exposure(src, universe, diag):
    """How far the source is from the base vintage, at six and at four digits.

    Two different questions with two different answers, so two names. Six
    digits is the exposure of the chain, which operates at six digits. Four
    digits is what would visibly break if the chain were skipped and codes were
    simply truncated and matched - the smaller number, and quoting it alone
    would understate how much work the chain is doing.
    """
    total = src["dollars"].sum()
    base6 = universe[2012]
    base4 = {c[:4] for c in base6}
    off6 = ~src["industry"].isin(base6)
    off4 = ~src["industry"].str[:4].isin(base4)
    diag.append(("off_base_vintage_dollar_share_naics6", "", "",
                 src.loc[off6, "dollars"].sum() / total))
    diag.append(("off_base_vintage_dollar_share_naics4", "", "",
                 src.loc[off4, "dollars"].sum() / total))
    diag.append(("off_base_vintage_naics4_codes", "", "",
                 " ".join(sorted({c[:4] for c in src.loc[off4, "industry"]}))))
    print(f"  off the base vintage: {src.loc[off6, 'dollars'].sum() / total:.4%} "
          f"of dollars at six digits, "
          f"{src.loc[off4, 'dollars'].sum() / total:.4%} at four", flush=True)


def check_cells(tab, rows, src, diag):
    """Restating the CONCORDANCE equals restating the CELLS, on sample years.

    Path A pushes county x FSC dollars through the restated table. Path B
    pushes them through the ORIGINAL NAICS6 shares and converts the resulting
    NAICS6 cells with the per-(fsc, naics6) vintage mixture. Same object, two
    orders of operation; if they ever disagree, the cheap path is not the one
    to keep.
    """
    # Path B never touches `tab`. It takes the COMMITTED per-(fsc, naics6)
    # shares and converts each NAICS6 cell with that pair's own vintage
    # mixture, so the two paths differ in the order they pool and in which
    # share numbers they start from.
    pair = rows.groupby(["fsc", "industry", "axis4"], as_index=False)["dv"].sum()
    pair["w"] = pair["dv"] / pair.groupby(["fsc", "industry"])["dv"].transform("sum")
    b_map = (src[["fsc", "industry", "share"]]
             .merge(pair[["fsc", "industry", "axis4", "w"]],
                    on=["fsc", "industry"], how="inner")
             .assign(bw=lambda x: x["share"] * x["w"])
             .groupby(["fsc", "axis4"], as_index=False)["bw"].sum())
    a_map = tab[["fsc", "axis4", "share"]]

    worst = 0.0
    for y in CHECK_YEARS:
        cells = pd.read_parquet(PIPE / "contracts_category_ccg_year.parquet",
                                filters=[("year", "==", y)])
        cells["dollars"] = cells["dollars"].astype(np.float64)

        def push(mp, col):
            e = cells.merge(mp, left_on="cat", right_on="fsc")
            e["d"] = e["dollars"] * e[col]
            return e.groupby(["rule", "ccg", "axis4"], as_index=False)["d"].sum()

        # Outer-joined rather than positionally compared: a cell that one path
        # produces and the other does not is a real disagreement worth seeing
        # as a number, not an unequal-length crash with no magnitude attached.
        j = push(a_map, "share").merge(push(b_map, "bw"), on=["rule", "ccg", "axis4"],
                                       how="outer", suffixes=("_a", "_b")).fillna(0.0)
        gap = (j["d_a"] - j["d_b"]).abs()
        rel = gap.max() / max(j["d_a"].abs().max(), 1.0)
        worst = max(worst, rel)
        assert rel < RTOL_CELLS, \
            f"{y}: restating the concordance != restating the cells ({rel:.2e})"
        diag.append(("cell_equivalence_relgap", y, "", rel))
    print(f"  concordance-vs-cells equivalence: worst relative gap {worst:.2e} "
          f"over {CHECK_YEARS} (tolerance {RTOL_CELLS:g})", flush=True)


def serialize(tab):
    buf = io.StringIO()
    for line in META:
        buf.write(f"# {line}\n")
    cols = ["fsc", "axis4", "dollars", "share", "n_years", "is_construction",
            "low_support", "top1_axis4", "top1_share"]
    tab[cols].to_csv(buf, index=False, float_format="%.10g")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate the reference CSV and the diagnostics "
                         "(default is --verify: recompute and byte-compare)")
    args = ap.parse_args()
    t0 = time.time()
    diag = []
    src = pd.read_csv(SOURCE, comment="#", dtype={"fsc": str, "industry": str})
    ch, universe = load_chain()
    report_exposure(src, universe, diag)

    p06 = load_p06()
    cells = p06.cell_table(p06.NAICS_POOL, "naics", "naics", [])
    tie_to_committed(cells, src, diag)
    yv = year_vintages(cells, universe, diag)
    cells = place_rows(cells, yv, universe, diag)
    rows = to_axis(cells, ch, diag)
    tab = build_table(rows, diag)
    check_cells(tab, rows, src, diag)

    text = serialize(tab)
    if args.write:
        OUT.write_text(text)
        print(f"  wrote {OUT.name}  ({len(tab):,} rows)")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dd = pd.DataFrame(diag, columns=["metric", "key1", "key2", "value"])
        dd.to_csv(RESULTS_DIR / "fsc_axis_restatement.csv", index=False)
        print(f"  wrote {RESULTS_DIR / 'fsc_axis_restatement.csv'}  ({len(dd)} rows)")
    else:
        if not OUT.exists():
            raise AssertionError(f"{OUT.name} is not committed; run --write first")
        committed = OUT.read_text()
        if text != committed:
            a, b = text.splitlines(), committed.splitlines()
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    raise AssertionError(
                        f"{OUT.name} differs from the recompute at line {i+1}:\n"
                        f"  recomputed: {x[:120]}\n  committed:  {y[:120]}")
            raise AssertionError(f"{OUT.name}: length differs ({len(a)} vs {len(b)} lines)")
        print(f"  {OUT.name}: recompute is byte-identical to the committed table")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
