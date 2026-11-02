"""Stage 23: county x industry contract dollars on the EFSY axis.

WS1 step 1. The composition instrument is built on a contract-category (FSC)
axis; this producer relabels the same dollars onto an industry axis so the
instrument can be rebuilt there. It creates no shock and no estimand - it moves
dollars between labels and asserts that it neither makes nor destroys any.

    contracts_category_ccg_year.parquet        county x FSC x year, real $
              |
              |  FY1966-2006: reference/fsc_efsy_axis_concordance.csv
              |               (producer 08, already on the EFSY axis)
              |  FY2007-2018: the record's OWN naics_code, pushed through
              |               reference/naics_to_efsy_axis.csv at the fiscal
              |               year's own vintage
              v
    industry_cells_naics4.parquet               county x axis4 x year, real $

THE ERA RULE. USAspending records carry an industry code; DCADS records before
FY2001 do not, which is the whole reason the concordance exists. Where the
record has its own code, the record wins: FY2007-18 is assigned natively. The
concordance is still applied to FY2007-18 as well, into a separate cross-check
file, because agreement between the two on the years where both are available
is the only direct evidence about the years where only one is.

NOTHING IS EVER DROPPED FOR BEING UNMAPPABLE. Three sentinel cells carry what
cannot be placed, so county x year totals are conserved exactly against stage 1
and every residual is a number in the file rather than a silent leak:

    UNMAPPED   the FSC category is not a key in the concordance (18 of the 800
               harmonized categories; namespace-era codes absent from the
               concordance's FY2001-06 estimation window)
    INVALID    the NAICS code is recognised by no vintage (producer 08)
    NA__       a native record whose naics_code is missing or malformed

TWO ALLOCATIONS ARE WRITTEN, NOT ONE. WS1 lists probabilistic-share versus
modal-industry assignment as a decision for Adrien, so both are emitted per
cell - `dollars_prob` spreads a category over its industries by the
concordance shares, `dollars_modal` puts all of it on the category's modal
industry - and they share county x year totals by construction. Natively
assigned rows get the same treatment, because a record names one CODE and an
older-vintage code can still split across several cells of the current
classification; there the modal column puts the record whole on its code's
largest destination.

Outputs, all LOCAL and gitignored, under 4_Macro_version/data/temp/industry_axis/:
    industry_cells_naics4.parquet       source, rule, ccg, year, axis4,
                                        dollars_prob, dollars_modal
    industry_cells_sic4.parquet         the SIC-1987 companion, same schema
    industry_cells_crosscheck.parquet   concordance FY2007-18 and native
                                        FY2001-06, for the drift tests
    coverage_by_year.csv                the dollar-coverage assertion, by year
    uncovered_by_category.csv           which categories the residual is on

They are local because they are an intermediate: the copy that matters gets
made when the instrument is built on top of them, and rebuilding them on every
run is waste. The directory mirrors the Dropbox path exactly, so that copy is
a copy and not a path rewrite.

  --target naics|sic|both   default both
  --dry-run                 coverage report only; writes nothing
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
import contracts  # noqa: E402
import harmonize as hz  # noqa: E402
import panel  # noqa: E402

REF_DIR = PIPELINE.parent / "reference"
PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"
OUT_DIR = PIPELINE.parents[2] / "data" / "temp" / "industry_axis"

UNMAPPED, INVALID, NA = "UNMAPPED", "INVALID", "NA__"
RULES = (("net", "net"), ("positive", "pos"))
RTOL_CONSERVE = 1e-6      # within a pass: cells against the dollars that fed
                          # them, in the same precision
# Across passes, the native side is recomputed in float64 while the stage-1
# panel is STORED float32, so the floor here is that storage, measured at up to
# 1.7e-6 on the overlap years. It is a precision floor, not slack for a real
# disagreement: the pre-fix gaps this caught were 6e-3 and 7.5e-5.
RTOL_CROSS_SOURCE = 1e-5


def cats_set():
    """The locked top-NCAT category set, from stage 1's own record of it."""
    c = pd.read_csv(PIPE / "cats.csv", dtype={"cat": str})
    assert len(c) == locked.NCAT, (len(c), locked.NCAT)
    return set(c["cat"])


def load_axis_map(target):
    """(long fsc -> axis4 share frame, fsc -> modal axis4 series)."""
    if target == "naics":
        t = pd.read_csv(REF_DIR / "fsc_efsy_axis_concordance.csv", comment="#",
                        dtype={"fsc": str, "axis4": str, "top1_axis4": str})
        shares = t[["fsc", "axis4", "share"]]
        modal = t.drop_duplicates("fsc").set_index("fsc")["top1_axis4"]
    else:
        # SIC-1987 is a single vintage, so the SIC side needs no chain: the
        # concordance's `industry` is already the 4-digit target.
        t = pd.read_csv(REF_DIR / "fsc_sic_concordance.csv", comment="#",
                        dtype={"fsc": str, "industry": str})
        shares = (t.assign(axis4=t["industry"].str[:4])
                  .groupby(["fsc", "axis4"], as_index=False)["share"].sum())
        modal = (shares.sort_values(["share", "axis4"], ascending=[False, True])
                 .drop_duplicates("fsc").set_index("fsc")["axis4"])
    assert np.allclose(shares.groupby("fsc")["share"].sum().to_numpy(), 1.0,
                       atol=1e-9), "per-category shares must sum to 1"
    return shares, modal


def load_chain():
    ch = pd.read_csv(REF_DIR / "naics_to_efsy_axis.csv", comment="#",
                     dtype={"vintage": int, "naics6": str, "axis4": str})
    return ch, {v: set(g["naics6"]) for v, g in ch.groupby("vintage")}


def map_concordance_year(year, shares, modal, keys, cov, unc, target):
    """One year of stage-1 cells, spread over industries by the concordance."""
    cells = pd.read_parquet(PIPE / "contracts_category_ccg_year.parquet",
                            filters=[("year", "==", year)])
    # float64 BEFORE the multiply: the panel is stored float32 and the products
    # are what the conservation assertion is measured on.
    cells["dollars"] = cells["dollars"].astype(np.float64)
    known = cells["cat"].isin(keys)

    hit = cells[known]
    e = hit.merge(shares, left_on="cat", right_on="fsc")
    e["dollars_prob"] = e["dollars"] * e["share"]
    m = hit.assign(axis4=hit["cat"].map(modal), dollars_modal=hit["dollars"])
    out = pd.concat([
        e.groupby(["rule", "ccg", "axis4"], as_index=False)["dollars_prob"].sum(),
        m.groupby(["rule", "ccg", "axis4"], as_index=False)["dollars_modal"].sum(),
    ]).groupby(["rule", "ccg", "axis4"], as_index=False).sum(min_count=1).fillna(0.0)

    miss = cells[~known]
    if len(miss):
        s = (miss.groupby(["rule", "ccg"], as_index=False)["dollars"].sum()
             .assign(axis4=UNMAPPED)
             .rename(columns={"dollars": "dollars_prob"}))
        s["dollars_modal"] = s["dollars_prob"]
        out = pd.concat([out, s[out.columns]], ignore_index=True)

    for rule, tag in RULES:
        sub = cells[cells["rule"] == tag]
        got = out[out["rule"] == tag]
        tot, cvd = sub["dollars"].sum(), sub.loc[known[sub.index], "dollars"].sum()
        for col in ("dollars_prob", "dollars_modal"):
            rel = abs(got[col].sum() - tot) / max(abs(tot), 1.0)
            assert rel < RTOL_CONSERVE, (
                f"{target} {year} {tag} {col}: industry cells do not conserve "
                f"stage-1 dollars (relative gap {rel:.2e})")
        cov.append(dict(target=target, source="conc", rule=tag, year=year,
                        total_dollars=tot, mapped_dollars=cvd,
                        uncovered_dollars=tot - cvd,
                        uncovered_share=(tot - cvd) / tot if tot else np.nan,
                        n_uncovered_cats=int(sub.loc[~known[sub.index], "cat"].nunique())))
    if len(miss):
        g = miss.groupby(["rule", "cat"], as_index=False)["dollars"].sum()
        unc.append(g.assign(target=target, source="conc", year=year))
    out["year"] = year
    return out


def native_actions(year, fips_to_ccg):
    """The pipeline's own record universe, with each record's industry kept.

    lib/contracts.py is under the 05 gate and is not edited; this mirrors
    `category_actions` and is asserted equal to it below. Keeping the industry
    column aligned is legitimate because `harmonize_arrays` is a pure 1:1
    relabel for year >= 1979 - the row-expanding 1979 service bridge fires only
    at year <= 1978 - so row count and order survive it.
    """
    if year < dc.SEAM_YEAR:
        df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet",
                             columns=["dollars", "dollars_unit", "geo_class",
                                      "county_fips5", "fsc", "naics"])
        df = df[df["geo_class"] == "county"]
        d = df["dollars"].to_numpy(np.float64)
        d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
        src, ccg, ind = df["fsc"], df["county_fips5"].map(fips_to_ccg), df["naics"]
    else:
        df = pd.read_parquet(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
                             columns=["federal_action_obligation",
                                      "award_or_idv_flag", "geo_class", "ccg",
                                      "product_or_service_code", "naics_code"])
        df = df[df["award_or_idv_flag"].eq("AWARD") & df["geo_class"].eq("county")]
        d = df["federal_action_obligation"].to_numpy(np.float64)
        src, ccg, ind = df["product_or_service_code"], df["ccg"], df["naics_code"]
    cat = src.astype(str).str[:locked.NDIG]
    ok = (ccg.notna().to_numpy() & src.notna().to_numpy()
          & cat.str.len().eq(locked.NDIG).to_numpy()
          & ~cat.str.startswith("Y").to_numpy())
    d, ccg, cat = d[ok], ccg[ok].astype(np.int64).to_numpy(), cat[ok].to_numpy()
    ind = ind.astype(str).str.strip().str[:6].to_numpy()[ok]
    n_in = len(d)
    d, ccg, cat = hz.harmonize_arrays(year, d, ccg, cat)
    assert len(d) == n_in, (
        f"{year}: harmonisation changed the row count, so the industry column "
        f"is no longer aligned ({n_in} -> {len(d)})")
    keep = ~pd.Series(cat).str.startswith("Y").to_numpy()
    return d[keep], ccg[keep], cat[keep], ind[keep]


def verify_reader(year, fips_to_ccg, d, ccg, cat):
    """The mirrored reader must reproduce the pipeline's arrays exactly."""
    rd, rccg, rcat = contracts.category_actions(
        year, fips_to_ccg, drop_construction=True, harmonize=True)
    assert len(rd) == len(d), (year, len(rd), len(d))
    assert np.array_equal(rccg, ccg) and np.array_equal(rcat, cat), \
        f"{year}: the mirrored reader disagrees with contracts.category_actions"
    assert np.allclose(rd, d, rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC), year


def native_vintage(ind, d, universes, year, cov_rows):
    """The fiscal year's NAICS vintage, read off its own records.

    Which vintage a year's codes are written in is recorded nowhere, so it is
    inferred the only way available - whichever vintage's six-digit universe
    covers the most of that year's dollars - and the whole coverage row is
    reported, not just the winner, so a marginal call is visible. Federal
    archives keep old codes on long-running awards, so no year is ever fully
    one vintage; that is exactly why the per-record fallback below exists.
    """
    usable = pd.Series(ind).str.fullmatch(r"\d{6}").fillna(False).to_numpy()
    tot = d[usable].sum()
    cov = {v: d[usable & pd.Series(ind).isin(u).to_numpy()].sum() / tot
           for v, u in universes.items()}
    best = max(cov, key=cov.get)
    cov_rows.append(dict(year=year, vintage=best,
                         **{f"cov_{v}": c for v, c in sorted(cov.items())}))
    return best, usable


def map_native_year(year, keys, units, chain, universes, fips_to_ccg, cov, vint,
                    target, source, verify=True):
    """One year assigned from the records' own industry codes."""
    d, ccg, cat, ind = native_actions(year, fips_to_ccg)
    if verify:
        verify_reader(year, fips_to_ccg, d, ccg, cat)
    v = d * dc.deflate(year)
    # Stage 1 builds its tensor on the screened base panel's 3,058 counties and
    # silently drops records outside it (`unit_pos` misses become non-finite
    # and are masked). The mirrored reader is kept faithful to
    # category_actions so verify_reader stays a true equality test, so the
    # county restriction is applied here instead - without it the native pass
    # counts contracts in counties the estimation sample does not contain.
    in_cats = (pd.Series(cat).isin(keys).to_numpy()
               & pd.Series(ccg).isin(units).to_numpy())
    home, usable = native_vintage(ind, v, universes, year, vint)

    # The year's own vintage first, then the rest nearest-first: a code the
    # year's vintage does not have is a straggler on a long-running award.
    order = [home] + sorted((x for x in universes if x != home),
                            key=lambda x: (abs(x - home), x))
    vin = np.zeros(len(ind), dtype=np.int64)          # 0 = not placed
    inds = pd.Series(ind)
    for cand in order:
        todo = (vin == 0) & usable
        if not todo.any():
            break
        vin[todo & inds.isin(universes[cand]).to_numpy()] = cand

    # A record names one CODE, not one modern industry. An older-vintage code
    # can split across several cells of the current classification - 111998,
    # 238190, 315211 and others do - and the Census concordance offers nothing
    # finer than an equal split to divide it by. So the native path spreads
    # exactly as the concordance path does, and `dollars_modal` puts the
    # record whole on its code's largest destination instead.
    modal_map = (chain.sort_values(["weight", "axis4"], ascending=[False, True])
                 .drop_duplicates(["vintage", "naics6"])
                 .rename(columns={"axis4": "axis4_modal"})
                 [["vintage", "naics6", "axis4_modal"]])
    rec = pd.DataFrame({"ccg": ccg, "cat": cat, "v": v, "vintage": vin,
                        "naics6": ind, "in_cats": in_cats})
    placed, lost = rec[rec["vintage"] > 0], rec[rec["vintage"] == 0]
    e = placed.merge(chain, on=["vintage", "naics6"], how="left")
    assert e["axis4"].notna().all(), f"{year}: a placed code failed to reach the axis"
    e = e.merge(modal_map, on=["vintage", "naics6"], how="left")
    # Fan-out is a property of the CODE, so it is counted on the chain. Counting
    # it on the merged frame instead groups every record that shares a code and
    # reports ~100% of dollars as split, which is meaningless.
    fan = (chain.groupby(["vintage", "naics6"], as_index=False)["axis4"].size()
           .rename(columns={"size": "fan"}))
    e = e.merge(fan, on=["vintage", "naics6"], how="left")
    e["split"] = e["fan"] > 1
    # See producer 08: SPLIT is the exposure of the equal-split weighting,
    # CHANGED is the exposure of the vintage conversion itself.
    e["changed"] = e["axis4"] != e["naics6"].str[:4]
    e = e.drop(columns="fan")
    lo = lost.assign(axis4=NA, axis4_modal=NA, weight=1.0, split=False,
                     changed=False)
    allrec = pd.concat([e, lo[e.columns]], ignore_index=True)
    allrec["dollars_prob"] = allrec["v"] * allrec["weight"]
    # The modal column reassigns each RECORD, so its dollars are the record's
    # own v on the fanned-out row set - v * weight summed over a record's rows
    # is v again, which is why this is written as a weighted sum rather than a
    # deduplication.
    allrec["dollars_modal"] = allrec["dollars_prob"]

    frames = []
    for rule, tag in RULES:
        keep = allrec["in_cats"].to_numpy() & (
            (allrec["v"].to_numpy() > 0) if rule == "positive"
            else np.ones(len(allrec), bool))
        s = allrec[keep]
        prob = s.groupby(["ccg", "axis4"], as_index=False)["dollars_prob"].sum()
        md = (s.groupby(["ccg", "axis4_modal"], as_index=False)["dollars_modal"].sum()
              .rename(columns={"axis4_modal": "axis4"}))
        frames.append(prob.merge(md, on=["ccg", "axis4"], how="outer")
                      .fillna(0.0).assign(rule=tag))
        tot = s["dollars_prob"].sum()
        ok_d = s.loc[s["axis4"] != NA, "dollars_prob"].sum()
        cov.append(dict(target=target, source=source, rule=tag, year=year,
                        total_dollars=tot, mapped_dollars=ok_d,
                        uncovered_dollars=tot - ok_d,
                        uncovered_share=(tot - ok_d) / tot if tot else np.nan,
                        n_uncovered_cats=int(s.loc[s["axis4"] == NA, "cat"].nunique()),
                        split_dollar_share=(s.loc[s["split"], "dollars_prob"].sum() / tot
                                            if tot else np.nan),
                        changed_dollar_share=(s.loc[s["changed"], "dollars_prob"].sum() / tot
                                              if tot else np.nan)))
    out = pd.concat(frames, ignore_index=True)
    out["year"] = year
    for col in ("dollars_prob", "dollars_modal"):
        g = out.groupby("rule")[col].sum()
        for rule, tag in RULES:
            want = allrec.loc[allrec["in_cats"].to_numpy() & (
                (allrec["v"].to_numpy() > 0) if rule == "positive"
                else np.ones(len(allrec), bool)), "dollars_prob"].sum()
            rel = abs(g[tag] - want) / max(abs(want), 1.0)
            assert rel < RTOL_CONSERVE, (
                f"{target} {year} {tag} {col}: native cells do not conserve "
                f"record dollars (relative gap {rel:.2e})")
    return out[["rule", "ccg", "axis4", "year", "dollars_prob", "dollars_modal"]]


COLUMNS = ["source", "rule", "ccg", "year", "axis4",
           "dollars_prob", "dollars_modal"]
SCHEMA = pa.schema([("source", pa.string()), ("rule", pa.string()),
                    ("ccg", pa.int32()), ("year", pa.int16()),
                    ("axis4", pa.string()), ("dollars_prob", pa.float64()),
                    ("dollars_modal", pa.float64())])


class Writer:
    """A parquet file written one year at a time, or nothing at all on a dry run."""

    def __init__(self, path, dry):
        self.path, self.dry, self.rows, self._w = path, dry, 0, None

    def __enter__(self):
        if not self.dry:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._w = pq.ParquetWriter(self.path, SCHEMA)
        return self

    def write(self, frame):
        self.rows += len(frame)
        if self._w is None:
            return
        f = frame[COLUMNS].sort_values(["rule", "ccg", "axis4"])
        self._w.write_table(pa.Table.from_pandas(f, schema=SCHEMA,
                                                 preserve_index=False))

    def __exit__(self, *exc):
        if self._w is not None:
            self._w.close()


def run_target(target, dry, cov, unc, vint):
    shares, modal = load_axis_map(target)
    keys = set(shares["fsc"])
    # The native pass takes the industry from the RECORD, so the concordance's
    # key list is irrelevant to it: it must see the whole locked top-NCAT set,
    # including the 18 categories the concordance cannot map. Restricting it to
    # the intersection would silently delete exactly the dollars the era rule
    # exists to rescue - and native codes cover 100% of them.
    allcats = cats_set()
    print(f"  {target}: {len(keys):,} FSC keys in the map; "
          f"{len(allcats & keys)}/{locked.NCAT} locked categories covered by "
          f"the concordance", flush=True)
    chain, universes = load_chain()
    fips_to_ccg, _, _ = dc.load_ccg_map()
    units = set(panel.base_panel()["ccg"].unique())
    assert len(units) == locked.N_COUNTIES, (len(units), locked.N_COUNTIES)

    name = ("industry_cells_naics4.parquet" if target == "naics"
            else "industry_cells_sic4.parquet")
    native_first = dc.SEAM_YEAR if target == "naics" else locked.LAST_YEAR + 1

    # Streamed year by year rather than concatenated: 1966-2018 on the NAICS
    # side is some 40M rows, and holding them all as Python strings before the
    # write costs several GB for no benefit. One row group per year also makes
    # the filters=[("year","==",y)] read the rest of the pipeline uses cheap.
    # The file is therefore YEAR-MAJOR, sorted (rule, ccg, axis4) within a year.
    with Writer(OUT_DIR / name, dry) as w:
        for y in range(locked.FIRST_YEAR, locked.LAST_YEAR + 1):
            if y < native_first:
                b = map_concordance_year(y, shares, modal, keys, cov, unc, target)
                b["source"] = "conc"
            else:
                b = map_native_year(y, allcats, units, chain, universes,
                                    fips_to_ccg, cov, vint, target, "native")
                b["source"] = "native"
            w.write(b)
            if y % 10 == 0:
                print(f"    ... {y}", flush=True)
        if not dry:
            print(f"  wrote {name}  ({w.rows:,} rows)", flush=True)

    if target != "naics":
        return
    # Both drift tests: the concordance where a native code also exists
    # (FY2007-18), and the native code over the window the concordance was
    # estimated on (FY2001-06), which tests the estimation itself.
    with Writer(OUT_DIR / "industry_cells_crosscheck.parquet", dry) as w:
        for y in range(dc.SEAM_YEAR, locked.LAST_YEAR + 1):
            b = map_concordance_year(y, shares, modal, keys, cov, unc,
                                     target + "_cross")
            w.write(b.assign(source="conc"))
        for y in range(2001, dc.SEAM_YEAR):
            b = map_native_year(y, allcats, units, chain, universes,
                                fips_to_ccg, cov, vint, target + "_cross",
                                "native")
            w.write(b.assign(source="native"))
        if not dry:
            print(f"  wrote industry_cells_crosscheck.parquet "
                  f"({w.rows:,} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["naics", "sic", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="coverage report only; writes nothing")
    args = ap.parse_args()
    t0 = time.time()
    cov, unc, vint = [], [], []
    targets = ["naics", "sic"] if args.target == "both" else [args.target]
    for t in targets:
        run_target(t, args.dry_run, cov, unc, vint)

    cv = pd.DataFrame(cov)
    # The two sources must see the SAME dollars on the years where both run.
    # This is the end-to-end proof that the mirrored native reader reproduces
    # the stage-1 panel: same records, same geography, same deflator, same
    # top-NCAT restriction, differing only in where the industry comes from.
    a = cv[(cv["target"] == "naics") & (cv["source"] == "native")]
    b = cv[(cv["target"] == "naics_cross") & (cv["source"] == "conc")]
    both = a.merge(b, on=["rule", "year"], suffixes=("_nat", "_con"))
    if len(both):
        rel = ((both["total_dollars_nat"] - both["total_dollars_con"]).abs()
               / both["total_dollars_con"].abs()).max()
        assert rel < RTOL_CROSS_SOURCE, (
            "the native and concordance passes do not see the same dollars on "
            f"the overlap years (worst relative gap {rel:.2e}) - the mirrored "
            "reader is not reproducing the stage-1 panel")
        print(f"  native-vs-concordance dollar totals agree on FY"
              f"{both['year'].min()}-{both['year'].max()}: worst relative gap "
              f"{rel:.2e}")

    for (tgt, src), g in cv.groupby(["target", "source"]):
        p = g[g["rule"] == "pos"]
        tot = p["total_dollars"].sum()
        print(f"  {tgt:12s} {src:7s} pooled {p['year'].min()}-{p['year'].max()}: "
              f"${tot / 1e9:,.1f}bn, uncovered "
              f"{p['uncovered_dollars'].sum() / tot:.4%} "
              f"(${p['uncovered_dollars'].sum() / 1e9:,.2f}bn)")
    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cv.to_csv(OUT_DIR / "coverage_by_year.csv", index=False)
        if unc:
            pd.concat(unc, ignore_index=True).to_csv(
                OUT_DIR / "uncovered_by_category.csv", index=False)
        if vint:
            pd.DataFrame(vint).drop_duplicates(subset=["year"]).to_csv(
                OUT_DIR / "native_vintage_by_year.csv", index=False)
        print(f"  wrote coverage_by_year.csv, uncovered_by_category.csv, "
              f"native_vintage_by_year.csv -> {OUT_DIR}")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
