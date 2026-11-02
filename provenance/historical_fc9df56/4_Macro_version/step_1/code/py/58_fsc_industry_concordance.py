"""FSC-to-industry concordance, estimated from the dual-coded contract years.

The early panel (1966-1987) carries a 4-digit FSC category code but no
industry code. Records from 1988 carry both, so the map from category to
industry can be estimated where both exist and applied where only the
category exists. This producer builds two probabilistic share tables:

    fsc_sic_concordance.csv      FSC x 4-digit SIC-1987, pooled 1990-2000
    fsc_naics_concordance.csv    FSC x 6-digit NAICS (1997/2002 era),
                                 pooled 2001-2006

The SIC table is the designated assignment for the early panel; the NAICS
table exists for triangulation. 1988-1989 are excluded from estimation (SIC
covers only 82.7% of 1988 county dollars, and that missingness need not be
random across categories) and serve as held-out validation years instead.
USASpending (2007+) is deliberately not an input: its category axis is PSC
and its NAICS spans later vintages, and no consumer needs a PSC-era map.

Record universe: geo_class == "county" ONLY. Unlike 54's category_actions,
no county-to-CCG mapping is required, because the concordance is a property
of the category, not of a county panel; the ccg condition is dropped on
purpose and no county identifier is read.

Sign rule: dollars are netted within each (category, industry, year) cell;
net-negative cells are floored at zero and the floored mass is reported.
Every share therefore lies in [0, 1] by construction, and this is asserted
rather than assumed.

Placeholder industry codes (SIC 9999, NAICS beginning 9999) are excluded
from the shares; their dollar mass is reported in the coverage diagnostics.

External check: FSC->SIC composed with the Census 1987-SIC-to-1997-NAICS
concordance (reference/sic87_naics97_census.csv, equal-split weights; source
https://www.census.gov/naics/concordances/1987_SIC_to_1997_NAICS.xls,
sha256 ba04e2332bbe5d56eec834908f7d0e253b377b59d906dbc3053088a0aa09701d)
is compared with the directly estimated FSC->NAICS at the 2- and 3-digit
levels. If the crosswalk file is absent the diagnostic reports
`unavailable`; it never blocks the tables.

Diagnostics land in Results/ns14_diagnostics/fsc_concordance_diagnostics.csv
as long-format rows (metric, key1, key2, value). The acceptance number for
the whole artifact is the 1966-1987 coverage row set: the dollar-weighted
share of each early year's county dollars whose FSC appears in the SIC
table's key set.
"""

from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

CODE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = CODE_DIR / "reference"
RESULTS_DIR = CODE_DIR.parent.parent / "Results" / "ns14_diagnostics"
LOG_DIR = CODE_DIR.parent.parent.parent / "batch_logs" / "macro_version"
LOG_FILE = LOG_DIR / f"fsc_industry_concordance_{datetime.now():%Y%m%d_%H%M%S}.log"

NDIG = 4                      # category granularity, matching producer 54
SIC_POOL = range(1990, 2001)  # estimation window for the SIC table
SIC_HOLDOUT = (1988, 1989)    # validation-only years (partial SIC coverage)
NAICS_POOL = range(2001, 2007)
EARLY = range(1966, 1988)     # the years the concordance exists to serve
LOW_SUPPORT_DOLLARS = 1_000_000.0


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def read_year(year, ind_col):
    """County records for one year: (dollars, category, industry).

    Category normalization follows 54's category_actions: missingness is
    tested on the source column BEFORE the string cast, because the cast
    writes the literal "nan" and a later case-insensitive match on that
    string would delete every code carrying an N. Codes shorter than NDIG
    after the cut are unusable. industry is None when ind_col is None;
    otherwise it is the raw stripped string with usability left to the
    caller, so coverage can be measured before any industry filter.
    """
    cols = ["dollars", "dollars_unit", "geo_class", "fsc"]
    if ind_col:
        cols.append(ind_col)
    df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet", columns=cols)
    df = df[df["geo_class"] == "county"]
    d = df["dollars"].to_numpy(np.float64)
    d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
    src = df["fsc"]
    cat = src.astype(str).str.strip().str[:NDIG]
    cat_ok = src.notna().to_numpy() & cat.str.len().eq(NDIG).to_numpy()
    ind = None
    if ind_col:
        ind = df[ind_col].astype(str).str.strip()
    return d, cat.to_numpy(), cat_ok, (ind.to_numpy() if ind is not None else None)


def usable_industry(ind, kind):
    s = pd.Series(ind)
    if kind == "sic":
        return (s.str.len().eq(4) & s.str.isdigit()).to_numpy()
    return (s.str.len().eq(6) & s.str.isdigit()).to_numpy()


def is_placeholder(ind, kind):
    s = pd.Series(ind)
    if kind == "sic":
        return s.eq("9999").to_numpy()
    return s.str.startswith("9999").to_numpy()


def cell_table(years, ind_col, kind, diag):
    """Netted (year, fsc, industry) cells and per-year coverage diagnostics.

    Netting happens within the cell-year; flooring happens here too, so
    every later aggregation (pooling, splits, leave-one-year-out) starts
    from the same nonnegative cells.
    """
    frames, floored = [], 0.0
    for y in years:
        d, cat, cat_ok, ind = read_year(y, ind_col)
        total = d.sum()
        ind_ok = usable_industry(ind, kind)
        ph = is_placeholder(ind, kind)
        both = cat_ok & ind_ok & ~ph
        diag.append(("coverage_category_dollars", y, "", d[cat_ok].sum() / total))
        diag.append(("coverage_industry_dollars", y, kind, d[ind_ok].sum() / total))
        diag.append(("placeholder_dollar_share", y, kind, d[ind_ok & ph].sum() / total))
        g = (pd.DataFrame({"fsc": cat[both], "industry": ind[both], "d": d[both]})
             .groupby(["fsc", "industry"], as_index=False)["d"].sum())
        floored += -g.loc[g["d"] < 0, "d"].sum()
        g.loc[g["d"] < 0, "d"] = 0.0
        g["year"] = y
        frames.append(g)
    diag.append(("floored_negative_dollars", kind, "", floored))
    return pd.concat(frames, ignore_index=True)


def shares_from_cells(cells):
    """Pool cell-years into a per-category share table; shares in [0,1]."""
    pooled = cells.groupby(["fsc", "industry"], as_index=False).agg(
        dollars=("d", "sum"), n_years=("d", lambda x: int((x > 0).sum())))
    pooled = pooled[pooled["dollars"] > 0]
    tot = pooled.groupby("fsc")["dollars"].transform("sum")
    pooled["share"] = pooled["dollars"] / tot
    assert (pooled["share"] >= 0).all() and (pooled["share"] <= 1).all()
    sums = pooled.groupby("fsc")["share"].sum()
    assert np.allclose(sums, 1.0, atol=1e-9), "per-category shares must sum to 1"
    return pooled


def prefix_shares(tab, k):
    g = tab.assign(p=tab["industry"].str[:k]).groupby(["fsc", "p"], as_index=False)["share"].sum()
    return g


def modal_columns(tab):
    for k in (2, 3, 4):
        g = prefix_shares(tab, k)
        top = g.sort_values("share", ascending=False).drop_duplicates("fsc")
        tab = tab.merge(top.rename(columns={"p": f"top1_{k}digit",
                                            "share": f"top1_share_{k}digit"}),
                        on="fsc", how="left")
    return tab


def loyo(cells, score_years, kind, diag, full_table=None):
    """Leave-one-year-out validation; held-out years score the full table."""
    for y in score_years:
        # Held-out years are scored against the full estimation-window table;
        # in-pool years get the genuine leave-one-year-out table.
        tab = full_table if full_table is not None else shares_from_cells(
            cells[cells["year"] != y])
        realized = cells[cells["year"] == y]
        if realized.empty:
            continue
        rtot = realized.groupby("fsc", as_index=False)["d"].sum().rename(columns={"d": "fsc_d"})
        for k in (2, 3, 4):
            g = prefix_shares(tab, k)
            top = g.sort_values("share", ascending=False).drop_duplicates("fsc")[["fsc", "p"]]
            r = realized.assign(p=realized["industry"].str[:k]).merge(top, on="fsc",
                                                                      suffixes=("", "_modal"))
            agree = r.loc[r["p"] == r["p_modal"], "d"].sum() / r["d"].sum()
            diag.append((f"loyo_modal_agreement_{k}digit", y, kind, agree))
        rp = realized.assign(p=realized["industry"].str[:3]).groupby(
            ["fsc", "p"], as_index=False)["d"].sum()
        rp["rshare"] = rp["d"] / rp.groupby("fsc")["d"].transform("sum")
        m = rp.merge(prefix_shares(tab, 3), on=["fsc", "p"], how="outer").fillna(0.0)
        l1 = (m.assign(a=(m["rshare"] - m["share"]).abs())
              .groupby("fsc")["a"].sum().rename("l1").reset_index().merge(rtot, on="fsc"))
        diag.append(("loyo_share_l1_3digit", y, kind,
                     (l1["l1"] * l1["fsc_d"]).sum() / l1["fsc_d"].sum()))


def held_out_cells(years, ind_col, kind, diag):
    return cell_table(years, ind_col, kind, diag)


def write_table(tab, path, meta):
    with open(path, "w") as f:
        for line in meta:
            f.write(f"# {line}\n")
        cols = ["fsc", "industry", "dollars", "share", "n_years",
                "is_construction", "low_support",
                "top1_2digit", "top1_share_2digit", "top1_3digit",
                "top1_share_3digit", "top1_4digit", "top1_share_4digit"]
        tab[cols].to_csv(f, index=False, float_format="%.10g")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    diag = []

    log("== FSC-to-industry concordance (producer 58) ==")

    # --- SIC side ---------------------------------------------------------
    sic_cells = cell_table(SIC_POOL, "sic", "sic", diag)
    sic = modal_columns(shares_from_cells(sic_cells))
    assert set(sic_cells["year"]) == set(SIC_POOL), "every estimation year must contribute"

    cov = {y: v for m, y, k, v in diag if m == "coverage_industry_dollars" and k == "sic"}
    hold_diag = []
    hold_cells = held_out_cells(SIC_HOLDOUT, "sic", "sic", hold_diag)
    cov.update({y: v for m, y, k, v in hold_diag if m == "coverage_industry_dollars" and k == "sic"})
    diag += hold_diag
    assert abs(cov[1988] - 0.827) <= 0.005, f"1988 SIC coverage {cov[1988]:.3f} off its pin"
    assert all(cov[y] >= 0.999 for y in SIC_POOL), "1990-2000 SIC coverage must be complete"
    log(f"SIC table: {sic['fsc'].nunique()} categories, {len(sic)} pairs; "
        f"1988 coverage {cov[1988]:.3f} (pin 0.827±0.005)")

    loyo(sic_cells, list(SIC_POOL), "sic", diag)
    loyo(pd.concat([sic_cells, hold_cells]), list(SIC_HOLDOUT), "sic", diag,
         full_table=sic)

    # --- NAICS side -------------------------------------------------------
    naics_cells = cell_table(NAICS_POOL, "naics", "naics", diag)
    naics = modal_columns(shares_from_cells(naics_cells))
    assert set(naics_cells["year"]) == set(NAICS_POOL), "every estimation year must contribute"
    log(f"NAICS table: {naics['fsc'].nunique()} categories, {len(naics)} pairs")
    loyo(naics_cells, list(NAICS_POOL), "naics", diag)

    # --- concentration ----------------------------------------------------
    for kind, tab in (("sic", sic), ("naics", naics)):
        fsc_d = tab.groupby("fsc")["dollars"].sum()
        for k in (2, 3, 4):
            g = prefix_shares(tab, k)
            top1 = g.sort_values("share", ascending=False).drop_duplicates("fsc").set_index("fsc")["share"]
            hhi = g.assign(s2=g["share"] ** 2).groupby("fsc")["s2"].sum()
            w = fsc_d / fsc_d.sum()
            for name, s in (("top1", top1), ("hhi", hhi)):
                q = np.array([np.interp(p, np.cumsum(w[s.sort_values().index]),
                                        s.sort_values()) for p in (0.10, 0.50, 0.90)])
                for p, v in zip(("p10", "p50", "p90"), q):
                    diag.append((f"concentration_{name}_{k}digit_{p}", kind, "", v))

    # --- stability, SIC 1990-1994 vs 1995-2000 ---------------------------
    a = shares_from_cells(sic_cells[sic_cells["year"] <= 1994])
    b = shares_from_cells(sic_cells[sic_cells["year"] >= 1995])
    pa, pb = prefix_shares(a, 3), prefix_shares(b, 3)
    both = set(pa["fsc"]) & set(pb["fsc"])
    diag.append(("stability_categories_one_half_only", "sic", "",
                 len(set(pa["fsc"]) ^ set(pb["fsc"]))))
    m = pa[pa["fsc"].isin(both)].merge(pb[pb["fsc"].isin(both)], on=["fsc", "p"],
                                       how="outer").fillna(0.0)
    l1 = m.assign(a=(m["share_x"] - m["share_y"]).abs()).groupby("fsc")["a"].sum()
    fsc_d = sic.groupby("fsc")["dollars"].sum()
    w = fsc_d.reindex(l1.index).fillna(0.0)
    diag.append(("stability_l1_3digit_dollarwtd", "sic", "", (l1 * w).sum() / w.sum()))
    ta = pa.sort_values("share", ascending=False).drop_duplicates("fsc").set_index("fsc")["p"]
    tb = pb.sort_values("share", ascending=False).drop_duplicates("fsc").set_index("fsc")["p"]
    common = ta.index.intersection(tb.index)
    wc = fsc_d.reindex(common).fillna(0.0)
    diag.append(("stability_modal_agreement_3digit", "sic", "",
                 (wc * (ta[common] == tb[common])).sum() / wc.sum()))

    # --- early-panel coverage, the acceptance number ---------------------
    keys = set(sic["fsc"])
    for y in EARLY:
        d, cat, cat_ok, _ = read_year(y, None)
        total = d.sum()
        diag.append(("coverage_category_dollars", y, "", d[cat_ok].sum() / total))
        in_keys = cat_ok & pd.Series(cat).isin(keys).to_numpy()
        diag.append(("early_coverage_sic_keys", y, "", d[in_keys].sum() / total))
    early = {y: v for m, y, k, v in diag if m == "early_coverage_sic_keys"}
    log("\n1966-1987 dollars in SIC-concordance categories (acceptance numbers):")
    for y in EARLY:
        log(f"  {y}: {early[y]:6.1%}")

    # --- triangulation through the Census crosswalk ----------------------
    xwalk_path = REF_DIR / "sic87_naics97_census.csv"
    if not xwalk_path.exists():
        diag.append(("triangulation", "status", "", "unavailable"))
        log("\nTriangulation: crosswalk file missing, diagnostic unavailable.")
    else:
        xw = pd.read_csv(xwalk_path, comment="#", dtype=str)
        xw["weight"] = xw["weight"].astype(float)
        implied = (sic[["fsc", "industry", "share"]]
                   .merge(xw.rename(columns={"sic": "industry"}), on="industry")
                   .assign(s=lambda x: x["share"] * x["weight"]))
        matched = implied.groupby("fsc")["s"].sum()
        fsc_d = sic.groupby("fsc")["dollars"].sum()
        for k in (2, 3):
            imp = (implied.assign(p=implied["naics97"].str[:k])
                   .groupby(["fsc", "p"], as_index=False)["s"].sum())
            imp["s"] = imp["s"] / imp.groupby("fsc")["s"].transform("sum")
            direct = prefix_shares(naics, k).rename(columns={"share": "sd"})
            m = imp.merge(direct, on=["fsc", "p"], how="outer")
            m = m[m["fsc"].isin(set(imp["fsc"]) & set(direct["fsc"]))].fillna(0.0)
            ov = 1 - 0.5 * (m.assign(a=(m["s"] - m["sd"]).abs()).groupby("fsc")["a"].sum())
            w = fsc_d.reindex(ov.index).fillna(0.0)
            diag.append((f"triangulation_overlap_{k}digit", "sic_vs_naics", "",
                         (ov * w).sum() / w.sum()))
        top20 = fsc_d.reindex(ov.index).nlargest(20).index
        log("\nTriangulation overlap (3-digit) for the top-20 dollar categories:")
        for f in top20:
            diag.append(("triangulation_overlap_3digit_fsc", f, "", ov[f]))
            log(f"  {f}: {ov[f]:5.2f}")
        diag.append(("triangulation_sic_dollars_bridged", "", "",
                     (matched.clip(upper=1.0) * fsc_d.reindex(matched.index)).sum()
                     / fsc_d.sum()))

    # --- flags and outputs ------------------------------------------------
    for kind, tab, meta in (
        ("sic", sic, ["FSC x 4-digit SIC-1987 dollar-share concordance.",
                      "Estimation window 1990-2000, DCADS county records only "
                      "(geo_class==county, no CCG condition).",
                      "Sign rule: netted within (fsc, industry, year), negative "
                      "cells floored at zero.",
                      "Designated assignment table for the 1966-1987 panel."]),
        ("naics", naics, ["FSC x 6-digit NAICS dollar-share concordance.",
                      "Estimation window 2001-2006 (NAICS-1997/2002 era), DCADS "
                      "county records only (geo_class==county, no CCG condition).",
                      "Sign rule: netted within (fsc, industry, year), negative "
                      "cells floored at zero.",
                      "Triangulation companion to the SIC table; not the "
                      "early-panel assignment."]),
    ):
        tab["is_construction"] = tab["fsc"].str.startswith("Y")
        tab["low_support"] = (tab["dollars"] < LOW_SUPPORT_DOLLARS) | (tab["n_years"] <= 1)
        out = REF_DIR / f"fsc_{kind}_concordance.csv"
        write_table(tab, out, meta)
        log(f"\nwrote {out}  ({len(tab)} rows)")

    dd = pd.DataFrame(diag, columns=["metric", "key1", "key2", "value"])
    dd.to_csv(RESULTS_DIR / "fsc_concordance_diagnostics.csv", index=False)
    log(f"wrote {RESULTS_DIR / 'fsc_concordance_diagnostics.csv'}  ({len(dd)} rows)")

    for k in (2, 3, 4):
        vals = dd[(dd["metric"] == f"loyo_modal_agreement_{k}digit")
                  & (dd["key2"] == "sic")]["value"].astype(float)
        log(f"LOYO modal agreement, SIC {k}-digit: "
            f"min {vals.min():.3f}  median {vals.median():.3f}  max {vals.max():.3f}")
    l1v = dd[(dd["metric"] == "loyo_share_l1_3digit") & (dd["key2"] == "sic")]["value"].astype(float)
    log(f"LOYO share-vector L1 (3-digit, SIC): median {l1v.median():.3f}")


if __name__ == "__main__":
    main()
