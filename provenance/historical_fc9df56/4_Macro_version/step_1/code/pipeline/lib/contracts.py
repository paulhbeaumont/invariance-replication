"""Contract ingestion: per-fiscal-year records to county and category panels.

Reads the parsed per-FY parquets (DCADS before the 2007 seam, USAspending
after), applies the thousands rescale and CPI deflation, and collapses to:

- county x year total dollars under the net and positive rules
  (contract_matrices);
- county x category x year dollar tensors on the harmonized top-NCAT
  category axis (category_panel), with national totals per category.

Guards preserved from the old producers: missingness is tested on the source
column BEFORE the string cast (a cast-then-match deletes every PSC code
containing "N"); the construction drop (leading-Y codes) is reapplied after
harmonization. A county absent from a year has ZERO contracts, not missing.

Extracted from producers 51 (county_actions 51:167, contract_matrices 51:203)
and 54 (category_actions 54:340, category_panel 54:388); operation order
preserved so outputs are bit-identical. Run this file directly to tie its
output to the golden fixtures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402

RULES = ("net", "positive")
FLOORS = ((0.0, "none"), (25_000.0, "25k nominal"))


def county_actions(year, fips_to_ccg, drop_construction=True):
    """One year of county-level contract actions, in dollars.

    drop_construction removes FSC letter Y (construction), matching the
    Nakamura-Steinsson state series.
    """
    if year < dc.SEAM_YEAR:
        cols = ["dollars", "dollars_unit", "geo_class", "county_fips5"]
        df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet",
                             columns=cols + (["fsc"] if drop_construction else []))
        df = df[df["geo_class"] == "county"]
        if drop_construction:
            df = df[~df["fsc"].astype(str).str[0].eq("Y")]
        d = df["dollars"].to_numpy(np.float64)
        d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
        ccg = df["county_fips5"].map(fips_to_ccg)
        ok = ccg.notna().to_numpy()
        return d[ok], ccg[ok].to_numpy(np.int64)
    cols = ["federal_action_obligation", "award_or_idv_flag", "geo_class", "ccg"]
    df = pd.read_parquet(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
                         columns=cols + (["product_or_service_code"]
                                         if drop_construction else []))
    d = df["federal_action_obligation"].to_numpy(np.float64)
    ok = (df["award_or_idv_flag"].eq("AWARD") & df["geo_class"].eq("county")
          & df["ccg"].notna()).to_numpy()
    if drop_construction:
        ok &= ~df["product_or_service_code"].astype(str).str[0].eq("Y").to_numpy()
    return d[ok], df.loc[ok, "ccg"].to_numpy(np.int64)


def contract_matrices(units, first, last, drop_construction=True):
    """County x year real-dollar matrices, one per (rule, floor) cell,
    rectangular on units x years with absent cells zero-filled."""
    fips_to_ccg, _, _ = dc.load_ccg_map()
    parts = {(r, f): [] for r in RULES for f, _ in FLOORS}
    for year in range(first, last + 1):
        d, ccg = county_actions(year, fips_to_ccg, drop_construction)
        defl = dc.deflate(year)
        for rule in RULES:
            base = (d > 0) if rule == "positive" else np.ones(len(d), bool)
            for floor, _ in FLOORS:
                keep = base & (np.abs(d) >= floor)
                g = pd.DataFrame({"ccg": ccg[keep], "v": d[keep] * defl})
                parts[(rule, floor)].append(
                    g.groupby("ccg", sort=False)["v"].sum()
                     .rename("v").reset_index().assign(year=year))
    return {k: (pd.concat(v, ignore_index=True)
                .pivot(index="ccg", columns="year", values="v")
                .reindex(units).reindex(columns=range(first, last + 1)).fillna(0.0))
            for k, v in parts.items()}


def category_actions(year, fips_to_ccg, drop_construction=True,
                     harmonize=False):
    """One year of county contract actions, carrying the category code.

    Rows with no readable 4-character code are dropped: they cannot enter a
    composition. They still enter the TREATMENT (county_actions above), which
    uses the unfiltered records; the two rest on slightly different record
    sets by construction.
    """
    if year < dc.SEAM_YEAR:
        df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet",
                             columns=["dollars", "dollars_unit", "geo_class",
                                      "county_fips5", "fsc"])
        df = df[df["geo_class"] == "county"]
        d = df["dollars"].to_numpy(np.float64)
        d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
        src = df["fsc"]
        ccg = df["county_fips5"].map(fips_to_ccg)
    else:
        df = pd.read_parquet(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
                             columns=["federal_action_obligation", "award_or_idv_flag",
                                      "geo_class", "ccg", "product_or_service_code"])
        df = df[df["award_or_idv_flag"].eq("AWARD") & df["geo_class"].eq("county")]
        d = df["federal_action_obligation"].to_numpy(np.float64)
        src = df["product_or_service_code"]
        ccg = df["ccg"]
    # Missingness on the SOURCE column, before the string cast (see module
    # docstring).
    cat = src.astype(str).str[:locked.NDIG]
    ok = (ccg.notna().to_numpy() & src.notna().to_numpy()
          & cat.str.len().eq(locked.NDIG).to_numpy())
    if drop_construction:
        ok &= ~cat.str.startswith("Y").to_numpy()
    d, ccg, cat = d[ok], ccg[ok].astype(np.int64).to_numpy(), cat[ok].to_numpy()
    if harmonize:
        import harmonize as hz
        d, ccg, cat = hz.harmonize_arrays(year, d, ccg, cat)
        if drop_construction:
            keep = ~pd.Series(cat).str.startswith("Y").to_numpy()
            d, ccg, cat = d[keep], ccg[keep], cat[keep]
    return d, ccg, cat


def category_panel(units, first, last, drop_construction=True, harmonize=False):
    """County x category x year dollar tensors, one per (rule, floor) cell.

    Returns (ARR, NAT, cats, years). ARR[(rule, floor)] has shape
    (n_units, NCAT, n_years) in real dollars, float32; NAT[(rule, floor)] is
    the national total by category and year. The category set is chosen ONCE,
    from the unrestricted net universe, so a floor cannot silently change
    which things a county is treated as producing. The parquets are read
    twice by design (the first pass only needs national dollars, to choose
    the category set).
    """
    fips_to_ccg, _, _ = dc.load_ccg_map()
    years = list(range(first, last + 1))
    nat_tot = {}
    for year in years:
        d, _, cat = category_actions(year, fips_to_ccg, drop_construction, harmonize)
        s = pd.Series(d * dc.deflate(year)).groupby(cat).sum()
        for k, v in s.items():
            nat_tot[k] = nat_tot.get(k, 0.0) + v
    cats = list(pd.Series(nat_tot).nlargest(locked.NCAT).index)
    cat_pos = {c: j for j, c in enumerate(cats)}
    unit_pos = {u: i for i, u in enumerate(units)}
    yr_pos = {y: t for t, y in enumerate(years)}

    ARR = {(r, f): np.zeros((len(units), len(cats), len(years)), np.float32)
           for r in RULES for f, _ in FLOORS}
    for year in years:
        d, ccg, cat = category_actions(year, fips_to_ccg, drop_construction, harmonize)
        v = d * dc.deflate(year)
        i = pd.Series(ccg).map(unit_pos).to_numpy(dtype=np.float64)
        j = pd.Series(cat).map(cat_pos).to_numpy(dtype=np.float64)
        t = yr_pos[year]
        keep0 = np.isfinite(i) & np.isfinite(j)
        for rule in RULES:
            base = (v > 0) if rule == "positive" else np.ones(len(v), bool)
            for floor, _ in FLOORS:
                m = keep0 & base & (np.abs(v) >= floor)
                if not m.any():
                    continue
                np.add.at(ARR[(rule, floor)],
                          (i[m].astype(np.int64), j[m].astype(np.int64), t),
                          v[m].astype(np.float32))
    NAT = {k: pd.DataFrame(a.sum(axis=0), index=cats, columns=years)
           for k, a in ARR.items()}
    return ARR, NAT, cats, years


if __name__ == "__main__":
    # Micro tie-out against the golden fixtures (raw extraction only; the
    # harmonized variant is harmonize.py's tie-out).
    GOLD = dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "_golden"
    gold = pd.read_parquet(GOLD / "category_actions_samples.parquet")
    fips_to_ccg, _, _ = dc.load_ccg_map()
    for y in sorted(gold["year"].unique()):
        ref = gold[(gold["year"] == y) & (gold["variant"] == "raw")]
        d, ccg, cat = category_actions(y, fips_to_ccg, drop_construction=True,
                                       harmonize=False)
        assert len(ref) == len(d), (y, len(ref), len(d))
        assert np.array_equal(ref["ccg"].to_numpy(), ccg), y
        assert np.array_equal(ref["cat"].to_numpy(), cat), y
        assert np.allclose(ref["dollars"].to_numpy(), d,
                           rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC), y
        print(f"category_actions tie-out {y}: {len(d):,} records match")

    base = pd.read_parquet(GOLD / "base_panel.parquet")
    units = np.array(sorted(base["ccg"].unique()))
    CON = contract_matrices(units, locked.FIRST_YEAR, locked.LAST_YEAR)
    for rule, tag in (("net", "net"), ("positive", "pos")):
        ref = (pd.read_parquet(GOLD / f"con_{tag}.parquet")
                 .pivot(index="ccg", columns="year", values="dollars"))
        new = CON[(rule, 0.0)]
        assert ref.index.equals(new.index) and list(ref.columns) == list(new.columns)
        a, b = ref.to_numpy(), new.to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a, b, rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC,
                           equal_nan=True), rule
        print(f"contract_matrices tie-out ({rule}): "
              f"{new.shape[0]:,} x {new.shape[1]} matches")
    print("contracts.py: PASS")
