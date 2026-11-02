#!/usr/bin/env python3
"""The county grid again, with the instrument built from WHAT a county sells.

51's instrument is a county's contract INTENSITY times a single national shift:

    z_ct = (sum_w C_cw / sum_w X_cw) * (national spending change per unit of X)

One national number per year, scaled up and down by how contract-heavy the county
is. This file replaces it with an instrument that asks a different question. Take
what the county sold the military over a predetermined window, category by
category, and ask what happened to NATIONAL demand for those particular things:

    z_ct = ( sum_k Cbar_ck,window * g_kt ) / X_ct-h

    Cbar_ck,window   the county's average annual dollars in Federal Supply
                     Classification k over a window that closes before the
                     treatment window opens
    g_kt             the national change in category k over the horizon, divided
                     by the category's own mean size so a small category cannot
                     produce a large rate, then clipped

A county that built ships and a county that built electronics get different
instruments in the same year, which is the whole point: identification comes from
the composition of what a place produces meeting a national demand shift for those
categories, not from the place being large or contract-heavy.

WHAT IS IDENTICAL TO 51, ON PURPOSE

The outcome, the treatment, the sample, the fixed effects, the weights, the
winsorization and every grid axis. The treatment comes from 51's own
`contract_matrices`, not from a rebuild, so it is the same numbers in the same
cells. ONLY the instrument changes. Every cell written here has a twin in
`county_multiplier_grid_full.csv` under an identical set of axis labels, so the
two are joined on those labels and compared without any alignment step.

A UNIT WITH NO CONTRACTS HAS A ZERO SHOCK, NOT A MISSING ONE

A county with no procurement never appears in a groupby over contract records, so
reindexing onto the unit list returns NaN and the county leaves the sample without
saying so. That is a bug, not a boundary: this instrument is denominated in
dollars per head, so no contracts means every term in the sum is zero times a
growth rate, and the shock is exactly zero. It is not the undefined 0/0 of a
composition normalized to sum to one, which this is not.

Leaving it unfilled costs 33 counties: 30 with no county-coded contract across
1966-2018, and 3 whose contracts never carried a usable 4-digit code. That is
33 x 39 years = 1,287 observations against a reference cell of 119,262. Filling
them moves the estimate from 3.945 to 3.983, t 3.29 to 3.30, F 41.2 to 41.1,
because a restored county carries a zero treatment and a zero instrument. The
reference-cell assertion below exists so this cannot come back silently.

READING THE FIRST STAGE HERE

A high F says the instrument is not weak. It does not say the instrument carries
procurement information, because a contract-free placebo built from 1/P times the
national shift reaches F = 34.5 to 35.5 off the denominator that treatment and
exposure share. That placebo's F is NOT a floor other instruments must clear: an
instrument orthogonal to 1/P draws nothing from that channel. This one runs at
corr -0.0021 with 1/P, so its F is its own. Both diagnostics print on every run
and are written to the CSV, so neither question has to be answered from memory.

THE HEADLINE SPECIFICATION

Ten-year share window, five-year lag, clip 1.0, FSC 4-digit, top 300 categories,
at the four-year horizon: the grid's roll10 exposure at h = 4. There the
instrument reaches t 6.77 with F 48.4 on 97,856 county-years, against the
incumbent's t 6.05 on the same 97,856. At h = 2 and h = 3 the incumbent is still
the sharper of the two, 4.69 against 3.48 and 6.75 against 5.13, so this is a
horizon-specific claim and not a general one.

Output:
    batch_logs/macro_version/county_composition_grid.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

PROJECT_ROOT = CODE.parents[3]
STEP2_CODE = PROJECT_ROOT / "4_Macro_version" / "step_2" / "code" / "py"
OUT = dc.LOG_DIR / "county_composition_grid.csv"
INCUMBENT = dc.LOG_DIR / "county_multiplier_grid_full.csv"

# Category granularity. 4-digit codes at the top 800 categories, chosen on the
# HARMONIZED panel per the flow-construction audit's reversal rule
# (Table_flow_audit.csv). On clean codes the count is nearly inert: the
# headline cell reads t 7.43 / F 36.5 at 800 against 7.39 / 34.2 at 300, and
# grid-wide median |t| is 1.54 at 800, 1.50 at 300, 1.44 at 100; 800 and 300
# give identical axis winners, while 100 loses outright and flips the
# exposure, FE, and weight winners. The digit choice stays at 4: 2-digit
# top-50 splits the cells (t 2.77 vs 2.73 at h=2, 6.41 vs 7.39 at h=4).
NDIG = 4
NCAT = 800

# The national category rate. A category's change divided by its OWN mean size,
# rather than by its base, because a base-scaled rate lets a category that was
# nearly zero in the base year produce an enormous growth rate on a trivial
# dollar move. Base-scaling gave t 1.65 against 3.29 here. The clip is the
# backstop for whatever survives that.
CLIP = 1.0

# A reference cell whose observation count is known from the incumbent grid and
# from the denominator probe, asserted on every run. It stays on roll5 because
# roll5 anchors the LARGEST sample, 119,262, which is the count that catches a
# zero-fill regression; it is not the headline specification.
REF = dict(era="full", horizon=2, denom="population", rule="net", floor="none",
           screen=False, exposure="roll5")
REF_N, REF_COUNTIES = 119_262, 3_058

# THE HEADLINE CELL. The share window runs ten years, not five, which is the one
# setting that moved after the grid was first read. A county's product mix is a
# persistent characteristic, so five years of lumpy contract awards estimate it
# noisily and the noise attenuates the first stage. Lengthening the window
# measures the same object better rather than measuring a different one, which is
# why it is defensible ahead of seeing any coefficient: with lag held at 5, the
# standard error falls from 0.913 at span 5 to 0.712 at span 10, and t rises at
# every horizon (3.29 to 3.48 at h = 2, 3.84 to 5.13 at h = 3, 4.75 to 6.77 at
# h = 4). Span 10 with lag 5 is exactly the grid's roll10 exposure, so no cell
# outside the existing grid is involved.
#
# Span 15 is better still on t alone and is NOT taken: with lag 5 it costs the
# sample needed to compute it, falling to 82,566 observations and t 3.67 at
# h = 4. It only wins when paired with a two-year lag, which buys precision by
# measuring the county's mix closer to the treatment window and weakens the claim
# that the mix is predetermined.
#
# The clip stays at 1.0. At span 10 it holds F between 44.1 and 48.4 across the
# three horizons where a clip of 2.0 gives 26.7 to 37.5, and a looser clip trades
# first-stage F against second-stage precision in opposite directions.
HEADLINE = dict(era="full", horizon=4, denom="population", rule="net",
                floor="none", screen=False, exposure="roll10", lam=1.0,
                fe="divisionyear", weight="division-EW", winsor_group="year")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p51 = _load("p51", CODE / "51_county_multiplier_grid.py")

REF_DIR = CODE.parent / "reference"
_HARMONIZATION = None


def load_harmonization():
    """The seam repairs, loaded once.

    Returns (bridge, relabel, orphan_map). `bridge` maps a pre-1979 service
    code to a list of (modern code, weight) summing to 1; codes the bridge
    left unmatched are absent, so their records drop out of the harmonized
    composition panel (1.0% of pre-1979 service dollars).

    `relabel` is a single label map applied SYMMETRICALLY to every year, so a
    renamed code carries one name over the whole panel and no seam can churn
    on it. It holds three layers: the FY1989 renames (architect-engineer R2xx
    to C2xx, ADP R3xx to D3xx, discovered as exact suffix matches between the
    1988 dying and 1989 born sets); the FY2012 unique-predecessor renames
    (new code to official predecessor); and the FY2012 consolidations (each
    ended predecessor to its merged successor).

    `orphan_map` closes the FY2012 newborns that have no predecessor into
    their family's `XX99` catch-all (or largest FY2011 code); it applies only
    to post-2011 years, because a label absent from FY2011 for the opposite
    reason (a code that died earlier) must not be rewritten.
    """
    global _HARMONIZATION
    if _HARMONIZATION is not None:
        return _HARMONIZATION
    b = pd.read_csv(REF_DIR / "service_code_bridge_1979.csv", comment="#",
                    dtype={"old_code": str, "new_code": str})
    b = b[b["new_code"].notna() & (b["new_code"] != "") & (b["weight"] > 0)]
    bridge = {oc: list(zip(g["new_code"], g["weight"]))
              for oc, g in b.groupby("old_code")}

    def year_codes(year):
        if year < dc.SEAM_YEAR:
            df = pd.read_parquet(dc.OUT_DIR / f"fy{year}.parquet",
                                 columns=["dollars", "geo_class", "fsc"])
            df = df[df["geo_class"] == "county"]
            code = df["fsc"].astype(str).str.strip().str[:4]
            keep = df["fsc"].notna() & code.str.len().eq(4)
            return set(code[keep][df.loc[keep, "dollars"] > 0])
        df = pd.read_parquet(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
                             columns=["federal_action_obligation",
                                      "award_or_idv_flag", "geo_class",
                                      "product_or_service_code"])
        df = df[df["award_or_idv_flag"].eq("AWARD")
                & df["geo_class"].eq("county")
                & (df["federal_action_obligation"] > 0)]
        code = df["product_or_service_code"].astype(str).str.strip().str[:4]
        return set(code[code.str.len().eq(4)])

    # FY1989: dying R-codes whose exact suffix reappears under C, D, or E.
    dead89 = year_codes(1988) - year_codes(1989)
    born89 = year_codes(1989) - year_codes(1988)
    relabel = {}
    for c in dead89:
        if c[0] == "R":
            for pre in ("C", "D", "E"):
                if pre + c[1:] in born89:
                    relabel[c] = pre + c[1:]
                    break

    # FY2012: unique-predecessor renames applied new-to-old, consolidations
    # applied old-to-new; both symmetric, exactly the construction validated
    # against the 2011-2018 churn (0.55% at the seam).
    xw = pd.read_csv(REF_DIR / "psc2011_crosswalk.csv", comment="#",
                     dtype=str).fillna("")
    preds = {}
    for o, n in zip(xw.previous_psc, xw.current_psc):
        if n and n != o:
            preds.setdefault(n, []).append(o)
    for n, olds in preds.items():
        if len(olds) == 1:
            relabel[n] = olds[0]
        else:
            for o in olds:
                relabel[o] = n

    known11 = year_codes(2011)
    handled = set(relabel)
    orphan_map = {}
    fam_sets = {}
    for c in sorted(known11):
        fam_sets.setdefault(c[:2], []).append(c)
    for c in year_codes(2012) - known11 - handled:
        fam = c[:2]
        if fam + "99" in known11:
            orphan_map[c] = fam + "99"
        elif fam_sets.get(fam):
            orphan_map[c] = fam_sets[fam][0]

    # Resolve chains once (an FY1989 target that FY2012 renames, etc.), so a
    # single map application lands on the final label.
    for k in list(relabel):
        v, seen = relabel[k], {k}
        while v in relabel and v not in seen:
            seen.add(v)
            v = relabel[v]
        relabel[k] = v

    # Letter spreads for pre-1979 service codes the bridge never saw (codes
    # dead before its 1976-1978 window, up to 6.7% of the earliest years'
    # dollars): distribute over the letter's modern codes by their 1979-1981
    # national dollar shares, the same mechanism as the bridge's A/Y
    # letter-level fallback.
    nat = {}
    for y in (1979, 1980, 1981):
        df = pd.read_parquet(dc.OUT_DIR / f"fy{y}.parquet",
                             columns=["dollars", "dollars_unit", "geo_class",
                                      "fsc"])
        df = df[df["geo_class"] == "county"]
        dv = df["dollars"].to_numpy(np.float64)
        dv = np.where((df["dollars_unit"] == "thousands").to_numpy(),
                      dv * 1000.0, dv)
        code = df["fsc"].astype(str).str.strip().str[:4]
        keep = (df["fsc"].notna() & code.str.len().eq(4)
                & code.str[0].str.isalpha() & (dv > 0))
        s = pd.Series(dv[keep.to_numpy()]).groupby(
            code[keep].to_numpy()).sum()
        for k, v in s.items():
            nat[k] = nat.get(k, 0.0) + v
    nat = pd.Series(nat)
    letter_spread = {}
    for letter, g in nat.groupby(nat.index.str[0]):
        # top 30 codes per letter, renormalized: bounds the record expansion
        # while keeping >90% of each letter's mass in its own proportions
        g = g.nlargest(30)
        letter_spread[letter] = list(zip(g.index, (g / g.sum()).to_numpy()))

    _HARMONIZATION = (bridge, relabel, orphan_map, letter_spread)
    return _HARMONIZATION


def harmonize_arrays(year, d, ccg, cat):
    """Rewrite one year's (dollars, ccg, code) arrays into the harmonized
    namespace. Pre-1979 service records are expanded through the bridge;
    the symmetric relabel map applies to every year; orphan closure applies
    to post-2011 years only."""
    bridge, relabel, orphan_map, letter_spread = load_harmonization()
    if year <= 1978:
        # Goods pass through; bridged service records are expanded through
        # the weights; service codes the bridge never saw (dead before its
        # 1976-1978 window) get the letter-level national spread; only codes
        # whose letter has no modern counterpart drop.
        s = pd.Series(cat)
        svc = s.str[0].str.isalpha().to_numpy()
        in_bridge = svc & s.isin(bridge.keys()).to_numpy()
        unb = svc & ~in_bridge
        goods = ~svc
        out_d = [d[goods]]
        out_c = [ccg[goods]]
        out_k = [cat[goods]]
        bt = pd.DataFrame([(oc, nc, w) for oc, pairs in bridge.items()
                           for nc, w in pairs],
                          columns=["code", "new", "w"])
        if in_bridge.any():
            m = pd.DataFrame({"d": d[in_bridge], "ccg": ccg[in_bridge],
                              "code": cat[in_bridge]})
            m = m.merge(bt, on="code", how="inner")
            out_d.append((m["d"] * m["w"]).to_numpy(np.float64))
            out_c.append(m["ccg"].to_numpy())
            out_k.append(m["new"].to_numpy(dtype=object))
        if unb.any():
            lt = pd.DataFrame([(letter, nc, w)
                               for letter, pairs in letter_spread.items()
                               for nc, w in pairs],
                              columns=["letter", "new", "w"])
            m = pd.DataFrame({"d": d[unb], "ccg": ccg[unb],
                              "letter": pd.Series(cat[unb]).str[0]})
            m = m.merge(lt, on="letter", how="inner")
            out_d.append((m["d"] * m["w"]).to_numpy(np.float64))
            out_c.append(m["ccg"].to_numpy())
            out_k.append(m["new"].to_numpy(dtype=object))
        d, ccg, cat = (np.concatenate(out_d), np.concatenate(out_c),
                       np.concatenate(out_k))
    s = pd.Series(cat)
    mapped = s.map(relabel)
    if year >= 2012:
        orph = s.map(orphan_map)
        mapped = mapped.fillna(orph)
    cat = mapped.fillna(s).to_numpy()
    return d, ccg, cat


def category_actions(year, fips_to_ccg, drop_construction=True,
                     harmonize=False):
    """One year of county contract actions, carrying the category code.

    This is 51's `county_actions` with the classification kept rather than
    discarded, and with rows whose code is unusable dropped. A row with no
    readable code cannot be assigned to a category, so it cannot enter a
    composition; it still enters the TREATMENT, which is built by 51 from the
    unfiltered records. The two therefore rest on slightly different record sets
    by construction, which is a property of the instrument and not a sample
    difference: no county-year is lost to it once the zero-fill is in place.
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
    # Missingness is tested on the SOURCE column, before the string cast. Testing
    # the cast instead means matching the literal "nan" the cast produces, and a
    # case-insensitive match for that string silently deletes every PSC service
    # code carrying an N, which is a large and non-random block of the post-2000
    # records rather than a handful of unreadable ones.
    cat = src.astype(str).str[:NDIG]
    ok = (ccg.notna().to_numpy() & src.notna().to_numpy()
          & cat.str.len().eq(NDIG).to_numpy())
    if drop_construction:
        ok &= ~cat.str.startswith("Y").to_numpy()
    d, ccg, cat = d[ok], ccg[ok].astype(np.int64).to_numpy(), cat[ok].to_numpy()
    if harmonize:
        d, ccg, cat = harmonize_arrays(year, d, ccg, cat)
        if drop_construction:
            keep = ~pd.Series(cat).str.startswith("Y").to_numpy()
            d, ccg, cat = d[keep], ccg[keep], cat[keep]
    return d, ccg, cat


def category_panel(units, first, last, drop_construction=True, harmonize=False):
    """county x category x year dollar tensors, one per (rule, floor) cell.

    Returns (ARR, NAT, cats, years). ARR[(rule, floor)] has shape
    (n_units, NCAT, n_years) in real dollars; NAT[(rule, floor)] is the national
    total by category and year. The category set is chosen ONCE, from the
    unrestricted net universe, so every cell of the grid weighs the same
    categories and a floor cannot silently change which things a county is
    treated as producing.

    The tensor is float32. At 3,058 counties, 300 categories and 53 years that is
    194 MB per cell and 778 MB for the four, against 1.6 GB in float64. The
    numbers are dollar aggregates read into a growth rate that is then clipped, so
    float32 precision is not the binding constraint anywhere downstream.

    The parquets are read twice. The first pass only needs national dollars by
    category, to choose the category set; holding every year's records in memory
    to save that pass would buffer tens of millions of object-dtype code strings,
    which costs far more than reading the files again.
    """
    fips_to_ccg, _, _ = dc.load_ccg_map()
    years = list(range(first, last + 1))
    nat_tot = {}
    for year in years:
        d, _, cat = category_actions(year, fips_to_ccg, drop_construction, harmonize)
        s = pd.Series(d * dc.deflate(year)).groupby(cat).sum()
        for k, v in s.items():
            nat_tot[k] = nat_tot.get(k, 0.0) + v
    cats = list(pd.Series(nat_tot).nlargest(NCAT).index)
    cat_pos = {c: j for j, c in enumerate(cats)}
    unit_pos = {u: i for i, u in enumerate(units)}
    yr_pos = {y: t for t, y in enumerate(years)}

    ARR = {(r, f): np.zeros((len(units), len(cats), len(years)), np.float32)
           for r in p51.RULES for f, _ in p51.FLOORS}
    for year in years:
        d, ccg, cat = category_actions(year, fips_to_ccg, drop_construction, harmonize)
        v = d * dc.deflate(year)
        i = pd.Series(ccg).map(unit_pos).to_numpy(dtype=np.float64)
        j = pd.Series(cat).map(cat_pos).to_numpy(dtype=np.float64)
        t = yr_pos[year]
        keep0 = np.isfinite(i) & np.isfinite(j)
        for rule in p51.RULES:
            base = (v > 0) if rule == "positive" else np.ones(len(v), bool)
            for floor, _ in p51.FLOORS:
                m = keep0 & base & (np.abs(v) >= floor)
                if not m.any():
                    continue
                np.add.at(ARR[(rule, floor)],
                          (i[m].astype(np.int64), j[m].astype(np.int64), t),
                          v[m].astype(np.float32))
    NAT = {k: pd.DataFrame(a.sum(axis=0), index=cats, columns=years)
           for k, a in ARR.items()}
    return ARR, NAT, cats, years


def category_rates(nat, h, clip=CLIP):
    """National change in each category over h years, scaled by its own mean size.

    Returns a (n_cats, n_years) array. A category whose h-year lag is outside the
    panel has no rate and is held out of the sum for that year rather than being
    read as zero growth, which is why the caller counts contributing categories.
    """
    v = nat.to_numpy(np.float64)
    out = np.full_like(v, np.nan)
    out[:, h:] = v[:, h:] - v[:, :-h]
    scale = np.nanmean(np.abs(v), axis=1, keepdims=True)
    scale[scale == 0] = np.nan
    return np.clip(out / scale, -clip, clip)


def window_sums(arr, years, lo, hi):
    """Mean annual dollars per category over the inclusive year window [lo, hi]."""
    a = years.index(lo)
    b = years.index(hi)
    return arr[:, :, a:b + 1].mean(axis=2)


def predicted_dollars(arr, gk, years, target_years, h, kind, spec, roll_lag,
                      first_window_year):
    """The instrument's numerator: predicted dollar change, county by year.

    For each target year t the county's category mix is measured over a window
    that closes at t-h-roll_lag, before the treatment window opens at t-h+1, and
    is met with the national rate for that same t. Returns a (n_units,
    n_target_years) array with NaN in columns whose window falls outside the data,
    and the count of categories that contributed.

    A rolling window stops at `first_window_year`, which is 51's DATA_FIRST rather
    than the first year of contract data. This instrument does not divide by
    income or population over the window, so on its own terms it could reach back
    to 1966 and buy three more years of sample. Taking them would put the cell on
    a different sample from its twin in the incumbent grid, which is the one thing
    this file exists not to do: roll5 at h = 2 would run on 128,436 observations
    against the incumbent's 119,262 and the two coefficients would no longer be
    comparable. The extra years are a robustness to run deliberately, not a
    default to acquire by accident.
    """
    out = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    ncontrib = np.zeros(len(target_years), np.int64)
    for c, t in enumerate(target_years):
        if kind == "fixed":
            lo, hi = spec
        else:
            hi = t - h - roll_lag
            lo = hi - spec + 1
            if lo < first_window_year:
                continue
        if lo < years[0] or hi > years[-1] or lo > hi:
            continue
        if t not in years:
            continue
        g = gk[:, years.index(t)]
        live = np.isfinite(g)
        if not live.any():
            continue
        cbar = window_sums(arr, years, lo, hi)
        out[:, c] = cbar[:, live].astype(np.float64) @ g[live]
        ncontrib[c] = int(live.sum())
    return out, ncontrib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", nargs="+", default=list(p51.ERAS),
                    choices=list(p51.ERAS))
    ap.add_argument("--out", default=None,
                    help="CSV to write; defaults to %s" % OUT.name)
    ap.add_argument("--include-construction", action="store_true",
                    help="keep FSC/PSC Y in both the treatment and the category "
                         "panel. The default excludes it, as 51 does")
    ap.add_argument("--nominal", action="store_true",
                    help="build both the treatment and the category panel in "
                         "NOMINAL dollars. The default deflates, as 51 does, so "
                         "that cells are comparable to the incumbent grid")
    ap.add_argument("--skip-assert", action="store_true",
                    help="run without the reference-cell check. Use only when "
                         "deliberately running a cell the reference does not cover")
    ap.add_argument("--ncat", type=int, default=None,
                    help="override the top-N category count (default %d). The "
                         "output filename carries the override so grids at "
                         "different counts never collide" % NCAT)
    ap.add_argument("--harmonize", action="store_true",
                    help="build the category panel on the harmonized namespace "
                         "(1979 bridge, 1989 renames, 2012 crosswalk). Writes to "
                         "county_composition_grid_harmonized.csv so the incumbent "
                         "grid is never overwritten")
    args = ap.parse_args()
    if args.ncat:
        globals()["NCAT"] = args.ncat
    base = ("county_composition_grid_harmonized" if args.harmonize
            else OUT.stem)
    base += f"_ncat{NCAT}"
    out_path = Path(args.out) if args.out else dc.LOG_DIR / f"{base}.csv"
    if args.nominal:
        # 51 deflates. Running nominal here makes the coefficient a different
        # object from the incumbent's, in nominal rather than real dollars per
        # head, so the two are not put side by side without saying which is which.
        dc.deflate = lambda year: 1.0
        p51.dc.deflate = lambda year: 1.0

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 78)
    print("  County grid, instrument built from contract COMPOSITION x national demand")
    print("=" * 78)

    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    drop_con = not args.include_construction

    # The treatment comes from 51, unmodified, so it is the same object in both
    # grids and any difference between them is the instrument alone.
    CON = p51.contract_matrices(units, 1966, 2018, drop_construction=drop_con)
    print(f"  {len(units):,} counties, 51's contract matrices built, "
          f"{time.time()-t0:.1f}s", flush=True)

    ARR, NAT, cats, cyears = category_panel(units, 1966, 2018,
                                            drop_construction=drop_con,
                                            harmonize=args.harmonize)
    print(f"  category panel: {len(cats)} FSC {NDIG}-digit categories, "
          f"{len(cyears)} years, {time.time()-t0:.1f}s", flush=True)

    STATE = B.drop_duplicates("ccg").set_index("ccg")["state"]
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]

    rows = []
    ref_seen = {}
    for era in args.era:
        cfg = p51.ERAS[era]
        for h in p51.HORIZONS:
            years = [t for t in range(max(cfg["lo"], cfg["first_base"] + h),
                                      cfg["hi"] + 1)]
            idx = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
            f = pd.DataFrame(index=idx).reset_index()
            f["state"] = f["ccg"].map(STATE)
            f["division"] = f["ccg"].map(DIVI)
            piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
                   for c in ("Y", "P", "E")}
            for c in ("Y", "P", "E"):
                f[c] = piv[c].stack().reindex(idx).to_numpy()
                f[c + "_b"] = piv[c].reindex(columns=[y - h for y in years]) \
                    .set_axis(years, axis=1).stack().reindex(idx).to_numpy()
            den_m = f["E"] + f["E_b"]
            f["y"] = np.where(den_m != 0, 2.0 * (f["E"] - f["E_b"]) / den_m, np.nan)
            f["invP"] = 1.0 / f["P_b"]

            for rule in p51.RULES:
                for floor, flab in p51.FLOORS:
                    C = CON[(rule, floor)]
                    lag = {k: C.reindex(columns=[y - k for y in years])
                           .set_axis(years, axis=1) for k in range(h + 1)}
                    f["G_b"] = lag[h].stack().reindex(idx).to_numpy()
                    allpos = (sum(lag[k] for k in range(h)) > 0).all(axis=1)
                    kept_units = set(units[allpos.to_numpy()])

                    gk = category_rates(NAT[(rule, floor)], h)
                    arr = ARR[(rule, floor)]
                    EXPO = {k: ("fixed", v) for k, v in cfg["expo"].items()}
                    EXPO.update({k: ("roll", v) for k, v in p51.ROLLING.items()})
                    LNUM = {}
                    for elab, (kind, spec) in EXPO.items():
                        L, nc = predicted_dollars(arr, gk, cyears, years, h,
                                                  kind, spec, p51.ROLL_LAG,
                                                  p51.DATA_FIRST)
                        LNUM[elab] = (L, nc, kind, spec)

                    for lam in p51.LAMBDAS:
                        wj = p51.lam_weights(h, lam)
                        win = sum(wj[k] * lag[k] for k in range(h))
                        f["G_win"] = win.stack().reindex(idx).to_numpy()
                        for screen in p51.SCREENS:
                            keep = (f["ccg"].isin(kept_units) if screen
                                    else pd.Series(True, index=f.index))
                            for X, xlab in p51.DENOMS:
                                xv = (f["G_win"] - f["G_b"]) / f[X + "_b"]
                                for elab, (L, nc, kind, spec) in LNUM.items():
                                    # A county absent from the contract records has
                                    # no contracts, so its predicted dollar change
                                    # is zero. A column whose window leaves the
                                    # data stays NaN and drops the year.
                                    Lf = np.where(np.isnan(L) & (nc[None, :] > 0),
                                                  0.0, L)
                                    zser = pd.DataFrame(Lf, index=units,
                                                        columns=years) \
                                        .stack().reindex(idx).to_numpy()
                                    zv = pd.Series(zser, index=f.index) / f[X + "_b"]
                                    if kind == "fixed":
                                        a_rep, b_rep = spec
                                    else:
                                        live = [t for t, n in zip(years, nc) if n]
                                        if not live:
                                            continue
                                        a_rep, b_rep = min(live), max(live)
                                    ok = (keep & np.isfinite(xv) & np.isfinite(zv)
                                          & np.isfinite(f["y"])).to_numpy()
                                    g = f[ok]
                                    if len(g) < 500:
                                        continue
                                    x0 = xv[ok].to_numpy(np.float64)
                                    z0 = zv[ok].to_numpy(np.float64)
                                    y0 = g["y"].to_numpy(np.float64)   # never capped
                                    f1 = pd.factorize(g["ccg"])[0].astype(np.int64)

                                    yr_ = g["year"].astype(str)
                                    FEV = {
                                        "year":
                                            pd.factorize(g["year"])[0].astype(np.int64),
                                        "divisionyear":
                                            pd.factorize(g["division"].astype(str) + "_"
                                                         + yr_)[0].astype(np.int64),
                                        "stateyear":
                                            pd.factorize(g["state"].astype(str) + "_"
                                                         + yr_)[0].astype(np.int64)}
                                    NOFE = np.zeros(len(g), np.int64)
                                    WV = {"none": np.ones(len(g))}
                                    for wl, kk in (("state-EW", "state"),
                                                   ("division-EW", "division")):
                                        tot = g.groupby([kk, "year"])["E_b"] \
                                               .transform("sum")
                                        v = (g["E_b"] / tot).to_numpy(np.float64)
                                        WV[wl] = v / v.mean()
                                    keys = {"pooled": np.zeros(len(g), np.int64),
                                            "year": g["year"].to_numpy(),
                                            "state": g["state"].to_numpy()}
                                    ip = g["invP"].to_numpy(np.float64)
                                    corr_raw = float(np.corrcoef(z0, ip)[0, 1])

                                    for wg in p51.WGROUPS:
                                        xw_ = p51.wins(x0, keys[wg])
                                        for felab, use_cty, f2key, wlab in p51.SPECS:
                                            try:
                                                r = t28.fit2sls(
                                                    y0, xw_, z0, WV[wlab],
                                                    f1 if use_cty else FEV[f2key],
                                                    FEV[f2key] if use_cty else NOFE,
                                                    f1.copy(), singleton=False)
                                                st_ = "ok"
                                            except Exception as exc:     # noqa: BLE001
                                                r = dict(beta=np.nan, se=np.nan,
                                                         F=np.nan)
                                                st_ = f"failed: {type(exc).__name__}"
                                            rows.append(dict(
                                                era=era, horizon=h, lam=lam,
                                                denom=xlab, exposure=elab,
                                                expo_first=a_rep, expo_last=b_rep,
                                                rule=rule, floor=flab, screen=screen,
                                                fe=felab, weight=wlab, winsor_group=wg,
                                                n_obs=len(g),
                                                n_counties=int(g["ccg"].nunique()),
                                                n_categories=len(cats),
                                                beta=r["beta"], se=r["se"],
                                                beta_scaled=r["beta"] * p51.SCALE[xlab],
                                                se_scaled=r["se"] * p51.SCALE[xlab],
                                                unit=p51.UNIT[xlab],
                                                t=(r["beta"] / r["se"]) if r["se"]
                                                else np.nan,
                                                first_stage_F=r["F"],
                                                corr_inv_pop=corr_raw,
                                                dollars="nominal" if args.nominal
                                                else "real", status=st_))
                                            if (era == REF["era"] and h == REF["horizon"]
                                                    and xlab == REF["denom"]
                                                    and rule == REF["rule"]
                                                    and flab == REF["floor"]
                                                    and screen == REF["screen"]
                                                    and elab == REF["exposure"]):
                                                ref_seen = dict(
                                                    n_obs=len(g),
                                                    n_counties=int(
                                                        g["ccg"].nunique()))
            print(f"  {era:>4} h={h}  {len(rows):>8,} cells  {time.time()-t0:>8.1f}s",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    print(f"\n  {len(out):,} cells, {int((~out['status'].eq('ok')).sum())} failed")
    print(f"  wrote {out_path}")

    # The reference cell. Without the zero-fill this comes back 117,975 on 3,025
    # counties, which is the 33 counties that never show a usable contract code.
    if REF["era"] in args.era and not args.skip_assert:
        if not ref_seen:
            raise SystemExit("  FAIL: the reference cell was never estimated; the "
                             "grid axes no longer contain it")
        okn = (ref_seen["n_obs"] == REF_N
               and ref_seen["n_counties"] == REF_COUNTIES)
        print(f"\n  reference cell {REF}")
        print(f"    n_obs      {ref_seen['n_obs']:>9,} against {REF_N:>9,}"
              f"   {'PASS' if ref_seen['n_obs'] == REF_N else 'FAIL'}")
        print(f"    counties   {ref_seen['n_counties']:>9,} against "
              f"{REF_COUNTIES:>9,}   "
              f"{'PASS' if ref_seen['n_counties'] == REF_COUNTIES else 'FAIL'}")
        if not okn:
            raise SystemExit("  FAIL: the composition instrument is not on the "
                             "incumbent's sample. A county with no contracts has a "
                             "ZERO shock, not a missing one; check the zero-fill.")

    # The two diagnostics that must not be answered from memory.
    good = out[out["status"].eq("ok")]
    if len(good):
        print(f"\n  correlation with 1/P across cells: median "
              f"{good['corr_inv_pop'].median():.4f}, "
              f"max |.| {good['corr_inv_pop'].abs().max():.4f}")
        print("  a high F says the instrument is not weak; it does NOT say the "
              "instrument\n  carries procurement information. The contract-free "
              "placebo reaches 34.5 to 35.5\n  off the shared 1/P denominator, and "
              "is not a floor an orthogonal instrument\n  must clear.")
    q = " and ".join(f"{k}=={v!r}" for k, v in HEADLINE.items())
    hit = out.query(q) if len(out) else out
    if len(hit) == 1:
        r = hit.iloc[0]
        print(f"\n  HEADLINE CELL  {HEADLINE}")
        print(f"    {'':<14}{'beta':>9}{'se':>8}{'t':>7}{'F':>8}{'N':>10}")
        print(f"    {'composition':<14}{r['beta_scaled']:>9.3f}{r['se_scaled']:>8.3f}"
              f"{r['t']:>7.2f}{r['first_stage_F']:>8.1f}{r['n_obs']:>10,}")
        if INCUMBENT.exists():
            twin = pd.read_csv(INCUMBENT).query(q)
            if len(twin) == 1:
                w = twin.iloc[0]
                same = "same sample" if w["n_obs"] == r["n_obs"] else "DIFFERENT sample"
                print(f"    {'incumbent':<14}{w['beta_scaled']:>9.3f}"
                      f"{w['se_scaled']:>8.3f}{w['t']:>7.2f}"
                      f"{w['first_stage_F']:>8.1f}{w['n_obs']:>10,}   {same}")
    else:
        print(f"\n  NOTE: the headline cell is not in this run's output")
    if INCUMBENT.exists():
        print(f"\n  join to the incumbent on era, horizon, lam, denom, exposure, "
              f"rule,\n  floor, screen, fe, weight, winsor_group: {INCUMBENT}")
    else:
        print(f"\n  NOTE: {INCUMBENT.name} is not on disk, so no twin grid to join to")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
