#!/usr/bin/env python3
"""From Nakamura and Steinsson's specification to ours, one change per row.

WHAT THIS FILE IS FOR

50 and 53 reproduce NS at their own rung. Nothing walks from there to the
specification this project uses, so the distance between their number and ours is
undocumented. This file is that walk: row 1 IS 53's replication, every row in
between changes one thing, and the whole ladder stays at the STATE rung, which is
where they ran it.

TWO COLUMNS, ONE SET OF ROWS

    single        one predetermined exposure times one national series. This is
                  the county analogue of NS's own instrument, and the note in
                  document/ shows why such a design is rank one in the cross
                  section: the national series decides which YEARS count, never
                  which units are treated more than which.

    composition   the unit's mix across the top 800 categories of the
                  harmonized code panel (54's baseline construction: the 1979
                  service bridge, the 1989 renames, and the 2012 crosswalk
                  applied) meeting the national change in demand for each. The
                  shares do not sum to one, they sum to contract dollars per unit
                  of the scaling denominator, so the share sum is partialled out,
                  which is what Borusyak, Hull and Jaravel require of incomplete
                  shares.

Replacing one national series with 300 is the COLUMN, not a row. Every row means
the same thing in both columns, including row 4: when the treatment becomes a
window average the national category rates become window averages too, so that
row moves one thing rather than moving the treatment in both columns and the
instrument in only one.

Two rows do different amounts of work in the two columns, and the table says so.
Row 7 does less in the composition column, since the scaling denominator enters
the single-shift instrument in three places and the composition instrument in
one. Row 6 does MORE: NS's basis divides by the national figure, which for one
shock is a common scalar that provably cannot move beta, and across 300
categories is a reweighting of them against each other that does move it.

ONE ANCHOR, AND ONE END THAT HAS NONE

    row 1   single    NS's own specification, checked against 53's output file

This file re-implements the panel, the contracts, the exposure and the estimator
rather than calling 53's, so an independent implementation reproducing 53's number
is what makes the rows below it a path rather than a third codebase that resembles
the other two.

The LAST row has no such check. No committed producer estimates this specification
at the state rung: 51 and 54 are county grids, and 50 and 53 both run NS's
specification, not ours. So a construction error introduced in the middle of the
ladder would leave row 1 landing on NS exactly and nothing would catch it at the
bottom. That gap is printed on every run rather than left to be noticed. Giving it
a real anchor means adding a state-rung cell to one of the county grids.

The composition column has no anchor at either end for the same reason.

WHAT IS NOT RUN

The reversal. Protocol B.3 pairs a ladder with a leave-one-out, because a path
shows each change given what precedes it and only the reversal shows each change
given everything else. It is not run here and its absence is printed on every
run rather than left to be noticed.

THE RULE THE ROWS FOLLOW                              protocol B.3

One change per row, so movement can be attributed, EXCEPT where changing one
thing alone would leave a row that is not a legitimate estimate. Those rows
declare a bundle and name what is inside it. There are four here, and they are
NOT all the same kind of bundle, which the table has to say rather than hide:

    raw employment      B.3.2's own case. A level outcome with no growth rate
                        chosen is not an estimate at all, so the two move together
                        and the step between them does not exist.

    the sample window   merged for LENGTH. Both halves are legitimate on their own.
    the rolling window  merged for LENGTH.
    the exposure basis  merged for LENGTH.

Compression is not what B.3.2's escape hatch is for, so the three length merges are
declared as such and `--full` runs them as separate rows, which is where their
separate values come from. The run fails if any other row moves more than one key.

WHAT THE SCREEN DOES HERE

Nothing. A.3.4's exclusion is applied only where a row asks for it, and no row does,
because every row is at the state rung and no state has zero covered employment. The
state totals therefore sum every county group, the 42 with a zero EFSY year included.
That makes the state rung a slightly larger aggregate than the sum of the counties
51 estimates on, which matters only if a row is ever anchored to a county grid.

WHAT IS NOT A ROW

The contract definition. NS describe DD-350's own reporting thresholds and we
keep every action and net the deobligations, which is what 13 already builds, so
there is no step between their definition and ours to show.

Output:
    batch_logs/macro_version/journey_ladder.csv
    batch_logs/macro_version/journey_ladder_full.csv        with --full
    batch_logs/macro_version/52_journey_ladder.log
    4_Macro_version/Results/ns14_diagnostics/Table_journey_ladder.tex
"""

from __future__ import annotations

import argparse
import importlib.util
import re
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
OUT = dc.LOG_DIR / "journey_ladder.csv"
OUT_FULL = dc.LOG_DIR / "journey_ladder_full.csv"
TAB_DIR = dc.MACRO_RESULTS_ROOT / "ns14_diagnostics"

POP_BASE_YEAR = 1975            # 57's fixed weighting year, and predetermined here too
ROLL_LAG = 5                    # 51's rolling rule: the window ends five years before the base
COMP_CLIP = 1.0                 # 54's clip on a national category rate
FIXED_SHARE = (1966, 1971)      # their code: year > 1965 & year < 1972
DATA_FIRST = {"state": 1966, "county": 1969}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------
# The ladder. Every row names every key, so any row is reproducible on its own and
# the diff between two rows is machine-checkable rather than a matter of reading.
# --------------------------------------------------------------------------------

BASE = dict(rung="state", emp="ces", outcome="per_capita", growth="standard",
            treat="endpoint", scale="gdp", share="fixed", share_basis="ns",
            percap_diff=True, shift="national",
            fe="unit_year", weight="none", winsor="none", sample="all",
            first_year=None, last_year=2006,
            # The horizon and the share window are ROWS, not settings passed in
            # from outside. The horizon becomes a real choice only once the
            # treatment is a window average, so it moves immediately after that
            # row; the window length becomes a choice only once the share is
            # rolling, so it moves immediately after that one. Holding either
            # fixed outside the ladder would hide two of the changes that carry
            # the distance between NS's number and ours.
            horizon=2, share_span=5)

LADDER = (
    ("1",  "NS's Bartik on their specification, at their own rung", {}),
    ("2",  "1977 to 2018, the years EFSY covers",
     dict(first_year=1977, last_year=2018), ("first_year", "last_year"),
     "nine years off the front, the Vietnam drawdown among them, eleven on at the back"),
    ("3",  "EFSY employment in place of CES", dict(emp="efsy")),
    ("4",  "window average in place of the endpoint", dict(treat="window")),
    ("5",  "horizon of four years in place of two", dict(horizon=4), (),
     "a longer horizon needs history in front of each base year"),
    ("6",  "rolling ten-year share in place of the fixed 1966-71 one",
     dict(share="rolling", share_span=10), ("share", "share_span"),
     "a rolling ten-year window needs history before each base year"),
    ("7",  "exposure over the scaling denominator, in levels not per capita",
     dict(share_basis="intensity", percap_diff=False),
     ("share_basis", "percap_diff")),
    ("8",  "population scaling in place of state output", dict(scale="population"),
     (), "population reaches further back than state output does"),
    ("9",  "raw employment, growth measured by MPGR",
     dict(outcome="level", growth="midpoint"), ("outcome", "growth")),
    ("10", "drop the state fixed effect", dict(fe="year")),
    ("11", "region-by-year effects", dict(fe="regionyear")),
    ("12", "weight by population, fixed at 1975", dict(weight="pop_fixed")),
)

# The ladder ends unwinsorized because the cross-level exercise it walks toward
# (57's baseline) estimates on raw outcome and shock at every rung: a per-level
# winsor would break the reconstruction identity 57 asserts. At 51 units a year
# a 1/99 winsor would clip the most and least exposed state each year whether
# or not either is an outlier, moving 70 of 1,785 observations.

# The three length merges, expanded by --full. Each entry replaces its row with the
# sub-steps listed, so the compact table and the long one walk the same path and end
# on the same config; only the number of stops differs.
SPLITS = {
    "2": (("2a", "restrict to 1977-2006, the window EFSY covers",
           dict(first_year=1977), (),
           "nine years off the front, the Vietnam drawdown among them"),
          ("2b", "extend the sample to 2018", dict(last_year=2018), (),
           "eleven years on at the back")),
    "6": (("6a", "rolling five-year share in place of the fixed 1966-71 one",
           dict(share="rolling"), (),
           "a rolling window needs history before each base year"),
          ("6b", "ten-year share window in place of five", dict(share_span=10), (),
           "a longer share window needs more history still")),
    "7": (("7a", "exposure as contracts over the scaling denominator",
           dict(share_basis="intensity")),
          ("7b", "difference in levels, not per capita, off the base year",
           dict(percap_diff=False))),
}


def resolve_ladder(full=False):
    """Walk the deltas into full configs and enforce B.3.1 while doing it."""
    items = []
    for item in LADDER:
        items.extend(SPLITS[item[0]] if full and item[0] in SPLITS else [item])
    rows, cur = [], dict(BASE)
    for item in items:
        lab, desc, delta = item[0], item[1], item[2]
        bundle = item[3] if len(item) > 3 else ()
        why = item[4] if len(item) > 4 else None
        moved = tuple(k for k, v in delta.items() if cur.get(k) != v)
        if len(moved) > 1 and set(moved) != set(bundle):
            raise SystemExit(f"row {lab} moves {moved} but declares bundle {bundle}; "
                             "protocol B.3.1 allows one change per row and B.3.2 "
                             "allows a bundle only where the row would otherwise "
                             "not be a legitimate estimate")
        cur = {**cur, **delta}
        rows.append((lab, desc, dict(cur), bundle, why))
    return rows


# --------------------------------------------------------------------------------
# The panel. Built once at the county level with NO screen applied, so a row that
# does not ask for A.3.4's exclusion gets the full universe and the state rung is
# summed from every county rather than from a screened subset.
# --------------------------------------------------------------------------------

def county_frame():
    b14 = _load("b14", CODE / "14_build_instrument.py")
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"]
    st = (xw.assign(ccg=xw["ccg_id"].astype(int),
                    state=xw["county_fips5"].str[:2].astype(int))
          .drop_duplicates("ccg")[["ccg", "state"]])
    st = st[st["state"].isin(b14.DIVISION)]

    inc = pd.read_parquet(dc.PANEL_OUT_DIR / "income_ccg_year.parquet",
                          columns=["ccg", "year", "income_real"])
    pop = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    efs = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    bea = pd.read_parquet(dc.PANEL_OUT_DIR / "bea_employment_ccg_year.parquet")

    d = (inc.merge(pop, on=["ccg", "year"])
            .merge(efs, on=["ccg", "year"], how="left")
            .merge(bea, on=["ccg", "year"], how="left")
            .merge(st, on="ccg"))
    d = d.rename(columns={"income_real": "Y", "population": "P",
                          "emp": "E_efsy", "bea_emp": "E_bea"})
    d["division"] = d["state"].map(b14.DIVISION)
    # A.3.4's units, computed once and applied only where a row asks for it.
    z = d.loc[d["E_efsy"].notna() & (d["E_efsy"] <= 0), "ccg"].unique()
    return d.sort_values(["ccg", "year"]).reset_index(drop=True), set(z)


def contracts_by_unit(units_index, level, first, last):
    """51's contract matrices, kept at the county or summed to the state."""
    p51 = _load("p51", CODE / "51_county_multiplier_grid.py")
    C = p51.contract_matrices(np.array(sorted(units_index["ccg"].unique())), first, last)
    C = C[("net", 0.0)]                         # the definition 13 already builds
    if level == "county":
        return C
    key = units_index.drop_duplicates("ccg").set_index("ccg")["state"]
    return C.groupby(key.reindex(C.index)).sum()


_CAT_PANEL = {}


def category_panel_by_unit(units_index, level, first, last):
    """county x category x year contract dollars, kept at the county or summed
    to the state. 54 owns the construction; this only changes the rung.

    Returns (ARR, cats, years, units) with ARR aligned to `units` in sorted order.
    """
    key = ("panel", first, last)
    if key not in _CAT_PANEL:
        p54 = _load("p54", CODE / "54_composition_instrument_grid.py")
        p51 = _load("p51", CODE / "51_county_multiplier_grid.py")
        cty = np.array(sorted(units_index["ccg"].unique()))
        ARR, _, cats, cy = p54.category_panel(cty, first, last, harmonize=True)
        _CAT_PANEL[key] = (ARR[("net", 0.0)].astype(np.float64), cats, list(cy), cty)
    arr, cats, cy, cty = _CAT_PANEL[key]
    if level == "county":
        return arr, cats, cy, cty
    st = units_index.drop_duplicates("ccg").set_index("ccg")["state"]
    codes = st.reindex(cty).to_numpy()
    states = np.array(sorted(pd.unique(codes[pd.notna(codes)])))
    pos = {v: i for i, v in enumerate(states)}
    out = np.zeros((len(states), arr.shape[1], arr.shape[2]), np.float64)
    for i, c in enumerate(codes):
        if pd.notna(c):
            out[pos[c]] += arr[i]
    return out, cats, cy, states


def category_rates(arr, cy, h, treat):
    """The national change in each category, scaled by that category's own mean
    size and clipped, exactly as 54 defines it.

    `treat` follows the ladder's own row: when the treatment becomes a window
    average the category rate becomes one too, so that row means one thing in
    both columns rather than moving the treatment in both and the instrument in
    only one of them.
    """
    v = arr.sum(axis=0)                                   # (K, T) national
    out = np.full_like(v, np.nan)
    if treat == "endpoint":
        out[:, h:] = v[:, h:] - v[:, :-h]
    else:
        win = np.full_like(v, np.nan)
        for j in range(h - 1, v.shape[1]):
            win[:, j] = v[:, j - h + 1:j + 1].mean(axis=1)
        out[:, h:] = win[:, h:] - v[:, :-h]
    scale = np.nanmean(np.abs(v), axis=1, keepdims=True)
    scale[scale == 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(out / scale, -COMP_CLIP, COMP_CLIP)


_STATE_EXTRAS = {}


def state_extras():
    """State GDP and CES employment, from 53, which owns their construction."""
    if not _STATE_EXTRAS:
        p53 = _load("p53", CODE / "53_ns14_exact_replication.py")
        _STATE_EXTRAS["gdp"] = p53.state_gdp()
        _STATE_EXTRAS["ces"] = p53.ces_employment()
        _STATE_EXTRAS["pop"] = p53.state_population()
    return _STATE_EXTRAS


def unit_panel(cfg, county, zero_ccgs):
    """Collapse the county frame to the row's rung, applying A.3.4 if asked."""
    d = county
    if cfg["sample"] == "drop_zero_emp":
        d = d[~d["ccg"].isin(zero_ccgs)]
    E = "E_bea" if cfg["emp"] == "bea" else "E_efsy"
    d = d.rename(columns={E: "E"})
    keep = ["ccg", "year", "Y", "P", "E", "state", "division"]
    if cfg["rung"] == "county":
        return d[keep].rename(columns={"ccg": "unit"}), d
    g = (d.groupby(["state", "year"], as_index=False)[["Y", "P", "E"]].sum(min_count=1)
           .rename(columns={"state": "unit"}))
    g["state"] = g["unit"]
    g["division"] = g["unit"].map(d.drop_duplicates("state").set_index("state")["division"])
    # The county frame begins in 1969 because BEA county income does. State GDP
    # reaches 1963 and SAINC population 1929, so the state rung is not bound by
    # that and is widened to the years its own inputs cover before they merge in.
    x = state_extras()
    yrs = range(min(int(x["gdp"].year.min()), int(g["year"].min())), 2019)
    full = pd.MultiIndex.from_product([sorted(g["unit"].unique()), yrs],
                                      names=["unit", "year"])
    g = (g.set_index(["unit", "year"]).reindex(full).reset_index())
    for c in ("state", "division"):
        g[c] = g["unit"].map(g.dropna(subset=[c]).drop_duplicates("unit")
                             .set_index("unit")[c])
    # GDP and CES exist only at this rung, and only rows 1 to 6 ask for them; by
    # the time the ladder reaches the county it has already moved to population
    # and to EFSY, so nothing downstream needs a county analogue of either.
    g = g.merge(x["gdp"].rename(columns={"state": "unit"}), on=["unit", "year"], how="left")
    g["gdp"] = g["gdp"] * 1e6 * g["year"].map(dc.deflate)
    g = g.merge(x["pop"].rename(columns={"state": "unit"}), on=["unit", "year"], how="left")
    g["P"] = g["pop"].fillna(g["P"])                 # SAINC reaches 1929; county sums do not
    ces = x["ces"].rename(columns={"state": "unit", "emp": "E_ces"})
    g = g.merge(ces, on=["unit", "year"], how="left")
    g["E_ces"] = g["E_ces"] * 1000.0                 # CES is in thousands of jobs
    if cfg["emp"] == "ces":
        g["E"] = g["E_ces"]
    return g, d


# --------------------------------------------------------------------------------
# One row's variables. Levels throughout: x = (spending change) / X at the base
# year, which equals 50's per-capita construction (dG/P)/(Y/P) exactly when
# X = Y, and generalises to any denominator without a per-capita detour.
# --------------------------------------------------------------------------------

def make_vars(cfg, panel, C, h=None, CAT=None):
    h = cfg["horizon"] if h is None else h
    X = {"income": "Y", "population": "P", "employment": "E",
         "gdp": "gdp"}[cfg["scale"]]
    units = np.array(sorted(panel["unit"].unique()))
    have = [c for c in ("Y", "P", "E", "gdp") if c in panel.columns]
    piv = {c: panel.pivot(index="unit", columns="year", values=c).reindex(units)
           for c in have}
    C = C.reindex(units)
    # Two different first years. `first` is what the DATA support and governs how
    # far a share window may reach back; `first_year` is the earliest OUTCOME year
    # a row chooses to estimate on. Conflating them makes a restriction meant for
    # one rung silently shorten another rung's lookback.
    first = DATA_FIRST[cfg["rung"]]
    lo_y = max(first + h, panel["year"].min() + h, cfg.get("first_year") or 0)
    hi_y = cfg["last_year"]
    years = [t for t in range(lo_y, hi_y + 1)
             if t in piv["E"].columns and (t - h) in piv["E"].columns]
    idx = pd.MultiIndex.from_product([units, years], names=["unit", "year"])
    f = pd.DataFrame(index=idx).reset_index()
    attr = panel.drop_duplicates("unit").set_index("unit")[["state", "division"]]
    f["state"] = f["unit"].map(attr["state"])
    f["division"] = f["unit"].map(attr["division"])

    # The weight's population is read at a FIXED calendar year, carried on every
    # row of that unit in every year, so the weighting cannot itself respond to the
    # treatment. A weight taken at t-h would move with the shock through whatever
    # the shock does to population.
    if POP_BASE_YEAR not in piv["P"].columns:
        raise SystemExit(f"  FAIL: population has no {POP_BASE_YEAR} at the "
                         f"{cfg['rung']} rung")
    f["P0"] = f["unit"].map(piv["P"][POP_BASE_YEAR])

    for c in have:
        f[c] = piv[c].stack().reindex(idx).to_numpy()
        f[c + "_b"] = (piv[c].reindex(columns=[t - h for t in years])
                       .set_axis(years, axis=1).stack().reindex(idx).to_numpy())

    # outcome
    if cfg["outcome"] == "per_capita":
        L, Lb = f["E"] / f["P"], f["E_b"] / f["P_b"]
        f["y"] = (L - Lb) / Lb
    elif cfg["growth"] == "standard":
        f["y"] = (f["E"] - f["E_b"]) / f["E_b"]
    else:
        den = f["E"] + f["E_b"]
        f["y"] = np.where(den != 0, 2.0 * (f["E"] - f["E_b"]) / den, np.nan)

    # treatment
    lag = {k: C.reindex(columns=[t - k for t in years]).set_axis(years, axis=1)
           for k in range(h + 1)}
    f["G_b"] = lag[h].stack().reindex(idx).to_numpy()
    win = (lag[0] if cfg["treat"] == "endpoint"
           else sum(lag[k] for k in range(h)) / h)
    f["G_w"] = win.stack().reindex(idx).to_numpy()
    if cfg.get("percap_diff", False):
        # NS divide each year's spending by THAT year's population, so the two
        # ends of the difference carry different deflators. Working in levels
        # deflates both by the base year's instead, which is not the same object.
        f["x"] = ((f["G_w"] / f["P"] - f["G_b"] / f["P_b"]) / (f[X + "_b"] / f["P_b"]))
    else:
        f["x"] = (f["G_w"] - f["G_b"]) / f[X + "_b"]

    # the shift, national or leave-one-out
    loo = cfg["shift"] != "national"
    def agg(col):
        t = f.groupby("year")[col].transform("sum")
        return t - f[col] if loo else t
    gw, gb, nx = agg("G_w"), agg("G_b"), agg(X + "_b")
    if cfg.get("percap_diff", False):
        pn, pb = agg("P"), agg("P_b")
        shift = (gw / pn - gb / pb) / (nx / pb)
    else:
        shift = (gw - gb) / nx

    # the share, fixed or rolling
    ns_basis = cfg.get("share_basis", "intensity") == "ns"

    def share_over(yy):
        """NS: mean per-capita spending over the window, relative to the national
        mean per-capita spending over the same window. The national term is one
        number common to every unit, so it rescales a just-identified instrument
        and cannot move beta; what it fixes is the BASIS, per head rather than per
        dollar of the scaling denominator. Intensity: contracts over X, which is
        what the county design uses."""
        if ns_basis:
            pc = C.loc[:, yy] / piv["P"].loc[:, yy]
            nat = (C.loc[:, yy].sum(axis=0) / piv["P"].loc[:, yy].sum(axis=0)).mean()
            return pc.mean(axis=1) / nat
        return (C.loc[:, yy].sum(axis=1)
                / piv[X].loc[:, yy].sum(axis=1, min_count=len(yy)))

    # ------------------------------------------------------------------
    # The column. `single` multiplies one exposure by one national series, which
    # is the county analogue of NS's own instrument. `composition` meets the
    # unit's category mix with the national change in demand for each category.
    # Every window rule above is shared, so a row means the same thing in both.
    # ------------------------------------------------------------------
    comp = cfg["instrument"] == "composition"
    if comp:
        if CAT is None:
            return None
        arr, cats, cy, cunits = CAT
        pos = {u: i for i, u in enumerate(cunits)}
        take = np.array([pos.get(u, -1) for u in units])
        if (take < 0).any():
            return None
        arr = arr[take]
        gk = category_rates(arr, cy, h, cfg["treat"])
        ycol = {y: j for j, y in enumerate(cy)}

        def comp_share(yy):
            """The unit's category mix over the window, on the row's own basis.

            NS basis: category dollars per head, each category divided by the
            national per-head figure for THAT category. For one shock the
            national term is a common scalar and cannot move beta; across 300
            categories it reweights them against each other and does move it,
            which is the single-factor and multicomponent distinction showing up
            inside a row rather than in the header.
            """
            js = [ycol[y] for y in yy]
            num = arr[:, :, js].sum(axis=2)                       # (n_units, K)
            if ns_basis:
                pd_ = piv["P"].loc[:, yy].sum(axis=1).to_numpy(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pc = num / pd_[:, None]
                    nat = num.sum(axis=0) / pd_.sum()
                    nat = np.where(nat == 0, np.nan, nat)
                    return pc / nat[None, :]
            den = piv[X].loc[:, yy].sum(axis=1, min_count=len(yy)).to_numpy(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                return num / den[:, None]

        zc = np.full((len(units), len(years)), np.nan)
        s0 = np.full((len(units), len(years)), np.nan)
        for j, t in enumerate(years):
            if cfg["share"] == "fixed":
                yy = [y for y in range(FIXED_SHARE[0], FIXED_SHARE[1] + 1) if y in cy]
            else:
                span = cfg["share_span"]
                a, b = t - h - ROLL_LAG - span + 1, t - h - ROLL_LAG
                yy = [y for y in range(a, b + 1)]
                if a < first or not all(y in cy for y in yy):
                    continue
            if t not in ycol:
                continue
            g = gk[:, ycol[t]]
            live = np.isfinite(g)
            if not live.any():
                continue
            sh = comp_share(yy)
            zc[:, j] = np.nan_to_num(sh[:, live], nan=0.0) @ g[live]
            s0[:, j] = np.nansum(sh, axis=1)
        zs = pd.DataFrame(zc, index=units, columns=years).stack().reindex(idx)
        ss = pd.DataFrame(s0, index=units, columns=years).stack().reindex(idx)
        f["z"] = zs.to_numpy()
        f["s0"] = ss.to_numpy()
    elif cfg["share"] == "fixed":
        yy = [y for y in range(FIXED_SHARE[0], FIXED_SHARE[1] + 1) if y in C.columns]
        f["z"] = f["unit"].map(share_over(yy)) * shift
        f["s0"] = 0.0
    else:
        cols = {}
        for t in years:
            span = cfg["share_span"]
            a, b = t - h - ROLL_LAG - span + 1, t - h - ROLL_LAG
            yy = [y for y in range(a, b + 1)]
            if a < first or not all(y in C.columns for y in yy):
                continue
            cols[t] = share_over(yy)
        if not cols:
            return None
        sv = pd.DataFrame(cols).reindex(units).stack().reindex(idx).to_numpy()
        f["z"] = pd.Series(sv, index=f.index) * shift
        f["s0"] = 0.0

    need = ["x", "y", "z"] + (["s0"] if comp else [])
    ok = np.isfinite(f[need].to_numpy()).all(axis=1)
    return f[ok].reset_index(drop=True)


_REGION = {}


def region_of():
    """Division to Census region, read from 57 rather than restated here, so the
    region-by-year effect on this ladder is the same object as the one the
    aggregation baseline uses."""
    if not _REGION:
        _REGION.update(_load("p57", CODE / "57_aggregation_invariance.py").REGION)
    return _REGION


def wins_p1p99(v, key):
    """Winsorize within `key`, and report how many observations actually moved.

    The count is not decoration. At the state rung there are 51 units in a year, so
    numpy's 1st percentile falls half way between the smallest and the second
    smallest and the 99th half way between the second largest and the largest.
    Exactly one unit per tail per year exceeds the cutoff, which makes this a
    min-max clip wearing a percentile's name, and it bites every year whether or
    not that year has an outlier. Printing the count is how a reader sees that.
    """
    out = v.astype(np.float64).copy()
    for g in np.unique(key):
        m = key == g
        lo, hi = np.nanpercentile(out[m], [1.0, 99.0])
        out[m] = np.clip(out[m], lo, hi)
    return out, int((out != v.astype(np.float64)).sum())


SCALE = {"income": 1.0, "gdp": 1.0, "population": 1e5, "employment": 1e5}
UNIT = {"income": "pp per 1% of income", "gdp": "pp per 1% of output",
        "population": "pp per $1,000 per head",
        "employment": "pp per $1,000 per job"}


def fit(cfg, g, t28):
    y = g["y"].to_numpy(np.float64)
    x = g["x"].to_numpy(np.float64)
    z = g["z"].to_numpy(np.float64)
    n_clipped = 0
    if cfg["winsor"] == "p1p99_year":
        x, n_clipped = wins_p1p99(x, g["year"].to_numpy())

    yr = g["year"].astype(str)
    unit = pd.factorize(g["unit"])[0].astype(np.int64)
    yrf = pd.factorize(g["year"])[0].astype(np.int64)
    if cfg["fe"] == "unit_year":
        f1, f2 = unit, yrf
    elif cfg["fe"] == "year":
        f1, f2 = yrf, np.zeros(len(g), np.int64)
    elif cfg["fe"] == "divisionyear":
        f1 = pd.factorize(g["division"].astype(str) + "_" + yr)[0].astype(np.int64)
        f2 = np.zeros(len(g), np.int64)
    else:
        # Four regions rather than nine divisions. At the state rung a
        # division-by-year cell holds three to nine states, which leaves little
        # cross-sectional variation standing; a region-by-year cell holds nine to
        # sixteen.
        reg = g["division"].map(region_of()).astype(str)
        f1 = pd.factorize(reg + "_" + yr)[0].astype(np.int64)
        f2 = np.zeros(len(g), np.int64)

    if cfg["weight"] == "none":
        w = np.ones(len(g))
    elif cfg["weight"] == "pop_fixed":
        w = g["P0"].to_numpy(np.float64)
        w = w / w.mean()
    else:
        tot = g.groupby(["division", "year"])["E_b"].transform("sum")
        w = (g["E_b"] / tot).to_numpy(np.float64)
        w = w / w.mean()

    # The share sum. Our category shares sum to total contract dollars per unit
    # of the scaling denominator rather than to one, which is the incomplete-shares
    # case: Borusyak, Hull and Jaravel require the sum to be held constant or
    # controlled. Partialling it out is their remedy, and it is applied only to the
    # composition column, where the shares exist. It is NOT a device for making the
    # two columns orthogonal; the columns are allowed to overlap, and that they
    # largely do not is a fact about the data rather than something imposed here.
    if cfg["instrument"] == "composition":
        w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
        M = w22.demean(np.column_stack([y, x, z, g["s0"].to_numpy(np.float64)]),
                       f1, f2, w)
        y, x, z, st = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        sw = float(st @ (w * st))
        if sw > 0:
            y = y - float(st @ (w * y)) / sw * st
            x = x - float(st @ (w * x)) / sw * st
            z = z - float(st @ (w * z)) / sw * st
        f1 = np.zeros(len(g), np.int64)
        f2 = np.zeros(len(g), np.int64)

    r = t28.fit2sls(y, x, z, w, f1, f2, unit.copy(), singleton=False)
    sc = SCALE[cfg["scale"]]
    return dict(n_obs=len(g), n_units=int(g["unit"].nunique()), n_clipped=n_clipped,
                first_year_obs=int(g["year"].min()),
                last_year_obs=int(g["year"].max()),
                beta=r["beta"], se=r["se"],
                beta_scaled=r["beta"] * sc, se_scaled=r["se"] * sc,
                t=r["beta"] / r["se"] if r["se"] else np.nan,
                first_stage_F=r["F"], unit_label=UNIT[cfg["scale"]])


def anchors():
    """Read from 53's OWN output file, never typed here. Row 1 is its replication
    of NS Table 3 col 2. There is no second anchor: nothing else in the repo
    estimates this ladder's last row, and `main` prints that rather than letting
    the silence read as a passing check."""
    a = pd.read_csv(dc.LOG_DIR / "ns14_exact_replication.csv")
    a = a[a.outcome.eq("employment") & a.share_first.eq(1966) & a.share_last.eq(1971)].iloc[0]
    return {"1": dict(beta=float(a.beta), se=float(a.se))}


def tex_escape(t):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_")):
        t = t.replace(a, b)
    # A year range takes an en dash; a hyphen in the source would set as one.
    return re.sub(r"\b(\d{4})-(\d{2,4})\b", r"\1--\2", t)


def write_tex(out, full):
    """The paper's table. Two columns of estimates, one block of rows.

    The unit of the coefficient CHANGES inside the table, at the row where the
    scaling denominator moves from state output to population, and a reader who
    misses that will compare two numbers that are not the same object. So the
    unit is written across the table as its own line wherever it changes, rather
    than parked in a note or repeated in a column nobody reads.
    """
    cols = [c for c in ("single", "composition") if c in set(out["instrument"])]
    if len(cols) < 2:
        return None
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    dest = TAB_DIR / ("Table_journey_ladder_full.tex" if full
                      else "Table_journey_ladder.tex")

    body, unit = [], None
    for lab in out["row"].drop_duplicates():
        blk = {c: out[out["row"].eq(lab) & out["instrument"].eq(c)] for c in cols}
        ref = blk[cols[0]]
        if ref.empty:
            continue
        r0 = ref.iloc[0]
        if r0["unit_label"] != unit:
            unit = r0["unit_label"]
            pretty = {"pp per 1% of output": r"percentage points per 1\% of state output",
                      "pp per $1,000 per head":
                          r"percentage points per \$1{,}000 of spending per head",
                      "pp per 1% of income": r"percentage points per 1\% of income",
                      "pp per $1,000 per job":
                          r"percentage points per \$1{,}000 per job"}.get(unit,
                                                                          tex_escape(unit))
            if body:
                body.append(r"\addlinespace")
            body.append(r"\multicolumn{6}{l}{\textit{Coefficient in %s}} \\" % pretty)

        def cell(c, key, fmt):
            d = blk[c]
            v = float(d.iloc[0][key]) if len(d) else float("nan")
            return format(v, fmt) if np.isfinite(v) else ""

        desc = tex_escape(f"{lab}. {r0['change']}")
        body.append(f"{desc} & {int(r0['n_obs']):,} & "
                    + " & ".join(f"{cell(c, 'beta_scaled', '.3f')} & "
                                 f"{cell(c, 'first_stage_F', '.1f')}" for c in cols)
                    + r" \\")
        se = " & ".join(f"({cell(c, 'se_scaled', '.3f')}) & " for c in cols)
        body.append(f" & & {se}" + r"\\")

    clipped = out[out["n_clipped"] > 0]
    clipnote = ""
    if len(clipped):
        c0 = clipped.iloc[0]
        yrs = max(int(c0["last_year_obs"]) - int(c0["first_year_obs"]) + 1, 1)
        clipnote = (" The winsor in the last row moves %d of %s observations, %.1f a "
                    "year: with 51 states the first and ninety-ninth percentiles fall "
                    "between the extreme observation and its neighbour, so it clips the "
                    "most and the least exposed state every year whether or not either "
                    "is an outlier."
                    % (int(c0["n_clipped"]), f"{int(c0['n_obs']):,}",
                       int(c0["n_clipped"]) / yrs))

    merged = ("" if full else
              " Three rows each carry two changes that were merged for length, the "
              "sample window, the rolling window and the exposure basis; running the "
              "producer with \\texttt{-{}-full} splits them and reaches the same "
              "last row.")

    dest.write_text(r"""\begin{table}[!htbp]
\caption{From Nakamura and Steinsson's Specification to Ours}
\label{tab:journey_ladder}
{\centering
\adjustbox{max width=\linewidth}{%
\begin{tabular}{lrrrrr}
\toprule
 & & \multicolumn{2}{c}{Single shift} & \multicolumn{2}{c}{Composition} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textit{Change from the row above} & $N$ & $\hat{\beta}$ & $F$ & $\hat{\beta}$ & $F$ \\
 & (1) & (2) & (3) & (4) & (5) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\par}
\medskip
{\scriptsize \noindent\textit{Notes:} Every row is a state-year panel of 51 states
estimated by two-stage least squares, and every row differs from the row above it in
exactly one choice, so the movement down a column is attributable.%s Row 1 reproduces
Nakamura and Steinsson (2014), Table 3 column 2, whose published employment coefficient
is 1.81 with a standard error of 0.41; it is checked against the replication producer's
own output file on every run. Column pair (2)--(3) instruments with one predetermined
exposure times one national procurement series, the analogue of their alternative
instrument. Column pair (4)--(5) meets the unit's contract mix across 300 four-digit
Federal Supply Classification categories with the national change in demand for each,
and partials out the share sum, which is what incomplete shares require. Standard errors
in parentheses are clustered on the state; $F$ is the first-stage $F$ statistic. The
coefficient's unit changes where the scaling denominator does, and the two blocks are
not comparable in level.%s The leave-one-out reversal that pairs with a ladder of this
kind is not run, so each row is measured given the rows above it and never given
everything else.\par}
\end{table}
""" % (merged, clipnote))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", nargs="+", default=["single", "composition"],
                    choices=["single", "composition"],
                    help="the columns to run. single is one national series, the\ncounty analogue of NS's own instrument; composition is the\n300-category multicomponent version")
    ap.add_argument("--no-anchor-check", action="store_true",
                    help="report the anchor rows without failing on them")
    ap.add_argument("--full", action="store_true",
                    help="split the three rows that were merged for length back\ninto their separate steps. Same path, same endpoint, more stops")
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 100)
    print("  From Nakamura and Steinsson's specification to ours, one change per row")
    print("=" * 100)
    print("  NS report, Table 3 cols 1-2, the Bartik columns:")
    print("             employment 1.81 (0.41) states, 2.51 (0.31) regions\n")

    ladder = resolve_ladder(full=args.full)
    county, zero_ccgs = county_frame()
    print(f"  every row is at the STATE rung, summed from "
          f"{len(county['ccg'].unique()):,} county groups. A.3.4's exclusion is not "
          f"applied,\n  because no row asks for it: the {len(zero_ccgs)} groups with "
          f"a zero EFSY year are inside the\n  state totals, and no state has zero "
          f"covered employment. {time.time()-t0:.1f}s\n", flush=True)

    C_cache = {}
    CAT_cache = {}
    rows = []
    sample_moves = []
    for inst in args.instrument:
        print(f"  COLUMN: {inst}"
              + ("   one national series, the county analogue of NS's instrument"
                 if inst == "single" else
                 "   300 category series, share sum controlled"))
        print(f"  {'row':<4}{'change':<52}{'N':>8}{'beta':>10}{'se':>9}"
              f"{'t':>7}{'F':>8}  units")
        prev = None
        for lab, desc, cfg0, bundle, why in ladder:
            cfg = {**cfg0, "instrument": inst}
            key = (cfg["rung"], cfg["sample"])
            sub = (county if cfg["sample"] == "all"
                   else county[~county["ccg"].isin(zero_ccgs)])
            if key not in C_cache:
                C_cache[key] = contracts_by_unit(sub, cfg["rung"], 1966, 2018)
            if inst == "composition" and key not in CAT_cache:
                CAT_cache[key] = category_panel_by_unit(sub, cfg["rung"], 1966, 2018)
            panel, _ = unit_panel(cfg, county, zero_ccgs)
            g = make_vars(cfg, panel, C_cache[key], cfg["horizon"], CAT_cache.get(key))
            if g is None or len(g) < 100:
                print(f"  {lab:<4}{desc:<52}{'no cells':>8}")
                continue
            r = fit(cfg, g, t28)
            print(f"  {lab:<4}{desc:<52}{r['n_obs']:>8,}{r['beta_scaled']:>10.3f}"
                  f"{r['se_scaled']:>9.3f}{r['t']:>7.2f}{r['first_stage_F']:>8.1f}"
                  f"  {r['unit_label']}", flush=True)
            span = (r["n_obs"], r["first_year_obs"], r["last_year_obs"])
            if prev is not None and span != prev and not why:
                sample_moves.append(
                    f"row {lab} moved the sample {prev} -> {span} without saying why; "
                    "protocol B.3.1 counts a change the reader can see, and the "
                    "config keys are not the whole of what a row changes")
            prev = span
            rows.append(dict(row=lab, change=desc,
                             bundle="|".join(bundle), sample_note=why or "",
                             **cfg, **r))
        print()

    out = pd.DataFrame(rows)
    dest = OUT_FULL if args.full else OUT
    out.to_csv(dest, index=False)

    # The table itself: one block of three lines per row, the coefficient over
    # its t over its first-stage F, so the columns can be read against each other
    # down the page rather than by holding two separate blocks in mind.
    order = [lab for lab, *_ in ladder]
    cols = [c for c in ("single", "composition") if c in set(out["instrument"])]
    W = 15
    print("\n" + "=" * (56 + 8 + W * len(cols)))
    print("  THE LADDER      coefficient, (t), [first-stage F]")
    print("=" * (56 + 8 + W * len(cols)))
    head = f"  {'row':<4}{'change':<50}{'N':>8}" + "".join(f"{c:>{W}}" for c in cols)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for lab in order:
        blk = out[out["row"].eq(lab)]
        if blk.empty:
            continue
        desc = blk.iloc[0]["change"]
        n = int(blk.iloc[0]["n_obs"])
        unit = blk.iloc[0]["unit_label"]
        cells = {c: blk[blk["instrument"].eq(c)] for c in cols}

        def pick(c, key, fmt, wrap=("", "")):
            d = cells[c]
            if d.empty or not np.isfinite(d.iloc[0][key]):
                return f"{'':>{W}}"
            return f"{wrap[0] + format(d.iloc[0][key], fmt) + wrap[1]:>{W}}"

        print(f"  {lab:<4}{desc[:50]:<50}{n:>8,}"
              + "".join(pick(c, "beta_scaled", ".3f") for c in cols)
              + f"   {unit}")
        print(f"  {'':<4}{'':<50}{'':>8}"
              + "".join(pick(c, "t", ".2f", ("(", ")")) for c in cols))
        print(f"  {'':<4}{'':<50}{'':>8}"
              + "".join(pick(c, "first_stage_F", ".1f", ("[", "]")) for c in cols))
    clip = out[out["n_clipped"] > 0]
    if len(clip):
        print("  THE WINSOR, in observations actually moved")
        for _, r in clip.iterrows():
            print(f"  {'':<4}row {r['row']:<4}{r['instrument']:<12}"
                  f"{int(r['n_clipped']):>6} of {int(r['n_obs']):,} "
                  f"({100*r['n_clipped']/r['n_obs']:.1f}%), "
                  f"{int(r['n_clipped'])/max(r['last_year_obs']-r['first_year_obs']+1,1):.1f} "
                  "per year")
        print("  At 51 units a year the 1st and 99th percentiles fall between the "
              "extreme\n  observation and its neighbour, so this clips the most and "
              "least exposed state\n  each year whether or not either is an outlier. "
              "Read the row above it.\n")
    print(f"  wrote {dest}")
    tex = write_tex(out, args.full)
    if tex is not None:
        print(f"  wrote {tex}")
    else:
        print("  no .tex: it needs both columns, and this run had one")
    print("  NOT RUN: the reversal that protocol B.3 pairs with a ladder. Each "
          "row here is\n  measured given the rows above it, never given "
          "everything else.")

    # The anchor. Row 1 belongs to the single-shift column and is checked against
    # 53's own output file. There is no anchor at the other end, and the print
    # below says so, because a check that does not exist must not read as one that
    # passed.
    bad = list(sample_moves)
    for lab, want in anchors().items():
        got = out[out["row"].eq(lab) & out["instrument"].eq("single")]
        if got.empty:
            bad.append(f"row {lab} did not run in the single-shift column")
            continue
        got = got.iloc[0]
        for k, v in want.items():
            have = got["beta"] if k == "beta" else got["se"] if k == "se" else \
                   got["t"] if k == "t" else got["first_stage_F"]
            if not np.isfinite(have) or abs(have - v) > max(1e-9, abs(v) * 2e-4):
                bad.append(f"row {lab} {k}: {have:.6g} against {v:.6g}")
    if bad:
        print("\n  CHECKS FAILED")
        for b in bad:
            print(f"    {b}")
        if not args.no_anchor_check:
            return 1
    else:
        print("  checks: row 1 matches 53, no row moved the sample unannounced")
    last = ladder[-1][0]
    print(f"  NOT CHECKED: row {last}. No committed producer estimates this "
          "specification at the\n  state rung, so nothing independent lands on the "
          "bottom of the ladder. The\n  composition column is unanchored at both "
          "ends for the same reason.")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
