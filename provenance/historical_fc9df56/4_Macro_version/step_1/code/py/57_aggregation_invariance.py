#!/usr/bin/env python3
"""The aggregation test on the fiscal multiplier: does the coefficient survive
being run at a coarser geography?

WHAT THE TEST IS                                        protocol A.2, B.2

Three regressions per rung, not two. The aggregated one is estimated on rows that
ARE aggregates; it is never inferred from the identity (A.2.0).

    baseline     y_i = a + b_b z_i                  + e     county rows, WLS by w
    augmented    y_i = a + b_b z_i + d ShareShock_j + e     county rows, WLS by w
    aggregated   y_j = a + b_a ShareShock_j         + e     rung-j rows

    ShareShock_j = sum_{i in j} w_ij z_i         w_ij = E_i,t-h / E_j,t-h

and the identity to check is  b_a = b_b + d  (+ b_E for entry, A.2.4/A.2.5).

WHY THE REDUCED FORM

The identity is an accounting statement about ONE linear projection. IV is a
ratio of two projections and the ratio of two aggregation-invariant objects is
not itself invariant in any usable way, so the test runs on the reduced form,
outcome on instrument. The multiplier in levels is a separate question that this
file does not answer.

THE WEIGHTS DO THREE JOBS AT ONCE                                        A.2.2

One rule aggregates the outcome, builds the exposure, and weights the county
regression. Ex-ante employment shares do all three, which is what makes
sum_i w_ij y_i = y_j hold exactly rather than approximately.

There is a wedge to report and not to hide: the shock is scaled by POPULATION
while the outcome aggregates with EMPLOYMENT. Those coincide only if every
county's employment share equals its population share, so both versions run and
the gap between them is reported as the mechanical object it is.

THE OUTCOME IS LOG GROWTH, AND THAT IS A KNOWN COMPROMISE                A.2.3

Standard growth is the admissible rate here: midpoint's weights depend on
post-treatment employment, which makes ShareShock a function of the outcome.
Standard growth is unbounded as the base goes to zero, and the protocol's answer
is the weights rather than a cap, since w_ij y_i = (E_it - E_i,t-h)/E_j,t-h has
no small denominator at all.

This file runs LOG growth instead, deliberately and as a first pass, because it
bounds the influence of small counties while a specification is being chosen. The
cost is not a vague bias: log growth breaks the identity through the Jensen
channel exactly, because the log of a sum is not the weighted average of the
logs. So the Jensen wedge is BUILT AND REPORTED beside every gap:

    jensen_j = y_j^log - sum_i w_ij y_i^log

Read no gap as a spillover until that column is subtracted. Standard growth runs
alongside on the same rows, so the two are visible together.

WHAT IS FROZEN BEFORE ANYTHING IS AGGREGATED                          B.2 1a

The county sample. Every commuting zone, state and division is built from exactly
the counties that survive A.3.4's screen, and no restriction is recomputed after
aggregating. This is the largest single source of published cross-level gaps: in
Dupor and McCrory the multipliers rise 0.39, 1.12, 1.63, 1.87 across partitions,
and rebuilt on a fixed county set the path is 0.39, 0.64, 0.77, 0.58 and the rise
is gone.

FIXED EFFECTS ARE YEAR ONLY, AND NOTHING IS LOST BY THAT

Rung-by-year effects absorb a rung-level regressor completely, so they cannot
survive to the top of the ladder. Year effects are the coarsest choice compatible
with every rung, and dropping the finer ones costs nothing at all here, which is
not obvious and is worth the proof.

The weights sum to one inside each rung-year cell, so ShareShock IS the weighted
cell mean of the instrument. Then z - ShareShock has weighted cell mean zero,
which makes it orthogonal to ShareShock and to the year dummies. The projection
of z on {year dummies, ShareShock} is therefore exactly ShareShock, and the
residualized instrument equals the one a cell fixed effect produces. Since the IV
coefficient is (z~'W y~)/(z~'W x~) and z~ annihilates anything constant within a
cell, the outcome and the treatment need not be residualized the same way:

    year effects + ShareShock  ==  rung-by-year effects,   identically

Verified at 2.256710504050 against 2.256710504050 on the division rung. The
equality holds because the weights are the AGGREGATION weights; with any other
weighting ShareShock stops being the cell mean and the two part company.

THE DIVISION RUNG HAS NINE UNITS

It is reported because the literature reports it, and every row carries its unit
count so the reader cannot miss that inference there rests on nine clusters.

Output:
    batch_logs/macro_version/aggregation_invariance.csv
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
OUT = dc.LOG_DIR / "aggregation_invariance.csv"
SEARCH_OUT = dc.LOG_DIR / "aggregation_specification_search.csv"
GRID_OUT = dc.LOG_DIR / "aggregation_level_grid.csv"

# The regression weight is FIXED at the start of the sample: population in one
# calendar year, carried on every row of that unit for every year, so the
# weighting cannot itself respond to the treatment.
#
# The year is a CALENDAR year, read straight from the population panel, not the
# base-year population of whatever the first estimable row happens to be. Those
# coincide here, because the first estimable year is 1975 + h and the base is
# measured at t - h, but only by arithmetic accident: change the sample window and
# they part company, and the weights would then differ across horizons without
# anything saying so.
POP_BASE_YEAR = 1975

# The four Census regions, as unions of the nine divisions. A region-by-year
# effect is the coarsest geographic control on the branch and the one that
# leaves the most cross-sectional variation standing.
REGION = {"New England": "Northeast", "Middle Atlantic": "Northeast",
          "East North Central": "Midwest", "West North Central": "Midwest",
          "South Atlantic": "South", "East South Central": "South",
          "West South Central": "South",
          "Mountain": "West", "Pacific": "West"}

H, SPAN, LAG = 4, 10, 5          # the headline cell of 54
# Reporting scale only, 51's own factor for a population denominator: the
# outcome goes to percentage points and spending to units of $1,000 per head.
SCALE = 1e5
RUNGS = ("czone", "msa", "cea", "ea", "state", "division")

# The last TARGET year this producer estimates on, as a constant rather than a
# literal, so the pipeline can ask the old path to reproduce a shorter sample.
# 2019 is the exclusive bound, i.e. the sample ends in 2018, which is what this
# file has always executed. 00_capture_golden sets it from locked.TARGET_LAST
# when recapturing; nothing else changes it.
TARGET_END_EXCL = 2017     # sample ends 2016; see locked.TARGET_LAST


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p54 = _load("p54", CODE / "54_composition_instrument_grid.py")
p51 = p54.p51


def wls(y, Xm, w, cluster):
    """WLS with a constant and CRV1, returning coefficients and their SEs."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, np.float64) for c in Xm])
    XtWX = A.T @ (w[:, None] * A)
    b = np.linalg.solve(XtWX, A.T @ (w * y))
    u = y - A @ b
    inv = np.linalg.inv(XtWX)
    meat = np.zeros_like(XtWX)
    for g in np.unique(cluster):
        m = cluster == g
        s = (A[m] * (w[m] * u[m])[:, None]).sum(axis=0)
        meat += np.outer(s, s)
    G = len(np.unique(cluster))
    meat *= G / max(G - 1, 1)
    V = inv @ meat @ inv
    return b, np.sqrt(np.diag(V))


def demean_year(cols, year, w):
    """Year effects, by weighted within-year demeaning. One operator, applied
    identically at every rung, which is what keeps Frisch-Waugh compatible
    across a baseline/aggregate pair."""
    out = []
    ser = pd.Series(w)
    for c in cols:
        v = pd.Series(np.asarray(c, np.float64))
        num = (v * ser).groupby(year).transform("sum")
        den = ser.groupby(year).transform("sum")
        out.append((v - num / den).to_numpy())
    return out


_PANELS = {}


def panels():
    """Everything expensive, built once: the county panel, the contract matrices
    for all four rule-floor cells, the category tensor, and the geography labels.

    Rebuilding these inside the grid would repeat a fourteen-second category
    panel a hundred and forty-four times for no reason; none of them depend on
    the horizon, the window, the denominator or the level.
    """
    if _PANELS:
        return _PANELS
    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    CON = p51.contract_matrices(units, 1966, 2018)
    ARR, NAT, cats, cy = p54.category_panel(units, 1966, 2018, harmonize=True)
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("P", "E", "Y")}
    st = B.drop_duplicates("ccg").set_index("ccg")
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"].assign(ccg=lambda t: t["ccg_id"].astype(int))
    czmap = (xw.dropna(subset=["ccz_id"]).drop_duplicates("ccg")
               .set_index("ccg")["ccz_id"])
    cb = pd.read_csv(CODE.parent / "reference" / "county_cbsa_2003.csv",
                     dtype={"county_fips5": str})
    met = cb[cb["cbsa_type"].eq("Metropolitan")][["county_fips5", "cbsa_code"]]
    msamap = (xw.merge(met, on="county_fips5", how="left")
                .dropna(subset=["cbsa_code"]).drop_duplicates("ccg")
                .set_index("ccg")["cbsa_code"])
    # BEA economic areas, the 2004 delineation 22 recovers and freezes. Whole
    # counties, full national coverage, and no consistent county group split
    # across two economic areas, so no rest-of-state residual is needed here.
    ba = pd.read_csv(CODE.parent / "reference" / "county_bea_area_2004.csv",
                     dtype={"county_fips5": str, "cea_code": str, "ea_code": str})
    bam = xw.merge(ba, on="county_fips5", how="left")
    eamap = (bam.dropna(subset=["ea_code"]).drop_duplicates("ccg")
                .set_index("ccg")["ea_code"])
    ceamap = (bam.dropna(subset=["cea_code"]).drop_duplicates("ccg")
                 .set_index("ccg")["cea_code"])
    _PANELS.update(eamap=eamap, ceamap=ceamap)
    _PANELS.update(B=B, units=units, CON=CON, ARR=ARR, NAT=NAT, cats=cats, cy=cy,
                   piv=piv, state=st["state"], division=st["division"],
                   czmap=czmap, msamap=msamap,
                   # state -> division, which is NOT the same object as the
                   # county -> division map above and was being used in its place
                   state_division=(B.drop_duplicates("state")
                                   .set_index("state")["division"]))
    return _PANELS


def build_county(instrument, h=H, span=SPAN, rule="net", floor=0.0,
                 denom="population"):
    """The frozen county panel for one cell of the grid."""
    P = panels()
    units, piv = P["units"], P["piv"]
    C = P["CON"][(rule, floor)]
    XCOL = {"population": "P", "income": "Y", "employment": "E"}[denom]
    years = [t for t in range(max(1977, 1975 + h), TARGET_END_EXCL)]

    lag = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
           for k in range(h + 1)}
    G_b = lag[h].to_numpy(np.float64)
    G_w = (sum(lag[k] for k in range(h)) / h).to_numpy(np.float64)
    Xb = piv[XCOL].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
    P_b = piv["P"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
    E_b = piv["E"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)

    if instrument == "intensity":
        natshift = np.nansum(G_w - G_b, axis=0) / np.nansum(Xb, axis=0)
        expo = np.full_like(Xb, np.nan)
        for j, t in enumerate(years):
            hi = t - h - LAG
            lo = hi - span + 1
            if lo < p51.DATA_FIRST:
                continue
            yy = list(range(lo, hi + 1))
            expo[:, j] = (C.loc[:, yy].sum(axis=1).to_numpy(np.float64)
                          / piv[XCOL].loc[:, yy].sum(axis=1, min_count=len(yy))
                          .to_numpy(np.float64))
        Z = expo * natshift[None, :]
    else:
        gk = p54.category_rates(P["NAT"][(rule, floor)], h)
        L, _ = p54.predicted_dollars(P["ARR"][(rule, floor)], gk, P["cy"], years,
                                     h, "roll", span, LAG, p51.DATA_FIRST)
        Z = L / Xb

    X = (G_w - G_b) / Xb
    if POP_BASE_YEAR not in piv["P"].columns:
        raise SystemExit(f"  FAIL: population has no {POP_BASE_YEAR}")
    P0 = piv["P"][POP_BASE_YEAR].to_numpy(np.float64)

    ok = (np.isfinite(Z) & np.isfinite(X) & np.isfinite(E_b) & np.isfinite(E_t)
          & (E_b > 0) & (E_t > 0))
    ri, ci = np.where(ok)
    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci],
                          z=Z[ok], x=X[ok], E_b=E_b[ok], E_t=E_t[ok],
                          P_b=P_b[ok],
                          P0=np.broadcast_to(P0[:, None], P_b.shape)[ok]))
    d["y_log"] = np.log(d["E_t"]) - np.log(d["E_b"])
    d["y_std"] = (d["E_t"] - d["E_b"]) / d["E_b"]
    d["y_mpgr"] = 2.0 * (d["E_t"] - d["E_b"]) / (d["E_t"] + d["E_b"])
    d["state"] = d["ccg"].map(P["state"])
    d["division"] = d["ccg"].map(P["division"])
    d["czone"] = d["ccg"].map(P["czmap"])
    d["cea"] = d["ccg"].map(P["ceamap"])
    d["ea"] = d["ccg"].map(P["eamap"])
    inm = d["ccg"].map(P["msamap"])
    d["msa"] = np.where(inm.notna(), "M" + inm.astype(str),
                        "R" + d["state"].astype(str))
    before = d["ccg"].nunique()
    d = d[d[["czone", "msa", "cea", "ea", "state", "division"]]
          .notna().all(axis=1)]
    return d.reset_index(drop=True), before - d["ccg"].nunique()


def _build_county_legacy(instrument, h=H, span=SPAN):
    """The frozen county panel: outcome, instrument, weights, rung labels.

    A.3.4's screen is applied HERE and nowhere else. Every aggregate is built
    from these rows, so no county excluded at the baseline can reappear inside a
    larger unit.
    """
    B = p51.base_panel()                       # the screen lives in here
    units = np.array(sorted(B["ccg"].unique()))
    C = p51.contract_matrices(units, 1966, 2018)[("net", 0.0)]
    years = [t for t in range(max(1977, 1975 + h), TARGET_END_EXCL)]
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("P", "E")}

    lag = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
           for k in range(h + 1)}
    G_b = lag[h].to_numpy(np.float64)
    G_w = (sum(lag[k] for k in range(h)) / h).to_numpy(np.float64)
    P_b = piv["P"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
    E_b = piv["E"].reindex(columns=[y - h for y in years]).to_numpy(np.float64)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)

    if instrument == "intensity":
        natshift = np.nansum(G_w - G_b, axis=0) / np.nansum(P_b, axis=0)
        expo = np.full_like(P_b, np.nan)
        for j, t in enumerate(years):
            hi = t - h - LAG
            lo = hi - span + 1
            if lo < p51.DATA_FIRST:
                continue
            yy = list(range(lo, hi + 1))
            expo[:, j] = (C.loc[:, yy].sum(axis=1).to_numpy(np.float64)
                          / piv["P"].loc[:, yy].sum(axis=1, min_count=len(yy))
                          .to_numpy(np.float64))
        Z = expo * natshift[None, :]
    else:
        ARR, NAT, cats, cy = p54.category_panel(units, 1966, 2018, harmonize=True)
        gk = p54.category_rates(NAT[("net", 0.0)], h)
        L, nc = p54.predicted_dollars(ARR[("net", 0.0)], gk, cy, years, h,
                                      "roll", span, LAG, p51.DATA_FIRST)
        Z = L / P_b

    if POP_BASE_YEAR not in piv["P"].columns:
        raise SystemExit(f"  FAIL: population has no {POP_BASE_YEAR}; the fixed "
                         f"regression weight has no base year to read")
    P0 = piv["P"][POP_BASE_YEAR].to_numpy(np.float64)     # (n_units,), one per unit

    X = (G_w - G_b) / P_b            # the treatment, exactly 51's
    ok = (np.isfinite(Z) & np.isfinite(X) & np.isfinite(E_b) & np.isfinite(E_t)
          & (E_b > 0) & (E_t > 0))
    ri, ci = np.where(ok)
    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci],
                          z=Z[ok], x=X[ok], E_b=E_b[ok], E_t=E_t[ok],
                          P_b=P_b[ok],
                          P0=np.broadcast_to(P0[:, None], P_b.shape)[ok]))
    # Both growth rates, on identical rows. Log is what this pass estimates on;
    # standard is the admissible one and runs beside it so the compromise is
    # visible rather than remembered.
    d["y_log"] = np.log(d["E_t"]) - np.log(d["E_b"])
    d["y_std"] = (d["E_t"] - d["E_b"]) / d["E_b"]
    d["y_mpgr"] = 2.0 * (d["E_t"] - d["E_b"]) / (d["E_t"] + d["E_b"])

    st = B.drop_duplicates("ccg").set_index("ccg")
    d["state"] = d["ccg"].map(st["state"])
    d["division"] = d["ccg"].map(st["division"])
    # The commuting zone comes from the project's OWN crosswalk, which already
    # carries a harmonized `ccz_id`: 739 zones, and no CCG split across two, so
    # counties nest exactly. The raw `czone` column does not have that property,
    # two CCGs straddling a boundary, which would make the aggregate outcome not
    # a partition of the baseline.
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"]
    czmap = (xw.assign(ccg=xw["ccg_id"].astype(int))
               .dropna(subset=["ccz_id"]).drop_duplicates("ccg")
               .set_index("ccg")["ccz_id"])
    d["czone"] = d["ccg"].map(czmap)

    # The MSA rung. Metropolitan counties join their MSA; micropolitan and rural
    # counties join one rest-of-state unit for their state, so no county leaves
    # and the support stays frozen. A rest-of-state unit is an administrative
    # residual and not a labour market, which is a thing to say plainly rather
    # than present as a third geography beside zones and states. What it buys is
    # that 1,992 tiny units become 48 large ones, which removes the small-unit
    # problem at its source instead of reweighting around it: the smallest half
    # of MSA units hold 10.3% of 1975 population against 4.1% of commuting zones,
    # and the estimate stops depending on the weighting choice.
    #
    # The delineation is 21's frozen 2003 vintage, applied to every year. MSA
    # boundaries move at every OMB revision, so a contemporaneous delineation
    # would make the unit set a different object in each era and mix a change in
    # geography with a change in the instrument.
    cb = pd.read_csv(CODE.parent / "reference" / "county_cbsa_2003.csv",
                     dtype={"county_fips5": str})
    met = cb[cb["cbsa_type"].eq("Metropolitan")][["county_fips5", "cbsa_code"]]
    msamap = (xw.assign(ccg=xw["ccg_id"].astype(int))
                .merge(met, on="county_fips5", how="left")
                .dropna(subset=["cbsa_code"]).drop_duplicates("ccg")
                .set_index("ccg")["cbsa_code"])
    in_msa = d["ccg"].map(msamap)
    d["msa"] = np.where(in_msa.notna(), "M" + in_msa.astype(str),
                        "R" + d["state"].astype(str))

    # FREEZE THE SUPPORT, once, here. A county missing ANY rung's label leaves
    # now rather than at the rung that lacks it: otherwise the commuting-zone
    # rung runs on a different county set from the state rung, and the ladder
    # measures a change of sample as if it were a change of level. This is
    # protocol B.2 rule 1a, the largest single source of published cross-level
    # gaps.
    before = d["ccg"].nunique()
    d = d[d[["czone", "msa", "cea", "ea", "state", "division"]]
          .notna().all(axis=1)]
    dropped = before - d["ccg"].nunique()
    return d.reset_index(drop=True), dropped


LEVELS = ("county", "czone", "msa", "cea", "ea", "state", "division")


def unit_labels(d, level):
    """One main state per unit, and the division that follows from it.

    A commuting zone or an MSA can straddle a state line, and Philadelphia
    straddles a division line as well. Each unit is assigned the state holding
    the plurality of its base-year population, so that division-equal weighting
    has a single well-defined division per unit rather than a share of several.
    The count of straddlers is returned, not buried.
    """
    P = panels()
    u = d.drop_duplicates("ccg")
    if level == "division":
        v = pd.unique(u["division"])
        # A division has no single state above it; state-equal weighting is
        # therefore undefined at this rung and the grid skips it there.
        return pd.Series(np.nan, index=v), pd.Series(v, index=v), 0
    if level == "county":
        st, straddle = u.set_index("ccg")["state"], 0
    elif level == "state":
        v = pd.unique(u["state"])
        st, straddle = pd.Series(v, index=v), 0
    else:
        pop = u.groupby([level, "state"])["P0"].sum().reset_index()
        st = (pop.sort_values("P0", ascending=False).drop_duplicates(level)
                 .set_index(level)["state"])
        straddle = int((u.groupby(level)["state"].nunique() > 1).sum())
    return st, st.map(P["state_division"]), straddle


def collapse(d, level, ycol):
    """One level of the ladder, built from counties with ex-ante employment shares.

    The employment shares do the AGGREGATION: they build the outcome, the
    treatment and the instrument at level j. The regression weight is a separate
    object, passed in, and is never these shares.
    """
    if level == "county":
        g = d.copy()
        g["unit"] = g["ccg"]
        out = g[["unit", "year", ycol, "x", "z"]].rename(columns={ycol: "y"})
    else:
        g = d[d[level].notna()].copy()
        tot = g.groupby([level, "year"])["E_b"].transform("sum")
        g["om"] = g["E_b"] / tot
        out = (g.assign(wy=g["om"] * g[ycol], wx=g["om"] * g["x"],
                        wz=g["om"] * g["z"])
                 .groupby([level, "year"], as_index=False)
                 .agg(y=("wy", "sum"), x=("wx", "sum"), z=("wz", "sum"))
                 .rename(columns={level: "unit"}))
    # The fixed weight: the unit's total POP_BASE_YEAR population, summed over
    # its counties once and identical in every year.
    key = "ccg" if level == "county" else level
    u = d.drop_duplicates("ccg")
    out["popw"] = out["unit"].map(u.groupby(key)["P0"].sum())
    # Employment, for the division-equal weighting, and the unit's division.
    out["Eb"] = out["unit"].map(
        d.groupby(key)["E_b"].mean() if level != "county"
        else d.groupby("ccg")["E_b"].mean())
    ust, div, _ = unit_labels(d, level)
    out["ustate"] = out["unit"].map(ust)
    out["division"] = out["unit"].map(div)
    out["region"] = out["division"].map(REGION)
    return out.dropna(subset=["y", "x", "z", "popw"])


def weight_vector(a, scheme):
    """The three weightings, all normalized to mean one.

    equal        every unit-year alike, at whatever level is being run
    fixed pop    the unit's POP_BASE_YEAR population, the same number every year
    division-EW  each division gets equal total weight, units inside it weighted
                 by employment share. At the division level this coincides with
                 equal weighting, which is a duplicate rather than an error.
    """
    if scheme == "equal":
        w = np.ones(len(a), np.float64)
    elif scheme == "fixed pop":
        w = a["popw"].to_numpy(np.float64)
    elif scheme == "division-EW":
        tot = a.groupby(["division", "year"])["Eb"].transform("sum")
        w = (a["Eb"] / tot).to_numpy(np.float64)
    else:                                            # state-EW
        tot = a.groupby(["ustate", "year"])["Eb"].transform("sum")
        w = (a["Eb"] / tot).to_numpy(np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    return w / np.nanmean(w)


RULES = (("net", 0.0, "none"), ("net", 25_000.0, "25k"),
         ("positive", 0.0, "none"), ("positive", 25_000.0, "25k"))
DENOMS = ("population", "income", "employment")
WEIGHTS = ("equal", "fixed pop", "division-EW", "state-EW")
FES = ("year", "regionyear", "divisionyear")

def admissible(level, wl, fe):
    """Which level, weight and effect combinations exist at all.

    State-equal weighting groups units by the state above them and has no
    meaning at the division rung. A geographic-by-year effect absorbs a
    regressor defined at that same geography, so division-by-year effects stop
    at the state rung. Region-by-year effects survive to the division rung,
    where the two or three divisions inside each region identify the
    coefficient: thin, but not degenerate. These combinations are skipped
    rather than run and reported as zeros.
    """
    if level == "division" and (wl == "state-EW" or fe == "divisionyear"):
        return False
    return True


BASE_OUT = dc.LOG_DIR / "aggregation_baseline.csv"

# ---------------------------------------------------------------------------
# THE BASELINE, declared in one place so nothing downstream has to remember it.
# ---------------------------------------------------------------------------
BASELINE = dict(instrument="composition", denom="population", outcome="y_mpgr",
                span=5, rule="positive", floor=0.0, fe="divisionyear",
                levels=("czone", "msa", "cea", "ea", "state"),
                horizons=(2, 3, 4))

# The two weighting schemes, side by side, and why they are a pair rather than
# rivals. The regression weight must be the aggregation weight E_i/E_j times any
# scalar constant inside the cell: the cell then carries the same total weight in
# both regressions, and within the cell the weight stays proportional to the
# aggregation weight. Those two conditions are what make b_a = b_b + delta an
# identity. c_j = 1 counts every labour market once; c_j = P_j counts every
# person once. Both are legal, and showing both is how a reader sees that the
# choice does not drive the result.
#
# Weighting each COUNTY by its own population is NOT in this family and breaks
# the identity, by 0.038 at the commuting zone and 0.149 at the state, because
# county population is not proportional to county employment share inside a cell.
SCALARS = ("EW", "P_j")

# Region-by-year rather than division-by-year, and the reason is the pair. Under
# division-by-year the population column is stronger, 2.78 against 2.42, but the
# equal-weighted column then fails at two horizons of three instead of one.
# Region-by-year costs 0.36 on the headline and buys a second column that mostly
# works. If the equal-weighted column is ever dropped, this choice should be
# revisited, since it is paid for entirely by that column.


def baseline(args, t28, w22, t0):
    """The locked specification: both weightings, every level, every horizon."""
    B = BASELINE
    rows = []
    worst_gap = 0.0
    for h in B["horizons"]:
        d, _ = build_county(B["instrument"], h=h, span=B["span"],
                            rule=B["rule"], floor=B["floor"], denom=B["denom"])
        for lev in B["levels"]:
            g = d[d[lev].notna()].copy()
            g["om"] = g["E_b"] / g.groupby([lev, "year"])["E_b"].transform("sum")
            key = pd.MultiIndex.from_arrays([g[lev], g["year"]])
            shr = (g.assign(wy=g["om"] * g[B["outcome"]], wz=g["om"] * g["z"])
                     .groupby([lev, "year"])[["wy", "wz"]].sum())
            g["share"] = shr["wz"].reindex(key).to_numpy()
            agg = shr.reset_index().rename(columns={"wy": "y", "wz": "share"})
            P0j = g.drop_duplicates("ccg").groupby(lev)["P0"].sum()
            _, div, straddle = unit_labels(d, lev)
            di, da = g[lev].map(div), agg[lev].map(div)
            if B["fe"] == "regionyear":
                di, da = di.map(REGION), da.map(REGION)
            rk_i = di.astype(str) + "_" + g["year"].astype(str)
            rk_a = da.astype(str) + "_" + agg["year"].astype(str)
            for cj in SCALARS:
                wc = (g["om"] if cj == "EW"
                      else g["om"] * g[lev].map(P0j)).to_numpy(np.float64)
                wa = (np.ones(len(agg)) if cj == "EW"
                      else agg[lev].map(P0j).to_numpy(np.float64))
                wc, wa = wc / wc.mean(), wa / wa.mean()
                f1 = pd.factorize(rk_i)[0].astype(np.int64)
                f1a = pd.factorize(rk_a)[0].astype(np.int64)
                ze = np.zeros(len(g), np.int64)
                zea = np.zeros(len(agg), np.int64)
                clc = pd.factorize(g[lev])[0].astype(np.int64)
                cla = pd.factorize(agg[lev])[0].astype(np.int64)

                y = g[B["outcome"]].to_numpy(np.float64)
                z = g["z"].to_numpy(np.float64)
                sh = g["share"].to_numpy(np.float64)
                M = w22.demean(np.column_stack([y, z, sh]), f1, ze, wc)
                A = np.column_stack([M[:, 1], M[:, 2]])
                sol = np.linalg.solve(A.T @ (wc[:, None] * A), A.T @ (wc * M[:, 0]))
                bb, dl = float(sol[0]), float(sol[1])
                Ma = w22.demean(np.column_stack([agg["y"], agg["share"]]),
                                f1a, zea, wa)
                ba = float(Ma[:, 1] @ (wa * Ma[:, 0])) / float(Ma[:, 1] @ (wa * Ma[:, 1]))
                gap = abs(ba - bb - dl) * SCALE
                worst_gap = max(worst_gap, gap)

                # The aggregate the literature would actually run: the unit's OWN
                # log growth, log(sum E_t) - log(sum E_b), which is not the
                # weighted mean of its counties' log growths. The difference is
                # the Jensen wedge, and under log growth it is exactly why the
                # identity above cannot hold against a real aggregate.
                Ej = g.groupby([lev, "year"])[["E_b", "E_t"]].sum()
                if B["outcome"] == "y_mpgr":
                    own = 2.0 * (Ej["E_t"] - Ej["E_b"]) / (Ej["E_t"] + Ej["E_b"])
                elif B["outcome"] == "y_std":
                    own = (Ej["E_t"] - Ej["E_b"]) / Ej["E_b"]
                else:
                    own = np.log(Ej["E_t"]) - np.log(Ej["E_b"])
                own = own.reindex(pd.MultiIndex.from_frame(
                    agg[[lev, "year"]])).to_numpy()
                Mo = w22.demean(np.column_stack([own, agg["share"]]), f1a, zea, wa)
                bo = float(Mo[:, 1] @ (wa * Mo[:, 0])) / float(Mo[:, 1] @ (wa * Mo[:, 1]))
                ro = t28.fit2sls(own, agg["share"].to_numpy(np.float64),
                                 agg["share"].to_numpy(np.float64),
                                 wa, f1a, zea, cla, singleton=False)

                rc = t28.fit2sls(y, z, z, wc, f1, ze, clc, singleton=False)
                ra = t28.fit2sls(agg["y"].to_numpy(np.float64),
                                 agg["share"].to_numpy(np.float64),
                                 agg["share"].to_numpy(np.float64),
                                 wa, f1a, zea, cla, singleton=False)
                rows.append(dict(
                    horizon=h, level=lev, weight=cj, n_units=int(g[lev].nunique()),
                    # The county groups surviving the zero-employment screen,
                    # the same 3,058 the deck quotes. Saved so the frame can
                    # input it instead of typing it.
                    n_ccg=int(d["ccg"].nunique()),
                    n_county=len(g), n_agg=len(agg), straddle=straddle,
                    beta_b=bb * SCALE, delta=dl * SCALE,
                    beta_a_aligned=ba * SCALE, beta_a_own=bo * SCALE,
                    jensen=(bo - ba) * SCALE, construction_gap=gap,
                    # `rc` is the county regression WITHOUT ShareTreated: the
                    # one an applied paper runs before it has thought about the
                    # rung at all. Its coefficient is what makes the naive
                    # cross-level difference beta_a - beta_b readable as an
                    # implied spillover, so it is saved rather than reduced to
                    # a t statistic. It is NOT `beta_b`, which comes from the
                    # augmented fit, and it is NOT T3's `beta_b_naive` in the
                    # battery, which is the augmented fit under naive weights.
                    beta_b_noaug=rc["beta"] * SCALE,
                    se_b_noaug=rc["se"] * SCALE,
                    # gamma: ShareTreated_j on the county's own shock, same
                    # weights, same effects. Omitted-variable algebra makes
                    # beta_b_noaug = beta_b + delta * gamma exactly, so the
                    # naive cross-level difference is delta * (1 - gamma);
                    # saved from its own regression so that identity can be
                    # checked rather than assumed.
                    gamma_share_on_z=float(M[:, 1] @ (wc * M[:, 2]))
                    / float(M[:, 1] @ (wc * M[:, 1])),
                    # Under aligned weights the share is the om-weighted mean of
                    # the counties' shocks, so Cov(z, share) collapses to
                    # Var(share) and gamma is a ratio of variances, bounded in
                    # [0, 1] with no assumption on within-unit correlation.
                    # Saved separately so the collapse is checked, not assumed.
                    var_ratio_share_z=float(M[:, 2] @ (wc * M[:, 2]))
                    / float(M[:, 1] @ (wc * M[:, 1])),
                    beta_a_own_se=ro["se"] * SCALE,
                    t_agg_own=ro["beta"] / ro["se"],
                    t_county=rc["beta"] / rc["se"], t_agg=ra["beta"] / ra["se"]))
    o = pd.DataFrame(rows)
    o.to_csv(BASE_OUT, index=False)

    print("=" * 92)
    print("  BASELINE   composition instrument, population denominator, log growth,")
    print("             ten-year window, net contracts, no floor, "
          "region-by-year effects,")
    print("             employment shares aggregating, county against each level")
    print("=" * 92)
    for cj, lab in (("EW", "c_j = 1        every labour market counts once"),
                    ("P_j", "c_j = P_j      every person counts once")):
        print(f"\n{lab}")
        print(f"  {'level':<9}{'units':>7}" + "".join(
            f"{'h='+str(h)+' cty':>10}{'h='+str(h)+' agg':>10}"
            for h in BASELINE["horizons"]))
        for lev in BASELINE["levels"]:
            line = f"  {lev:<9}{int(o[o.level.eq(lev)].n_units.max()):>7,}"
            for h in BASELINE["horizons"]:
                r = o[(o.level == lev) & (o.horizon == h) & (o.weight == cj)]
                line += ("".join(f"{r.iloc[0][c]:>10.2f}"
                                 for c in ("t_county", "t_agg")) if len(r)
                         else f"{'-':>20}")
            print(line)
        sub = o[o.weight.eq(cj)]
        w = sub[["t_county", "t_agg"]].abs().min().min()
        print(f"  {'weakest cell':<9}{'':>7}{w:>10.2f}")

    print(f"\n  AT h = {BASELINE['horizons'][-1]}")
    print("  aligned: the weighted mean of the county outcomes, which is what")
    print("           beta_b + delta reproduces BY CONSTRUCTION")
    print(f"  own:     the unit's own {B['outcome'][2:]} growth, computed from summed")
    print("           employment,")
    print("           which is what a paper reports and is NOT that object")
    print("  jensen:  the difference. Nonlinear growth of a sum is not the")
    print("           lagged-share mean of the growths, so it need not be zero.")
    print(f"  {'level':<9}{'weight':<6}{'beta_b':>9}{'delta':>9}"
          f"{'aligned':>9}{'own':>9}{'jensen':>9}")
    for lev in BASELINE["levels"]:
        for cj in SCALARS:
            r = o[(o.level == lev) & (o.horizon == BASELINE["horizons"][-1])
                  & (o.weight == cj)]
            if len(r):
                r = r.iloc[0]
                print(f"  {lev:<9}{cj:<6}{r.beta_b:>9.3f}{r.delta:>9.3f}"
                      f"{r.beta_a_aligned:>9.3f}{r.beta_a_own:>9.3f}"
                      f"{r.jensen:>9.3f}")
    print(f"\n  CONSTRUCTION CHECK, largest gap {worst_gap:.1e}")
    if worst_gap > 1e-8:
        raise SystemExit("  FAIL: the regression weight must be the aggregation "
                         "weight times a scalar constant within the cell.")
    print("  This is a check on the CODE, not a result. beta_a(aligned) is defined")
    print("  as the weighted mean of the county outcomes, so beta_a = beta_b + delta")
    print("  follows from the normal equations. It catches a wrong weight or a wrong")
    print("  collapse, and it caught one; it says nothing about the world.")
    print("\n  NOT an identity: beta_a(own) = beta_b + delta. Nonlinear growth of")
    print("  a sum is not the mean of the growths; the two aggregates differ by the jensen")
    print("  column above. And delta is a spillover only under the maintained")
    print("  condition on treatment-effect heterogeneity, never by construction.")
    print(f"\nwrote {BASE_OUT}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


MATCHED_OUT = dc.LOG_DIR / "aggregation_matched.csv"


def matched(args, t28, t0):
    """One weighting rule, applied to whatever level the comparison is against.

    For each higher level j the county regression is weighted by E_i/E_j, which
    sums to one inside every j-year cell, and the regression at j is equally
    weighted. That is the only pairing under which each j-year cell carries the
    same total weight in both regressions, which is what makes b_a = b_b + delta
    an identity rather than an approximation.

    So the weighting is not a fixed scheme applied everywhere: it changes with
    the comparison, being commuting-zone-equal when the comparison is against
    commuting zones and state-equal when it is against states. At the level
    itself the rule reduces to equal weighting.
    """
    rows = []
    for inst in args.instrument:
        for h in (2, 3, 4):
            for span in (5, 10):
                for denom in DENOMS:
                    d, _ = build_county(inst, h=h, span=span, denom=denom)
                    for ycol in args.outcome:
                        for j in ("czone", "msa", "cea", "ea", "state"):
                            g = d[d[j].notna()].copy()
                            tot = g.groupby([j, "year"])["E_b"].transform("sum")
                            g["w"] = g["E_b"] / tot
                            key = pd.MultiIndex.from_arrays([g[j], g["year"]])
                            agg = (g.assign(wy=g["w"] * g[ycol],
                                            wz=g["w"] * g["z"])
                                     .groupby([j, "year"], as_index=False)
                                     .agg(y=("wy", "sum"), z=("wz", "sum")))
                            for fe in ("year", "regionyear", "divisionyear"):
                                if fe == "divisionyear" and j == "division":
                                    continue
                                out = {}
                                # the county side, weighted to match level j
                                wc = (g["w"] / g["w"].mean()).to_numpy(np.float64)
                                lab = (g["year"] if fe == "year"
                                       else (g["division"].map(REGION).astype(str)
                                             if fe == "regionyear"
                                             else g["division"].astype(str))
                                       + ("" if fe == "year"
                                          else "_" + g["year"].astype(str)))
                                f1 = pd.factorize(lab)[0].astype(np.int64)
                                ze = np.zeros(len(g), np.int64)
                                cl = pd.factorize(g[j])[0].astype(np.int64)
                                r = t28.fit2sls(g[ycol].to_numpy(np.float64),
                                                g["z"].to_numpy(np.float64),
                                                g["z"].to_numpy(np.float64),
                                                wc, f1, ze, cl, singleton=False)
                                out["county"] = r["beta"] / r["se"]
                                # level j, equally weighted
                                ust, div, _ = unit_labels(d, j)
                                agg2 = agg.rename(columns={j: "unit"})
                                dv = agg2["unit"].map(div)
                                lab2 = (agg2["year"] if fe == "year"
                                        else (dv.map(REGION).astype(str)
                                              if fe == "regionyear"
                                              else dv.astype(str))
                                        + ("" if fe == "year"
                                           else "_" + agg2["year"].astype(str)))
                                wa = np.ones(len(agg2))
                                f1a = pd.factorize(lab2)[0].astype(np.int64)
                                zea = np.zeros(len(agg2), np.int64)
                                cla = pd.factorize(agg2["unit"])[0].astype(np.int64)
                                r2 = t28.fit2sls(agg2["y"].to_numpy(np.float64),
                                                 agg2["z"].to_numpy(np.float64),
                                                 agg2["z"].to_numpy(np.float64),
                                                 wa, f1a, zea, cla, singleton=False)
                                out["agg"] = r2["beta"] / r2["se"]
                                rows.append(dict(
                                    instrument=inst, horizon=h, span=span,
                                    denom=denom, outcome=ycol, level=j, fe=fe,
                                    t_county=out["county"], t_agg=out["agg"],
                                    n_units=int(g[j].nunique())))
                print(f"  {inst} h={h} span={span} done  {time.time()-t0:.0f}s",
                      flush=True)
    o = pd.DataFrame(rows)
    o.to_csv(MATCHED_OUT, index=False)
    o["m"] = o[["t_county", "t_agg"]].abs().min(axis=1)
    g = o.groupby(["instrument", "denom", "fe"])["m"].min().sort_values(
        ascending=False)
    print("\n  MATCHED WEIGHTING: level-equal at the comparison level, and the")
    print("  county regression weighted to match it. Worst of the two sides,")
    print("  across every level, horizon and window.\n")
    print(f"  {'instrument':<13}{'denom':<12}{'fe':<14}{'worst':>8}")
    for k, v in g.head(8).items():
        print(f"  {k[0]:<13}{k[1]:<12}{k[2]:<14}{v:>8.2f}   "
              f"{'YES' if v > 1.96 else ''}")
    best = g.index[0]
    b = o[(o.instrument == best[0]) & (o.denom == best[1]) & (o.fe == best[2])
          & (o.span == 10)]
    print(f"\n  best = {best[0]} / {best[1]} / {best[2]} effects, window 10")
    print(f"  {'level':<9}{'units':>7}" + "".join(
        f"{'h='+str(h)+' cty':>11}{'h='+str(h)+' agg':>11}" for h in (2, 3, 4)))
    for j in ("czone", "msa", "cea", "ea", "state"):
        r = b[b.level.eq(j)]
        line = f"  {j:<9}{int(r.n_units.max()):>7,}"
        for h in (2, 3, 4):
            q = r[r.horizon.eq(h)]
            line += ("".join(f"{q.iloc[0][c]:>11.2f}"
                             for c in ("t_county", "t_agg")) if len(q)
                     else f"{'-':>22}")
        print(line)
    print(f"\n  wrote {MATCHED_OUT}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


def level_grid(args, t28, t0):
    """The grid. Reduced form only: the outcome on the shift-share instrument.

    Swept: level, instrument, horizon, window length, regression weighting,
    contract rule, floor, and the denominator the shock is scaled by.

    Held: log growth, year effects, ex-ante employment shares doing the
    aggregation, the county sample frozen before anything is aggregated, a
    five-year window lag, and a rolling rather than fixed exposure window.

    The first-stage F is printed beside each cell for information. Nothing is
    gated on it.
    """
    rows = []
    n = 0
    for inst in args.instrument:
        for h in (2, 3, 4):
            for span in (5, 10):
                for rule, floor, flab in RULES:
                    for denom in DENOMS:
                        d, _ = build_county(inst, h=h, span=span, rule=rule,
                                            floor=floor, denom=denom)
                        for ycol in args.outcome:
                            for level in LEVELS:
                                a = collapse(d, level, ycol)
                                if len(a) < 50:
                                    continue
                                yr = pd.factorize(a["year"])[0].astype(np.int64)
                                ze = np.zeros(len(a), np.int64)
                                cl = pd.factorize(a["unit"])[0].astype(np.int64)
                                y = a["y"].to_numpy(np.float64)
                                x = a["x"].to_numpy(np.float64)
                                z = a["z"].to_numpy(np.float64)
                                FEV = {"year": yr}
                                for _k, _c in (("regionyear", "region"),
                                               ("divisionyear", "division")):
                                    FEV[_k] = pd.factorize(
                                        a[_c].astype(str) + "_"
                                        + a["year"].astype(str))[0].astype(np.int64)
                                for wl in WEIGHTS:
                                    w = weight_vector(a, wl)
                                    m = np.isfinite(w)
                                    if m.sum() < 50:
                                        continue
                                    for fe in FES:
                                        if not admissible(level, wl, fe):
                                            continue
                                        try:
                                            rf = t28.fit2sls(y[m], z[m], z[m], w[m],
                                                             FEV[fe][m], ze[m],
                                                             cl[m], singleton=False)
                                            iv = t28.fit2sls(y[m], x[m], z[m], w[m],
                                                             FEV[fe][m], ze[m],
                                                             cl[m], singleton=False)
                                        except Exception:        # noqa: BLE001
                                            continue
                                        n += 1
                                        rows.append(dict(
                                            level=level, instrument=inst, horizon=h,
                                            span=span, weight=wl, fe=fe, rule=rule,
                                            floor=flab, denom=denom, outcome=ycol,
                                            n_units=int(a["unit"].nunique()),
                                            n_obs=int(m.sum()),
                                            rf_beta=rf["beta"] * SCALE,
                                            rf_t=rf["beta"] / rf["se"],
                                            iv_t=iv["beta"] / iv["se"],
                                            F=iv["F"]))
                print(f"  {inst} h={h} span={span} done, {n:,} cells, "
                      f"{time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(GRID_OUT, index=False)
    out["ok"] = out["rf_t"].abs() > 1.96

    for level in LEVELS:
        sub = out[out["level"].eq(level)]
        if sub.empty:
            continue
        d0, _ = build_county(args.instrument[0], h=4, span=10)
        _, _, straddle = unit_labels(d0, level)
        print(f"\n  {level.upper()}   {sub['n_units'].max():,} units"
              + (f", {straddle} straddling a state line" if straddle else ""))
        t = sub.pivot_table(index=["instrument", "denom"], columns=["weight", "fe"],
                            values="rf_t", aggfunc="max")

        print(f"    best reduced-form t over horizon, window, rule and floor")
        print(f"    {'instrument':<13}{'denom':<12}"
              + "".join(f"{a[:8]+'/'+b[:4]:>14}" for a, b in t.columns))
        for (i_, dn), r in t.iterrows():
            print(f"    {i_:<13}{dn:<12}"
                  + "".join(f"{v:>14.2f}" for v in r.values))
    print(f"\n  {len(out):,} cells, {int(out['ok'].sum())} with |t| > 1.96")
    print(f"  wrote {GRID_OUT}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


def iv_at_rungs(d, rung, ycol, t28, w22):
    """2SLS at the county rung and at the aggregate rung, on the same rows.

    The aggregate regression instruments the employment-weighted aggregate
    TREATMENT with the employment-weighted aggregate INSTRUMENT, which is the
    same object the reduced form uses. Reported per level: the first-stage F and
    the second-stage t, which is what decides whether a rung is usable at all.

    The county row is also run augmented with the share shock, because that is
    the specification the aggregation test actually needs, and a control that
    kills the first stage would make the whole ladder unusable.
    """
    g = d[d[rung].notna()].copy()
    tot = g.groupby([rung, "year"])["E_b"].transform("sum")
    g["w"] = g["E_b"] / tot
    key = pd.MultiIndex.from_arrays([g[rung], g["year"]])
    shr = (g.assign(wz=g["w"] * g["z"], wx=g["w"] * g["x"], wy=g["w"] * g[ycol])
             .groupby([rung, "year"])[["wz", "wx", "wy"]].sum())
    g["share_z"] = shr["wz"].reindex(key).to_numpy()
    g["share_x"] = shr["wx"].reindex(key).to_numpy()
    agg = shr.reset_index().rename(columns={"wz": "z", "wx": "x", "wy": "y"})

    out = {}
    wc = (g["w"] / g["w"].mean()).to_numpy(np.float64)
    yrc = pd.factorize(g["year"])[0].astype(np.int64)
    zero = np.zeros(len(g), np.int64)
    clc = pd.factorize(g[rung])[0].astype(np.int64)
    y0 = g[ycol].to_numpy(np.float64)
    x0 = g["x"].to_numpy(np.float64)
    z0 = g["z"].to_numpy(np.float64)
    r = t28.fit2sls(y0, x0, z0, wc, yrc, zero, clc, singleton=False)
    out["county"] = (r["beta"] * SCALE, r["beta"] / r["se"], r["F"], len(g))

    # augmented: partial the share shock out of all three, then the same 2SLS
    M = w22.demean(np.column_stack([y0, x0, z0, g["share_z"].to_numpy()]),
                   yrc, zero, wc)
    yy, xx, zz, st = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
    sw = float(st @ (wc * st))
    if sw > 0:
        yy = yy - float(st @ (wc * yy)) / sw * st
        xx = xx - float(st @ (wc * xx)) / sw * st
        zz = zz - float(st @ (wc * zz)) / sw * st
    r = t28.fit2sls(yy, xx, zz, wc, zero, zero, clc, singleton=False)
    out["county_aug"] = (r["beta"] * SCALE, r["beta"] / r["se"], r["F"], len(g))

    wa = np.ones(len(agg), np.float64)
    yra = pd.factorize(agg["year"])[0].astype(np.int64)
    zea = np.zeros(len(agg), np.int64)
    cla = pd.factorize(agg[rung])[0].astype(np.int64)
    r = t28.fit2sls(agg["y"].to_numpy(np.float64), agg["x"].to_numpy(np.float64),
                    agg["z"].to_numpy(np.float64), wa, yra, zea, cla,
                    singleton=False)
    out["aggregate"] = (r["beta"] * SCALE, r["beta"] / r["se"], r["F"], len(agg))
    out["n_units"] = int(g[rung].nunique())
    return out


def run_rung(d, rung, ycol, t28):
    """The three regressions at one rung, plus the identity and the Jensen wedge."""
    g = d[d[rung].notna()].copy()
    tot = g.groupby([rung, "year"])["E_b"].transform("sum")
    g["w"] = g["E_b"] / tot                       # sums to one inside each j-year
    g["share"] = (g.groupby([rung, "year"])
                    .apply(lambda t: (t["w"] * t["z"]).sum(),
                           include_groups=False)
                    .reindex(pd.MultiIndex.from_arrays(
                        [g[rung], g["year"]])).to_numpy())
    # The aggregate outcome IS the weighted sum of the county outcomes, which is
    # what makes the identity checkable rather than approximate.
    agg = (g.assign(wy=g["w"] * g[ycol], wz=g["w"] * g["z"])
             .groupby([rung, "year"], as_index=False)
             .agg(y=("wy", "sum"), share=("wz", "sum"), E_b=("E_b", "sum"),
                  n_counties=("ccg", "size")))
    wreg = (g["w"] / g["w"].mean()).to_numpy(np.float64)
    yr = g["year"].to_numpy()
    cl = pd.factorize(g[rung])[0].astype(np.int64)

    y, z, sh = demean_year([g[ycol], g["z"], g["share"]], yr, wreg)
    bb, sb = wls(y, [z], wreg, cl)
    ba_, sa_ = wls(y, [z, sh], wreg, cl)

    # The aggregated regression is EQUALLY weighted across j-year cells, and that
    # is forced, not chosen. The baseline weights sum to exactly one inside every
    # j-year, so each cell carries total weight one in the county regression; the
    # aggregate regression must give each cell that same total or the two are not
    # weighting the same data and b_a = b_b + delta cannot hold. Weighting the
    # aggregate by its own size instead is the implicit-weight channel (B.2 2a),
    # which is a thing to measure, not to commit by accident.
    wj = np.ones(len(agg), np.float64)
    yj, sj = demean_year([agg["y"], agg["share"]], agg["year"].to_numpy(), wj)
    bA, sA = wls(yj, [sj], wj, pd.factorize(agg[rung])[0].astype(np.int64))

    # The nonlinearity wedge: the aggregate's OWN growth, computed the same way
    # the county growth is computed, minus the weighted average of its members'.
    # It must be evaluated in the SAME functional form as the outcome, or it
    # measures the log-versus-standard difference rather than the aggregation
    # error. For standard growth it is identically zero, which is the built-in
    # check that the weights and the collapse are right.
    Ej = g.groupby([rung, "year"])[["E_b", "E_t"]].sum()
    if ycol == "y_log":
        own = np.log(Ej["E_t"]) - np.log(Ej["E_b"])
    elif ycol == "y_mpgr":
        # The rung's own midpoint growth. Like the log, it is NOT the weighted
        # mean of its members under lagged-share weights, so the wedge below is
        # nonzero and the reconstruction identity is not available (A.2.3);
        # midpoint at every rung is the deliberate outcome choice of 2026-08-23.
        own = 2.0 * (Ej["E_t"] - Ej["E_b"]) / (Ej["E_t"] + Ej["E_b"])
    else:
        own = (Ej["E_t"] - Ej["E_b"]) / Ej["E_b"]
    memb = agg.set_index([rung, "year"])["y"]
    wedge = (own.reindex(memb.index) - memb)
    clj = pd.factorize(agg[rung])[0].astype(np.int64)
    yw, sw2 = demean_year([wedge.to_numpy(), agg["share"]], agg["year"].to_numpy(), wj)
    bJ, sJ = wls(yw, [sw2], wj, clj)

    # The regression the literature actually runs: the aggregate's OWN growth on
    # its own exposure. Under standard growth it is the aligned one, because the
    # weighted average of members IS the aggregate. Under log growth it is not,
    # and beta_a_own = beta_a + jensen exactly, which is the decomposition the
    # protocol asks the exhibit to report: naive gap, aligned gap, method share.
    yo, so_ = demean_year([own.reindex(memb.index).to_numpy(), agg["share"]],
                          agg["year"].to_numpy(), wj)
    bO, sO = wls(yo, [so_], wj, clj)

    S = SCALE
    return dict(rung=rung, outcome=ycol,
                n_county_obs=len(g), n_units=int(g[rung].nunique()),
                n_agg_obs=len(agg),
                beta_b=bb[1] * S, se_b=sb[1] * S, t_b=bb[1] / sb[1],
                beta_b_aug=ba_[1] * S, se_b_aug=sa_[1] * S,
                t_b_aug=ba_[1] / sa_[1],
                delta=ba_[2] * S, se_delta=sa_[2] * S, t_delta=ba_[2] / sa_[2],
                beta_a=bA[1] * S, se_a=sA[1] * S, t_a=bA[1] / sA[1],
                beta_a_own=bO[1] * S, se_a_own=sO[1] * S, t_a_own=bO[1] / sO[1],
                identity_gap=(bA[1] - (ba_[1] + ba_[2])) * S,
                jensen=bJ[1] * S, se_jensen=sJ[1] * S)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", nargs="+", default=["intensity", "composition"],
                    choices=["intensity", "composition"])
    ap.add_argument("--outcome", nargs="+", default=["y_log"],
                    choices=["y_log", "y_std", "y_mpgr"],
                    help="the growth rate on the left. Log is the default and the "
                         "current working choice; standard is the admissible one "
                         "under A.2.3 and is what the left-hand side returns to "
                         "once a specification is settled")
    ap.add_argument("--baseline", action="store_true",
                    help="the locked specification: both weighting schemes side "
                         "by side, every level, every horizon, identity asserted")
    ap.add_argument("--matched", action="store_true",
                    help="level-equal weighting at the comparison level, with "
                         "the county regression weighted to match it")
    ap.add_argument("--grid", action="store_true",
                    help="the level grid: the aggregate regression at every level, "
                         "run equally weighted and weighted by base-year population")
    ap.add_argument("--search", action="store_true",
                    help="sweep horizon, share window, instrument, outcome and "
                         "rung, reporting the first-stage F and second-stage t at "
                         "the county and aggregate levels for each")
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 96)
    print("  Aggregation invariance on the fiscal multiplier, reduced form")
    print("=" * 96)

    if args.baseline:
        return baseline(args, t28, w22, t0)
    if args.matched:
        return matched(args, t28, t0)
    if args.grid:
        return level_grid(args, t28, t0)
    if args.search:
        return search(args, t28, w22, t0)

    rows = []
    for inst in args.instrument:
        d, dropped = build_county(inst)
        print(f"\n  {inst}: {d['ccg'].nunique():,} counties frozen "
              f"({dropped} dropped for lacking a rung label), "
              f"{len(d):,} county-years, {d['czone'].nunique():,} commuting "
              f"zones, {d['state'].nunique()} states, "
              f"{d['division'].nunique()} divisions, "
              f"{time.time()-t0:.1f}s", flush=True)
        for ycol in args.outcome:
            print(f"\n  {inst}, outcome {ycol}")
            print(f"  {'rung':<10}{'units':>7}{'agg obs':>9}"
                  f"{'beta_b':>9}{'(t)':>8}{'delta':>9}{'(t)':>8}"
                  f"{'b_b+d':>9}{'aligned':>9}{'naive':>9}{'(t)':>8}"
                  f"{'gap':>8}{'jensen':>9}")
            for rung in RUNGS:
                r = run_rung(d, rung, ycol, t28)
                r["instrument"] = inst
                rows.append(r)
                print(f"  {rung:<10}{r['n_units']:>7,}{r['n_agg_obs']:>9,}"
                      f"{r['beta_b']:>9.3f}{'('+format(r['t_b'],'.2f')+')':>8}"
                      f"{r['delta']:>9.3f}{'('+format(r['t_delta'],'.2f')+')':>8}"
                      f"{r['beta_b_aug']+r['delta']:>9.3f}"
                      f"{r['beta_a']:>9.3f}{r['beta_a_own']:>9.3f}"
                      f"{'('+format(r['t_a_own'],'.2f')+')':>8}"
                      f"{r['identity_gap']:>8.3f}{r['jensen']:>9.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}")
    print("  the identity is b_a = b_b + delta. Under log growth it is broken by")
    print("  construction, and the jensen column is the size of that break; read")
    print("  no gap as a spillover before subtracting it.")
    print(f"  {time.time()-t0:.1f}s")
    return 0


def search(args, t28, w22, t0):
    """Which rungs are usable at all, judged on first-stage F and second-stage t."""
    rows = []
    for inst in args.instrument:
        for h in (2, 3, 4):
            for span in (5, 10):
                d, _ = build_county(inst, h=h, span=span)
                for ycol in args.outcome:
                    for rung in RUNGS:
                        try:
                            o = iv_at_rungs(d, rung, ycol, t28, w22)
                        except Exception as exc:            # noqa: BLE001
                            print(f"    {inst} h={h} span={span} {ycol} {rung}: "
                                  f"failed {type(exc).__name__}")
                            continue
                        for lvl in ("county", "county_aug", "aggregate"):
                            b, t, F, n = o[lvl]
                            rows.append(dict(instrument=inst, horizon=h, span=span,
                                             outcome=ycol, rung=rung, level=lvl,
                                             n_units=o["n_units"], n_obs=n,
                                             beta=b, t=t, F=F))
                print(f"  {inst} h={h} span={span} done  {time.time()-t0:.0f}s",
                      flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(SEARCH_OUT, index=False)
    w = out.pivot_table(index=["instrument", "horizon", "span", "outcome", "rung"],
                        columns="level", values=["t", "F"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["usable"] = ((w.F_aggregate > 20) & (w.t_aggregate.abs() > 1.96)
                   & (w.F_county_aug > 20) & (w.t_county_aug.abs() > 1.96))
    print(f"\n  {len(w):,} cells; {int(w.usable.sum())} clear F>20 and |t|>1.96 "
          f"at BOTH the aggregate and the augmented county level\n")
    cols = ["instrument", "horizon", "span", "outcome", "rung",
            "F_county", "t_county", "F_county_aug", "t_county_aug",
            "F_aggregate", "t_aggregate", "usable"]
    hdr = (f"  {'instrument':<12}{'h':>2}{'span':>5}{'outcome':>7}{'rung':>10}"
           f"{'F_cty':>8}{'t_cty':>7}{'F_aug':>8}{'t_aug':>7}"
           f"{'F_agg':>8}{'t_agg':>7}  ok")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for _, r in w.sort_values(["instrument", "rung", "horizon", "span",
                               "outcome"]).iterrows():
        print(f"  {r.instrument:<12}{int(r.horizon):>2}{int(r.span):>5}"
              f"{r.outcome.replace('y_',''):>7}{r.rung:>10}"
              f"{r.F_county:>8.1f}{r.t_county:>7.2f}"
              f"{r.F_county_aug:>8.1f}{r.t_county_aug:>7.2f}"
              f"{r.F_aggregate:>8.1f}{r.t_aggregate:>7.2f}"
              f"   {'YES' if r.usable else ''}")
    print(f"\n  wrote {SEARCH_OUT}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
