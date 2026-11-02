#!/usr/bin/env python3
"""Table VI's grid, OLS only, under five restrictions of the county set.

WHAT THIS ASKS, AND WHY IT IS NOT MORE WEIGHTING

20_first_stage_grid.py ran every fit twice, OLS and WLS on the unit's lagged
employment, and found that weighting raises the post-1998 county F fivefold -
4.85 to 24.96 - without bringing the sign back. That buys precision on a negative
coefficient, so the weighting is dropped and this file is OLS throughout. Table VI
stays as the single record of what it was worth.

The alternative to reweighting the counties is removing them, and the reference
for that is Auerbach, Gorodnichenko and Murphy (step_1/s5-auerbach.pdf). Their
unit of analysis is the core-based statistical area, "one or more counties (or
equivalents) with an urban center of at least 10,000 people" (footnote 4), and
their data filters include, verbatim: "we exclude geographical locations with
zero DOD spending because these locations likely provide a poor control group"
(section 3C). Three rules follow, and they are reported in this order because
this is their order of promise - (b) is the only one of the three that does not
select on the treatment:

    b   pop_b500, pop_b1000   drop the smallest counties by 1969 population
    c   cbsa_any, cbsa_metro  drop counties in no 2003 CBSA, and the harsher
                              variant that also drops micropolitan
    a   nonzero               drop counties whose h-year contract window is
                              zero in any estimated year

THE GRID IS TABLE VI's, LESS TWO THINGS

    era        pre and post only; 20's third era, post-on-DCADS-alone
               1999-2006, is dropped
    weighting  OLS only; 20's emp-weighted column is dropped
    exposure   PAIRED to the era, as in 20: fixed 1969-74 on pre, fixed_9397
               1993-97 on post, rolling_t5 on both
    rungs      county, commuting zone, state
    h          2, 3 and 4

h AND THE ROLLING EXPOSURE BOTH MOVED ON 2026-08-13

Tables VII and VIII were h=1 and are now h=2; Table IX keeps h=3 and h=4. And the
rolling exposure is rolling_t5, the trailing-5 window ending at t-5 rather than at
t-h-1, so its distance from the sample no longer tracks the horizon - 14's
docstring carries why. Both come in through 20: DESIGN below is a filter of
g20.DESIGN, so this file cannot end up pairing differently from Table VI.

`rolling` is still built, as an anchor exposure, and is not a design row.

fixed_9397 and not fixed_9398, which is 14's own post window: a window ending in
1998 bars every year y with y - h <= 1998, so at h=1 it would contaminate 1999
and the post era would have to start at 2000. 19 built the 1993-97 window for
exactly this reason and Table VI uses it, which is also what lets the six post
cells here anchor against Table VI.

h=4 HAS NO PRODUCTION BLOCK, AND CHECK 3 IS WHERE THAT BITES

bi.build_horizon is generic in h and sym.periods_for(4) gives pre 1979-1997 and
post 2002-2018, with the seam masking 2007-2009. But 14's sweep is h=1,2,3, so
there is no h=4 block in instrument_ccg_year.parquet, and a check that filters the
production file on h.isin(horizons) merges zero rows, reports "max dev 0.0" and
passes vacuously. The two-branch version that repairs this was written here and
now lives in 19 as rebuild_deviation, because 19 reports h=4 too and needed it;
check 3 below calls it. 20's copy was replaced by the same call on 2026-08-13, so
the vacuity README 1.8.8 records against 20 by name is closed.

THE SCREENS ARE COUNTY SETS, NOT COUNTY-YEAR MASKS

Each screen is one frozen set of CCGs, applied before the era split and outside
the era loop, in the same place 20 puts its finiteness screen. That is what keeps
the two eras differing only in which years they contain. `nonzero` is the only
h-dependent one: its window length is h, so it is one set per horizon, defined on
the union of that horizon's two eras rather than one set per era.

Every screen reports the share of defense dollars it drops beside the count of
counties it drops. That is the statistic that says whether a screen is cheap, and
a table of F values without it invites the reader to assume the worst.

Output:
    dc.LOG_DIR/sample_screen_grid.csv
    dc.LOG_DIR/23_sample_screen_grid.log
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 14 owns the build and the estimator, 19 the eras and the logging, 20 the
# windows, the era registration and the unit frame. All three are imported and
# none is restated; none writes anything on import, their writers being reachable
# only from main().
#
# 14 AND 19 ARE TAKEN FROM 20 AND NOT LOADED AGAIN, WHICH MATTERS.
# importlib.util.module_from_spec builds a NEW module object every call, so a
# third `_load("build_instrument", ...)` here would give this file a private copy
# of 14 with its own PERIODS dict. g20.register_variants writes the era bounds
# into the copy 20 holds, and g20.unit_frame reads the estimator from that same
# copy - so a private copy here would be registered into by nobody and
# bi.contaminated_years would raise KeyError on the first design row. It did.
g20 = _load("first_stage_grid", "20_first_stage_grid.py")
bi = g20.bi
sym = g20.sym

# Table VI's design with the two post_dcads rows removed. Taken from 20 by
# filtering rather than retyped, so a change there cannot leave this file
# reporting a different pairing under the same era names.
#
# WIDENED 2026-08-13, having been narrowed the same day. `fixed` x post was
# excluded for one revision because the screen tables did not show it; both it and
# `fixed` x post_pre2017 are now in, on request, and Tables VII-IX and XIV-XVI
# carry all six pairs. The exclusion list is gone rather than inverted - the eras
# this file reports are the whitelist, and post_dcads is still the only thing it
# drops from 20.
ERAS = ("pre", "post", "post_pre2017")
DESIGN = tuple((e, x) for e, x in g20.DESIGN if x in ERAS)
ERA_LABEL = {"pre": "pre", "post": "post",
             "post_pre2017": "post, pre-vintage-break"}

# Everything that has to be BUILT: the design's exposures plus the anchors'.
# `rolling` is here and not in DESIGN, which is exactly the distinction - it earns
# its build by reproducing a published number in check 4, not by being reported.
EXPOSURES = tuple(dict.fromkeys([e for e, _ in DESIGN]
                                + list(sym.ANCHOR_EXPOSURES)))
ROLL_NAMES = tuple(n for n in bi.ROLLING if n in set(EXPOSURES))
RUNGS = g20.RUNGS
WINDOWS = g20.WINDOWS

OMEGA_COL = g20.OMEGA_COL
DEFAULT_OMEGA = g20.DEFAULT_OMEGA

HORIZONS = (2, 3, 4)
POP_YEAR = 1969        # CAINC1's first year, and the first year of the `fixed`
                       # exposure window, so the ranking is pre-sample for both
                       # eras. There is no public flat file of 1960 county
                       # population that does not need an NHGIS or ICPSR account.

# Rule (b)'s two cuts, in counties.
POP_DROPS = (500, 1000)

TOL_BUILD = 1e-12
RTOL_FIT = g20.RTOL_FIT
ATOL_FIT = g20.ATOL_FIT
TOL_TABLE_F = g20.TOL_TABLE_F

# Check 6's target: the screen sizes, measured once and asserted from then on. A
# screen whose size drifts is a screen that stopped meaning what the README says
# it means, and nothing else in the run would notice.
# `nonzero` falls with h and it falls steeply: a one-year window is empty far
# more often than a four-year one, so the screen that is meant to remove counties
# with no treatment removes two thirds of the country at h=1 and less than half
# at h=4. That is the reason rule (a) was posed at h=3 and h=4 and not at h=1.
EXPECT_DROPPED = {"all": 0, "pop_b500": 494, "pop_b1000": 994,
                  "cbsa_any": 1358, "cbsa_metro": 2026,
                  "nonzero": {1: 2149, 2: 1838, 3: 1628, 4: 1438}}

DEMEANER = None


def check(failures, ok, num, label, detail=""):
    """One numbered PASS/FAIL line, in 20_first_stage_grid.py's format."""
    print(f"    [{num}] {label:<54s} {detail:<44s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


def eras_for(h):
    """The two eras this table reports, from 19's periods_for by way of 20."""
    eras, straddle = g20.eras_for(h)
    return {k: v for k, v in eras.items() if k in ERA_LABEL}, straddle


def era_years(h):
    """Every year either era estimates at this horizon, as a sorted list."""
    eras, _ = eras_for(h)
    ys = set()
    for y0, y1 in eras.values():
        ys |= set(range(y0, y1 + 1))
    return sorted(ys)


def fit(frame, fe, weights=None):
    """bi.fit_rung with the demeaner pinned, exactly as 20 does it."""
    return bi.fit_rung(frame, fe, weights=weights, demeaner=DEMEANER)


def only_added(before):
    """(ok, detail) - 19's only_added, read against the module 20 mutated.

    Restated here rather than called, which is the one place in this file that
    does restate something, and the reason is the module-identity trap the import
    block records. sym.only_added closes over 19's OWN copy of 14, which
    g20.register_variants never touches, so calling it compares a pristine
    snapshot against a pristine dict and passes whatever happened. This version
    reads `bi`, which is the copy that was registered into and the copy the fits
    come from.
    """
    fw, pe, ad = before
    bad = []
    for k, v in fw.items():
        if bi.FIXED_WINDOWS.get(k) != v:
            bad.append(f"FIXED_WINDOWS[{k}]")
    for k, v in pe.items():
        if bi.PERIODS.get(k) != v:
            bad.append(f"PERIODS[{k}]")
    for k, v in ad.items():
        if set(bi.ADMISSIBLE.get(k, ())) != v:
            bad.append(f"ADMISSIBLE[{k}]")
    added = (set(bi.PERIODS) - set(pe)) | set()
    return not bad, (f"nothing redefined, {len(added)} era key(s) added"
                     if not bad else ", ".join(bad))


# --------------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------------

def load_population():
    """ccg -> population in POP_YEAR, from 12_build_county_income.py's parquet.

    LineCode 2 of the same CAINC1 file the income denominator comes from, put on
    the CCG spine through the same four declared reconciliations - the Virginia
    combinations, the Alaska time-split, the two straddles, the window filter.
    Kalawao is absent for the same reason it has no income, and is therefore
    absent from every population screen as well as from the estimation support.
    """
    p = dc.PANEL_OUT_DIR / "population_ccg_year.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run 12_build_county_income.py, which now writes it")
    d = pd.read_parquet(p)
    d = d[d["year"] == POP_YEAR]
    return d.set_index("ccg")["population"].sort_values()


def load_cbsa():
    """(any_cbsa, metro_cbsa) as sets of county FIPS, from 21's reference file."""
    if not dc.CBSA_CROSSWALK.exists():
        raise FileNotFoundError(
            f"{dc.CBSA_CROSSWALK} not found - run 21_ingest_cbsa.py")
    x = pd.read_csv(dc.CBSA_CROSSWALK, dtype={"county_fips5": str})
    return (set(x["county_fips5"]),
            set(x.loc[x["cbsa_type"] == "Metropolitan", "county_fips5"]))


def build_screens(universe, county, horizons):
    """{name: {h: frozenset of CCGs kept}} plus a per-screen label.

    A CCG is a GROUP of counties, so "in a CBSA" means ANY member is. The
    alternative - every member - would drop a metropolitan county for having
    been merged with a rural neighbour by the 1990 harmonisation, which is a
    property of our crosswalk and not of the delineation.
    """
    uni = set(int(c) for c in universe)
    pop = load_population()
    any_cbsa, metro_cbsa = load_cbsa()
    fips_to_ccg, _, _ = dc.load_ccg_map()

    in_any = {int(fips_to_ccg[f]) for f in any_cbsa if f in fips_to_ccg}
    in_metro = {int(fips_to_ccg[f]) for f in metro_cbsa if f in fips_to_ccg}

    fixed = {"all": (uni, "no screen")}
    for n in POP_DROPS:
        cut = int(pop.iloc[n - 1])
        fixed[f"pop_b{n}"] = (uni - set(int(c) for c in pop.index[:n]),
                              f"drop the {n} smallest by {POP_YEAR} population, "
                              f"cutoff {cut:,}")
    fixed["cbsa_any"] = (uni & in_any, "in a 2003 CBSA, metro or micro")
    fixed["cbsa_metro"] = (uni & in_metro, "in a 2003 METROPOLITAN CBSA")

    screens = {k: {h: frozenset(v) for h in horizons} for k, (v, _) in fixed.items()}
    labels = {k: lab for k, (_, lab) in fixed.items()}

    # Rule (a). own_dD is the h-year window sum of contract dollars, the exact
    # numerator of the shock, taken from the frame that built it rather than
    # recomputed - a second rolling sum here is how this screen would end up
    # describing a quantity the regression does not use. NaN rows are the seam
    # mask and are not estimated anywhere, so they are not evidence of a zero.
    keep_nonzero = {}
    for h in horizons:
        d = county[(county["h"] == h) & county["year"].isin(era_years(h))]
        d = d[d["own_dD"].notna()]
        dark = set(int(c) for c in d.loc[d["own_dD"] == 0, "ccg"].unique())
        keep_nonzero[h] = frozenset(uni - dark)
    screens["nonzero"] = keep_nonzero
    labels["nonzero"] = "no zero h-year contract window in any estimated year"
    return screens, labels


def dollar_share(levels, kept, h):
    """{era: % of real contract dollars the screen drops}, per era at this h."""
    eras, _ = eras_for(h)
    out = {}
    for era, (y0, y1) in eras.items():
        d = levels[levels["year"].between(y0, y1)]
        tot = float(d["dollars_real"].sum())
        drop = float(d.loc[~d["ccg"].isin(kept), "dollars_real"].sum())
        out[era] = 100.0 * drop / tot if tot else np.nan
    return out


# --------------------------------------------------------------------------------
# The build and the fits
# --------------------------------------------------------------------------------

def build(horizons):
    """The instrument on every exposure, in memory, at 14's production settings.

    20's build(), unchanged but for the horizons. The levels reach back to
    EXPOSURE_FIRST because 1969-74 needs years the estimation sample does not
    contain.
    """
    levels, universe, _ = bi.build_levels(bi.DEFAULT_FLOOR, True,
                                          bi.EXPOSURE_FIRST, sym.LAST_YEAR)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)

    fixed = {}
    for name, window in WINDOWS.items():
        s = bi.fixed_exposure(levels, window, name)
        fixed[name] = s
        print(f"    {name:<11} {window[0]}-{window[1]}   mean {s.mean():.5f}   "
              f"sd {s.std():.5f}   {int((s == 0).sum()):,} counties at zero")
    for n in ROLL_NAMES:
        print(f"    {n:<11} {g20.window_label(n)}, at least {bi.ROLL_MIN} usable")

    frames = []
    for h in horizons:
        ex = dict(fixed)
        ex.update(bi.rolling_exposures(levels, h, bi.ROLL_K, bi.ROLL_MIN,
                                       names=ROLL_NAMES))
        f = bi.build_horizon(levels, ex, h, bi.FIRST_YEAR, sym.LAST_YEAR, True,
                             delta=bi.DEFAULT_DELTA)
        frames.append(f)
        v = f["shock"].dropna()
        print(f"    h={h}   {len(v):,} CCG-years carry a shock   sd {v.std():.5f}")
    county = pd.concat(frames, ignore_index=True)

    pre = {name: bi.fixed_window_sums(levels, w).set_index("ccg")
           for name, w in WINDOWS.items()}
    return county, levels, universe, pre


def windows_for(name, h, design=None):
    """Every (role, era, period_key, y0, y1) this exposure reports at h.

    `design` is a parameter rather than a closure over the module constant because
    24 calls this function with ITS OWN design - README 1.8.9 is frozen on the
    original `rolling` and must not follow this file to rolling_t5. It defaults to
    None so that DESIGN is resolved at call time and not at definition time.
    """
    out = []
    eras, _ = eras_for(h)
    for exposure, era in (DESIGN if design is None else design):
        if exposure != name:
            continue
        y0, y1 = eras[era]
        out.append(("variant", era, sym.era_key(era, h), y0, y1))
    if name in sym.ANCHOR_EXPOSURES:
        y0, y1 = bi.PERIODS[sym.ANCHOR_PERIOD]
        out.append(("anchor", sym.ANCHOR_PERIOD, sym.ANCHOR_PERIOD, y0, y1))
    return out


def fit_grid(county, rungs, pre, horizons, exposures, rung_names, omega,
             screens, labels, levels):
    """One row per screen x exposure x era x h x rung. OLS throughout.

    The finiteness screen and the sample screen are applied in the same place,
    OUTSIDE the era loop, so an era differs from another only in which years it
    contains. The order matters and is deliberate: the sample screen goes first,
    because it is a statement about which counties exist, and the finiteness
    screen second, because it is a statement about which of those counties have
    a usable row.
    """
    wanted = [(r, c) for r, c in RUNGS if r in set(rung_names)]
    omega_col = OMEGA_COL[omega]
    rows, repeat_devs = [], []
    for h in horizons:
        share = {k: dollar_share(levels, screens[k][h], h) for k in screens}
        for name in exposures:
            base_full = county[county["h"] == h].copy()
            base_full["z"] = base_full[f"z_{name}"]
            if name in pre:
                base_full["pre_D"] = base_full["ccg"].map(pre[name]["pre_D"])
                base_full["pre_Y"] = base_full["ccg"].map(pre[name]["pre_Y"])
            else:
                base_full["pre_D"] = np.nan
                base_full["pre_Y"] = np.nan
            base_full = base_full.merge(rungs, on="ccg", how="left")
            keep = np.isfinite(base_full[["shock", "emp_growth", "z",
                                          "income_real_lag", "emp_lag"]]
                               .to_numpy()).all(axis=1)
            base_full = base_full[keep & (base_full["emp_lag"] > 0)
                                  & (base_full["income_real_lag"] > 0)]

            for screen, per_h in screens.items():
                kept = per_h[h]
                full = base_full[base_full["ccg"].isin(kept)]
                for role, era, pkey, y0, y1 in windows_for(name, h):
                    # The anchor is the frozen 1975-1998 convention and exists to
                    # reproduce a published number, so it runs on the unscreened
                    # sample only.
                    if role == "anchor" and screen != "all":
                        continue
                    b = full[full["year"].between(y0, y1)]
                    n_con = len(bi.contaminated_years(pkey, name, h))
                    for rung, unit_col in wanted:
                        d = g20.unit_frame(b, rung, unit_col, omega_col)
                        r = fit(d, "unit + year")
                        # min(horizons) and not the literal 1: the probe has to
                        # fire somewhere or check 11 has no cells and fails.
                        if (role == "variant" and screen == "all"
                                and h == min(horizons)):
                            again = fit(d, "unit + year")
                            worst = max(abs(again[k] - r[k])
                                        for k in ("fs_F", "beta", "beta_se"))
                            repeat_devs.append(
                                (f"{name} x {era} x {rung} h={h}", worst))
                        rows.append(dict(
                            role=role, screen=screen, screen_label=labels[screen],
                            n_ccg_screen=len(kept),
                            n_ccg_dropped=len(screens["all"][h]) - len(kept),
                            pct_dollars_dropped=(np.nan if role == "anchor"
                                                 else share[screen].get(era)),
                            era=era, era_label=ERA_LABEL.get(era, era),
                            era_span=f"{y0}-{y1}", exposure=name,
                            exposure_window=g20.window_label(name), rung=rung,
                            n_units=int(d["unit"].nunique()), omega=omega,
                            reg_weights="unweighted", h=h,
                            year_first=y0, year_last=y1, fe="unit+year",
                            contaminated_years=n_con, n_frame=len(d), **r))
    return pd.DataFrame(rows), repeat_devs


# --------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------

def check_eras(failures, horizons):
    """Check 1. All three eras are 20's, at every horizon including 4.

    pre and post are asserted against 19's periods_for directly; the truncation is
    asserted against 20's eras_for, which is where its bound is derived from 11's
    vintage break. Asserting it here against a retyped 2016 would defeat the point
    of deriving it there.
    """
    ok, detail = True, []
    for h in horizons:
        eras, _ = eras_for(h)
        base, _ = sym.periods_for(h)
        ref, _ = g20.eras_for(h)
        ok &= (eras["pre"] == base["pre"] and eras["post"] == base["post"]
               and eras["post_pre2017"] == ref["post_pre2017"])
        detail.append(f"h={h} {eras['pre'][0]}-{eras['pre'][1]}, "
                      f"{eras['post'][0]}-{eras['post'][1]}, "
                      f"{eras['post_pre2017'][0]}-{eras['post_pre2017'][1]}")
    return check(failures, ok, 1, "all three eras are 20's",
                 "; ".join(detail))


def check_build(failures, county, horizons):
    """Check 3. The rebuild is right at every horizon, production or not.

    This file used to carry its own two-branch version of this. It now calls 19's
    rebuild_deviation, which IS that version, moved up to the module every caller
    already imports - 19 needs it too now that it reports h=4, and two copies of a
    check whose whole subject is "do not pass vacuously" is how one of them ends up
    vacuous again. It also compares z_rolling_t5, which the local copy did not.
    """
    ok, detail = sym.rebuild_deviation(county, horizons)
    return check(failures, ok, 3, "the rebuild is right at every horizon", detail)


def check_anchor(failures, grid, horizons, rung_names, omega):
    """Check 4. Every anchor exposure x the frozen 1975-1998 convention, vs 14's sweep.

    THE EXPECTED COUNT COMES FROM THE GRID, not from a module constant, because
    this function has two callers with different anchor sets: this file builds both
    `rolling` and `rolling_t5` and so anchors twice, while 24 is frozen on `rolling`
    alone and anchors once. A constant would be wrong for one of them, and wrong in
    the direction that reads as a missing anchor rather than as a mis-specified
    check.
    """
    a = grid[grid["role"] == "anchor"]
    anchored = sorted(set(a["exposure"]))
    o = pd.read_csv(dc.LOG_DIR / "instrument_first_stage.csv")
    o = o[(o["fe"] == "unit+year") & (o["period"] == sym.ANCHOR_PERIOD)
          & (o["exposure"].isin(anchored))
          & (o["weights"].isin(("unweighted", omega)))]
    m = o.merge(a, on=["rung", "h", "exposure"], suffixes=("_frozen", "_new"))
    # The frozen sweep only ran h=1,2,3, so the anchor is available at whichever
    # of this run's horizons it covers. Asserting on all of them would fail at
    # h=4 for a reason that is not about this file.
    covered = [h for h in horizons if h in set(o["h"])]
    want = len(anchored) * len(covered) * len(rung_names)
    return g20._compare(m, ("fs_F", "beta", "beta_se", "n_obs"), want, 4,
                        f"{'/'.join(anchored)} anchors reproduce 14's sweep",
                        failures)


def check_tablevi(failures, grid, rung_names):
    """Check 5. The unscreened grid IS Table VI's OLS column, on the pairs kept.

    This is the check the whole file rests on. Every screened number is read as a
    movement away from Table VI, so a baseline that is not Table VI makes all of
    them unreadable - and the twelve cells are design rows here, so it costs
    nothing but the assertion. F to 0.01, units and n_frame exact.
    """
    d = grid[(grid["role"] == "variant") & (grid["screen"] == "all")
             & (grid["h"] == g20.HORIZON)]
    if d.empty:
        return check(failures, False, 5, "the unscreened grid is Table VI",
                     f"no h={g20.HORIZON} rows in this run")
    n = d.set_index(["exposure", "era", "rung"])
    bad, checked = [], 0
    for key, (units, obs, f_ols, _f_wls) in g20.REQUESTED.items():
        # Keyed on DESIGN and not on the era name alone: 20's REQUESTED covers
        # blocks this file does not fit - post_dcads, and since 2026-08-13 the
        # `fixed` x post block SKIP excludes - and an era-name test reads those as
        # missing cells rather than as out of scope. The count assertion below is
        # what stops this from quietly checking nothing.
        if (key[0], key[1]) not in DESIGN or key[2] not in set(rung_names):
            continue
        checked += 1
        if key not in n.index:
            bad.append(f"{key} MISSING")
            continue
        row = n.loc[key]
        if abs(float(row["fs_F"]) - f_ols) > TOL_TABLE_F:
            bad.append(f"{key} F {float(row['fs_F']):.2f} want {f_ols:.2f}")
        if int(row["n_units"]) != units:
            bad.append(f"{key} units {int(row['n_units'])} want {units}")
        if int(row["n_frame"]) != obs:
            bad.append(f"{key} n_frame {int(row['n_frame'])} want {obs}")
    ok = not bad and checked == len(DESIGN) * len(rung_names)
    check(failures, ok, 5,
          f"the unscreened h={g20.HORIZON} grid IS Table VI's OLS column",
          f"{checked} cells" if ok else f"{len(bad)} deviation(s) of {checked}")
    for b in bad:
        print(f"          {b}")
    return ok


def check_tablev(failures, grid, horizons, rung_names):
    """Check 6. The unscreened h=3 grid against 19's own output.

    rolling x pre, rolling x post and fixed_9397 x post at h=3 are Table V cells,
    so the one horizon in this run that 19 also reports has nine of its cells
    already published. h=1 is covered by check 5 and h=4 by neither, which is the
    honest scope of what is new here.
    """
    p = dc.LOG_DIR / "first_stage_table.csv"
    if 3 not in horizons:
        return check(failures, True, 6, "the h=3 cells reproduce 19",
                     f"skipped: h={horizons} does not include 3")
    if not p.exists():
        return check(failures, False, 6, "the h=3 cells reproduce 19",
                     f"{p.name} not found - run 19 first")
    cells = g20.TABLEV_CELLS
    is_v = pd.Series(list(zip(grid["exposure"], grid["era"])),
                     index=grid.index).isin(cells)
    d = grid[(grid["role"] == "variant") & (grid["screen"] == "all")
             & (grid["h"] == 3) & is_v]
    o = pd.read_csv(p)
    o = o[(o["role"] == "variant") & (o["fe"] == "unit+year") & (o["h"] == 3)]
    m = o.merge(d, on=["rung", "h", "exposure", "era"], suffixes=("_frozen", "_new"))
    return g20._compare(m, ("fs_F", "beta", "beta_se", "n_obs"),
                        len(cells) * len(rung_names), 6,
                        "section 1.8.6 Table V's h=3 cells reproduce 19", failures)


def check_screens(failures, grid, screens, horizons):
    """Check 7. Every screen is a declared subset that bites and does not empty."""
    bad = []
    base = screens["all"]
    for name, per_h in screens.items():
        want = EXPECT_DROPPED[name]
        for h in horizons:
            kept = per_h[h]
            if not kept <= base[h]:
                bad.append(f"{name} h={h} is not a subset of the universe")
            dropped = len(base[h]) - len(kept)
            target = want[h] if isinstance(want, dict) else want
            if dropped != target:
                bad.append(f"{name} h={h} drops {dropped}, declared {target}")
            if not kept:
                bad.append(f"{name} h={h} is empty")
    ok = not bad
    check(failures, ok, 7, "every screen is the declared subset of the universe",
          f"{len(screens)} screens x {len(horizons)} horizons"
          if ok else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")
    return ok


def check_applied(failures, grid, screens):
    """Check 8. The county rung's unit count tracks the screen.

    A screen that silently failed to apply would produce a full-sample fit under
    a screened label, which reads as "the screen changes nothing" - the exact
    null this table is looking for, arrived at by a bug. The county rung is where
    it is visible, because there n_units IS the retained county count less the
    counties with no usable row.
    """
    d = grid[(grid["role"] == "variant") & (grid["rung"] == "county")]
    bad = []
    for (screen, h), block in d.groupby(["screen", "h"]):
        cap = len(screens[screen][h])
        if int(block["n_units"].max()) > cap:
            bad.append(f"{screen} h={h}: {int(block['n_units'].max())} units > "
                       f"{cap} retained")
        if screen != "all" and int(block["n_units"].max()) >= len(screens["all"][h]):
            bad.append(f"{screen} h={h}: the screen did not bite")
    ok = not bad
    check(failures, ok, 8, "the county unit count tracks each screen",
          f"{d['screen'].nunique()} screens" if ok else f"{len(bad)} deviation(s)")
    for b in bad:
        print(f"          {b}")
    return ok


def check_rest(failures, grid, repeat_devs, horizons, rung_names, screens,
               canonical):
    """Checks 2 is elsewhere; 9, 10 and 11 here."""
    d = grid[grid["role"] == "variant"]
    check(failures, int(d["contaminated_years"].max()) == 0, 9,
          "zero contaminated years on every design row",
          f"max {int(d['contaminated_years'].max())} over {len(d)} rows")

    w_var = (len(DESIGN) * len(horizons) * len(rung_names) * len(screens))
    w_anc = len(sym.ANCHOR_EXPOSURES) * len(horizons) * len(rung_names)
    n_var = int((grid["role"] == "variant").sum())
    n_anc = int((grid["role"] == "anchor").sum())
    label = ("row count is the announced grid" if canonical
             else "row count is the requested subset")
    check(failures, n_var == w_var and n_anc == w_anc, 10, label,
          f"{n_var} of {w_var} design, {n_anc} of {w_anc} anchor")

    worst = max((v for _, v in repeat_devs), default=np.inf)
    where = max(repeat_devs, key=lambda t: t[1])[0] if repeat_devs else "none"
    check(failures, bool(repeat_devs) and worst == 0.0, 11,
          "a refit is bit-identical, the demeaner being pinned",
          f"{len(repeat_devs)} cells refitted, max dev {worst:.1e} ({where})")


# --------------------------------------------------------------------------------

def main():
    global DEMEANER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS),
                    help="horizons to report; each sets its own era bounds")
    ap.add_argument("--rung", nargs="+", default=[r for r, _ in RUNGS],
                    choices=[r for r, _ in RUNGS])
    ap.add_argument("--omega", choices=sorted(OMEGA_COL), default=DEFAULT_OMEGA,
                    help="the AGGREGATION weight above the county. There is no "
                         "regression weight: this table is OLS")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    canonical = (horizons == sorted(HORIZONS)
                 and set(args.rung) == {r for r, _ in RUNGS}
                 and args.omega == DEFAULT_OMEGA)
    sfx = "" if canonical else ("__h" + "-".join(map(str, horizons)) + "_"
                                + "-".join(sorted(args.rung)).replace(" ", "")
                                + "_" + args.omega)

    tee = sym.Tee(dc.LOG_DIR / f"23_sample_screen_grid{sfx}.log")
    try:
        import pyfixest as pf
        DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

        failures = []
        print("=" * 78)
        print("Table VI's grid, OLS only, under five restrictions of the county set")
        print("=" * 78)
        print("  design     " + ", ".join(f"{e} -> {ERA_LABEL[x]}"
                                          for e, x in DESIGN))
        print(f"  h          {horizons}")
        for h in horizons:
            eras, straddle = eras_for(h)
            print(f"    h={h}   " + "   ".join(
                f"{ERA_LABEL[k]} {v[0]}-{v[1]}" for k, v in eras.items())
                + f"   straddle {straddle[0]}-{straddle[1]}, estimated nowhere")
        print(f"  rungs      {', '.join(args.rung)}")
        print(f"  omega      {args.omega} ({OMEGA_COL[args.omega]}) above the "
              f"county, omega = 1 at the county")
        print(f"  regression OLS only - section 1.8.7 is the record of the "
              f"weighted variant")
        print(f"  demeaner   PINNED, fixef_tol=1e-12, fixef_maxiter=100,000")

        print("\n[1] the eras, and the registration")
        check_eras(failures, horizons)
        before = g20.register_variants(horizons)
        ok, detail = only_added(before)
        check(failures, ok, 2, "registering the variants only ADDED keys", detail)

        print("\n[2] the instrument, every exposure, in memory")
        exposures = list(EXPOSURES)
        county, levels, universe, pre = build(horizons)
        rungs = bi.load_rungs(universe)

        print("\n[3] the build, against production and against source")
        check_build(failures, county, horizons)
        sym.check_rolling_pair(failures, county, horizons)

        print("\n[4] the screens")
        screens, labels = build_screens(universe, county, horizons)
        for name, per_h in screens.items():
            for h in horizons:
                kept = per_h[h]
                sh = dollar_share(levels, kept, h)
                tag = f"{name} h={h}" if name == "nonzero" else name
                if name != "nonzero" and h != horizons[0]:
                    continue
                print(f"    {tag:<14} keep {len(kept):>5,}   "
                      f"drop {len(screens['all'][h]) - len(kept):>5,}   "
                      f"drops {sh['pre']:5.2f}% of pre-era $ and "
                      f"{sh['post']:5.2f}% of post-era $   {labels[name]}")

        print("\n[5] the fits")
        grid, repeat_devs = fit_grid(county, rungs, pre, horizons, exposures,
                                     args.rung, args.omega, screens, labels,
                                     levels)
        print(f"    {len(grid)} fits: "
              f"{int((grid['role'] == 'variant').sum())} design, "
              f"{int((grid['role'] == 'anchor').sum())} anchor")

        print("\n[6] the checks")
        check_anchor(failures, grid, horizons, args.rung, args.omega)
        check_tablevi(failures, grid, args.rung)
        check_tablev(failures, grid, horizons, args.rung)
        check_screens(failures, grid, screens, horizons)
        check_applied(failures, grid, screens)
        check_rest(failures, grid, repeat_devs, horizons, args.rung, screens,
                   canonical)

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            report(grid, horizons, args.rung)
            return 1

        if not canonical:
            print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; "
                  f"this run writes to *{sfx}.* instead.")

        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.LOG_DIR / f"sample_screen_grid{sfx}.csv"
        grid.to_csv(out, index=False)
        print(f"\nall checks passed   {len(grid)} fits -> {out}")
        report(grid, horizons, args.rung)
        return 0
    finally:
        tee.close()


# The order the README reports the screens in: (b) first, because it is the only
# one of the three that does not select on the treatment.
REPORT_GROUPS = (("Table VII. Population screens",
                  ("all", "pop_b500", "pop_b1000")),
                 ("Table VIII. The rural screens",
                  ("all", "cbsa_any", "cbsa_metro")),
                 ("Table IX. The zero-contract screen",
                  ("all", "nonzero")))


def report(grid, horizons, rung_names):
    """The three tables, in the shape the README carries them.

    Printed even on failure: a run that failed check 5 is exactly the run whose
    numbers need reading.
    """
    d = grid[grid["role"] == "variant"]
    short = {"fixed": "fixed 1969-74", "fixed_9397": "fixed 1993-97",
             "rolling": f"rolling K={bi.ROLL_K}",
             # 14 chars at most: the report pads this column to 15 and a
             # 15-char label runs into the rung beside it.
             "rolling_t5": f"rolling K={bi.ROLL_K} t5"}
    for title, cols in REPORT_GROUPS:
        cols = [c for c in cols if c in set(d["screen"])]
        for h in horizons:
            s = d[(d["h"] == h) & d["screen"].isin(cols)]
            if s.empty:
                continue
            print(f"\n  {title} - first-stage F, h={h}")
            head = f"    {'era':<11}{'exposure':<15}{'rung':<17}"
            for c in cols:
                head += f"{c:>16}"
            print(head)
            sub = f"    {'':<11}{'':<15}{'counties dropped':<17}"
            for c in cols:
                v = s[s["screen"] == c]
                sub += f"{int(v['n_ccg_dropped'].iloc[0]):>16,}"
            print(sub)
            sub = f"    {'':<11}{'':<15}{'% of pre-era $':<17}"
            for c in cols:
                v = s[(s["screen"] == c) & (s["era"] == "pre")]
                sub += f"{float(v['pct_dollars_dropped'].iloc[0]):>16.2f}"
            print(sub)
            for exposure, era in DESIGN:
                first = True
                for rung, _ in RUNGS:
                    if rung not in set(rung_names):
                        continue
                    r = s[(s["exposure"] == exposure) & (s["era"] == era)
                          & (s["rung"] == rung)]
                    if r.empty:
                        continue
                    lab = r["era_span"].iloc[0] if first else ""
                    exp = short[exposure] if first else ""
                    first = False
                    line = f"    {lab:<11}{exp:<15}{rung:<17}"
                    for c in cols:
                        v = r[r["screen"] == c]
                        line += (f"{float(v['fs_F'].iloc[0]):>16.2f}"
                                 if len(v) else f"{'':>16}")
                    print(line)
    print(f"\n  2SLS beta (se), the same fits")
    for h in horizons:
        for exposure, era in DESIGN:
            s = d[(d["exposure"] == exposure) & (d["era"] == era)
                  & (d["h"] == h)]
            if s.empty:
                continue
            print(f"    h={h}  {short[exposure]} x {ERA_LABEL[era]} "
                  f"({s['era_span'].iloc[0]})")
            for rung, _ in RUNGS:
                r = s[s["rung"] == rung]
                if r.empty:
                    continue
                parts = []
                for c in [x for _, g in REPORT_GROUPS for x in g]:
                    v = r[r["screen"] == c]
                    if len(v):
                        parts.append(f"{c} {float(v['beta'].iloc[0]):+.3f} "
                                     f"({float(v['beta_se'].iloc[0]):.3f})")
                seen, uniq = set(), []
                for p in parts:
                    if p.split()[0] not in seen:
                        seen.add(p.split()[0])
                        uniq.append(p)
                print(f"      {rung:<17}" + "   ".join(uniq))


if __name__ == "__main__":
    raise SystemExit(main())
