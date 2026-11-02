#!/usr/bin/env python3
"""The county-rung multiplier grid, on the design the NS replication validates.

Producer 50 reproduces Nakamura and Steinsson at their own rung on our DD-350
data: state output 1.350 against their 1.43, state employment 1.791 against
their 1.28. This file takes that specification one rung down.

THE TREATMENT, and the one parameter that nests everything we tried

    x_ct = ( sum_j w_j G_c,t-j  -  G_c,t-h ) / X_c,t-h        j = 0 .. h-1
    w_j proportional to LAMBDA^(-j), normalised to sum to one

    LAMBDA = 1     flat window average. Reproduces NS at the state rung to
                   within 3% and gives a stronger first stage.
    LAMBDA -> inf  all weight on the most recent year, which IS NS's own
                   endpoint difference G_t - G_t-h.
    LAMBDA > 1     recency-weighted, between the two. The free-weight estimate
                   at h = 2 wants (+1, -0.32, -0.68) on (G_t, G_t-1, G_t-2),
                   so the data do ask for more weight on the recent year.

    A LEVEL of window spending, with no base subtracted, is not on this line
    and does not work: at the state rung it takes employment from 1.79 to 0.14,
    because a level cannot tell a buildup from a drawdown. The base has to be
    subtracted once per window year, which these weights do by construction.

CONSTRUCTION IS OUT OF THE CONTRACT MEASURE, and putting it back is a robustness
row rather than an alternative default. Federal Supply Classification Y, the
construction of structures and facilities, is excluded by Nakamura and Steinsson's
own state series, and excluding it is what makes the replication of their Table 3
column 2 land at 1.805 against their 1.81 rather than at 1.645. It is not a small
filter and it does not fall on counties evenly: 7.0% of national contract dollars
in FY1975 but 69.9% of Idaho's against 0.4% of Connecticut's.

At this rung it RAISES the estimate. On the baseline cell, excluding construction
moves h = 2 from 4.054 to 4.655, h = 3 from 4.112 to 4.705 and h = 4 from 3.387 to
3.802, on an identical sample of 119,262 county-years, and across 15,529
comparable grid cells the median ratio is 1.176 with 3.7% sign flips. A dollar of
military construction buys less local covered employment than a dollar of
procurement, which is what a March-reference-date establishment count would show
if construction crews are short-lived or brought in.

    --include-construction   the robustness, written to its own CSV

THE OUTCOME is county midpoint growth, 2(E_t - E_t-h)/(E_t + E_t-h), and it is
NEVER winsorized. Midpoint is defined where a log is not, and it is the
transform under which weighted cell growth reconstructs the parent exactly.

THE SAMPLE excludes, whole, any county reporting zero CBP-covered employment in
any year EFSY covers. 42 counties of 3,100 qualify, carrying 0.0065% of national
employment and 0.0226% of population. They are the smallest counties in the
country (Loving County, Texas, median population 83) plus a handful of coverage
holes. Because the outcome is a midpoint growth rate that is never winsorized, a
zero at either end of a window returns exactly the bound |y| = 2, nine standard
deviations from zero against a panel sd of 0.23.

WINSORIZATION touches the endogenous variable and nothing else. Not the
outcome, and NEVER the instrument: capping the instrument is what manufactured
a first-stage F of 492 from a true 7 in the design this replaces, while moving
the coefficient hardly at all. The rule is the 1st and 99th percentiles. IQR
rules are unusable here because more than half of county-years have no shock,
so the interquartile range is degenerate and IQR5 clips 17% of the sample
against 1-99's 2%.

THE DENOMINATOR OF AN EXPOSURE IS SUMMED WITH min_count, so a window containing a
year in which X is missing returns NaN and the cell drops. Without it pandas skips
the missing years and divides a full contract numerator by a partial X sum: EFSY
employment begins in 1975, so a roll5 window on 1971-1975 would sum contracts over
five years and employment over one, inflating that county's exposure fivefold. The
guard binds only where a series starts inside the panel; income and population
reach 1969 and are never touched by it.

THE EXPOSURE is an intensity, contracts over X, never a share of the national
total. A national share is a size measure: under it Los Angeles alone carried
20% of the identifying variance and the effective sample size was 12. Fixed
windows are era-specific and predetermined; rolling windows end a fixed five
years before each observation's own base year, so they are never stale.

THE THREE SPECIFICATIONS pair a fixed effect with the weight that aggregates
counties to the level that effect works within:

    county + year              state-EW      states equally weighted
    county + division x year   division-EW   the nine Census divisions, equally
    county + year              none          every county-year equal

Output (--out overrides the CSV):
    batch_logs/macro_version/county_multiplier_grid.csv
    batch_logs/macro_version/51_county_multiplier_grid.log
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
OUT = dc.LOG_DIR / "county_multiplier_grid.csv"

HORIZONS = (2, 3, 4)
DENOMS = (("P", "population"), ("Y", "income"), ("E", "employment"))
# Reporting scale only; the estimation is untouched. The outcome goes to
# PERCENTAGE POINTS (x100) and spending to UNITS OF $1,000 (x1,000), so
# beta_scaled = beta x 1e5. Income needs no factor: spending as a PERCENT of
# income against growth in PERCENTAGE POINTS cancels, which is the scale
# Nakamura and Steinsson report on and why their coefficient is order one.
SCALE = {"population": 1e5, "income": 1.0, "employment": 1e5}
UNIT = {"population": "pp per $1,000 per head",
        "income": "pp per 1% of income",
        "employment": "pp per $1,000 per job"}

RULES = ("net", "positive")
FLOORS = ((0.0, "none"), (25_000.0, "25k nominal"))
SCREENS = (True, False)
WGROUPS = ("pooled", "year", "state")
LAMBDAS = (0.7, 1.0, 1.3, 1.6, 2.0, 3.0)
# (label, county effect on/off, the time-by-geography effect, weight).
# The division and state levels are run BOTH ways on the county effect so its
# contribution is a clean on-off comparison at each level.
SPECS = (("county+year",         True,  "year",         "state-EW"),
         ("county+year",         True,  "year",         "none"),
         ("county+divisionyear", True,  "divisionyear", "division-EW"),
         ("divisionyear",        False, "divisionyear", "division-EW"),
         ("county+stateyear",    True,  "stateyear",    "state-EW"),
         ("stateyear",           False, "stateyear",    "state-EW"))

# The national shift is the FULL national aggregate. Leaving the focal unit out
# is a robustness, not the default. At this rung it can hardly matter either way,
# since one county of 3,058 is about 0.03% of the aggregate, but the default
# should be the simpler object and the deviation should be the one that is asked
# for. --leave-one-out turns it on.
LEAVE_ONE_OUT = False

ROLL_LAG = 5                      # step 1's rolling_t5 rule
ROLLING = {"roll4": 4, "roll5": 5, "roll10": 10}
DATA_FIRST = 1969                 # income and population both start here

ERAS = {
    "full": dict(lo=1977, hi=2018, first_base=1975,
                 expo={"exp4": (1971, 1974), "exp5": (1970, 1974),
                       "exp10": (1969, 1974), "expNS": (1977, 1981)}),
    "pre":  dict(lo=1977, hi=1997, first_base=1975,
                 expo={"exp4": (1971, 1974), "exp5": (1970, 1974),
                       "exp10": (1969, 1974), "expNS": (1977, 1981)}),
    "post": dict(lo=1998, hi=2018, first_base=1996,
                 expo={"exp4": (1992, 1995), "exp5": (1991, 1995),
                       "exp10": (1986, 1995), "expNS": (1998, 2002)}),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def county_actions(year, fips_to_ccg, drop_construction=True):
    """One year of county-level contract actions, in dollars.

    drop_construction removes Federal Supply Classification Y, "construction of
    structures and facilities". Nakamura and Steinsson's state series excludes it:
    with construction in, our contract totals run a median 1.063 of theirs with a
    cross-state sd of 0.181; with it out, 0.998 and 0.088. It is 69.9% of Idaho's
    FY1975 dollars and 0.4% of Connecticut's, which is why it looks like a
    geography problem and is not one. Whether OUR OWN specification should drop it
    is a separate question, since military construction is local demand like any
    other; the flag exists so both can be estimated rather than assumed.
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


def base_panel():
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
    d = (inc.merge(pop, on=["ccg", "year"]).merge(efs, on=["ccg", "year"], how="left")
         .merge(st, on="ccg"))
    d = d.rename(columns={"income_real": "Y", "population": "P", "emp": "E"})
    d["division"] = d["state"].map(b14.DIVISION)

    # A county reporting no CBP-covered employment in ANY year EFSY covers is
    # dropped whole, every year of it, not just the offending year. The outcome is
    # midpoint growth and is never winsorized, so E = 0 at either end of a window
    # returns exactly the midpoint bound |y| = 2, which is 9 sd from zero against a
    # panel sd of 0.23. Dropping only the zero year would leave the county's other
    # windows in the sample with a base or an endpoint that the same coverage
    # problem touches.
    #
    # The test runs on E.notna() and not on the whole frame: income and population
    # reach back to 1969 while EFSY starts in 1975, so every county is missing
    # employment for 1969-1974 and testing every row would drop all 3,100.
    n_before = d["ccg"].nunique()
    zero_ever = d.loc[d["E"].notna() & (d["E"] <= 0), "ccg"].unique()
    d = d[~d["ccg"].isin(zero_ever)]
    n_after = d["ccg"].nunique()
    if n_after < 0.95 * n_before:
        raise SystemExit(f"zero-employment screen dropped {n_before - n_after} of "
                         f"{n_before} counties; EFSY coverage is not what this "
                         f"screen assumes")
    print(f"  zero-employment screen: dropped {n_before - n_after} counties, "
          f"{n_after:,} kept", flush=True)

    return d.sort_values(["ccg", "year"]).reset_index(drop=True)


def lam_weights(h, lam):
    """Window weights on years t, t-1, ... t-h+1, most recent first."""
    w = np.array([lam ** (-j) for j in range(h)], dtype=np.float64)
    return w / w.sum()


def wins(v, key):
    out = v.astype(np.float64).copy()
    for g in np.unique(key):
        m = key == g
        lo, hi = np.nanpercentile(out[m], [1.0, 99.0])
        out[m] = np.clip(out[m], lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", nargs="+", default=list(ERAS), choices=list(ERAS))
    ap.add_argument("--out", default=None,
                    help="CSV to write; defaults to %s" % OUT.name)
    ap.add_argument("--leave-one-out", action="store_true",
                    help="build the national shift leaving the focal unit out; "
                         "the default uses the full national aggregate")
    ap.add_argument("--include-construction", action="store_true",
                    help="keep FSC/PSC Y, construction of structures and "
                         "facilities. The default excludes it; see the docstring")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else OUT

    global LEAVE_ONE_OUT
    LEAVE_ONE_OUT = args.leave_one_out

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 78)
    print("  County multiplier grid: midpoint employment growth on the contract shock")
    print("=" * 78)

    B = base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    CON = contract_matrices(units, 1966, 2018,
                            drop_construction=not args.include_construction)
    STATE = B.drop_duplicates("ccg").set_index("ccg")["state"]
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]
    print(f"  {len(units):,} counties, contracts built, {time.time()-t0:.1f}s", flush=True)

    rows = []
    for era in args.era:
        cfg = ERAS[era]
        for h in HORIZONS:
            years = [t for t in range(max(cfg["lo"], cfg["first_base"] + h), cfg["hi"] + 1)]
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
            for rule in RULES:
                for floor, flab in FLOORS:
                    C = CON[(rule, floor)]
                    lag = {k: C.reindex(columns=[y - k for y in years])
                           .set_axis(years, axis=1) for k in range(h + 1)}
                    f["G_b"] = lag[h].stack().reindex(idx).to_numpy()
                    allpos = (sum(lag[k] for k in range(h)) > 0).all(axis=1)
                    kept_units = set(units[allpos.to_numpy()])
                    for lam in LAMBDAS:
                        wj = lam_weights(h, lam)
                        win = sum(wj[k] * lag[k] for k in range(h))
                        f["G_win"] = win.stack().reindex(idx).to_numpy()
                        natwin = f.groupby("year")["G_win"].transform("sum")
                        natbas = f.groupby("year")["G_b"].transform("sum")
                        if LEAVE_ONE_OUT:
                            natwin = natwin - f["G_win"]
                            natbas = natbas - f["G_b"]
                        for screen in SCREENS:
                            keep = (f["ccg"].isin(kept_units) if screen
                                    else pd.Series(True, index=f.index))
                            for X, xlab in DENOMS:
                                natX = f.groupby("year")[X + "_b"].transform("sum")
                                if LEAVE_ONE_OUT:
                                    natX = natX - f[X + "_b"]
                                xv = (f["G_win"] - f["G_b"]) / f[X + "_b"]
                                shift = (natwin - natbas) / natX
                                EXPO = {k: ("fixed", v) for k, v in cfg["expo"].items()}
                                EXPO.update({k: ("roll", v) for k, v in ROLLING.items()})
                                for elab, (kind, spec) in EXPO.items():
                                    if kind == "fixed":
                                        a, b = spec
                                        yy = list(range(a, b + 1))
                                        s = (C.loc[:, yy].sum(axis=1)
                                             / piv[X].loc[:, yy].sum(
                                                 axis=1, min_count=len(yy))).reindex(units)
                                        zv = f["ccg"].map(s) * shift
                                        a_rep, b_rep = a, b
                                    else:
                                        cols = {}
                                        for t_ in years:
                                            lo_ = t_ - h - ROLL_LAG - spec + 1
                                            hi_ = t_ - h - ROLL_LAG
                                            if lo_ < DATA_FIRST:
                                                continue
                                            yy = list(range(lo_, hi_ + 1))
                                            cols[t_] = (C.loc[:, yy].sum(axis=1)
                                                        / piv[X].loc[:, yy].sum(
                                                            axis=1, min_count=len(yy)))
                                        if not cols:
                                            continue
                                        sv = pd.DataFrame(cols).reindex(units) \
                                            .stack().reindex(idx).to_numpy()
                                        zv = pd.Series(sv, index=f.index) * shift
                                        a_rep, b_rep = min(cols), max(cols)
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
                                    for wg in WGROUPS:
                                        xw_ = wins(x0, keys[wg])
                                        for felab, use_cty, f2key, wlab in SPECS:
                                            try:
                                                r = t28.fit2sls(
                                                    y0, xw_, z0, WV[wlab],
                                                    f1 if use_cty else FEV[f2key],
                                                    FEV[f2key] if use_cty else NOFE,
                                                    f1.copy(), singleton=False)
                                                st_ = "ok"
                                            except Exception as exc:     # noqa: BLE001
                                                r = dict(beta=np.nan, se=np.nan, F=np.nan)
                                                st_ = f"failed: {type(exc).__name__}"
                                            rows.append(dict(
                                                era=era, horizon=h, lam=lam,
                                                denom=xlab, exposure=elab,
                                                expo_first=a_rep, expo_last=b_rep,
                                                rule=rule, floor=flab, screen=screen,
                                                fe=felab, weight=wlab, winsor_group=wg,
                                                n_obs=len(g),
                                                n_counties=int(g["ccg"].nunique()),
                                                beta=r["beta"], se=r["se"],
                                                beta_scaled=r["beta"] * SCALE[xlab],
                                                se_scaled=r["se"] * SCALE[xlab],
                                                unit=UNIT[xlab],
                                                t=(r["beta"] / r["se"]) if r["se"]
                                                else np.nan,
                                                first_stage_F=r["F"], status=st_))
            print(f"  {era:>4} h={h}  {len(rows):>8,} cells  {time.time()-t0:>8.1f}s",
                  flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"\n  {len(out):,} cells, {int((~out['status'].eq('ok')).sum())} failed")
    print(f"  wrote {out_path}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
