#!/usr/bin/env python3
"""The second-stage grid at h = 4: log employment growth on the contract shock.

Reads the panels 47 writes. Same design as 48's first stage, with the outcome
added:

    outcome      dlog_emp     ln E(t0+4) - ln E(t0)
                 ns_emp_rate  (L(t0+4) - L(t0)) / L(t0), L = E / population.
                              This is Nakamura and Steinsson's own left side:
                              "a two-year change divided by the initial value
                              ... employment are per capita" (Table 2 note)
                 demp_pop     (E(t0+4) - E(t0)) / population(t0)

The last two are LEVEL changes, so no disaggregation result applies to them.
They are here to diagnose why the pre era carries no signal, not as candidate
specifications.
    era          pre, post                    run separately, never pooled
    rule         net, positive
    trim         none, gt5pct, top15
    endogenous   level_inc, level_emp, level_pop, std, log, mpgr
    share        fixed4, fixed5, fixed10
    winsor       none, tail1, iq4, iq5, iq10, iq20
    group        pooled, year, division
    fixed effect division x year, state x year

20,736 cells. Division x year fixed effects, no county fixed effect, clustered on
the county, no weights, just-identified 2SLS.

THE WINSORIZATION IS APPLIED TO BOTH SIDES, NEVER TO THE INSTRUMENT. A cap on
the regressor with a raw outcome leaves the outcome's tail free to drive the
second stage while the regressor's tail has been pulled in, which is not one
sample. The rule and the grouping are shared; the cutoffs are computed
separately per variable because a growth rate and a dollar ratio are not in the
same units. The instrument stays raw at every setting, because capping it was
what manufactured the first stage this whole exercise replaced.

Step 2's A.3.2 bars winsorizing a growth rate, and that bar does not reach here:
its reason is that capping cell growth breaks the reconstruction identity, and
no reconstruction runs at the county rung.

THE LOG OUTCOME NEEDS POSITIVE EMPLOYMENT AT BOTH ENDPOINTS of every window, so
a county failing that is dropped from the era outright, the same way the
contract screen works in 47. It costs 0 counties pre and 2 to 3 post.

Output:
    batch_logs/macro_version/symmetric_second_stage_grid.csv
    batch_logs/macro_version/49_symmetric_second_stage_grid.log
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
IN_DIR = dc.MACRO_DATA_ROOT / "temp" / "symmetric_growth_grid"
OUT = dc.LOG_DIR / "symmetric_second_stage_grid.csv"
OUTCOMES = ("dlog_emp", "ns_emp_rate", "demp_pop")

HORIZON = 4
ERAS = ("pre", "post")
RULES = ("net", "positive")
TRIMS = ("none", "gt5pct", "top15")
FES = ("division_year", "state_year")
TRIM_N = 15
TRIM_PCT = 0.05
ENDOG = ("level_inc", "level_emp", "level_pop", "std", "log", "mpgr")
SHARES = ("fixed4", "fixed5", "fixed10")
WINSOR = (("none", "none"),) + tuple(
    (r, g) for r in ("tail1", "iq4", "iq5", "iq10", "iq20")
    for g in ("pooled", "year", "division"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trimmed_counties(exposure):
    """Counties each trim rule REMOVES. `exposure` is one share per county."""
    ranked = exposure.sort_values(ascending=False)
    return {"none": set(),
            "top15": set(ranked.index[:TRIM_N]),
            "gt5pct": set(exposure.index[exposure > TRIM_PCT])}


def winsorize(v, rule, group_key):
    """Median +/- level x IQR, or the 1st and 99th percentiles, within group."""
    if rule == "none":
        return v
    out = v.astype(np.float64).copy()
    for g in np.unique(group_key):
        m = group_key == g
        x = out[m]
        if rule == "tail1":
            lo, hi = np.nanpercentile(x, [1.0, 99.0])
        else:
            level = float(rule[2:])
            q25, q50, q75 = np.nanpercentile(x, [25.0, 50.0, 75.0])
            lo, hi = q50 - level * (q75 - q25), q50 + level * (q75 - q25)
        out[m] = np.clip(x, lo, hi)
    return out


def attach_outcome(panel, emp, pop):
    """The three outcomes, and the counties with a non-positive employment endpoint."""
    cur = np.empty(len(panel)); lag = np.empty(len(panel))
    pcur = np.empty(len(panel)); plag = np.empty(len(panel))
    for t0, sub in panel.groupby("year", sort=True):
        i = sub.index.to_numpy()
        u = sub["ccg"].to_numpy()
        lag[i] = emp[t0].reindex(u).to_numpy(np.float64)
        cur[i] = emp[t0 + HORIZON].reindex(u).to_numpy(np.float64)
        plag[i] = pop[t0].reindex(u).to_numpy(np.float64)
        pcur[i] = pop[t0 + HORIZON].reindex(u).to_numpy(np.float64)
    good = (lag > 0) & (cur > 0) & (plag > 0) & (pcur > 0)
    bad_counties = set(panel.loc[~good, "ccg"].unique())
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_lag = lag / plag
        panel = panel.assign(
            dlog_emp=np.where(good, np.log(cur) - np.log(lag), np.nan),
            ns_emp_rate=np.where(good, (cur / pcur - rate_lag) / rate_lag, np.nan),
            demp_pop=np.where(good, (cur - lag) / plag, np.nan))
    return panel, bad_counties


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", nargs="+", default=list(ERAS), choices=list(ERAS))
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    efsy = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                           columns=["ccg", "year", "emp"])
    emp = {y: g.set_index("ccg")["emp"] for y, g in efsy.groupby("year")}
    popp = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    pop = {y: g.set_index("ccg")["population"] for y, g in popp.groupby("year")}

    t0 = time.time()
    print("=" * 78)
    print(f"  Second stage, h = {HORIZON}: log employment growth on the contract shock")
    print("=" * 78)
    print("  division x year FE, no county FE, clustered on the county")
    print("  the endogenous AND the outcome carry the same cap; the instrument never does")

    rows = []
    for era in args.era:
        for rule in RULES:
            panel = pd.read_parquet(IN_DIR / f"{era}_h{HORIZON}_{rule}.parquet")
            panel = panel.reset_index(drop=True)
            panel, bad = attach_outcome(panel, emp, pop)
            if bad:
                panel = panel[~panel["ccg"].isin(bad)]
            panel["division_year"] = (panel["division"] + "_"
                                      + panel["year"].astype(str))
            panel["state_year"] = (panel["state"].astype(str) + "_"
                                   + panel["year"].astype(str))
            print(f"  {era:>4} {rule:>8}  {panel['ccg'].nunique():,} counties "
                  f"({len(bad)} dropped for a non-positive employment endpoint)")
            for share in SHARES:
                dall = panel[panel["share_label"] == share]
                cuts = trimmed_counties(dall.groupby("ccg")["share"].first())
                for trim, drop in cuts.items():
                    d = dall[~dall["ccg"].isin(drop)] if drop else dall
                    for endog in ENDOG:
                        ycol, zcol = f"y_{endog}", f"z_{endog}"
                        ok = (np.isfinite(d[ycol].to_numpy())
                              & np.isfinite(d[zcol].to_numpy())
                              & np.isfinite(d["dlog_emp"].to_numpy()))
                        sub = d[ok]
                        if sub.empty:
                            continue
                        xv = sub[ycol].to_numpy(np.float64)
                        zv = sub[zcol].to_numpy(np.float64)
                        GV = {o: sub[o].to_numpy(np.float64) for o in OUTCOMES}
                        ccg = pd.factorize(sub["ccg"])[0].astype(np.int64)
                        FEV = {k: pd.factorize(sub[k])[0].astype(np.int64)
                               for k in FES}
                        zero = np.zeros(len(sub), np.int64)
                        one = np.ones(len(sub))
                        keys = {"pooled": np.zeros(len(sub), np.int64),
                                "year": sub["year"].to_numpy(),
                                "division": pd.factorize(sub["division"])[0],
                                "none": np.zeros(len(sub), np.int64)}
                        for wrule, wgroup in WINSOR:
                            key = keys[wgroup]
                            xw = winsorize(xv, wrule, key)
                            GW = {o: winsorize(GV[o], wrule, key) for o in OUTCOMES}
                            for outcome in OUTCOMES:
                              gw = GW[outcome]
                              for fename in FES:
                                fe = FEV[fename]
                                try:
                                    r = t28.fit2sls(gw, xw, zv, one, fe, zero, ccg,
                                                    singleton=False)
                                    beta, se, F = r["beta"], r["se"], r["F"]
                                    status = "ok"
                                except Exception as exc:            # noqa: BLE001
                                    status = f"failed: {type(exc).__name__}"
                                    beta = se = F = np.nan
                                rows.append(dict(
                                    era=era, rule=rule, horizon=HORIZON, trim=trim,
                                    n_trimmed=len(drop), outcome=outcome,
                                    endogenous=endog, share=share,
                                    fe=fename, winsor_rule=wrule, winsor_group=wgroup,
                                    n_obs=len(sub), n_counties=int(sub["ccg"].nunique()),
                                    n_fe_cells=int(sub[fename].nunique()),
                                    beta_2sls=beta, se_2sls=se, first_stage_F=F,
                                    status=status))
            print(f"       {len(rows):>6,} cells  {time.time() - t0:>6.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n  {len(out):,} cells, {int((~out['status'].eq('ok')).sum())} failed")
    print(f"  wrote {OUT}")
    print(f"  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
