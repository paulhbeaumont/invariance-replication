#!/usr/bin/env python3
"""The symmetric first-stage grid: geography-by-year effects, no county effect.

Reads the panels 47 writes and estimates the first stage in every cell of

    era          pre 1979-1997, post 1998-2018      run separately, never pooled
    rule         net, positive
    horizon      2, 3, 4
    endogenous   level_inc, level_emp, level_pop, std, log, mpgr
    share        fixed4, fixed5, fixed10
    trim         none, top15, gt5pct
    fixed effect division x year, state x year
    winsor       none, tail1, iq4, iq5, iq10, iq20   ON THE ENDOGENOUS ONLY
    group        pooled, year, division

10,368 cells per era. The instrument is never winsorized, at any setting.

THE TRIM EXISTS BECAUSE THE EXPOSURE IS EXTREMELY CONCENTRATED. Untrimmed, ten
county-years out of 12,320 carry two thirds of the demeaned instrument's
variance and the effective sample size is about 12, half of it Los Angeles. A
clustered standard error on 1,395 counties is then reporting information the
design does not have.

    none     the naive estimate, reported so the concentration is visible
    top15    drop the 15 counties with the largest exposure. A fixed count is
             symmetric across eras by construction, and it lands the effective
             sample size at 381 pre against 370 post
    gt5pct   drop counties above 5% of national award dollars in the exposure
             window. A threshold is the more natural rule to state but it is
             NOT symmetric here: the exposure distribution shifts between eras,
             so it catches one county pre and two post

The trim is defined on the exposure, so it is computed within each (era, rule,
share window) and never on the outcome or the F.

NO COUNTY FIXED EFFECT ANYWHERE. In a stacked-differences design the level
effect is already gone, so a unit dummy is a unit-specific trend rather than a
unit level, and at four or five windows per county it consumes the identifying
variation instead of cleaning it. Division x year is the only absorbed term.

What that costs, and it should be said rather than discovered: within a
division-year the national factor is common to every county, so the instrument
is the fixed share rescaled. Identification is cross-sectional exposure inside a
Census division and window, which is the standard shift-share assumption and a
stronger one than a within-county comparison.

The estimator is 28_table_d1.py's `fit2sls` called with the instrument in both
the regressor and the instrument slot, which makes it the OLS first stage with
the same demeaning, the same CRV1 and the same small-sample correction every
other exhibit in this project uses.

Output:
    batch_logs/macro_version/symmetric_growth_first_stage_grid.csv
    batch_logs/macro_version/48_symmetric_growth_first_stage_grid.log
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
OUT = dc.LOG_DIR / "symmetric_growth_first_stage_grid.csv"

ERAS = ("pre", "post")
RULES = ("net", "positive")
HORIZONS = (2, 3, 4)
ENDOG = ("level_inc", "level_emp", "level_pop", "std", "log", "mpgr")
SHARES = ("fixed4", "fixed5", "fixed10")
TRIMS = ("none", "top15", "gt5pct")
FES = ("division_year", "state_year")
TRIM_N = 15
TRIM_PCT = 0.05
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", nargs="+", default=list(ERAS), choices=list(ERAS))
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 78)
    print("  Symmetric first-stage grid: division x year FE, no county FE")
    print("=" * 78)
    print("  endogenous winsorized, instrument never; clustered on the county")

    rows = []
    for era in args.era:
        for rule in RULES:
            for h in HORIZONS:
                panel = pd.read_parquet(IN_DIR / f"{era}_h{h}_{rule}.parquet")
                panel["division_year"] = (panel["division"] + "_"
                                          + panel["year"].astype(str))
                panel["state_year"] = (panel["state"].astype(str) + "_"
                                       + panel["year"].astype(str))
                for share in SHARES:
                    dall = panel[panel["share_label"] == share]
                    cuts = trimmed_counties(dall.groupby("ccg")["share"].first())
                    for trim, drop in cuts.items():
                      d = dall[~dall["ccg"].isin(drop)] if drop else dall
                      for endog in ENDOG:
                        ycol = f"y_{endog}"
                        zcol = f"z_{endog}"
                        sub = d[np.isfinite(d[ycol].to_numpy())
                                & np.isfinite(d[zcol].to_numpy())]
                        if sub.empty:
                            continue
                        yv = sub[ycol].to_numpy(np.float64)
                        zv = sub[zcol].to_numpy(np.float64)
                        ccg = pd.factorize(sub["ccg"])[0].astype(np.int64)
                        FEV = {k: pd.factorize(sub[k])[0].astype(np.int64)
                               for k in FES}
                        zero = np.zeros(len(sub), np.int64)
                        one = np.ones(len(sub))
                        for wrule, wgroup in WINSOR:
                            key = {"pooled": np.zeros(len(sub), np.int64),
                                   "year": sub["year"].to_numpy(),
                                   "division": pd.factorize(sub["division"])[0],
                                   "none": np.zeros(len(sub), np.int64)}[wgroup]
                            yw = winsorize(yv, wrule, key)
                            for fename in FES:
                                fe = FEV[fename]
                                try:
                                    # z is x, so fit2sls takes its OLS branch: beta
                                    # and se ARE the first-stage coefficient and its
                                    # CRV1 standard error, F their squared ratio.
                                    r = t28.fit2sls(yw, zv, zv, one, fe, zero, ccg,
                                                    singleton=False)
                                    beta, se = r["beta"], r["se"]
                                    F = (beta / se) ** 2 \
                                        if se and np.isfinite(se) and se > 0 else np.nan
                                    status = "ok"
                                except Exception as exc:            # noqa: BLE001
                                    status = f"failed: {type(exc).__name__}"
                                    beta = se = F = np.nan
                                rows.append(dict(
                                    era=era, rule=rule, horizon=h, endogenous=endog,
                                    share=share, trim=trim, n_trimmed=len(drop),
                                    fe=fename, winsor_rule=wrule, winsor_group=wgroup,
                                    n_obs=len(sub), n_counties=int(sub["ccg"].nunique()),
                                    n_windows=int(sub["year"].nunique()),
                                    n_fe_cells=int(sub[fename].nunique()),
                                    fs_coef=beta, fs_se=se, fs_F=F, status=status))
                print(f"  {era:>4} {rule:>8} h={h}  {len(rows):>6,} cells  "
                      f"{time.time() - t0:>6.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    ok = out["status"].eq("ok")
    print(f"\n  {len(out):,} cells, {int((~ok).sum())} failed")
    print(f"  wrote {OUT}")
    print(f"  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
