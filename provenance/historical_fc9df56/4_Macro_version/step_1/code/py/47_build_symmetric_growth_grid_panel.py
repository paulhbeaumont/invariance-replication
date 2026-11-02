#!/usr/bin/env python3
"""Build the symmetric pre/post first-stage panels for the division-year grid.

One construction, run identically on both eras, so a pre-versus-post difference
in a coefficient is a difference in the data and not in how the data were made.

    contract quantity   G_ck = sum of award-year real obligations over the
                        window t0+1 .. t0+h, on the $25,000 NOMINAL floor
    obligation rule     net      every action, signed
                        positive only actions with a positive obligation
    county screen       a county enters an era only if G_ck > 0 in EVERY window
                        of that era. A zero or negative window drops the county
                        outright, because the growth transforms below are not
                        defined on it and a panel that silently changes support
                        between transforms is not one grid
    endogenous          level_inc  G_ck / Y_c,t0        income scaled
                        level_emp  G_ck / E_c,t0        employment scaled
                        level_pop  G_ck / P_c,t0        population scaled
                        std        (G_ck - G_c,k-1) / G_c,k-1
                        log        ln G_ck - ln G_c,k-1
                        mpgr       2 (G_ck - G_c,k-1) / (G_ck + G_c,k-1)
    instrument          level_inc  s_c * N_k^-c / Y^nat_t0
                        level_emp  s_c * N_k^-c / E^nat_t0
                        level_pop  s_c * N_k^-c / P^nat_t0
                        growth     s_c * g(N_k^-c), the same transform g applied
                        to the leave-one-county-out national stock

    share s_c           the county's share of national award dollars over a
                        fixed pre-era window, symmetric across eras:
                        pre  FY1975-78 / FY1974-78 / FY1969-78
                        post FY1994-97 / FY1993-97 / FY1988-97

The three level variants carry a denominator; the three growth variants are
ratios of dollars to dollars and carry none, so the scaling choice applies to the
level rows only. Every national denominator is the sum over the SAME candidate
county universe the local one is drawn from, which is what makes the ratio of the
two a national rate rather than two unrelated aggregates. Population is BEA
CAINC1 LineCode 2, from population_ccg_year.parquet, on the areas the income
denominator uses.

THE GROWTH INSTRUMENT CANNOT BE g(s_c * N_k). Both log and mpgr are scale
invariant, so g(s_c N_k) = g(N_k) exactly, which is common to every county in a
window and is absorbed whole by the division-year fixed effect. Exposure times
the national growth rate is the form that survives, and it is the Bartik shape.

The county set, the window start years, the income denominators and the state
codes all come from the certified candidate panels, so this producer changes the
contract quantity and nothing else.

Output: one parquet per (era, horizon, obligation rule) under
_data/temp/symmetric_growth_grid/, plus a support audit in batch_logs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

PROJECT_ROOT = CODE.parents[3]
CANDIDATE = PROJECT_ROOT / "Candidate for first stage" / "data" / "temp"
OUT_DIR = dc.MACRO_DATA_ROOT / "temp" / "symmetric_growth_grid"
AUDIT = dc.LOG_DIR / "symmetric_growth_grid_support_audit.csv"

FLOOR = 25_000.0
HORIZONS = (2, 3, 4)
RULES = ("net", "positive")
ERAS = {"pre": (1979, 1997), "post": (1998, 2018)}
SHARE_WINDOWS = {
    "pre": {"fixed4": (1975, 1978), "fixed5": (1974, 1978), "fixed10": (1969, 1978)},
    "post": {"fixed4": (1994, 1997), "fixed5": (1993, 1997), "fixed10": (1988, 1997)},
}
CONTRACT_FIRST = 1969
CONTRACT_LAST = 2018

DIVISION = {}
for _name, _states in (
        ("New England", "09 23 25 33 44 50"),
        ("Middle Atlantic", "34 36 42"),
        ("East North Central", "17 18 26 39 55"),
        ("West North Central", "19 20 27 29 31 38 46"),
        ("South Atlantic", "10 11 12 13 24 37 45 51 54"),
        ("East South Central", "01 21 28 47"),
        ("West South Central", "05 22 40 48"),
        ("Mountain", "04 08 16 30 32 35 49 56"),
        ("Pacific", "02 06 15 41 53")):
    for _s in _states.split():
        DIVISION[int(_s)] = _name


def county_actions(year, fips_to_ccg):
    """Signed nominal dollars and CCG for every county-classed action in `year`.

    Mirrors dc.build_contract_panel: geo_class == 'county', the thousands unit
    change at FY1989, AWARD rows only on the USAspending side of the seam.
    """
    if year < dc.SEAM_YEAR:
        df = pd.read_parquet(
            dc.OUT_DIR / f"fy{year}.parquet",
            columns=["dollars", "dollars_unit", "geo_class", "county_fips5"])
        df = df[df["geo_class"] == "county"]
        d = df["dollars"].to_numpy(np.float64)
        d = np.where((df["dollars_unit"] == "thousands").to_numpy(), d * 1000.0, d)
        ccg = df["county_fips5"].map(fips_to_ccg)
        ok = ccg.notna().to_numpy()
        return d[ok], ccg[ok].to_numpy(np.int64)
    df = pd.read_parquet(
        dc.USASPENDING_OUT_DIR / f"fy{year}.parquet",
        columns=["federal_action_obligation", "award_or_idv_flag", "geo_class", "ccg"])
    d = df["federal_action_obligation"].to_numpy(np.float64)
    ok = (df["award_or_idv_flag"].eq("AWARD")
          & df["geo_class"].eq("county")
          & df["ccg"].notna()).to_numpy()
    return d[ok], df.loc[ok, "ccg"].to_numpy(np.int64)


def contract_matrix(units, rule, quiet=False):
    """CCG x year real award dollars on the floor, rectangular and zero-filled."""
    fips_to_ccg, _, _ = dc.load_ccg_map()
    parts = []
    for year in range(CONTRACT_FIRST, CONTRACT_LAST + 1):
        d, ccg = county_actions(year, fips_to_ccg)
        keep = np.abs(d) >= FLOOR
        if rule == "positive":
            keep &= d > 0
        g = pd.DataFrame({"ccg": ccg[keep], "v": d[keep] * dc.deflate(year)})
        parts.append(g.groupby("ccg", sort=False)["v"].sum()
                     .rename("v").reset_index().assign(year=year))
        if not quiet:
            print(f"    {year}  {int(keep.sum()):>9,} actions  "
                  f"${g['v'].sum() / 1e9:>8,.1f}bn", flush=True)
    wide = (pd.concat(parts, ignore_index=True)
            .pivot(index="ccg", columns="year", values="v")
            .reindex(units).reindex(columns=range(CONTRACT_FIRST, CONTRACT_LAST + 1))
            .fillna(0.0))
    return wide


def window_starts(era, h):
    first, last = ERAS[era]
    return list(range(first, last - h + 1, h))


def population_matrix(units):
    """CCG x year BEA population, on the candidate county universe."""
    pop = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    wide = (pop.pivot(index="ccg", columns="year", values="population")
            .reindex(units))
    missing = int(wide.isna().to_numpy().sum())
    assert missing == 0, f"population is missing for {missing} county-years"
    return wide


def build(era, h, rule, contracts, population, audit_rows):
    """One (era, horizon, rule) panel: level and growth endogenous plus instruments."""
    cand = pd.read_stata(CANDIDATE / f"{era}_h{h}.dta",
                         columns=["unit", "state", "year", "income_real",
                                  "national_income", "employment",
                                  "national_employment"])
    cand["unit"] = cand["unit"].astype(np.int64)
    cand["year"] = cand["year"].astype(np.int64)
    starts = window_starts(era, h)
    assert sorted(cand["year"].unique()) == starts, \
        f"{era} h={h}: candidate start years {sorted(cand['year'].unique())} != {starts}"
    units = np.array(sorted(cand["unit"].unique()))

    C = contracts.reindex(units)
    G = pd.DataFrame(
        {t0: C.loc[:, list(range(t0 + 1, t0 + h + 1))].sum(axis=1) for t0 in starts},
        index=units)

    keep = (G > 0).all(axis=1).to_numpy()
    audit_rows.append(dict(era=era, horizon=h, rule=rule, windows=len(starts),
                           units_candidate=len(units), units_kept=int(keep.sum()),
                           pct_kept=round(100 * keep.mean(), 2),
                           pct_dollars_kept=round(
                               100 * G.to_numpy()[keep].sum() / G.to_numpy().sum(), 3)))
    units = units[keep]
    G = G.loc[units]

    lo, hi = SHARE_WINDOWS[era]["fixed4"][0], max(
        w[1] for w in SHARE_WINDOWS[era].values())
    assert lo >= CONTRACT_FIRST and hi <= CONTRACT_LAST

    rows = []
    nat_all = {t0: float(contracts.loc[:, list(range(t0 + 1, t0 + h + 1))].sum().sum())
               for t0 in starts}
    for share_label, (s0, s1) in SHARE_WINDOWS[era].items():
        cty_share_num = contracts.reindex(units).loc[:, list(range(s0, s1 + 1))] \
            .sum(axis=1).to_numpy()
        share = cty_share_num / float(contracts.loc[:, list(range(s0, s1 + 1))].sum().sum())
        for j, t0 in enumerate(starts):
            g_now = G[t0].to_numpy()
            nat_loo = nat_all[t0] - g_now          # leave the county out
            sub = cand[cand["year"] == t0].set_index("unit").reindex(units)
            y_c = sub["income_real"].to_numpy(np.float64)
            e_c = sub["employment"].to_numpy(np.float64)
            p_c = population.loc[units, t0].to_numpy(np.float64)
            y_nat = float(sub["national_income"].iloc[0])
            e_nat = float(sub["national_employment"].iloc[0])
            p_nat = float(population.loc[:, t0].sum())
            rec = dict(ccg=units, year=t0, window=j,
                       state=sub["state"].to_numpy(np.int64),
                       share_label=share_label, share=share,
                       income_real=y_c, employment=e_c, population=p_c,
                       G=g_now, nat_loo=nat_loo,
                       y_level_inc=g_now / y_c, z_level_inc=share * nat_loo / y_nat,
                       y_level_emp=g_now / e_c, z_level_emp=share * nat_loo / e_nat,
                       y_level_pop=g_now / p_c, z_level_pop=share * nat_loo / p_nat)
            if j == 0:
                for nm in ("y_std", "y_log", "y_mpgr", "z_std", "z_log", "z_mpgr"):
                    rec[nm] = np.full(len(units), np.nan)
            else:
                g_lag = G[starts[j - 1]].to_numpy()
                n_lag = nat_all[starts[j - 1]] - g_lag
                rec["y_std"] = (g_now - g_lag) / g_lag
                rec["y_log"] = np.log(g_now) - np.log(g_lag)
                rec["y_mpgr"] = 2.0 * (g_now - g_lag) / (g_now + g_lag)
                rec["z_std"] = share * ((nat_loo - n_lag) / n_lag)
                rec["z_log"] = share * (np.log(nat_loo) - np.log(n_lag))
                rec["z_mpgr"] = share * (2.0 * (nat_loo - n_lag) / (nat_loo + n_lag))
            rows.append(pd.DataFrame(rec))

    panel = pd.concat(rows, ignore_index=True)
    panel["division"] = panel["state"].map(DIVISION)
    assert panel["division"].notna().all(), "a state code carries no Census division"
    panel["era"] = era
    panel["horizon"] = h
    panel["rule"] = rule
    for c in ("y_level_inc", "z_level_inc", "y_level_emp", "z_level_emp",
              "y_level_pop", "z_level_pop", "y_std", "y_log", "y_mpgr",
              "z_std", "z_log", "z_mpgr"):
        bad = ~np.isfinite(panel[c].to_numpy()) & (panel["window"].to_numpy() > 0)
        assert not bad.any(), f"{era} h={h} {rule}: {int(bad.sum())} non-finite {c}"
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 78)
    print("  Symmetric pre/post first-stage panels, $25,000 nominal floor")
    print("=" * 78)

    cand_units = np.array(sorted(
        pd.read_stata(CANDIDATE / "pre_h4.dta", columns=["unit"])["unit"]
        .astype(np.int64).unique()))
    print(f"  candidate county universe: {len(cand_units):,}")

    audit_rows = []
    for rule in RULES:
        print(f"\n[contracts] award flow, {rule} obligations, floor ${FLOOR:,.0f} nominal")
        C = contract_matrix(cand_units, rule, quiet=args.quiet)
        P = population_matrix(cand_units)
        for era in ERAS:
            for h in HORIZONS:
                panel = build(era, h, rule, C, P, audit_rows)
                path = OUT_DIR / f"{era}_h{h}_{rule}.parquet"
                panel.to_parquet(path, index=False)
                a = audit_rows[-1]
                print(f"  {era:>4} h={h} {rule:>8}  "
                      f"{a['units_kept']:>5,} of {a['units_candidate']:,} counties "
                      f"({a['pct_kept']:>4.1f}%), {a['pct_dollars_kept']:>6.2f}% of "
                      f"window dollars  ->  {path.name}")

    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)
    print(f"\n  wrote {AUDIT}")
    print(f"  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
