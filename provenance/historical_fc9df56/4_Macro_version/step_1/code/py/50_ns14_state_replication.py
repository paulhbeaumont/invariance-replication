#!/usr/bin/env python3
"""Nakamura and Steinsson (2014) at their own rung, on our contract data.

Before anything is claimed one rung down, the design has to reproduce their
result where they ran it. Their spending source IS ours: "the electronic
database of DD-350 military procurement forms ... compiled ... by state and year
for 1966-2006", using "the location where the majority of the work was
performed" (p. 757-758). So this is a replication on shared source data, not an
analogue.

THEIR SPECIFICATION, equation (1) p. 759 and the Table 2 note p. 763:

    (Y_it - Y_it-2)/Y_it-2 = a_i + g_t + b (G_it - G_it-2)/Y_it-2 + e_it

    outcome     output    (Y_it - Y_it-2)/Y_it-2
                employment (L_it - L_it-2)/L_it-2, L = employment / population
    regressor   the two-year change in spending divided by LAGGED OUTPUT
    per capita  every variable, spending included
    effects     unit and year
    cluster     the unit, which is how they handle the overlapping windows
    instrument  Bartik: national spending scaled by "the average level of
                spending in that state relative to state output in the first
                five years of our sample" (p. 762)

    their result, Table 3 cols 1-2, the Bartik columns
                    output      2.48 (0.94) states, 2.75 (0.69) regions
                    employment  1.81 (0.41) states, 2.51 (0.31) regions

Their Bartik share window is 1966-1971 (Table 3 note); ours is 1969-1973, the
first five years of a sample that starts where BEA county income does. Rescaling
a single instrument cannot move a just-identified beta, so the share's
normalisation is immaterial; its window is not. Their employment is CES and ours
is BEA.

FOUR DECLARED DEVIATIONS, none of them optional given what this project holds.

1. OUTPUT is BEA personal income, not BEA GDP-by-state. We never acquired
   SAGDP. Personal income runs about four fifths of GDP, so the output-equation
   coefficient is expected LOW against their 1.43 by roughly that factor, and
   the regressor's denominator carries the same substitution.
2. EMPLOYMENT is the BEA county measure summed to the state, not the BLS CES
   payroll survey. NS report both and say the BEA series "is available since
   1969"; ours ends in 2000, so the employment sample is 1971-2000 against
   their 1966-2006.
3. THE SAMPLE STARTS IN 1969, because BEA county income does. At h = 2 the
   first estimable year is 1971. They start in 1966.
4. REGIONS are the nine Census divisions. NS use ten, splitting South Atlantic
   "because of its large size" (p. 759 n. 20). Not replicated.

No floor is imposed on the contracts, because NS impose none either: they simply
note the source's own thresholds, "greater than $10,000 up to 1983 and greater
than $25,000 thereafter" (p. 758).

Output:
    batch_logs/macro_version/ns14_state_replication.csv
    batch_logs/macro_version/50_ns14_state_replication.log
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
OUT = dc.LOG_DIR / "ns14_state_replication.csv"

HORIZON = 2
SHARE_YEARS = (1969, 1973)      # "the first five years of our sample", ours starts 1969
FIRST_YEAR = 1969


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def state_panel():
    """One row per (state, year): contracts, income, population, employment."""
    b14 = _load("b14", CODE / "14_build_instrument.py")
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"]
    st = (xw.assign(ccg=xw["ccg_id"].astype(int),
                    state=xw["county_fips5"].str[:2].astype(int))
          .drop_duplicates("ccg")[["ccg", "state"]])

    con = pd.read_parquet(dc.PANEL_OUT_DIR / "contracts_harmonized_ccg_year.parquet",
                          columns=["ccg", "year", "dollars_real"])
    inc = pd.read_parquet(dc.PANEL_OUT_DIR / "income_ccg_year.parquet",
                          columns=["ccg", "year", "income_real"])
    pop = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    bea = pd.read_parquet(dc.PANEL_OUT_DIR / "bea_employment_ccg_year.parquet")

    d = (inc.merge(pop, on=["ccg", "year"], how="left")
            .merge(bea, on=["ccg", "year"], how="left")
            .merge(con.groupby(["ccg", "year"], as_index=False)["dollars_real"].sum(),
                   on=["ccg", "year"], how="left")
            .merge(st, on="ccg", how="left"))
    d["dollars_real"] = d["dollars_real"].fillna(0.0)
    d = d[d["state"].notna() & d["state"].isin(b14.DIVISION)]
    d["division"] = d["state"].map(b14.DIVISION)

    out = {"county": d.rename(columns={"ccg": "unit", "income_real": "Y",
                                       "population": "P", "bea_emp": "E",
                                       "dollars_real": "G"})[
        ["unit", "state", "division", "year", "G", "Y", "P", "E"]]}
    out["county"] = out["county"][out["county"]["year"] >= FIRST_YEAR].reset_index(drop=True)
    for lab, key in (("state", "state"), ("region", "division")):
        g = (d.groupby([key, "year"], as_index=False)
               .agg(G=("dollars_real", "sum"), Y=("income_real", "sum"),
                    P=("population", "sum"), E=("bea_emp", "sum")))
        g = g.rename(columns={key: "unit"})
        g["state"] = g["unit"] if key == "state" else np.nan
        out[lab] = g[g["year"] >= FIRST_YEAR].reset_index(drop=True)
    return out


def build(g, outcome, delta="flow"):
    """NS's left and right sides, per capita, on lagged output."""
    g = g.sort_values(["unit", "year"]).copy()
    if outcome == "employment":
        g = g[g["E"] > 0]
    g["g_pc"] = g["G"] / g["P"]
    g["y_pc"] = g["Y"] / g["P"]
    g["L"] = g["E"] / g["P"]
    for c in ("g_pc", "y_pc", "L"):
        g[c + "_lag"] = g.groupby("unit")[c].shift(HORIZON)
    # the window's own years, for the cumulative treatment: t-h+1 .. t
    g["g_pc_win"] = sum(g.groupby("unit")["g_pc"].shift(k) for k in range(HORIZON))
    g["lag_year"] = g.groupby("unit")["year"].shift(HORIZON)
    # The national series is built from the panel BEFORE this filter. The filter
    # keeps only rows whose own t-h exists, so it removes the first h years, and a
    # national aggregate built on what survives cannot look up its own t-h: the
    # shift returns missing and the first h years of every unit drop with it. The
    # spending that lag needs is in the panel either way, so the aggregate is taken
    # from the unfiltered frame and only the estimation rows are filtered.
    g_full = g.copy()
    g = g[g["lag_year"] == g["year"] - HORIZON]
    # flow        NS:   G_t - G_t-2
    # win_less_1   yours: (G_t + G_t-1) - G_t-2
    # win_less_h   balanced: (G_t + G_t-1) - h x G_t-2, the window's total against
    #              what the base year's rate would have delivered over the window
    # cumulative   the level: G_t + G_t-1, no base subtracted at all
    NUM = {"flow": g["g_pc"] - g["g_pc_lag"],
           "win_less_1": g["g_pc_win"] - g["g_pc_lag"],
           "win_less_h": g["g_pc_win"] - HORIZON * g["g_pc_lag"],
           "cumulative": g["g_pc_win"]}
    g["x"] = NUM[delta] / g["y_pc_lag"]
    g["y"] = ((g["L"] - g["L_lag"]) / g["L_lag"] if outcome == "employment"
              else (g["y_pc"] - g["y_pc_lag"]) / g["y_pc_lag"])

    # Bartik: the unit's average spending relative to its output over the first
    # five years, times the national spending change on lagged national output.
    base = g[(g["year"] >= SHARE_YEARS[0]) & (g["year"] <= SHARE_YEARS[1])]
    s = (base.groupby("unit").apply(lambda t: (t["G"] / t["Y"]).mean(),
                                    include_groups=False).rename("s"))
    nat = (g_full.groupby("year").agg(G=("G", "sum"), Y=("Y", "sum"), P=("P", "sum")))
    nat["g_pc"] = nat["G"] / nat["P"]
    nat["y_pc"] = nat["Y"] / nat["P"]
    nat["g_pc_win"] = sum(nat["g_pc"].shift(k) for k in range(HORIZON))
    NNUM = {"flow": nat["g_pc"] - nat["g_pc"].shift(HORIZON),
            "win_less_1": nat["g_pc_win"] - nat["g_pc"].shift(HORIZON),
            "win_less_h": nat["g_pc_win"] - HORIZON * nat["g_pc"].shift(HORIZON),
            "cumulative": nat["g_pc_win"]}
    nat["shift"] = NNUM[delta] / nat["y_pc"].shift(HORIZON)
    g = g.merge(s, on="unit", how="left")
    g["z"] = g["s"] * g["year"].map(nat["shift"])
    g = g[np.isfinite(g[["x", "y", "z"]].to_numpy()).all(axis=1)].reset_index(drop=True)
    g["delta"] = delta
    # THE AGGREGATING WEIGHT IS THE LAGGED LEVEL IN THE GROWTH RATE'S DENOMINATOR.
    # Standard growth aggregates EXACTLY under it:
    #   sum_c [(Y_ct - Y_c,t-2)/Y_c,t-2] x [Y_c,t-2 / Y_s,t-2] = (Y_st - Y_s,t-2)/Y_s,t-2
    # so the output equation takes income shares and the employment equation takes
    # employment shares. Using one for the other breaks the identity.
    g["E_lag"] = g["L_lag"] * g["P"]
    g["Y_lag"] = g["y_pc_lag"] * g["P"]
    g["base"] = g["E_lag"] if outcome == "employment" else g["Y_lag"]
    if g["state"].notna().all():
        tot = g.groupby(["state", "year"])["base"].transform("sum")
        g["w_state_equal"] = np.where(tot > 0, g["base"] / tot, np.nan)
    else:
        g["w_state_equal"] = np.nan
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--last-year", type=int, default=2006)
    ap.add_argument("--rung", nargs="+", default=["region", "state", "county"])
    ap.add_argument("--delta", nargs="+",
                    default=["flow", "win_less_1", "win_less_h", "cumulative"],
                    choices=["flow", "win_less_1", "win_less_h", "cumulative"])
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 78)
    print("  Nakamura and Steinsson (2014) replicated at the state and region rung")
    print("=" * 78)
    print("  NS report  output 2.48 (0.94) states, 2.75 (0.69) regions")
    print("             employment 1.81 (0.41) states, 2.51 (0.31) regions")
    print("  ours uses BEA personal income for output and BEA employment, see header\n")

    panels = state_panel()
    rows = []
    print(f"{'rung':>8}{'treatment':>13}{'outcome':>12}{'fixed effects':>15}"
          f"{'weight':>20}{'years':>12}{'units':>7}{'N':>8}"
          f"{'beta':>10}{'se':>9}{'t':>7}{'FS F':>9}")
    for rung in args.rung:
      for delta in args.delta:
        for outcome in ("output", "employment"):
            g0 = build(panels[rung], outcome, delta)
            g0 = g0[g0["year"] <= args.last_year]
            if g0.empty:
                continue
            wcols = [(None, "none")] if rung != "county" else [
                (None, "none"), ("base", "size (matched base)"),
                ("w_state_equal", "state-EW (matched base)")]
            fes = [("unit", "unit + year")] if rung != "county" else [
                ("unit", "county + year"), ("state", "state + year")]
            for fecol, felab in fes:
              for wcol, wlab in wcols:
                g = g0 if wcol is None else g0[np.isfinite(g0[wcol].to_numpy())
                                               & (g0[wcol].to_numpy() > 0)]
                w = np.ones(len(g)) if wcol is None else g[wcol].to_numpy(float)
                w = w / w.mean()
                f1 = pd.factorize(g[fecol])[0].astype(np.int64)
                f2 = pd.factorize(g["year"])[0].astype(np.int64)
                cl = pd.factorize(g["unit"])[0].astype(np.int64)
                r = t28.fit2sls(g["y"].to_numpy(float), g["x"].to_numpy(float),
                                g["z"].to_numpy(float), w, f1, f2,
                                cl, singleton=False)
                span = f"{int(g['year'].min())}-{int(g['year'].max())}"
                print(f"{rung:>8}{delta:>13}{outcome:>12}{felab:>15}{wlab:>20}{span:>12}"
                      f"{g['unit'].nunique():>7}{len(g):>8,}{r['beta']:>10.3f}"
                      f"{r['se']:>9.3f}{r['beta'] / r['se']:>7.2f}{r['F']:>9.2f}")
                rows.append(dict(rung=rung, delta=delta, outcome=outcome,
                                 fe=felab, weight=wlab,
                                 first_year=int(g["year"].min()),
                                 last_year=int(g["year"].max()),
                                 n_units=g["unit"].nunique(), n_obs=len(g),
                                 beta=r["beta"], se=r["se"],
                                 t=r["beta"] / r["se"], first_stage_F=r["F"]))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}")
    print(f"  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
