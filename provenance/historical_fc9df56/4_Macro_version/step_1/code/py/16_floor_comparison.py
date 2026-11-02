#!/usr/bin/env python3
"""What removing the $25,000 floor cost each instrument. Part 1.5, after 1.7.2.

The floor came off on 2026-08-10 and step_1 was rebuilt without it. 14 reports the
unfloored first stage and 15 the unfloored Cragg-Donald, but neither can say what
the CHANGE did, because each one knows only the floor it was run at. This file
runs both floors and reports the pair, for all three exposures rather than for the
headline one.

IT WRITES NOTHING THAT ANY OTHER SCRIPT READS.

That is the whole reason it exists as a separate file. 14 writes
instrument_ccg_year.parquet and instrument_first_stage.csv at fixed paths, so
`14 --floor 25000` silently replaces the production panel with a floored one -
open item 9 of the README. Getting the comparison by running 14 twice therefore
means backing up and restoring by hand, and a forgotten restore leaves the project
running on a floored panel that every table claims is unfloored. Here both floors
are built in memory and only a comparison CSV is written.

BOTH STATISTICS, BECAUSE THEY DISAGREE ABOUT THE LADDER

    clustered F   Kleibergen-Paap rk Wald, clustered on the unit. What plan 1.5
                  asks for and what decides whether a rung is usable.
    CD F          Cragg-Donald, homoskedastic. Comparable to NS14 footnote 30,
                  and the only statistic their published numbers can be laid
                  beside. README 1.8.5 explains why the two move in opposite
                  directions up the ladder.

Reporting one alone would say the floor change helped or hurt depending on which
one was picked, which is exactly the trap 1.8.5 was written to close.

Output:
    dc.LOG_DIR/floor_comparison.csv
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


# 14 owns the build, 15 owns the Cragg-Donald machinery. Import both rather than
# restate either, so this file cannot drift into measuring something else.
bi = _load("build_instrument", "14_build_instrument.py")
cf = _load("ns14_compare", "15_ns14_f_comparison.py")

FLOORS = (0, 25_000)
RUNGS = (("county", "ccg"), ("commuting zone", "cz"),
         ("state", "state"), ("Census division", "division"))

# This file stops at FY2006, and it is the one file here that SHOULD.
#
# A floor is a threshold on an individual contract ACTION, and only the DCADS
# years are action-level. The post-seam block arrives already aggregated to the
# county-year, so there is nothing there to apply a floor to and comparing
# floor 0 against floor $25,000 over 1975-2018 would compare two windows that
# are identical after 2006 - which would dilute the very contrast this exhibit
# exists to show. bi.LAST_YEAR moved to 2018 with the splice; this does not.
LAST_YEAR = 2006

# Likewise only the exposures this file actually BUILDS below. bi.EXPOSURES
# gained fixed_9398, which belongs to the post-1998 window and therefore to a
# sample this exhibit does not cover; defaulting to it produced a KeyError on
# z_fixed_9398 rather than a wrong number, but stating the set is better than
# relying on that.
EXPOSURES = ("fixed", "rolling")


def build_at(floor, horizons, names, emp, universe):
    """Every horizon's instrument frame at one floor, plus the pre-window sums.

    A transcription of 14's main() up to the point where it writes, and nothing
    after it. The pre-window sums are carried because bi.collapse needs them.
    """
    levels, _, _ = bi.build_levels(floor, True, bi.EXPOSURE_FIRST, LAST_YEAR)
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)

    fixed = bi.fixed_exposure(levels, bi.PRE_WINDOW)
    pre = bi.fixed_window_sums(levels, bi.PRE_WINDOW)

    frames = []
    for h in horizons:
        exposures = {}
        if "fixed" in names:
            exposures["fixed"] = fixed
        if "rolling" in names:
            exposures["rolling"] = bi.rolling_exposure(
                levels, h, bi.ROLL_K, bi.ROLL_MIN).to_numpy()
        frames.append(bi.build_horizon(levels, exposures, h,
                                       bi.FIRST_YEAR, LAST_YEAR, True))
    out = pd.concat(frames, ignore_index=True)
    return out.merge(pre, on="ccg", how="left"), levels


def sweep(county_all, rungs, names, horizons, floor):
    """Clustered F, Cragg-Donald F and the 2SLS point estimate at every rung."""
    rows = []
    for h in horizons:
        for name in names:
            base = county_all[county_all["h"] == h].copy()
            base["z"] = base[f"z_{name}"]
            base = base.merge(rungs, on="ccg", how="left")
            keep = np.isfinite(base[["shock", "emp_growth", "z", "income_real_lag",
                                     "emp_lag"]].to_numpy()).all(axis=1)
            base = base[keep & (base["emp_lag"] > 0) & (base["income_real_lag"] > 0)]

            for rung, unit_col in RUNGS:
                if rung == "county":
                    d = base.rename(columns={"emp_growth": "y", "shock": "x"}).copy()
                    d["unit"] = d["ccg"]
                    d = d[["unit", "year", "y", "x", "z"]]
                else:
                    # Employment weights: the invariance version of README 3.3,
                    # which is what Table I reports and what Part 3 will run.
                    d = bi.collapse(base, unit_col, "emp_lag")[
                        ["unit", "year", "y", "x", "z"]]
                d = d[np.isfinite(d[["x", "y", "z"]].to_numpy()).all(axis=1)]

                clustered = bi.fit_rung(d, "unit + year")
                codes, dffe = cf.codes_for(d, ("unit", "year"))
                cd = cf.f_stat(d["x"].to_numpy(), d[["z"]].to_numpy(), codes, dffe)
                rows.append(dict(
                    floor=floor, rung=rung, h=h, exposure=name,
                    n_units=int(d["unit"].nunique()), n_obs=int(len(d)),
                    clustered_F=clustered["fs_F"], cd_F=cd["cd_F"],
                    partial_R2=cd["partial_R2"],
                    beta=clustered["beta"], beta_se=clustered["beta_se"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--exposure", nargs="+", default=list(EXPOSURES),
                    choices=list(EXPOSURES))
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    names = [e for e in EXPOSURES if e in set(args.exposure)]
    print(f"\nwhat the floor change cost each instrument"
          f"   floors {[f'${f:,}' if f else 'none' for f in FLOORS]}, "
          f"h={horizons}, exposures {names}")

    _, _, universe = dc.load_analysis_ccg_map()
    rungs = bi.load_rungs(universe)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])

    parts, totals = [], {}
    for floor in FLOORS:
        label = "none" if floor == 0 else f"${floor:,}"
        print(f"\n[{FLOORS.index(floor)}] building at floor {label}")
        county_all, levels = build_at(floor, horizons, names, emp, universe)
        totals[floor] = float(levels.loc[
            levels["year"].between(bi.FIRST_YEAR, LAST_YEAR), "dollars_real"].sum())
        print(f"    ${totals[floor] / 1e9:,.1f}bn of 2006 dollars, "
              f"{len(county_all):,} rows")
        parts.append(sweep(county_all, rungs, names, horizons, floor))

    s = pd.concat(parts, ignore_index=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(dc.LOG_DIR / "floor_comparison.csv", index=False)
    print(f"\n[2] {len(s):,} fits -> {dc.LOG_DIR / 'floor_comparison.csv'}")
    d0, d25 = totals[0], totals[25_000]
    print(f"    removing the floor adds ${(d0 - d25) / 1e9:,.1f}bn, "
          f"{(d0 / d25 - 1) * 100:.2f}% more dollars")

    # Indexed lookup on the long frame rather than a pivot: pivot_table returns
    # MultiIndex columns whose reset_index collides with the index names, and the
    # collision turns a scalar cell into a Series only at format time.
    at = s.set_index(["floor", "h", "rung", "exposure"]).sort_index()
    for h in horizons:
        print(f"\n[3] h={h}   floored $25,000  ->  unfloored")
        print(f"    {'rung':<17s}{'exposure':<9s}"
              f"{'clustered F':>22s}{'Cragg-Donald F':>24s}{'2SLS beta':>22s}")
        for rung, _ in RUNGS:
            for name in names:
                lo = at.loc[(25_000, h, rung, name)]
                hi = at.loc[(0, h, rung, name)]

                def pair(col, fmt):
                    return f"{lo[col]:{fmt}} -> {hi[col]:{fmt}}"
                print(f"    {rung:<17s}{name:<9s}"
                      f"{pair('clustered_F', '8.1f'):>22s}"
                      f"{pair('cd_F', '9.1f'):>24s}"
                      f"{pair('beta', '+8.3f'):>22s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
