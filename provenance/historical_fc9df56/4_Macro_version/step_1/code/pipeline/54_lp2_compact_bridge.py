"""Bridge the locked compact h=2 coefficient to the direct-shock LP impact.

Every row uses the persisted compact h=2 composition instrument, two-year log
employment growth, and division-by-year fixed effects with equal county
weights. The specifications sequentially introduce the LP sample and controls.

Writes Results/lp_2yr/lp2_compact_bridge.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import oldpath as op  # noqa: E402
import store  # noqa: E402
from clusterstats import wls_fe  # noqa: E402


OUT = ROOT / "Results" / "lp_2yr" / "lp2_compact_bridge.csv"
COMPACT_RESULTS = ROOT / "step_2" / "result" / \
    "coeff_table_decomposition_baseline.csv"
LP_RESULTS = ROOT / "Results" / "lp_2yr" / "lp2_dynamics.csv"
H = 2
FULL_LO, FULL_HI = 1980, locked.TARGET_LAST
LP_LO, LP_HI = 1986, 2012
SHOCK_LAGS = (2, 4, 6)
OUTCOME_LAGS = (2, 3, 4)
SCALE = locked.SCALE


def _columns(pivot, years):
    """Return a float matrix after checking that every requested year exists."""
    missing = sorted(set(years) - set(pivot.columns))
    if missing:
        raise SystemExit(f"FAIL: source panel lacks years {missing}")
    return pivot.reindex(columns=years).to_numpy(np.float64)


def load_frame():
    """Construct the compact frame and the variables needed by the LP impact."""
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E"))
    division = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    target = store.target_years(H)
    zwide = store.load_instrument(H, units, "z")
    zpos = {year: j for j, year in enumerate(target)}

    years = list(range(FULL_LO, FULL_HI + 1))
    z = zwide[:, [zpos[t] for t in years]]
    e_t = _columns(piv["E"], years)
    e_tm2 = _columns(piv["E"], [t - H for t in years])
    log_growth = np.log(e_t) - np.log(e_tm2)

    ok = (np.isfinite(z) & np.isfinite(log_growth)
          & (e_t > 0) & (e_tm2 > 0))
    ri, ci = np.where(ok)
    year_array = np.asarray(years)
    d = pd.DataFrame({
        "ccg": units[ri],
        "year": year_array[ci],
        "z": z[ok],
        "y_log": log_growth[ok],
    })
    d["division"] = d["ccg"].map(division)
    if d["division"].isna().any():
        raise SystemExit("FAIL: division is missing on the compact frame")
    if len(d) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: compact frame has {len(d):,} rows, expected "
            f"{locked.N_COUNTY[H]:,}")

    common = d["year"].between(LP_LO, LP_HI)
    if int(common.sum()) != locked.N_COUNTIES * (LP_HI - LP_LO + 1):
        raise SystemExit("FAIL: LP rectangle is not balanced")

    common_years = list(range(LP_LO, LP_HI + 1))
    common_z = zwide[:, [zpos[t] for t in common_years]]
    common_pbase = _columns(piv["P"], [t - H for t in common_years])
    units2, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")

    s0 = np.full_like(common_z, np.nan)
    for j, t in enumerate(common_years):
        hi = t - H - locked.LAG
        lo = hi - locked.SPAN + 1
        a, b = contract_years.index(lo), contract_years.index(hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        s0[:, j] = exposure.sum(axis=1) / common_pbase[:, j]

    common_data = {
        "s0": s0,
        **{
            f"zlag{lag}": zwide[:, [zpos[t - lag] for t in common_years]]
            for lag in SHOCK_LAGS
        },
        **{
            f"gl{j}": (
                np.log(_columns(piv["E"],
                                [t - H - (j - 1) for t in common_years]))
                - np.log(_columns(piv["E"],
                                  [t - H - j for t in common_years])))
            for j in range(1, len(OUTCOME_LAGS) + 1)
        },
    }
    dc = d.loc[common].copy()
    for name, values in common_data.items():
        dc[name] = values.reshape(-1)
    if not np.isfinite(dc[list(common_data)]).all().all():
        raise SystemExit("FAIL: the LP controls contain missing values")
    return d, dc


def fit(d, outcome, controls):
    """Run one county regression with the locked weighting and fixed effects."""
    variables = ["z", *controls]
    w = np.ones(len(d), np.float64)
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cluster = pd.factorize(d["ccg"])[0].astype(np.int64)
    b, se, _ = wls_fe(
        d[outcome].to_numpy(np.float64),
        d[variables].to_numpy(np.float64), w, fe, nofe, cluster)
    return float(b[0]), float(se[0])


def reference_values():
    """Read the two independently produced coefficients used as tie-outs."""
    compact = pd.read_csv(COMPACT_RESULTS)
    row = compact[(compact["h"] == H) & (compact["rung"] == "naics2")
                  & (compact["source"] == "efsy")]
    if len(row) != 1:
        raise SystemExit("FAIL: compact h=2 reference row is not unique")
    lp = pd.read_csv(LP_RESULTS)
    impact = lp[(lp["variant"] == "three_lags")
                & (lp["event_time"] == 0)]
    if len(impact) != 1:
        raise SystemExit("FAIL: direct-shock LP impact row is not unique")
    return float(row.iloc[0]["county"]) / SCALE, \
        float(impact.iloc[0]["coef"]) / SCALE


def main():
    start = time.time()
    print("== stage 54: compact-to-LP h=2 bridge")
    d_full, d_lp = load_frame()
    specs = [
        ("compact", "Compact h=2", d_full, "y_log", ()),
        ("lp_sample", "Restrict to LP sample", d_lp, "y_log", ()),
        ("share_sum", "Add share-sum control", d_lp, "y_log", ("s0",)),
        ("shock_lags", "Add three past h=2 shocks", d_lp, "y_log",
         ("s0", "zlag2", "zlag4", "zlag6")),
        ("outcome_lags", "Add three annual outcome lags", d_lp, "y_log",
         ("s0", "zlag2", "zlag4", "zlag6", "gl1", "gl2", "gl3")),
    ]
    rows = []
    prior = np.nan
    for order, (name, label, data, outcome, controls) in enumerate(specs):
        beta, se = fit(data, outcome, controls)
        rows.append({
            "order": order,
            "specification": name,
            "label": label,
            "outcome": "two-year log employment growth",
            "sample_start": int(data["year"].min()),
            "sample_end": int(data["year"].max()),
            "controls": ",".join(controls) if controls else "none",
            "coefficient_x1e5": beta * SCALE,
            "se_county_x1e5": se * SCALE,
            "change_from_previous_x1e5":
                (beta - prior) * SCALE if np.isfinite(prior) else np.nan,
            "n": len(data),
            "n_counties": int(data["ccg"].nunique()),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
            "shock": "persisted compact h=2 composition instrument",
        })
        prior = beta
        print(f"   {label:<34s} beta={beta * SCALE:+.6f} "
              f"se={se * SCALE:.6f} N={len(data):,}")

    compact_ref, lp_ref = reference_values()
    compact_gap = abs(rows[0]["coefficient_x1e5"] - compact_ref * SCALE)
    lp_gap = abs(rows[-1]["coefficient_x1e5"] - lp_ref * SCALE)
    if compact_gap >= locked.RTOL_COEF * max(abs(compact_ref * SCALE), 1e-300):
        raise SystemExit(
            f"FAIL: compact tie-out gap is {compact_gap:.3e} in reported units")
    if lp_gap >= locked.RTOL_COEF * max(abs(lp_ref * SCALE), 1e-300):
        raise SystemExit(
            f"FAIL: LP impact tie-out gap is {lp_gap:.3e} in reported units")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"   compact tie-out gap: {compact_gap:.3e}")
    print(f"   LP impact tie-out gap: {lp_gap:.3e}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
