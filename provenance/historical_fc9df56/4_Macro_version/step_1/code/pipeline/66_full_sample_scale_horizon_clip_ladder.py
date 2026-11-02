"""Estimate the full-sample-scale instrument across horizons and clip values.

For category k and horizon h, the national shifter is

    clip((G_k,t - G_k,t-h) / mean_s(abs(G_k,s)), -a, a).

County exposures use the locked five-year window, the county treatment uses
the locked average-window contract change, and the first stage uses
division-by-year fixed effects, equal county weights, and county clustering.

Writes Results/instrument_normalization/
    full_sample_scale_horizon_clip_ladder.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-batch-approved: explicit
# mpgr-new-estimates: 15
# mpgr-variation-scope: four-character|full-sample-scale|h1-h4|clip1-clip4

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
STEP1_PY = PIPELINE.parent / "py"
STEP2_PY = ROOT / "step_2" / "code" / "py"
OUT = (ROOT / "Results" / "instrument_normalization"
       / "full_sample_scale_horizon_clip_ladder.csv")
SAVED_H2 = (ROOT / "Results" / "instrument_normalization"
            / "h2_county_first_stage_sequence.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import instrument  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


HORIZONS = (1, 2, 3, 4)
CAPS = (1.0, 2.0, 3.0, 4.0)
N_NEW_EXPECTED = len(HORIZONS) * len(CAPS) - 1


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_h2_clip1():
    """Load the persisted h=2, clip-1 first stage."""
    saved = pd.read_csv(SAVED_H2)
    row = saved.loc[saved["instrument"].eq("A_current")]
    if len(row) != 1:
        raise SystemExit("FAIL: the h=2 sequence lacks one A_current row")
    row = row.iloc[0]
    if int(row["n"]) != locked.N_COUNTY[2]:
        raise SystemExit(
            f"FAIL: saved h=2, clip-1 N={int(row['n']):,}, expected "
            f"{locked.N_COUNTY[2]:,}")
    return {
        "first_stage": float(row["first_stage"]),
        "first_stage_se": float(row["first_stage_se"]),
        "first_stage_t": float(row["first_stage_t"]),
        "first_stage_F": float(row["first_stage_F"]),
        "partial_r2": float(row["partial_r2"]),
        "instrument_sd": float(row["instrument_sd"]),
    }


def estimation_arrays(baseline):
    """Build arrays for the locked county first-stage specification."""
    x = baseline["x"].to_numpy(np.float64)
    weight = np.ones(len(baseline), np.float64)
    fe = pd.factorize(
        baseline["division"].astype(str) + "_" + baseline["year"].astype(str)
    )[0].astype(np.int64)
    nofe = np.zeros(len(baseline), np.int64)
    cluster = pd.factorize(baseline["ccg"])[0].astype(np.int64)
    return x, weight, fe, nofe, cluster


def estimate_first_stage(x, z, weight, fe, nofe, cluster, t28):
    """Estimate one county-clustered first stage and partial R-squared."""
    fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster, singleton=False)
    demeaned = t28.w22.demean(
        np.column_stack([x, z]), fe, nofe, weight)
    xt, zt = demeaned[:, 0], demeaned[:, 1]
    partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
    tstat = float(fit["beta"] / fit["se"])
    return {
        "first_stage": float(fit["beta"]),
        "first_stage_se": float(fit["se"]),
        "first_stage_t": tstat,
        "first_stage_F": tstat ** 2,
        "partial_r2": partial_r2,
        "instrument_sd": float(np.std(z, ddof=1)),
        "within_instrument_sd": float(np.std(zt, ddof=1)),
    }


def build_instruments(arr, nat, contract_years, population, h, years):
    """Construct county instruments for every requested clipping value."""
    shifters = {
        cap: instrument.category_rates(nat, h, clip=cap) for cap in CAPS
    }
    instruments = {
        cap: np.full((arr.shape[0], len(years)), np.nan, np.float64)
        for cap in CAPS
    }
    n_contributing = {}
    for cap in CAPS:
        dollars, counts = instrument.predicted_dollars(
            arr, shifters[cap], contract_years, years, h, "roll",
            locked.SPAN, locked.LAG, locked.DATA_FIRST)
        filled = np.where(np.isnan(dollars) & (counts[None, :] > 0),
                          0.0, dollars)
        pbase = population.reindex(columns=[t - h for t in years]).to_numpy(
            np.float64)
        instruments[cap] = filled / pbase
        n_contributing[cap] = counts
    return shifters, instruments, n_contributing


def main():
    """Run the horizon-by-clip first-stage ladder."""
    np.random.seed(0)
    start = time.time()
    print("== full-sample-scale horizon-by-clip first-stage ladder")

    p57 = load_module("p57_full_sample_ladder",
                      STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_full_sample_ladder",
                      STEP2_PY / "28_table_d1.py")

    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=("P",))
    tensor_units, categories, contract_years, arr, nat = store.load_tensor(
        locked.RULE)
    if not np.array_equal(units, tensor_units):
        raise SystemExit("FAIL: county order differs between persisted stores")
    if len(categories) != locked.NCAT:
        raise SystemExit(
            f"FAIL: category store has {len(categories)} categories, expected "
            f"{locked.NCAT}")

    loaded = saved_h2_clip1()
    rows = []
    n_new = 0
    for h in HORIZONS:
        baseline, dropped = p57.build_county(
            "composition", h=h, span=locked.SPAN, rule=locked.RULE,
            floor=locked.FLOOR, denom=locked.DENOM)
        if dropped != 0:
            raise SystemExit(
                f"FAIL: h={h} baseline dropped {dropped} additional counties")
        expected_n = (locked.N_COUNTY_UNGATED[h]
                      if h in locked.EXTRA_HORIZONS else locked.N_COUNTY[h])
        if len(baseline) != expected_n:
            raise SystemExit(
                f"FAIL: h={h} baseline has {len(baseline):,} rows, expected "
                f"{expected_n:,}")

        years = store.target_years(h)
        shifters, wide, contributing = build_instruments(
            arr, nat, contract_years, pivots["P"], h, years)
        rowpos = {unit: i for i, unit in enumerate(units)}
        colpos = {year: j for j, year in enumerate(years)}
        ri = baseline["ccg"].map(rowpos).to_numpy()
        ci = baseline["year"].map(colpos).to_numpy()
        if pd.isna(ri).any() or pd.isna(ci).any():
            raise SystemExit(f"FAIL: h={h} baseline does not align to stores")
        ri, ci = ri.astype(int), ci.astype(int)
        zsample = {cap: matrix[ri, ci] for cap, matrix in wide.items()}
        if any(not np.isfinite(z).all() for z in zsample.values()):
            raise SystemExit(f"FAIL: h={h} has a missing county instrument")
        if not np.allclose(zsample[1.0], baseline["z"].to_numpy(np.float64),
                           rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC):
            gap = np.max(np.abs(
                zsample[1.0] - baseline["z"].to_numpy(np.float64)))
            raise SystemExit(
                f"FAIL: h={h}, clip-1 does not reproduce baseline z; "
                f"max gap {gap:.3e}")

        x, weight, fe, nofe, cluster = estimation_arrays(baseline)
        raw = instrument.category_rates(nat, h, clip=np.inf)
        sample_columns = [contract_years.index(t) for t in sorted(
            baseline["year"].unique())]
        raw_sample = raw[:, sample_columns]
        live = np.isfinite(raw_sample)
        total_live = int(live.sum())

        for cap in CAPS:
            z = zsample[cap]
            demeaned = t28.w22.demean(
                np.column_stack([x, z]), fe, nofe, weight)
            within_sd = float(np.std(demeaned[:, 1], ddof=1))
            if h == 2 and cap == 1.0:
                fit = {**loaded, "within_instrument_sd": within_sd}
                source = "loaded saved h=2 clip-1 result"
            else:
                fit = estimate_first_stage(
                    x, z, weight, fe, nofe, cluster, t28)
                source = "estimated in horizon-by-clip ladder"
                n_new += 1

            modified = int((live & (np.abs(raw_sample) > cap)).sum())
            rows.append({
                "horizon": h,
                "clip": cap,
                "horizon_status": (
                    "ungated exploratory" if h in locked.EXTRA_HORIZONS
                    else "locked horizon"),
                "category_level": "four-character",
                "category_universe": "persisted top-800 categories",
                "national_numerator": "G_k,t minus G_k,t-h",
                "national_denominator": (
                    "category mean absolute spending over 1966-2018"),
                "county_exposure": (
                    "five-year mean ending five years before base year"),
                "county_treatment": (
                    "average-window contract change per capita"),
                "sample_start": int(baseline["year"].min()),
                "sample_end": int(baseline["year"].max()),
                "n": len(baseline),
                "n_counties": int(baseline["ccg"].nunique()),
                "fixed_effects": "division-by-year",
                "weight": "equal county",
                "cluster": "county",
                **fit,
                "correlation_with_clip_1": float(np.corrcoef(
                    z, zsample[1.0])[0, 1]),
                "live_category_years": total_live,
                "clipped_category_years": modified,
                "clipped_share_of_live": modified / total_live,
                "min_contributing_categories": int(
                    contributing[cap][contributing[cap] > 0].min()),
                "max_contributing_categories": int(contributing[cap].max()),
                "estimation_source": source,
            })

        print(f"   h={h}: N={len(baseline):,}, years "
              f"{baseline['year'].min()}-{baseline['year'].max()}")

    if n_new != N_NEW_EXPECTED:
        raise SystemExit(
            f"FAIL: estimated {n_new} new rows, expected {N_NEW_EXPECTED}")

    result = pd.DataFrame(rows).sort_values(["horizon", "clip"])
    loaded_row = result.loc[
        result["horizon"].eq(2) & result["clip"].eq(1.0)
    ].iloc[0]
    for key in ("first_stage", "first_stage_se", "first_stage_t",
                "first_stage_F", "partial_r2", "instrument_sd"):
        if not np.isclose(float(loaded_row[key]), loaded[key],
                          rtol=locked.RTOL_COEF, atol=0.0):
            raise SystemExit(f"FAIL: loaded h=2 clip-1 {key} drifted")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"   clip-1 h=2 loaded; {n_new} new first stages estimated")
    for _, row in result.iterrows():
        print(f"   h={int(row['horizon'])}, clip={row['clip']:g}: "
              f"pi={row['first_stage']:+.6f}, "
              f"SE={row['first_stage_se']:.6f}, "
              f"F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"clipped={100 * row['clipped_share_of_live']:.2f}%")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
