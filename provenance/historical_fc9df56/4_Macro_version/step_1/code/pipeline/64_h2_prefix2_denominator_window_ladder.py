"""Estimate the h=2 prefix-2 denominator-window first-stage ladder.

For window length L, the national category shifter is

    [0.5 * (G_k,t + G_k,t-1) - G_k,t-2]
    / mean(G_k,t-2, ..., G_k,t-L-1).

The numerator, category aggregation, county exposure, estimation sample,
weights, fixed effects, clustering, and tail rules are held fixed. The saved
L=1 coefficient rows are loaded rather than re-estimated.

Writes Results/instrument_normalization/
    h2_county_prefix2_denominator_window_ladder.csv
    h2_county_prefix2_denominator_window_correlations.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-batch-approved: explicit
# mpgr-new-estimates: 8
# mpgr-variation-scope: prefix2|L2,L3,L5,L7|clip1,clip2|h2|division-year-FE

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
PREFIX2_OUT = (ROOT / "Results" / "instrument_normalization"
               / "h2_county_prefix2_tail_comparison.csv")
OUT = (ROOT / "Results" / "instrument_normalization"
       / "h2_county_prefix2_denominator_window_ladder.csv")
CORR_OUT = (ROOT / "Results" / "instrument_normalization"
            / "h2_county_prefix2_denominator_window_correlations.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
PREFIX_DIGITS = 2
WINDOWS = (1, 2, 3, 5, 7)
NEW_WINDOWS = (2, 3, 5, 7)
CAPS = (1.0, 2.0)


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_shifters(national, contract_years, target_years,
                      windows=WINDOWS, caps=CAPS):
    """Build every denominator-window and cap combination."""
    shifters = {
        (window, cap): np.zeros(
            (national.shape[0], len(target_years)), np.float64)
        for window in windows for cap in caps
    }
    diagnostics = {
        key: [] for key in shifters
    }
    for j, end_year in enumerate(target_years):
        post2 = 0.5 * (
            national[:, contract_years.index(end_year)]
            + national[:, contract_years.index(end_year - 1)])
        base = national[:, contract_years.index(end_year - H)]
        for window in windows:
            denominator_years = [
                end_year - H - ell for ell in range(window)
            ]
            missing = [year for year in denominator_years
                       if year not in contract_years]
            if missing:
                raise SystemExit(
                    f"FAIL: denominator window L={window} at {end_year} "
                    f"requires missing contract years {missing}")
            denominator = np.column_stack([
                national[:, contract_years.index(year)]
                for year in denominator_years
            ]).mean(axis=1)
            live = (np.isfinite(post2) & np.isfinite(base)
                    & np.isfinite(denominator) & (denominator > 0))
            raw = np.zeros(len(base), np.float64)
            raw[live] = (post2[live] - base[live]) / denominator[live]
            for cap in caps:
                values = np.zeros(len(base), np.float64)
                values[live] = np.clip(raw[live], -cap, cap)
                shifters[(window, cap)][:, j] = values
                diagnostics[(window, cap)].append({
                    "year": end_year,
                    "n_live": int(live.sum()),
                    "n_modified": int(
                        (live & (np.abs(raw) > cap)).sum()),
                })
    return shifters, diagnostics


def county_instruments(arr, mapping, shifters, contract_years,
                       target_years, population):
    """Aggregate prefix shifters with the locked county exposures."""
    instruments = {
        key: np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
        for key in shifters
    }
    keys = list(shifters)
    for j, end_year in enumerate(target_years):
        exposure_hi = end_year - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure4 = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        exposure2 = exposure4 @ mapping
        pbase = population[end_year - H].to_numpy(np.float64)
        matrix = np.column_stack([shifters[key][:, j] for key in keys])
        values = exposure2 @ matrix / pbase[:, None]
        for col, key in enumerate(keys):
            instruments[key][:, j] = values[:, col]
    return instruments


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


def loaded_l1_rows():
    """Load the two saved L=1 first-stage rows."""
    saved = pd.read_csv(PREFIX2_OUT)
    expected = {f"clip_{cap:g}" for cap in CAPS}
    if len(saved) != len(CAPS) or set(saved["tail_rule"]) != expected:
        raise SystemExit("FAIL: saved prefix-2 L=1 rows are incomplete")
    return {
        float(row["clip"]): row
        for _, row in saved.iterrows()
    }


def main():
    """Run the requested prefix-2 denominator-window ladder."""
    start = time.time()
    print("== h=2 prefix-2 denominator-window first-stage ladder")
    p57 = load_module("p57_window_ladder",
                      STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_window_ladder",
                      STEP2_PY / "28_table_d1.py")
    p63 = load_module("p63_window_ladder",
                      PIPELINE / "63_h2_prefix3_category_shifter.py")
    helper = load_module("clip_window_ladder",
                         PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12,
                                      fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0:
        raise SystemExit(f"FAIL: baseline dropped {dropped} additional counties")
    if len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows, expected "
            f"{locked.N_COUNTY[H]:,}")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    prefixes, assignment, mapping = p63.prefix_mapping(
        categories, PREFIX_DIGITS)
    national = p63.grouped_national(arr, assignment, len(prefixes))
    years = store.target_years(H)
    shifters, diagnostics = category_shifters(
        national, contract_years, years)
    zwide = county_instruments(
        arr, mapping, shifters, contract_years, years, pivots["P"])

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)
    zsample = {key: matrix[ri, ci] for key, matrix in zwide.items()}
    if any(not np.isfinite(z).all() for z in zsample.values()):
        raise SystemExit("FAIL: a denominator-window instrument is missing")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    loaded = loaded_l1_rows()
    sample_years = set(baseline["year"].unique())
    rows = []
    n_new = 0
    for window in WINDOWS:
        for cap in CAPS:
            z = zsample[(window, cap)]
            if window == 1:
                saved = loaded[cap]
                fit = {
                    name: float(saved[name])
                    for name in (
                        "first_stage", "first_stage_se", "first_stage_t",
                        "first_stage_F", "partial_r2", "instrument_sd",
                        "within_instrument_sd")
                }
                source = "loaded existing prefix-2 result"
            else:
                fit = estimate_first_stage(
                    x, z, weight, fe, nofe, cluster, t28)
                source = "estimated in denominator-window ladder"
                n_new += 1
            records = [
                record for record in diagnostics[(window, cap)]
                if record["year"] in sample_years
            ]
            n_live = sum(record["n_live"] for record in records)
            n_modified = sum(record["n_modified"] for record in records)
            rows.append({
                "denominator_window_years": window,
                "clip": cap,
                "tail_rule": f"clip_{cap:g}",
                "category_level": "2-character prefix",
                "category_universe": "persisted top-800 four-character codes",
                "n_four_character_categories": len(categories),
                "n_prefix_categories": len(prefixes),
                "horizon": H,
                "numerator": "post2 minus final pre-shock year",
                "denominator": (
                    f"mean national prefix spending over {window} pre-shock "
                    "year(s) ending at t-2"),
                "zero_denominator_rule": "exclude prefix-year",
                "sample_start": int(baseline["year"].min()),
                "sample_end": int(baseline["year"].max()),
                "n": len(baseline),
                "n_counties": int(baseline["ccg"].nunique()),
                "fixed_effects": "division-by-year",
                "weight": "equal county",
                "cluster": "county",
                **fit,
                "live_prefix_years": int(n_live),
                "modified_prefix_years": int(n_modified),
                "modified_share_of_live": n_modified / n_live,
                "correlation_with_L1_same_clip": float(np.corrcoef(
                    z, zsample[(1, cap)])[0, 1]),
                "correlation_with_L5_same_clip": float(np.corrcoef(
                    z, zsample[(5, cap)])[0, 1]),
                "estimation_source": source,
            })
    if n_new != 8:
        raise SystemExit(f"FAIL: estimated {n_new} new rows, expected 8")

    keys = [(window, cap) for window in WINDOWS for cap in CAPS]
    correlation_rows = []
    for i, left in enumerate(keys):
        for right in keys[i:]:
            correlation_rows.append({
                "left_window_years": left[0],
                "left_clip": left[1],
                "right_window_years": right[0],
                "right_clip": right[1],
                "correlation": float(np.corrcoef(
                    zsample[left], zsample[right])[0, 1]),
                "n": len(baseline),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    pd.DataFrame(correlation_rows).to_csv(CORR_OUT, index=False)
    print(f"   category universe: {len(categories)} four-character codes "
          f"collapsed to {len(prefixes)} two-character prefixes")
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    print("   L=1 rows loaded; eight new first stages estimated")
    for row in rows:
        print(f"   L={row['denominator_window_years']}, "
              f"clip={row['clip']:g}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"modified={100 * row['modified_share_of_live']:.2f}%, "
              f"corr(L1)={row['correlation_with_L1_same_clip']:+.3f}, "
              f"corr(L5)={row['correlation_with_L5_same_clip']:+.3f}")
    print(f"   wrote {OUT}")
    print(f"   wrote {CORR_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
