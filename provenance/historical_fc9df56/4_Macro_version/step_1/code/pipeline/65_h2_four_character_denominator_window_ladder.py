"""Estimate the h=2 four-character denominator-window first-stage ladder.

For window length L, the national category shifter is

    [0.5 * (G_k,t + G_k,t-1) - G_k,t-2]
    / mean(G_k,t-2, ..., G_k,t-L-1).

The numerator, four-character category axis, county exposure, estimation
sample, weights, fixed effects, clustering, and tail rules are held fixed.
Saved clip-1 and clip-2 coefficient rows are loaded rather than re-estimated.

Writes Results/instrument_normalization/
    h2_county_four_character_denominator_window_ladder.csv
    h2_county_four_character_denominator_window_correlations.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-batch-approved: explicit
# mpgr-new-estimates: 14
# mpgr-variation-scope: four-character|L1-L7|clip3,clip4|h2|division-year-FE

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
       / "h2_county_four_character_denominator_window_ladder.csv")
CORR_OUT = (ROOT / "Results" / "instrument_normalization"
            / "h2_county_four_character_denominator_window_correlations.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
WINDOWS = (1, 2, 3, 4, 5, 6, 7)
CAPS = (1.0, 2.0, 3.0, 4.0)
NEW_CAPS = (3.0, 4.0)
FIT_COLUMNS = (
    "first_stage", "first_stage_se", "first_stage_t", "first_stage_F",
    "partial_r2", "instrument_sd", "within_instrument_sd",
)


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaded_results():
    """Load the saved clip-1 and clip-2 coefficient rows."""
    saved = {}
    prior = pd.read_csv(OUT)
    for window in WINDOWS:
        for cap in (1.0, 2.0):
            selected = prior.loc[
                prior["denominator_window_years"].eq(window)
                & prior["clip"].eq(cap)
            ]
            if len(selected) != 1:
                raise SystemExit(
                    f"FAIL: prior ladder has {len(selected)} rows for "
                    f"L={window}, clip={cap:g}")
            row = selected.iloc[0]
            if int(row["n"]) != locked.N_COUNTY[H]:
                raise SystemExit(
                    f"FAIL: prior L={window}, clip={cap:g} row has "
                    f"N={int(row['n']):,}, expected {locked.N_COUNTY[H]:,}")
            saved[(window, cap)] = row
    return saved


def main():
    """Run the requested four-character denominator-window ladder."""
    start = time.time()
    print("== h=2 four-character denominator-window first-stage ladder")
    p57 = load_module("p57_four_character_window",
                      STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_four_character_window",
                      STEP2_PY / "28_table_d1.py")
    helper = load_module("clip_four_character_window",
                         PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p64 = load_module("p64_four_character_window",
                      PIPELINE / "64_h2_prefix2_denominator_window_ladder.py")
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

    units, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    national = arr.sum(axis=0, dtype=np.float64)
    years = store.target_years(H)
    shifters, diagnostics = p64.category_shifters(
        national, contract_years, years, windows=WINDOWS, caps=CAPS)
    units2, years2, zwide, _, _ = helper.county_instruments(shifters)
    if not np.array_equal(units, units2) or years != years2:
        raise SystemExit("FAIL: instrument arrays do not match the stored panel")

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
    loaded = loaded_results()
    sample_years = set(baseline["year"].unique())
    rows = []
    n_new = 0
    for window in WINDOWS:
        for cap in CAPS:
            key = (window, cap)
            z = zsample[key]
            if key in loaded:
                saved = loaded[key]
                fit = {name: float(saved[name]) for name in FIT_COLUMNS}
                source = f"loaded existing L={window} result"
            else:
                if cap not in NEW_CAPS:
                    raise SystemExit(
                        f"FAIL: no saved or approved estimate for L={window}, "
                        f"clip={cap:g}")
                fit = p64.estimate_first_stage(
                    x, z, weight, fe, nofe, cluster, t28)
                source = "estimated in four-character window ladder"
                n_new += 1
            records = [
                record for record in diagnostics[key]
                if record["year"] in sample_years
            ]
            n_live = sum(record["n_live"] for record in records)
            n_modified = sum(record["n_modified"] for record in records)
            rows.append({
                "denominator_window_years": window,
                "clip": cap,
                "tail_rule": f"clip_{cap:g}",
                "category_level": "four-character",
                "category_universe": "persisted top-800 four-character codes",
                "n_categories": len(categories),
                "horizon": H,
                "numerator": "post2 minus final pre-shock year",
                "denominator": (
                    f"mean national category spending over {window} pre-shock "
                    "year(s) ending at t-2"),
                "zero_denominator_rule": "exclude category-year",
                "sample_start": int(baseline["year"].min()),
                "sample_end": int(baseline["year"].max()),
                "n": len(baseline),
                "n_counties": int(baseline["ccg"].nunique()),
                "fixed_effects": "division-by-year",
                "weight": "equal county",
                "cluster": "county",
                **fit,
                "live_category_years": int(n_live),
                "modified_category_years": int(n_modified),
                "modified_share_of_live": n_modified / n_live,
                "correlation_with_L1_same_clip": float(np.corrcoef(
                    z, zsample[(1, cap)])[0, 1]),
                "correlation_with_L5_same_clip": float(np.corrcoef(
                    z, zsample[(5, cap)])[0, 1]),
                "estimation_source": source,
            })
    if n_new != 14:
        raise SystemExit(f"FAIL: estimated {n_new} new rows, expected 14")

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
    print(f"   category universe: {len(categories)} four-character codes")
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    print("   clip-1 and clip-2 rows loaded; 14 new first stages estimated")
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
