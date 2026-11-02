"""Estimate the h=2 exposure-window market-share instrument first stage.

For county c, category k, and target year t, define predetermined exposure as
mean annual county procurement over t-11 through t-7. The category weight is
that county exposure divided by national exposure over the identical window.
The national shift is the average of t-1 and t procurement minus t-2
procurement. Their product predicts a county dollar change and is divided by
population at t-2.

Writes Results/instrument_normalization/
    h2_county_exposure_window_market_share_shifter.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_county_exposure_window_market_share_shifter.csv
# mpgr-new-estimates: 1
# mpgr-variation-scope: h2|four-character|same-exposure-window-market-shares|no-clip|county-first-stage

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

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
       / "h2_county_exposure_window_market_share_shifter.csv")
REFERENCE_OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
                 / "h2_county_first_stage_sequence.csv")
IDENTITY_RTOL = 1e-12


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_instrument(arr, contract_years, target_years, population):
    """Build county-share times national-dollar-shift instruments."""
    national = arr.sum(axis=0, dtype=np.float64)
    result = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    diagnostics = []

    for j, t in enumerate(target_years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        required = (exposure_lo, exposure_hi, t - H, t - 1, t)
        missing = [year for year in required if year not in contract_years]
        if missing:
            raise SystemExit(
                f"FAIL: target year {t} requires missing contract years {missing}")

        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = exposure.sum(axis=0, dtype=np.float64)
        live = np.isfinite(national_exposure) & (national_exposure > 0)
        if not live.any():
            raise SystemExit(f"FAIL: target year {t} has no live categories")
        if np.any(exposure[:, ~live] != 0):
            raise SystemExit(
                f"FAIL: target year {t} has county exposure with zero national exposure")

        post2 = 0.5 * (
            national[:, contract_years.index(t)]
            + national[:, contract_years.index(t - 1)])
        base = national[:, contract_years.index(t - H)]
        national_shift = post2 - base

        shares = exposure[:, live] / national_exposure[live]
        share_sum_error = float(np.max(np.abs(shares.sum(axis=0) - 1.0)))
        if share_sum_error > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: target year {t} category-share sum error is "
                f"{share_sum_error:.3e}")

        predicted_from_levels = shares @ national_shift[live]
        growth = national_shift[live] / national_exposure[live]
        predicted_from_growth = exposure[:, live] @ growth
        identity_scale = max(
            float(np.max(np.abs(predicted_from_levels))), 1.0)
        representation_gap = float(np.max(np.abs(
            predicted_from_levels - predicted_from_growth)))
        representation_relative_gap = representation_gap / identity_scale
        if representation_relative_gap > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: target year {t} level-growth identity relative gap is "
                f"{representation_relative_gap:.3e}")

        pbase = population[t - H].to_numpy(np.float64)
        if np.any(~np.isfinite(pbase)) or np.any(pbase <= 0):
            raise SystemExit(
                f"FAIL: target year {t} has missing or nonpositive base population")
        result[:, j] = predicted_from_levels / pbase
        diagnostics.append({
            "year": t,
            "exposure_start": exposure_lo,
            "exposure_end": exposure_hi,
            "live_categories": int(live.sum()),
            "zero_exposure_categories": int((~live).sum()),
            "max_category_share_sum_error": share_sum_error,
            "max_level_growth_identity_gap_dollars": representation_gap,
            "max_level_growth_identity_relative_gap":
                representation_relative_gap,
        })

    return result, diagnostics


def reference_rows():
    """Load the locked and matched rolling-five-year first-stage references."""
    reference = pd.read_csv(REFERENCE_OUT)
    rows = {}
    for key in ("A_current", "C_matched_roll5"):
        selected = reference.loc[reference["instrument"].eq(key)]
        if len(selected) != 1:
            raise SystemExit(
                f"FAIL: first-stage sequence has {len(selected)} rows for {key}")
        row = selected.iloc[0]
        if int(row["n"]) != locked.N_COUNTY[H]:
            raise SystemExit(
                f"FAIL: reference {key} has N={int(row['n']):,}, expected "
                f"{locked.N_COUNTY[H]:,}")
        rows[key] = row
    return rows


def main():
    """Estimate the same-window market-share first stage on the locked sample."""
    start = time.time()
    print("== h=2 exposure-window market-share shifter")
    p57 = load_module(
        "p57_exposure_market_share", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_exposure_market_share", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_exposure_market_share",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p64 = load_module(
        "p64_exposure_market_share",
        PIPELINE / "64_h2_prefix2_denominator_window_ladder.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0:
        raise SystemExit(f"FAIL: baseline dropped {dropped} additional counties")
    if len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows, expected "
            f"{locked.N_COUNTY[H]:,}")
    if baseline["ccg"].nunique() != locked.N_COUNTIES:
        raise SystemExit(
            f"FAIL: baseline has {baseline['ccg'].nunique():,} counties, "
            f"expected {locked.N_COUNTIES:,}")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    zwide, diagnostics = build_instrument(
        arr, contract_years, years, pivots["P"])

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)
    z = zwide[ri, ci]
    if not np.isfinite(z).all():
        raise SystemExit(
            f"FAIL: new instrument has {int((~np.isfinite(z)).sum()):,} "
            "missing sample values")

    locked_wide = store.load_instrument(H, units, "z")
    z_locked = locked_wide[ri, ci]
    if not np.isfinite(z_locked).all():
        raise SystemExit("FAIL: locked instrument has missing sample values")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    fit = p64.estimate_first_stage(
        x, z, weight, fe, nofe, cluster, t28)
    refs = reference_rows()
    diag = pd.DataFrame(diagnostics)
    estimation_years = set(baseline["year"].unique())
    diag = diag.loc[diag["year"].isin(estimation_years)]
    if len(diag) != baseline["year"].nunique():
        raise SystemExit(
            "FAIL: diagnostic years do not match the estimation sample")

    row = {
        "variant": "exposure_window_market_share",
        "label": "Same-window county share times matched national dollar shift",
        "horizon": H,
        "national_shift": "mean(t-1,t) minus t-2 national category dollars",
        "category_weight": (
            "county mean category dollars divided by national mean category "
            "dollars over t-11 through t-7"),
        "tail_rule": "none",
        "category_level": "four-character",
        "category_universe": "persisted top-800 four-character codes",
        "n_categories": len(categories),
        "sample_start": int(baseline["year"].min()),
        "sample_end": int(baseline["year"].max()),
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "fixed_effects": "division-by-year",
        "weight": "equal county",
        "cluster": "county",
        **fit,
        "correlation_with_locked_instrument": float(np.corrcoef(
            z, z_locked)[0, 1]),
        "locked_first_stage_F": float(refs["A_current"]["first_stage_F"]),
        "matched_pre5_clip1_first_stage_F": float(
            refs["C_matched_roll5"]["first_stage_F"]),
        "min_live_categories": int(diag["live_categories"].min()),
        "max_live_categories": int(diag["live_categories"].max()),
        "max_category_share_sum_error": float(
            diag["max_category_share_sum_error"].max()),
        "max_level_growth_identity_gap_dollars": float(
            diag["max_level_growth_identity_gap_dollars"].max()),
        "max_level_growth_identity_relative_gap": float(
            diag["max_level_growth_identity_relative_gap"].max()),
        "estimation_source": "estimated from persisted pipeline tensor",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(f"   sample: {row['n']:,} county-years, "
          f"{row['n_counties']:,} counties, "
          f"{row['sample_start']}-{row['sample_end']}")
    print(f"   first stage: pi={row['first_stage']:+.6f}, "
          f"SE={row['first_stage_se']:.6f}, "
          f"F={row['first_stage_F']:.3f}, "
          f"partial R2={row['partial_r2']:.6f}")
    print(f"   corr(locked)={row['correlation_with_locked_instrument']:+.6f}")
    print(f"   share-sum max error={row['max_category_share_sum_error']:.3e}")
    print("   level-growth identity max relative gap="
          f"{row['max_level_growth_identity_relative_gap']:.3e}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
