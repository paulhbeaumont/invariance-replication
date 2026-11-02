"""Estimate an unclipped h=2 leave-one-county-out market-share shifter.

For each county and category, the national growth shifter excludes the focal
county's procurement from both the national change and its predetermined
national denominator. The county's predetermined category exposure remains
measured over t-11 through t-7, and the resulting predicted dollar change is
divided by population at t-2.

Writes Results/instrument_normalization/
    h2_leave_one_out_market_share_first_stage.csv
    h2_leave_one_out_market_share_cluster_diagnostics.csv
    h2_leave_one_out_market_share_cluster_score_lags.csv
    h2_leave_one_out_market_share_austin_year.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_leave_one_out_market_share_first_stage.csv
# mpgr-new-estimates: 1
# mpgr-variation-scope: h2|four-character|same-window-market-share|leave-one-county-out|no-clip

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
TARGET_CCG = 48015
FOCUS_CATEGORY = "2320"
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
RESULT_OUT = OUTDIR / "h2_leave_one_out_market_share_first_stage.csv"
CLUSTER_OUT = OUTDIR / "h2_leave_one_out_market_share_cluster_diagnostics.csv"
LAG_OUT = OUTDIR / "h2_leave_one_out_market_share_cluster_score_lags.csv"
AUSTIN_OUT = OUTDIR / "h2_leave_one_out_market_share_austin_year.csv"
IDENTITY_RTOL = 1e-12


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_loo_instrument(
        arr, categories, contract_years, target_years, population, units):
    """Build county exposure times rest-of-nation category growth."""
    national = arr.sum(axis=0, dtype=np.float64)
    result = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    diagnostics = []
    focus = categories.index(FOCUS_CATEGORY)
    target_positions = np.flatnonzero(units == TARGET_CCG)
    if len(target_positions) != 1:
        raise SystemExit(
            f"FAIL: county axis has {len(target_positions)} rows for {TARGET_CCG}")
    target = int(target_positions[0])

    for j, t in enumerate(target_years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        required = (exposure_lo, exposure_hi, t - H, t - 1, t)
        missing = [year for year in required if year not in contract_years]
        if missing:
            raise SystemExit(
                f"FAIL: target year {t} requires missing years {missing}")

        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = exposure.sum(axis=0, dtype=np.float64)

        post2_county = 0.5 * (
            arr[:, :, contract_years.index(t)].astype(np.float64)
            + arr[:, :, contract_years.index(t - 1)].astype(np.float64))
        base_county = arr[:, :, contract_years.index(t - H)].astype(np.float64)
        county_shift = post2_county - base_county
        national_shift = county_shift.sum(axis=0, dtype=np.float64)
        rest_exposure = national_exposure[None, :] - exposure
        rest_shift = national_shift[None, :] - county_shift

        identity_exposure = np.max(np.abs(
            rest_exposure + exposure - national_exposure[None, :]))
        identity_shift = np.max(np.abs(
            rest_shift + county_shift - national_shift[None, :]))
        exposure_scale = max(float(np.max(np.abs(national_exposure))), 1.0)
        shift_scale = max(float(np.max(np.abs(national_shift))), 1.0)
        if identity_exposure / exposure_scale > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: exposure leave-out identity failed in {t}")
        if identity_shift / shift_scale > IDENTITY_RTOL:
            raise SystemExit(f"FAIL: shift leave-out identity failed in {t}")

        live = np.isfinite(rest_exposure) & (rest_exposure > 0)
        holes = (~live) & (exposure > 0)
        growth = np.zeros_like(rest_shift)
        np.divide(rest_shift, rest_exposure, out=growth, where=live)
        if not np.isfinite(growth).all():
            raise SystemExit(
                f"FAIL: leave-one-out growth has nonfinite values in {t}")
        absolute_growth = np.where(live, np.abs(growth), -np.inf)
        max_unit, max_category = np.unravel_index(
            np.argmax(absolute_growth), absolute_growth.shape)

        pbase = population[t - H].to_numpy(np.float64)
        if np.any(~np.isfinite(pbase)) or np.any(pbase <= 0):
            raise SystemExit(
                f"FAIL: target year {t} has invalid base population")
        result[:, j] = np.einsum("uk,uk->u", exposure, growth) / pbase

        diagnostics.append({
            "year": int(t),
            "exposure_start": int(exposure_lo),
            "exposure_end": int(exposure_hi),
            "county_category_holes_with_positive_exposure": int(holes.sum()),
            "counties_with_at_least_one_hole": int(holes.any(axis=1).sum()),
            "max_exposure_identity_relative_gap":
                float(identity_exposure / exposure_scale),
            "max_shift_identity_relative_gap":
                float(identity_shift / shift_scale),
            "max_absolute_loo_growth": float(np.max(np.abs(growth[live]))),
            "max_absolute_loo_growth_ccg": int(units[max_unit]),
            "max_absolute_loo_growth_category": categories[max_category],
            "max_absolute_loo_growth_own_exposure":
                float(exposure[max_unit, max_category]),
            "max_absolute_loo_growth_rest_exposure":
                float(rest_exposure[max_unit, max_category]),
            "max_absolute_loo_growth_own_shift":
                float(county_shift[max_unit, max_category]),
            "max_absolute_loo_growth_rest_shift":
                float(rest_shift[max_unit, max_category]),
            "austin_raw_instrument": float(result[target, j]),
            "austin_category_2320_exposure": float(exposure[target, focus]),
            "austin_category_2320_rest_national_exposure":
                float(rest_exposure[target, focus]),
            "austin_category_2320_rest_national_shift":
                float(rest_shift[target, focus]),
            "austin_category_2320_loo_growth":
                float(growth[target, focus]),
            "austin_category_2320_inclusive_growth":
                float(national_shift[focus] / national_exposure[focus]),
        })

    return result, pd.DataFrame(diagnostics)


def main():
    """Run the leave-one-county-out first-stage diagnostic."""
    start = time.time()
    print("== h=2 leave-one-county-out market-share shifter")
    p57 = load_module(
        "p57_loo_market", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_loo_market", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_loo_market",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_loo_market",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_loo_market",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    loo_wide, loo_diagnostics = build_loo_instrument(
        arr, categories, contract_years, years, pivots["P"], units)
    inclusive_wide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    locked_wide = store.load_instrument(H, units, "z")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    instruments = {
        "market_share_inclusive": inclusive_wide[ri, ci],
        "market_share_loo": loo_wide[ri, ci],
        "locked": locked_wide[ri, ci],
    }
    if any(not np.isfinite(z).all() for z in instruments.values()):
        raise SystemExit("FAIL: an instrument has missing sample values")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    results = {
        name: p69.within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        for name, z in instruments.items()
    }
    fixtures = p69.saved_results()
    p69.assert_fixture(
        "market_share", results["market_share_inclusive"],
        fixtures["market_share"])
    p69.assert_fixture("locked", results["locked"], fixtures["locked"])

    sample_years = baseline["year"].to_numpy(np.int64)
    county_ids = baseline["ccg"].to_numpy()
    cluster_rows, lag_rows = [], []
    for name, result in results.items():
        row, lags = p69.cluster_diagnostics(
            name, result, cluster, sample_years, county_ids, t28)
        cluster_rows.append(row)
        lag_rows.extend(lags)
    cluster_frame = pd.DataFrame(cluster_rows)

    target_mask = county_ids == TARGET_CCG
    if int(target_mask.sum()) != baseline["year"].nunique():
        raise SystemExit(
            f"FAIL: Austin County has {int(target_mask.sum())} rows")
    target_code = np.unique(cluster[target_mask])
    if len(target_code) != 1:
        raise SystemExit("FAIL: Austin County maps to multiple clusters")
    target_code = int(target_code[0])

    for row in cluster_rows:
        name = row["instrument"]
        result = results[name]
        austin_signal = float(
            result["zt"][target_mask] @ result["xt"][target_mask])
        austin_zz = float(
            result["zt"][target_mask] @ result["zt"][target_mask])
        austin_score = float(result["score_cluster"][target_code])
        row["austin_share_of_total_within_zx"] = (
            austin_signal / result["zx"])
        row["austin_share_of_total_within_z_sum_squares"] = (
            austin_zz / result["zz"])
        row["austin_share_of_cluster_meat"] = (
            austin_score ** 2 / result["meat"])
    cluster_frame = pd.DataFrame(cluster_rows)

    loo = results["market_share_loo"]
    inclusive = results["market_share_inclusive"]
    locked_result = results["locked"]
    diagnostic_sample = loo_diagnostics.loc[
        loo_diagnostics["year"].isin(np.unique(sample_years))]
    if len(diagnostic_sample) != baseline["year"].nunique():
        raise SystemExit("FAIL: LOO diagnostics do not match estimation years")
    maximum_growth = diagnostic_sample.loc[
        diagnostic_sample["max_absolute_loo_growth"].idxmax()]
    row = {
        "variant": "leave_one_county_out_market_share",
        "label": "County exposure times rest-of-nation category growth",
        "horizon": H,
        "tail_rule": "none",
        "category_level": "four-character",
        "n_categories": len(categories),
        "sample_start": int(baseline["year"].min()),
        "sample_end": int(baseline["year"].max()),
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "fixed_effects": "division-by-year",
        "weight": "equal county",
        "cluster": "county",
        "first_stage": loo["beta"],
        "first_stage_se": loo["se"],
        "first_stage_t": loo["t"],
        "first_stage_F": loo["F"],
        "partial_r2": loo["partial_r2"],
        "instrument_sd": float(np.std(instruments["market_share_loo"], ddof=1)),
        "within_instrument_sd": float(np.std(loo["zt"], ddof=1)),
        "correlation_with_inclusive_market_share": float(np.corrcoef(
            instruments["market_share_loo"],
            instruments["market_share_inclusive"])[0, 1]),
        "correlation_with_locked_instrument": float(np.corrcoef(
            instruments["market_share_loo"], instruments["locked"])[0, 1]),
        "inclusive_market_share_first_stage_F": inclusive["F"],
        "locked_first_stage_F": locked_result["F"],
        "county_category_holes_with_positive_exposure": int(
            diagnostic_sample[
                "county_category_holes_with_positive_exposure"].sum()),
        "max_counties_with_at_least_one_hole_in_year": int(
            diagnostic_sample["counties_with_at_least_one_hole"].max()),
        "max_exposure_identity_relative_gap": float(
            diagnostic_sample["max_exposure_identity_relative_gap"].max()),
        "max_shift_identity_relative_gap": float(
            diagnostic_sample["max_shift_identity_relative_gap"].max()),
        "max_absolute_loo_growth": float(
            diagnostic_sample["max_absolute_loo_growth"].max()),
        "max_absolute_loo_growth_year": int(maximum_growth["year"]),
        "max_absolute_loo_growth_ccg": int(
            maximum_growth["max_absolute_loo_growth_ccg"]),
        "max_absolute_loo_growth_category":
            maximum_growth["max_absolute_loo_growth_category"],
        "max_absolute_loo_growth_own_exposure": float(
            maximum_growth["max_absolute_loo_growth_own_exposure"]),
        "max_absolute_loo_growth_rest_exposure": float(
            maximum_growth["max_absolute_loo_growth_rest_exposure"]),
        "max_absolute_loo_growth_own_shift": float(
            maximum_growth["max_absolute_loo_growth_own_shift"]),
        "max_absolute_loo_growth_rest_shift": float(
            maximum_growth["max_absolute_loo_growth_rest_shift"]),
        "estimation_source": "persisted pipeline tensor",
    }

    austin_rows = []
    diagnostic_by_year = diagnostic_sample.set_index("year")
    for pos in np.flatnonzero(target_mask):
        year = int(sample_years[pos])
        detail = diagnostic_by_year.loc[year]
        austin_rows.append({
            **detail.to_dict(),
            "year": year,
            "ccg": TARGET_CCG,
            "county": "Austin County",
            "state": "Texas",
            "realized_county_procurement_change_per_capita": float(x[pos]),
            "realized_county_procurement_change_per_capita_within":
                float(loo["xt"][pos]),
            "inclusive_instrument": float(instruments[
                "market_share_inclusive"][pos]),
            "inclusive_instrument_within": float(inclusive["zt"][pos]),
            "inclusive_cluster_score_contribution":
                float(inclusive["score_obs"][pos]),
            "loo_instrument": float(instruments["market_share_loo"][pos]),
            "loo_instrument_within": float(loo["zt"][pos]),
            "loo_first_stage_residual": float(loo["residual"][pos]),
            "loo_cluster_score_contribution": float(loo["score_obs"][pos]),
            "locked_instrument": float(instruments["locked"][pos]),
            "locked_instrument_within": float(locked_result["zt"][pos]),
            "locked_cluster_score_contribution":
                float(locked_result["score_obs"][pos]),
        })

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(RESULT_OUT, index=False)
    cluster_frame.to_csv(CLUSTER_OUT, index=False)
    pd.DataFrame(lag_rows).to_csv(LAG_OUT, index=False)
    pd.DataFrame(austin_rows).to_csv(AUSTIN_OUT, index=False)

    loo_cluster = cluster_frame.loc[
        cluster_frame["instrument"].eq("market_share_loo")].iloc[0]
    print(f"   sample: {row['n']:,} county-years, "
          f"{row['n_counties']:,} counties, "
          f"{row['sample_start']}-{row['sample_end']}")
    print(f"   LOO first stage: pi={row['first_stage']:+.6f}, "
          f"SE={row['first_stage_se']:.6f}, "
          f"F={row['first_stage_F']:.3f}, "
          f"partial R2={row['partial_r2']:.6f}")
    print(f"   F inclusive={row['inclusive_market_share_first_stage_F']:.3f}; "
          f"F locked={row['locked_first_stage_F']:.3f}")
    print("   Austin: variance share="
          f"{100 * loo_cluster['austin_share_of_total_within_z_sum_squares']:.2f}%, "
          "signal share="
          f"{100 * loo_cluster['austin_share_of_total_within_zx']:.2f}%, "
          "cluster-meat share="
          f"{100 * loo_cluster['austin_share_of_cluster_meat']:.2f}%")
    print(f"   corr(inclusive)="
          f"{row['correlation_with_inclusive_market_share']:+.6f}; "
          f"corr(locked)={row['correlation_with_locked_instrument']:+.6f}")
    print(f"   positive-exposure LOO holes="
          f"{row['county_category_holes_with_positive_exposure']:,}")
    print(f"   wrote {RESULT_OUT}")
    print(f"   wrote {CLUSTER_OUT}")
    print(f"   wrote {LAG_OUT}")
    print(f"   wrote {AUSTIN_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
