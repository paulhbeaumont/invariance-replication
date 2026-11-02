"""Compare 5, 7, and 10-year predetermined market-share windows at h=2.

Each instrument allocates the same national category dollar change using the
county's market share over a rolling predetermined window that ends at t-7.
The comparison is estimated on each window's available sample and on the common
1985-2016 sample. No category shifter is clipped or winsorized.

Writes Results/instrument_normalization/
    h2_market_share_window_length_first_stage.csv
    h2_market_share_window_length_concentration.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_window_length_first_stage.csv
# mpgr-new-estimates: 6
# mpgr-variation-scope: h2|market-share|span-5-7-10|available-and-common-sample|no-clip

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
SPANS = (5, 7, 10)
COMMON_START = 1985
TARGET_CCG = 48015
FOCUS_CATEGORY = "2320"
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
RESULT_OUT = OUTDIR / "h2_market_share_window_length_first_stage.csv"
CONCENTRATION_OUT = OUTDIR / "h2_market_share_window_length_concentration.csv"
IDENTITY_RTOL = 1e-12


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_instrument(
        arr, categories, contract_years, target_years, population, units, span):
    """Build the market-share instrument for one exposure-window length."""
    national = arr.sum(axis=0, dtype=np.float64)
    result = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    concentration = []
    focus = categories.index(FOCUS_CATEGORY)
    target_positions = np.flatnonzero(units == TARGET_CCG)
    if len(target_positions) != 1:
        raise SystemExit(
            f"FAIL: county axis has {len(target_positions)} Austin rows")
    target = int(target_positions[0])

    for j, t in enumerate(target_years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - span + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        required = (exposure_lo, exposure_hi, t - H, t - 1, t)
        missing = [year for year in required if year not in contract_years]
        if missing:
            raise SystemExit(
                f"FAIL: span {span}, target {t} requires missing years {missing}")

        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = exposure.sum(axis=0, dtype=np.float64)
        live = np.isfinite(national_exposure) & (national_exposure > 0)
        if not live.any():
            raise SystemExit(
                f"FAIL: span {span}, target {t} has no live categories")

        post2 = 0.5 * (
            national[:, contract_years.index(t)]
            + national[:, contract_years.index(t - 1)])
        base = national[:, contract_years.index(t - H)]
        national_shift = post2 - base
        shares = np.zeros_like(exposure)
        np.divide(
            exposure, national_exposure[None, :], out=shares,
            where=live[None, :])
        share_sum_error = float(np.max(np.abs(
            shares[:, live].sum(axis=0) - 1.0)))
        if share_sum_error > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: span {span}, target {t} share-sum error is "
                f"{share_sum_error:.3e}")

        predicted = shares[:, live] @ national_shift[live]
        pbase = population[t - H].to_numpy(np.float64)
        if np.any(~np.isfinite(pbase)) or np.any(pbase <= 0):
            raise SystemExit(
                f"FAIL: span {span}, target {t} has invalid population")
        result[:, j] = predicted / pbase

        hhi = np.square(shares[:, live]).sum(axis=0)
        dollar_weights = national_exposure[live] / national_exposure[live].sum()
        weighted_hhi = float(dollar_weights @ hhi)
        max_unit, max_category_live = np.unravel_index(
            np.argmax(shares[:, live]), shares[:, live].shape)
        live_positions = np.flatnonzero(live)
        max_category = int(live_positions[max_category_live])
        focus_shares = shares[:, focus]
        concentration.append({
            "span": span,
            "year": int(t),
            "exposure_start": int(exposure_lo),
            "exposure_end": int(exposure_hi),
            "live_categories": int(live.sum()),
            "max_county_category_share":
                float(shares[max_unit, max_category]),
            "max_share_ccg": int(units[max_unit]),
            "max_share_category": categories[max_category],
            "exposure_dollar_weighted_hhi": weighted_hhi,
            "exposure_dollar_weighted_effective_counties": 1.0 / weighted_hhi,
            "austin_category_2320_share": float(focus_shares[target]),
            "category_2320_hhi": float(np.square(focus_shares).sum()),
            "category_2320_effective_counties":
                float(1.0 / np.square(focus_shares).sum()),
            "category_2320_austin_absolute_rank":
                int(np.sum(focus_shares > focus_shares[target]) + 1),
            "max_category_share_sum_error": share_sum_error,
        })

    return result, pd.DataFrame(concentration)


def estimation_row(
        span, panel, baseline, z, result, cluster, county_ids, t28,
        p69, reference_z):
    """Collect first-stage and Austin County leverage diagnostics."""
    sample_years = baseline["year"].to_numpy(np.int64)
    row, _ = p69.cluster_diagnostics(
        f"span_{span}_{panel}", result, cluster, sample_years, county_ids, t28)
    target = county_ids == TARGET_CCG
    target_codes = np.unique(cluster[target])
    if len(target_codes) != 1:
        raise SystemExit(
            f"FAIL: span {span}, panel {panel} has invalid Austin cluster")
    target_code = int(target_codes[0])
    target_signal = float(result["zt"][target] @ result["xt"][target])
    target_zz = float(result["zt"][target] @ result["zt"][target])
    target_score = float(result["score_cluster"][target_code])
    return {
        "panel": panel,
        "span": span,
        "exposure_window": f"t-{H + locked.LAG + span - 1} to t-{H + locked.LAG}",
        "sample_start": int(sample_years.min()),
        "sample_end": int(sample_years.max()),
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "fixed_effects": "division-by-year",
        "weight": "equal county",
        "cluster": "county",
        "tail_rule": "none",
        "first_stage": result["beta"],
        "first_stage_se": result["se"],
        "first_stage_t": result["t"],
        "first_stage_F": result["F"],
        "partial_r2": result["partial_r2"],
        "instrument_sd": float(np.std(z, ddof=1)),
        "within_instrument_sd": float(np.std(result["zt"], ddof=1)),
        "correlation_with_span5_on_same_sample":
            float(np.corrcoef(z, reference_z)[0, 1]),
        "cross_year_share_of_cluster_meat":
            row["cross_year_share_of_cluster_meat"],
        "effective_counties_in_cluster_meat":
            row["effective_counties_in_cluster_meat"],
        "largest_county_meat_share": row["largest_county_meat_share"],
        "largest_county_meat_ccg": row["largest_county_meat_ccg"],
        "austin_share_of_total_within_zx": target_signal / result["zx"],
        "austin_share_of_total_within_z_sum_squares":
            target_zz / result["zz"],
        "austin_share_of_cluster_meat": target_score ** 2 / result["meat"],
    }


def main():
    """Run the rolling-window first-stage and concentration comparison."""
    start = time.time()
    print("== h=2 market-share exposure-window length grid")
    p57 = load_module(
        "p57_window_grid", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_window_grid", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_window_grid",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p69 = load_module(
        "p69_window_grid",
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

    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)

    instruments, concentration_parts = {}, []
    for span in SPANS:
        instrument, concentration = build_instrument(
            arr, categories, contract_years, years, pivots["P"], units, span)
        instruments[span] = instrument
        concentration_parts.append(concentration)

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    zsample = {span: instrument[ri, ci]
               for span, instrument in instruments.items()}

    rows = []
    for panel in ("available sample", "common 1985-2016 sample"):
        for span in SPANS:
            if panel == "available sample":
                keep = np.isfinite(zsample[span])
            else:
                keep = (baseline["year"].to_numpy() >= COMMON_START)
            if not np.isfinite(zsample[span][keep]).all():
                raise SystemExit(
                    f"FAIL: span {span}, panel {panel} has missing instrument")
            selected = baseline.loc[keep].reset_index(drop=True)
            x, weight, fe, nofe, cluster = helper.estimation_arrays(selected)
            result = p69.within_first_stage(
                x, zsample[span][keep], weight, fe, nofe, cluster, t28)
            reference = zsample[5][keep]
            if not np.isfinite(reference).all():
                raise SystemExit(
                    f"FAIL: span-5 reference is missing for {panel}, span {span}")
            rows.append(estimation_row(
                span, panel, selected, zsample[span][keep], result, cluster,
                selected["ccg"].to_numpy(), t28, p69, reference))

    results = pd.DataFrame(rows)
    concentration = pd.concat(concentration_parts, ignore_index=True)
    span5_full = results.loc[
        results["panel"].eq("available sample") & results["span"].eq(5)]
    if len(span5_full) != 1:
        raise SystemExit("FAIL: span-5 fixture row is missing")
    fixture = p69.saved_results()["market_share"]
    for calculated, saved in {
            "first_stage": "first_stage",
            "first_stage_se": "first_stage_se",
            "first_stage_F": "first_stage_F",
            "partial_r2": "partial_r2"}.items():
        left = float(span5_full.iloc[0][calculated])
        right = float(fixture[saved])
        gap = abs(left - right) / max(abs(right), 1e-300)
        if gap >= locked.RTOL_COEF:
            raise SystemExit(
                f"FAIL: span-5 {calculated} fixture gap is {gap:.3e}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_OUT, index=False)
    concentration.to_csv(CONCENTRATION_OUT, index=False)

    for panel in results["panel"].unique():
        print(f"   {panel}:")
        for _, row in results.loc[results["panel"].eq(panel)].iterrows():
            print(
                f"      L={int(row['span']):2d}: "
                f"{int(row['sample_start'])}-{int(row['sample_end'])}, "
                f"F={row['first_stage_F']:.3f}, "
                f"partial R2={row['partial_r2']:.6f}, "
                f"Austin z-variance={100 * row['austin_share_of_total_within_z_sum_squares']:.2f}%, "
                f"Austin meat={100 * row['austin_share_of_cluster_meat']:.2f}%")
    print(f"   wrote {RESULT_OUT}")
    print(f"   wrote {CONCENTRATION_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
