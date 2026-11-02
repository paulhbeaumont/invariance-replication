"""Test unclipped h=2 leave-one-county-out shifters by category detail.

County procurement categories are kept at four characters or collapsed to
three- and two-character prefixes. For each county-group-year cell, the
shifter is the rest-of-country procurement change divided by rest-of-country
predetermined exposure. The county's own predetermined exposure multiplies
that shifter, and predicted dollars are divided by population at t-2.

Writes Results/instrument_normalization/
    h2_leave_one_out_category_coarsening_first_stage.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_leave_one_out_category_coarsening_first_stage.csv
# mpgr-new-estimates: 3
# mpgr-variation-scope: h2|market-share-loo|prefix-4-3-2|common-sample|five-year-exposure|no-clip

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
PREFIX_DIGITS = (4, 3, 2)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
OUT = OUTDIR / "h2_leave_one_out_category_coarsening_first_stage.csv"
INCLUSIVE_REFERENCE = OUTDIR / "h2_market_share_category_coarsening_first_stage.csv"
LOO_REFERENCE = OUTDIR / "h2_leave_one_out_market_share_first_stage.csv"
IDENTITY_RTOL = 1e-12


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_loo(arr, categories, contract_years, target_years, population,
              units, prefix_digits, p63):
    """Build county exposure times rest-of-country group growth."""
    prefixes, assignment, mapping = p63.prefix_mapping(
        categories, prefix_digits)
    national = p63.grouped_national(arr, assignment, len(prefixes))
    result = np.full((len(units), len(target_years)), np.nan, np.float64)
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
                f"FAIL: target year {t} requires missing years {missing}")

        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure4 = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        exposure = exposure4 @ mapping
        national_exposure = exposure.sum(axis=0, dtype=np.float64)

        post4 = 0.5 * (
            arr[:, :, contract_years.index(t)].astype(np.float64)
            + arr[:, :, contract_years.index(t - 1)].astype(np.float64))
        base4 = arr[:, :, contract_years.index(t - H)].astype(np.float64)
        county_shift = (post4 - base4) @ mapping
        national_shift = county_shift.sum(axis=0, dtype=np.float64)

        rest_exposure = national_exposure[None, :] - exposure
        rest_shift = national_shift[None, :] - county_shift
        exposure_gap = float(np.max(np.abs(
            rest_exposure + exposure - national_exposure[None, :])))
        shift_gap = float(np.max(np.abs(
            rest_shift + county_shift - national_shift[None, :])))
        exposure_scale = max(float(np.max(np.abs(national_exposure))), 1.0)
        shift_scale = max(float(np.max(np.abs(national_shift))), 1.0)
        if exposure_gap / exposure_scale > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: prefix {prefix_digits} exposure identity failed in {t}")
        if shift_gap / shift_scale > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: prefix {prefix_digits} shift identity failed in {t}")

        live = np.isfinite(rest_exposure) & (rest_exposure > 0)
        holes = (~live) & (exposure > 0)
        growth = np.zeros_like(rest_shift)
        np.divide(rest_shift, rest_exposure, out=growth, where=live)
        if not np.isfinite(growth).all():
            raise SystemExit(
                f"FAIL: prefix {prefix_digits} has nonfinite growth in {t}")
        absolute_growth = np.abs(growth[live])
        max_unit, max_group = np.unravel_index(
            np.argmax(np.where(live, np.abs(growth), -np.inf)), growth.shape)

        pbase = population[t - H].to_numpy(np.float64)
        if np.any(~np.isfinite(pbase)) or np.any(pbase <= 0):
            raise SystemExit(
                f"FAIL: prefix {prefix_digits} has invalid population in {t}")
        result[:, j] = np.einsum("ug,ug->u", exposure, growth) / pbase
        diagnostics.append({
            "year": int(t),
            "positive_exposure_holes": int(holes.sum()),
            "counties_with_holes": int(holes.any(axis=1).sum()),
            "max_absolute_growth": float(absolute_growth.max()),
            "p99_absolute_growth": float(np.quantile(absolute_growth, 0.99)),
            "p999_absolute_growth": float(np.quantile(absolute_growth, 0.999)),
            "max_growth_ccg": int(units[max_unit]),
            "max_growth_group": prefixes[max_group],
            "max_growth_rest_exposure": float(rest_exposure[max_unit, max_group]),
            "max_growth_rest_shift": float(rest_shift[max_unit, max_group]),
            "max_exposure_identity_relative_gap": exposure_gap / exposure_scale,
            "max_shift_identity_relative_gap": shift_gap / shift_scale,
        })

    return result, pd.DataFrame(diagnostics), prefixes


def main():
    """Estimate the three category-detail specifications on one sample."""
    start = time.time()
    print("== h=2 leave-one-out category coarsening")
    p57 = load_module(
        "p57_loo_coarsen", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_loo_coarsen", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_loo_coarsen",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p63 = load_module(
        "p63_loo_coarsen", PIPELINE / "63_h2_prefix3_category_shifter.py")
    p69 = load_module(
        "p69_loo_coarsen",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    p73 = load_module(
        "p73_loo_coarsen",
        PIPELINE / "73_h2_market_share_category_coarsening.py")
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
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)

    inclusive_reference = pd.read_csv(INCLUSIVE_REFERENCE)
    inclusive_reference = inclusive_reference.loc[
        inclusive_reference["panel"].eq("common 1980-2016 sample")]
    loo_reference = pd.read_csv(LOO_REFERENCE)
    if len(inclusive_reference) != len(PREFIX_DIGITS) or len(loo_reference) != 1:
        raise SystemExit("FAIL: reference result rows are incomplete")

    rows = []
    for digits in PREFIX_DIGITS:
        loo_wide, diagnostics, prefixes = build_loo(
            arr, categories, contract_years, years, pivots["P"], units,
            digits, p63)
        inclusive_wide, _, inclusive_prefixes = p73.build_instrument(
            arr, categories, contract_years, years, pivots["P"], units,
            digits, p63)
        if prefixes != inclusive_prefixes:
            raise SystemExit(
                f"FAIL: prefix {digits} group order differs across builders")
        z_loo = loo_wide[ri, ci]
        z_inclusive = inclusive_wide[ri, ci]
        if not np.isfinite(z_loo).all() or not np.isfinite(z_inclusive).all():
            raise SystemExit(
                f"FAIL: prefix {digits} has missing sample instruments")

        result_loo = p69.within_first_stage(
            x, z_loo, weight, fe, nofe, cluster, t28)
        result_inclusive = p69.within_first_stage(
            x, z_inclusive, weight, fe, nofe, cluster, t28)
        reference = inclusive_reference.loc[
            inclusive_reference["prefix_digits"].eq(digits)]
        if len(reference) != 1:
            raise SystemExit(
                f"FAIL: prefix {digits} inclusive reference is missing")
        p69.assert_fixture(
            f"inclusive prefix {digits}", result_inclusive,
            reference.iloc[0])
        if digits == 4:
            p69.assert_fixture(
                "leave-one-out prefix 4", result_loo,
                loo_reference.iloc[0])

        cluster_row, _ = p69.cluster_diagnostics(
            f"loo_prefix_{digits}", result_loo, cluster,
            baseline["year"].to_numpy(np.int64),
            baseline["ccg"].to_numpy(), t28)
        diagnostic_sample = diagnostics.loc[
            diagnostics["year"].isin(baseline["year"].unique())]
        maximum = diagnostic_sample.loc[
            diagnostic_sample["max_absolute_growth"].idxmax()]
        rows.append({
            "prefix_digits": digits,
            "category_level": (
                "four-character" if digits == 4
                else f"{digits}-character prefix"),
            "n_groups": len(prefixes),
            "horizon": H,
            "exposure_window": "t-11 to t-7",
            "tail_rule": "none",
            "sample_start": int(baseline["year"].min()),
            "sample_end": int(baseline["year"].max()),
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
            "first_stage": result_loo["beta"],
            "first_stage_se": result_loo["se"],
            "first_stage_t": result_loo["t"],
            "first_stage_F": result_loo["F"],
            "partial_r2": result_loo["partial_r2"],
            "instrument_sd": float(np.std(z_loo, ddof=1)),
            "within_instrument_sd": float(np.std(result_loo["zt"], ddof=1)),
            "inclusive_first_stage_F": result_inclusive["F"],
            "correlation_with_inclusive": float(np.corrcoef(
                z_loo, z_inclusive)[0, 1]),
            "positive_exposure_holes": int(
                diagnostic_sample["positive_exposure_holes"].sum()),
            "max_counties_with_holes": int(
                diagnostic_sample["counties_with_holes"].max()),
            "max_absolute_growth": float(maximum["max_absolute_growth"]),
            "max_growth_year": int(maximum["year"]),
            "max_growth_ccg": int(maximum["max_growth_ccg"]),
            "max_growth_group": maximum["max_growth_group"],
            "max_growth_rest_exposure": float(
                maximum["max_growth_rest_exposure"]),
            "max_growth_rest_shift": float(maximum["max_growth_rest_shift"]),
            "max_p99_absolute_growth": float(
                diagnostic_sample["p99_absolute_growth"].max()),
            "max_p999_absolute_growth": float(
                diagnostic_sample["p999_absolute_growth"].max()),
            "effective_counties_in_cluster_meat":
                cluster_row["effective_counties_in_cluster_meat"],
            "largest_county_meat_share":
                cluster_row["largest_county_meat_share"],
            "largest_county_meat_ccg":
                cluster_row["largest_county_meat_ccg"],
            "max_exposure_identity_relative_gap": float(
                diagnostic_sample[
                    "max_exposure_identity_relative_gap"].max()),
            "max_shift_identity_relative_gap": float(
                diagnostic_sample[
                    "max_shift_identity_relative_gap"].max()),
        })

    output = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    for row in rows:
        print(
            f"   {row['prefix_digits']}-character ({row['n_groups']} groups): "
            f"LOO F={row['first_stage_F']:.3f}, "
            f"inclusive F={row['inclusive_first_stage_F']:.3f}, "
            f"corr={row['correlation_with_inclusive']:+.4f}")
        print(
            f"      max |growth|={row['max_absolute_growth']:.3g}, "
            f"rest exposure=${row['max_growth_rest_exposure']:.2f}, "
            f"holes={row['positive_exposure_holes']:,}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
