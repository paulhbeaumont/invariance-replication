"""Estimate growth-tail treatments for the h=2 market-share instrument.

The category shifter is the two-year national procurement change divided by
the same five-year national exposure window that forms county market shares.
This producer leaves every other part of the county specification fixed and
winsorizes the category-year shifter either over the pooled distribution or
within target year.

Writes Results/instrument_normalization/
    h2_market_share_growth_tail_grid.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_growth_tail_grid.csv
# mpgr-new-estimates: 7
# mpgr-variation-scope: h2|market-share|growth-tail|first-stage-and-2sls

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
TAILS = (0.005, 0.01, 0.025)
OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
       / "h2_market_share_growth_tail_grid.csv")
REFERENCE = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
             / "h2_county_exposure_window_market_share_shifter.csv")


def load_module(name, path):
    """Load one project producer without invoking its command-line entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_growth_inputs(arr, contract_years, target_years, population):
    """Build raw national rates and county exposure ingredients by year."""
    national = arr.sum(axis=0, dtype=np.float64)
    rates = np.full((arr.shape[1], len(target_years)), np.nan, np.float64)
    exposures = []
    denominators = []
    for j, year in enumerate(target_years):
        exposure_hi = year - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        required = (exposure_lo, exposure_hi, year - H, year - 1, year)
        missing = [value for value in required if value not in contract_years]
        if missing:
            raise SystemExit(
                f"FAIL: target year {year} requires missing years {missing}")
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = exposure.sum(axis=0, dtype=np.float64)
        live = np.isfinite(national_exposure) & (national_exposure > 0)
        if not live.any() or np.any(exposure[:, ~live] != 0):
            raise SystemExit(f"FAIL: invalid category support in {year}")
        post = 0.5 * (
            national[:, contract_years.index(year - 1)]
            + national[:, contract_years.index(year)])
        base = national[:, contract_years.index(year - H)]
        rates[live, j] = (post[live] - base[live]) / national_exposure[live]
        denominator = population[year - H].to_numpy(np.float64)
        if not np.isfinite(denominator).all() or (denominator <= 0).any():
            raise SystemExit(f"FAIL: population denominator is invalid in {year}")
        exposures.append(exposure)
        denominators.append(denominator)
    if not np.isfinite(rates[np.isfinite(rates)]).all():
        raise SystemExit("FAIL: category growth rates are invalid")
    return rates, exposures, denominators


def transformed_rates(raw, scope, tail):
    """Winsorize live category-year rates and return threshold diagnostics."""
    result = raw.copy()
    live = np.isfinite(raw)
    lower_values = []
    upper_values = []
    if scope == "pooled":
        lower, upper = np.quantile(
            raw[live], [tail, 1.0 - tail], method="linear")
        result[live] = np.clip(raw[live], lower, upper)
        lower_values.append(float(lower))
        upper_values.append(float(upper))
    elif scope == "within_year":
        for j in range(raw.shape[1]):
            year_live = live[:, j]
            if not year_live.any():
                raise SystemExit(f"FAIL: no live growth rates in column {j}")
            lower, upper = np.quantile(
                raw[year_live, j], [tail, 1.0 - tail], method="linear")
            result[year_live, j] = np.clip(
                raw[year_live, j], lower, upper)
            lower_values.append(float(lower))
            upper_values.append(float(upper))
    else:
        raise SystemExit(f"FAIL: unknown winsorization scope {scope}")
    changed = live & ~np.isclose(result, raw, rtol=0.0, atol=0.0)
    return result, {
        "modified_category_years": int(changed.sum()),
        "modified_share": float(changed.sum() / live.sum()),
        "lower_threshold_min": min(lower_values),
        "lower_threshold_max": max(lower_values),
        "upper_threshold_min": min(upper_values),
        "upper_threshold_max": max(upper_values),
    }


def county_instrument(rates, exposures, denominators):
    """Combine county exposure with transformed national category rates."""
    result = np.full((exposures[0].shape[0], rates.shape[1]), np.nan)
    for j, exposure in enumerate(exposures):
        live = np.isfinite(rates[:, j])
        result[:, j] = (
            exposure[:, live] @ rates[live, j] / denominators[j])
    if not np.isfinite(result).all():
        raise SystemExit("FAIL: transformed county instrument is incomplete")
    return result


def main():
    """Run the pooled and within-year winsorization grid."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 market-share category-growth tail grid")
    p57 = load_module(
        "p57_growth_tail", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_growth_tail", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_growth_tail",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_growth_tail",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_growth_tail",
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
    units2, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    target_years = store.target_years(H)
    raw_rates, exposures, denominators = build_growth_inputs(
        arr, contract_years, target_years, pivots["P"])
    raw_wide = county_instrument(raw_rates, exposures, denominators)
    reference_wide, _ = p68.build_instrument(
        arr, contract_years, target_years, pivots["P"])
    gap = float(np.nanmax(np.abs(raw_wide - reference_wide)))
    scale = max(float(np.nanmax(np.abs(reference_wide))), 1.0)
    if gap / scale > 1e-12:
        raise SystemExit(
            f"FAIL: raw instrument tie-out relative gap is {gap / scale:.3e}")

    rowpos = {unit: index for index, unit in enumerate(units)}
    colpos = {year: index for index, year in enumerate(target_years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    y = baseline[locked.OUTCOME].to_numpy(np.float64)
    x = baseline["x"].to_numpy(np.float64)
    _, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)

    variants = [("none", "none", np.nan, raw_rates, {
        "modified_category_years": 0,
        "modified_share": 0.0,
        "lower_threshold_min": np.nan,
        "lower_threshold_max": np.nan,
        "upper_threshold_min": np.nan,
        "upper_threshold_max": np.nan,
    })]
    for scope in ("pooled", "within_year"):
        for tail in TAILS:
            rates, diagnostics = transformed_rates(raw_rates, scope, tail)
            variants.append((
                f"winsor_{scope}_{100 * tail:g}_{100 * (1 - tail):g}",
                scope, tail, rates, diagnostics))

    rows = []
    reference = pd.read_csv(REFERENCE)
    if len(reference) != 1:
        raise SystemExit("FAIL: first-stage reference is incomplete")
    for name, scope, tail, rates, diagnostics in variants:
        wide = raw_wide if name == "none" else county_instrument(
            rates, exposures, denominators)
        z = wide[ri, ci]
        first = p69.within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        second = t28.fit2sls(
            y, x, z, weight, fe, nofe, cluster,
            singleton=False, want_F=False)
        if name == "none" and not np.isclose(
                first["F"], float(reference.iloc[0]["first_stage_F"]),
                rtol=locked.RTOL_COEF):
            raise SystemExit("FAIL: unrestricted first-stage fixture drift")
        rows.append({
            "variant": name,
            "winsor_scope": scope,
            "lower_percentile": np.nan if name == "none" else 100 * tail,
            "upper_percentile": (
                np.nan if name == "none" else 100 * (1 - tail)),
            "quantile_method": "linear",
            "live_category_years": int(np.isfinite(raw_rates).sum()),
            **diagnostics,
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "sample_start": int(baseline["year"].min()),
            "sample_end": int(baseline["year"].max()),
            "first_stage": first["beta"],
            "first_stage_se": first["se"],
            "first_stage_t": first["t"],
            "first_stage_F": first["F"],
            "partial_r2": first["partial_r2"],
            "second_stage_beta": second["beta"],
            "second_stage_se": second["se"],
            "second_stage_t": second["beta"] / second["se"],
            "second_stage_beta_x1e5": second["beta"] * locked.SCALE,
            "second_stage_se_x1e5": second["se"] * locked.SCALE,
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
        })

    output = pd.DataFrame(rows)
    if len(output) != 7 or output["variant"].duplicated().any():
        raise SystemExit("FAIL: growth-tail grid is incomplete")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    for row in output.itertuples():
        print(
            f"   {row.variant}: modified={row.modified_category_years:,} "
            f"({100 * row.modified_share:.2f}%), "
            f"F={row.first_stage_F:.3f}, IV t={row.second_stage_t:.3f}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
