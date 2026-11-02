"""Compare unclipped h=2 market-share instruments across category detail.

The top-800 four-character FSC/PSC categories are kept as-is or collapsed to
three- and two-character prefixes before both county market shares and national
dollar changes are formed. Results are reported for the contract-feasible
1977-2016 sample and the 1980-2016 common sample. All rows use the same
five-year exposure window, division-by-year fixed effects, equal county
weights, and county clusters.

Writes Results/instrument_normalization/
    h2_market_share_category_coarsening_first_stage.csv
    h2_market_share_category_coarsening_concentration.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_category_coarsening_first_stage.csv
# mpgr-new-estimates: 6
# mpgr-variation-scope: h2|market-share|prefix-4-3-2|sample-1977-2016-and-1980-2016|five-year-exposure|no-clip

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
TARGET_CCG = 48015
FOCUS_FSC = "2320"
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
RESULT_OUT = OUTDIR / "h2_market_share_category_coarsening_first_stage.csv"
CONCENTRATION_OUT = (
    OUTDIR / "h2_market_share_category_coarsening_concentration.csv")
IDENTITY_RTOL = 1e-12


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_instrument(
        arr, categories, contract_years, target_years, population, units,
        prefix_digits, p63):
    """Build one market-share instrument after prefix aggregation."""
    prefixes, assignment, mapping = p63.prefix_mapping(
        categories, prefix_digits)
    national = p63.grouped_national(arr, assignment, len(prefixes))
    result = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    diagnostics = []
    focus_prefix = FOCUS_FSC[:prefix_digits]
    focus = prefixes.index(focus_prefix)
    target_positions = np.flatnonzero(units == TARGET_CCG)
    if len(target_positions) != 1:
        raise SystemExit(
            f"FAIL: county axis has {len(target_positions)} Austin rows")
    target = int(target_positions[0])

    for j, t in enumerate(target_years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.FIRST_YEAR:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure4 = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        exposure = exposure4 @ mapping
        national_exposure = exposure.sum(axis=0, dtype=np.float64)
        live = np.isfinite(national_exposure) & (national_exposure > 0)
        if not live.any():
            raise SystemExit(
                f"FAIL: prefix {prefix_digits}, target {t} has no live groups")

        post2 = 0.5 * (
            national[:, contract_years.index(t)]
            + national[:, contract_years.index(t - 1)])
        base = national[:, contract_years.index(t - H)]
        national_shift = post2 - base
        shares = np.zeros_like(exposure)
        np.divide(
            exposure, national_exposure[None, :], out=shares,
            where=live[None, :])
        category_share_sums = shares[:, live].sum(axis=0)
        share_sum_error = float(np.max(np.abs(category_share_sums - 1.0)))
        if share_sum_error > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: prefix {prefix_digits}, target {t} share-sum error is "
                f"{share_sum_error:.3e}")

        county_category_prediction = shares[:, live] * national_shift[live]
        allocation_error = float(np.max(np.abs(
            county_category_prediction.sum(axis=0) - national_shift[live])))
        allocation_scale = max(
            float(np.max(np.abs(national_shift[live]))), 1.0)
        allocation_relative_error = allocation_error / allocation_scale
        if allocation_relative_error > IDENTITY_RTOL:
            raise SystemExit(
                f"FAIL: prefix {prefix_digits}, target {t} allocation "
                f"relative error is {allocation_relative_error:.3e}")

        predicted_dollars = county_category_prediction.sum(axis=1)
        pbase = population[t - H].to_numpy(np.float64)
        if np.any(~np.isfinite(pbase)) or np.any(pbase <= 0):
            raise SystemExit(
                f"FAIL: prefix {prefix_digits}, target {t} has invalid population")
        result[:, j] = predicted_dollars / pbase

        hhi = np.square(shares[:, live]).sum(axis=0)
        dollar_weights = national_exposure[live] / national_exposure[live].sum()
        weighted_hhi = float(dollar_weights @ hhi)
        max_unit, max_group_live = np.unravel_index(
            np.argmax(shares[:, live]), shares[:, live].shape)
        live_positions = np.flatnonzero(live)
        max_group = int(live_positions[max_group_live])
        focus_shares = shares[:, focus]
        diagnostics.append({
            "prefix_digits": prefix_digits,
            "category_level": (
                "four-character" if prefix_digits == 4
                else f"{prefix_digits}-character prefix"),
            "n_groups": len(prefixes),
            "year": int(t),
            "exposure_start": int(exposure_lo),
            "exposure_end": int(exposure_hi),
            "live_groups": int(live.sum()),
            "max_county_group_share": float(shares[max_unit, max_group]),
            "max_share_ccg": int(units[max_unit]),
            "max_share_group": prefixes[max_group],
            "exposure_dollar_weighted_hhi": weighted_hhi,
            "exposure_dollar_weighted_effective_counties": 1.0 / weighted_hhi,
            "austin_focus_group": focus_prefix,
            "austin_focus_group_share": float(focus_shares[target]),
            "focus_group_hhi": float(np.square(focus_shares).sum()),
            "focus_group_effective_counties":
                float(1.0 / np.square(focus_shares).sum()),
            "focus_group_austin_absolute_rank":
                int(np.sum(focus_shares > focus_shares[target]) + 1),
            "max_category_share_sum_error": share_sum_error,
            "max_national_allocation_relative_error":
                allocation_relative_error,
        })

    return result, pd.DataFrame(diagnostics), prefixes


def build_extended_sample(county_panel, units):
    """Build the h=2 county treatment panel for target years 1977-2016."""
    years = store.target_years(H)
    panel_units, pivots = store.load_pivots(
        county_panel, cols=("P", "E"))
    if not np.array_equal(units, panel_units):
        raise SystemExit("FAIL: county order differs within persisted stores")
    contracts = store.load_con(locked.RULE).reindex(index=units)

    base = contracts.reindex(columns=[t - H for t in years]).to_numpy(
        np.float64)
    post = 0.5 * (
        contracts.reindex(columns=years).to_numpy(np.float64)
        + contracts.reindex(columns=[t - 1 for t in years]).to_numpy(
            np.float64))
    population = pivots["P"].reindex(
        columns=[t - H for t in years]).to_numpy(np.float64)
    employment_base = pivots["E"].reindex(
        columns=[t - H for t in years]).to_numpy(np.float64)
    employment_end = pivots["E"].reindex(
        columns=years).to_numpy(np.float64)
    treatment = (post - base) / population

    valid = (
        np.isfinite(treatment)
        & np.isfinite(employment_base)
        & np.isfinite(employment_end)
        & (employment_base > 0)
        & (employment_end > 0))
    if not valid.all():
        raise SystemExit(
            f"FAIL: extended sample has {int((~valid).sum()):,} invalid "
            "county-years")
    ri, ci = np.where(valid)
    division = (county_panel.drop_duplicates("ccg").set_index("ccg")
                ["division"].reindex(units))
    baseline = pd.DataFrame({
        "ccg": units[ri],
        "year": np.asarray(years, dtype=np.int64)[ci],
        "x": treatment[valid],
        "division": division.to_numpy()[ri],
    })
    if baseline["division"].isna().any():
        raise SystemExit("FAIL: extended sample has missing divisions")
    counts = baseline.groupby("year", sort=False)["ccg"].nunique()
    if len(counts) != len(years) or not (counts == len(units)).all():
        raise SystemExit("FAIL: extended county-year sample is not balanced")
    return baseline.reset_index(drop=True)


def main():
    """Run the category-coarsening first-stage comparison."""
    start = time.time()
    print("== h=2 market-share category coarsening")
    p57 = load_module(
        "p57_market_coarsen", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_market_coarsen", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_market_coarsen",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p63 = load_module(
        "p63_market_coarsen",
        PIPELINE / "63_h2_prefix3_category_shifter.py")
    p69 = load_module(
        "p69_market_coarsen",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    common_reference, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(common_reference) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: common reference has {len(common_reference):,} rows and "
            f"{dropped} drops")

    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    extended = build_extended_sample(county_panel, units)
    common = extended.loc[extended["year"] >= locked.DATA_FIRST + 11]
    common = common.reset_index(drop=True)
    reference_columns = ["ccg", "year", "x", "division"]
    if not common[reference_columns[:2]].equals(
            common_reference[reference_columns[:2]]):
        raise SystemExit("FAIL: common-sample county-year keys changed")
    if not np.allclose(
            common["x"], common_reference["x"], rtol=1e-13, atol=1e-15):
        raise SystemExit("FAIL: common-sample county treatment changed")
    if not common["division"].equals(common_reference["division"]):
        raise SystemExit("FAIL: common-sample division labels changed")

    instruments, concentration_parts, group_counts = {}, [], {}
    for digits in PREFIX_DIGITS:
        instrument, concentration, prefixes = build_instrument(
            arr, categories, contract_years, years, pivots["P"], units,
            digits, p63)
        instruments[digits] = instrument
        concentration_parts.append(concentration)
        group_counts[digits] = len(prefixes)

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    samples = (
        ("extended 1977-2016 sample", extended),
        ("common 1980-2016 sample", common),
    )
    rows = []
    for panel_label, baseline in samples:
        ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
        ci = baseline["year"].map(colpos).to_numpy(np.int64)
        zsample = {
            digits: instrument[ri, ci]
            for digits, instrument in instruments.items()
        }
        if any(not np.isfinite(z).all() for z in zsample.values()):
            raise SystemExit(
                f"FAIL: {panel_label} has missing instrument values")

        x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
        results = {
            digits: p69.within_first_stage(
                x, z, weight, fe, nofe, cluster, t28)
            for digits, z in zsample.items()
        }
        if panel_label == "common 1980-2016 sample":
            p69.assert_fixture(
                "market_share", results[4],
                p69.saved_results()["market_share"])

        sample_years = baseline["year"].to_numpy(np.int64)
        county_ids = baseline["ccg"].to_numpy()
        target = county_ids == TARGET_CCG
        target_codes = np.unique(cluster[target])
        if len(target_codes) != 1:
            raise SystemExit("FAIL: Austin County maps to multiple clusters")
        target_code = int(target_codes[0])

        for digits in PREFIX_DIGITS:
            result = results[digits]
            cluster_row, _ = p69.cluster_diagnostics(
                f"{panel_label}_prefix_{digits}", result, cluster,
                sample_years, county_ids, t28)
            target_signal = float(
                result["zt"][target] @ result["xt"][target])
            target_zz = float(
                result["zt"][target] @ result["zt"][target])
            target_score = float(result["score_cluster"][target_code])
            rows.append({
                "panel": panel_label,
                "prefix_digits": digits,
                "category_level": (
                    "four-character" if digits == 4
                    else f"{digits}-character prefix"),
                "n_four_character_categories": len(categories),
                "n_groups": group_counts[digits],
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
                "first_stage": result["beta"],
                "first_stage_se": result["se"],
                "first_stage_t": result["t"],
                "first_stage_F": result["F"],
                "partial_r2": result["partial_r2"],
                "instrument_sd": float(np.std(zsample[digits], ddof=1)),
                "within_instrument_sd": float(
                    np.std(result["zt"], ddof=1)),
                "correlation_with_four_character": float(np.corrcoef(
                    zsample[digits], zsample[4])[0, 1]),
                "cross_year_share_of_cluster_meat":
                    cluster_row["cross_year_share_of_cluster_meat"],
                "effective_counties_in_cluster_meat":
                    cluster_row["effective_counties_in_cluster_meat"],
                "largest_county_meat_share":
                    cluster_row["largest_county_meat_share"],
                "largest_county_meat_ccg":
                    cluster_row["largest_county_meat_ccg"],
                "austin_share_of_total_within_zx":
                    target_signal / result["zx"],
                "austin_share_of_total_within_z_sum_squares":
                    target_zz / result["zz"],
                "austin_share_of_cluster_meat":
                    target_score ** 2 / result["meat"],
            })

    results_frame = pd.DataFrame(rows)
    concentration = pd.concat(concentration_parts, ignore_index=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(RESULT_OUT, index=False)
    concentration.to_csv(CONCENTRATION_OUT, index=False)

    for panel_label, panel_rows in results_frame.groupby("panel", sort=False):
        print(f"   {panel_label}: "
              f"{int(panel_rows.iloc[0]['n']):,} county-years, "
              f"{int(panel_rows.iloc[0]['n_counties']):,} counties")
        for _, row in panel_rows.iterrows():
            print(
                f"      {int(row['prefix_digits'])}-character "
                f"({int(row['n_groups'])} groups): "
                f"pi={row['first_stage']:+.6f}, "
                f"SE={row['first_stage_se']:.6f}, "
                f"F={row['first_stage_F']:.3f}, "
                f"partial R2={row['partial_r2']:.6f}, "
                f"Austin variance={100 * row['austin_share_of_total_within_z_sum_squares']:.2f}%, "
                f"Austin meat={100 * row['austin_share_of_cluster_meat']:.2f}%")
    print(f"   wrote {RESULT_OUT}")
    print(f"   wrote {CONCENTRATION_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
