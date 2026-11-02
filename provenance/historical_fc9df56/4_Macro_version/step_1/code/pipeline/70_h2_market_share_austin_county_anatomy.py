"""Trace Austin County's role in the h=2 market-share first stage.

The script decomposes Austin County's clustered score by year and procurement
category, separates its contribution to first-stage signal and instrument
variation, and re-estimates the unchanged first stage after excluding the
county.

Writes Results/instrument_normalization/
    h2_market_share_austin_county_summary.csv
    h2_market_share_austin_county_year.csv
    h2_market_share_austin_county_category.csv
    h2_market_share_austin_county_leaveout.csv
    h2_market_share_austin_category_2320_contractors.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_austin_county_summary.csv
# mpgr-new-estimates: 2
# mpgr-variation-scope: h2|market-share-vs-locked|austin-county-anatomy

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
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
SUMMARY_OUT = OUTDIR / "h2_market_share_austin_county_summary.csv"
YEAR_OUT = OUTDIR / "h2_market_share_austin_county_year.csv"
CATEGORY_OUT = OUTDIR / "h2_market_share_austin_county_category.csv"
LEAVEOUT_OUT = OUTDIR / "h2_market_share_austin_county_leaveout.csv"
CONTRACTOR_OUT = OUTDIR / "h2_market_share_austin_category_2320_contractors.csv"
TOL = 1e-10


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descending_absolute_rank(values, position):
    """Return the one-based descending absolute rank of one array element."""
    target = abs(float(values[position]))
    return int(np.sum(np.abs(values) > target) + 1)


def county_identity():
    """Resolve the county name and metropolitan area from project crosswalks."""
    county = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    row = county.loc[county["ccg_id"].astype(str).eq(str(TARGET_CCG))]
    if len(row) != 1:
        raise SystemExit(
            f"FAIL: county crosswalk has {len(row)} rows for {TARGET_CCG}")
    metro = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_cbsa_2003.csv")
    mrow = metro.loc[metro["county_fips5"].eq(TARGET_CCG)]
    if len(mrow) != 1:
        raise SystemExit(
            f"FAIL: CBSA crosswalk has {len(mrow)} rows for {TARGET_CCG}")
    return str(row.iloc[0]["ccg_name"]), str(mrow.iloc[0]["cbsa_title"])


def focus_category_contractors():
    """Aggregate source records for Austin County's category 2320."""
    import duckdb

    paths = [str(path) for path in sorted(dc.OUT_DIR.glob("fy*.parquet"))]
    if not paths:
        raise SystemExit("FAIL: no DCADS fiscal-year files are available")
    connection = duckdb.connect()
    result = connection.execute("""
        SELECT
            fiscal_year,
            city_name,
            county_name,
            contractor_name,
            fsc_description,
            COUNT(*) AS contract_actions,
            SUM(CASE WHEN dollars_unit = 'thousands'
                     THEN dollars * 1000.0 ELSE dollars END) AS net_dollars,
            SUM(ABS(CASE WHEN dollars_unit = 'thousands'
                         THEN dollars * 1000.0 ELSE dollars END))
                AS absolute_dollars
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(county_fips5 AS VARCHAR) = ?
          AND SUBSTR(UPPER(TRIM(CAST(fsc AS VARCHAR))), 1, 4) = '2320'
        GROUP BY ALL
        ORDER BY fiscal_year, net_dollars DESC
    """, [paths, str(TARGET_CCG)]).df()
    connection.close()
    if result.empty:
        raise SystemExit("FAIL: no Austin County category 2320 records found")
    return result


def category_components(
        arr, categories, contract_years, baseline, ri, population, g_locked,
        results):
    """Decompose Austin County's raw and within instruments by category."""
    national = arr.sum(axis=0, dtype=np.float64)
    names = tuple(results)
    aggregate = {
        name: {
            "raw_sum": np.zeros(len(categories), np.float64),
            "raw_absolute_sum": np.zeros(len(categories), np.float64),
            "within_sum": np.zeros(len(categories), np.float64),
            "within_absolute_sum": np.zeros(len(categories), np.float64),
            "signal": np.zeros(len(categories), np.float64),
            "score": np.zeros(len(categories), np.float64),
        } for name in names
    }
    year_rows = []
    years = baseline["year"].to_numpy(np.int64)
    divisions = baseline["division"].to_numpy()
    focus_category = "2320"
    if focus_category not in categories:
        raise SystemExit("FAIL: category 2320 is absent from the tensor")
    focus = categories.index(focus_category)

    for t in sorted(np.unique(years)):
        sample = np.flatnonzero(years == t)
        unit_rows = ri[sample]
        target_positions = np.flatnonzero(
            baseline.iloc[sample]["ccg"].to_numpy() == TARGET_CCG)
        if len(target_positions) != 1:
            raise SystemExit(
                f"FAIL: year {t} has {len(target_positions)} Austin rows")
        target = int(target_positions[0])

        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[unit_rows, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = arr[:, :, a:b + 1].mean(
            axis=2).sum(axis=0, dtype=np.float64)
        live = national_exposure > 0

        post2 = 0.5 * (
            national[:, contract_years.index(t)]
            + national[:, contract_years.index(t - 1)])
        base = national[:, contract_years.index(t - H)]
        g_market = np.zeros(len(categories), np.float64)
        g_market[live] = (
            post2[live] - base[live]) / national_exposure[live]
        g_lock_t = g_locked[:, contract_years.index(t)]
        if not np.isfinite(g_lock_t).all():
            raise SystemExit(f"FAIL: locked shifter is missing in {t}")

        pbase = population[t - H].to_numpy(np.float64)[unit_rows]
        scale = exposure / pbase[:, None]
        components = {
            "market_share": scale * g_market[None, :],
            "locked": scale * g_lock_t[None, :],
        }
        division_t = divisions[sample]
        focus_values = {}
        for name, component in components.items():
            component_within = component.copy()
            for division in np.unique(division_t):
                positions = division_t == division
                component_within[positions] -= component[positions].mean(axis=0)

            sample_position = sample[target]
            raw_target = component[target]
            within_target = component_within[target]
            if not np.isclose(
                    raw_target.sum(), results[name]["z_raw"][sample_position],
                    rtol=TOL, atol=1e-9):
                raise SystemExit(
                    f"FAIL: {name} raw category identity failed in {t}")
            if not np.isclose(
                    within_target.sum(), results[name]["zt"][sample_position],
                    rtol=TOL, atol=1e-9):
                raise SystemExit(
                    f"FAIL: {name} within category identity failed in {t}")

            xt = results[name]["xt"][sample_position]
            residual = results[name]["residual"][sample_position]
            signal = within_target * xt
            score = within_target * residual
            values = aggregate[name]
            values["raw_sum"] += raw_target
            values["raw_absolute_sum"] += np.abs(raw_target)
            values["within_sum"] += within_target
            values["within_absolute_sum"] += np.abs(within_target)
            values["signal"] += signal
            values["score"] += score
            focus_values[name] = {
                "raw": float(raw_target[focus]),
                "within": float(within_target[focus]),
                "signal": float(signal[focus]),
                "score": float(score[focus]),
            }

        market = results["market_share"]
        locked_result = results["locked"]
        pos = sample[target]
        exposure_target = exposure[target]
        p_target = pbase[target]
        row = {
            "ccg": TARGET_CCG,
            "county": "Austin County",
            "state": "Texas",
            "year": int(t),
            "division": divisions[pos],
            "population_t_minus_2": p_target,
            "mean_procurement_exposure_dollars": float(exposure_target.sum()),
            "mean_procurement_exposure_per_capita":
                float(exposure_target.sum() / p_target),
            "category_2320_exposure_dollars": float(exposure_target[focus]),
            "category_2320_exposure_per_capita":
                float(exposure_target[focus] / p_target),
            "category_2320_national_exposure_dollars":
                float(national_exposure[focus]),
            "category_2320_national_shift_dollars":
                float(post2[focus] - base[focus]),
            "category_2320_market_share_shifter": float(g_market[focus]),
            "category_2320_locked_shifter": float(g_lock_t[focus]),
            "county_procurement_change_per_capita":
                float(baseline.iloc[pos]["x"]),
            "county_procurement_change_per_capita_within":
                float(market["xt"][pos]),
        }
        for name, result in results.items():
            prefix = name
            row[f"{prefix}_instrument"] = float(result["z_raw"][pos])
            row[f"{prefix}_instrument_within"] = float(result["zt"][pos])
            row[f"{prefix}_first_stage_residual"] = float(
                result["residual"][pos])
            row[f"{prefix}_signal_contribution"] = float(
                result["zt"][pos] * result["xt"][pos])
            row[f"{prefix}_cluster_score_contribution"] = float(
                result["score_obs"][pos])
            row[f"{prefix}_instrument_absolute_rank_in_year"] = (
                descending_absolute_rank(result["z_raw"][sample], target))
            row[f"{prefix}_within_instrument_absolute_rank_in_year"] = (
                descending_absolute_rank(result["zt"][sample], target))
            row[f"{prefix}_score_absolute_rank_in_year"] = (
                descending_absolute_rank(result["score_obs"][sample], target))
            row[f"{prefix}_category_2320_raw_contribution"] = (
                focus_values[name]["raw"])
            row[f"{prefix}_category_2320_within_contribution"] = (
                focus_values[name]["within"])
            row[f"{prefix}_category_2320_signal_contribution"] = (
                focus_values[name]["signal"])
            row[f"{prefix}_category_2320_score_contribution"] = (
                focus_values[name]["score"])
        year_rows.append(row)

    category_rows = []
    for name, values in aggregate.items():
        total_score = float(values["score"].sum())
        cluster_code = int(np.unique(results[name]["cluster_target_code"])[0])
        expected_score = float(results[name]["score_cluster"][cluster_code])
        if not np.isclose(total_score, expected_score, rtol=TOL, atol=1e-5):
            raise SystemExit(
                f"FAIL: {name} Austin category scores sum to {total_score}, "
                f"expected {expected_score}")
        absolute_score = float(np.abs(values["score"]).sum())
        order = np.argsort(-np.abs(values["score"]))
        ranks = np.empty(len(categories), np.int64)
        ranks[order] = np.arange(1, len(categories) + 1)
        for k, category in enumerate(categories):
            category_rows.append({
                "instrument": name,
                "ccg": TARGET_CCG,
                "category": category,
                "raw_instrument_contribution_sum_over_years":
                    float(values["raw_sum"][k]),
                "raw_instrument_absolute_contribution_sum_over_years":
                    float(values["raw_absolute_sum"][k]),
                "within_instrument_contribution_sum_over_years":
                    float(values["within_sum"][k]),
                "within_instrument_absolute_contribution_sum_over_years":
                    float(values["within_absolute_sum"][k]),
                "first_stage_signal_contribution": float(values["signal"][k]),
                "cluster_score_contribution": float(values["score"][k]),
                "signed_share_of_austin_cluster_score":
                    float(values["score"][k] / total_score),
                "absolute_share_of_absolute_austin_cluster_scores":
                    float(abs(values["score"][k]) / absolute_score),
                "absolute_cluster_score_rank": int(ranks[k]),
            })
    return pd.DataFrame(year_rows), pd.DataFrame(category_rows)


def main():
    """Produce the Austin County score anatomy and leave-county-out check."""
    start = time.time()
    print("== h=2 market-share Austin County anatomy")
    p57 = load_module(
        "p57_austin", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_austin", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_austin", PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_austin", PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_austin", PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    county_name, metro_name = county_identity()
    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")
    target_mask = baseline["ccg"].eq(TARGET_CCG).to_numpy()
    if int(target_mask.sum()) != baseline["year"].nunique():
        raise SystemExit(
            f"FAIL: Austin County has {int(target_mask.sum())} rows")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    market_wide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    locked_wide = store.load_instrument(H, units, "z")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    zraw = {
        "market_share": market_wide[ri, ci],
        "locked": locked_wide[ri, ci],
    }
    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    fixtures = p69.saved_results()
    results = {}
    target_cluster_code = np.unique(cluster[target_mask])
    if len(target_cluster_code) != 1:
        raise SystemExit("FAIL: Austin County maps to multiple clusters")
    for name, z in zraw.items():
        result = p69.within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        p69.assert_fixture(name, result, fixtures[name])
        result["z_raw"] = z
        result["cluster_target_code"] = target_cluster_code
        results[name] = result

    g_locked = store.load_gk(H, categories, contract_years)
    year, category = category_components(
        arr, categories, contract_years, baseline, ri, pivots["P"],
        g_locked, results)
    contractors = focus_category_contractors()

    summary_rows = []
    leaveout_rows = []
    for name, result in results.items():
        code = int(target_cluster_code[0])
        target_signal = float(
            result["zt"][target_mask] @ result["xt"][target_mask])
        target_zz = float(result["zt"][target_mask] @ result["zt"][target_mask])
        score = float(result["score_cluster"][code])
        meat_share = float(score ** 2 / result["meat"])
        summary_rows.append({
            "instrument": name,
            "ccg": TARGET_CCG,
            "county": county_name,
            "state": "Texas",
            "metropolitan_area": metro_name,
            "sample_years": int(target_mask.sum()),
            "first_stage": result["beta"],
            "first_stage_se": result["se"],
            "first_stage_F": result["F"],
            "partial_r2": result["partial_r2"],
            "austin_within_zx_contribution": target_signal,
            "austin_share_of_total_within_zx": target_signal / result["zx"],
            "austin_within_z_sum_squares": target_zz,
            "austin_share_of_total_within_z_sum_squares":
                target_zz / result["zz"],
            "austin_cluster_score": score,
            "austin_share_of_cluster_meat": meat_share,
        })
        leaveout_rows.append({
            "instrument": name,
            "sample": "all counties",
            "excluded_ccg": "none",
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "first_stage": result["beta"],
            "first_stage_se": result["se"],
            "first_stage_t": result["t"],
            "first_stage_F": result["F"],
            "partial_r2": result["partial_r2"],
        })

        keep = ~target_mask
        reduced = baseline.loc[keep]
        xr, wr, fer, nofer, clusterr = helper.estimation_arrays(reduced)
        reduced_result = p69.within_first_stage(
            xr, zraw[name][keep], wr, fer, nofer, clusterr, t28)
        leaveout_rows.append({
            "instrument": name,
            "sample": "excluding Austin County, Texas",
            "excluded_ccg": TARGET_CCG,
            "n": len(reduced),
            "n_counties": int(reduced["ccg"].nunique()),
            "first_stage": reduced_result["beta"],
            "first_stage_se": reduced_result["se"],
            "first_stage_t": reduced_result["t"],
            "first_stage_F": reduced_result["F"],
            "partial_r2": reduced_result["partial_r2"],
        })

    summary = pd.DataFrame(summary_rows)
    leaveout = pd.DataFrame(leaveout_rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False)
    year.to_csv(YEAR_OUT, index=False)
    category.to_csv(CATEGORY_OUT, index=False)
    leaveout.to_csv(LEAVEOUT_OUT, index=False)
    contractors.to_csv(CONTRACTOR_OUT, index=False)

    print(f"   county: {county_name}, Texas; {metro_name}")
    for name in results:
        full = leaveout.loc[
            leaveout["instrument"].eq(name)
            & leaveout["sample"].eq("all counties")].iloc[0]
        drop = leaveout.loc[
            leaveout["instrument"].eq(name)
            & leaveout["sample"].str.startswith("excluding")].iloc[0]
        srow = summary.loc[summary["instrument"].eq(name)].iloc[0]
        top_years = year.assign(
            absolute_score=year[f"{name}_cluster_score_contribution"].abs()
        ).nlargest(3, "absolute_score")["year"].tolist()
        top_category = category.loc[
            category["instrument"].eq(name)].nsmallest(
                1, "absolute_cluster_score_rank").iloc[0]
        print(
            f"   {name}: Austin meat share="
            f"{100 * srow['austin_share_of_cluster_meat']:.2f}%, "
            f"signal share={100 * srow['austin_share_of_total_within_zx']:.2f}%, "
            f"F full={full['first_stage_F']:.3f}, "
            f"F excluding Austin={drop['first_stage_F']:.3f}, "
            f"top score years={top_years}, "
            f"top score category={top_category['category']}")
    print(f"   wrote {SUMMARY_OUT}")
    print(f"   wrote {YEAR_OUT}")
    print(f"   wrote {CATEGORY_OUT}")
    print(f"   wrote {LEAVEOUT_OUT}")
    print(f"   wrote {CONTRACTOR_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
