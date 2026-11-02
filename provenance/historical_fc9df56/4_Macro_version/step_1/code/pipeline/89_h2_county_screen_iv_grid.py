"""Consolidated h=2 first-stage and 2SLS county-screen grid.

Saved first-stage screens from producers 83-88 supply fixed county sets and
first-stage statistics. This producer also constructs pooled within-division
procurement shares and fixed upper-tail screens from raw procurement intensity
and the instrument's rolling exposure. It estimates the missing second stages
for every screen and records the rules in one comparison file. The instrument,
outcome, treatment, weights, fixed effects, and clustering remain fixed.

Writes Results/instrument_normalization/
    h2_county_screen_iv_grid.csv
    h2_county_screen_iv_exclusions.csv
    h2_county_exposure_screen_scores.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_county_screen_iv_grid.csv
# mpgr-new-estimates: 59
# mpgr-variation-scope: h2|market-share|county-screen-grid|first-stage-and-2sls

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
SCALE = locked.SCALE
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
OUT = OUTDIR / "h2_county_screen_iv_grid.csv"
EXCLUSIONS_OUT = OUTDIR / "h2_county_screen_iv_exclusions.csv"
SCORES_OUT = OUTDIR / "h2_county_exposure_screen_scores.csv"
REFERENCE = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
DIVISION_THRESHOLDS = (0.10, 0.05, 0.02, 0.01, 0.005)
UPPER_TAIL_SHARES = (0.001, 0.0025, 0.005, 0.01, 0.02)
SOURCE_FILES = {
    "instrument_leverage": OUTDIR / "h2_market_share_fixed_county_leverage_grid.csv",
    "national_procurement_share": OUTDIR / "h2_raw_procurement_county_share_grid.csv",
    "rolling_category_dominance": OUTDIR / "h2_raw_category_dominance_first_stage.csv",
    "county_category_cell_share": OUTDIR / "h2_pooled_county_category_cell_share_first_stage.csv",
    "procurement_intensity": OUTDIR / "h2_procurement_intensity_first_stage.csv",
}
FAMILY_META = {
    "instrument_leverage": {
        "classification": "diagnostic_only",
        "formula": "county share of residualized instrument sum of squares",
        "uses_population": False,
        "uses_constructed_instrument": True,
        "uses_time_varying_window": False,
    },
    "national_procurement_share": {
        "classification": "clean_raw_one_cutoff",
        "formula": "pooled county procurement divided by pooled national procurement",
        "uses_population": False,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
    },
    "rolling_category_dominance": {
        "classification": "raw_but_complex_window_maximum",
        "formula": "maximum county share within category across rolling five-year windows",
        "uses_population": False,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": True,
    },
    "county_category_cell_share": {
        "classification": "clean_raw_one_cutoff",
        "formula": "largest pooled county-category cell divided by pooled national procurement",
        "uses_population": False,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
    },
    "procurement_intensity": {
        "classification": "clean_full_sample_scale_one_cutoff",
        "formula": "1966-2018 average annual county procurement divided by 1975 population",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
    },
    "procurement_intensity_early15": {
        "classification": "clean_initial_scale_one_cutoff",
        "formula": "1966-1980 average annual county procurement divided by 1975 population",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
    },
    "rolling_exposure_data_start_early15": {
        "classification": "clean_exposure_tail_one_cutoff",
        "formula": "mean rolling exposure for the first 15 exposure vintages beginning with the 1966-1970 window",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": True,
    },
    "rolling_exposure_data_start_full": {
        "classification": "clean_exposure_tail_one_cutoff",
        "formula": "mean rolling exposure for all vintages beginning with the 1966-1970 window",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": True,
    },
    "rolling_exposure_estimation_start_early15": {
        "classification": "clean_exposure_tail_one_cutoff",
        "formula": "mean rolling exposure for the first 15 estimation-sample target years",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": True,
    },
    "rolling_exposure_estimation_start_full": {
        "classification": "clean_exposure_tail_one_cutoff",
        "formula": "mean rolling exposure for all estimation-sample target years",
        "uses_population": True,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": True,
    },
    "division_procurement_share": {
        "classification": "clean_raw_one_cutoff",
        "formula": "pooled county procurement divided by pooled procurement in its Census division",
        "uses_population": False,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
    },
}


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ccgs(value):
    """Read a pipe-delimited saved county set."""
    if pd.isna(value) or str(value).strip() == "":
        return tuple()
    return tuple(sorted(int(part) for part in str(value).split("|")))


def cutoff_text(value, suffix="%"):
    """Format a numeric cutoff without introducing false precision."""
    return f"{float(value):g}{suffix}"


def county_labels():
    """Map CCG identifiers to available county or stable CCG labels."""
    frame = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    frame["ccg"] = pd.to_numeric(frame["ccg_id"], errors="coerce")
    frame = frame.dropna(subset=["ccg"]).drop_duplicates("ccg")
    frame["ccg"] = frame["ccg"].astype(np.int64)
    return frame.set_index("ccg")["ccg_name"].to_dict()


def base_row(family, rule_id, cutoff, cutoff_label, excluded, source):
    """Create fields shared by saved and newly estimated screens."""
    meta = FAMILY_META[family]
    return {
        "rule_id": rule_id,
        "rule_family": family,
        "classification": meta["classification"],
        "rule_formula": meta["formula"],
        "cutoff": cutoff,
        "cutoff_label": cutoff_label,
        "uses_population": meta["uses_population"],
        "uses_constructed_instrument": meta["uses_constructed_instrument"],
        "uses_time_varying_window": meta["uses_time_varying_window"],
        "fixed_county_set": True,
        "source_first_stage_file": (
            source.name if isinstance(source, Path) else str(source)),
        "excluded_ccgs": "|".join(map(str, excluded)),
        "n_counties_excluded": len(excluded),
        "austin_excluded": int(48015 in excluded),
        "sagadahoc_excluded": int(23023 in excluded),
    }


def saved_rows():
    """Normalize the saved first-stage rule families."""
    rows = []
    for family, path in SOURCE_FILES.items():
        if not path.exists():
            raise SystemExit(f"FAIL: missing saved first-stage file {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise SystemExit(f"FAIL: empty saved first-stage file {path}")
        unrestricted = frame.loc[frame["n_counties_excluded"].eq(0)]
        if unrestricted.empty:
            raise SystemExit(f"FAIL: no unrestricted fixture in {path}")
        for value in unrestricted["first_stage_F"]:
            if not np.isclose(value, 16.35086173530424, rtol=1e-6):
                raise SystemExit(f"FAIL: unrestricted first stage drift in {path}")

        for _, item in frame.iterrows():
            parameter_name = (
                "upper_tail_share" if family == "procurement_intensity"
                else "threshold")
            if str(item[parameter_name]).strip().lower() == "none":
                continue
            excluded = parse_ccgs(item.get("excluded_ccgs"))
            if len(excluded) != int(item["n_counties_excluded"]):
                raise SystemExit(
                    f"FAIL: saved exclusion count mismatch in {path}")
            if family == "procurement_intensity":
                cutoff = float(item["upper_tail_percent"])
                label = f"top {cutoff_text(cutoff)} of counties"
                key = f"tail_{cutoff:g}pct"
                parameter_2 = np.nan
                parameter_2_label = ""
            else:
                cutoff = float(item["threshold_percent"])
                label = cutoff_text(cutoff)
                key = f"cap_{cutoff:g}pct"
                parameter_2 = np.nan
                parameter_2_label = ""
            rule_id = f"{family}__{key}"
            row = base_row(
                family, rule_id, cutoff, label, excluded, path)
            score_start = item.get(
                "procurement_sample_start", item.get("sample_start", np.nan))
            score_end = item.get(
                "procurement_sample_end", item.get("sample_end", np.nan))
            row.update({
                "secondary_cutoff": parameter_2,
                "secondary_cutoff_label": parameter_2_label,
                "score_period_start": score_start,
                "score_period_end": score_end,
                "score_period_index": (
                    "fiscal_year" if "procurement_sample_start" in item
                    else "estimation_year"),
                "score_cutoff_value": item.get("intensity_cutoff", np.nan),
                "n": int(item["n"]),
                "n_counties": int(item["n_counties"]),
                "years_per_county": int(item["years_per_county"]),
                "first_stage": float(item["first_stage"]),
                "first_stage_se": float(item["first_stage_se"]),
                "first_stage_t": float(item["first_stage_t"]),
                "first_stage_F": float(item["first_stage_F"]),
                "partial_r2": float(item["partial_r2"]),
            })
            rows.append(row)
    return rows


def division_rows(baseline, arr, units, p69, helper, t28, z):
    """Construct pooled within-division procurement-share screens."""
    pooled = arr.sum(axis=(1, 2), dtype=np.float64)
    unit_division = (baseline.drop_duplicates("ccg")
                     .set_index("ccg")["division"].reindex(units))
    if unit_division.isna().any():
        raise SystemExit("FAIL: division is missing for a county")
    totals = pd.Series(pooled).groupby(
        unit_division.to_numpy(), sort=False).transform("sum").to_numpy()
    shares = pooled / totals
    share_by_ccg = pd.Series(shares, index=units)
    rows = []
    for threshold in DIVISION_THRESHOLDS:
        excluded = tuple(sorted(
            share_by_ccg.index[share_by_ccg.to_numpy() > threshold].astype(int)))
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if not (counts == baseline["year"].nunique()).all():
            raise SystemExit("FAIL: division-share screen is unbalanced")
        xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
        first = p69.within_first_stage(
            xs, z[keep], ws, fes, nofes, clusters, t28)
        pct = 100.0 * threshold
        row = base_row(
            "division_procurement_share",
            f"division_procurement_share__cap_{pct:g}pct",
            pct, cutoff_text(pct), excluded, "computed_in_89")
        row.update({
            "secondary_cutoff": np.nan,
            "secondary_cutoff_label": "",
            "score_period_start": locked.FIRST_YEAR,
            "score_period_end": locked.LAST_YEAR,
            "score_period_index": "fiscal_year",
            "score_cutoff_value": threshold,
            "n": len(sample),
            "n_counties": int(sample["ccg"].nunique()),
            "years_per_county": int(sample["year"].nunique()),
            "first_stage": first["beta"],
            "first_stage_se": first["se"],
            "first_stage_t": first["t"],
            "first_stage_F": first["F"],
            "partial_r2": first["partial_r2"],
        })
        rows.append(row)
    return rows


def rolling_exposure(arr, contract_years, target_years, population):
    """Return the instrument's county exposure scale for selected vintages."""
    result = np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
    for j, year in enumerate(target_years):
        exposure_hi = year - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < contract_years[0] or exposure_hi > contract_years[-1]:
            raise SystemExit(
                f"FAIL: exposure window {exposure_lo}-{exposure_hi} is missing")
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        annual_procurement = arr[:, :, a:b + 1].mean(axis=2).sum(axis=1)
        population_year = year - H
        if population_year not in population:
            raise SystemExit(
                f"FAIL: population year {population_year} is missing")
        denominator = population[population_year].to_numpy(np.float64)
        if not np.isfinite(denominator).all() or (denominator <= 0).any():
            raise SystemExit(
                f"FAIL: population year {population_year} is invalid")
        result[:, j] = annual_procurement / denominator
    if not np.isfinite(result).all() or (result < 0).any():
        raise SystemExit("FAIL: rolling exposure score is invalid")
    return result


def upper_tail_rows(family, scores, score_years, score_period_index,
                    baseline, units, z, p69, helper, t28):
    """Estimate fixed upper-tail county screens for one exposure score."""
    if scores.shape != (len(units),):
        raise SystemExit(f"FAIL: score shape is invalid for {family}")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise SystemExit(f"FAIL: score values are invalid for {family}")
    order = np.lexsort((units, -scores))
    rows = []
    for upper_tail_share in UPPER_TAIL_SHARES:
        n_excluded = int(np.ceil(upper_tail_share * locked.N_COUNTIES))
        selected = units[order[:n_excluded]].astype(np.int64)
        excluded = tuple(sorted(selected.tolist()))
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if len(counts) != locked.N_COUNTIES - n_excluded:
            raise SystemExit(f"FAIL: retained county count differs for {family}")
        if not (counts == baseline["year"].nunique()).all():
            raise SystemExit(f"FAIL: {family} screen is unbalanced")
        xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
        first = p69.within_first_stage(
            xs, z[keep], ws, fes, nofes, clusters, t28)
        pct = 100.0 * upper_tail_share
        row = base_row(
            family, f"{family}__tail_{pct:g}pct", pct,
            f"top {cutoff_text(pct)} of counties", excluded,
            "computed_in_89")
        row.update({
            "secondary_cutoff": np.nan,
            "secondary_cutoff_label": "",
            "score_period_start": min(score_years),
            "score_period_end": max(score_years),
            "score_period_index": score_period_index,
            "score_cutoff_value": float(scores[order[n_excluded - 1]]),
            "n": len(sample),
            "n_counties": int(sample["ccg"].nunique()),
            "years_per_county": int(sample["year"].nunique()),
            "first_stage": first["beta"],
            "first_stage_se": first["se"],
            "first_stage_t": first["t"],
            "first_stage_F": first["F"],
            "partial_r2": first["partial_r2"],
        })
        rows.append(row)
    return rows


def main():
    """Build the consolidated first-stage and second-stage grid."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 consolidated county-screen IV grid")
    p57 = load_module(
        "p57_screen_iv", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_screen_iv", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_screen_iv",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_screen_iv",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_screen_iv",
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
    years = store.target_years(H)
    zwide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    z = zwide[ri, ci]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: market-share instrument has missing values")

    raw_early_years = list(range(1966, 1981))
    raw_early_positions = [contract_years.index(year)
                           for year in raw_early_years]
    population_1975 = pivots["P"][locked.POP_BASE_YEAR].to_numpy(np.float64)
    if not np.isfinite(population_1975).all() or (
            population_1975 <= 0).any():
        raise SystemExit("FAIL: 1975 population is invalid")
    raw_early_score = (
        arr[:, :, raw_early_positions].mean(axis=2).sum(axis=1)
        / population_1975)
    raw_full_score = arr.mean(axis=2).sum(axis=1) / population_1975
    raw_full_order = np.lexsort((units, -raw_full_score))
    intensity_reference = pd.read_csv(SOURCE_FILES["procurement_intensity"])
    for upper_tail_share in UPPER_TAIL_SHARES:
        reference_row = intensity_reference.loc[np.isclose(
            pd.to_numeric(
                intensity_reference["upper_tail_share"], errors="coerce"),
            upper_tail_share)]
        if len(reference_row) != 1:
            raise SystemExit("FAIL: full-sample intensity fixture is incomplete")
        n_excluded = int(np.ceil(upper_tail_share * locked.N_COUNTIES))
        calculated = tuple(sorted(
            units[raw_full_order[:n_excluded]].astype(np.int64).tolist()))
        expected = parse_ccgs(reference_row.iloc[0]["excluded_ccgs"])
        if calculated != expected:
            raise SystemExit("FAIL: full-sample intensity ranking drift")

    data_start_targets = list(range(1977, locked.TARGET_LAST + 1))
    exposure_by_vintage = rolling_exposure(
        arr, contract_years, data_start_targets, pivots["P"])
    vintage_position = {
        year: index for index, year in enumerate(data_start_targets)}
    data_start_early_targets = list(range(1977, 1992))
    estimation_early_targets = list(range(1980, 1995))
    estimation_full_targets = list(range(1980, locked.TARGET_LAST + 1))

    def mean_exposure(selected_years):
        positions = [vintage_position[year] for year in selected_years]
        return exposure_by_vintage[:, positions].mean(axis=1)

    exposure_scores = {
        "rolling_exposure_data_start_early15": (
            mean_exposure(data_start_early_targets),
            data_start_early_targets),
        "rolling_exposure_data_start_full": (
            mean_exposure(data_start_targets), data_start_targets),
        "rolling_exposure_estimation_start_early15": (
            mean_exposure(estimation_early_targets),
            estimation_early_targets),
        "rolling_exposure_estimation_start_full": (
            mean_exposure(estimation_full_targets),
            estimation_full_targets),
    }

    labels = county_labels()
    score_table = pd.DataFrame({
        "ccg": units,
        "county": [
            labels.get(int(ccg), f"CCG {int(ccg):05d}") for ccg in units],
        "procurement_intensity_early15": raw_early_score,
        "procurement_intensity_full": raw_full_score,
    })
    for family, (scores, _) in exposure_scores.items():
        score_table[family] = scores
    for column in [
            "procurement_intensity_early15",
            "procurement_intensity_full",
            *exposure_scores.keys()]:
        order = np.lexsort((units, -score_table[column].to_numpy()))
        rank = np.empty(len(units), dtype=np.int64)
        rank[order] = np.arange(1, len(units) + 1)
        score_table[f"{column}_rank"] = rank

    y = baseline[locked.OUTCOME].to_numpy(np.float64)
    x = baseline["x"].to_numpy(np.float64)
    weight, fe, nofe, cluster = helper.estimation_arrays(baseline)[1:]
    first_full = p69.within_first_stage(
        x, z, weight, fe, nofe, cluster, t28)
    reference = pd.read_csv(REFERENCE)
    if len(reference) != 1:
        raise SystemExit("FAIL: unrestricted first-stage reference is incomplete")
    if not np.isclose(
            first_full["F"], float(reference.iloc[0]["first_stage_F"]),
            rtol=locked.RTOL_COEF):
        raise SystemExit("FAIL: unrestricted first-stage fixture drift")

    rows = [{
        "rule_id": "none",
        "rule_family": "none",
        "classification": "baseline",
        "rule_formula": "no county exclusion",
        "cutoff": np.nan,
        "cutoff_label": "none",
        "secondary_cutoff": np.nan,
        "secondary_cutoff_label": "",
        "score_period_start": np.nan,
        "score_period_end": np.nan,
        "score_period_index": "",
        "score_cutoff_value": np.nan,
        "uses_population": False,
        "uses_constructed_instrument": False,
        "uses_time_varying_window": False,
        "fixed_county_set": True,
        "source_first_stage_file": REFERENCE.name,
        "excluded_ccgs": "",
        "n_counties_excluded": 0,
        "austin_excluded": 0,
        "sagadahoc_excluded": 0,
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "years_per_county": int(baseline["year"].nunique()),
        "first_stage": first_full["beta"],
        "first_stage_se": first_full["se"],
        "first_stage_t": first_full["t"],
        "first_stage_F": first_full["F"],
        "partial_r2": first_full["partial_r2"],
    }]
    rows.extend(saved_rows())
    rows.extend(division_rows(
        baseline, arr, units, p69, helper, t28, z))
    rows.extend(upper_tail_rows(
        "procurement_intensity_early15", raw_early_score,
        raw_early_years, "fiscal_year", baseline, units, z,
        p69, helper, t28))
    for family, (scores, score_years) in exposure_scores.items():
        rows.extend(upper_tail_rows(
            family, scores, score_years, "exposure_target_year",
            baseline, units, z, p69, helper, t28))

    fit_cache = {}
    exclusion_rows = []
    for row in rows:
        excluded = parse_ccgs(row["excluded_ccgs"])
        key = excluded
        if key not in fit_cache:
            keep = ~baseline["ccg"].isin(excluded).to_numpy()
            sample = baseline.loc[keep]
            counts = sample.groupby("ccg", sort=False)["year"].nunique()
            if len(counts) != row["n_counties"] or not (
                    counts == baseline["year"].nunique()).all():
                raise SystemExit(
                    f"FAIL: sample mismatch for {row['rule_id']}")
            _, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
            iv = t28.fit2sls(
                y[keep], x[keep], z[keep], ws, fes, nofes, clusters,
                singleton=False, want_F=False)
            fit_cache[key] = iv
        iv = fit_cache[key]
        if iv["n"] != row["n"]:
            raise SystemExit(f"FAIL: 2SLS N mismatch for {row['rule_id']}")
        row["second_stage_beta"] = iv["beta"]
        row["second_stage_se"] = iv["se"]
        row["second_stage_t"] = iv["beta"] / iv["se"]
        row["second_stage_beta_x1e5"] = iv["beta"] * SCALE
        row["second_stage_se_x1e5"] = iv["se"] * SCALE
        row["outcome"] = locked.OUTCOME
        row["treatment"] = "two-year average-window procurement change per capita"
        row["instrument"] = "h2 exposure-window market-share shifter"
        row["fixed_effects"] = "division-by-year"
        row["weight"] = "equal county"
        row["cluster"] = "county"
        for ccg in excluded:
            exclusion_rows.append({
                "rule_id": row["rule_id"],
                "rule_family": row["rule_family"],
                "ccg": ccg,
                "county": labels.get(ccg, f"CCG {ccg:05d}"),
            })

    output = pd.DataFrame(rows)
    if output["rule_id"].duplicated().any():
        duplicate = output.loc[output["rule_id"].duplicated(), "rule_id"].iloc[0]
        raise SystemExit(f"FAIL: duplicate rule id {duplicate}")
    if not output["years_per_county"].eq(37).all():
        raise SystemExit("FAIL: at least one screen is not balanced")
    if len(output) != 59:
        raise SystemExit(f"FAIL: grid has {len(output)} rows instead of 59")
    output = output.sort_values(
        ["classification", "rule_family", "n_counties_excluded", "rule_id"]
    ).reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows)
    if not exclusions.empty:
        exclusions = exclusions.sort_values(
            ["rule_family", "rule_id", "ccg"]).reset_index(drop=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    exclusions.to_csv(EXCLUSIONS_OUT, index=False)
    score_table.to_csv(SCORES_OUT, index=False)
    print(
        f"   rows={len(output)}, unique county sets={len(fit_cache)}, "
        f"second stages={len(fit_cache)}")
    for family, group in output.groupby("rule_family", sort=False):
        best = group.loc[group["first_stage_F"].idxmax()]
        print(
            f"   {family}: rows={len(group)}, best F={best['first_stage_F']:.3f}, "
            f"2SLS t={best['second_stage_t']:.3f}, "
            f"excluded={int(best['n_counties_excluded'])}")
    print(f"   wrote {OUT}")
    print(f"   wrote {EXCLUSIONS_OUT}")
    print(f"   wrote {SCORES_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
