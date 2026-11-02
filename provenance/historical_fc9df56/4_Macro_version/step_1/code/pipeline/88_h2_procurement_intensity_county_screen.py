"""First-stage sensitivity to fixed procurement-intensity county screens.

For each county, positive real procurement in the 800 instrument categories is
pooled over fiscal years 1966-2018 and divided by the number of fiscal years
and 1975 county population. The screen removes the counties in the upper tail
of this single predetermined procurement-intensity measure for every year,
preserving a balanced estimation panel.

Writes Results/instrument_normalization/
    h2_procurement_intensity_county.csv
    h2_procurement_intensity_first_stage.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_procurement_intensity_first_stage.csv
# mpgr-new-estimates: 5
# mpgr-variation-scope: h2|market-share|procurement-intensity|balanced-panel|first-stage

import importlib.util
import math
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
UPPER_TAIL_SHARES = (None, 0.001, 0.0025, 0.005, 0.01, 0.02)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
COUNTY_OUT = OUTDIR / "h2_procurement_intensity_county.csv"
FIRST_STAGE_OUT = OUTDIR / "h2_procurement_intensity_first_stage.csv"
REFERENCE = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_reference(result, reference):
    """Require the unrestricted estimate to match its saved result."""
    mapping = {
        "beta": "first_stage",
        "se": "first_stage_se",
        "t": "first_stage_t",
        "F": "first_stage_F",
        "partial_r2": "partial_r2",
    }
    for calculated, saved in mapping.items():
        left = float(result[calculated])
        right = float(reference[saved])
        gap = abs(left - right) / max(abs(right), 1e-300)
        if gap >= locked.RTOL_COEF:
            raise SystemExit(
                f"FAIL: unrestricted {calculated} relative gap is {gap:.3e}")


def county_labels():
    """Map CCG identifiers to available county or stable CCG labels."""
    frame = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    frame["ccg"] = pd.to_numeric(frame["ccg_id"], errors="coerce")
    frame = frame.dropna(subset=["ccg"]).drop_duplicates("ccg")
    frame["ccg"] = frame["ccg"].astype(np.int64)
    return frame.set_index("ccg")["ccg_name"].to_dict()


def main():
    """Estimate the procurement-intensity county-screen grid."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 procurement-intensity county screen")
    p57 = load_module(
        "p57_intensity", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_intensity", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_intensity",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_intensity",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_intensity",
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
    expected_years = list(range(locked.FIRST_YEAR, locked.LAST_YEAR + 1))
    if contract_years != expected_years:
        raise SystemExit("FAIL: intensity screen does not use the full data span")

    population = pivots["P"][locked.POP_BASE_YEAR].to_numpy(np.float64)
    if not np.isfinite(population).all() or (population <= 0).any():
        raise SystemExit("FAIL: 1975 population is missing or nonpositive")
    county_procurement = arr.sum(axis=(1, 2), dtype=np.float64)
    intensity = county_procurement / (len(contract_years) * population)
    if not np.isfinite(intensity).all() or (intensity < 0).any():
        raise SystemExit("FAIL: procurement intensity is invalid")

    labels = county_labels()
    concentration = pd.DataFrame({
        "ccg": units,
        "county": pd.Series(units).map(labels),
        "population_1975": population,
        "pooled_positive_real_procurement": county_procurement,
        "annual_real_procurement_per_1975_resident": intensity,
    })
    concentration = concentration.sort_values(
        ["annual_real_procurement_per_1975_resident", "ccg"],
        ascending=[False, True]).reset_index(drop=True)
    concentration["rank"] = np.arange(1, len(concentration) + 1)

    years = store.target_years(H)
    zwide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    z = zwide[ri, ci]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: market-share instrument has missing sample values")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    full = p69.within_first_stage(
        x, z, weight, fe, nofe, cluster, t28)
    reference = pd.read_csv(REFERENCE)
    if len(reference) != 1:
        raise SystemExit("FAIL: unrestricted reference row is incomplete")
    assert_reference(full, reference.iloc[0])

    sample_years = baseline["year"].nunique()
    rows = []
    for upper_tail_share in UPPER_TAIL_SHARES:
        n_excluded = (
            0 if upper_tail_share is None
            else int(math.ceil(upper_tail_share * locked.N_COUNTIES)))
        excluded = concentration.loc[
            concentration["rank"] <= n_excluded, "ccg"].to_numpy(np.int64)
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        retained_counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if len(retained_counts) != locked.N_COUNTIES - len(excluded):
            raise SystemExit("FAIL: retained county count is inconsistent")
        if not (retained_counts == sample_years).all():
            raise SystemExit("FAIL: intensity screen creates an unbalanced panel")

        if upper_tail_share is None:
            result = full
            cutoff = np.nan
        else:
            xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
            result = p69.within_first_stage(
                xs, z[keep], ws, fes, nofes, clusters, t28)
            cutoff = float(concentration.loc[
                concentration["rank"].eq(n_excluded),
                "annual_real_procurement_per_1975_resident"].iloc[0])
        excluded_labels = [
            f"{ccg}:{labels.get(int(ccg)) or f'CCG {int(ccg):05d}'}"
            for ccg in excluded]
        rows.append({
            "upper_tail_share": (
                "none" if upper_tail_share is None else upper_tail_share),
            "upper_tail_percent": (
                np.nan if upper_tail_share is None
                else 100.0 * upper_tail_share),
            "screen_basis": (
                "upper tail of pooled positive real procurement per 1975 "
                "resident per fiscal year in the 800 instrument categories"),
            "screen_computed_before_instrument": True,
            "procurement_sample_start": int(contract_years[0]),
            "procurement_sample_end": int(contract_years[-1]),
            "procurement_dollar_basis": f"CPI {dc.CPI_BASE} dollars",
            "population_year": locked.POP_BASE_YEAR,
            "intensity_cutoff": cutoff,
            "balanced_panel": True,
            "estimation_sample_start": int(sample["year"].min()),
            "estimation_sample_end": int(sample["year"].max()),
            "years_per_county": sample_years,
            "n": len(sample),
            "n_counties": int(sample["ccg"].nunique()),
            "n_counties_excluded": len(excluded),
            "austin_excluded": int(48015 in set(excluded)),
            "sagadahoc_excluded": int(23023 in set(excluded)),
            "excluded_ccgs": "|".join(map(str, excluded)),
            "excluded_counties": "|".join(excluded_labels),
            "first_stage": result["beta"],
            "first_stage_se": result["se"],
            "first_stage_t": result["t"],
            "first_stage_F": result["F"],
            "partial_r2": result["partial_r2"],
            "instrument_sd": float(np.std(z[keep], ddof=1)),
            "within_instrument_sd": float(np.std(result["zt"], ddof=1)),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
        })

    output = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    concentration.to_csv(COUNTY_OUT, index=False)
    output.to_csv(FIRST_STAGE_OUT, index=False)
    for ccg, label in ((48015, "Austin"), (23023, "Sagadahoc")):
        row = concentration.loc[concentration["ccg"].eq(ccg)]
        if len(row) != 1:
            raise SystemExit(f"FAIL: {label} intensity row is incomplete")
        row = row.iloc[0]
        print(
            f"   {label}: rank={int(row['rank'])}, "
            f"annual real procurement per 1975 resident="
            f"${row['annual_real_procurement_per_1975_resident']:,.2f}")
    for row in rows:
        label = (
            "none" if pd.isna(row["upper_tail_percent"])
            else f"top {row['upper_tail_percent']:g}%")
        print(
            f"   {label:>9}: excluded={row['n_counties_excluded']:>2}, "
            f"Austin={row['austin_excluded']}, "
            f"Sagadahoc={row['sagadahoc_excluded']}, "
            f"beta={row['first_stage']:+.6f}, "
            f"SE={row['first_stage_se']:.6f}, "
            f"F={row['first_stage_F']:.3f}, "
            f"partial R2={row['partial_r2']:.6f}")
    print(f"   wrote {COUNTY_OUT}")
    print(f"   wrote {FIRST_STAGE_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
