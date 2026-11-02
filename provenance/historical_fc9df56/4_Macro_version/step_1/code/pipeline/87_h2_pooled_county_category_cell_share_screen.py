"""First-stage sensitivity to pooled county-category cell-share screens.

For each county, the screen statistic is the largest pooled procurement amount
in one county-category cell divided by pooled national procurement in all 800
instrument categories. The statistic uses positive real contract dollars from
fiscal years 1966-2018 and is computed before constructing the instrument. A
threshold removes every year of a flagged county and preserves a balanced
estimation panel.

Writes Results/instrument_normalization/
    h2_pooled_county_category_cell_share.csv
    h2_pooled_county_category_cell_share_first_stage.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_pooled_county_category_cell_share_first_stage.csv
# mpgr-new-estimates: 7
# mpgr-variation-scope: h2|market-share|pooled-county-category-cell-share|balanced-panel|first-stage

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
THRESHOLDS = (None, 0.01, 0.005, 0.0025, 0.002, 0.0015, 0.001, 0.0005)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
COUNTY_OUT = OUTDIR / "h2_pooled_county_category_cell_share.csv"
FIRST_STAGE_OUT = (
    OUTDIR / "h2_pooled_county_category_cell_share_first_stage.csv")
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


def pooled_cell_share(units, categories, arr):
    """Calculate each county's largest category cell share of procurement."""
    pooled = arr.sum(axis=2, dtype=np.float64)
    national_category = pooled.sum(axis=0, dtype=np.float64)
    national_total = float(national_category.sum())
    if not np.isfinite(national_total) or national_total <= 0:
        raise SystemExit("FAIL: pooled national procurement is invalid")
    cell_share = pooled / national_total
    argmax = np.argmax(cell_share, axis=1)
    maximum = cell_share[np.arange(len(units)), argmax]
    category_total = national_category[argmax]
    category_dominance = np.divide(
        pooled[np.arange(len(units)), argmax], category_total,
        out=np.zeros(len(units), np.float64), where=category_total > 0)
    category_national_share = category_total / national_total
    if not np.allclose(
            maximum, category_dominance * category_national_share,
            rtol=1e-12, atol=1e-15):
        raise SystemExit("FAIL: cell-share product identity does not hold")
    return pd.DataFrame({
        "ccg": units,
        "maximum_cell_share": maximum,
        "maximum_category": np.asarray(categories, dtype=object)[argmax],
        "maximum_category_dominance": category_dominance,
        "maximum_category_national_procurement_share": (
            category_national_share),
        "maximum_cell_real_dollars": pooled[
            np.arange(len(units)), argmax],
    })


def main():
    """Estimate the pooled county-category cell-share threshold grid."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 pooled county-category cell-share screen")
    p57 = load_module(
        "p57_cell_share", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_cell_share", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_cell_share",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_cell_share",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_cell_share",
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
    expected_years = list(range(locked.FIRST_YEAR, locked.LAST_YEAR + 1))
    if contract_years != expected_years:
        raise SystemExit("FAIL: cell-share screen does not use the full data span")

    concentration = pooled_cell_share(units, categories, arr)
    labels = county_labels()
    concentration["county"] = concentration["ccg"].map(labels)
    concentration["rank"] = concentration["maximum_cell_share"].rank(
        method="first", ascending=False).astype(np.int64)
    concentration = concentration.sort_values("rank").reset_index(drop=True)

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

    share_by_ccg = concentration.set_index("ccg")["maximum_cell_share"]
    sample_years = baseline["year"].nunique()
    rows = []
    for threshold in THRESHOLDS:
        excluded = (
            np.array([], dtype=np.int64) if threshold is None
            else share_by_ccg.index[
                share_by_ccg.to_numpy() > threshold].to_numpy(np.int64))
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        retained_counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if len(retained_counts) != locked.N_COUNTIES - len(excluded):
            raise SystemExit("FAIL: retained county count is inconsistent")
        if not (retained_counts == sample_years).all():
            raise SystemExit("FAIL: cell-share screen creates an unbalanced panel")

        if threshold is None:
            result = full
        else:
            xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
            result = p69.within_first_stage(
                xs, z[keep], ws, fes, nofes, clusters, t28)
        excluded_labels = [
            f"{ccg}:{labels.get(int(ccg)) or f'CCG {int(ccg):05d}'}"
            for ccg in excluded]
        retained_share = share_by_ccg.drop(index=excluded)
        rows.append({
            "threshold": "none" if threshold is None else threshold,
            "threshold_percent": (
                np.nan if threshold is None else 100.0 * threshold),
            "screen_basis": (
                "largest pooled county-category cell share of positive real "
                "national procurement in the 800 instrument categories"),
            "screen_computed_before_instrument": True,
            "procurement_sample_start": int(contract_years[0]),
            "procurement_sample_end": int(contract_years[-1]),
            "procurement_dollar_basis": f"CPI {dc.CPI_BASE} dollars",
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
            "largest_retained_cell_share": float(retained_share.max()),
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
    austin = concentration.loc[concentration["ccg"].eq(48015)]
    if len(austin) != 1:
        raise SystemExit("FAIL: Austin County concentration row is incomplete")
    austin = austin.iloc[0]
    print(
        f"   Austin: max cell share={100 * austin['maximum_cell_share']:.4f}%, "
        f"category={austin['maximum_category']}, "
        f"dominance={100 * austin['maximum_category_dominance']:.2f}%, "
        f"category share="
        f"{100 * austin['maximum_category_national_procurement_share']:.3f}%, "
        f"rank={int(austin['rank'])}")
    for row in rows:
        label = (
            "none" if pd.isna(row["threshold_percent"])
            else f"{row['threshold_percent']:g}%")
        print(
            f"   threshold {label:>5}: excluded={row['n_counties_excluded']:>3}, "
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
