"""First-stage sensitivity to fixed raw category-dominance county screens.

For each county, the screen statistic is its largest share of national positive
real procurement in any instrument category and any rolling five-year window
from fiscal years 1966-2018. The statistic uses contract dollars only and is
computed before constructing the instrument. A threshold removes every year of
a flagged county and therefore preserves a balanced estimation panel.

Writes Results/instrument_normalization/
    h2_raw_category_dominance_county.csv
    h2_raw_category_dominance_first_stage.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_raw_category_dominance_first_stage.csv
# mpgr-new-estimates: 5
# mpgr-variation-scope: h2|market-share|raw-category-dominance|balanced-panel|first-stage

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
WINDOW = 5
THRESHOLDS = (None, 0.50, 0.33, 0.25, 0.20, 0.10)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
COUNTY_OUT = OUTDIR / "h2_raw_category_dominance_county.csv"
FIRST_STAGE_OUT = OUTDIR / "h2_raw_category_dominance_first_stage.csv"
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


def raw_dominance(units, categories, years, arr):
    """Find each county's maximum national category share over five years."""
    if len(years) < WINDOW:
        raise SystemExit("FAIL: contract panel is shorter than the window")
    n_counties = len(units)
    maximum = np.full(n_counties, -np.inf, np.float64)
    maximum_category = np.full(n_counties, -1, np.int64)
    maximum_start = np.full(n_counties, -1, np.int64)
    window = arr[:, :, :WINDOW].sum(axis=2, dtype=np.float64)

    for start_index in range(len(years) - WINDOW + 1):
        if start_index > 0:
            window += arr[:, :, start_index + WINDOW - 1]
            window -= arr[:, :, start_index - 1]
        national = window.sum(axis=0, dtype=np.float64)
        shares = np.divide(
            window, national[None, :],
            out=np.zeros_like(window), where=national[None, :] > 0)
        if shares.min() < -1e-12 or shares.max() > 1.0 + 1e-12:
            raise SystemExit("FAIL: raw category share lies outside [0, 1]")
        argmax = np.argmax(shares, axis=1)
        rowmax = shares[np.arange(n_counties), argmax]
        improve = rowmax > maximum
        maximum[improve] = rowmax[improve]
        maximum_category[improve] = argmax[improve]
        maximum_start[improve] = years[start_index]

    if not np.isfinite(maximum).all() or (maximum_category < 0).any():
        raise SystemExit("FAIL: county dominance statistic is incomplete")
    return pd.DataFrame({
        "ccg": units,
        "maximum_category_share": maximum,
        "maximum_category": np.asarray(categories, dtype=object)[
            maximum_category],
        "window_start": maximum_start,
        "window_end": maximum_start + WINDOW - 1,
    })


def main():
    """Estimate the raw category-dominance county-screen grid."""
    np.random.seed(0)
    start = time.time()
    print("== h=2 raw category-dominance county screen")
    p57 = load_module(
        "p57_raw_dominance", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_raw_dominance", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_raw_dominance",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_raw_dominance",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_raw_dominance",
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
        raise SystemExit("FAIL: dominance screen does not use the full data span")

    dominance = raw_dominance(units, categories, contract_years, arr)
    labels = county_labels()
    dominance["county"] = dominance["ccg"].map(labels)
    dominance["rank"] = dominance["maximum_category_share"].rank(
        method="first", ascending=False).astype(np.int64)
    dominance = dominance.sort_values("rank").reset_index(drop=True)

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

    dominance_by_ccg = dominance.set_index("ccg")["maximum_category_share"]
    sample_years = baseline["year"].nunique()
    rows = []
    for threshold in THRESHOLDS:
        excluded = (
            np.array([], dtype=np.int64) if threshold is None
            else dominance_by_ccg.index[
                dominance_by_ccg.to_numpy() > threshold].to_numpy(np.int64))
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        retained_counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if len(retained_counts) != locked.N_COUNTIES - len(excluded):
            raise SystemExit("FAIL: retained county count is inconsistent")
        if not (retained_counts == sample_years).all():
            raise SystemExit("FAIL: dominance screen creates an unbalanced panel")

        xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
        result = p69.within_first_stage(
            xs, z[keep], ws, fes, nofes, clusters, t28)
        excluded_labels = [
            f"{ccg}:{labels.get(int(ccg)) or f'CCG {int(ccg):05d}'}"
            for ccg in excluded]
        retained_dominance = dominance_by_ccg.drop(index=excluded)
        rows.append({
            "threshold": "none" if threshold is None else threshold,
            "threshold_percent": (
                np.nan if threshold is None else 100.0 * threshold),
            "screen_basis": (
                "maximum county share of positive real national procurement "
                "within any category and rolling five-year window"),
            "screen_computed_before_instrument": True,
            "procurement_sample_start": int(contract_years[0]),
            "procurement_sample_end": int(contract_years[-1]),
            "procurement_dollar_basis": f"CPI {dc.CPI_BASE} dollars",
            "window_years": WINDOW,
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
            "largest_retained_category_share": float(
                retained_dominance.max()),
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
    dominance.to_csv(COUNTY_OUT, index=False)
    output.to_csv(FIRST_STAGE_OUT, index=False)
    austin = dominance.loc[dominance["ccg"].eq(48015)]
    if len(austin) != 1:
        raise SystemExit("FAIL: Austin County dominance row is incomplete")
    austin = austin.iloc[0]
    print(
        f"   Austin: max share={100 * austin['maximum_category_share']:.2f}%, "
        f"category={austin['maximum_category']}, "
        f"window={austin['window_start']}-{austin['window_end']}, "
        f"rank={int(austin['rank'])}")
    for row in rows:
        label = (
            "none" if pd.isna(row["threshold_percent"])
            else f"{row['threshold_percent']:g}%")
        print(
            f"   threshold {label:>4}: excluded={row['n_counties_excluded']:>4}, "
            f"Austin={row['austin_excluded']}, "
            f"beta={row['first_stage']:+.6f}, "
            f"SE={row['first_stage_se']:.6f}, "
            f"F={row['first_stage_F']:.3f}, "
            f"partial R2={row['partial_r2']:.6f}")
    print(f"   wrote {COUNTY_OUT}")
    print(f"   wrote {FIRST_STAGE_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
