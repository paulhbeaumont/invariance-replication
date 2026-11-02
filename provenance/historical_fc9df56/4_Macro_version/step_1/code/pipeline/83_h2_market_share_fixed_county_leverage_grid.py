"""First-stage sensitivity to fixed full-sample county leverage screens.

County leverage is each county's share of full-sample within-instrument sum of
squares, computed once on the complete 1980-2016 sample after division-by-year
residualization. A threshold removes every year of each flagged county. The
instrument is not reconstructed after screening.

Writes Results/instrument_normalization/
    h2_market_share_fixed_county_leverage_grid.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_fixed_county_leverage_grid.csv
# mpgr-new-estimates: 6
# mpgr-variation-scope: h2|market-share|fixed-full-sample-county-leverage|balanced-panel|first-stage

import argparse
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
THRESHOLDS = (None, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
OUT = OUTDIR / "h2_market_share_fixed_county_leverage_grid.csv"
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


def county_names():
    """Map CCG identifiers to available county or stable CCG labels."""
    frame = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    frame["ccg"] = pd.to_numeric(frame["ccg_id"], errors="coerce")
    frame = frame.dropna(subset=["ccg"]).drop_duplicates("ccg")
    frame["ccg"] = frame["ccg"].astype(np.int64)
    return frame.set_index("ccg")["ccg_name"].to_dict()


def parse_args():
    """Parse an optional single-threshold fast path."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-threshold", type=float,
        help="Estimate one listed numeric threshold and append it to OUT.")
    args = parser.parse_args()
    if args.only_threshold is not None:
        matches = [
            threshold for threshold in THRESHOLDS if threshold is not None
            and np.isclose(threshold, args.only_threshold)]
        if len(matches) != 1:
            raise SystemExit(
                f"FAIL: threshold {args.only_threshold:g} is not listed")
        args.only_threshold = matches[0]
    return args


def main():
    """Estimate the fixed county-leverage threshold grid."""
    args = parse_args()
    start = time.time()
    print("== h=2 fixed county-leverage threshold grid")
    p57 = load_module(
        "p57_fixed_leverage", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_fixed_leverage", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_fixed_leverage",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_fixed_leverage",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_fixed_leverage",
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
        raise SystemExit("FAIL: market-share instrument has missing sample values")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    full = p69.within_first_stage(
        x, z, weight, fe, nofe, cluster, t28)
    reference = pd.read_csv(REFERENCE)
    if len(reference) != 1:
        raise SystemExit("FAIL: unrestricted reference row is incomplete")
    assert_reference(full, reference.iloc[0])

    n_clusters = int(cluster.max()) + 1
    cluster_ids = np.empty(n_clusters, np.int64)
    for code in range(n_clusters):
        values = baseline.loc[cluster == code, "ccg"].unique()
        if len(values) != 1:
            raise SystemExit(
                f"FAIL: cluster {code} maps to {len(values)} counties")
        cluster_ids[code] = int(values[0])
    county_zz = np.bincount(
        cluster, weights=np.square(full["zt"]), minlength=n_clusters)
    leverage = county_zz / county_zz.sum()
    leverage_by_ccg = pd.Series(leverage, index=cluster_ids)
    names = county_names()
    sample_years = baseline["year"].nunique()

    selected = (
        THRESHOLDS if args.only_threshold is None
        else (args.only_threshold,))
    rows = []
    for threshold in selected:
        excluded = (
            np.array([], dtype=np.int64) if threshold is None
            else leverage_by_ccg.index[
                leverage_by_ccg.to_numpy() > threshold].to_numpy(np.int64))
        keep = ~baseline["ccg"].isin(excluded).to_numpy()
        sample = baseline.loc[keep]
        retained_counts = sample.groupby("ccg", sort=False)["year"].nunique()
        if len(retained_counts) != locked.N_COUNTIES - len(excluded):
            raise SystemExit("FAIL: retained county count is inconsistent")
        if not (retained_counts == sample_years).all():
            raise SystemExit("FAIL: leverage screen creates an unbalanced panel")

        xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
        result = p69.within_first_stage(
            xs, z[keep], ws, fes, nofes, clusters, t28)
        excluded_labels = [
            f"{ccg}:{names.get(int(ccg)) or f'CCG {int(ccg):05d}'}"
            for ccg in excluded]
        retained_leverage = leverage_by_ccg.drop(index=excluded)
        rows.append({
            "threshold": "none" if threshold is None else threshold,
            "threshold_percent": (
                np.nan if threshold is None else 100.0 * threshold),
            "screen_basis": (
                "county share of full-sample within-instrument sum of squares"),
            "screen_computed_once": True,
            "balanced_panel": True,
            "sample_start": int(sample["year"].min()),
            "sample_end": int(sample["year"].max()),
            "years_per_county": sample_years,
            "n": len(sample),
            "n_counties": int(sample["ccg"].nunique()),
            "n_counties_excluded": len(excluded),
            "excluded_ccgs": "|".join(map(str, excluded)),
            "excluded_counties": "|".join(excluded_labels),
            "largest_retained_original_leverage_share": float(
                retained_leverage.max()),
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

    new_output = pd.DataFrame(rows)
    if args.only_threshold is None:
        output = new_output
    else:
        if not OUT.exists():
            raise SystemExit(
                "FAIL: single-threshold fast path requires the existing grid")
        existing = pd.read_csv(OUT)
        numeric_threshold = pd.to_numeric(
            existing["threshold"], errors="coerce")
        existing = existing.loc[
            ~np.isclose(
                numeric_threshold.fillna(np.inf), args.only_threshold)]
        output = pd.concat([existing, new_output], ignore_index=True)
        order = {
            "none": 0,
            **{str(threshold): rank for rank, threshold in enumerate(
                THRESHOLDS[1:], start=1)},
        }
        output["_order"] = output["threshold"].astype(str).map(order)
        if output["_order"].isna().any():
            raise SystemExit("FAIL: output contains an unlisted threshold")
        output = output.sort_values("_order").drop(columns="_order")
    same = output.loc[output["threshold"].astype(str).isin(["0.1", "0.05"])]
    if len(same) != 2 or same["excluded_ccgs"].nunique() != 1:
        raise SystemExit("FAIL: 10% and 5% screens do not select the same counties")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    for row in rows:
        label = (
            "none" if pd.isna(row["threshold_percent"])
            else f"{row['threshold_percent']:g}%")
        print(
            f"   threshold {label:>4}: excluded={row['n_counties_excluded']:>2}, "
            f"beta={row['first_stage']:+.6f}, "
            f"SE={row['first_stage_se']:.6f}, "
            f"F={row['first_stage_F']:.3f}, "
            f"partial R2={row['partial_r2']:.6f}")
        if row["excluded_counties"]:
            print(f"      {row['excluded_counties']}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
