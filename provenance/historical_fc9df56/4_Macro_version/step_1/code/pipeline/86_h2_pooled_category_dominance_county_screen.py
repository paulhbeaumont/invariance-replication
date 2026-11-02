"""First-stage sensitivity to pooled raw category-dominance county screens.

The screen pools positive real procurement over fiscal years 1966-2018. It
retains categories satisfying a fixed minimum share of pooled national
procurement in the 800 instrument categories. For each county, the screen
statistic is its largest share of national procurement in any retained
category. The statistic uses contract dollars only and is computed before
constructing the instrument. A threshold removes every year of a flagged
county and therefore preserves a balanced estimation panel.

Writes Results/instrument_normalization/
    h2_pooled_category_dominance_county.csv
    h2_pooled_category_dominance_first_stage.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_pooled_category_dominance_first_stage.csv
# mpgr-new-estimates: 15
# mpgr-variation-scope: h2|market-share|pooled-category-dominance|balanced-panel|first-stage

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
MATERIAL_CATEGORY_SHARES = (0.001, 0.005, 0.01)
THRESHOLDS = (None, 0.50, 0.33, 0.25, 0.20, 0.10)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
COUNTY_OUT = OUTDIR / "h2_pooled_category_dominance_county.csv"
FIRST_STAGE_OUT = OUTDIR / "h2_pooled_category_dominance_first_stage.csv"
REFERENCE = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"


def parse_args():
    """Parse an optional materiality-filter fast path."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-material-shares", type=float, nargs="+",
        help="Estimate listed materiality filters and append them to outputs.")
    args = parser.parse_args()
    if args.only_material_shares is not None:
        selected = []
        for requested in args.only_material_shares:
            matches = [
                value for value in MATERIAL_CATEGORY_SHARES
                if np.isclose(value, requested)]
            if len(matches) != 1:
                raise SystemExit(
                    f"FAIL: materiality filter {requested:g} is not listed")
            if matches[0] not in selected:
                selected.append(matches[0])
        args.only_material_shares = tuple(selected)
    return args


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


def pooled_dominance(units, categories, pooled, national_category,
                     national_category_share, material_threshold):
    """Calculate raw category shares and county maxima for one filter."""
    material = national_category_share >= material_threshold
    if not material.any():
        raise SystemExit("FAIL: no category passes the materiality threshold")
    shares = np.divide(
        pooled[:, material], national_category[material][None, :],
        out=np.zeros((len(units), int(material.sum())), np.float64),
        where=national_category[material][None, :] > 0)
    if shares.min() < -1e-12 or shares.max() > 1.0 + 1e-12:
        raise SystemExit("FAIL: pooled category share lies outside [0, 1]")
    argmax = np.argmax(shares, axis=1)
    maximum = shares[np.arange(len(units)), argmax]
    material_categories = np.asarray(categories, dtype=object)[material]
    material_national_shares = national_category_share[material]
    return pd.DataFrame({
        "ccg": units,
        "maximum_category_share": maximum,
        "maximum_category": material_categories[argmax],
        "maximum_category_national_procurement_share": (
            material_national_shares[argmax]),
        "material_category_threshold": material_threshold,
        "n_material_categories": int(material.sum()),
    }), material


def append_blocks(existing_path, new_block, key_column, selected):
    """Replace selected parameter blocks while preserving saved rows."""
    if not existing_path.exists():
        raise SystemExit(
            f"FAIL: fast path requires existing output {existing_path}")
    existing = pd.read_csv(existing_path)
    key = pd.to_numeric(existing[key_column], errors="coerce")
    remove = np.zeros(len(existing), dtype=bool)
    for value in selected:
        remove |= np.isclose(key, value)
    return pd.concat([existing.loc[~remove], new_block], ignore_index=True)


def main():
    """Estimate the pooled raw category-dominance county-screen grid."""
    np.random.seed(0)
    args = parse_args()
    selected = (
        MATERIAL_CATEGORY_SHARES if args.only_material_shares is None
        else args.only_material_shares)
    start = time.time()
    print("== h=2 pooled raw category-dominance county screen")
    p57 = load_module(
        "p57_pooled_dominance", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_pooled_dominance", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_pooled_dominance",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_pooled_dominance",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_pooled_dominance",
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

    pooled = arr.sum(axis=2, dtype=np.float64)
    national_category = pooled.sum(axis=0, dtype=np.float64)
    national_total = float(national_category.sum())
    if not np.isfinite(national_total) or national_total <= 0:
        raise SystemExit("FAIL: pooled national procurement is invalid")
    category_shares = national_category / national_total

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

    labels = county_labels()
    sample_years = baseline["year"].nunique()
    county_blocks = []
    first_stage_rows = []
    print_rows = []
    for material_threshold in selected:
        dominance, material = pooled_dominance(
            units, categories, pooled, national_category, category_shares,
            material_threshold)
        dominance["county"] = dominance["ccg"].map(labels)
        dominance["rank"] = dominance["maximum_category_share"].rank(
            method="first", ascending=False).astype(np.int64)
        dominance = dominance.sort_values("rank").reset_index(drop=True)
        county_blocks.append(dominance)
        dominance_by_ccg = dominance.set_index("ccg")[
            "maximum_category_share"]

        austin = dominance.loc[dominance["ccg"].eq(48015)]
        if len(austin) != 1:
            raise SystemExit("FAIL: Austin County dominance row is incomplete")
        austin = austin.iloc[0]
        print(
            f"   materiality={100 * material_threshold:g}%: "
            f"categories={int(material.sum())}/{len(categories)}, "
            f"procurement share={100 * category_shares[material].sum():.2f}%")
        print(
            f"      Austin max={100 * austin['maximum_category_share']:.2f}%, "
            f"category={austin['maximum_category']}, "
            f"category national share="
            f"{100 * austin['maximum_category_national_procurement_share']:.3f}%, "
            f"rank={int(austin['rank'])}")

        for threshold in THRESHOLDS:
            excluded = (
                np.array([], dtype=np.int64) if threshold is None
                else dominance_by_ccg.index[
                    dominance_by_ccg.to_numpy() > threshold].to_numpy(
                        np.int64))
            keep = ~baseline["ccg"].isin(excluded).to_numpy()
            sample = baseline.loc[keep]
            retained_counts = sample.groupby(
                "ccg", sort=False)["year"].nunique()
            if len(retained_counts) != locked.N_COUNTIES - len(excluded):
                raise SystemExit("FAIL: retained county count is inconsistent")
            if not (retained_counts == sample_years).all():
                raise SystemExit(
                    "FAIL: dominance screen creates an unbalanced panel")

            if threshold is None:
                result = full
            else:
                xs, ws, fes, nofes, clusters = helper.estimation_arrays(sample)
                result = p69.within_first_stage(
                    xs, z[keep], ws, fes, nofes, clusters, t28)
            excluded_labels = [
                f"{ccg}:{labels.get(int(ccg)) or f'CCG {int(ccg):05d}'}"
                for ccg in excluded]
            retained_dominance = dominance_by_ccg.drop(index=excluded)
            row = {
                "threshold": "none" if threshold is None else threshold,
                "threshold_percent": (
                    np.nan if threshold is None else 100.0 * threshold),
                "screen_basis": (
                    "maximum county share of pooled positive real national "
                    "procurement within a material category"),
                "screen_computed_before_instrument": True,
                "procurement_sample_start": int(contract_years[0]),
                "procurement_sample_end": int(contract_years[-1]),
                "procurement_dollar_basis": f"CPI {dc.CPI_BASE} dollars",
                "material_category_share_threshold": material_threshold,
                "material_category_share_threshold_percent": (
                    100.0 * material_threshold),
                "n_material_categories": int(material.sum()),
                "material_categories_procurement_share": float(
                    category_shares[material].sum()),
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
                "within_instrument_sd": float(
                    np.std(result["zt"], ddof=1)),
                "fixed_effects": "division-by-year",
                "weight": "equal county",
                "cluster": "county",
            }
            first_stage_rows.append(row)
            print_rows.append(row)

    new_county = pd.concat(county_blocks, ignore_index=True)
    new_first_stage = pd.DataFrame(first_stage_rows)
    if args.only_material_shares is None:
        county_output = new_county
        first_stage_output = new_first_stage
    else:
        county_output = append_blocks(
            COUNTY_OUT, new_county, "material_category_threshold", selected)
        first_stage_output = append_blocks(
            FIRST_STAGE_OUT, new_first_stage,
            "material_category_share_threshold", selected)

    county_output = county_output.sort_values(
        ["material_category_threshold", "rank"]).reset_index(drop=True)
    threshold_order = {
        "none": 0,
        **{str(value): rank for rank, value in enumerate(
            THRESHOLDS[1:], start=1)},
    }
    first_stage_output["_threshold_order"] = (
        first_stage_output["threshold"].astype(str).map(threshold_order))
    if first_stage_output["_threshold_order"].isna().any():
        raise SystemExit("FAIL: output contains an unlisted dominance threshold")
    first_stage_output = first_stage_output.sort_values([
        "material_category_share_threshold", "_threshold_order"
    ]).drop(columns="_threshold_order").reset_index(drop=True)

    for material_threshold in MATERIAL_CATEGORY_SHARES:
        county_group = county_output.loc[np.isclose(
            county_output["material_category_threshold"], material_threshold)]
        stage_group = first_stage_output.loc[np.isclose(
            first_stage_output["material_category_share_threshold"],
            material_threshold)]
        if len(county_group) != locked.N_COUNTIES or len(stage_group) != len(
                THRESHOLDS):
            raise SystemExit(
                f"FAIL: output block {material_threshold:g} is incomplete")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    county_output.to_csv(COUNTY_OUT, index=False)
    first_stage_output.to_csv(FIRST_STAGE_OUT, index=False)
    for row in print_rows:
        label = (
            "none" if pd.isna(row["threshold_percent"])
            else f"{row['threshold_percent']:g}%")
        print(
            f"      threshold {label:>4}: "
            f"excluded={row['n_counties_excluded']:>3}, "
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
