"""Estimate the factorial h=2 exposure-window by category-detail grid.

This derivative uses the certified sparse all-category cache from producer 98.
It holds the h=2 outcome, five-year gap before the shock, unclipped
market-share construction, positive actions, zero nominal floor, and the
existing nonconstruction screen fixed. It estimates every combination of
rolling exposure length (3, 5, 7, 9, or 10 years) and category detail (four,
three, or two characters), plus fixed 1966--1974 exposure at each category
detail, on the common 1982--2016 sample. Complete saved cells are reused.

Writes Results/instrument_normalization/
    h2_all_category_window_by_category_grid.csv
    h2_all_category_window_by_category_grid_categories.csv
"""

# mpgr-run-class: estimate_only
# mpgr-cache: data/temp/all_category_market_share_grid/ and contracts_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_all_category_window_by_category_grid.csv
# mpgr-new-estimates: 12
# mpgr-variation-scope: h2|all-valid-nonconstruction-categories|window-by-category-factorial|no-clip

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


OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
       / "h2_all_category_window_by_category_grid.csv")
CATEGORY_OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
                / "h2_all_category_window_by_category_grid_categories.csv")
SPANS = (3, 5, 7, 9, 10)
LEVELS = ((4, "four-character"), (3, "three-character"),
          (2, "two-character"))


def load_module(name, path):
    """Load an existing producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_sparse_cache(p98):
    """Fail before estimation if the certified all-category cache is incomplete."""
    missing = [
        p98.cache_path("positive", 0.0, year)
        for year in range(locked.FIRST_YEAR, locked.LAST_YEAR + 1)
        if not p98.cache_path("positive", 0.0, year).exists()
    ]
    if missing:
        raise SystemExit(
            f"FAIL: sparse cache is incomplete, first missing file is {missing[0]}")


def matching_one_factor_variant(span, digits):
    """Return the producer-98 one-factor row for a factorial identity check."""
    if digits == 4 and span != 9:
        return f"allcat_span{span}"
    if span == 5:
        return f"allcat_prefix{digits}"
    return None


def verify_overlap(results, p98):
    """Verify six factorial cells against the previously saved one-factor rows."""
    prior = pd.read_csv(p98.OUT)
    fields = ("first_stage", "first_stage_se", "first_stage_t", "first_stage_F",
              "second_stage", "second_stage_se", "second_stage_t")
    checks = 0
    for span in SPANS:
        for digits, _ in LEVELS:
            old_variant = matching_one_factor_variant(span, digits)
            if old_variant is None:
                continue
            candidate = results.loc[(results["span"] == span)
                                    & (results["category_digits"] == digits)]
            reference = prior.loc[prior["variant"] == old_variant]
            for weight in candidate["weight"].unique():
                a = candidate.loc[candidate["weight"] == weight].iloc[0]
                b = reference.loc[reference["weight"] == weight].iloc[0]
                for field in fields:
                    if not np.isclose(a[field], b[field], rtol=1e-11, atol=1e-13):
                        raise SystemExit(
                            f"FAIL: {span}-year/{digits}-character {weight} does not "
                            f"reproduce producer 98 for {field}")
                checks += 1
    if checks != 12:
        raise SystemExit(f"FAIL: expected 12 overlap checks, found {checks}")


def main():
    """Construct and estimate every window-by-category combination."""
    started = time.time()
    np.random.seed(0)
    print("== h=2 all-category exposure-window by category-detail grid")
    p98 = load_module("p98_factorial", PIPELINE / "98_h2_all_category_market_share_design_grid.py")
    t28 = load_module("t28_factorial", STEP2_PY / "28_table_d1.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    require_sparse_cache(p98)
    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=("P",))
    county_dollars = (store.load_con("positive")
                      .reindex(index=units,
                               columns=range(locked.FIRST_YEAR, locked.LAST_YEAR + 1))
                      .fillna(0.0))
    panel = p98.make_panel(county_dollars, county_panel, units)
    metadata = {
        "exposure_action_rule": "positive",
        "national_shift_action_rule": "positive",
        "treatment_action_rule": "positive",
        "nominal_floor": 0.0,
        "category_universe": "all valid nonconstruction four-character codes",
        "gap_before_base": p98.GAP,
        "tail_rule": "none",
    }

    rows, category_rows = [], []
    if OUT.exists():
        saved = pd.read_csv(OUT)
        if "exposure_mode" not in saved:
            saved["exposure_mode"] = "rolling"
        expected_n = len(units) * len(p98.TARGET_YEARS)
        if (not saved["n"].eq(expected_n).all()
                or not saved["sample_start"].eq(min(p98.TARGET_YEARS)).all()
                or not saved["sample_end"].eq(max(p98.TARGET_YEARS)).all()):
            raise SystemExit("FAIL: saved grid sample does not match input panel")
        rows = saved.to_dict("records")
        category_rows.append(pd.read_csv(CATEGORY_OUT))
    computed = 0
    designs = [("rolling", span, None) for span in SPANS]
    designs.append(("fixed_1966_1974", 9, (1966, 1974)))
    for mode, span, fixed_window in designs:
        for digits, label in LEVELS:
            variant = (f"allcat_span{span}_prefix{digits}" if mode == "rolling"
                       else f"allcat_fixed1966_1974_prefix{digits}")
            cached = [r for r in rows if r["variant"] == variant]
            if cached:
                if len(cached) != 2 or {r["weight"] for r in cached} != {
                        "equal county", "fixed 1975 population"}:
                    raise SystemExit(f"FAIL: incomplete saved cell {variant}")
                continue
            category_rows.append(p98.add_variant(
                rows, variant, "exposure window by category detail", label, span,
                "positive", 0.0, "positive", 0.0, panel, units, pivots["P"],
                t28, {**metadata, "exposure_mode": mode,
                      "gap_before_base": p98.GAP if mode == "rolling" else np.nan,
                      "minimum_gap_before_base": p98.GAP},
                fixed_window=fixed_window))
            computed += 2
    results = pd.DataFrame(rows)
    results["category_digits"] = results["category_level"].map({
        "four-character": 4, "three-character": 3, "two-character": 2})
    if len(results) != len(designs) * len(LEVELS) * 2:
        raise SystemExit("FAIL: factorial output has an unexpected number of estimates")
    if results.duplicated(["exposure_mode", "span", "category_digits", "weight"]).any():
        raise SystemExit("FAIL: duplicate factorial design cell")
    if results[["first_stage_F", "second_stage_t"]].isna().any().any():
        raise SystemExit("FAIL: incomplete factorial estimation output")
    verify_overlap(results, p98)

    categories = pd.concat(category_rows, ignore_index=True)
    fixed = categories.loc[categories["variant"].str.contains("fixed1966_1974")]
    if not (fixed["exposure_start"].eq(1966).all()
            and fixed["exposure_end"].eq(1974).all()
            and (fixed["year"] - p98.H - fixed["exposure_end"]).ge(p98.GAP).all()):
        raise SystemExit("FAIL: fixed exposure window or minimum gap mismatch")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    categories.to_csv(CATEGORY_OUT, index=False)
    print(f"   wrote {len(results):,} estimates to {OUT}")
    print(f"   estimated {computed} rows; reused {len(results) - computed}")
    print("   verified 12 overlapping cells against producer 98")
    print(f"   done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
