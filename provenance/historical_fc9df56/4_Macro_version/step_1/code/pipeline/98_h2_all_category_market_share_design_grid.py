"""Run the h=2 all-category market-share design grid.

The grid holds the five-year gap before the shock and the unclipped,
same-window market-share construction fixed. It varies one design margin at a
time: the exposure-window length, category aggregation (four, three, or two
characters), observed-contract floor, and the use of positive or net actions.

Unlike the locked pipeline, this producer uses every valid four-character
FSC/PSC code that survives the existing construction screen. It does not select
the top 800 categories. The locked top-800 row is rebuilt only as a numerical
reference. The treatment and outcome are unchanged except in the matched net
and floor rows, where the county procurement change is rebuilt under the named
action rule and nominal floor.

Writes Results/instrument_normalization/
    h2_all_category_market_share_design_grid.csv
    h2_all_category_market_share_design_grid_categories.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-cache: parsed annual contract actions and county_panel_pipeline.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_all_category_market_share_design_grid.csv
# mpgr-new-estimates: 26
# mpgr-variation-scope: h2|all-valid-nonconstruction-categories|market-share|no-clip

import importlib.util
import json
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

import contracts  # noqa: E402
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
GAP = 5
BASE_SPAN = 5
SPANS = (3, 5, 7, 10)
PREFIX_DIGITS = (4, 3, 2)
FLOORS = (0.0, 10_000.0, 25_000.0)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
OUT = OUTDIR / "h2_all_category_market_share_design_grid.csv"
CATEGORY_OUT = OUTDIR / "h2_all_category_market_share_design_grid_categories.csv"
CACHE = ROOT / "data" / "temp" / "all_category_market_share_grid"
TOL = 1e-10
# The longest proposed ten-year exposure window needs contract data from 1966
# at the earliest common target year, 1982.  Holding this sample fixed makes
# every row in the design grid directly comparable.
TARGET_YEARS = list(range(1982, locked.TARGET_LAST + 1))


def load_module(name, path):
    """Load one existing project module without changing its interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_path(rule, floor, year):
    """Return the one-year sparse category cache for a rule and nominal floor."""
    return CACHE / f"rule={rule}_floor={int(floor):d}" / f"fy={year}.parquet"


def build_sparse_cache(units):
    """Persist one category-year aggregate at a time and build county totals.

    The construction screen is held fixed: contracts with FSC/PSC codes starting
    with Y remain outside both the treatment and the composition universe. Floors
    apply to nominal action dollars before CPI deflation.
    """
    fips_to_ccg, _, _ = dc.load_ccg_map()
    keys = (("positive", 0.0), ("positive", 10_000.0),
            ("positive", 25_000.0), ("net", 0.0))
    county_parts = {key: [] for key in keys}
    for year in range(locked.FIRST_YEAR, locked.LAST_YEAR + 1):
        d_all, ccg_all = contracts.county_actions(
            year, fips_to_ccg, drop_construction=True)
        d_cat, ccg_cat, cat = contracts.category_actions(
            year, fips_to_ccg, drop_construction=True, harmonize=True)
        deflator = dc.deflate(year)
        for rule, floor in keys:
            keep_all_rule = d_all > 0 if rule == "positive" else np.ones(len(d_all), bool)
            keep_cat_rule = d_cat > 0 if rule == "positive" else np.ones(len(d_cat), bool)
            keep_all = keep_all_rule & (np.abs(d_all) >= floor)
            keep_cat = keep_cat_rule & (np.abs(d_cat) >= floor)
            county_parts[(rule, floor)].append(
                pd.DataFrame({"ccg": ccg_all[keep_all],
                              "dollars": d_all[keep_all] * deflator})
                .groupby("ccg", as_index=False, sort=False)["dollars"].sum()
                .assign(year=year))
            path = cache_path(rule, floor, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            (pd.DataFrame({"ccg": ccg_cat[keep_cat], "cat": cat[keep_cat],
                           "dollars": d_cat[keep_cat] * deflator})
             .groupby(["ccg", "cat"], as_index=False, sort=False)["dollars"].sum()
             .to_parquet(path, index=False))
    years = list(range(locked.FIRST_YEAR, locked.LAST_YEAR + 1))
    county = {}
    for key, parts in county_parts.items():
        wide = (pd.concat(parts, ignore_index=True)
                .pivot(index="ccg", columns="year", values="dollars")
                .reindex(index=units, columns=years).fillna(0.0))
        county[key] = wide
    manifest = {"first_year": locked.FIRST_YEAR, "last_year": locked.LAST_YEAR,
                "construction_screen": "exclude FSC/PSC codes beginning with Y",
                "floors": [int(floor) for _, floor in keys]}
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return county


def load_group_year(rule, floor, year, digits, keep_categories=None):
    """Load and aggregate one cached category year, keeping memory bounded."""
    frame = pd.read_parquet(cache_path(rule, floor, year))
    if keep_categories is not None:
        frame = frame.loc[frame["cat"].isin(keep_categories)]
    return (frame.assign(group=frame["cat"].str[:digits])
            .groupby(["ccg", "group"], as_index=False, sort=False)["dollars"].sum())


def build_instrument(exposure_rule, exposure_floor, shift_rule, shift_floor,
                     digits, units, population, span, keep_categories=None,
                     fixed_window=None):
    """Build an unclipped h=2 market-share instrument from sparse category data."""
    target_years = TARGET_YEARS
    national_parts = []
    for year in range(locked.FIRST_YEAR, locked.LAST_YEAR + 1):
        national_parts.append(
            load_group_year(shift_rule, shift_floor, year, digits, keep_categories)
            .groupby("group", as_index=False, sort=False)["dollars"].sum()
            .assign(year=year))
    nat = (pd.concat(national_parts, ignore_index=True)
           .pivot(index="group", columns="year", values="dollars")
           .reindex(columns=range(locked.FIRST_YEAR, locked.LAST_YEAR + 1)).fillna(0.0))
    z = np.zeros((len(units), len(target_years)), np.float64)
    category_counts = []
    unit_position = pd.Series(np.arange(len(units)), index=units)
    for j, t in enumerate(target_years):
        lo, hi = (fixed_window if fixed_window is not None
                  else (t - H - GAP - span + 1, t - H - GAP))
        if hi - lo + 1 != span or hi > t - H - GAP:
            raise SystemExit(f"FAIL: invalid exposure length or gap at target {t}")
        if lo < locked.FIRST_YEAR:
            raise SystemExit(f"FAIL: target {t} needs exposure before {locked.FIRST_YEAR}")
        exposure = pd.concat([
            load_group_year(exposure_rule, exposure_floor, year, digits,
                            keep_categories)
            for year in range(lo, hi + 1)], ignore_index=True)
        exposure = (exposure.groupby(["ccg", "group"], as_index=False, sort=False)
                    ["dollars"].sum())
        exposure["dollars"] /= span
        national_exposure = exposure.groupby("group", sort=False)["dollars"].sum()
        live = national_exposure[national_exposure > 0.0]
        shift = 0.5 * (nat[t] + nat[t - 1]) - nat[t - H]
        shift = shift.reindex(live.index).fillna(0.0).rename("shift")
        work = exposure.merge(live.rename("national_exposure"),
                              left_on="group", right_index=True, how="inner")
        work = work.merge(shift, left_on="group", right_index=True, how="left")
        work["prediction"] = work["dollars"] / work["national_exposure"] * work["shift"]
        predicted = work.groupby("ccg", sort=False)["prediction"].sum()
        values = predicted.reindex(units).fillna(0.0).to_numpy(np.float64)
        pbase = population[t - H].reindex(units).to_numpy(np.float64)
        if not np.isfinite(pbase).all() or (pbase <= 0).any():
            raise SystemExit(f"FAIL: invalid base population at target year {t}")
        z[:, j] = values / pbase
        allocation = work.groupby("group", sort=False)["prediction"].sum()
        expected = shift.reindex(allocation.index).to_numpy(np.float64)
        gap = np.max(np.abs(allocation.to_numpy(np.float64) - expected), initial=0.0)
        scale = max(np.max(np.abs(expected), initial=0.0), 1.0)
        if gap / scale > TOL:
            raise SystemExit(f"FAIL: national allocation gap {gap / scale:.3e} in {t}")
        category_counts.append({"year": t, "exposure_start": lo,
                                "exposure_end": hi,
                                "actual_gap_before_base": t - H - hi,
                                "live_groups": int(len(live))})
    return z, pd.DataFrame(category_counts)


def make_panel(county_dollars, panel, units):
    """Construct the unchanged h=2 outcome and a named procurement treatment."""
    years = TARGET_YEARS
    piv = {col: panel.pivot(index="ccg", columns="year", values=col).reindex(units)
           for col in ("P", "E")}
    base = county_dollars.reindex(columns=[t - H for t in years]).to_numpy(np.float64)
    post = 0.5 * (county_dollars.reindex(columns=years).to_numpy(np.float64)
                  + county_dollars.reindex(columns=[t - 1 for t in years]).to_numpy(np.float64))
    population = piv["P"].reindex(columns=[t - H for t in years]).to_numpy(np.float64)
    e0 = piv["E"].reindex(columns=[t - H for t in years]).to_numpy(np.float64)
    e1 = piv["E"].reindex(columns=years).to_numpy(np.float64)
    x = (post - base) / population
    y = 2.0 * (e1 - e0) / (e1 + e0)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise SystemExit("FAIL: constructed county panel has nonfinite outcome or treatment")
    division = (panel.drop_duplicates("ccg").set_index("ccg")["division"]
                .reindex(units).to_numpy())
    pop1975 = (piv["P"][1975].reindex(units).to_numpy(np.float64))
    if pd.isna(division).any() or not np.isfinite(pop1975).all() or (pop1975 <= 0).any():
        raise SystemExit("FAIL: missing division or fixed population weight")
    return pd.DataFrame({"ccg": np.repeat(units, len(years)),
                         "year": np.tile(years, len(units)),
                         "x": x.ravel(), "y": y.ravel(),
                         "division": np.repeat(division, len(years)),
                         "pop1975": np.repeat(pop1975, len(years))})


def estimate(panel, zwide, units, t28):
    """Estimate first stage and 2SLS under the two pre-specified weights."""
    years = TARGET_YEARS
    row = panel["ccg"].map(pd.Series(np.arange(len(units)), index=units)).to_numpy(np.int64)
    col = panel["year"].map(pd.Series(np.arange(len(years)), index=years)).to_numpy(np.int64)
    z = zwide[row, col]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: missing instrument in regression panel")
    fe = pd.factorize(panel["division"].astype(str) + "_" + panel["year"].astype(str))[0].astype(np.int64)
    cluster = pd.factorize(panel["ccg"])[0].astype(np.int64)
    nofe = np.zeros(len(panel), np.int64)
    rows = []
    for weight_name, weight in (("equal county", np.ones(len(panel))),
                                ("fixed 1975 population", panel["pop1975"].to_numpy(np.float64))):
        first = t28.fit2sls(panel["x"].to_numpy(np.float64), z, z, weight, fe, nofe,
                             cluster, singleton=False)
        second = t28.fit2sls(panel["y"].to_numpy(np.float64), panel["x"].to_numpy(np.float64),
                              z, weight, fe, nofe, cluster, singleton=False)
        rows.append({"weight": weight_name,
                     "first_stage": first["beta"], "first_stage_se": first["se"],
                     "first_stage_t": first["beta"] / first["se"],
                     "first_stage_F": (first["beta"] / first["se"]) ** 2,
                     "second_stage": second["beta"], "second_stage_se": second["se"],
                     "second_stage_t": second["beta"] / second["se"],
                     "n": second["n"], "n_counties": int(panel["ccg"].nunique()),
                     "sample_start": int(panel["year"].min()),
                     "sample_end": int(panel["year"].max())})
    return rows


def add_variant(rows, name, margin, category_level, span, exposure_rule,
                exposure_floor, shift_rule, shift_floor, panel, units,
                population, t28, metadata, keep_categories=None,
                fixed_window=None):
    """Construct one instrument and append estimates for both weights."""
    digits = {"four-character": 4, "three-character": 3,
              "two-character": 2}.get(category_level, category_level)
    z, counts = build_instrument(
        exposure_rule, exposure_floor, shift_rule, shift_floor, digits,
        units, population, span, keep_categories, fixed_window=fixed_window)
    for fit in estimate(panel, z, units, t28):
        rows.append({"variant": name, "margin": margin,
                     "category_level": category_level, "span": span,
                     "gap_before_base": GAP, "tail_rule": "none", **metadata, **fit})
    return counts.assign(variant=name, category_level=category_level, span=span)


def main():
    """Build all-category inputs once and run each requested margin separately."""
    started = time.time()
    print("== h=2 all-category market-share design grid")
    t28 = load_module("t28_all_category_grid", STEP2_PY / "28_table_d1.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    panel = store.load_county_panel()
    units, piv = store.load_pivots(panel, cols=("P",))
    population = piv["P"]
    county = build_sparse_cache(units)
    all_category_totals = pd.concat([
        pd.read_parquet(cache_path("net", 0.0, year)).groupby("cat", as_index=False)["dollars"].sum()
        for year in range(locked.FIRST_YEAR, locked.LAST_YEAR + 1)], ignore_index=True)
    all_categories = sorted(all_category_totals["cat"].unique())
    top800 = (all_category_totals.groupby("cat", sort=False)["dollars"].sum()
              .nlargest(locked.NCAT).index.tolist())
    if len(all_categories) <= locked.NCAT or len(top800) != locked.NCAT:
        raise SystemExit("FAIL: all-category universe did not exceed the locked top-800 universe")

    panels = {("positive", 0.0): make_panel(county[("positive", 0.0)], panel, units),
              ("positive", 10_000.0): make_panel(county[("positive", 10_000.0)], panel, units),
              ("positive", 25_000.0): make_panel(county[("positive", 25_000.0)], panel, units),
              ("net", 0.0): make_panel(county[("net", 0.0)], panel, units)}
    rows, category_rows = [], []
    base_meta = {"exposure_action_rule": "positive", "national_shift_action_rule": "positive",
                 "treatment_action_rule": "positive", "nominal_floor": 0.0,
                 "category_universe": "all valid nonconstruction four-character codes"}

    # Reference only: prove that sparse raw aggregation reconstructs the former top-800 design.
    category_rows.append(add_variant(rows, "legacy_top800_reference", "reference", "four-character", BASE_SPAN,
                                     "positive", 0.0, "positive", 0.0,
                                     panels[("positive", 0.0)], units, population, t28,
                                     {**base_meta, "category_universe": "legacy top-800 nonconstruction codes"}, top800))

    # Exposure length, with the all-category four-character instrument.
    for span in SPANS:
        category_rows.append(add_variant(rows, f"allcat_span{span}", "exposure window length", "four-character", span,
                                         "positive", 0.0, "positive", 0.0,
                                         panels[("positive", 0.0)], units, population, t28, base_meta))

    # Category aggregation, with full coverage in every row.
    for digits in PREFIX_DIGITS:
        category_rows.append(add_variant(rows, f"allcat_prefix{digits}", "category aggregation",
                                         {4: "four-character", 3: "three-character", 2: "two-character"}[digits],
                                         BASE_SPAN, "positive", 0.0, "positive", 0.0,
                                         panels[("positive", 0.0)], units, population, t28, base_meta))

    # Nominal reporting floors apply before deflation to both treatment and instrument.
    for floor in FLOORS:
        metadata = {**base_meta, "nominal_floor": floor,
                    "category_universe": "all valid nonconstruction four-character codes"}
        category_rows.append(add_variant(rows, f"allcat_positive_floor{int(floor):d}", "nominal action floor",
                                         "four-character", BASE_SPAN, "positive", floor,
                                         "positive", floor, panels[("positive", floor)], units,
                                         population, t28, metadata))

    # Net action robustness. The first row changes national changes only; the second uses net actions throughout.
    category_rows.append(add_variant(rows, "allcat_net_national_shift", "positive versus net actions",
                                     "four-character", BASE_SPAN, "positive", 0.0, "net", 0.0,
                                     panels[("positive", 0.0)], units, population, t28,
                                     {**base_meta, "national_shift_action_rule": "net"}))
    category_rows.append(add_variant(rows, "allcat_net_matched", "positive versus net actions",
                                     "four-character", BASE_SPAN, "net", 0.0, "net", 0.0,
                                     panels[("net", 0.0)], units, population, t28,
                                     {"exposure_action_rule": "net", "national_shift_action_rule": "net",
                                      "treatment_action_rule": "net", "nominal_floor": 0.0,
                                      "category_universe": "all valid nonconstruction four-character codes"}))

    results = pd.DataFrame(rows)
    category_summary = pd.concat(category_rows, ignore_index=True)
    if results.duplicated(["variant", "weight"]).any():
        raise SystemExit("FAIL: duplicate variant-weight output row")
    incomplete = results.loc[results["first_stage_F"].isna() | results["second_stage_t"].isna(),
                             ["variant", "weight", "first_stage", "first_stage_se",
                              "first_stage_F", "second_stage", "second_stage_se",
                              "second_stage_t"]]
    if not incomplete.empty:
        print(incomplete.to_string(index=False))
        raise SystemExit("FAIL: incomplete estimation output")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    category_summary.to_csv(CATEGORY_OUT, index=False)
    print(f"   valid four-character categories: {len(all_categories):,}")
    print(f"   wrote {len(results):,} estimates to {OUT}")
    print(f"   done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
