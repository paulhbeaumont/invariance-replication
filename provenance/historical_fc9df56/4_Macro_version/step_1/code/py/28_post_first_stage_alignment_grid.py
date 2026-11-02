#!/usr/bin/env python3
"""First-stage alignment grid with state controls at county and CZ rungs.

The main grid uses fixed five-year exposure windows, common calendar periods,
and unit fixed effects plus state-by-year fixed effects. County observations use
their actual state. Commuting zones use the predetermined state assignment
produced by 27_build_cz_state_assignment.py. Both rungs retain only states with
at least two CZs under the applicable CZ-state assignment.

Main design:
    pre   fixed_7074, 1979-1997
    post  fixed_9397, 2002-2018
    horizons 2, 3, 4
    rungs county, commuting zone
    FE unit + state-by-year
    regression weights unweighted, lagged employment
    clusters unit, state

Targeted diagnostics use h=2, unweighted regressions clustered by unit:
    post sample ending in 2006 or 2016
    1970-1974 population CZ-state assignment
    exclusion of all cross-state CZs
    leave-one-state-out under unit + state-by-year FE

Output:
    batch_logs/macro_version/post_first_stage_alignment_grid.csv
    batch_logs/macro_version/28_post_first_stage_alignment_grid.log
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402


CODE = Path(__file__).resolve().parent
OUT = dc.LOG_DIR / "post_first_stage_alignment_grid.csv"
LOG = dc.LOG_DIR / "28_post_first_stage_alignment_grid.log"
CZ_STATE = dc.REFERENCE_DIR / "cz_state_1969_population.csv"


def load_numbered(name, filename):
    """Import a numbered producer without invoking its main function."""
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bi = load_numbered("build_instrument_alignment", "14_build_instrument.py")
sym = load_numbered("first_stage_symmetric", "19_first_stage_table.py")

HORIZONS = (2, 3, 4)
RUNGS = (("county", "ccg"), ("commuting zone", "cz"))
DESIGN = (("pre", "fixed_7074", 1979, 1997),
          ("post", "fixed_9397", 2002, 2018))
FE_SPECS = (
    ("unit+state-by-year", "unit + state_year"),
)
REG_WEIGHTS = (("unweighted", None), ("employment", "w"))
CLUSTERS = (("unit", "unit"), ("state", "state"))
SAMPLE_DIAGNOSTICS = (("post_dcads", 2002, dc.SEAM_YEAR - 1),
                      ("post_pre2017", 2002, 2016))
PRIMARY_HORIZON = 2
PRIMARY_FE = "unit+state-by-year"
PRIMARY_FORMULA = "unit + state_year"
OMEGA = "employment"
OMEGA_COL = "emp_lag"
MIN_CZS_PER_STATE = 2
EXPECTED_MAIN_ROWS = (len(DESIGN) * len(HORIZONS) * len(RUNGS)
                      * len(FE_SPECS) * len(REG_WEIGHTS) * len(CLUSTERS))
EXPECTED_SAMPLE_ROWS = len(SAMPLE_DIAGNOSTICS) * len(RUNGS) * len(FE_SPECS)
EXPECTED_ASSIGNMENT_ROWS = 2

DEMEANER = None


def check(failures, ok, number, label, detail=""):
    """Print one check and retain failures for the write gate."""
    print(f"    [{number}] {label:<54s} {detail:<38s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(number), f"{label}: {detail}"))
    return ok


def register_periods():
    """Register common periods so the canonical contamination guard can inspect them."""
    before_periods = dict(bi.PERIODS)
    before_admissible = {key: set(value) for key, value in bi.ADMISSIBLE.items()}
    for era, exposure, first, last in DESIGN:
        key = f"aligned_{era}"
        bi.PERIODS[key] = (first, last)
        bi.ADMISSIBLE[key] = {exposure}
    for label, first, last in SAMPLE_DIAGNOSTICS:
        key = f"aligned_{label}"
        bi.PERIODS[key] = (first, last)
        bi.ADMISSIBLE[key] = {"fixed_9397"}
    return before_periods, before_admissible


def registration_only_added(before):
    """Verify that registration preserves every producer-owned key."""
    periods, admissible = before
    bad = []
    for key, value in periods.items():
        if bi.PERIODS.get(key) != value:
            bad.append(f"PERIODS[{key}]")
    for key, value in admissible.items():
        if set(bi.ADMISSIBLE.get(key, ())) != value:
            bad.append(f"ADMISSIBLE[{key}]")
    return not bad, ("nothing redefined" if not bad else ", ".join(bad))


def load_cz_assignment():
    """Read and validate the deterministic one-state-per-CZ reference."""
    a = pd.read_csv(CZ_STATE, dtype={
        "cz": str,
        "assigned_state_fips2": str,
        "alternate_state_fips2_1970_1974": str,
    })
    a["cz"] = a["cz"].astype(np.int32).astype(str)
    a["state_1969"] = a["assigned_state_fips2"].astype(np.int32)
    a["state_1970_1974"] = a["alternate_state_fips2_1970_1974"].astype(np.int32)
    if a["cz"].duplicated().any():
        raise ValueError("The CZ-state reference has duplicate commuting zones")
    if a[["state_1969", "state_1970_1974"]].isna().any().any():
        raise ValueError("The CZ-state reference has a missing state")
    return a[["cz", "state_1969", "state_1970_1974", "cross_state"]]


def build_analysis():
    """Build the symmetric fixed exposures in memory through producer 19."""
    sym.register_variants(HORIZONS)
    county, levels, universe, pre = sym.build(HORIZONS)
    rungs = bi.load_rungs(universe)
    return county, levels, universe, pre, rungs


def estimation_base(county, pre, rungs, exposure, horizon, first, last):
    """County-level estimation support frozen before the calendar-period filter."""
    full = county[county["h"] == horizon].copy()
    full["z"] = full[f"z_{exposure}"]
    full["pre_D"] = full["ccg"].map(pre[exposure]["pre_D"])
    full["pre_Y"] = full["ccg"].map(pre[exposure]["pre_Y"])
    full = full.merge(rungs, on="ccg", how="left")
    finite = np.isfinite(full[["shock", "emp_growth", "z", "income_real_lag",
                               "emp_lag"]].to_numpy()).all(axis=1)
    full = full[finite & (full["emp_lag"] > 0) & (full["income_real_lag"] > 0)]
    return full[full["year"].between(first, last)]


def unit_frame(base, rung, assignment, assignment_mode="1969",
               drop_cross_state=False):
    """One row per unit-year with a state, FE codes, and employment weight."""
    state_col = "state_1969" if assignment_mode == "1969" else "state_1970_1974"
    analysis_czs = set(base["cz"].dropna().astype(str).unique())
    assignment_used = assignment[assignment["cz"].isin(analysis_czs)].copy()
    if drop_cross_state:
        assignment_used = assignment_used[~assignment_used["cross_state"]]
    cz_counts = assignment_used.groupby(state_col, sort=False)["cz"].nunique()
    eligible_states = set(cz_counts[cz_counts >= MIN_CZS_PER_STATE].index.astype(int))

    if rung == "county":
        d = base.rename(columns={"emp_growth": "y", "shock": "x",
                                 "emp_lag": "w"})
        d = d.assign(unit=d["ccg"].astype(np.int32),
                     state=d["state"].astype(np.int32),
                     cross_state=False)
        states_before = set(d["state"].unique())
        d = d[d["state"].isin(eligible_states)]
        d = d[["unit", "year", "state", "cross_state", "y", "x", "z", "w"]]
        state_assignment = "actual_county_state"
    else:
        d = bi.collapse(base, "cz", OMEGA_COL)[["unit", "year", "y", "x", "z"]]
        weights = (base.groupby(["cz", "year"], sort=False)["emp_lag"].sum()
                   .rename("w").reset_index().rename(columns={"cz": "unit"}))
        d = d.merge(weights, on=["unit", "year"], how="left")
        d = d.rename(columns={"unit": "cz"})
        d = d.merge(assignment, on="cz", how="left")
        d["state"] = d[state_col].astype(np.int32)
        states_before = set(d["state"].unique())
        if drop_cross_state:
            d = d[~d["cross_state"]]
        d = d[d["state"].isin(eligible_states)]
        d["unit"] = d["cz"].astype(np.int32)
        d = d[["unit", "year", "state", "cross_state", "y", "x", "z", "w"]]
        state_assignment = ("largest_1969_population" if assignment_mode == "1969"
                            else "largest_1970_1974_average_population")
        if drop_cross_state:
            state_assignment += "_drop_cross_state_czs"

    d = d[np.isfinite(d[["x", "y", "z", "w"]].to_numpy()).all(axis=1)].copy()
    d["unit"] = d["unit"].astype(np.int32)
    d["year"] = d["year"].astype(np.int32)
    d["state"] = d["state"].astype(np.int32)
    d["state_year"] = (d["state"] * 10_000 + d["year"]).astype(np.int32)
    excluded_states = sorted(states_before - set(d["state"].unique()))
    singleton_rows = 0
    while True:
        unit_size = d.groupby("unit", sort=False)["year"].transform("size")
        state_year_size = d.groupby("state_year", sort=False)["unit"].transform("size")
        keep = (unit_size > 1) & (state_year_size > 1)
        if keep.all():
            break
        singleton_rows += int((~keep).sum())
        d = d[keep].copy()
    d.attrs["excluded_state_fips"] = " ".join(f"{state:02d}" for state in excluded_states)
    d.attrs["explicit_singleton_fe_rows_excluded"] = singleton_rows
    if not ((d["w"] > 0).all() and np.isfinite(d["w"]).all()):
        raise ValueError(f"Invalid regression weight in {rung} frame")
    return d, state_assignment


def fit_first_stage(frame, fe_formula, cluster_col, weights=None):
    """Estimate one first stage and report clustered F plus within partial R-squared."""
    formula = f"x ~ z | {fe_formula}"
    kwargs = {"vcov": {"CRV1": cluster_col}, "demeaner": DEMEANER}
    if weights is not None:
        kwargs["weights"] = weights
    fit = pf.feols(formula, frame, **kwargs)
    coefficient = float(fit.coef()["z"])
    standard_error = float(fit.se()["z"])
    return {
        "n_obs": int(fit._N),
        "fs_coef": coefficient,
        "fs_se": standard_error,
        "fs_F": (coefficient / standard_error) ** 2,
        "partial_r2": float(fit._r2_within),
    }


def result_row(frame, role, sample, era, exposure, horizon, rung, fe_name,
               fe_formula, cluster_name, cluster_col, weight_name, weight_col,
               state_assignment, first, last, contaminated_years,
               dropped_state=np.nan):
    """Fit one cell and attach its complete design metadata."""
    result = fit_first_stage(frame, fe_formula, cluster_col, weight_col)
    return dict(
        role=role, sample=sample, era=era, exposure=exposure,
        exposure_window=sym.window_label(exposure), h=horizon, rung=rung,
        omega=("unweighted" if rung == "county" else OMEGA),
        fe=fe_name, cluster=cluster_name, reg_weights=weight_name,
        state_assignment=state_assignment, dropped_state=dropped_state,
        state_eligibility_rule=f"at_least_{MIN_CZS_PER_STATE}_assigned_czs",
        excluded_state_fips=frame.attrs.get("excluded_state_fips", ""),
        explicit_singleton_fe_rows_excluded=int(
            frame.attrs.get("explicit_singleton_fe_rows_excluded", 0)),
        year_first=first, year_last=last,
        contaminated_years=contaminated_years, n_frame=len(frame),
        estimator_rows_dropped=len(frame) - result["n_obs"],
        n_units=int(frame["unit"].nunique()),
        n_states=int(frame["state"].nunique()), **result)


def main_grid(county, pre, rungs, assignment):
    """The fully crossed, predeclared two-era alignment grid."""
    rows = []
    for era, exposure, first, last in DESIGN:
        period_key = f"aligned_{era}"
        for horizon in HORIZONS:
            base = estimation_base(county, pre, rungs, exposure, horizon, first, last)
            contaminated = len(bi.contaminated_years(period_key, exposure, horizon))
            for rung, _ in RUNGS:
                frame, state_assignment = unit_frame(base, rung, assignment)
                for fe_name, fe_formula in FE_SPECS:
                    for weight_name, weight_col in REG_WEIGHTS:
                        for cluster_name, cluster_col in CLUSTERS:
                            rows.append(result_row(
                                frame, "main", "common_calendar", era, exposure,
                                horizon, rung, fe_name, fe_formula, cluster_name,
                                cluster_col, weight_name, weight_col,
                                state_assignment, first, last, contaminated))
    return rows


def sample_diagnostics(county, pre, rungs, assignment):
    """Targeted post-period source and employment-vintage diagnostics."""
    rows = []
    exposure = "fixed_9397"
    for sample, first, last in SAMPLE_DIAGNOSTICS:
        base = estimation_base(county, pre, rungs, exposure, PRIMARY_HORIZON,
                               first, last)
        contaminated = len(bi.contaminated_years(
            f"aligned_{sample}", exposure, PRIMARY_HORIZON))
        for rung, _ in RUNGS:
            frame, state_assignment = unit_frame(base, rung, assignment)
            for fe_name, fe_formula in FE_SPECS:
                rows.append(result_row(
                    frame, "sample_sensitivity", sample, "post", exposure,
                    PRIMARY_HORIZON, rung, fe_name, fe_formula, "unit", "unit",
                    "unweighted", None, state_assignment, first, last, contaminated))
    return rows


def assignment_diagnostics(county, pre, rungs, assignment):
    """Targeted CZ-state assignment diagnostics at the primary post specification."""
    era, exposure, first, last = DESIGN[1]
    base = estimation_base(county, pre, rungs, exposure, PRIMARY_HORIZON, first, last)
    contaminated = len(bi.contaminated_years(
        "aligned_post", exposure, PRIMARY_HORIZON))
    rows = []
    for mode, drop in (("1970_1974", False), ("1969", True)):
        frame, state_assignment = unit_frame(
            base, "commuting zone", assignment, assignment_mode=mode,
            drop_cross_state=drop)
        sample = "alternate_state_assignment" if not drop else "drop_cross_state_czs"
        rows.append(result_row(
            frame, "assignment_sensitivity", sample, era, exposure,
            PRIMARY_HORIZON, "commuting zone", PRIMARY_FE, PRIMARY_FORMULA,
            "unit", "unit", "unweighted", None, state_assignment,
            first, last, contaminated))
    return rows


def leave_one_state_out(county, pre, rungs, assignment):
    """Drop each state in turn at the primary post first stage."""
    era, exposure, first, last = DESIGN[1]
    base = estimation_base(county, pre, rungs, exposure, PRIMARY_HORIZON, first, last)
    contaminated = len(bi.contaminated_years(
        "aligned_post", exposure, PRIMARY_HORIZON))
    rows = []
    for rung, _ in RUNGS:
        frame, state_assignment = unit_frame(base, rung, assignment)
        for state in sorted(frame["state"].unique()):
            reduced = frame[frame["state"] != state]
            rows.append(result_row(
                reduced, "leave_one_state_out", "common_calendar", era, exposure,
                PRIMARY_HORIZON, rung, PRIMARY_FE, PRIMARY_FORMULA,
                "unit", "unit", "unweighted", None, state_assignment,
                first, last, contaminated, dropped_state=int(state)))
    return rows


def exposure_check(failures, county, pre):
    """Verify each fixed exposure against its window sums at every horizon."""
    deviations = []
    for _, exposure, _, _ in DESIGN:
        expected = pre[exposure]["pre_D"] / pre[exposure]["pre_Y"]
        for horizon in HORIZONS:
            frame = county[county["h"] == horizon]
            actual = frame[["ccg", f"exposure_{exposure}"]].drop_duplicates("ccg")
            actual = actual.set_index("ccg")[f"exposure_{exposure}"]
            joined = pd.concat([expected.rename("expected"), actual.rename("actual")],
                               axis=1).dropna()
            deviations.append(float((joined["expected"] - joined["actual"]).abs().max()))
    worst = max(deviations, default=np.inf)
    return check(failures, len(deviations) == len(DESIGN) * len(HORIZONS)
                 and worst < 1e-12, 5,
                 "fixed exposures reproduce their window sums",
                 f"{len(deviations)} cells, max deviation {worst:.1e}")


def main():
    global DEMEANER
    np.random.seed(0)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = sym.Tee(LOG)
    try:
        failures = []
        DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
        print("=" * 78)
        print("Post-period first-stage alignment grid")
        print("=" * 78)
        print(f"  design       {DESIGN}")
        print(f"  horizons     {HORIZONS}")
        print("  rungs        county, commuting zone")
        print("  fixed effects " + "; ".join(name for name, _ in FE_SPECS))
        print("  clusters     unit, assigned state")
        print("  weights      unweighted, lagged employment WLS")
        print("  CZ state     largest 1969 population, tie lowest state FIPS")

        print("\n[1] registration and inputs")
        before = register_periods()
        ok, detail = registration_only_added(before)
        check(failures, ok, 1, "period registration only adds keys", detail)
        assignment = load_cz_assignment()
        check(failures, len(assignment) == 739 and assignment["cz"].nunique() == 739,
              2, "CZ-state assignment is one-to-one", f"{len(assignment)} CZs")
        check(failures, int(assignment["cross_state"].sum()) == 136, 3,
              "cross-state flag matches the assignment audit",
              f"{int(assignment['cross_state'].sum())} CZs")

        print("\n[2] instrument build through producer 19")
        county, _, _, pre, rungs = build_analysis()

        print("\n[3] fully crossed main grid")
        rows = main_grid(county, pre, rungs, assignment)
        print(f"    {len(rows)} main cells estimated")

        print("\n[4] targeted diagnostics")
        rows.extend(sample_diagnostics(county, pre, rungs, assignment))
        rows.extend(assignment_diagnostics(county, pre, rungs, assignment))
        rows.extend(leave_one_state_out(county, pre, rungs, assignment))
        results = pd.DataFrame(rows)
        print(f"    {len(results) - EXPECTED_MAIN_ROWS} diagnostic cells estimated")

        print("\n[5] checks")
        main_rows = results[results["role"] == "main"]
        sample_rows = results[results["role"] == "sample_sensitivity"]
        assignment_rows = results[results["role"] == "assignment_sensitivity"]
        loo_rows = results[results["role"] == "leave_one_state_out"]
        check(failures, len(main_rows) == EXPECTED_MAIN_ROWS, 4,
              "main row count equals the announced grid",
              f"{len(main_rows)}/{EXPECTED_MAIN_ROWS}")
        exposure_check(failures, county, pre)
        check(failures, int(main_rows["contaminated_years"].max()) == 0, 6,
              "every main cell has zero exposure contamination",
              f"max {int(main_rows['contaminated_years'].max())}")
        check(failures, len(sample_rows) == EXPECTED_SAMPLE_ROWS, 7,
              "sample diagnostic row count is exact",
              f"{len(sample_rows)}/{EXPECTED_SAMPLE_ROWS}")
        check(failures, len(assignment_rows) == EXPECTED_ASSIGNMENT_ROWS, 8,
              "assignment diagnostic row count is exact",
              f"{len(assignment_rows)}/{EXPECTED_ASSIGNMENT_ROWS}")
        loo_ok = True
        loo_detail = []
        for rung, group in loo_rows.groupby("rung", sort=False):
            states = group["dropped_state"].nunique()
            loo_ok &= len(group) == states and (group["n_states"] == states - 1).all()
            loo_detail.append(f"{rung} {states} states")
        check(failures, loo_ok and set(loo_rows["rung"]) == {r for r, _ in RUNGS}, 9,
              "leave-one-state-out grid covers every represented state",
              f"{len(loo_rows)} cells: " + ", ".join(loo_detail))
        drops = results["estimator_rows_dropped"]
        check(failures, (drops == 0).all(), 10,
              "the explicit state screen leaves no singleton drops",
              f"{int((drops > 0).sum())} cells, max {int(drops.max())} rows")
        finite_cols = ["fs_coef", "fs_se", "fs_F", "partial_r2"]
        check(failures, np.isfinite(results[finite_cols].to_numpy()).all(), 11,
              "all first-stage statistics are finite",
              f"{len(results)} cells")

        deterministic = main_rows[
            (main_rows["era"] == "post") & (main_rows["h"] == PRIMARY_HORIZON)
            & (main_rows["rung"] == "county") & (main_rows["fe"] == PRIMARY_FE)
            & (main_rows["cluster"] == "unit")
            & (main_rows["reg_weights"] == "unweighted")
        ].iloc[0]
        base = estimation_base(county, pre, rungs, "fixed_9397", PRIMARY_HORIZON,
                               2002, 2018)
        frame, _ = unit_frame(base, "county", assignment)
        repeated = fit_first_stage(frame, PRIMARY_FORMULA, "unit")
        repeat_dev = max(abs(repeated[key] - deterministic[key])
                         for key in ("fs_coef", "fs_se", "fs_F", "partial_r2"))
        check(failures, repeat_dev == 0.0, 12,
              "pinned primary refit is bit-identical", f"max deviation {repeat_dev:.1e}")

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for number, message in failures:
                print(f"    [{number}] {message}")
            return 1

        results.to_csv(OUT, index=False)
        print(f"\nall checks passed   {len(results)} fits -> {OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
