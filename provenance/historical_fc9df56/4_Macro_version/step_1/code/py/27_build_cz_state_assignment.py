#!/usr/bin/env python3
"""Assign each commuting zone to one state using predetermined population.

The assignment is the state containing the largest share of a commuting zone's
1969 population. Ties go to the lowest state FIPS code. A 1970-1974 average-
population assignment is carried as a sensitivity measure. Counties remain in
their commuting zones; the assigned state is used only for fixed effects and
clustering in county-to-CZ comparisons.

Outputs:
    step_1/code/reference/cz_state_1969_population.csv
    batch_logs/macro_version/cz_state_assignment_audit.csv
    batch_logs/macro_version/27_build_cz_state_assignment.log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402


OUT = dc.REFERENCE_DIR / "cz_state_1969_population.csv"
AUDIT = dc.LOG_DIR / "cz_state_assignment_audit.csv"
LOG = dc.LOG_DIR / "27_build_cz_state_assignment.log"

EXPECTED_CCGS = 3_101
EXPECTED_CZS = 739
EXPECTED_CROSS_STATE_CZS = 136
DECLARED_MISSING_POPULATION_CCGS = {15005}  # Kalawao County, Hawaii


class Tee:
    """Mirror stdout and stderr to a flushed log file."""

    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def check(failures, ok, number, label, detail=""):
    """Print one check and retain failures for the write gate."""
    print(f"    [{number}] {label:<52s} {detail:<34s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(number), f"{label}: {detail}"))
    return ok


def load_ccg_cz():
    """One row per harmonized county group with its CZ and actual state."""
    raw = pd.read_csv(dc.CCG_CROSSWALK, dtype={"ccg_id": str, "ccz_id": str,
                                               "county_fips5": str})
    raw = raw.dropna(subset=["ccz_id"])
    raw = raw[raw["ccg_id"] != "unplaceable"].copy()
    raw["ccg"] = raw["ccg_id"].astype(np.int32)
    raw["cz"] = raw["ccz_id"]
    raw["state"] = raw["ccg"] // 1000

    by_ccg = raw.groupby("ccg", sort=False)
    if (by_ccg["cz"].nunique() != 1).any():
        raise ValueError("At least one CCG maps to multiple commuting zones")
    if (by_ccg["state"].nunique() != 1).any():
        raise ValueError("At least one CCG spans multiple states")
    return raw[["ccg", "cz", "state"]].drop_duplicates().sort_values("ccg")


def population_weights(ccg_cz, population, years):
    """CCG population averaged over the requested years, with declared gaps zero."""
    pop = population[population["year"].isin(years)]
    pop = (pop.groupby("ccg", sort=False)["population"].mean()
           .rename("weight").reset_index())
    out = ccg_cz.merge(pop, on="ccg", how="left")
    out["population_missing"] = out["weight"].isna()
    out["weight"] = out["weight"].fillna(0.0)
    return out


def assign_state(weighted):
    """Return the plurality state and assignment statistics for every CZ."""
    state = (weighted.groupby(["cz", "state"], sort=False)
             .agg(state_population=("weight", "sum"),
                  state_ccgs=("ccg", "nunique"),
                  state_missing_ccgs=("population_missing", "sum"))
             .reset_index())
    totals = (state.groupby("cz", sort=False)
              .agg(total_population=("state_population", "sum"),
                   n_states=("state", "nunique"))
              .reset_index())
    ranked = state.sort_values(["cz", "state_population", "state"],
                               ascending=[True, False, True])
    chosen = ranked.drop_duplicates("cz").rename(
        columns={"state": "assigned_state",
                 "state_population": "assigned_state_population"})
    chosen = chosen.merge(totals, on="cz", how="left")
    chosen["assigned_state_share"] = (
        chosen["assigned_state_population"] / chosen["total_population"])
    return chosen[["cz", "assigned_state", "assigned_state_population",
                   "total_population", "assigned_state_share", "n_states"]]


def metric(rows, name, value, unit, period, definition):
    """Append one long-form audit metric."""
    rows.append(dict(metric=name, value=float(value), unit=unit, period=period,
                     definition=definition))


def build_outputs():
    """Build the mapping and exact audit metrics without writing files."""
    ccg_cz = load_ccg_cz()
    population = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet",
                                 columns=["ccg", "year", "population"])

    w1969 = population_weights(ccg_cz, population, [1969])
    w7074 = population_weights(ccg_cz, population, list(range(1970, 1975)))
    main = assign_state(w1969)
    alt = assign_state(w7074)[["cz", "assigned_state"]].rename(
        columns={"assigned_state": "alternate_state_1970_1974"})

    ccg_counts = (ccg_cz.groupby("cz", sort=False)
                  .agg(n_ccgs=("ccg", "nunique"))
                  .reset_index())
    out = main.merge(alt, on="cz", how="left").merge(ccg_counts, on="cz", how="left")
    out["cross_state"] = out["n_states"] > 1
    out["assignment_differs_1970_1974"] = (
        out["assigned_state"] != out["alternate_state_1970_1974"])

    ccg_assignment = ccg_cz.merge(
        out[["cz", "assigned_state", "cross_state"]], on="cz", how="left")
    ccg_assignment["secondary_state"] = (
        ccg_assignment["state"] != ccg_assignment["assigned_state"])
    secondary_counts = (ccg_assignment.groupby("cz", sort=False)
                        ["secondary_state"].sum().rename("n_secondary_ccgs")
                        .reset_index())
    missing_counts = (w1969.groupby("cz", sort=False)["population_missing"].sum()
                      .rename("n_missing_population_ccgs").reset_index())
    out = out.merge(secondary_counts, on="cz", how="left")
    out = out.merge(missing_counts, on="cz", how="left")

    rows = []
    metric(rows, "ccgs", ccg_cz["ccg"].nunique(), "count", "all",
           "Harmonized county groups with a commuting-zone assignment")
    metric(rows, "commuting_zones", out["cz"].nunique(), "count", "all",
           "Commuting zones in the harmonized CCG universe")
    metric(rows, "cross_state_commuting_zones", out["cross_state"].sum(), "count",
           "all", "Commuting zones containing CCGs from at least two states")
    metric(rows, "secondary_state_ccgs", ccg_assignment["secondary_state"].sum(),
           "count", "all", "CCGs outside their CZ's assigned state")
    metric(rows, "missing_1969_population_ccgs", w1969["population_missing"].sum(),
           "count", "1969", "CCGs absent from the 1969 population panel")
    metric(rows, "assigned_state_agreement", (~out["assignment_differs_1970_1974"]).sum(),
           "count", "1969 versus 1970-1974",
           "CZ assignments agreeing under the two population windows")

    cross = out[out["cross_state"]]
    metric(rows, "median_assigned_state_share", cross["assigned_state_share"].median(),
           "share", "1969", "Median plurality-state population share among cross-state CZs")
    metric(rows, "minimum_assigned_state_share", cross["assigned_state_share"].min(),
           "share", "1969", "Minimum plurality-state population share among cross-state CZs")
    metric(rows, "plurality_below_one_half", (cross["assigned_state_share"] < 0.5).sum(),
           "count", "1969", "Cross-state CZs whose plurality state has under half of population")

    pop_ccg = w1969[["ccg", "weight"]].merge(
        ccg_assignment[["ccg", "secondary_state"]], on="ccg", how="left")
    total_pop = pop_ccg["weight"].sum()
    secondary_pop = pop_ccg.loc[pop_ccg["secondary_state"], "weight"].sum()
    metric(rows, "total_population", total_pop, "persons", "1969",
           "Population in all CCGs")
    metric(rows, "secondary_state_population", secondary_pop, "persons", "1969",
           "Population in CCGs outside their CZ's assigned state")
    metric(rows, "secondary_state_population_share", secondary_pop / total_pop,
           "share", "1969", "Secondary-state population divided by total population")

    panel = pd.read_parquet(dc.PANEL_OUT_DIR / "county_panel.parquet",
                            columns=["ccg", "year", "emp", "income_real",
                                     "dollars_real"])
    panel = panel[panel["year"].between(2002, 2018)].merge(
        ccg_assignment[["ccg", "secondary_state"]], on="ccg", how="left")
    activities = {
        "employment": panel["emp"],
        "real_personal_income": panel["income_real"],
        "absolute_real_contract_dollars": panel["dollars_real"].abs(),
    }
    for label, values in activities.items():
        total = values.sum()
        secondary = values[panel["secondary_state"]].sum()
        metric(rows, f"total_{label}", total, "level", "2002-2018",
               f"Sum of {label.replace('_', ' ')} across CCG-years")
        metric(rows, f"secondary_state_{label}", secondary, "level", "2002-2018",
               f"{label.replace('_', ' ').capitalize()} in secondary-state CCGs")
        metric(rows, f"secondary_state_{label}_share", secondary / total, "share",
               "2002-2018", "Secondary-state activity divided by total activity")

    reference = out.rename(columns={
        "assigned_state": "assigned_state_fips2",
        "alternate_state_1970_1974": "alternate_state_fips2_1970_1974",
        "assigned_state_population": "assigned_state_population_1969",
        "total_population": "total_population_1969",
        "assigned_state_share": "assigned_state_population_share_1969",
    })
    reference["assigned_state_fips2"] = reference["assigned_state_fips2"].map(
        lambda x: f"{int(x):02d}")
    reference["alternate_state_fips2_1970_1974"] = reference[
        "alternate_state_fips2_1970_1974"].map(lambda x: f"{int(x):02d}")
    reference["assignment_rule"] = "largest_1969_population_tie_lowest_state_fips"
    reference = reference[[
        "cz", "assigned_state_fips2", "alternate_state_fips2_1970_1974",
        "assignment_rule", "n_states", "cross_state", "n_ccgs",
        "n_secondary_ccgs", "n_missing_population_ccgs",
        "assigned_state_population_1969", "total_population_1969",
        "assigned_state_population_share_1969", "assignment_differs_1970_1974",
    ]].sort_values("cz").reset_index(drop=True)
    return reference, pd.DataFrame(rows), w1969


def main():
    np.random.seed(0)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        failures = []
        print("=" * 78)
        print("Commuting-zone state assignment from 1969 population")
        print("=" * 78)
        reference, audit, w1969 = build_outputs()

        missing = set(w1969.loc[w1969["population_missing"], "ccg"].astype(int))
        check(failures, len(reference) == EXPECTED_CZS, 1,
              "one row per commuting zone", f"{len(reference):,} rows")
        check(failures, int(reference["n_ccgs"].sum()) == EXPECTED_CCGS, 2,
              "all CCGs assigned once", f"{int(reference['n_ccgs'].sum()):,} CCGs")
        check(failures, int(reference["cross_state"].sum()) == EXPECTED_CROSS_STATE_CZS,
              3, "cross-state CZ count matches the declared universe",
              f"{int(reference['cross_state'].sum()):,} CZs")
        check(failures, missing == DECLARED_MISSING_POPULATION_CCGS, 4,
              "only the declared population gap is zero-filled",
              f"missing {sorted(missing)}")
        check(failures, reference["assigned_state_fips2"].notna().all(), 5,
              "every CZ has an assigned state")
        check(failures, reference["assigned_state_population_share_1969"].between(
            0, 1).all(), 6, "assigned-state shares lie in [0,1]")

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for number, message in failures:
                print(f"    [{number}] {message}")
            return 1

        reference.to_csv(OUT, index=False)
        audit.to_csv(AUDIT, index=False)
        print(f"\nall checks passed   {len(reference):,} CZs -> {OUT}")
        print(f"audit metrics        {len(audit):,} rows -> {AUDIT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
