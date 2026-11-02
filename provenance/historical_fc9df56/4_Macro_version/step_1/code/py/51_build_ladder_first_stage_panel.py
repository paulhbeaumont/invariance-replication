#!/usr/bin/env python3
"""Build the county panel the aggregation ladder is estimated on.

41 through 50 and the candidate package are county-rung only: every one of them
hard-asserts G == 3080 and none carries a geography above the county. So there
is no first-stage F at the commuting zone or the state on the instrument step_2
actually uses. This file is the input that closes that gap.

WHAT IS STORED, AND WHY IT IS ONLY THE COUNTY

The panel carries county rows and nothing else, by request. Two reasons, and the
second is the operative one.

  1. The winsorized series are stored, not the recipe, because the capped shock
     and capped instrument are what future analysis uses. A reader of this file
     does not have to know the cutoff rule to use the right variable.

  2. The rungs above the county are NOT stored, because climbing needs a weight
     vector and the weight vector is not settled for all time. Income weights are
     what the settled specification uses; employment weights are the invariance
     version of PLAN 3.3 and may be wanted later. Storing county rows plus both
     denominators keeps every rung one collapse away, and storing the collapsed
     rows would freeze a choice that is still live.

THE CAP, AND WHERE IT IS COMPUTED

`winsor4 ..., method(iq) level(5)` within start year, applied SEPARATELY to the
shock and to the instrument. Computed once here, at the county, and never
recomputed after aggregating. That is protocol A.3.3 and PLAN 3.5 rule 2: a cap
computed a second time on the aggregate distribution is a different cutoff
binding on different observations, and the rungs then stop measuring the same
object. 52 asserts the aggregate is a weighted mean of these capped county rows
and of nothing else.

The uncapped series sit beside the capped one. The two winsorization arms are
co-equal; neither is the robustness check of the other.

THE PIN

The rebuild below is 01_build_candidate_panel.py's arithmetic restated, not
imported, so that a divergence shows up here rather than three scripts later.
Every rebuilt column is compared against the candidate .dta it reproduces and
must agree at 0.0 - not a tolerance, an equality, because both sides are the
same floating-point expression over the same inputs.

Outputs:
    _data/temp/ladder_first_stage/panel.parquet
    _data/temp/ladder_first_stage/<period>_h<h>.dta
    batch_logs/macro_version/ladder_first_stage_design.json
    batch_logs/macro_version/ladder_first_stage_share_audit.csv
    batch_logs/macro_version/ladder_first_stage_panel_audit.csv
    batch_logs/macro_version/51_build_ladder_first_stage_panel.log
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: I001


np.random.seed(0)

CODE = Path(__file__).resolve().parent
INSTRUMENT_BUILDER = CODE / "14_build_instrument.py"
SOURCE = dc.SYMMETRIC_FIRST_STAGE_DIR / "panel.parquet"
CANDIDATE = dc.PROJECT_ROOT / "Candidate for first stage"
CANDIDATE_TEMP = CANDIDATE / "data" / "temp"
CZ_STATE = dc.REFERENCE_DIR / "cz_state_1969_population.csv"

TEMP = dc.MACRO_DATA_ROOT / "temp" / "ladder_first_stage"
LOG = dc.LOG_DIR / "51_build_ladder_first_stage_panel.log"
DESIGN_OUT = dc.LOG_DIR / "ladder_first_stage_design.json"
SHARE_AUDIT_OUT = dc.LOG_DIR / "ladder_first_stage_share_audit.csv"
PANEL_AUDIT_OUT = dc.LOG_DIR / "ladder_first_stage_panel_audit.csv"

PERIODS = {
    "pre": {"first": 1979, "last": 1997},
    "post": {"first": 1998, "last": 2018},
}
HORIZONS = (2, 3, 4)
HEADLINE_HORIZON = 4
SHARES = ("fixed4", "fixed5", "fixed10")
HEADLINE_SHARE = "fixed5"
WINSOR_LEVEL = 5.0
EXPECTED_ANALYSIS_UNITS = 3_080
EXPECTED_STATES = 48

# Every rung the panel can be collapsed to. The division is carried as a key so
# that a later run does not have to rebuild the panel to reach it; the settled
# specification stops at the state and 52 estimates three rungs, not four.
RUNG_KEYS = ("unit", "cz", "state", "division")
ESTIMATED_RUNGS = ("county", "cz", "state")

# The two absorption arms 52 estimates, in the order it reports them. `sxy` is
# the arm this ladder was frozen on; `uy` is what step_2 estimates on. Both are
# admissible: the treatment varies at county x start-year and each arm is
# strictly coarser than that at every rung. See the design contract's `fe_arms`.
FE_ARMS = ("sxy", "uy")
HEADLINE_FE_ARM = "sxy"


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, value):
        self.stdout.write(value)
        self.file.write(value)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def load_numbered(path: Path):
    spec = importlib.util.spec_from_file_location("instrument_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Stata's percentile, lifted verbatim from
# step_2/code/py/29_candidate_iv_mpgr_check.py lines 103-127, which is the
# function the candidate table's own cutoffs were reproduced with. Copied rather
# than imported: step_1 does not import from step_2, and a shared winsorization
# path would put every grid in this pipeline one edit away from moving together.
# 52 re-derives the same cap inside Stata and asserts equality, so the copy is
# pinned rather than trusted.
#
# `summarize, detail` sets j = n p / 100; when j is an integer the percentile is
# the midpoint of x(j) and x(j+1), otherwise it is x(ceil(j)). Integer arithmetic
# on n*p, so the integer case is exact.
# ------------------------------------------------------------------------------

def stata_pctile(xs, n, p):
    num = n * p
    if num % 100 == 0:
        j = num // 100
        return 0.5 * (xs[j - 1] + xs[j]) if j < n else xs[n - 1]
    return xs[-(-num // 100) - 1]


def winsor4_iq(v, level):
    """`winsor4 varname, method(iq) level(#)`: clip at p50 -/+ level * IQR."""
    xs = np.sort(v)
    n = len(xs)
    p25 = stata_pctile(xs, n, 25)
    p50 = stata_pctile(xs, n, 50)
    p75 = stata_pctile(xs, n, 75)
    iqr = p75 - p25
    return np.clip(v, p50 - level * iqr, p50 + level * iqr)


def winsor4_by_year(v, year, level):
    out = v.astype(np.float64).copy()
    for y in np.unique(year):
        m = year == y
        out[m] = winsor4_iq(out[m], level)
    return out


def cutoffs_by_year(v, year, level):
    """The cap the rule implies, reported per start year so the audit can show
    what bound and by how much rather than only that something did."""
    rows = []
    for y in np.unique(year):
        m = year == y
        xs = np.sort(v[m])
        n = len(xs)
        p25 = stata_pctile(xs, n, 25)
        p50 = stata_pctile(xs, n, 50)
        p75 = stata_pctile(xs, n, 75)
        lo, hi = p50 - level * (p75 - p25), p50 + level * (p75 - p25)
        rows.append(
            {
                "year": int(y),
                "n": int(n),
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "lo": lo,
                "hi": hi,
                "n_below": int((v[m] < lo).sum()),
                "n_above": int((v[m] > hi).sum()),
            }
        )
    return rows


# ------------------------------------------------------------------------------
# Geography
# ------------------------------------------------------------------------------

def load_geography(units, panel_states):
    """One row per analysis county: commuting zone, state, division, and the
    state the commuting zone is assigned to for its fixed effects.

    The commuting zone comes from 14's `load_rungs`, which reads `ccz_id` and not
    the raw county-to-CZ bridge - a CCG can straddle a 1990 zone boundary and 10
    already union-finds those, so the raw bridge would put one county in two
    zones and silently break the frozen-support rule.

    The CZ's state is 27's largest-1969-population assignment. A commuting zone
    crosses state lines 135 times on this support, so it is not nested in a state
    and the state-by-year effect at that rung has to be built from a declared
    assignment rather than inferred.

    ONE REPAIR, AND WHY IT IS NOT OPTIONAL

    27 computed its assignment on the full 3,101-county universe, before the
    analysis support existed. The support keeps 48 states - 41's rule drops any
    state without at least two assigned commuting zones - and 27's assignment can
    therefore point a zone at a state the support does not contain. It does so
    exactly once: CZ 20401 is assigned to Rhode Island, and the only county of it
    that survives into the support is in Massachusetts.

    Left alone that zone would sit alone in every one of its state-by-year cells,
    reghdfe would absorb it perfectly and drop it, and the commuting-zone rung
    would then cover a different set of counties from the county rung. That is
    the support artefact the whole exercise exists to measure, arriving through
    the geography labels instead of the data.

    So a zone whose assigned state is outside the support is reassigned to the
    state holding most of its RETAINED counties, ties to the lowest FIPS - 27's
    tie rule, evaluated on the frozen support, which is the order PLAN 3.5 rule 1
    prescribes. Every reassignment is named below rather than counted.
    """
    builder = load_numbered(INSTRUMENT_BUILDER)
    rungs = builder.load_rungs(set(units))
    rungs = rungs.rename(columns={"ccg": "unit"})
    rungs["cz"] = rungs["cz"].astype(np.int64)

    missing = set(units) - set(rungs["unit"])
    if missing:
        raise AssertionError(
            f"{len(missing)} analysis counties have no commuting zone: "
            f"{sorted(missing)[:10]}"
        )
    if rungs["division"].isna().any():
        raise AssertionError("a county has no Census division")

    # Division as an integer code; the name lives in the design contract only, so
    # the .dta carries no string column.
    names = sorted(rungs["division"].unique())
    codes = {name: i + 1 for i, name in enumerate(names)}
    rungs["division_name"] = rungs["division"]
    rungs["division"] = rungs["division"].map(codes).astype(np.int32)

    cz_state = pd.read_csv(CZ_STATE, usecols=["cz", "assigned_state_fips2"])
    cz_state = cz_state.rename(columns={"assigned_state_fips2": "cz_state"})
    cz_state["cz"] = cz_state["cz"].astype(np.int64)
    rungs = rungs.merge(cz_state, on="cz", how="left", validate="many_to_one")
    if rungs["cz_state"].isna().any():
        raise AssertionError("a commuting zone has no assigned state")
    rungs["cz_state"] = rungs["cz_state"].astype(np.int32)

    rungs = rungs[rungs["unit"].isin(units)].sort_values("unit").reset_index(drop=True)

    orphans = sorted(set(rungs.loc[~rungs["cz_state"].isin(panel_states), "cz"]))
    repairs = []
    for cz in orphans:
        block = rungs[rungs["cz"] == cz]
        counts = block.groupby("state").size().sort_values(
            ascending=False, kind="stable"
        )
        top = int(counts.max())
        replacement = int(min(s for s, n in counts.items() if n == top))
        repairs.append(
            {
                "cz": int(cz),
                "assigned_state": int(block["cz_state"].iloc[0]),
                "reassigned_state": replacement,
                "n_retained_counties": int(len(block)),
                "retained_counties": ",".join(str(u) for u in sorted(block["unit"])),
            }
        )
        rungs.loc[rungs["cz"] == cz, "cz_state"] = np.int32(replacement)

    print(
        f"geography: {len(rungs):,} counties, {rungs['cz'].nunique():,} commuting "
        f"zones, {rungs['state'].nunique()} states, "
        f"{rungs['division'].nunique()} divisions"
    )
    if repairs:
        print(
            f"           {len(repairs)} commuting zone(s) assigned outside the "
            f"{len(panel_states)}-state support, reassigned on retained counties:"
        )
        for row in repairs:
            print(
                f"             cz {row['cz']}: state {row['assigned_state']} -> "
                f"{row['reassigned_state']} on {row['n_retained_counties']} retained "
                f"county/ies ({row['retained_counties']})"
            )
    if not set(rungs["cz_state"]).issubset(panel_states):
        raise AssertionError("a commuting zone is still assigned outside the support")

    # Every state-by-year cell at the CZ rung must hold at least two zones, or
    # reghdfe absorbs the cell exactly and drops it, which would silently shrink
    # the CZ rung's county coverage below the county rung's.
    per_state = rungs.groupby("cz_state")["cz"].nunique()
    if int(per_state.min()) < 2:
        raise AssertionError(
            "these assigned states hold a single commuting zone and would be "
            f"absorbed away: {sorted(per_state[per_state < 2].index.tolist())}"
        )
    print(
        f"           commuting zones assigned across {rungs['cz_state'].nunique()} "
        f"states, thinnest holding {int(per_state.min())} zones "
        f"({sorted(per_state[per_state == per_state.min()].index.tolist())})"
    )
    return rungs, codes, pd.DataFrame(repairs)


# ------------------------------------------------------------------------------
# The rebuild, and the pin that makes it trustworthy
# ------------------------------------------------------------------------------

def rebuild_candidate_columns(frame: pd.DataFrame):
    """01_build_candidate_panel.py lines 68-79, restated.

    The instrument divides by NATIONAL income. It never divides by the focal
    county's own income, and it is not leave-one-out: the national contract
    numerator and the national income denominator are both plain sums over the
    3,080 analysis counties. That is the whole content of the candidate
    correction, so it is written out here rather than inherited.
    """
    out = pd.DataFrame(index=frame.index)
    out["shock_raw"] = frame["future"] / frame["income_real"]
    for share in SHARES:
        out[f"z_raw_{share}"] = (
            frame[share] * frame["national_future"] / frame["national_income"]
        )
    return out


def pin_to_candidate(period: str, horizon: int, frame: pd.DataFrame):
    """Every rebuilt column against the candidate .dta it reproduces, at 0.0.

    Not a tolerance. Both sides evaluate the same expression over the same
    float64 inputs, so anything other than an exact match is a build difference
    and not arithmetic noise.
    """
    path = CANDIDATE_TEMP / f"{period}_h{horizon}.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"candidate pin target is missing: {path}\n"
            "Run 'Candidate for first stage/code/01_build_candidate_panel.py' first; "
            "its data/temp is gitignored."
        )
    ref = pd.read_stata(path)
    ref = ref.sort_values(["unit", "year"]).reset_index(drop=True)
    got = frame.sort_values(["unit", "year"]).reset_index(drop=True)

    if len(ref) != len(got):
        raise AssertionError(
            f"{period} h={horizon}: candidate has {len(ref):,} rows, rebuild has "
            f"{len(got):,}"
        )
    for key in ("unit", "year", "state"):
        if not np.array_equal(ref[key].to_numpy(np.int64), got[key].to_numpy(np.int64)):
            raise AssertionError(f"{period} h={horizon}: {key} differs from candidate")

    pairs = [("y_income", "shock_raw")] + [
        (f"z_income_{share}", f"z_raw_{share}") for share in SHARES
    ]
    worst = 0.0
    for ref_col, got_col in pairs:
        deviation = float(
            np.max(np.abs(ref[ref_col].to_numpy(np.float64) - got[got_col].to_numpy(np.float64)))
        )
        worst = max(worst, deviation)
        if deviation != 0.0:
            raise AssertionError(
                f"{period} h={horizon}: {got_col} deviates from candidate "
                f"{ref_col} by {deviation:g}, expected exactly 0.0"
            )
    print(f"    pinned to {path.name}: {len(pairs)} columns, max deviation {worst:g}")
    return len(pairs)


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

def period_starts(period: str, horizon: int):
    bounds = PERIODS[period]
    return list(range(bounds["first"], bounds["last"] - horizon + 1, horizon))


def build_block(period: str, horizon: int, source: pd.DataFrame, rungs: pd.DataFrame):
    frame = source[
        (source["period"] == period) & (source["horizon"] == horizon)
    ].copy()
    if frame.empty:
        raise AssertionError(f"symmetric panel has no {period} h={horizon} block")

    starts = period_starts(period, horizon)
    if sorted(frame["year"].unique().tolist()) != starts:
        raise AssertionError(
            f"{period} h={horizon}: start years {sorted(frame['year'].unique())} "
            f"do not match the nonoverlapping schedule {starts}"
        )

    rebuilt = rebuild_candidate_columns(frame)
    frame = pd.concat([frame.reset_index(drop=True), rebuilt.reset_index(drop=True)], axis=1)
    n_pinned = pin_to_candidate(period, horizon, frame)

    # `rungs` derives state as ccg // 1000 and the panel carries its own copy.
    # They must agree; assert that rather than letting the merge suffix them into
    # two columns that quietly disagree.
    check = frame[["unit", "state"]].drop_duplicates().merge(
        rungs[["unit", "state"]], on="unit", suffixes=("_panel", "_rungs")
    )
    if not check["state_panel"].equals(check["state_rungs"].astype(check["state_panel"].dtype)):
        raise AssertionError(f"{period} h={horizon}: panel and rung state disagree")

    frame = frame.merge(
        rungs.drop(columns=["state", "division_name"]),
        on="unit",
        how="left",
        validate="many_to_one",
    )
    if frame[["cz", "cz_state", "division"]].isna().any().any():
        raise AssertionError(f"{period} h={horizon}: a county lost its geography")

    year = frame["year"].to_numpy(np.int64)
    cut_rows = []
    frame["shock_iq5"] = winsor4_by_year(
        frame["shock_raw"].to_numpy(np.float64), year, WINSOR_LEVEL
    )
    for row in cutoffs_by_year(frame["shock_raw"].to_numpy(np.float64), year, WINSOR_LEVEL):
        cut_rows.append({"period": period, "horizon": horizon, "variable": "shock", **row})
    for share in SHARES:
        frame[f"z_iq5_{share}"] = winsor4_by_year(
            frame[f"z_raw_{share}"].to_numpy(np.float64), year, WINSOR_LEVEL
        )
        for row in cutoffs_by_year(
            frame[f"z_raw_{share}"].to_numpy(np.float64), year, WINSOR_LEVEL
        ):
            cut_rows.append(
                {"period": period, "horizon": horizon, "variable": f"z_{share}", **row}
            )

    frame["cz_state_year"] = (
        frame["cz_state"].astype(np.int32) * 10_000 + frame["year"].astype(np.int32)
    )

    n_units = frame["unit"].nunique()
    if n_units != EXPECTED_ANALYSIS_UNITS:
        raise AssertionError(f"{period} h={horizon}: {n_units} counties")
    if frame["state"].nunique() != EXPECTED_STATES:
        raise AssertionError(f"{period} h={horizon}: {frame['state'].nunique()} states")
    if len(frame) != EXPECTED_ANALYSIS_UNITS * len(starts):
        raise AssertionError(
            f"{period} h={horizon}: {len(frame):,} rows, expected "
            f"{EXPECTED_ANALYSIS_UNITS * len(starts):,}"
        )
    # Balance is what makes the county support frozen across rungs: if every
    # county appears in every start year, so does every commuting zone and every
    # state, and no rung can quietly cover a different population.
    counts = frame.groupby("unit", sort=False).size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != len(starts):
        raise AssertionError(f"{period} h={horizon}: county panel is unbalanced")

    n_capped = {
        "shock": int((frame["shock_iq5"] != frame["shock_raw"]).sum()),
        **{
            f"z_{share}": int((frame[f"z_iq5_{share}"] != frame[f"z_raw_{share}"]).sum())
            for share in SHARES
        },
    }
    print(
        f"  {period} h={horizon}: {len(frame):,} rows, {len(starts)} windows "
        f"{starts[0]}-{starts[-1]}, {n_pinned} columns pinned"
    )
    for name, count in n_capped.items():
        print(
            f"      IQR{WINSOR_LEVEL:g}-by-start-year capped {name:<12s} "
            f"{count:>6,} of {len(frame):,} ({100 * count / len(frame):.2f}%)"
        )
    return frame, starts, cut_rows, n_capped


COLUMNS = [
    "period",
    "horizon",
    "unit",
    "state",
    "cz",
    "cz_state",
    "division",
    "year",
    "state_year",
    "cz_state_year",
    "future",
    "income_real",
    "employment",
    "national_future",
    "national_income",
    *SHARES,
    "shock_raw",
    "shock_iq5",
    *(f"z_raw_{share}" for share in SHARES),
    *(f"z_iq5_{share}" for share in SHARES),
]


def design_contract(codes: dict, starts: dict, repairs: pd.DataFrame):
    return {
        "contract": "ladder_first_stage_v1",
        "purpose": (
            "first-stage F at the county, commuting zone and state rungs on the "
            "candidate income instrument"
        ),
        "source": {
            "symmetric_panel": str(SOURCE),
            "candidate_pin": str(CANDIDATE_TEMP),
            "contract_flow": "signed award-year obligations",
            "dollar_treatment": "no nominal floor",
            "price_basis": "real 2006 dollars",
        },
        "sample": {
            "periods": PERIODS,
            "horizons": list(HORIZONS),
            "headline_horizon": HEADLINE_HORIZON,
            "window_mode": "period-anchored nonoverlapping start years",
            "start_years": starts,
            "expected_counties": EXPECTED_ANALYSIS_UNITS,
            "expected_states": EXPECTED_STATES,
            "sample_floor": "none: no nominal dollar floor, no population floor",
        },
        "shares": {
            "lengths": list(SHARES),
            "headline": HEADLINE_SHARE,
            "windows": {
                period: {
                    share: [
                        bounds["first"] - int(share.removeprefix("fixed")),
                        bounds["first"] - 1,
                    ]
                    for share in SHARES
                }
                for period, bounds in PERIODS.items()
            },
            "definition": (
                "county share of national real contract dollars over the window, "
                "national denominator, time invariant within era"
            ),
        },
        "scaling": {
            "shock": "county contracts from t+1 through t+h divided by county real personal income in t",
            "instrument": (
                "county fixed share times national contracts from t+1 through t+h "
                "divided by national real personal income in t"
            ),
            "leave_one_out": False,
            "instrument_divides_by_own_income": False,
        },
        "winsorization": {
            "variables": "shock and instrument, separately",
            "rule": "winsor4, method(iq) level(5)",
            "group": "start year",
            "level": WINSOR_LEVEL,
            "computed_at": "county rung only, never recomputed after aggregating",
            "rules_reported": ["none", "iq5"],
            "unwinsorized_is_co_equal": True,
        },
        "rungs": {
            "stored": "county only",
            "estimated": list(ESTIMATED_RUNGS),
            "keys_carried": list(RUNG_KEYS),
            "aggregation": (
                "income-weighted mean of the capped county values, weights "
                "omega_ct = income_ct / income_jt rebuilt within each start year"
            ),
            "division_codes": codes,
            # Two co-equal absorption arms, both admissible under A.1.4: the
            # treatment varies at county x start-year and each arm is strictly
            # coarser than that at every rung it is estimated on. `sxy` is the
            # arm this ladder was first built on; `uy` is the arm step_2's
            # Tables 1-5 estimate on, added so the exhibit reports the
            # specification the application actually uses.
            "fe_arms": {
                "sxy": {
                    "county": "unit + state x year",
                    "cz": "unit + assigned state x year",
                    "state": "unit + year",
                },
                "uy": {
                    "county": "unit + year",
                    "cz": "unit + year",
                    "state": "unit + year",
                },
            },
            "fe_arms_reported": list(FE_ARMS),
            # At the state rung the unit IS the state, so a state-by-year effect
            # would absorb every observation and 52 already sets that rung to
            # unit + year. The two arms are therefore the same regression there,
            # which 52 asserts agrees at 0.0 as a check on the arm plumbing.
            "fe_arms_coincide_at": "state",
            "cz_state_assignment": {
                "source": str(CZ_STATE),
                "rule": "largest 1969 population, ties to the lowest state FIPS",
                "evaluated_on": (
                    "27's full-universe assignment, repaired on the frozen support "
                    "where it points outside the retained states"
                ),
                "repairs": repairs.to_dict("records"),
            },
        },
    }


def main():
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        if not SOURCE.exists():
            raise FileNotFoundError(f"symmetric panel is missing: {SOURCE}")
        source = pd.read_parquet(SOURCE)
        units = sorted(source["unit"].unique().tolist())
        panel_states = set(int(s) for s in source["state"].unique())
        print(
            f"symmetric panel: {len(source):,} rows, {len(units):,} counties, "
            f"{len(panel_states)} states"
        )

        rungs, codes, repairs = load_geography(units, panel_states)

        TEMP.mkdir(parents=True, exist_ok=True)
        blocks, audit_rows, cut_rows, starts_by_cell = [], [], [], {}
        for period in PERIODS:
            for horizon in HORIZONS:
                frame, starts, cuts, n_capped = build_block(
                    period, horizon, source, rungs
                )
                frame = frame[COLUMNS]
                blocks.append(frame)
                cut_rows.extend(cuts)
                starts_by_cell[f"{period}_h{horizon}"] = starts

                path = TEMP / f"{period}_h{horizon}.dta"
                frame.drop(columns=["period"]).assign(
                    horizon=frame["horizon"].astype(np.int8)
                ).sort_values(["unit", "year"]).to_stata(
                    path, write_index=False, version=118
                )
                audit_rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "n_obs": len(frame),
                        "n_units": frame["unit"].nunique(),
                        "n_cz": frame["cz"].nunique(),
                        "n_states": frame["state"].nunique(),
                        "n_cz_states": frame["cz_state"].nunique(),
                        "n_windows": len(starts),
                        "first_start_year": starts[0],
                        "last_start_year": starts[-1],
                        "first_outcome_year": starts[0] + 1,
                        "last_outcome_year": starts[-1] + horizon,
                        **{f"n_capped_{k}": v for k, v in n_capped.items()},
                        "output_path": str(path),
                    }
                )

        panel = pd.concat(blocks, ignore_index=True)
        panel.to_parquet(TEMP / "panel.parquet", index=False)

        audit = pd.DataFrame(audit_rows)
        audit.to_csv(PANEL_AUDIT_OUT, index=False)
        pd.DataFrame(cut_rows).to_csv(SHARE_AUDIT_OUT, index=False)
        DESIGN_OUT.write_text(
            json.dumps(design_contract(codes, starts_by_cell, repairs), indent=2) + "\n",
            encoding="utf-8",
        )

        if len(audit) != len(PERIODS) * len(HORIZONS):
            raise AssertionError(f"{len(audit)} audit rows, expected 6")
        if not audit["n_units"].eq(EXPECTED_ANALYSIS_UNITS).all():
            raise AssertionError("a block does not carry 3,080 counties")
        if audit["n_cz"].nunique() != 1 or audit["n_states"].nunique() != 1:
            raise AssertionError("the rung sizes are not constant across blocks")

        print()
        print(audit[
            ["period", "horizon", "n_obs", "n_units", "n_cz", "n_states", "n_windows"]
        ].to_string(index=False))
        print()
        print(f"{len(panel):,} county rows -> {TEMP / 'panel.parquet'}")
        print(f"6 Stata inputs -> {TEMP}")
        print(f"design contract -> {DESIGN_OUT}")
        print(f"cutoffs by start year -> {SHARE_AUDIT_OUT}")
        print(f"panel audit -> {PANEL_AUDIT_OUT}")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
