#!/usr/bin/env python3
"""The Nakamura-Steinsson instrument for the county contract shock. Part 1.5.

13_build_county_panel.py stops at the endogenous regressor. This file builds the
instrument that Part 2.4 and every regression in Part 3 run on, and answers the
fifth check of plan section 1.5, first-stage strength at every rung.

THE INSTRUMENT

    Z_ct = s_c  x  ( G_t^nat,-c - G_t-h^nat,-c ) / Y_t-h^nat,-c

s_c is pre-period county defense exposure, a contracts-over-income ratio; the
second factor is the national military buildup rate. One functional form, the one
the plan specifies, one level down from Nakamura and Steinsson's states.

THE ACCUMULATION, ON BOTH SIDES                                        --delta

G is the CUMULATIVE stock of contract dollars, sum over s <= t of D_s, so

    dG_t = G_t - G_t-h = sum of D over t-h+1..t

at the county and at the national level alike, and the start year of the
cumulation cancels in the difference. 13_build_county_panel.py's docstring carries
why: DCADS records obligations rather than a spending flow, so D_t - D_t-h
measures award timing and turns one award into a positive shock followed by a
spurious negative one h years later. --delta flow recovers the old difference and
exists only to keep README 1.8.4's record of the abandoned design reproducible.

Accumulating leaves the aggregation property intact - a window sum of sums is a
sum of window sums - so the leave-one-out identity below and A.2.2's ω-weighting
are untouched, and check 5 is the proof rather than the claim.

It is just identified, so rescaling Z leaves the 2SLS coefficient untouched and
moves only the first-stage coefficient and its F. Nobody needs to "fix" the units.

At a higher rung the unit instrument is the omega-weighted average of the county
instruments, sum_c omega_cj Z_ct, with omega the same weight vector that
aggregates the outcome. That is what INVARIANCE_TEST_PROTOCOL.md A.2.2 requires
and it is why one functional form is enough: the aggregation property belongs to
how the unit instrument is built, not to the county instrument's algebra. The
other construction - build exposure at the unit and interact that - is computed
too, as a comparison, and the deviation between them is reported as a
construction error in the sense of plan section 3.5.

LEAVE-ONE-OUT, ON BY DEFAULT

The national aggregate nets out the county's own contribution. At the county rung
that changes almost nothing. At the state and division rungs, where California or
the South Atlantic is a large share of national defense spending, an instrument
that contains the unit's own shock is mechanically endogenous, and the same
instrument has to run at every rung.                                   --no-loo

THE NATIONAL SERIES IS OURS, NOT NIPA

D_t^nat is the sum of dollars_real over the fixed positive-employment CCG set,
under no dollar floor
and the same geo_class == "county" filter as the shock. The protocol's minimum
audit wants the treatment and the instrument built from the same retained rows,
and it makes the leave-one-out subtraction exact rather than approximate.

THREE EXPOSURES                                                      --exposure

  fixed       sum of 1969-1974 real contracts over sum of 1969-1974 real income.
              BEA county income starts in 1969 and the estimation panel starts in
              1975, so this window touches no estimation year: predetermined by
              construction rather than by argument. 611 counties come out at zero.
  rolling     the same ratio of sums over a trailing ROLL_K-year window ending at
              t-h-1, needing at least ROLL_MIN usable years. Tracks a defense base
              that moves over thirty-two years.
  rolling_t5  the same window, ending at t-5 WHATEVER h IS. Added 2026-08-13.

All are ratios of sums, so a single low-income year cannot dominate them.

WHY rolling_t5 EXISTS, AND WHY ITS END YEAR IS A CONSTANT

`rolling` ends at t-h-1, so its clearance from the sample is one year at every
horizon and the window MOVES with h. A change in beta across horizons therefore
mixes a change in the horizon with a change in where the loading was measured.
`rolling_t5` holds the window at t-9..t-5 for every h, so the loading sits the
same distance from the sample at h=2, 3 and 4 and only the horizon varies.

t-5 and not t-4 or t-6: the shock at t sums D over {t-h+1..t} and divides by Y at
t-h, so the earliest year the design touches is t-h. At h <= 4 that is t-4, and a
window ending t-5 clears it by one year at the widest horizon and by more at the
narrower ones. At h = 4 the two rolling exposures COINCIDE exactly, because
h+1 = 5 there, which is what check 8b asserts.

A third exposure, `ns` - the mean of the ANNUAL ratio D_ct/Y_ct over 1975-2006 -
was built until 2026-08-10 and is NOT built any more. Its window is the estimation
sample, so the loading is a function of the same D_ct the shock instruments and it
violates the exclusion restriction by construction, whatever its first stage.
README 1.8.1 carries the argument, so that nobody rebuilds it.

TWO HORIZONS OR THREE                                                --horizon

1, 2 and 3 are all built. At h=1 the accumulation window is one year, so
dG_ct = D_ct and the shock is bare contract intensity; the horizons differ only in
the width of the window and the lag of the denominator.

THREE ESTIMATION WINDOWS                                             --periods

The rung sweep runs 1975-2006, 1975-1998 and 1998-2006. The split is at the
post-Cold-War flattening of the national buildup, which the first stage does not
survive: see README 1.8.5. 1998 sits in BOTH sub-samples by request. step_2's
stricter convention drops the straddle years instead - PRE_END 1997, POST_START
2000, so that t-h is also on the far side of the SIC-to-NAICS employment break -
which matters for the outcome and not for a first stage that uses no industry data.

Counties with no pre-period defense keep an instrument of zero. They are counted
in the report and never dropped, because plan section 3.5.1 freezes the county
support and the protocol's question 7 asks for the same sample at every rung.

Output:
    dc.PANEL_OUT_DIR/instrument_ccg_year.parquet   long on (ccg, year, h)
    dc.LOG_DIR/instrument_report.csv               per year
    dc.LOG_DIR/instrument_first_stage.csv          the rung sweep, check 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

FIRST_YEAR = 1975          # the estimation panel, as in 13_build_county_panel.py
LAST_YEAR = 2018           # was 2006; the contract panel is spliced at dc.SEAM_YEAR
EXPOSURE_FIRST = 1969      # BEA county personal income starts here and not before

# TWO FIXED WINDOWS, one per era.
#
# A fixed exposure is only usable on a sample it strictly precedes, and 1969-74
# precedes everything here - but a loading measured in the Nixon administration
# is a weak description of where defense sat in 2010. The post-1998 window
# therefore gets its own pre-period, 1993-98, taken from the end of the
# post-Cold-War drawdown and before the sample it loads.
#
# The key "fixed" is deliberately unchanged rather than renamed to "fixed_6974":
# 16_floor_comparison.py imports PRE_WINDOW, step_2's winsorization exhibit reads
# the column z_fixed, and this file's own construction-error exhibit and summary
# table both filter on the literal string "fixed". A rename would leave all of
# them running and silently empty.
FIXED_WINDOWS = {"fixed": (1969, 1974), "fixed_9398": (1993, 1998)}
PRE_WINDOW = FIXED_WINDOWS["fixed"]   # the name 16_floor_comparison.py imports
DEFAULT_FLOOR = 0          # no floor as of 2026-08-10, and 13 uses the same one
DEFAULT_DELTA = "cumulative"   # G is a cumulative stock as of 2026-08-10; see the docstring
DELTAS = ("cumulative", "flow")
ROLL_K = 5                 # trailing years in the rolling exposure window
ROLL_MIN = 3               # usable years the window needs before it yields a number

# The two rolling exposures differ in ONE number: where the window ends. None
# means t-h-1, the original, whose clearance from the sample tracks the horizon;
# an integer means t-that, fixed across horizons. See the docstring for why 5.
#
# Written as a dict rather than as two functions so that the only difference
# between them is visible in one place and a third variant is a line rather than a
# copy of rolling_exposure.
ROLL_END_LAG = {"rolling": None, "rolling_t5": 5}
ROLLING = tuple(ROLL_END_LAG)

HORIZONS = (1, 2, 3)
EXPOSURES = ("fixed", "fixed_9398") + ROLLING

# The estimation windows the rung sweep reports.
#
# "1998-2006" is kept so README 1.8.5 Panel C stays reproducible, but it is NOT
# in DEFAULT_PERIODS: pass --periods to ask for it. That is the whole mechanism -
# --periods already exists with choices=list(PERIODS), so a separate legacy flag
# would be a second way to do the same thing.
#
# The post-1998 window starts at 1999 and not 1998 because 1993-98 has to sit
# strictly before it, and it is further trimmed by the horizon: see
# admissible_periods.
PERIODS = {"1975-2006": (1975, 2006),   # frozen, reproduces README 1.8.5
           "1975-1998": (1975, 1998),   # pre-1998
           "1998-2006": (1998, 2006),   # legacy, not in the default sweep
           "2002-2018": (2002, 2018)}   # post-1998, the extension
DEFAULT_PERIODS = ("1975-2006", "1975-1998", "2002-2018")

# Which exposure may be reported on which window.
#
# `fixed` (1969-74) is barred after 1998 BY INSTRUCTION, not by arithmetic:
# 1969-74 is disjoint from 1999-2018 and the predeterminedness guard below passes
# it. The bar is a judgement that a thirty-year-old loading is not a credible
# description of post-1998 defense, and it is enforced by its own named check so
# that it is not mistaken for the guard. Its cost is real and is recorded in
# README section 1.8: pre and post share no fixed exposure, so a pre/post
# difference in beta mixes a change in the multiplier with a change in the
# loading, and only `rolling` is common to both.
# Both rolling exposures are admissible everywhere for the same reason: they are
# predetermined by construction at every t, so no period can contaminate them.
ADMISSIBLE = {"1975-2006": {"fixed"} | set(ROLLING),
              "1975-1998": {"fixed"} | set(ROLLING),
              "1998-2006": {"fixed"} | set(ROLLING),
              "2002-2018": {"fixed_9398"} | set(ROLLING)}

# WHY THE POST-1998 WINDOW STARTS AT 2002 AND NOT 1999
#
# Because 1993-98 has to sit strictly before everything the shock touches, and
# the shock reaches further back than the estimation year. At t the numerator
# sums D over {t-h+1..t} and the denominator is Y at t-h, so the earliest year a
# period starting at y0 touches is y0 - h. With the exposure window ending in
# 1998 and h running to 3, y0 must exceed 2001.
#
#     y0 = 1999, h = 1   ->  divides by Y_1998, inside the window
#     y0 = 2001, h = 3   ->  divides by Y_1998, inside the window
#     y0 = 2002, h = 3   ->  divides by Y_1999, clear
#
# Overlapping would make the instrument a function of the same income that scales
# the shock it instruments - which is the exact defect that retired the `ns`
# exposure (README 1.8.1), and it is not less of one for being smaller.
#
# The bound is taken at max(HORIZONS) rather than per horizon so that h=1, 2 and
# 3 are reported on ONE sample. Letting each horizon start where it individually
# became clean would mean a change in beta across horizons mixed a change in the
# window with a change in the sample.
#
# The cost is three years off the front of an already short window, and it is
# the exposure window's end year that buys them: 1993-95 would allow a 1999
# start. That trade is left to whoever revisits this, and stated in README 1.8.

# Kalawao HI has no BEA income in any year and draws $0 of contracts in every
# year, which 12_build_county_income.py and 13_build_county_panel.py both assert.
# Restated here so this file declares its own universe.
INCOME_GAP_CCGS = (15005,)

# The nine Census divisions, which is what the plan means by "9 Census regions".
# Fifty states plus the District of Columbia, and nothing else: the CCG universe
# carries no territory.
DIVISION = {}
for _name, _states in (
        ("New England", "09 23 25 33 44 50"),
        ("Middle Atlantic", "34 36 42"),
        ("East North Central", "17 18 26 39 55"),
        ("West North Central", "19 20 27 29 31 38 46"),
        ("South Atlantic", "10 11 12 13 24 37 45 51 54"),
        ("East South Central", "01 21 28 47"),
        ("West South Central", "05 22 40 48"),
        ("Mountain", "04 08 16 30 32 35 49 56"),
        ("Pacific", "02 06 15 41 53")):
    for _s in _states.split():
        DIVISION[int(_s)] = _name


def check(failures, ok, num, label, detail=""):
    """Print one numbered PASS/FAIL line and record the failure."""
    print(f"    [{num}] {label:<54s} {detail:<32s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


def load_rungs(universe):
    """ccg -> commuting zone, state, Census division, one row per CCG.

    The commuting zone comes from ccz_id in the crosswalk that
    10_build_county_universe.py wrote, not from the raw county-to-CZ bridge. A
    CCG is a group of counties and some groups straddle a 1990 commuting-zone
    boundary; that script already merged the straddled zones with a union-find so
    that every CCG sits inside exactly one zone, and ccz_id is the result. Using
    the raw bridge here would put one CCG in two zones and silently break the
    frozen-support rule.
    """
    x = pd.read_csv(dc.CCG_CROSSWALK, dtype={"ccg_id": str, "ccz_id": str})
    x = x[x["ccg_id"] != "unplaceable"]
    cz = x.dropna(subset=["ccz_id"]).groupby("ccg_id")["ccz_id"].first()
    rungs = pd.DataFrame({"ccg": cz.index.astype(np.int64), "cz": cz.to_numpy()})
    rungs["state"] = rungs["ccg"] // 1000
    rungs["division"] = rungs["state"].map(DIVISION)
    rungs = rungs[rungs["ccg"].isin(universe)].sort_values("ccg").reset_index(drop=True)
    return rungs


def build_levels(floor, fold_tq76, first, last):
    """Contract dollars and BEA income on a rectangular CCG x year spine.

    The span reaches back to EXPOSURE_FIRST rather than to FIRST_YEAR because the
    pre-period exposure needs years the estimation sample does not contain. The
    estimation years inside it must come out identical to county_panel.parquet,
    which is check 4.
    """
    _, _, universe = dc.load_analysis_ccg_map()
    years = list(range(first, last + 1))
    contracts = dc.build_contract_panel(universe, years, floor=floor,
                                        fold_tq76=fold_tq76)
    income = pd.read_parquet(dc.PANEL_OUT_DIR / "income_ccg_year.parquet")
    income = income[income["year"].between(first, last)]

    spine = pd.MultiIndex.from_product(
        [np.sort(universe), years], names=["ccg", "year"]).to_frame(index=False)
    levels = spine.merge(contracts[["ccg", "year", "dollars_real", "dollars_nominal"]],
                         on=["ccg", "year"], how="left")
    levels = levels.merge(income[["ccg", "year", "income_real"]],
                          on=["ccg", "year"], how="left")
    return levels.sort_values(["ccg", "year"]).reset_index(drop=True), universe, years


def fixed_window_sums(levels, window):
    """Window contract dollars and window income, one row per CCG.

    Returned as the two components rather than the ratio because the aggregation
    exhibit needs to add them up inside a larger unit before dividing, and a
    ratio cannot be added.
    """
    w = levels[levels["year"].between(*window)]
    return pd.DataFrame({
        "pre_D": w.groupby("ccg")["dollars_real"].sum(),
        "pre_Y": w.groupby("ccg")["income_real"].sum(),
    }).reset_index()


def fixed_exposure(levels, window, name="fixed"):
    """s_c = sum of window contracts over sum of window income, one per CCG."""
    s = fixed_window_sums(levels, window).set_index("ccg")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (s["pre_D"] / s["pre_Y"]).rename(f"exposure_{name}")


def contaminated_years(period, exposure, h):
    """Estimation years whose shock reaches back into the exposure window.

    NOT the same as "years inside the window". The shock at t sums D over
    {t-h+1..t} and divides by Y at t-h, so year t reaches back to t-h. A fixed
    exposure ending at w1 is therefore a function of the same dollars the shock
    carries for every t with t - h <= w1.

    The old guard compared the exposure window against the estimation YEARS
    alone, which is why it passed designs it should not have. On (1993,1998)
    with a 1999 start it finds the two disjoint, while at h=2 the 1999 shock
    sums {1998, 1999} over Y_1997 - both inside the window.

    The contamination is a LEADING-EDGE effect: it hits the first years of a
    period and then stops. Which years, and how many, is the number worth
    reporting - "overlaps" alone cannot distinguish three years at the front of
    a thirty-two-year panel from a window that ruins the whole sample.
    """
    if exposure not in FIXED_WINDOWS:
        return []
    w1 = FIXED_WINDOWS[exposure][1]
    y0, y1 = PERIODS[period]
    return [y for y in range(y0, y1 + 1) if y - h <= w1]


def check_predetermined(period, exposure, horizons):
    """(hard_ok, contaminated, detail).

    hard_ok is False only when EVERY year of the period is contaminated - a
    window that does not precede its sample at all, which is the defect that
    retired the `ns` exposure (README 1.8.1). A leading edge is reported with its
    years rather than failing the run, because the frozen 1975-2006 design has
    one at h=2 and h=3 and this check was written after that design was
    published, not before it.
    """
    if exposure not in FIXED_WINDOWS:
        return True, {}, "rolling, predetermined by construction"
    w0, w1 = FIXED_WINDOWS[exposure]
    y0, y1 = PERIODS[period]
    bad = {h: contaminated_years(period, exposure, h) for h in horizons}
    bad = {h: v for h, v in bad.items() if v}
    n_years = y1 - y0 + 1
    hard_ok = all(len(v) < n_years for v in bad.values())
    if not bad:
        detail = f"window {w0}-{w1}, period {y0}-{y1}, clean at every h"
    else:
        worst = max(bad, key=lambda k: len(bad[k]))
        detail = (f"window {w0}-{w1}: h={worst} reaches it in "
                  f"{len(bad[worst])} of {n_years} years "
                  f"({bad[worst][0]}-{bad[worst][-1]})")
    return hard_ok, bad, detail


def roll_lag(h, end_lag):
    """How many years to shift a k-wide trailing window back by.

    ONE definition, used by the fast path, the slow path and every caller, so the
    two rolling exposures cannot drift apart. end_lag None reproduces the original
    t-h-1 window exactly; an integer pins the window's last year at t-end_lag.
    """
    return h + 1 if end_lag is None else end_lag


def rolling_exposure(levels, h, k, min_years, end_lag=None, name="rolling"):
    """s_ct over the k-year window ending at t-roll_lag(h, end_lag), aligned to rows.

    rolling(k) at row t sums t-k+1..t, so shifting the result by `lag` lands the
    window on t-lag-k+1..t-lag. With end_lag None, lag = h+1 and the window's last
    year is t-h-1, one year before the lagged period the shock is measured from -
    the original construction, byte-for-byte. With end_lag = 5 the last year is
    t-5 at every horizon, which is `rolling_t5`.

    Either way the window is strictly predetermined, and check 8 recomputes it the
    slow way rather than trusting this.
    """
    lag = roll_lag(h, end_lag)
    g = levels.groupby("ccg", sort=False)
    out = {}
    for col in ("dollars_real", "income_real"):
        roll = (g[col].rolling(k, min_periods=min_years).sum()
                .reset_index(level=0, drop=True).sort_index())
        out[col] = roll.groupby(levels["ccg"]).shift(lag)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (out["dollars_real"] / out["income_real"]).rename(f"exposure_{name}")


def slow_rolling_exposure(levels, h, k, min_years, ccgs, end_lag=None):
    """The same number, computed by an explicit loop over the window years.

    Existing only to be compared against rolling_exposure. A shift-by-one error
    in the fast version is invisible in the output and fatal to the design, since
    it would put the county's own contemporaneous contracts inside its exposure -
    and the arithmetic here is written from the window's DEFINITION rather than
    from a shift, which is what makes it an independent witness.
    """
    lag = roll_lag(h, end_lag)
    sub = levels[levels["ccg"].isin(ccgs)]
    d = sub.set_index(["ccg", "year"])["dollars_real"].to_dict()
    y = sub.set_index(["ccg", "year"])["income_real"].to_dict()
    rows = []
    for ccg, year in sub[["ccg", "year"]].itertuples(index=False):
        window = [t for t in range(year - lag - k + 1, year - lag + 1)
                  if (ccg, t) in d]
        if len(window) < min_years:
            rows.append((ccg, year, np.nan))
            continue
        dd = sum(d[(ccg, t)] for t in window)
        yy = sum(y[(ccg, t)] for t in window)
        rows.append((ccg, year, dd / yy if yy else np.nan))
    return pd.DataFrame(rows, columns=["ccg", "year", "slow"])


def rolling_exposures(levels, h, k, min_years, names=ROLLING):
    """{name: ndarray} for every rolling variant asked for, at this horizon.

    The one place a caller has to touch to get both, so 19, 20, 23 and 26 cannot
    each build a private version of `rolling_t5` with its own end_lag.
    """
    return {n: rolling_exposure(levels, h, k, min_years,
                                end_lag=ROLL_END_LAG[n], name=n).to_numpy()
            for n in names}


def window_sum(frame, col, h, by="ccg"):
    """The h-year rolling sum of `col` within `by`, aligned to the frame's rows.

    This is dG under `cumulative`: the cumulation's start year cancels in
    G_t - G_t-h, so the difference of cumulative sums is the same number wherever
    it is defined and the rolling form is additionally defined in the h-1 leading
    years. 13_build_county_panel.py asserts the two are equal (its check 4b) and
    that the value does not move with the start year (4c).
    """
    return (frame.groupby(by, sort=False)[col].rolling(h).sum()
            .reset_index(level=0, drop=True).sort_index())


def build_horizon(levels, exposures, h, first, last, loo, delta=DEFAULT_DELTA):
    """One horizon's rows: the shock, the outcome, the national buildup, the Zs."""
    p = levels.copy()
    g = p.groupby("ccg", sort=False)
    for col in ("dollars_real", "income_real", "emp"):
        p[f"{col}_lag"] = g[col].shift(h)
    p["year_lag"] = g["year"].shift(h)

    nat = p.groupby("year").agg(nat_D=("dollars_real", "sum"),
                                nat_Y=("income_real", "sum"))
    # The same accumulation on both sides. nat_D is one row per year already, so the
    # rolling sum runs over years rather than within a unit.
    if delta == "cumulative":
        nat["nat_dD"] = nat["nat_D"].rolling(h).sum()
        p["own_dD"] = window_sum(p, "dollars_real", h)
    else:
        nat["nat_dD"] = nat["nat_D"].diff(h)
        p["own_dD"] = p["dollars_real"] - p["dollars_real_lag"]
    nat["nat_Y_lag"] = nat["nat_Y"].shift(h)
    p = p.merge(nat[["nat_dD", "nat_Y_lag"]], left_on="year", right_index=True, how="left")

    if loo:
        p["nat_dD_used"] = p["nat_dD"] - p["own_dD"]
        p["nat_Y_lag_used"] = p["nat_Y_lag"] - p["income_real_lag"]
    else:
        p["nat_dD_used"] = p["nat_dD"]
        p["nat_Y_lag_used"] = p["nat_Y_lag"]

    with np.errstate(divide="ignore", invalid="ignore"):
        p["shock"] = p["own_dD"] / p["income_real_lag"]
        p["emp_growth"] = (p["emp"] - p["emp_lag"]) / p["emp_lag"]
        p["buildup"] = p["nat_dD_used"] / p["nat_Y_lag_used"]
        for name, s in exposures.items():
            e = s if isinstance(s, pd.Series) and s.index.name == "ccg" else None
            p[f"exposure_{name}"] = (p["ccg"].map(e) if e is not None else s)
            p[f"z_{name}"] = p[f"exposure_{name}"] * p["buildup"]

    p["emp_growth"] = p["emp_growth"].replace([np.inf, -np.inf], np.nan)

    # The seam, masked here as well as in 13. This file does NOT read 13's dG - it
    # recomputes the shock from levels a few lines above - so the mask has to be
    # applied a second time or the post-seam frames would carry the straddling
    # years that 13 dropped. Same rule, same years: dG at t sums {t-h+1..t}, so it
    # mixes sources exactly in SEAM_YEAR .. SEAM_YEAR+h-2.
    #
    # The national series is masked too. A county-level mask alone would leave
    # buildup contaminated, and buildup is what every z is built from - so the
    # instrument would carry the seam even where the shock no longer did.
    seam = [y for y in range(dc.SEAM_YEAR, dc.SEAM_YEAR + h - 1)]
    if seam:
        m = p["year"].isin(seam)
        p.loc[m, ["shock", "own_dD", "buildup", "nat_dD", "nat_dD_used"]] = np.nan
        for name in exposures:
            p.loc[m, f"z_{name}"] = np.nan

    p["h"] = h
    return p[p["year"].between(first, last)].reset_index(drop=True)


# --------------------------------------------------------------------------------
# The rung sweep. Plan section 1.5 check 5.
# --------------------------------------------------------------------------------

FE_SPECS = (("none", None), ("year", "year"), ("unit+year", "unit + year"))


def fit_rung(frame, fe, cluster="unit", with_iv=True, weights=None, **feols_kw):
    """First-stage coefficient, cluster-robust F, and the 2SLS point estimate.

    The instrument is single, so the first-stage t squared IS the Kleibergen-Paap
    F under the same variance estimator, which is why it is computed that way
    rather than by asking for a diagnostic that would have to be matched to this
    clustering by hand.

    feols_kw goes straight to pf.feols, both calls. Added for 20_first_stage_grid,
    which pins the demeaner the way 22_winsor_construction_error.py does; passing
    nothing reproduces this function's previous behaviour exactly, so 15, 16, 17
    and 19 are unaffected. It is a passthrough and not a copy on purpose: a second
    fit_rung is how this file and its callers would drift into measuring different
    things while both looking right.
    """
    import pyfixest as pf
    vcov = {"CRV1": cluster}
    kw = {"vcov": vcov, **feols_kw}
    if weights is not None:
        kw["weights"] = weights
    fs_f = "x ~ z" if fe is None else f"x ~ z | {fe}"
    fs = pf.feols(fs_f, frame, **kw)
    b = float(fs.coef()["z"])
    se = float(fs.se()["z"])
    out = {"n_obs": int(fs._N), "fs_coef": b, "fs_se": se,
           "fs_F": (b / se) ** 2 if se > 0 else np.nan,
           "beta": np.nan, "beta_se": np.nan}
    if with_iv:
        iv_f = "y ~ 1 | x ~ z" if fe is None else f"y ~ 1 | {fe} | x ~ z"
        iv = pf.feols(iv_f, frame, **kw)
        out["beta"] = float(iv.coef()["x"])
        out["beta_se"] = float(iv.se()["x"])
    return out


def collapse(county, unit_col, weight_col):
    """Aggregate the outcome, the shock and the instrument with ONE weight vector.

    omega_cj is the county's share of its unit's `weight_col`, and the SAME omega
    multiplies the shock and the instrument - protocol A.2.2. The outcome is
    always aggregated with lagged employment, because unit employment growth IS
    the employment-share-weighted average of its counties' employment growth and
    nothing else reproduces it.

    Also carried up: the pre-window contract and income sums, and the unit's own
    contract change and lagged income, which is what the alternative construction
    of the unit instrument needs - unit exposure interacted with a unit-level
    leave-one-out buildup, instead of the county instruments averaged.
    """
    # dict.fromkeys and not a set, because weight_col is one of the columns
    # already named and a duplicate label would make d[weight_col] a frame.
    cols = list(dict.fromkeys(
        [unit_col, "year", weight_col, "emp_lag", "emp_growth", "shock", "z",
         "pre_D", "pre_Y", "own_dD", "income_real_lag", "nat_dD", "nat_Y_lag"]))
    d = county[cols].copy()
    d = d.rename(columns={unit_col: "unit"})
    key = ["unit", "year"]
    tot = d.groupby(key)[weight_col].transform("sum")
    emp_tot = d.groupby(key)["emp_lag"].transform("sum")
    d["omega"] = d[weight_col] / tot
    d["omega_emp"] = d["emp_lag"] / emp_tot
    d["_y"] = d["omega_emp"] * d["emp_growth"]
    d["_x"] = d["omega"] * d["shock"]
    d["_z"] = d["omega"] * d["z"]
    agg = d.groupby(key).agg(
        y=("_y", "sum"), x=("_x", "sum"), z=("_z", "sum"),
        omega_sum=("omega", "sum"), omega_emp_sum=("omega_emp", "sum"),
        pre_D=("pre_D", "sum"), pre_Y=("pre_Y", "sum"),
        unit_dD=("own_dD", "sum"), unit_Y_lag=("income_real_lag", "sum"),
        nat_dD=("nat_dD", "first"), nat_Y_lag=("nat_Y_lag", "first"),
    ).reset_index()
    return agg


def rung_sweep(county_all, rungs, exposures, horizons, cell_counts, periods=PERIODS,
               verbose=True):
    """Every rung, every exposure, every horizon, every period, every FE setting.

    The aggregation rungs are what check 5 is about: going up genuinely rebuilds
    both the regressor and the instrument, and the cross-sectional variation the
    instrument lives on thins out as the units get bigger. The two cell rungs are
    reported next to them for completeness and are NOT a second first stage: the
    instrument is constant across a county's industries (protocol A.1.1), so the
    cell regression is the county regression reweighted by the county's cell
    count, which is exactly how it is computed here.

    The period loop sits INSIDE the finiteness screen, so the county support is
    frozen the same way in every sub-sample and a period differs from the full
    sample only in which years it contains.
    """
    rows = []
    for h in horizons:
        for name in exposures:
            full = county_all[county_all["h"] == h].copy()
            full["z"] = full[f"z_{name}"]
            full = full.merge(rungs, on="ccg", how="left")
            keep = np.isfinite(full[["shock", "emp_growth", "z", "income_real_lag",
                                     "emp_lag"]].to_numpy()).all(axis=1)
            full = full[keep & (full["emp_lag"] > 0) & (full["income_real_lag"] > 0)]
            if verbose:
                print(f"    h={h} {name:<8s} support {full['ccg'].nunique():,} CCGs, "
                      f"{len(full):,} county-years")

            for pname, (y0, y1) in periods.items():
              # ADMISSIBLE, enforced here and announced rather than left blank. An
              # omitted cell that simply does not appear reads as a cell that was
              # run and found nothing, which is the opposite of what it means.
              if name not in ADMISSIBLE.get(pname, set(exposures)):
                  if verbose:
                      print(f"      omitted: {pname} x {name} x h={h} - not "
                            f"admissible for this period")
                  continue
              base = full[full["year"].between(y0, y1)]
              for rung, unit_col in (("county", "ccg"), ("commuting zone", "cz"),
                                     ("state", "state"), ("Census division", "division")):
                # "unweighted" and not "n/a": pandas reads n/a back as a missing
                # value, which would silently delete the county rung from every
                # later reading of this file.
                schemes = (("unweighted", "income_real_lag"),) if rung == "county" else \
                          (("income", "income_real_lag"), ("employment", "emp_lag"))
                for wname, wcol in schemes:
                    if rung == "county":
                        d = base.rename(columns={"emp_growth": "y", "shock": "x"})
                        d["unit"] = d["ccg"]
                        d = d[["unit", "year", "y", "x", "z"]]
                        omega_dev = 0.0
                        con_err = con_corr = con_sd = np.nan
                    else:
                        d = collapse(base, unit_col, wcol)
                        omega_dev = float(np.abs(
                            pd.concat([d["omega_sum"], d["omega_emp_sum"]]) - 1).max())
                        con_err = con_corr = con_sd = np.nan
                        if name == "fixed":
                            # The other construction: build exposure at the unit
                            # and interact it with a unit-level leave-one-out
                            # buildup, instead of averaging the county
                            # instruments. Plan 3.5 wants the difference between
                            # two constructions named and measured, so it is
                            # measured here rather than assumed away. Only the
                            # headline exposure gets this, because the exhibit is
                            # about the aggregation step and not about which
                            # exposure went into it.
                            #
                            # Scaled by the instrument's own standard deviation
                            # and not by its level, so that the statistic means the
                            # same thing at every rung and in every period whatever
                            # the level of the buildup happens to be there.
                            with np.errstate(divide="ignore", invalid="ignore"):
                                direct = ((d["pre_D"] / d["pre_Y"])
                                          * (d["nat_dD"] - d["unit_dD"])
                                          / (d["nat_Y_lag"] - d["unit_Y_lag"]))
                            good = np.isfinite(direct) & np.isfinite(d["z"])
                            a, b = direct[good], d["z"][good]
                            sd = float(b.std())
                            if len(a) > 2 and sd > 0:
                                con_err = float(np.abs(a - b).max() / sd)
                                con_corr = float(np.corrcoef(a, b)[0, 1])
                                con_sd = float(a.std() / sd)
                    for fe_name, fe in FE_SPECS:
                        r = fit_rung(d, fe)
                        rows.append(dict(rung=rung, n_units=int(d["unit"].nunique()),
                                         weights=wname, h=h, period=pname,
                                         exposure=name, fe=fe_name, **r,
                                         omega_dev=omega_dev,
                                         construction_error=con_err,
                                         construction_corr=con_corr,
                                         construction_sd_ratio=con_sd))

              for cell_rung, counts in cell_counts.items():
                d = base.merge(counts, on=["ccg", "year"], how="inner")
                d = d.rename(columns={"emp_growth": "y", "shock": "x"})
                d["unit"] = d["ccg"]
                d = d[["unit", "year", "y", "x", "z", "n_cells"]]
                d = d[d["n_cells"] > 0]
                for fe_name, fe in FE_SPECS:
                    r = fit_rung(d, fe, with_iv=False, weights="n_cells")
                    r["n_obs"] = int(d["n_cells"].sum())
                    rows.append(dict(rung=cell_rung, n_units=int(d["unit"].nunique()),
                                     weights="cells", h=h, period=pname,
                                     exposure=name, fe=fe_name, **r, omega_dev=0.0,
                                     construction_error=np.nan,
                                     construction_corr=np.nan,
                                     construction_sd_ratio=np.nan))
    return pd.DataFrame(rows)


def load_cell_counts(years, rungs_wanted):
    """Cells per CCG-year, the only thing the cell rungs need.

    Materialising 24 million county-industry rows to run a regression whose
    regressor and instrument are both constant within a county would be a waste:
    the cell OLS moments are the county moments weighted by the cell count, and
    that equality is exact rather than an approximation.
    """
    out = {}
    for label, folder in rungs_wanted.items():
        frames = []
        for y in years:
            path = dc.EMP_OUT_DIR / folder / f"y{y}.parquet"
            if not path.exists():
                continue
            c = pd.read_parquet(path, columns=["ccg"])
            frames.append(c.groupby("ccg").size().rename("n_cells")
                          .reset_index().assign(year=y))
        if frames:
            out[label] = pd.concat(frames, ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--last-year", type=int, default=LAST_YEAR)
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="nominal dollar floor on |action|; must match 13")
    ap.add_argument("--drop-tq76", action="store_true")
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS),
                    help="horizons to build; the headline is 2, as in NS")
    ap.add_argument("--exposure", nargs="+", default=list(EXPOSURES),
                    choices=list(EXPOSURES))
    ap.add_argument("--delta", choices=DELTAS, default=DEFAULT_DELTA,
                    help="cumulative: G is a stock, dG is the h-year window sum. "
                         "flow: dG = D_t - D_t-h, the pre-2026-08-10 default")
    ap.add_argument("--periods", nargs="+", default=list(DEFAULT_PERIODS),
                    choices=list(PERIODS),
                    help="estimation windows the rung sweep reports")
    ap.add_argument("--roll-k", type=int, default=ROLL_K)
    ap.add_argument("--roll-min", type=int, default=ROLL_MIN)
    ap.add_argument("--no-loo", action="store_true",
                    help="use the national total including the county's own dollars")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="build the instrument but skip plan check 5")
    args = ap.parse_args()

    failures = []
    fold = not args.drop_tq76
    loo = not args.no_loo
    horizons = sorted(set(args.horizon))
    names = [e for e in EXPOSURES if e in set(args.exposure)]
    periods = {k: v for k, v in PERIODS.items() if k in set(args.periods)}
    print(f"\nthe Nakamura-Steinsson instrument   {args.first_year}-{args.last_year}, "
          f"h={horizons}, delta {args.delta}, exposure {names}, "
          f"national {'leave-one-out' if loo else 'including own'}")
    print(f"    sweep periods: {', '.join(periods)}")

    # --- levels, reaching back before the estimation window ---------------------------
    print("\n[0] contracts and income, 1969 onward")
    levels, universe, _ = build_levels(args.floor, fold, EXPOSURE_FIRST, args.last_year)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)
    n_ccg = len(universe)
    print(f"    {n_ccg:,} CCGs x {levels['year'].nunique()} years = {len(levels):,} rows")

    # --- exposures --------------------------------------------------------------------
    print("\n[1] pre-period defense exposure")
    fixed_by_name = {}
    for name, window in FIXED_WINDOWS.items():
        if name not in names:
            continue
        s = fixed_exposure(levels, window, name)
        fixed_by_name[name] = s
        # Counted apart, because they are different things: a county that drew no
        # pre-period defense has an exposure of zero and belongs in the panel, while
        # Kalawao has no income denominator at all and has none.
        print(f"    {name:<11} {window[0]}-{window[1]}   mean {s.mean():.5f}   "
              f"sd {s.std():.5f}   p90 {s.quantile(0.90):.5f}   "
              f"{int((s == 0).sum()):,} counties at zero, "
              f"{int(s.isna().sum())} with no income denominator")
    fixed = fixed_by_name.get("fixed")
    for n in [r for r in ROLLING if r in names]:
        end = ROLL_END_LAG[n]
        print(f"    {n:<11} trailing {args.roll_k} years ending "
              f"{'t-h-1' if end is None else f't-{end}'}, "
              f"at least {args.roll_min} usable")

    print("\n    predeterminedness, per (period, exposure). The shock at t reaches")
    print("    back to t-h, so a fixed window is clean only if it ends before that.")
    for pname in periods:
        for ename in names:
            if ename not in FIXED_WINDOWS:
                continue
            admitted = ename in ADMISSIBLE.get(pname, set())
            hard_ok, bad, detail = check_predetermined(pname, ename, horizons)
            if not admitted:
                print(f"      {pname:<11} {ename:<11} not admissible here; {detail}")
                continue
            check(failures, hard_ok, f"0.{pname}.{ename}",
                  "the exposure window precedes its own sample", detail)
            for h in sorted(bad):
                yrs = bad[h]
                print(f"        WARNING {pname} x {ename} x h={h}: "
                      f"{len(yrs)} leading year(s) {yrs[0]}-{yrs[-1]} have a shock "
                      f"window that dips into the exposure window.")
                print(f"                Those years' first stage is mechanically "
                      f"strengthened. Pre-existing in the frozen design, reported "
                      f"here, not silently dropped.")

    # --- one frame per horizon --------------------------------------------------------
    print("\n[2] the shock, the buildup and the instrument, one horizon at a time")
    frames = []
    roll_names = [r for r in ROLLING if r in names]
    for h in horizons:
        exposures = dict(fixed_by_name)
        exposures.update(rolling_exposures(levels, h, args.roll_k, args.roll_min,
                                           names=roll_names))
        f = build_horizon(levels, exposures, h, args.first_year, args.last_year, loo,
                          delta=args.delta)
        v = f["shock"].dropna()
        by_year = f.groupby("year")["buildup"].mean()
        print(f"    h={h}   {len(v):,} CCG-years carry a shock   sd {v.std():.5f}"
              f"   buildup {by_year.min():+.5f} to {by_year.max():+.5f}"
              f"   peak {int(by_year.idxmax())}   {int((by_year < 0).sum())} negative years")
        for n in roll_names:
            r = f.loc[f[f"exposure_{n}"].notna(), "year"]
            print(f"           {n} exposure defined from {int(r.min())}, "
                  f"{f[f'exposure_{n}'].notna().sum():,} of {len(f):,} CCG-years")
        frames.append(f)
    county_all = pd.concat(frames, ignore_index=True)
    county_all = county_all.merge(fixed_window_sums(levels, PRE_WINDOW),
                                  on="ccg", how="left")

    # --- the checks -------------------------------------------------------------------
    print("\n[3] the checks")
    years = list(range(args.first_year, args.last_year + 1))
    expected = n_ccg * len(years) * len(horizons)
    check(failures, len(county_all) == expected, 1,
          "row count is CCGs x years x horizons", f"{len(county_all):,} of {expected:,}")
    order_ok = True
    for h, f in zip(horizons, frames):
        order_ok &= np.array_equal(f["ccg"].to_numpy(), np.repeat(np.sort(universe), len(years)))
        order_ok &= np.array_equal(f["year"].to_numpy(), np.tile(np.array(years), n_ccg))
    check(failures, order_ok, "1b", "row ORDER is (ccg, year) ascending and complete",
          "identical to the spine")

    lag_ok = all(((f["year"] - f["year_lag"]).dropna() == h).all()
                 for h, f in zip(horizons, frames))
    check(failures, lag_ok, 2, "the h-year lag is a true h-year lag",
          f"h={horizons}")

    panel = pd.read_parquet(dc.PANEL_OUT_DIR / "county_panel.parquet")
    if 1 in horizons:
        m = county_all[county_all["h"] == 1].merge(
            panel[["ccg", "year", "shock"]], on=["ccg", "year"],
            how="inner", suffixes=("", "_13"))
        # 13 builds its lags inside 1975-2006, so its first year has no lagged
        # INCOME and therefore no shock. This file reaches back to 1969 for the
        # exposure window and has the 1974 income that 1975 needs, which gives
        # One extra county-year at h=1 per analysis CCG with an income denominator.
        # That is true under both delta definitions: at h=1 `cumulative` needs no
        # lagged dollars at all, dG_ct = D_ct, and the extra rows are still the
        # income lag's. More data correctly constructed rather than a discrepancy,
        # so the comparison runs where 13 defines a shock and the extra rows are
        # asserted to be exactly the first year rather than quietly matched away.
        have_13 = m["shock_13"].notna()
        dev = float(np.abs(m.loc[have_13, "shock"] - m.loc[have_13, "shock_13"]).max())
        covered = bool(m.loc[have_13, "shock"].notna().all())
        extra = int((m["shock"].notna() & ~have_13).sum())
        extra_pre = int((m["shock"].notna() & ~have_13
                         & (m["year"] == args.first_year)).sum())
        # A nonzero deviation almost always means county_panel.parquet was built
        # under the other --delta, which is a real inconsistency on disk and not a
        # tolerance question: both files are read downstream. Say which it is rather
        # than leave a bare maximum to be puzzled over.
        hint = "" if dev == 0.0 else f" - rerun 13 with --delta {args.delta}"
        check(failures, dev == 0.0 and covered and len(m) == len(panel)
              and extra == extra_pre, 3,
              "the h=1 shock reproduces 13 where 13 defines one",
              f"max dev {dev:.1e}, {int(have_13.sum()):,} compared, "
              f"{extra:,} extra in {args.first_year}{hint}")
    else:
        check(failures, True, 3, "the h=1 shock reproduces 13 (not built)", "skipped")

    est = levels[levels["year"].between(args.first_year, args.last_year)]
    m = est.merge(panel[["ccg", "year", "dollars_real", "income_real"]],
                  on=["ccg", "year"], how="inner", suffixes=("", "_13"))
    d_dev = float(np.abs(m["dollars_real"] - m["dollars_real_13"]).max())
    y_ok = bool((m["income_real"].isna() == m["income_real_13"].isna()).all())
    y_dev = float(np.abs(m["income_real"] - m["income_real_13"]).max())
    check(failures, d_dev == 0.0 and y_ok and (np.isnan(y_dev) or y_dev < 1e-9)
          and len(m) == len(panel), 4,
          "the 1969-extended levels equal the panel on 1975-2006",
          f"contracts {d_dev:.1f}, income {y_dev:.3e}")

    loo_dev = 0.0
    for h, f in zip(horizons, frames):
        if not loo:
            continue
        ok_rows = f["nat_dD"].notna() & f["income_real_lag"].notna()
        s = f.loc[ok_rows]
        loo_dev = max(loo_dev, float(np.abs(
            (s["nat_dD_used"] + s["own_dD"] - s["nat_dD"])
            / np.maximum(np.abs(s["nat_dD"]), 1.0)).max()))
        loo_dev = max(loo_dev, float(np.abs(
            (s["nat_Y_lag_used"] + s["income_real_lag"] - s["nat_Y_lag"])
            / np.abs(s["nat_Y_lag"])).max()))
    check(failures, loo_dev < 1e-10, 5,
          "leave-one-out identity: own + rest = national",
          "not requested" if not loo else f"max rel dev {loo_dev:.3e}")

    nat_ours = frames[0].groupby("year")["dollars_real"].sum()
    nat_13 = panel.groupby("year")["dollars_real"].sum()
    check(failures, float(np.abs(nat_ours - nat_13).max()) == 0.0, 6,
          "the national series equals the panel's year totals", "exact")

    zero_counts, fin_ok = {}, True
    for name in names:
        col = county_all[f"z_{name}"]
        fin_ok &= bool(np.isfinite(col.dropna()).all())
        zero_counts[name] = int((col.fillna(1) == 0).sum())
    check(failures, fin_ok, 7, "every instrument is finite wherever defined",
          "zeros kept: " + ", ".join(f"{k} {v:,}" for k, v in zero_counts.items()))

    pre_clean = not (set(range(*PRE_WINDOW)) | {PRE_WINDOW[1]}) & set(years)
    slow_ok = True
    slow_detail = "fixed window disjoint"
    if roll_names:
        probe = np.sort(universe)[:150]
        for h, f in zip(horizons, frames):
            for n in roll_names:
                slow = slow_rolling_exposure(levels, h, args.roll_k, args.roll_min,
                                             probe, end_lag=ROLL_END_LAG[n])
                cmp = f.merge(slow, on=["ccg", "year"], how="inner")
                a, b = cmp[f"exposure_{n}"], cmp["slow"]
                slow_ok &= bool((a.isna() == b.isna()).all())
                good = a.notna() & b.notna()
                slow_ok &= bool(np.allclose(a[good], b[good], rtol=0, atol=1e-12))
        slow_detail += (f", {'/'.join(roll_names)} verified on {len(probe)} CCGs")
    check(failures, pre_clean and slow_ok, 8,
          "exposure windows contain no estimation-period year", slow_detail)

    # 8b. The two rolling exposures COINCIDE at h=4 and differ everywhere else.
    #
    # h+1 = 5 at h=4, so rolling_t5 IS rolling there - and if the equality failed,
    # or if it held at a horizon where it should not, the end_lag plumbing is wrong
    # in a way that produces a plausible-looking instrument. Asserted in both
    # directions for that reason: an end_lag that was silently ignored would make
    # the two series identical at EVERY horizon and pass a one-sided check.
    if set(ROLLING) <= set(roll_names):
        ok_8b, detail, n_same, n_differ = True, [], 0, 0
        for h, f in zip(horizons, frames):
            a, b = f["exposure_rolling"], f["exposure_rolling_t5"]
            both = a.notna() & b.notna()
            dev = float((a[both] - b[both]).abs().max()) if both.any() else np.nan
            # Derived from the two end_lags rather than from the literal 4, so a
            # change to ROLL_END_LAG moves the horizon this fires at.
            if roll_lag(h, None) == roll_lag(h, ROLL_END_LAG["rolling_t5"]):
                ok_8b &= bool((a.isna() == b.isna()).all()) and dev == 0.0
                n_same += 1
                detail.append(f"h={h} identical")
            else:
                ok_8b &= not (dev == 0.0)
                n_differ += 1
                detail.append(f"h={h} max dev {dev:.4f}")
        # Both branches have to have been exercised somewhere, or the check is
        # one-sided: an end_lag silently ignored makes the two series identical at
        # EVERY horizon, which a differ-only run would catch and a same-only run
        # would not. The default sweep is h=1,2,3, so the identical branch needs
        # --horizon 4 and its absence is stated rather than passed over.
        ok_8b &= n_differ > 0
        if n_same == 0:
            detail.append("identical branch needs --horizon 4, not built here")
        check(failures, ok_8b, "8b",
              "rolling_t5 equals rolling at h=4 and differs elsewhere",
              ", ".join(detail))

    rungs = load_rungs(universe)
    one_each = (len(rungs) == n_ccg
                and rungs["cz"].notna().all()
                and rungs["division"].notna().all()
                and not rungs.duplicated("ccg").any())
    check(failures, one_each, 9,
          "each CCG has exactly one CZ, state and division",
          f"{rungs['cz'].nunique()} CZs, {rungs['state'].nunique()} states, "
          f"{rungs['division'].nunique()} divisions")

    # --- write ------------------------------------------------------------------------
    print("\n[4] writing")
    keep = ["ccg", "year", "h", "shock", "emp_growth", "income_real_lag", "emp_lag",
            "buildup", "nat_dD", "nat_Y_lag", "nat_dD_used", "nat_Y_lag_used"]
    keep += [f"exposure_{n}" for n in names] + [f"z_{n}" for n in names]
    dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet"
    county_all[keep].to_parquet(out_path, index=False)
    print(f"    {len(county_all):,} rows -> {out_path}")

    rep = county_all.groupby(["h", "year"]).agg(
        buildup=("buildup", "mean"),
        shock_sd=("shock", "std"),
        n_shock=("shock", "count"),
        **{f"{n}_sd": (f"z_{n}", "std") for n in names},
        **{f"{n}_zero": (f"z_{n}", lambda s: int((s == 0).sum())) for n in names},
    ).reset_index()
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(dc.LOG_DIR / "instrument_report.csv", index=False)
    print(f"    per-year report -> {dc.LOG_DIR / 'instrument_report.csv'}")

    # --- the rung sweep, plan check 5 -------------------------------------------------
    if not args.skip_sweep:
        print("\n[5] first-stage strength at every rung, plan section 1.5 check 5")
        cells = load_cell_counts(years, {"county x NAICS2": "naics2",
                                         "county x NAICS4": "naics4_alloc"})
        sweep = rung_sweep(county_all, rungs, names, horizons, cells, periods=periods)
        sweep_path = dc.LOG_DIR / "instrument_first_stage.csv"
        sweep.to_csv(sweep_path, index=False)
        print(f"    {len(sweep):,} fits -> {sweep_path}")

        worst = float(sweep["omega_dev"].max())
        check(failures, worst < 1e-12, 10, "unit weights sum to one at every rung",
              f"max |sum omega - 1| {worst:.3e}")

        # Printed period by period rather than pooled: the post-1998 sub-sample is
        # where the first stage collapses, and a full-sample-only table hides that.
        h_head = 2 if 2 in horizons else max(horizons)
        for pname in periods:
            # The headline exposure is PER PERIOD, not the literal "fixed". A
            # filter on "fixed" printed an empty table for 2002-2018, whose only
            # admissible fixed window is fixed_9398 - and an empty table reads as
            # "estimated, found nothing" rather than "asked the wrong question".
            head_exp = next((e for e in ("fixed", "fixed_9398")
                             if e in ADMISSIBLE.get(pname, set()) and e in names), None)
            if head_exp is None:
                continue
            head = sweep[(sweep["h"] == h_head) & (sweep["exposure"] == head_exp)
                         & (sweep["fe"] == "unit+year") & (sweep["period"] == pname)
                         & (sweep["weights"].isin(("unweighted", "employment", "cells")))]
            print(f"\n    h={h_head}, {head_exp} exposure "
                  f"({FIXED_WINDOWS[head_exp][0]}-{FIXED_WINDOWS[head_exp][1]}), "
                  f"unit and year FE, employment weights, {pname}")
            print(f"    {'rung':<18s} {'units':>7s} {'obs':>10s} {'first-stage F':>14s} "
                  f"{'2SLS beta':>11s} {'(se)':>10s}")
            for _, r in head.iterrows():
                beta = "         --" if np.isnan(r["beta"]) else f"{r['beta']:+11.4f}"
                se = "        --" if np.isnan(r["beta_se"]) else f"{r['beta_se']:>10.4f}"
                print(f"    {r['rung']:<18s} {r['n_units']:>7,} {r['n_obs']:>10,} "
                      f"{r['fs_F']:>14.1f} {beta} {se}")

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
