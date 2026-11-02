#!/usr/bin/env python3
"""The first-stage F on eight era x exposure blocks x three rungs, OLS and WLS.

WHAT THIS ADDS TO 19, AND WHY IT IS A SEPARATE FILE

19 reports one exposure per era on the symmetric design: fixed_7074 -> pre,
fixed_9397 -> post, rolling -> both, unweighted regressions, four rungs, h=1..3.
This file widens two dimensions and narrows one:

    era        pre, post, AND post-on-DCADS-alone, all from 19's periods_for(h)
    exposure   the ORIGINAL fixed 1969-74, not 19's fixed_7074, and on BOTH the
               pre era and 2000-2018
    weighting  every fit run TWICE, OLS and WLS on the unit's lagged employment
    rungs      county, commuting zone, state

FIXED 1969-74 ON 2000-2018, ADDED 2026-08-13

The 1969-74 loading was built for the pre era and until now ran only there. It now
also runs on 2000-2018, unchanged: same six years, same construction, aimed at a
sample that begins twenty-six years after the window closes. The question is
whether a cross-section that old still has first-stage power, and it is asked on
the era where fixed_9397 - a loading three years from the sample - is weak. The
DCADS-only era is deliberately NOT extended to this exposure: the splice
comparison is fixed_9397's and rolling_t5's, and one block is what was asked for.

Contamination is not an issue and is not merely assumed: bi.contaminated_years
keeps the sample years with y - h <= 1974, which on 2000-2018 is empty. Check 7
asserts it on this block as on every other.

AND ON 2000-2016, WHICH IS WHERE THE SECOND VINTAGE BREAK COMES IN

The same test runs a third time on the post era truncated at 2016. 2016 is not
typed: it is 11_build_employment_cells.py's VINTAGE_BREAKS[-1] - 1, loaded from
that file the way dc.SEAM_YEAR - 1 gives post_dcads its bound. EFSY breaks twice,
at 1998 and at 2017, and 11 records what the second one does - the row count
roughly halves and employment falls 11%.

The design already refuses to estimate across the FIRST break: 19's periods_for
drops the NAICS straddle, which is why the post era starts at 1998+h+1 rather than
at 1998. It does not refuse the second. A growth rate for year y reads employment
at y and y-h, so it crosses 2017 whenever 2017 <= y <= 2016+h, which on this panel
is {2017, 2018} at h=2, 3 and 4 alike. The post era therefore carries two straddle
years at every horizon, and the truncation removes exactly those two: nothing lies
wholly after the break, because y-h >= 2017 needs y >= 2019 and the panel ends at
2018. Check 1b asserts both halves of that sentence.

This is reported as a third block and NOT as a re-cut of the post era. Whether
1999-2018 should itself stop at 2016 is a separate decision; every published post
number in sections 1.8.5 through 1.8.11 rests on it.

h=2 AND rolling_t5, BOTH CHANGED IN PLACE ON 2026-08-13

This table was h=1 on `rolling`. It is now h=2 on `rolling_t5`, the trailing-5
window ending at t-5 rather than at t-h-1, so the loading sits the same distance
from the sample at every horizon - 14's docstring carries the argument. Both
weightings stay: this table is the single record of what employment weighting was
worth to the first stage, and only the horizon and the exposure moved.

`rolling` is still BUILT, because it is the anchor exposure and check 4 reproduces
a published number with it. It is not a design row.

Every F, beta and se below therefore differs from the 2026-08-11 table, which is
recorded in README 1.8.7 and superseded rather than corrected.

The eras are 19's own periods_for(h), so the NAICS straddle is dropped and every
block here is contamination-free - asserted as check 7, not assumed. At h=1 fixed
(1969-74) reaches back only to 1975 and the pre era starts at 1976; fixed_9397
reaches back only to 1998 and both post eras start at 1999.

WHY 1999-2006 SITS BESIDE 1999-2018 RATHER THAN INSIDE IT

1999-2006 is the post era on DCADS ALONE - it stops at dc.SEAM_YEAR - 1, one year
before the Muratori splice begins. It is not a shorter copy of 1999-2018: the pair
is what isolates the splice's contribution to the post-era first stage. The bound
is derived from dc.SEAM_YEAR and never typed, so if the seam moves the era follows.

WHY THIS IS NOT A FLAG ON 19

19's checks 6 and 7 assert that every fixed window is exactly five years and that
the first estimated year is window_end + h + 1. fixed 1969-74 is SIX years, so it
fails check 7 outright and satisfies check 6 only by the accident of h=1. Those
checks are the content of 19's redesign and should not be loosened to let this
table through. And, as in 19: this file writes nothing that any other script reads,
which is what keeps README section 4 open item 9 - a variant run of 14 replacing
the production panel at a fixed path - from applying to it.

THE DEMEANER IS PINNED HERE

19's own section of the README records that bi.fit_rung inherits pyfixest's default
demeaner, whose alternating projections stop on a tolerance rather than at a fixed
point, so three consecutive canonical runs of 19 reproduced the same frozen anchor
to 0.0, 1.4e-14 and 5.4e-08. That is far below any reported decimal, but it means a
table cannot be checked for exact reproduction. Here it is pinned the way
22_winsor_construction_error.py pins it, and check 10 asserts exact equality on a
refit rather than asserting a tolerance. 14, 15, 16, 17 and 19 are untouched, so no
published number moves.

THE THREE ANCHORS, SINCE FIVE OF THE BLOCKS REPRODUCE NOTHING

    check 4   rolling x the frozen 1975-1998 convention, against
              instrument_first_stage.csv - 19's anchor, carried over verbatim
    check 5   the nine cells section 1.8.6 already publishes - rolling x pre,
              rolling x post, fixed_9397 x post - against first_stage_table.csv
    check 6   all 48 F values against the table this file was asked to produce

Check 5 costs no extra fits: those nine cells are design rows here. Between 4, 5
and 6, only five blocks are genuinely unanchored - fixed 1969-74 on each of
1976-1997, 1999-2018 and the 2016 truncation, and the two 1999-2006 blocks.

Output:
    dc.LOG_DIR/first_stage_grid.csv
    dc.LOG_DIR/20_first_stage_grid.log
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 14 owns the build and the estimator; 19 owns the eras, the 1993-97 window and
# the logging. Both are imported rather than restated, and neither writes anything
# on import - their writers are reachable only from main(), which exec_module does
# not run.
bi = _load("build_instrument", "14_build_instrument.py")
sym = _load("first_stage_table", "19_first_stage_table.py")

# 11 is loaded for ONE constant, VINTAGE_BREAKS, and loaded rather than restated
# for the reason 14 and 19 are: a second copy of 2017 in this file could drift from
# the employment builder's own and nothing would notice. Its only module-level
# statement is the __main__ guard, so exec_module reads no data and writes nothing.
emp = _load("employment_cells", "11_build_employment_cells.py")

# The two fixed windows this table uses. `fixed` is 14's own key and is the
# ORIGINAL 1969-74 six-year window, not 19's fixed_7074; fixed_9397 is 19's.
WINDOWS = {"fixed": bi.FIXED_WINDOWS["fixed"],
           "fixed_9397": sym.NEW_WINDOWS["fixed_9397"]}

# Which exposure describes which era, in the order the table reports them. `fixed`
# runs on BOTH ends of the panel: 1969-74 is the loading the pre era was designed
# around, and pointing it at 2000-2018 asks whether a cross-section fixed a quarter
# of a century before the sample still moves anything in it. Nothing about the
# window is re-tuned for the second use - it is the same six years.
DESIGN = (("fixed", "pre"),
          ("fixed", "post"),
          ("fixed", "post_pre2017"),
          ("fixed_9397", "post_dcads"),
          ("fixed_9397", "post"),
          ("rolling_t5", "pre"),
          ("rolling_t5", "post_dcads"),
          ("rolling_t5", "post"))

# The horizon this table reports. h=1 until 2026-08-13.
HORIZON = 2

# Everything that has to be BUILT: the design's exposures plus the anchors'.
# `rolling` is in here and not in DESIGN, which is the whole distinction.
EXPOSURES = tuple(dict.fromkeys([e for e, _ in DESIGN]
                                + list(sym.ANCHOR_EXPOSURES)))
ROLL_NAMES = tuple(n for n in bi.ROLLING if n in set(EXPOSURES))

ERA_LABEL = {"pre": "pre", "post_dcads": "post, DCADS only", "post": "post",
             "post_pre2017": "post, pre-vintage-break"}

# EFSY's second vintage break, taken from 11 rather than typed. 11's own comment:
# at 2017 "the row count roughly halves while employment falls 11% - so the last
# two years are not comparable with what precedes them". The first break, 1998, is
# already the era boundary; this is the same objection at the other end.
VINTAGE_BREAK = emp.VINTAGE_BREAKS[-1]

RUNGS = (("county", "ccg"), ("commuting zone", "cz"), ("state", "state"))

# The aggregation scheme. omega = 1 at the county, so no weight is formed there at
# all and the label says so rather than naming a column that was never used.
OMEGA_COL = {"employment": "emp_lag", "income": "income_real_lag"}
DEFAULT_OMEGA = "employment"

# The regression weighting, which is the new dimension. "unweighted" is OLS on the
# collapsed frame; "employment" is WLS on the unit's own lagged employment - the
# county's for the county rung, the sum of its counties' for a rung above it.
REG_WEIGHTS = (("unweighted", None), ("employment", "w"))

N_DESIGN = len(DESIGN) * len(RUNGS) * len(REG_WEIGHTS)     # 48 per horizon
N_ANCHOR = len(RUNGS) * len(sym.ANCHOR_EXPOSURES)          # 6 per horizon, OLS only

TOL_BUILD = 1e-12    # rebuilt series against the production parquet

# Relative, for the reason 19 gives: the anchor F spans four orders of magnitude,
# so one absolute tolerance is simultaneously too strict at the division and too
# loose at the degenerate county cell.
RTOL_FIT = 1e-6
ATOL_FIT = 1e-9

# Check 6's tolerance. The requested table carries two decimals, so 0.01 on F is
# the precision it was stated at and anything looser would let a real regression
# through as rounding.
TOL_TABLE_F = 0.01

# ------------------------------------------------------------------------------
# THE TABLE THIS FILE PRODUCES - check 6's target.
# (exposure, era, rung) -> (units, n_frame, F_ols, F_wls)
#
# PROVENANCE CHANGED ON 2026-08-13, and saying so matters. Until then these were
# the h=1 numbers supplied WITH the request, so the check was against an outside
# source. h=1 is no longer reported and rolling became rolling_t5, so no outside
# table exists for this grid: these are FROZEN FROM THE CANONICAL RUN of
# 2026-08-13 instead. That is a weaker claim - it cannot catch an error that was
# in that run - and it is still worth asserting, because it is what stops the
# table moving unnoticed under a later edit. The checks that do compare against
# something independent are 4 (14's frozen sweep) and 5 (19's Table V).
#
# The old h=1 block is in README 1.8.7's superseded table and in git history.
#
# The three fixed x post cells were added on 2026-08-13 under the same convention:
# frozen from the canonical run that first produced them, so they assert stability
# and not correctness. Nothing independent covers that block - see check 6's entry
# in the anchor list above. The three fixed x post_pre2017 cells followed the same
# day and on the same footing. Note what they freeze: F barely moves off the
# untruncated block, 16.41 against 16.67, so a later change that moved F would be
# caught, and a later change that moved only beta - which is where the truncation
# actually bites - would not. beta is in the CSV and is read in the README.
# ------------------------------------------------------------------------------
REQUESTED = {
    ("fixed", "pre", "county"):                              (3094, 64974, 65.61, 63.50),
    ("fixed", "pre", "commuting zone"):                      (738, 15498, 31.25, 59.33),
    ("fixed", "pre", "state"):                               (51, 1071, 40.17, 57.86),
    ("fixed", "post", "county"):                             (3094, 55692, 16.68, 2.84),
    ("fixed", "post", "commuting zone"):                     (738, 13284, 6.69, 7.76),
    ("fixed", "post", "state"):                              (51, 918, 8.19, 3.94),
    ("fixed", "post_pre2017", "county"):                     (3094, 49504, 16.42, 2.64),
    ("fixed", "post_pre2017", "commuting zone"):             (738, 11808, 5.58, 9.20),
    ("fixed", "post_pre2017", "state"):                      (51, 816, 9.02, 3.56),
    ("fixed_9397", "post_dcads", "county"):                  (3094, 21658, 8.24, 2.05),
    ("fixed_9397", "post_dcads", "commuting zone"):          (738, 5166, 1.01, 3.08),
    ("fixed_9397", "post_dcads", "state"):                   (51, 357, 7.45, 9.60),
    ("fixed_9397", "post", "county"):                        (3094, 55692, 6.18, 15.63),
    ("fixed_9397", "post", "commuting zone"):                (738, 13284, 5.69, 40.33),
    ("fixed_9397", "post", "state"):                         (51, 918, 40.86, 45.12),
    ("rolling_t5", "pre", "county"):                         (3094, 64974, 1.69, 7.44),
    ("rolling_t5", "pre", "commuting zone"):                 (738, 15498, 9.62, 95.51),
    ("rolling_t5", "pre", "state"):                          (51, 1071, 43.58, 81.25),
    ("rolling_t5", "post_dcads", "county"):                  (3094, 21658, 0.03, 0.57),
    ("rolling_t5", "post_dcads", "commuting zone"):          (738, 5166, 0.26, 0.66),
    ("rolling_t5", "post_dcads", "state"):                   (51, 357, 8.31, 14.69),
    ("rolling_t5", "post", "county"):                        (3094, 55692, 0.22, 4.31),
    ("rolling_t5", "post", "commuting zone"):                (738, 13284, 12.97, 11.77),
    ("rolling_t5", "post", "state"):                         (51, 918, 22.71, 20.27),
}

# The nine cells section 1.8.6 Table V already publishes, which are design rows
# here. Check 5 joins on them; check 10 refits them to prove exact reproduction.
TABLEV_CELLS = (("rolling_t5", "pre"), ("rolling_t5", "post"),
                ("fixed_9397", "post"))

DEMEANER = None      # set in main(), once pyfixest is imported


def check(failures, ok, num, label, detail=""):
    """Print one numbered PASS/FAIL line and record the failure."""
    print(f"    [{num}] {label:<54s} {detail:<44s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


def eras_for(h):
    """The four (era -> bounds) this table reports, all derived from h.

    pre and post come from 19's periods_for verbatim - restating them is how this
    file would end up measuring a different sample than section 1.8.6 while both
    tables said "1976-1997". post_dcads is the post era truncated at the year
    before the splice, taken from dc.SEAM_YEAR so that moving the seam moves the
    era rather than leaving this bound behind. post_pre2017 is the post era
    truncated at the year before EFSY's second vintage break, taken from 11's
    VINTAGE_BREAKS on the same principle - 2016 is never typed here either.
    """
    base, straddle = sym.periods_for(h)
    return ({"pre": base["pre"],
             "post_dcads": (sym.NAICS_BREAK + h, dc.SEAM_YEAR - 1),
             "post": base["post"],
             "post_pre2017": (sym.NAICS_BREAK + h, VINTAGE_BREAK - 1)},
            straddle)


def straddling_years(bounds, h):
    """The years in `bounds` whose h-year growth rate crosses the 2017 break.

    A growth rate for year y reads employment at y and at y-h, so it straddles the
    break iff y >= VINTAGE_BREAK and y - h <= VINTAGE_BREAK - 1. The design already
    refuses the same overlap at the FIRST break - 19's periods_for estimates the
    NAICS straddle nowhere, which is why the post era starts at 1998+h+1 - and this
    is that rule read at the other end of the panel.
    """
    y0, y1 = bounds
    return [y for y in range(y0, y1 + 1)
            if y >= VINTAGE_BREAK and y - h <= VINTAGE_BREAK - 1]


def register_variants(horizons):
    """Add fixed_9397 and the h-indexed era bounds to 14's own dicts.

    bi.contaminated_years looks its arguments up BY NAME in bi.FIXED_WINDOWS and
    bi.PERIODS, so registering here reuses the canonical contamination logic
    instead of restating it - and restating it is exactly the drift that would let
    this file report zero contamination while the real design had some. 19's
    era_key is reused so the two files cannot disagree about what "pre-h1" means.
    Check 2 asserts the registration only ADDED.
    """
    before = (dict(bi.FIXED_WINDOWS),
              dict(bi.PERIODS),
              {k: set(v) for k, v in bi.ADMISSIBLE.items()})
    bi.FIXED_WINDOWS.update(WINDOWS)
    for h in horizons:
        eras, _ = eras_for(h)
        for era, bounds in eras.items():
            key = sym.era_key(era, h)
            bi.PERIODS[key] = bounds
            bi.ADMISSIBLE[key] = {e for e, x in DESIGN if x == era}
    return before


def fit(frame, fe, weights=None):
    """bi.fit_rung with the demeaner pinned.

    A wrapper and not a copy: bi.fit_rung is the estimator every other step_1
    number comes from, and a second implementation of it here would be free to
    drift while continuing to look right. The pin is what makes check 10's exact
    equality assertable - see 22_winsor_construction_error.py, which pins the same
    two knobs for the same reason.
    """
    return bi.fit_rung(frame, fe, weights=weights, demeaner=DEMEANER)


def build(horizons):
    """The instrument on every exposure, in memory, at 14's production settings.

    Floor 0, cumulative dG, leave-one-out on, the transitional quarter folded, and
    the levels reaching back to EXPOSURE_FIRST because 1969-74 needs years the
    estimation sample does not contain. 19's build(), with this file's windows.
    """
    levels, universe, _ = bi.build_levels(bi.DEFAULT_FLOOR, True,
                                          bi.EXPOSURE_FIRST, sym.LAST_YEAR)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)

    fixed = {}
    for name, window in WINDOWS.items():
        s = bi.fixed_exposure(levels, window, name)
        fixed[name] = s
        print(f"    {name:<11} {window[0]}-{window[1]}   mean {s.mean():.5f}   "
              f"sd {s.std():.5f}   {int((s == 0).sum()):,} counties at zero, "
              f"{int(s.isna().sum())} with no income denominator")
    for n in ROLL_NAMES:
        print(f"    {n:<11} {window_label(n)}, at least {bi.ROLL_MIN} usable")

    frames = []
    for h in horizons:
        ex = dict(fixed)
        ex.update(bi.rolling_exposures(levels, h, bi.ROLL_K, bi.ROLL_MIN,
                                       names=ROLL_NAMES))
        f = bi.build_horizon(levels, ex, h, bi.FIRST_YEAR, sym.LAST_YEAR, True,
                             delta=bi.DEFAULT_DELTA)
        frames.append(f)
        v = f["shock"].dropna()
        print(f"    h={h}   {len(v):,} CCG-years carry a shock   sd {v.std():.5f}")
    county = pd.concat(frames, ignore_index=True)

    # bi.collapse's column list carries pre_D and pre_Y unconditionally. They feed
    # only the construction-error exhibit, which this table does not compute, but
    # they have to exist or collapse raises.
    pre = {name: bi.fixed_window_sums(levels, w).set_index("ccg")
           for name, w in WINDOWS.items()}
    return county, levels, universe, pre


def windows_for(name, h):
    """Every (role, era, period_key, y0, y1) this exposure reports at h."""
    out = []
    eras, _ = eras_for(h)
    for exposure, era in DESIGN:
        if exposure != name:
            continue
        y0, y1 = eras[era]
        out.append(("variant", era, sym.era_key(era, h), y0, y1))
    if name in sym.ANCHOR_EXPOSURES:
        y0, y1 = bi.PERIODS[sym.ANCHOR_PERIOD]
        out.append(("anchor", sym.ANCHOR_PERIOD, sym.ANCHOR_PERIOD, y0, y1))
    return out


def window_label(name):
    """19's, reused rather than restated - the two files must not disagree about
    what `trailing 5 ending t-5` means."""
    if name in WINDOWS:
        w0, w1 = WINDOWS[name]
        return f"{w0}-{w1}"
    return sym.window_label(name)


def unit_frame(base, rung, unit_col, omega_col):
    """(frame, note) - one row per unit x year, with the WLS weight attached.

    The weight is the unit's own lagged employment: the county's at the county
    rung, the sum of its counties' above it. It is computed here rather than taken
    from bi.collapse, which returns the omega vectors and the aggregated moments
    but no unit employment total.
    """
    if rung == "county":
        d = base.rename(columns={"emp_growth": "y", "shock": "x",
                                 "emp_lag": "w"})
        d = d.assign(unit=d["ccg"])[["unit", "year", "y", "x", "z", "w"]]
    else:
        d = bi.collapse(base, unit_col, omega_col)[["unit", "year", "y", "x", "z"]]
        w = (base.groupby([unit_col, "year"])["emp_lag"].sum().rename("w")
             .reset_index().rename(columns={unit_col: "unit"}))
        d = d.merge(w, on=["unit", "year"], how="left")
    d = d[np.isfinite(d[["x", "y", "z"]].to_numpy()).all(axis=1)]
    return d


def fit_grid(county, rungs, pre, horizons, exposures, rung_names, omega):
    """One row per role x exposure x era x h x rung x regression weighting.

    The finiteness screen sits OUTSIDE the era loop, exactly as 14's rung_sweep and
    19's fit_grid do it, so the county support is frozen the same way in every
    sub-sample and an era differs from another only in which years it contains.

    Returns (rows, repeat_devs): repeat_devs holds check 10's refit deviations,
    collected here because it is the only place the estimation frame exists.
    """
    wanted = [(r, c) for r, c in RUNGS if r in set(rung_names)]
    omega_col = OMEGA_COL[omega]
    rows, repeat_devs, bad_w = [], [], []
    for h in horizons:
        for name in exposures:
            full = county[county["h"] == h].copy()
            full["z"] = full[f"z_{name}"]
            if name in pre:
                full["pre_D"] = full["ccg"].map(pre[name]["pre_D"])
                full["pre_Y"] = full["ccg"].map(pre[name]["pre_Y"])
            else:
                full["pre_D"] = np.nan
                full["pre_Y"] = np.nan
            full = full.merge(rungs, on="ccg", how="left")
            keep = np.isfinite(full[["shock", "emp_growth", "z", "income_real_lag",
                                     "emp_lag"]].to_numpy()).all(axis=1)
            full = full[keep & (full["emp_lag"] > 0) & (full["income_real_lag"] > 0)]

            for role, era, pkey, y0, y1 in windows_for(name, h):
                base = full[full["year"].between(y0, y1)]
                n_con = len(bi.contaminated_years(pkey, name, h))
                for rung, unit_col in wanted:
                    d = unit_frame(base, rung, unit_col, omega_col)
                    # Asserted rather than screened. A non-positive or missing
                    # weight would make WLS silently drop rows, and check 8 reads
                    # equal row counts as evidence that it did not.
                    if not (np.isfinite(d["w"]).all() and (d["w"] > 0).all()):
                        bad_w.append(f"{name} x {era} x {rung} h={h}")
                    schemes = REG_WEIGHTS if role == "variant" else REG_WEIGHTS[:1]
                    for rw_name, rw_col in schemes:
                        r = fit(d, "unit + year", weights=rw_col)
                        if (role == "variant" and rw_col is None
                                and (name, era) in TABLEV_CELLS):
                            again = fit(d, "unit + year", weights=rw_col)
                            worst = max(abs(again[k] - r[k])
                                        for k in ("fs_F", "beta", "beta_se"))
                            repeat_devs.append((f"{name} x {era} x {rung} h={h}",
                                                worst))
                        rows.append(dict(
                            role=role, era=era, era_label=ERA_LABEL.get(era, era),
                            era_span=f"{y0}-{y1}", exposure=name,
                            exposure_window=window_label(name), rung=rung,
                            n_units=int(d["unit"].nunique()), omega=omega,
                            reg_weights=rw_name, h=h, year_first=y0, year_last=y1,
                            fe="unit+year", contaminated_years=n_con,
                            # TWO counts, following 22_winsor_construction_error's
                            # own vocabulary. n_frame is the rows in the estimation
                            # frame; n_obs is what the fit actually used, which is
                            # smaller wherever pyfixest dropped a singleton unit.
                            # They differ by one in the 1999-2006 county cells and
                            # nowhere else in this grid, and a table that reported
                            # only one of them would make that look like a bug.
                            n_frame=len(d), **r))
    return pd.DataFrame(rows), repeat_devs, bad_w


# --------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------

def check_eras(failures, horizons):
    """Check 1. The four eras are 19's, plus the two derived truncations.

    Both truncated bounds are asserted to BE the constant they are derived from
    minus one - dc.SEAM_YEAR for the splice, 11's VINTAGE_BREAKS for the employment
    vintage. That is what stops either era being quietly re-typed as 2006 or 2016
    and then left behind when the constant it was supposed to track moves.
    """
    ok, detail = True, []
    for h in horizons:
        eras, _ = eras_for(h)
        base, _ = sym.periods_for(h)
        good = (eras["pre"] == base["pre"] and eras["post"] == base["post"]
                and eras["post_dcads"] == (sym.NAICS_BREAK + h, dc.SEAM_YEAR - 1)
                and eras["post_pre2017"] == (sym.NAICS_BREAK + h,
                                             VINTAGE_BREAK - 1))
        ok &= good
        detail.append(f"h={h} {eras['pre'][0]}-{eras['pre'][1]}, "
                      f"{eras['post_dcads'][0]}-{eras['post_dcads'][1]}, "
                      f"{eras['post'][0]}-{eras['post'][1]}, "
                      f"{eras['post_pre2017'][0]}-{eras['post_pre2017'][1]}")
    ok &= VINTAGE_BREAK == max(emp.VINTAGE_BREAKS)
    return check(failures, ok, 1, "eras are 19's periods_for plus the two bounds",
                 "; ".join(detail))


def check_straddle(failures, horizons):
    """Check 1b. post carries the 2017 straddle; post_pre2017 carries none.

    This is the whole reason the truncated era exists, asserted rather than
    described. A growth rate that reads employment on both sides of EFSY's 2017
    break measures the vintage change - 11 records the row count roughly halving
    and employment falling 11% across it - and the post era contains two such years
    at every horizon this file runs. The truncation removes exactly those and
    nothing else: no observation lies wholly after the break, because y - h >= 2017
    would need y >= 2019 and the panel stops at 2018.
    """
    ok, detail = True, []
    for h in horizons:
        eras, _ = eras_for(h)
        bad = straddling_years(eras["post"], h)
        clean = straddling_years(eras["post_pre2017"], h)
        ok &= bool(bad) and not clean
        detail.append(f"h={h} post {bad}, truncated {clean or 'none'}")
    return check(failures, ok, "1b",
                 "the 2017 straddle is in post and not in the truncation",
                 "; ".join(detail))


def check_build(failures, county, horizons):
    """Check 3. The rebuild is right at every horizon, production or not.

    19's rebuild_deviation, called rather than copied. The version this file used
    to carry filtered production on h.isin(horizons) and would report a clean pass
    on zero rows at any horizon 14 has not built - README 1.8.8 records that
    vacuity against this file by name, and calling 19's version is the repair.
    """
    ok, detail = sym.rebuild_deviation(county, horizons)
    return check(failures, ok, 3, "the rebuild is right at every horizon", detail)


def _compare(m, cols, want, num, label, failures):
    """(ok, detail) for a frozen-file join. An empty merge is a FAILURE.

    The whole value of an anchor is that it compares something, and a key that
    stopped matching would otherwise report "0 cells, max dev nan" and read green.
    """
    worst_rel, worst_abs, where = 0.0, 0.0, ""
    if not len(m):
        worst_rel = np.inf
    for col in cols:
        if not len(m):
            break
        f, n = m[f"{col}_frozen"], m[f"{col}_new"]
        dev = (f - n).abs()
        rel = dev / (ATOL_FIT + f.abs())
        if float(rel.max()) > worst_rel:
            i = rel.idxmax()
            where = f"{col} at {m.loc[i, 'rung']} h={m.loc[i, 'h']}"
        worst_rel = max(worst_rel, float(rel.max()))
        worst_abs = max(worst_abs, float(dev.max()))
    ok = len(m) == want and worst_rel < RTOL_FIT
    return check(failures, ok, num, label,
                 f"{len(m)}/{want} cells, rel {worst_rel:.1e} abs {worst_abs:.1e}"
                 + (f" ({where})" if where else ""))


def check_anchor(failures, grid, horizons, rung_names, omega):
    """Check 4. rolling x the frozen 1975-1998 convention, against 14's sweep."""
    a = grid[(grid["role"] == "anchor") & (grid["reg_weights"] == "unweighted")]
    o = pd.read_csv(dc.LOG_DIR / "instrument_first_stage.csv")
    o = o[(o["fe"] == "unit+year") & (o["period"] == sym.ANCHOR_PERIOD)
          & (o["exposure"].isin(sym.ANCHOR_EXPOSURES))
          & (o["weights"].isin(("unweighted", omega)))]
    covered = sorted(set(horizons) & set(o["h"]))
    m = o.merge(a, on=["rung", "h", "exposure"], suffixes=("_frozen", "_new"))
    return _compare(m, ("fs_F", "beta", "beta_se", "n_obs"),
                    len(sym.ANCHOR_EXPOSURES) * len(covered) * len(rung_names), 4,
                    "anchors reproduce instrument_first_stage.csv", failures)


def check_tablev(failures, grid, horizons, rung_names):
    """Check 5. The nine cells section 1.8.6 publishes, against 19's own output.

    These are design rows here, not extra fits: rolling x pre, rolling x post and
    fixed_9397 x post are the same (exposure, era, rung, h) cells 19 reports. The
    join is on 19's era labels, which is why this file reuses them.
    """
    p = dc.LOG_DIR / "first_stage_table.csv"
    if not p.exists():
        return check(failures, False, 5, "Table V cells reproduce 19's output",
                     f"{p.name} not found - run 19 first")
    is_tablev = pd.Series(list(zip(grid["exposure"], grid["era"])),
                          index=grid.index).isin(TABLEV_CELLS)
    d = grid[(grid["role"] == "variant") & (grid["reg_weights"] == "unweighted")
             & is_tablev]
    o = pd.read_csv(p)
    o = o[(o["role"] == "variant") & (o["fe"] == "unit+year")]
    m = o.merge(d, on=["rung", "h", "exposure", "era"], suffixes=("_frozen", "_new"))
    want = len(TABLEV_CELLS) * len(horizons) * len(rung_names)
    return _compare(m, ("fs_F", "beta", "beta_se", "n_obs"), want, 5,
                    "section 1.8.6 Table V cells reproduce 19", failures)


def check_requested(failures, grid, horizons, rung_names):
    """Check 6. Every F, unit count and observation count against the request.

    The count compared is n_frame, the rows in the estimation frame, because that
    is the convention the requested table was stated in. n_obs - what the fit used
    after pyfixest dropped singleton units - is one lower in the two 1999-2006
    county cells, and both are carried in the CSV so neither has to be inferred.
    """
    if horizons != [HORIZON]:
        return check(failures, True, 6, "the requested table, F to 0.01",
                     f"skipped: stated at h={HORIZON}, this run is h={horizons}")
    if not REQUESTED:
        return check(failures, True, 6, "the requested table, F to 0.01",
                     "skipped: REQUESTED is empty, this is the generating pass")
    d = grid[(grid["role"] == "variant") & (grid["h"] == HORIZON)]
    piv = d.pivot_table(index=["exposure", "era", "rung"], columns="reg_weights",
                        values="fs_F")
    n = d.set_index(["exposure", "era", "rung", "reg_weights"])
    bad, checked = [], 0
    for key, (units, obs, f_ols, f_wls) in REQUESTED.items():
        if key[2] not in set(rung_names):
            continue
        checked += 1
        if key not in piv.index:
            bad.append(f"{key} MISSING")
            continue
        got_ols = float(piv.loc[key, "unweighted"])
        got_wls = float(piv.loc[key, "employment"])
        got_u = int(n.loc[key + ("unweighted",), "n_units"])
        got_o = int(n.loc[key + ("unweighted",), "n_frame"])
        got_fit = int(n.loc[key + ("unweighted",), "n_obs"])
        if abs(got_ols - f_ols) > TOL_TABLE_F:
            bad.append(f"{key} F_ols {got_ols:.2f} want {f_ols:.2f}")
        if abs(got_wls - f_wls) > TOL_TABLE_F:
            bad.append(f"{key} F_wls {got_wls:.2f} want {f_wls:.2f}")
        if got_u != units:
            bad.append(f"{key} units {got_u} want {units}")
        if got_o != obs:
            bad.append(f"{key} n_frame {got_o} want {obs} (n_obs {got_fit})")
    ok = not bad
    check(failures, ok, 6, "the requested table, F to 0.01 and counts exact",
          f"{checked} cells" if ok else f"{len(bad)} deviation(s) of {checked} cells")
    for b in bad:
        print(f"          {b}")
    return ok


def check_rest(failures, grid, repeat_devs, bad_w, horizons, rung_names, canonical):
    """Checks 7-10."""
    d = grid[grid["role"] == "variant"]
    check(failures, int(d["contaminated_years"].max()) == 0, 7,
          "zero contaminated years on every design row",
          f"max {int(d['contaminated_years'].max())} over {len(d)} rows")

    # OLS and WLS must see the same frame. A weight that dropped rows would show up
    # as a weighting effect on F and nowhere else.
    k = ["role", "exposure", "era", "rung", "h"]
    gb = grid.groupby(k)
    paired = (gb[["n_frame", "n_obs", "n_units"]].nunique()
              [gb["reg_weights"].nunique() > 1])
    same = bool((paired == 1).all().all()) and not bad_w and len(paired) > 0
    check(failures, same, 8, "OLS and WLS agree on n_obs and n_units",
          f"{len(paired)} paired cells"
          + (f", bad weight in {bad_w[:2]}" if bad_w else ""))

    w_var = len(DESIGN) * len(horizons) * len(rung_names) * len(REG_WEIGHTS)
    w_anc = len(sym.ANCHOR_EXPOSURES) * len(horizons) * len(rung_names)
    n_var = int((grid["role"] == "variant").sum())
    n_anc = int((grid["role"] == "anchor").sum())
    label = ("row count is the announced grid" if canonical
             else "row count is the requested subset")
    check(failures, n_var == w_var and n_anc == w_anc, 9, label,
          f"{n_var} of {w_var} design, {n_anc} of {w_anc} anchor")

    worst = max((v for _, v in repeat_devs), default=np.inf)
    where = max(repeat_devs, key=lambda t: t[1])[0] if repeat_devs else "none"
    check(failures, bool(repeat_devs) and worst == 0.0, 10,
          "a refit is bit-identical, the demeaner being pinned",
          f"{len(repeat_devs)} cells refitted, max dev {worst:.1e} ({where})")


def main():
    global DEMEANER
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=[HORIZON],
                    help="horizons to report; each one sets its own era bounds. "
                         f"The requested table is h={HORIZON}, the default")
    ap.add_argument("--rung", nargs="+", default=[r for r, _ in RUNGS],
                    choices=[r for r, _ in RUNGS])
    ap.add_argument("--omega", choices=sorted(OMEGA_COL), default=DEFAULT_OMEGA,
                    help="the AGGREGATION weight above the county, not the "
                         "regression weight; both regression weightings are "
                         "always reported")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))

    # A PARTIAL RUN NEVER OVERWRITES THE CANONICAL OUTPUT, and that includes the
    # log. README section 4 item 9 records this hazard on 14 and prescribes the
    # fix: a variant path. A debugging run on one rung would otherwise replace the
    # table with a subset of itself, and the file on disk would look complete.
    canonical = (horizons == [HORIZON] and set(args.rung) == {r for r, _ in RUNGS}
                 and args.omega == DEFAULT_OMEGA)
    sfx = "" if canonical else ("__h" + "-".join(map(str, horizons)) + "_"
                                + "-".join(sorted(args.rung)).replace(" ", "")
                                + "_" + args.omega)

    tee = sym.Tee(dc.LOG_DIR / f"20_first_stage_grid{sfx}.log")
    try:
        import pyfixest as pf
        # Pinned, not inherited - see the docstring and 22_winsor_construction_
        # error.py lines 98-103. pyfixest's default stops the alternating
        # projections on a loose tolerance, which is why 19 can only assert a
        # relative agreement and this file can assert an exact one.
        DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

        failures = []
        print("=" * 78)
        print("The first-stage F grid - eight era x exposure blocks, OLS and WLS")
        print("=" * 78)
        print("  design     " + "; ".join(f"{e} -> {ERA_LABEL[x]}"
                                          for e, x in DESIGN))
        print("  exposures  " + ", ".join(
            f"{n} {window_label(n)}" for n in dict.fromkeys(e for e, _ in DESIGN)))
        print(f"  h          {horizons}")
        for h in horizons:
            eras, straddle = eras_for(h)
            print(f"    h={h}   " + "   ".join(
                f"{ERA_LABEL[k]} {v[0]}-{v[1]}" for k, v in eras.items())
                + f"   straddle {straddle[0]}-{straddle[1]}, estimated nowhere")
        print(f"  rungs      {', '.join(args.rung)}")
        print(f"  omega      {args.omega} ({OMEGA_COL[args.omega]}) above the "
              f"county, omega = 1 at the county")
        print(f"  regression {', '.join(n for n, _ in REG_WEIGHTS)} "
              f"(WLS weight = the unit's lagged employment)")
        print(f"  anchors    " + " and ".join(sym.ANCHOR_EXPOSURES)
              + f" x {sym.ANCHOR_PERIOD}, the frozen convention, against "
                f"instrument_first_stage.csv")
        print(f"  FE         unit + year, clustered on the unit")
        print(f"  demeaner   PINNED, fixef_tol=1e-12, fixef_maxiter=100,000")

        print("\n[1] the eras, and the registration")
        check_eras(failures, horizons)
        check_straddle(failures, horizons)
        before = register_variants(horizons)
        ok, detail = sym.only_added(before)
        check(failures, ok, 2, "registering the variants only ADDED keys", detail)

        print("\n[2] the instrument, every exposure, in memory")
        exposures = list(EXPOSURES)
        county, levels, universe, pre = build(horizons)
        rungs = bi.load_rungs(universe)

        print("\n[3] the build, against production and against source")
        check_build(failures, county, horizons)
        sym.check_rolling_pair(failures, county, horizons)

        print("\n[4] the fits")
        grid, repeat_devs, bad_w = fit_grid(county, rungs, pre, horizons,
                                            exposures, args.rung, args.omega)
        print(f"    {len(grid)} fits: "
              f"{int((grid['role'] == 'variant').sum())} design, "
              f"{int((grid['role'] == 'anchor').sum())} anchor")

        print("\n[5] the checks")
        check_anchor(failures, grid, horizons, args.rung, args.omega)
        check_tablev(failures, grid, horizons, args.rung)
        check_requested(failures, grid, horizons, args.rung)
        check_rest(failures, grid, repeat_devs, bad_w, horizons, args.rung,
                   canonical)

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            report(grid, horizons, args.rung)
            return 1

        if not canonical:
            print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; "
                  f"this run writes to *{sfx}.* instead.")
            print("  Rerun with no arguments to rebuild the canonical set.")

        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.LOG_DIR / f"first_stage_grid{sfx}.csv"
        grid.to_csv(out, index=False)
        print(f"\nall checks passed   {len(grid)} fits -> {out}")
        report(grid, horizons, args.rung)
        return 0
    finally:
        tee.close()


def report(grid, horizons, rung_names):
    """The table, in the shape the README carries it.

    Printed even on failure: a run that failed check 6 is exactly the run whose
    numbers need reading, and printing them only on success would mean the one
    diagnostic worth having is withheld when it is needed.
    """
    d = grid[grid["role"] == "variant"]
    short = {"fixed": "fixed 1969-74", "fixed_9397": "fixed 1993-97",
             "rolling": f"rolling K={bi.ROLL_K}",
             # 14 chars at most: the report pads this column to 15 and a
             # 15-char label runs into the rung beside it.
             "rolling_t5": f"rolling K={bi.ROLL_K} t5"}
    for h in horizons:
        print(f"\n  h={h}   first-stage F")
        print(f"    {'era':<11}{'exposure':<15}{'rung':<17}{'units':>7}"
              f"{'obs':>9}{'F unweighted':>15}{'F emp-weighted':>16}")
        for exposure, era in DESIGN:
            s = d[(d["exposure"] == exposure) & (d["era"] == era) & (d["h"] == h)]
            if s.empty:
                continue
            span, first = s["era_span"].iloc[0], True
            for rung, _ in RUNGS:
                if rung not in set(rung_names):
                    continue
                r = s[s["rung"] == rung]
                if r.empty:
                    continue
                o = r[r["reg_weights"] == "unweighted"].iloc[0]
                w = r[r["reg_weights"] == "employment"].iloc[0]
                # The era and the exposure are printed on the block's first rung
                # only, which is what the requested table does.
                lab = span if first else ""
                exp = short[exposure] if first else ""
                first = False
                # obs is n_frame, the convention the requested table used. Where
                # the fit dropped a singleton, n_obs is printed beside it rather
                # than the difference being left to be discovered.
                extra = ("" if int(o["n_obs"]) == int(o["n_frame"])
                         else f"   (n_obs {int(o['n_obs']):,}, "
                              f"{int(o['n_frame']) - int(o['n_obs'])} singleton)")
                print(f"    {lab:<11}{exp:<15}{rung:<17}"
                      f"{int(o['n_units']):>7,}{int(o['n_frame']):>9,}"
                      f"{o['fs_F']:>15.2f}{w['fs_F']:>16.2f}{extra}")
    print(f"\n  2SLS beta (se), the same fits - not in the table above")
    for h in horizons:
        for exposure, era in DESIGN:
            s = d[(d["exposure"] == exposure) & (d["era"] == era) & (d["h"] == h)]
            if s.empty:
                continue
            print(f"    h={h}  {short[exposure]} x {ERA_LABEL[era]} "
                  f"({s['era_span'].iloc[0]})")
            for rung, _ in RUNGS:
                r = s[s["rung"] == rung]
                if r.empty:
                    continue
                o = r[r["reg_weights"] == "unweighted"].iloc[0]
                w = r[r["reg_weights"] == "employment"].iloc[0]
                print(f"      {rung:<17}OLS {o['beta']:+.3f} ({o['beta_se']:.3f})"
                      f"     WLS {w['beta']:+.3f} ({w['beta_se']:.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
