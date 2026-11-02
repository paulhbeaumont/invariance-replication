#!/usr/bin/env python3
"""The clustered first-stage F on the symmetric two-era instrument. Part 1.5, check 5.

WHAT THIS REDESIGNS, AND WHY

Section 1.8.5 reports the first stage on two exposure windows that are not
comparable to each other:

    fixed        1969-1974, six years, disjoint from its sample 1975-1998
    fixed_9398   1993-1998, six years, and 1998 is the FIRST YEAR of its own sample

So a pre/post difference in beta mixes a change in the multiplier with a change in
how the loading was measured, and neither is length-comparable to `rolling`, which
is a FIVE-year trailing window (ROLL_K = 5). This file replaces both with the five
years immediately preceding each era:

    fixed_7074   1970-1974  ->  pre-break
    fixed_9397   1993-1997  ->  post-break
    rolling_t5   trailing 5 ending t-5    ->  both

Every window is now five years, so `fixed` and `rolling` differ only in PLACEMENT -
frozen versus sliding - and not in length. 1969-1974 is discarded.

THE ROLLING EXPOSURE IS rolling_t5, NOT rolling                     2026-08-13

`rolling` ends at t-h-1, so its distance from the sample tracks the horizon and a
change in beta across h mixes a change in the horizon with a change in where the
loading was measured. `rolling_t5` ends at t-5 whatever h is, which at h <= 4
clears the earliest year the shock touches (t-h) by at least one year and clears it
by the SAME amount at every horizon. 14 owns the construction; see its docstring.

`rolling` survives here in exactly one place, role="anchor", because that is the
fit that reproduces a published number out of instrument_first_stage.csv. It is
never a design row and never printed as one.

HORIZONS ARE 2, 3 AND 4                                             2026-08-13

h=1 is dropped and h=4 added, on request. h=4 has no block in
instrument_ccg_year.parquet, which check 3 handles by recomputing from source
rather than by comparing zero rows and passing - see rebuild_deviation.

THE ERAS FOLLOW THE EMPLOYMENT DATA, NOT THE CONTRACTS

Section 1.8.5 puts 1998 in both halves, on the argument that a first stage of
contracts on contracts cannot be touched by the CBP NAICS break. That is true of
the first stage read in isolation - and it is the wrong test, because this
instrument exists to be run against the employment dataset. So the split follows
the OUTCOME's convention, which step_2 already codified in periods_for():

    pre       EFSY_FIRST_YEAR + h  ..  NAICS_BREAK - 1
    post      NAICS_BREAK + h      ..  LAST_YEAR
    straddle  NAICS_BREAK          ..  NAICS_BREAK + h - 1     estimated nowhere

The straddle years go because their t-h sits on the imputed side of the break; the
pre era starts at 1975 + h because t-h has to reach EFSY's first year.

AND THE TWO TRIMS COINCIDE EXACTLY, WHICH IS THE POINT

A window ending w1 contaminates every year y with y - h <= w1. For fixed_9397 that
is {1998 .. 1997+h} - which IS the straddle set. Dropping the NAICS straddle
therefore removes the exposure contamination for free. Symmetrically on the pre
side: fixed_7074 contaminates {1975 .. 1974+h} and EFSY's 1975 start already forces
the era to begin at 1975+h. So in BOTH eras the first estimated year is exactly

    window_end + h + 1

for pre because the employment data's first year forces it, for post because the
break does. The two halves are the same design shifted twenty-three years, with
zero contamination anywhere, and 18-22 years each against the old 1975-1998 (24)
versus 2002-2018 (17).

IT WRITES NOTHING THAT ANY OTHER SCRIPT READS.

That is why this is a separate file rather than a flag on 14. 14 writes
instrument_ccg_year.parquet and instrument_first_stage.csv at fixed paths, so any
variant run of 14 silently replaces the production panel - README section 4 open
item 9. Everything here is built in memory.

THE ANCHOR, SINCE NO REPORTED CELL REPRODUCES A PUBLISHED NUMBER

Both fixed windows are new, so nothing in the design grid can be checked against
1.8.5, and a bug in the in-memory rebuild would be indistinguishable from a real
effect of the redesign. Two anchors, both x the frozen 1975-1998 convention, both
independent of the fixed windows, both already in instrument_first_stage.csv at
every rung and h:

    rolling      the exposure that has been on disk longest
    rolling_t5   the exposure this table actually reports, on disk since 2026-08-13

They run beside the forty-eight design fits and are marked role="anchor". The
frozen sweep is h=1,2,3 and this table reports h=4, so check 4 compares on the
horizons the frozen file covers and prints which they were.

Output:
    dc.LOG_DIR/first_stage_table.csv
    dc.LOG_DIR/19_first_stage_table.log
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


# 14 owns the build and the estimator. Import it rather than restate either, so
# this file cannot drift into measuring something else. 14's writers are reachable
# only from its main(), which exec_module does not run, so this import writes
# nothing.
bi = _load("build_instrument", "14_build_instrument.py")

# The two new exposure windows. Five years each, ending one year before the era
# they describe, which is what makes them comparable to rolling's ROLL_K = 5.
NEW_WINDOWS = {"fixed_7074": (1970, 1974), "fixed_9397": (1993, 1997)}
WINDOW_LEN = 5

# Which exposure describes which era. This pairing IS the design: a fixed loading
# is only usable on the era it precedes, and each era gets its own.
#
# The rolling exposure is rolling_t5 as of 2026-08-13. `rolling` is NOT a design
# row any more; it appears only as ANCHOR_EXPOSURE below, because that is the one
# fit here that reproduces a number already on disk.
DESIGN = (("fixed_7074", "pre"), ("fixed_9397", "post"),
          ("rolling_t5", "pre"), ("rolling_t5", "post"))

# The horizons this table reports. h=1 out and h=4 in as of 2026-08-13, so this is
# no longer bi.HORIZONS - 14's production sweep is still 1,2,3 and h=4 has no block
# in the panel, which check 3 handles rather than passes over.
HORIZONS = (2, 3, 4)

# The employment data's own constants, restated rather than imported across steps.
# periods_for below is pinned to step_2's documented bounds by check 1.
EFSY_FIRST_YEAR = 1975      # EFSY's NAICS long panel starts here
NAICS_BREAK = 1998          # CBP goes native NAICS; before it EFSY imputes from SIC
LAST_YEAR = 2018            # contracts reach FY2018 through the splice, EFSY ends 2018

# The frozen convention the anchor runs on. A key that already exists in
# bi.PERIODS, so the anchor is not a variant of anything this file invented.
#
# The anchor exposure stays `rolling` even though no design row uses it: the whole
# point of an anchor is that instrument_first_stage.csv already carries its F at
# every rung, and `rolling` is the exposure that has been on disk longest. 14 also
# writes rolling_t5 into that file as of 2026-08-13, so ANCHOR_ALSO adds a second,
# independent anchor on the exposure this table actually reports.
ANCHOR_PERIOD = "1975-1998"
ANCHOR_EXPOSURE = "rolling"
ANCHOR_ALSO = "rolling_t5"
ANCHOR_EXPOSURES = (ANCHOR_EXPOSURE, ANCHOR_ALSO)

RUNGS = (("county", "ccg"), ("commuting zone", "cz"),
         ("state", "state"), ("Census division", "division"))

# The aggregation scheme, and the labels instrument_first_stage.csv uses for it.
# "unweighted" at the county is not a third scheme - omega = 1 there, so no weight
# is formed at all. Using the frozen file's own two labels lets check 4 join on
# them directly instead of mapping.
WEIGHT_COL = "emp_lag"
WEIGHT_LABEL = {"county": "unweighted"}
DEFAULT_LABEL = "employment"

N_DESIGN_FITS = len(DESIGN) * len(HORIZONS) * len(RUNGS)             # 48
N_ANCHOR_FITS = len(ANCHOR_EXPOSURES) * len(HORIZONS) * len(RUNGS)   # 24

TOL_BUILD = 1e-12    # rebuilt series against the production parquet

# The anchor tolerance is RELATIVE, and that is not fussiness. Two things make an
# absolute tolerance the wrong shape here. The anchor F ranges over four orders of
# magnitude - 0.05 at the degenerate county cell to 300 at the division - so one
# absolute number is simultaneously far too strict at the top of the ladder and far
# too loose at the bottom. And the fits are not bit-reproducible: bi.fit_rung takes
# pyfixest's DEFAULT demeaner, whose alternating projections stop on a tolerance
# rather than at a fixed point, so repeated canonical runs of this file have agreed
# with the frozen sweep to 0.0, 1.4e-14 and 5.4e-08 on the same cells. 22 pins
# pf.MapDemeaner(fixef_tol=1e-12, ...) explicitly for exactly this reason; 14 does
# not, so this file inherits the looser default and states the fact instead of
# tightening a knob that belongs to 14.
#
# 1e-6 relative is still four orders tighter than the tenth of an F the README
# quotes, so it discriminates a real regression from projection noise.
RTOL_FIT = 1e-6
ATOL_FIT = 1e-9


def check(failures, ok, num, label, detail=""):
    """Print one numbered PASS/FAIL line and record the failure."""
    print(f"    [{num}] {label:<54s} {detail:<40s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


class Tee:
    """This file is invoked with no shell redirection, so it tees its own stdout
    into the log it declares. Flushed per write: Python buffers stdout whenever it
    is not a terminal, and an unattended run that hung would otherwise look exactly
    like one that had not started."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.out, self.err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()
        self.out.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        sys.stdout, sys.stderr = self.out, self.err
        self.f.close()


def periods_for(h):
    """(eras, straddle), derived from h and never hardcoded.

    A transcription of 22_winsor_construction_error.py's periods_for, with
    LAST_YEAR at 2018 rather than 2006 because contracts now reach FY2018. The
    first year is EFSY_FIRST_YEAR + h because t-h has to reach EFSY's first year;
    the straddle runs from NAICS_BREAK to NAICS_BREAK + h - 1 because those are the
    years whose t-h sits on the imputed side of the break.

    Restated rather than imported: reaching into step_2's code from step_1 would
    invert the dependency between the steps. Check 1 pins it to step_2's published
    bounds instead.
    """
    return ({"pre": (EFSY_FIRST_YEAR + h, NAICS_BREAK - 1),
             "post": (NAICS_BREAK + h, LAST_YEAR)},
            (NAICS_BREAK, NAICS_BREAK + h - 1))


def era_key(era, h):
    """The name this file registers an (era, h) bound under in bi.PERIODS.

    The eras are h-dependent, so they cannot share one key the way 14's fixed
    periods do - "pre" alone would mean three different windows.
    """
    return f"{era}-h{h}"


def register_variants(horizons):
    """Add the new windows and the h-indexed era bounds to 14's own dicts.

    bi.contaminated_years looks its arguments up BY NAME in bi.FIXED_WINDOWS and
    bi.PERIODS, so registering here reuses the canonical contamination logic
    verbatim instead of restating it - and restating it is exactly the drift that
    would let this file report zero contamination while the real design had some.

    Mutating an imported module's constants is safe here for one specific reason:
    this process never reaches 14's writers, which live behind its main(). Check 2
    asserts that the mutation only ADDED, so a key collision cannot silently
    redefine `fixed` or `1975-1998` underneath the anchor.
    """
    before = (dict(bi.FIXED_WINDOWS),
              dict(bi.PERIODS),
              {k: set(v) for k, v in bi.ADMISSIBLE.items()})
    bi.FIXED_WINDOWS.update(NEW_WINDOWS)
    for h in horizons:
        eras, _ = periods_for(h)
        for era, bounds in eras.items():
            key = era_key(era, h)
            bi.PERIODS[key] = bounds
            bi.ADMISSIBLE[key] = {e for e, x in DESIGN if x == era}
    return before


def only_added(before):
    """(ok, detail) - every pre-existing key still maps to its original value."""
    fw, pe, ad = before
    bad = []
    for k, v in fw.items():
        if bi.FIXED_WINDOWS.get(k) != v:
            bad.append(f"FIXED_WINDOWS[{k}]")
    for k, v in pe.items():
        if bi.PERIODS.get(k) != v:
            bad.append(f"PERIODS[{k}]")
    for k, v in ad.items():
        if set(bi.ADMISSIBLE.get(k, ())) != v:
            bad.append(f"ADMISSIBLE[{k}]")
    return not bad, ("nothing redefined" if not bad else ", ".join(bad))


def build(horizons):
    """The instrument on every exposure, in memory, at 14's production settings.

    Floor 0, cumulative dG, leave-one-out on, the transitional quarter folded, and
    the levels reaching back to EXPOSURE_FIRST because a pre-period window needs
    years the estimation sample does not contain. A transcription of 14's main() up
    to the point where it writes, and nothing after it.
    """
    levels, universe, _ = bi.build_levels(bi.DEFAULT_FLOOR, True,
                                          bi.EXPOSURE_FIRST, LAST_YEAR)
    emp = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    levels = levels.merge(emp, on=["ccg", "year"], how="left")
    levels = levels.sort_values(["ccg", "year"]).reset_index(drop=True)

    fixed = {}
    for name, window in NEW_WINDOWS.items():
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
        # Both rolling variants, from 14's own factory rather than from a local
        # call with a local end_lag: rolling_t5 is the design's exposure and
        # rolling is the anchor's, and a private copy of either is how this file
        # would end up anchoring one series while reporting another.
        ex.update(bi.rolling_exposures(levels, h, bi.ROLL_K, bi.ROLL_MIN,
                                       names=ROLL_NAMES))
        f = bi.build_horizon(levels, ex, h, bi.FIRST_YEAR, LAST_YEAR, True,
                             delta=bi.DEFAULT_DELTA)
        frames.append(f)
        v = f["shock"].dropna()
        print(f"    h={h}   {len(v):,} CCG-years carry a shock   sd {v.std():.5f}")
    county = pd.concat(frames, ignore_index=True)

    # bi.collapse's column list carries pre_D and pre_Y unconditionally. They feed
    # only the construction-error exhibit, which this table does not compute, but
    # they have to exist or collapse raises. Attached per window so the frame stays
    # self-describing if that exhibit is ever added here.
    pre = {name: bi.fixed_window_sums(levels, w).set_index("ccg")
           for name, w in NEW_WINDOWS.items()}
    return county, levels, universe, pre


def windows_for(name, h):
    """Every (role, era, period_key, y0, y1, straddle) this exposure reports at h."""
    out = []
    eras, straddle = periods_for(h)
    for exposure, era in DESIGN:
        if exposure != name:
            continue
        y0, y1 = eras[era]
        out.append(("variant", era, era_key(era, h), y0, y1,
                    f"{straddle[0]}-{straddle[1]}"))
    if name in ANCHOR_EXPOSURES:
        y0, y1 = bi.PERIODS[ANCHOR_PERIOD]
        # The frozen convention keeps 1998 in both halves, so it drops nothing.
        out.append(("anchor", ANCHOR_PERIOD, ANCHOR_PERIOD, y0, y1, ""))
    return out


def window_label(name):
    """The exposure's window, in words. One definition, used by the print, the
    CSV's exposure_window column and the report header."""
    if name in NEW_WINDOWS:
        w0, w1 = NEW_WINDOWS[name]
        return f"{w0}-{w1}"
    end = bi.ROLL_END_LAG[name]
    return (f"trailing {bi.ROLL_K} ending "
            f"{'t-h-1' if end is None else f't-{end}'}")


# Every exposure this file has to BUILD: the design's, plus the anchors'. Kept
# apart from DESIGN because `rolling` is built and never reported.
ROLL_NAMES = tuple(n for n in bi.ROLLING
                   if n in {e for e, _ in DESIGN} | set(ANCHOR_EXPOSURES))
EXPOSURES = tuple(dict.fromkeys([e for e, _ in DESIGN] + list(ANCHOR_EXPOSURES)))


def fit_grid(county, rungs, pre, horizons, exposures, rung_names):
    """One row per role x exposure x era x h x rung.

    The finiteness screen sits OUTSIDE the era loop, exactly as 14's rung_sweep
    does it, so the county support is frozen the same way in every sub-sample and
    an era differs from another only in which years it contains.
    """
    wanted = [(r, c) for r, c in RUNGS if r in set(rung_names)]
    rows = []
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

            for role, era, pkey, y0, y1, straddle in windows_for(name, h):
                base = full[full["year"].between(y0, y1)]
                n_con = len(bi.contaminated_years(pkey, name, h))
                for rung, unit_col in wanted:
                    if rung == "county":
                        d = base.rename(columns={"emp_growth": "y", "shock": "x"})
                        d = d.assign(unit=d["ccg"])[["unit", "year", "y", "x", "z"]]
                    else:
                        d = bi.collapse(base, unit_col, WEIGHT_COL)[
                            ["unit", "year", "y", "x", "z"]]
                    d = d[np.isfinite(d[["x", "y", "z"]].to_numpy()).all(axis=1)]
                    r = bi.fit_rung(d, "unit + year")
                    rows.append(dict(
                        role=role, rung=rung, n_units=int(d["unit"].nunique()),
                        weights=WEIGHT_LABEL.get(rung, DEFAULT_LABEL),
                        h=h, era=era, year_first=y0, year_last=y1,
                        straddle_dropped=straddle, exposure=name,
                        exposure_window=window_label(name), fe="unit+year",
                        contaminated_years=n_con, **r))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------

def check_periods(failures):
    """Check 1. periods_for reproduces step_2 README section 1.3's own bounds.

    Only the era STARTS and the straddle are compared. The post era's last year is
    2018 here against step_2's 2006 by design - contracts reach FY2018 through the
    splice - so asserting on it would be asserting that the splice never happened.
    """
    want = {1: dict(pre=(1976, 1997), post_first=1999, straddle=(1998, 1998)),
            2: dict(pre=(1977, 1997), post_first=2000, straddle=(1998, 1999))}
    ok, detail = True, []
    for h, w in want.items():
        eras, straddle = periods_for(h)
        good = (eras["pre"] == w["pre"] and eras["post"][0] == w["post_first"]
                and straddle == w["straddle"])
        ok &= good
        detail.append(f"h={h} {'ok' if good else 'BAD'}")
    return check(failures, ok, 1, "eras match step_2 1.3's documented bounds",
                 ", ".join(detail))


def rebuild_deviation(county, horizons, cols=("shock", "z_rolling", "z_rolling_t5")):
    """(ok, detail) - the rebuilt series against production, at EVERY horizon.

    Shared by 19, 20 and 23 so there is one horizon-robust version of this check
    rather than three. The compared columns are the ones that do NOT depend on the
    fixed windows, which is what makes them usable at all: z_fixed_7074 has no
    counterpart on disk. Values AND missingness both - a rebuild that silently
    turned NaN into 0 would pass a values-only test on the rows that survived it.

    WHERE PRODUCTION HAS NO BLOCK, PRODUCTION CANNOT BE THE WITNESS. 14's sweep is
    h=1,2,3, and filtering the production frame on h.isin(horizons) at h=4 merges
    zero rows, so `len(m) == len(iv)` compares nothing and reports a clean pass -
    the vacuity README 1.8.8 records in 20's own check 3. So the shock is
    recomputed at those horizons directly from contracts_ccg_year.parquet over
    income_ccg_year.parquet, which is an independent witness rather than a weaker
    one: it is written from the shock's definition instead of from a rolling sum.
    """
    iv = pd.read_parquet(dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet",
                         columns=["ccg", "year", "h"] + list(cols))
    have = sorted(set(iv["h"]) & set(horizons))
    absent = sorted(set(horizons) - set(iv["h"]))

    ok, worst = True, 0.0
    if have:
        sub = iv[iv["h"].isin(have)]
        m = county[county["h"].isin(have)][["ccg", "year", "h"] + list(cols)].merge(
            sub, on=["ccg", "year", "h"], suffixes=("_new", "_prod"))
        ok &= len(m) == len(sub) and len(m) > 0
        for col in cols:
            a, b = m[f"{col}_new"], m[f"{col}_prod"]
            if not a.isna().equals(b.isna()):
                ok = False
            both = a.notna() & b.notna()
            worst = max(worst, float((a[both] - b[both]).abs().max()))

    src_worst, n_src = 0.0, 0
    if absent:
        con = pd.read_parquet(dc.PANEL_OUT_DIR / "contracts_ccg_year.parquet",
                              columns=["ccg", "year", "dollars_real"])
        inc = pd.read_parquet(dc.PANEL_OUT_DIR / "income_ccg_year.parquet",
                              columns=["ccg", "year", "income_real"])
        s = con.merge(inc, on=["ccg", "year"], how="left").sort_values(
            ["ccg", "year"]).reset_index(drop=True)
        for h in absent:
            g = s.groupby("ccg", sort=False)
            num = g["dollars_real"].rolling(h).sum().reset_index(level=0, drop=True)
            den = g["income_real"].shift(h)
            want = pd.DataFrame({"ccg": s["ccg"], "year": s["year"],
                                 "want": (num / den).to_numpy()})
            seam = list(range(dc.SEAM_YEAR, dc.SEAM_YEAR + h - 1))
            want = want[~want["year"].isin(seam)]
            m = county[county["h"] == h][["ccg", "year", "shock"]].merge(
                want, on=["ccg", "year"], how="inner")
            m = m[m["shock"].notna() & m["want"].notna()]
            n_src += len(m)
            src_worst = max(src_worst, float((m["shock"] - m["want"]).abs().max()))
            ok &= len(m) > 0
    ok &= worst < TOL_BUILD and src_worst < TOL_BUILD
    return ok, (f"h{have} vs production {worst:.1e}; "
                f"h{absent} vs source {n_src:,} rows {src_worst:.1e}")


def check_build(failures, county, horizons):
    """Check 3. The rebuild is right at every horizon, production or not."""
    ok, detail = rebuild_deviation(county, horizons)
    return check(failures, ok, 3, "the rebuild is right at every horizon", detail)


def check_rolling_pair(failures, county, horizons):
    """Check 3b. rolling_t5 IS rolling at h=4 and is not at h=2 or 3.

    14's own check 8b, restated at the horizons this table reports - and this is
    the file where it bites, because h=4 is reported here and not there.

    WHICH BRANCH CAN FIRE DEPENDS ON THE HORIZONS, and the check says so rather
    than failing a single-horizon caller. The failure mode worth catching is an
    end_lag silently ignored, which would make the two series identical at EVERY
    horizon; the `differ` branch catches that and is reachable at h != 4. A run
    that can only reach the `same` branch cannot rule it out on its own, so the
    detail names the gap instead of a count pretending to cover it. 20 runs h=2
    alone and reaches only `differ`, which is why this is not an all-or-nothing.
    """
    ok, detail, n_same, n_diff = True, [], 0, 0
    for h in horizons:
        f = county[county["h"] == h]
        a, b = f["exposure_rolling"], f["exposure_rolling_t5"]
        both = a.notna() & b.notna()
        dev = float((a[both] - b[both]).abs().max()) if both.any() else np.nan
        if bi.roll_lag(h, None) == bi.roll_lag(h, bi.ROLL_END_LAG["rolling_t5"]):
            ok &= bool((a.isna() == b.isna()).all()) and dev == 0.0
            n_same += 1
            detail.append(f"h={h} identical")
        else:
            ok &= not (dev == 0.0)
            n_diff += 1
            detail.append(f"h={h} dev {dev:.3f}")
    # At least one horizon has to have been classified, or nothing was compared.
    ok &= (n_same + n_diff) == len(horizons) and len(horizons) > 0
    if n_diff == 0:
        detail.append("differ branch unreachable at these horizons")
    if n_same == 0:
        detail.append("identical branch needs h=4")
    return check(failures, ok, "3b",
                 "rolling_t5 equals rolling at h=4, differs below it",
                 ", ".join(detail))


def check_anchor(failures, fit, horizons, rung_names):
    """Check 4. Both anchors reproduce the frozen sweep.

    An empty merge is a FAILURE and not a pass. The whole value of this check is
    that it compares something, and a key that stopped matching would otherwise
    report "0 cells, max dev nan" and be read as green - which is why the expected
    count is derived from what this run asked for rather than asserted to be 12.

    THE COUNT IS DERIVED FROM WHAT THE FROZEN FILE COVERS, not from this run's
    horizons. 14's sweep is h=1,2,3 and this table reports h=4, so demanding a
    cell at every horizon would fail for a reason that is not about this file. The
    horizons actually compared are printed, so a silently shrinking anchor is
    visible rather than inferable.
    """
    a = fit[fit["role"] == "anchor"]
    o = pd.read_csv(dc.LOG_DIR / "instrument_first_stage.csv")
    o = o[(o["fe"] == "unit+year") & (o["period"] == ANCHOR_PERIOD)
          & (o["exposure"].isin(ANCHOR_EXPOSURES))]
    covered = sorted(set(horizons) & set(o["h"]))
    want = len(ANCHOR_EXPOSURES) * len(covered) * len(rung_names)
    k = ["rung", "weights", "h", "exposure"]
    m = o.merge(a, on=k, suffixes=("_frozen", "_new"))
    ok = len(m) == want
    worst_rel, worst_abs, where = 0.0, 0.0, ""
    for col in ("fs_F", "beta", "beta_se"):
        if not len(m):
            worst_rel = np.inf
            break
        f, n = m[f"{col}_frozen"], m[f"{col}_new"]
        dev = (f - n).abs()
        rel = dev / (ATOL_FIT + f.abs())
        if float(rel.max()) > worst_rel:
            i = rel.idxmax()
            where = f"{col} at {m.loc[i, 'rung']} h={m.loc[i, 'h']}"
        worst_rel = max(worst_rel, float(rel.max()))
        worst_abs = max(worst_abs, float(dev.max()))
    ok &= worst_rel < RTOL_FIT and want > 0
    return check(failures, ok, 4, "anchors reproduce instrument_first_stage.csv",
                 f"{len(m)}/{want} cells on h={covered}, "
                 f"rel {worst_rel:.1e} abs {worst_abs:.1e}"
                 + (f" ({where})" if where else ""))


def check_clean(failures, fit):
    """Checks 5-8. The properties the redesign exists to have."""
    d = fit[fit["role"] == "variant"]
    check(failures, int(d["contaminated_years"].max()) == 0, 5,
          "zero contaminated years on every design row",
          f"max {int(d['contaminated_years'].max())} over {len(d)} rows")

    # Symmetry. In both eras the first estimated year is window_end + h + 1: for
    # pre because EFSY's first year forces it, for post because the break does.
    sym, detail = True, []
    for exposure, era in DESIGN:
        if exposure not in NEW_WINDOWS:
            continue
        w1 = NEW_WINDOWS[exposure][1]
        for h in sorted(d["h"].unique()):
            y0 = int(d[(d["exposure"] == exposure) & (d["era"] == era)
                       & (d["h"] == h)]["year_first"].iloc[0])
            sym &= (y0 == w1 + h + 1)
            detail.append(f"{exposure} h={h} {y0}")
    check(failures, sym, 6, "first year is window_end + h + 1 in both eras",
          f"{len(detail)} cells, e.g. {detail[0]} and {detail[-1]}")

    lens = {n: w[1] - w[0] + 1 for n, w in NEW_WINDOWS.items()}
    check(failures, set(lens.values()) == {WINDOW_LEN}, 7,
          f"every fixed window is exactly {WINDOW_LEN} years", str(lens))

    # The straddle years must appear in no estimation window. Asserted against the
    # tuple periods_for returns rather than against the numbers in the note, so a
    # change to NAICS_BREAK cannot leave the note right and the windows wrong.
    strad_ok, hit = True, []
    for h in sorted(d["h"].unique()):
        eras, (s0, s1) = periods_for(h)
        bad = [y for y in range(s0, s1 + 1)
               for y0, y1 in eras.values() if y0 <= y <= y1]
        strad_ok &= not bad
        hit += bad
    check(failures, strad_ok, 8, "straddle years sit in no estimation window",
          "none" if strad_ok else f"in a window: {sorted(set(hit))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS),
                    help="horizons to report; each one sets its own era bounds")
    ap.add_argument("--rung", nargs="+", default=[r for r, _ in RUNGS],
                    choices=[r for r, _ in RUNGS])
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))

    # A PARTIAL RUN NEVER OVERWRITES THE CANONICAL OUTPUT, and that includes the
    # log. README section 4 item 9 records this hazard on 14 and prescribes the
    # fix: a variant path. A debugging run on one horizon would otherwise replace
    # the table with a subset of itself, and the file on disk would look complete.
    canonical = (horizons == sorted(HORIZONS)
                 and set(args.rung) == {r for r, _ in RUNGS})
    sfx = "" if canonical else ("__" + "-".join(map(str, horizons)) + "_h"
                                + "-".join(sorted(args.rung)).replace(" ", ""))

    tee = Tee(dc.LOG_DIR / f"19_first_stage_table{sfx}.log")
    try:
        failures = []
        print("=" * 78)
        print("Plan check 5 - the first stage on the symmetric two-era instrument")
        print("=" * 78)
        print(f"  exposures  " + ", ".join(
            f"{n} {window_label(n)}" for n in EXPOSURES))
        print(f"  pairing    " + ", ".join(f"{e} -> {x}" for e, x in DESIGN))
        print(f"  h          {horizons}")
        for h in horizons:
            eras, straddle = periods_for(h)
            print(f"    h={h}   pre {eras['pre'][0]}-{eras['pre'][1]}   "
                  f"post {eras['post'][0]}-{eras['post'][1]}   "
                  f"straddle {straddle[0]}-{straddle[1]}, estimated nowhere")
        print(f"  anchors    " + " and ".join(ANCHOR_EXPOSURES)
              + f" x {ANCHOR_PERIOD}, the frozen convention, against "
                f"instrument_first_stage.csv")
        print(f"             rolling is an ANCHOR ONLY - no design row uses it")
        print(f"  weights    employment above the county, omega = 1 at the county")
        print(f"  FE         unit + year, clustered on the unit")

        print("\n[1] the eras, against step_2's own derivation")
        check_periods(failures)
        before = register_variants(horizons)
        ok, detail = only_added(before)
        check(failures, ok, 2, "registering the variants only ADDED keys", detail)

        print("\n[2] the instrument, every exposure, in memory")
        exposures = list(EXPOSURES)
        county, levels, universe, pre = build(horizons)
        rungs = bi.load_rungs(universe)

        print("\n[3] the build, against production and against source")
        check_build(failures, county, horizons)
        check_rolling_pair(failures, county, horizons)

        print("\n[4] the fits")
        fit = fit_grid(county, rungs, pre, horizons, exposures, args.rung)
        n_var = int((fit["role"] == "variant").sum())
        n_anc = int((fit["role"] == "anchor").sum())
        print(f"    {len(fit)} fits: {n_var} design, {n_anc} anchor")

        print("\n[5] the checks")
        check_anchor(failures, fit, horizons, args.rung)
        check_clean(failures, fit)
        if canonical:
            check(failures, n_var == N_DESIGN_FITS and n_anc == N_ANCHOR_FITS, 9,
                  "row count is the announced grid",
                  f"{n_var} of {N_DESIGN_FITS} design, {n_anc} of {N_ANCHOR_FITS} anchor")
        else:
            # The same arithmetic, scaled to what was asked for. A partial run that
            # skipped this entirely could drop a whole rung and still look green.
            w_var = len(DESIGN) * len(horizons) * len(args.rung)
            w_anc = len(ANCHOR_EXPOSURES) * len(horizons) * len(args.rung)
            check(failures, n_var == w_var and n_anc == w_anc, 9,
                  "row count is the requested subset",
                  f"{n_var} of {w_var} design, {n_anc} of {w_anc} anchor")

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            return 1

        if not canonical:
            print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; "
                  f"this run writes to *{sfx}.* instead.")
            print("  Rerun with no arguments to rebuild the canonical set.")

        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = dc.LOG_DIR / f"first_stage_table{sfx}.csv"
        fit.to_csv(out, index=False)
        print(f"\nall checks passed   {len(fit)} fits -> {out}")

        # The table, as the README will carry it.
        for role in ("variant", "anchor"):
            d = fit[fit["role"] == role]
            if d.empty:
                continue
            print(f"\n  {role}s   first-stage F | 2SLS beta (se)")
            for name in exposures:
                for era in dict.fromkeys(d[d["exposure"] == name]["era"]):
                    s = d[(d["exposure"] == name) & (d["era"] == era)]
                    y = s.iloc[0]
                    print(f"\n    {name} {window_label(name)}  x  {era}"
                          f"   ({y['year_first']}-{y['year_last']})")
                    print(f"      {'rung':<17s}{'units':>7s}" + "".join(
                        f"{'h=' + str(h):>26s}" for h in horizons))
                    for rung, _ in RUNGS:
                        r = s[s["rung"] == rung]
                        if r.empty:
                            continue
                        line = f"      {rung:<17s}{int(r['n_units'].iloc[0]):>7,}"
                        for h in horizons:
                            c = r[r["h"] == h]
                            line += ("".rjust(26) if c.empty else
                                     f"{c['fs_F'].iloc[0]:>10.1f} | "
                                     f"{c['beta'].iloc[0]:+.3f} "
                                     f"({c['beta_se'].iloc[0]:.3f})".rjust(13))
                        print(line)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
